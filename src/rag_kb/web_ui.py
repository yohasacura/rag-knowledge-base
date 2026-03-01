"""Web-based UI for managing RAG knowledge bases (NiceGUI).

Replaces the PySide6 desktop UI with a browser-based interface built on
NiceGUI (FastAPI + Vue/Quasar).  Communicates with the daemon via
``DaemonClient`` — the same way CLI and MCP do.

Launch
------
::

    rag-kb ui              # opens browser tab
    rag-kb ui --native     # opens in a native desktop window
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any

from nicegui import app, run, ui

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Windows asyncio proactor fix
# ---------------------------------------------------------------------------


def _patch_proactor_connection_lost() -> None:
    """Monkey-patch asyncio proactor transport to handle ``OSError`` on shutdown.

    ``_ProactorBasePipeTransport._call_connection_lost()`` calls
    ``socket.shutdown(SHUT_RDWR)`` inside a ``finally`` block without catching
    ``OSError``.  On Windows, when the remote end has already reset the TCP
    connection (``WinError 10054``), the ``shutdown()`` call raises
    ``ConnectionResetError``.  Because it is in a ``finally`` block the
    subsequent ``socket.close()`` and flag updates are **skipped**, leaking the
    socket.

    This bug is present in CPython 3.9 – 3.14-dev (as of 2025-06).
    ``https://github.com/python/cpython/issues/87419``

    The patch wraps only the ``shutdown()`` call so that all other cleanup runs
    normally.  It is safe to call on non-Windows platforms (it will no-op).
    """
    if sys.platform != "win32":
        return

    import asyncio.proactor_events as _pe
    import socket as _socket

    def _safe_call_connection_lost(self, exc):
        if self._called_connection_lost:
            return
        try:
            self._protocol.connection_lost(exc)
        finally:
            if hasattr(self._sock, "shutdown") and self._sock.fileno() != -1:
                try:
                    self._sock.shutdown(_socket.SHUT_RDWR)
                except OSError:
                    pass  # connection already reset / closed – harmless
            self._sock.close()
            self._sock = None
            server = self._server
            if server is not None:
                try:
                    server._detach(self)  # CPython >= 3.13
                except TypeError:
                    server._detach()  # CPython <= 3.12
                self._server = None
            self._called_connection_lost = True

    _pe._ProactorBasePipeTransport._call_connection_lost = (  # type: ignore[assignment]
        _safe_call_connection_lost
    )


# ---------------------------------------------------------------------------
# Globals (initialised in launch_web_ui)
# ---------------------------------------------------------------------------

_client = None  # DaemonClient instance
_client_lock = threading.Lock()  # serialises _safe_call to prevent close-during-use races
_auto_refresh_interval = 10  # seconds


def _make_client():
    """Create and connect a fresh DaemonClient."""
    from rag_kb.daemon_client import DaemonClient

    c = DaemonClient()
    c.ensure_daemon()
    c.connect()
    return c


def _find_free_port(host: str, preferred: int, max_attempts: int = 10) -> int:
    """Return *preferred* if it is available, otherwise scan upward.

    Tries up to *max_attempts* consecutive ports starting from *preferred*.
    If none are free, returns the original port and lets the caller fail
    with a clear error.
    """
    for offset in range(max_attempts):
        candidate = preferred + offset
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind((host, candidate))
                if offset:
                    logger.info(
                        "Port %d in use — using port %d instead.",
                        preferred,
                        candidate,
                    )
                return candidate
        except OSError:
            continue
    logger.warning(
        "Could not find a free port in range %d–%d; will attempt %d anyway.",
        preferred,
        preferred + max_attempts - 1,
        preferred,
    )
    return preferred


def _get_client():
    """Return the shared DaemonClient, creating/connecting if needed."""
    global _client
    if _client is None:
        _client = _make_client()
    return _client


def _safe_call(method, *args, **kwargs):
    """Call a DaemonClient method with error handling and auto-reconnect.

    A global lock serialises all calls so that a reconnect cycle in one
    thread cannot close the socket while another thread is mid-call.
    """
    from rag_kb.rpc_protocol import RpcError

    with _client_lock:
        try:
            client = _get_client()
            fn = getattr(client, method)
            return fn(*args, **kwargs)
        except RpcError:
            # Application-level error from the daemon (e.g. "not found") —
            # the connection is fine, propagate directly.
            raise
        except Exception as exc:
            logger.warning("RPC call %s failed: %s — reconnecting", method, exc)
            global _client
            # If stale daemon detected, force-kill it so ensure_daemon spawns fresh
            _is_stale = "unframed JSON" in str(exc) or "stale daemon" in str(exc)
            if _is_stale:
                logger.warning("Stale daemon detected — killing and restarting")
                try:
                    from rag_kb.daemon_client import _PID_FILE

                    if _PID_FILE.exists():
                        pid = int(_PID_FILE.read_text(encoding="utf-8").strip())
                        os.kill(pid, signal.SIGTERM)
                        _PID_FILE.unlink(missing_ok=True)
                except Exception:
                    pass
                time.sleep(2)
            try:
                if _client:
                    _client.close()
            except Exception:
                pass
            _client = None
            # Retry once with a fresh daemon
            try:
                client = _get_client()
                fn = getattr(client, method)
                return fn(*args, **kwargs)
            except Exception as exc2:
                logger.exception("RPC call %s failed after reconnect: %s", method, exc2)  # noqa: TRY401
                raise


# ---------------------------------------------------------------------------
# Custom dark CSS — matches original PySide6 theme
# ---------------------------------------------------------------------------

_CUSTOM_CSS = """
:root {
    --bg-main: #14181f;
    --bg-sidebar: #11151c;
    --bg-card: #1d232e;
    --bg-surface: #22262e;
    --border-color: #313949;
    --border-input: #3e4a5f;
    --text-primary: #e4e8ef;
    --text-secondary: #9daec1;
    --text-dim: #788a9a;
    --accent-blue: #2f79c6;
    --accent-cyan: #67d3ff;
    --accent-light-blue: #4fc3f7;
    --accent-green: #27ae60;
    --accent-orange: #f39c12;
    --accent-red: #c0392b;
}

body {
    background-color: var(--bg-main) !important;
}

.nicegui-content {
    padding: 0 !important;
}

.stat-value {
    font-size: 2rem;
    font-weight: bold;
    color: var(--accent-cyan);
}

.stat-label {
    font-size: 0.85rem;
    color: var(--text-secondary);
}

.result-card {
    border-left: 4px solid var(--accent-blue) !important;
}

.score-high { color: #27ae60; }
.score-mid  { color: #f39c12; }
.score-low  { color: #c0392b; }

.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: bold;
    margin-right: 4px;
}
.badge-active { background: #1a5276; color: #67d3ff; }
.badge-local { background: #1e3a20; color: #27ae60; }
.badge-imported { background: #3a2a10; color: #f39c12; }
.badge-detached { background: #3a1a18; color: #c0392b; }

.config-section-title {
    font-size: 1rem;
    font-weight: bold;
    color: var(--accent-light-blue);
    margin-top: 16px;
    margin-bottom: 4px;
}

.file-change-badge {
    display: inline-block;
    padding: 1px 8px;
    border-radius: 10px;
    font-size: 0.75rem;
    font-weight: bold;
    margin-right: 4px;
}

/* Navigation button styles */
.nav-btn {
    width: 100% !important;
    text-align: left !important;
    padding: 10px 16px !important;
    border-left: 3px solid transparent;
    justify-content: flex-start !important;
    transition: background-color 0.15s ease, border-color 0.15s ease;
}
.nav-active {
    color: white !important;
    background: rgba(103, 211, 255, 0.18) !important;
    border-left-color: #67d3ff !important;
    font-weight: bold;
}
.nav-inactive {
    color: #9daec1 !important;
    border-left-color: transparent !important;
}
.nav-inactive:hover {
    background: rgba(255, 255, 255, 0.05) !important;
}

/* Page content fade-in */
.page-content {
    animation: pageIn 0.15s ease;
}
@keyframes pageIn {
    from { opacity: 0; }
    to { opacity: 1; }
}

/* Skeleton loader */
.skeleton {
    background: linear-gradient(90deg, #232830 25%, #2e3440 50%, #232830 75%);
    background-size: 200% 100%;
    animation: skeletonShimmer 1.5s ease-in-out infinite;
    border-radius: 6px;
}
@keyframes skeletonShimmer {
    0%   { background-position: 200% 0; }
    100% { background-position: -200% 0; }
}
.skeleton-text {
    height: 14px;
    margin: 4px 0;
}
.skeleton-heading {
    height: 22px;
    margin: 4px 0;
}
.skeleton-stat {
    height: 38px;
    margin-bottom: 6px;
}
.skeleton-card {
    border-radius: 12px;
    border: 1px solid var(--border-color);
    padding: 20px;
    background: var(--bg-card);
}

/* Responsive adjustments */
@media (max-width: 599px) {
    .stat-value { font-size: 1.5rem; }
}
"""

# ---------------------------------------------------------------------------
# Skeleton helpers
# ---------------------------------------------------------------------------


def _skeleton_line(width: str = "100%", height: str = "14px", classes: str = ""):
    """Render a single shimmer skeleton line."""
    ui.element("div").classes(f"skeleton skeleton-text {classes}").style(
        f"width: {width}; height: {height};"
    )


def _skeleton_block(width: str = "100%", height: str = "80px", classes: str = ""):
    """Render a rectangular skeleton block."""
    ui.element("div").classes(f"skeleton {classes}").style(
        f"width: {width}; height: {height}; border-radius: 12px;"
    )


def _skeleton_stat_cards(count: int = 4):
    """Render a row of skeleton stat cards."""
    with ui.row().classes("w-full gap-4 flex-wrap"):
        for _ in range(count):
            with ui.card().classes(
                "bg-[#1d232e] border border-[#313949] rounded-xl p-6 min-w-[180px] flex-1"
            ):
                _skeleton_line("60%", "38px")  # value
                _skeleton_line("80%", "12px")  # label


def _skeleton_detail_panel():
    """Render skeleton lines mimicking a detail panel."""
    _skeleton_line("35%", "22px")  # title
    ui.element("div").style("height: 8px")  # spacer
    with ui.row().classes("gap-8 flex-wrap"):
        for _ in range(4):
            with ui.column().classes("gap-1"):
                _skeleton_line("60px", "10px")
                _skeleton_line("100px", "14px")
    ui.element("div").style("height: 8px")
    _skeleton_line("45%", "12px")
    _skeleton_line("70%", "12px")


def _skeleton_table_rows(cols: int = 5, rows: int = 5):
    """Render skeleton rows mimicking a data table."""
    for _ in range(rows):
        with ui.row().classes("w-full gap-4 py-2 border-b border-[#232830]"):
            for c in range(cols):
                w = "15%" if c == 0 else f"{60 // (cols - 1)}%"
                _skeleton_line(w, "14px")


def _skeleton_model_cards(count: int = 3):
    """Render skeleton model cards."""
    for _ in range(count):
        with ui.card().classes("bg-[#1a1f2b] w-full p-4 border border-[#2a3040]"):
            with ui.row().classes("items-center gap-2"):
                _skeleton_line("180px", "18px")
                _skeleton_line("80px", "16px")
            _skeleton_line("250px", "11px", "mt-1")
            _skeleton_line("90%", "13px", "mt-2")
            with ui.row().classes("gap-2 mt-2"):
                _skeleton_line("50px", "20px")
                _skeleton_line("60px", "20px")
                _skeleton_line("70px", "20px")


def _skeleton_config_form():
    """Render skeleton mimicking a settings form."""
    for _section in range(3):
        _skeleton_line("140px", "16px", "mt-4 mb-1")
        _skeleton_block("100%", "40px", "max-w-[400px]")
        with ui.row().classes("gap-4 mt-2"):
            _skeleton_block("192px", "40px")
            _skeleton_block("192px", "40px")


def _skeleton_metric_cards(count: int = 5):
    """Render skeleton metric cards for Monitoring."""
    with ui.row().classes("w-full gap-4 flex-wrap"):
        for _ in range(count):
            with (
                ui.card().classes(
                    "bg-[#22262e] border border-[#313949] rounded-lg p-3 min-w-[140px] flex-1"
                ),
                ui.row().classes("items-center gap-2"),
            ):
                _skeleton_line("24px", "24px")
                with ui.column().classes("gap-1"):
                    _skeleton_line("70px", "18px")
                    _skeleton_line("50px", "10px")


def _skeleton_search_results(count: int = 3):
    """Render skeleton search result cards."""
    for _ in range(count):
        with ui.card().classes("w-full bg-[#1d232e] border border-[#313949] rounded-lg p-4"):
            with ui.row().classes("w-full justify-between items-center"):
                with ui.row().classes("items-center gap-2"):
                    _skeleton_line("30px", "18px")
                    _skeleton_line("200px", "14px")
                    _skeleton_line("60px", "12px")
                _skeleton_line("50px", "18px")
            _skeleton_block("100%", "6px", "mt-2")
            _skeleton_line("95%", "12px", "mt-3")
            _skeleton_line("80%", "12px")
            _skeleton_line("60%", "12px")


# ---------------------------------------------------------------------------
# Page: Dashboard
# ---------------------------------------------------------------------------


class DashboardPage:
    """Overview/status page."""

    def __init__(self):
        self._stat_rags = None
        self._stat_files = None
        self._stat_chunks = None
        self._stat_active = None
        self._details_container = None
        self._refresh_timer = None
        self._deferred_timer = None
        # Metrics at-a-glance value labels (created once, updated in-place)
        self._metric_idx_runs = None
        self._metric_search_queries = None
        self._metric_avg_latency = None
        self._metric_embed_batches = None

    def destroy(self):
        """Cancel auto-refresh timer when leaving the page."""
        if self._deferred_timer is not None:
            self._deferred_timer.cancel()
            self._deferred_timer = None
        if self._refresh_timer is not None:
            self._refresh_timer.cancel()
            self._refresh_timer = None

    def build(self):
        with ui.column().classes("w-full gap-4"):
            ui.label("Dashboard").classes("text-2xl font-bold text-white")
            ui.label("Overview of your RAG knowledge bases").classes("text-sm text-gray-400 -mt-2")

            # Stat cards row
            with ui.row().classes("w-full gap-4 flex-wrap"):
                self._stat_rags = self._stat_card("Knowledge Bases", "0")
                self._stat_files = self._stat_card("Indexed Files", "0")
                self._stat_chunks = self._stat_card("Text Chunks", "0")
                self._stat_active = self._stat_card("Active RAG", "—")

            # Metrics at-a-glance row (created once, updated in-place)
            with ui.row().classes("w-full gap-4 flex-wrap"):
                self._metric_idx_runs = self._stat_card("Indexing Runs", "0")
                self._metric_search_queries = self._stat_card("Search Queries", "0")
                self._metric_avg_latency = self._stat_card("Avg Search Latency", "— ms")
                self._metric_embed_batches = self._stat_card("Embed Batches", "0")

            # Details panel (skeleton placeholder until data loads)
            self._details_container = ui.column().classes(
                "w-full bg-[#1d232e] rounded-xl border border-[#313949] p-6 gap-2"
            )
            with self._details_container:
                _skeleton_detail_panel()

            # Initial async refresh (runs after build returns)
            self._deferred_timer = ui.timer(0.1, self.refresh, once=True)

            # Periodic auto-refresh
            self._refresh_timer = ui.timer(_auto_refresh_interval, self.refresh)

    def _stat_card(self, label: str, value: str):
        with ui.card().classes(
            "bg-[#1d232e] border border-[#313949] rounded-xl p-6 min-w-[180px] flex-1"
        ):
            val_label = ui.label(value).classes("stat-value")
            ui.label(label).classes("stat-label")
        return val_label

    async def refresh(self):
        try:
            rags = await run.io_bound(lambda: _safe_call("list_rags")) or []
            active = next((r for r in rags if r.get("is_active")), None)
            total_files = sum(r.get("file_count", 0) for r in rags)
            total_chunks = sum(r.get("chunk_count", 0) for r in rags)

            self._stat_rags.set_text(str(len(rags)))
            self._stat_files.set_text(str(total_files))
            self._stat_chunks.set_text(str(total_chunks))
            self._stat_active.set_text(active["name"] if active else "—")

            self._details_container.clear()
            with self._details_container:
                if active:
                    rtype = "Imported" if active.get("is_imported") else "Local"
                    status = "Detached" if active.get("detached") else rtype
                    ui.label(f"Active: {active['name']}").classes("text-lg font-bold text-white")
                    with ui.row().classes("gap-8 flex-wrap"):
                        self._detail("Type", status)
                        self._detail("Embedding Model", active.get("embedding_model", ""))
                        self._detail(
                            "Files / Chunks",
                            f"{active.get('file_count', 0)} / {active.get('chunk_count', 0)}",
                        )
                        self._detail("Created", active.get("created_at", "")[:19])
                    if active.get("source_folders"):
                        ui.label("Source Folders").classes("text-sm text-gray-400 mt-2")
                        for f in active["source_folders"]:
                            ui.label(f"  📁 {f}").classes("text-sm text-gray-300 font-mono")
                else:
                    ui.label("No active RAG selected").classes("text-gray-400")
                    ui.label("Create a knowledge base or select one from the Manage page.").classes(
                        "text-sm text-gray-500"
                    )

        except RuntimeError:
            pass  # UI elements deleted (page navigated away)
        except Exception as exc:
            logger.warning("Dashboard refresh error: %s", exc)

        # Metrics at-a-glance — update labels in-place (no clear/rebuild)
        try:
            dashboard = await run.io_bound(lambda: _safe_call("get_metrics_dashboard")) or {}
            idx = dashboard.get("indexing_aggregates") or {}
            srch = dashboard.get("search_aggregates") or {}
            emb = dashboard.get("embedding_aggregates") or {}
            self._metric_idx_runs.set_text(str(idx.get("total_runs", 0)))
            self._metric_search_queries.set_text(str(srch.get("total_queries", 0)))
            self._metric_avg_latency.set_text(f"{srch.get('avg_latency_ms', 0) or 0:.0f} ms")
            self._metric_embed_batches.set_text(str(emb.get("total_batches", 0)))
        except Exception:
            pass  # metrics not available yet

    @staticmethod
    def _detail(key: str, value: str):
        with ui.column().classes("gap-0"):
            ui.label(key).classes("text-xs text-gray-400")
            ui.label(value).classes("text-sm font-bold text-gray-200")


# ---------------------------------------------------------------------------
# Page: Search
# ---------------------------------------------------------------------------


class SearchPage:
    """Semantic search page."""

    def __init__(self):
        self._query_input = None
        self._top_k_input = None
        self._results_container = None
        self._search_btn = None
        self._status_label = None

    def destroy(self):
        """No active timers to cancel."""

    def build(self):
        with ui.column().classes("w-full gap-4"):
            ui.label("Search").classes("text-2xl font-bold text-white")
            ui.label("Hybrid semantic search across your knowledge base").classes(
                "text-sm text-gray-400 -mt-2"
            )

            # Search bar
            with ui.row().classes("w-full gap-2 items-end"):
                self._query_input = (
                    ui.input("Search query…")
                    .classes("flex-grow")
                    .props("outlined dark dense")
                    .on("keydown.enter", self._do_search)
                )
                self._top_k_input = (
                    ui.number("Top N", value=5, min=1, max=50, step=1)
                    .classes("w-24")
                    .props("outlined dark dense")
                )
                self._search_btn = ui.button("Search", on_click=self._do_search).props(
                    "color=primary"
                )

            self._status_label = ui.label("").classes("text-sm text-gray-400")
            self._results_container = ui.column().classes("w-full gap-3")

    async def _do_search(self):
        query = self._query_input.value
        if not query or not query.strip():
            ui.notify("Enter a search query", type="warning")
            return

        self._search_btn.props("loading")
        self._status_label.set_text("Searching…")
        self._results_container.clear()
        with self._results_container:
            _skeleton_search_results(int(self._top_k_input.value or 5))

        try:
            top_k = int(self._top_k_input.value or 5)
            results = await run.io_bound(lambda: _safe_call("search", query=query, top_k=top_k))

            self._results_container.clear()
            if not results:
                self._status_label.set_text("No results found.")
                with self._results_container:
                    ui.label("No matching documents found.").classes("text-gray-400 p-4")
            else:
                self._status_label.set_text(f"{len(results)} result(s)")
                with self._results_container:
                    for i, r in enumerate(results, 1):
                        self._render_result(i, r)
        except Exception as exc:
            self._status_label.set_text(f"Error: {exc}")
            ui.notify(f"Search failed: {exc}", type="negative")
        finally:
            self._search_btn.props(remove="loading")

    @staticmethod
    def _render_result(rank: int, r: dict):
        score = r.get("score", 0)
        score_pct = round(score * 100, 1)
        score_class = (
            "score-high" if score >= 0.6 else "score-mid" if score >= 0.35 else "score-low"
        )

        with ui.card().classes(
            "w-full bg-[#1d232e] border border-[#313949] rounded-lg result-card p-4"
        ):
            with ui.row().classes("w-full justify-between items-center"):
                with ui.row().classes("items-center gap-2"):
                    ui.label(f"#{rank}").classes("text-lg font-bold text-blue-400")
                    ui.label(r.get("source_file", "")).classes("text-sm font-mono text-gray-300")
                    ui.label(f"chunk {r.get('chunk_index', 0)}").classes("text-xs text-gray-500")
                ui.label(f"{score_pct}%").classes(f"text-lg font-bold {score_class}")

            # Score bar
            ui.linear_progress(value=score).classes("mt-1").props(
                f"color={'green' if score >= 0.6 else 'orange' if score >= 0.35 else 'red'}"
            )

            # Content
            ui.label(r.get("text", "")).classes(
                "text-sm text-gray-300 mt-2 whitespace-pre-wrap select-all"
            ).style("max-height: 200px; overflow-y: auto;")


# ---------------------------------------------------------------------------
# Page: Manage
# ---------------------------------------------------------------------------


class ManagePage:
    """RAG CRUD management page."""

    def __init__(self):
        self._table = None
        self._table_loading = None
        self._detail_container = None
        self._selected_rag = None
        self._deferred_timer = None

    def destroy(self):
        """Cancel deferred load timer when leaving the page."""
        if self._deferred_timer is not None:
            self._deferred_timer.cancel()
            self._deferred_timer = None

    def build(self):
        with ui.column().classes("w-full gap-4"):
            ui.label("Manage").classes("text-2xl font-bold text-white")
            ui.label("Create, configure, and manage your RAG knowledge bases").classes(
                "text-sm text-gray-400 -mt-2"
            )

            # Action buttons
            with ui.row().classes("gap-2"):
                ui.button("＋ New Knowledge Base", on_click=self._show_create_dialog).props(
                    "color=primary"
                )
                ui.button("📥 Import .rag…", on_click=self._show_import_dialog).props(
                    "outline color=white"
                )

            # RAG table
            self._table = (
                ui.table(
                    columns=[
                        {
                            "name": "name",
                            "label": "Name",
                            "field": "name",
                            "sortable": True,
                            "align": "left",
                        },
                        {"name": "status", "label": "Status", "field": "status", "align": "left"},
                        {
                            "name": "description",
                            "label": "Description",
                            "field": "description",
                            "align": "left",
                        },
                        {
                            "name": "files",
                            "label": "Files",
                            "field": "files",
                            "sortable": True,
                            "align": "right",
                        },
                        {
                            "name": "chunks",
                            "label": "Chunks",
                            "field": "chunks",
                            "sortable": True,
                            "align": "right",
                        },
                        {"name": "model", "label": "Model", "field": "model", "align": "left"},
                        {
                            "name": "created",
                            "label": "Created",
                            "field": "created",
                            "align": "left",
                        },
                    ],
                    rows=[],
                    row_key="name",
                    selection="single",
                    on_select=self._on_select,
                )
                .classes("w-full")
                .props("dark flat bordered dense")
            )

            # Detail panel
            self._detail_container = ui.column().classes(
                "w-full bg-[#1d232e] rounded-xl border border-[#313949] p-6 gap-3"
            )
            with self._detail_container:
                ui.label("Select a RAG to see details").classes("text-gray-400")

            # Show skeleton rows in table area while loading
            self._table_loading = ui.column().classes("w-full gap-1 px-2 py-2")
            with self._table_loading:
                _skeleton_table_rows(cols=5, rows=4)

            # Deferred async refresh
            self._deferred_timer = ui.timer(0.1, self.refresh, once=True)

    async def refresh(self):
        try:
            rags = await run.io_bound(lambda: _safe_call("list_rags")) or []
            rows = []
            for r in rags:
                status_parts = []
                if r.get("is_active"):
                    status_parts.append("✦ Active")
                if r.get("detached"):
                    status_parts.append("🔒 Detached")
                elif r.get("is_imported"):
                    status_parts.append("📥 Imported")
                else:
                    status_parts.append("📁 Local")

                rows.append(
                    {
                        "name": r["name"],
                        "status": " · ".join(status_parts),
                        "description": r.get("description", ""),
                        "files": r.get("file_count", 0),
                        "chunks": r.get("chunk_count", 0),
                        "model": r.get("embedding_model", ""),
                        "created": (r.get("created_at", "") or "")[:10],
                    }
                )
            self._table.rows = rows
            self._table.update()
            # Hide table skeleton once data is loaded
            if self._table_loading is not None:
                self._table_loading.set_visibility(False)
        except Exception as exc:
            logger.warning("Manage refresh error: %s", exc)
            if self._table_loading is not None:
                self._table_loading.set_visibility(False)

    async def _on_select(self, e):
        selected = e.selection
        if selected:
            self._selected_rag = selected[0]["name"]
            # Show skeleton in detail panel while loading
            self._detail_container.clear()
            with self._detail_container:
                _skeleton_detail_panel()
            await self._show_detail(selected[0]["name"])
        else:
            self._selected_rag = None
            self._detail_container.clear()
            with self._detail_container:
                ui.label("Select a RAG to see details").classes("text-gray-400")

    async def _show_detail(self, name: str):
        try:
            rag = await run.io_bound(lambda: _safe_call("get_rag", name))
        except Exception as exc:
            ui.notify(f"Error loading RAG: {exc}", type="negative")
            return

        self._detail_container.clear()
        with self._detail_container:
            # Header with badges
            with ui.row().classes("items-center gap-2"):
                ui.label(rag["name"]).classes("text-xl font-bold text-white")
                if rag.get("is_active"):
                    ui.html('<span class="badge badge-active">ACTIVE</span>')
                if rag.get("detached"):
                    ui.html('<span class="badge badge-detached">DETACHED</span>')
                elif rag.get("is_imported"):
                    ui.html('<span class="badge badge-imported">IMPORTED</span>')
                else:
                    ui.html('<span class="badge badge-local">LOCAL</span>')

            if rag.get("description"):
                ui.label(rag["description"]).classes("text-sm text-gray-300")

            # Properties
            with ui.row().classes("gap-8 flex-wrap"):
                with ui.column().classes("gap-1"):
                    ui.label("Embedding Model").classes("text-xs text-gray-400")
                    ui.label(rag.get("embedding_model", "")).classes(
                        "text-sm font-bold text-gray-200"
                    )
                with ui.column().classes("gap-1"):
                    ui.label("Files / Chunks").classes("text-xs text-gray-400")
                    ui.label(f"{rag.get('file_count', 0)} / {rag.get('chunk_count', 0)}").classes(
                        "text-sm font-bold text-gray-200"
                    )
                with ui.column().classes("gap-1"):
                    ui.label("Created").classes("text-xs text-gray-400")
                    ui.label((rag.get("created_at", "") or "")[:19]).classes(
                        "text-sm font-bold text-gray-200"
                    )

            # Source folders — editable
            is_detached = rag.get("detached", False)
            current_folders = list(rag.get("source_folders") or [])

            ui.label("Source Folders").classes("text-sm text-gray-400 mt-2")
            folders_container = ui.column().classes("w-full gap-1")

            def _refresh_folder_list():
                folders_container.clear()
                with folders_container:
                    if not current_folders:
                        ui.label("No source folders configured").classes(
                            "text-sm text-gray-500 italic ml-2"
                        )
                    for f in current_folders:
                        with ui.row().classes("items-center gap-1 w-full"):
                            ui.label(f"📁 {f}").classes("text-sm text-gray-300 font-mono flex-grow")
                            if not is_detached:
                                ui.button(
                                    icon="close",
                                    on_click=lambda _f=f: _remove_folder(_f),
                                ).props("flat dense round size=sm color=red").tooltip(
                                    "Remove folder"
                                )

            async def _remove_folder(folder: str):
                if folder in current_folders:
                    current_folders.remove(folder)
                    try:
                        await run.io_bound(
                            lambda: _safe_call(
                                "update_rag", name, source_folders=list(current_folders)
                            )
                        )
                        ui.notify("Removed folder", type="positive")
                        _refresh_folder_list()
                        await self.refresh()
                    except Exception as exc:
                        # Revert on failure
                        current_folders.append(folder)
                        ui.notify(f"Error: {exc}", type="negative")
                        _refresh_folder_list()

            async def _add_folder(path: str):
                path = path.strip()
                if not path:
                    return
                if path in current_folders:
                    ui.notify("Folder already added", type="warning")
                    return
                current_folders.append(path)
                _refresh_folder_list()
                folder_input.value = ""
                try:
                    await run.io_bound(
                        lambda: _safe_call("update_rag", name, source_folders=list(current_folders))
                    )
                    ui.notify("Added folder", type="positive")
                    await self.refresh()
                except Exception as exc:
                    current_folders.remove(path)
                    ui.notify(f"Error: {exc}", type="negative")
                    _refresh_folder_list()

            _refresh_folder_list()

            if not is_detached:
                with ui.row().classes("w-full items-center gap-2"):
                    folder_input = (
                        ui.input("Add folder path…")
                        .props("outlined dark dense")
                        .classes("flex-grow")
                    )

                    def _open_browser():
                        self._show_folder_browser(
                            on_select=_add_folder,
                            start_path=folder_input.value.strip() or None,
                        )

                    ui.button(icon="folder_open", on_click=_open_browser).props(
                        "flat dense color=primary"
                    ).tooltip("Browse folders")
                    ui.button("Add", on_click=lambda: _add_folder(folder_input.value)).props(
                        "flat dense color=primary"
                    )

            # Action buttons
            ui.separator().classes("my-2")
            with ui.row().classes("gap-2 flex-wrap"):
                if not rag.get("is_active"):
                    ui.button(
                        "Set as Active",
                        on_click=lambda n=name: self._set_active(n),
                    ).props("color=primary outline")

                ui.button(
                    "Export .rag",
                    on_click=lambda n=name: self._show_export_dialog(n),
                ).props("outline color=white")

                if rag.get("detached"):
                    ui.button(
                        "Re-attach",
                        on_click=lambda n=name: self._attach(n),
                    ).props("outline color=orange")
                elif not rag.get("is_imported"):
                    ui.button(
                        "Detach",
                        on_click=lambda n=name: self._detach(n),
                    ).props("outline color=orange")

                ui.button(
                    "Delete",
                    on_click=lambda n=name: self._confirm_delete(n),
                ).props("color=negative outline")

    async def _set_active(self, name: str):
        try:
            await run.io_bound(lambda: _safe_call("switch_rag", name))
            ui.notify(f"Switched to {name}", type="positive")
            await self.refresh()
            await self._show_detail(name)
        except Exception as exc:
            ui.notify(f"Error: {exc}", type="negative")

    async def _detach(self, name: str):
        try:
            await run.io_bound(lambda: _safe_call("detach_rag", name))
            ui.notify(f"Detached {name}", type="positive")
            await self.refresh()
            await self._show_detail(name)
        except Exception as exc:
            ui.notify(f"Error: {exc}", type="negative")

    async def _attach(self, name: str):
        try:
            await run.io_bound(lambda: _safe_call("attach_rag", name))
            ui.notify(f"Re-attached {name}", type="positive")
            await self.refresh()
            await self._show_detail(name)
        except Exception as exc:
            ui.notify(f"Error: {exc}", type="negative")

    def _confirm_delete(self, name: str):
        with ui.dialog() as dlg, ui.card().classes("bg-[#1d232e] min-w-[400px] max-w-[90vw]"):
            ui.label(f"Delete '{name}'?").classes("text-lg font-bold text-white")
            ui.label("This will permanently delete the RAG database and all indexed data.").classes(
                "text-sm text-gray-400"
            )
            with ui.row().classes("w-full justify-end gap-2 mt-4"):
                ui.button("Cancel", on_click=dlg.close).props("flat color=white")
                ui.button(
                    "Delete",
                    on_click=lambda: self._do_delete(name, dlg),
                ).props("color=negative")
        dlg.open()

    async def _do_delete(self, name: str, dlg):
        try:
            await run.io_bound(lambda: _safe_call("delete_rag", name))
            dlg.close()
            ui.notify(f"Deleted {name}", type="positive")
            self._detail_container.clear()
            with self._detail_container:
                ui.label("Select a RAG to see details").classes("text-gray-400")
            await self.refresh()
        except Exception as exc:
            ui.notify(f"Error: {exc}", type="negative")

    def _show_create_dialog(self):
        name_input = None
        desc_input = None
        folders_list = []
        model_input = None

        with ui.dialog() as dlg, ui.card().classes("bg-[#1d232e] min-w-[500px] max-w-[90vw]"):
            ui.label("Create New Knowledge Base").classes("text-xl font-bold text-white")

            name_input = ui.input("Name").props("outlined dark dense").classes("w-full")
            desc_input = (
                ui.input("Description (optional)").props("outlined dark dense").classes("w-full")
            )

            from rag_kb.models import get_embedding_model_names

            model_input = (
                ui.select(
                    options=get_embedding_model_names(),
                    value="paraphrase-multilingual-MiniLM-L12-v2",
                    label="Embedding Model",
                )
                .props("outlined dark dense")
                .classes("w-full")
            )

            ui.label("Source Folders").classes("text-sm text-gray-400 mt-2")
            folders_container = ui.column().classes("w-full gap-1")

            with ui.row().classes("w-full items-center gap-2"):
                folder_input = (
                    ui.input("Add folder path…").props("outlined dark dense").classes("flex-grow")
                )

                def _open_browser():
                    self._show_folder_browser(
                        on_select=_add_folder_path,
                        start_path=folder_input.value.strip() or None,
                    )

                ui.button(icon="folder_open", on_click=_open_browser).props(
                    "flat dense color=primary"
                ).tooltip("Browse folders")

            def _add_folder_path(path: str):
                if path and path.strip() and path not in folders_list:
                    folders_list.append(path.strip())
                    folder_input.value = ""
                    _refresh_folders()

            def add_folder():
                _add_folder_path(folder_input.value)

            def remove_folder(f):
                if f in folders_list:
                    folders_list.remove(f)
                    _refresh_folders()

            def _refresh_folders():
                folders_container.clear()
                with folders_container:
                    for f in folders_list:
                        with ui.row().classes("items-center gap-1"):
                            ui.label(f"📁 {f}").classes("text-sm text-gray-300 font-mono flex-grow")
                            ui.button(
                                icon="close",
                                on_click=lambda _f=f: remove_folder(_f),
                            ).props("flat dense round size=sm color=red")

            ui.button("Add Folder", on_click=add_folder).props("flat dense color=primary")

            with ui.row().classes("w-full justify-end gap-2 mt-4"):
                ui.button("Cancel", on_click=dlg.close).props("flat color=white")
                ui.button(
                    "Create",
                    on_click=lambda: self._do_create(
                        name_input.value,
                        desc_input.value,
                        list(folders_list),
                        model_input.value,
                        dlg,
                    ),
                ).props("color=primary")
        dlg.open()

    def _show_folder_browser(self, on_select, start_path: str | None = None):
        """Open a server-side folder browser dialog.

        Lets the user navigate the file system on the server and pick a
        directory.  *on_select* is called with the chosen absolute path.
        """
        # Determine starting directory
        if start_path and Path(start_path).is_dir():
            current_path = str(Path(start_path).resolve())
        else:
            current_path = str(Path.home())

        path_label = None
        listing_container = None

        def _list_dir(dirpath: str) -> list[dict]:
            """Return sorted directory entries for *dirpath*."""
            entries = []
            try:
                for entry in sorted(
                    Path(dirpath).iterdir(), key=lambda e: (not e.is_dir(), e.name.lower())
                ):
                    # Skip hidden/system dirs
                    if entry.name.startswith("."):
                        continue
                    try:
                        is_dir = entry.is_dir()
                    except PermissionError:
                        continue
                    entries.append(
                        {
                            "name": entry.name,
                            "path": str(entry),
                            "is_dir": is_dir,
                        }
                    )
            except PermissionError:
                pass
            return entries

        def _get_drives() -> list[str]:
            """Return available drive letters on Windows."""
            drives = []
            if sys.platform == "win32":
                import string

                for letter in string.ascii_uppercase:
                    dp = f"{letter}:\\"
                    if Path(dp).exists():
                        drives.append(dp)
            return drives

        def _navigate(new_path: str):
            nonlocal current_path
            resolved = Path(new_path).resolve()
            if resolved.is_dir():
                current_path = str(resolved)
                _refresh_listing()

        def _go_up():
            parent = str(Path(current_path).parent)
            if parent != current_path:
                _navigate(parent)

        def _refresh_listing():
            if path_label:
                path_label.set_text(current_path)
            if listing_container is None:
                return
            listing_container.clear()
            entries = _list_dir(current_path)
            with listing_container:
                if not entries:
                    ui.label("Empty or inaccessible").classes("text-gray-500 text-sm italic p-2")
                for entry in entries:
                    icon = "folder" if entry["is_dir"] else "description"
                    color = "text-amber-400" if entry["is_dir"] else "text-gray-500"
                    with ui.row().classes(
                        "items-center gap-2 px-2 py-1 w-full rounded "
                        "hover:bg-white/5 cursor-pointer"
                    ):
                        ui.icon(icon).classes(f"{color} text-lg")
                        if entry["is_dir"]:
                            ui.label(entry["name"]).classes("text-sm text-gray-200 flex-grow").on(
                                "click", lambda _e=entry: _navigate(_e["path"])
                            )
                        else:
                            ui.label(entry["name"]).classes("text-sm text-gray-500 flex-grow")

        with (
            ui.dialog() as browser_dlg,
            ui.card().classes("bg-[#1d232e] min-w-[550px] max-w-[90vw]"),
        ):
            ui.label("Select Folder").classes("text-xl font-bold text-white")

            # Navigation bar
            with ui.row().classes("w-full items-center gap-1"):
                ui.button(icon="arrow_upward", on_click=_go_up).props(
                    "flat dense round color=white"
                ).tooltip("Go up")

                # Drive buttons on Windows
                drives = _get_drives()
                if drives:
                    for drv in drives:
                        ui.button(
                            drv.rstrip("\\"),
                            on_click=lambda _d=drv: _navigate(_d),
                        ).props("flat dense size=sm color=grey-5")

                ui.button(
                    icon="home",
                    on_click=lambda: _navigate(str(Path.home())),
                ).props("flat dense round color=white").tooltip("Home")

            # Current path display
            path_label = ui.label(current_path).classes(
                "text-xs text-gray-400 font-mono w-full px-1 truncate"
            )

            # Directory listing
            listing_container = (
                ui.scroll_area()
                .classes("w-full border border-gray-700 rounded")
                .style("height: 350px")
            )
            _refresh_listing()

            # Manual path input
            with ui.row().classes("w-full items-center gap-2 mt-2"):
                manual_input = (
                    ui.input("Or type path…").props("outlined dark dense").classes("flex-grow")
                )
                ui.button(
                    "Go",
                    on_click=lambda: _navigate(manual_input.value),
                ).props("flat dense color=primary")

            # Action buttons
            with ui.row().classes("w-full justify-end gap-2 mt-2"):
                ui.button("Cancel", on_click=browser_dlg.close).props("flat color=white")

                async def _select_current():
                    result = on_select(current_path)
                    if asyncio.iscoroutine(result):
                        await result
                    browser_dlg.close()

                ui.button(
                    "Select This Folder",
                    on_click=_select_current,
                ).props("color=primary")

        browser_dlg.open()

    async def _do_create(self, name, description, folders, model, dlg):
        if not name or not name.strip():
            ui.notify("Name is required", type="warning")
            return
        try:
            await run.io_bound(
                lambda: _safe_call(
                    "create_rag",
                    name=name.strip(),
                    folders=folders,
                    description=description or "",
                    embedding_model=model,
                )
            )
            dlg.close()
            ui.notify(f"Created '{name.strip()}'", type="positive")
            await self.refresh()
        except Exception as exc:
            ui.notify(f"Error: {exc}", type="negative")

    def _show_import_dialog(self):
        path_input = None
        name_input = None

        with ui.dialog() as dlg, ui.card().classes("bg-[#1d232e] min-w-[500px] max-w-[90vw]"):
            ui.label("Import .rag File").classes("text-xl font-bold text-white")
            path_input = ui.input("File path (.rag)").props("outlined dark dense").classes("w-full")
            name_input = (
                ui.input("Custom name (optional)").props("outlined dark dense").classes("w-full")
            )

            with ui.row().classes("w-full justify-end gap-2 mt-4"):
                ui.button("Cancel", on_click=dlg.close).props("flat color=white")
                ui.button(
                    "Import",
                    on_click=lambda: self._do_import(path_input.value, name_input.value, dlg),
                ).props("color=primary")
        dlg.open()

    async def _do_import(self, path, name, dlg):
        if not path or not path.strip():
            ui.notify("File path is required", type="warning")
            return

        def _do_long_import():
            client = _make_client()
            try:
                return client.import_rag(path.strip(), name=name or None)
            finally:
                client.close()

        try:
            result = await run.io_bound(_do_long_import)
            dlg.close()
            ui.notify(f"Imported as '{result.get('name', '')}'", type="positive")
            await self.refresh()
        except Exception as exc:
            ui.notify(f"Error: {exc}", type="negative")

    def _show_export_dialog(self, name: str):
        path_input = None

        with ui.dialog() as dlg, ui.card().classes("bg-[#1d232e] min-w-[500px] max-w-[90vw]"):
            ui.label(f"Export '{name}'").classes("text-xl font-bold text-white")
            path_input = (
                ui.input("Output path (.rag)").props("outlined dark dense").classes("w-full")
            )
            path_input.value = f"{name}.rag"

            with ui.row().classes("w-full justify-end gap-2 mt-4"):
                ui.button("Cancel", on_click=dlg.close).props("flat color=white")
                ui.button(
                    "Export",
                    on_click=lambda: self._do_export(name, path_input.value, dlg),
                ).props("color=primary")
        dlg.open()

    async def _do_export(self, name, path, dlg):
        if not path or not path.strip():
            ui.notify("Output path is required", type="warning")
            return

        def _do_long_export():
            client = _make_client()
            try:
                return client.export_rag(name, path.strip())
            finally:
                client.close()

        try:
            result = await run.io_bound(_do_long_export)
            dlg.close()
            ui.notify(f"Exported to {result.get('path', path)}", type="positive")
        except Exception as exc:
            ui.notify(f"Error: {exc}", type="negative")


# ---------------------------------------------------------------------------
# Page: Indexing
# ---------------------------------------------------------------------------


class IndexingPage:
    """Document indexing and file change detection page."""

    _FILES_PAGE_SIZE = 50
    _PROGRESS_POLL_INTERVAL = 2.0  # seconds between progress polls

    def __init__(self):
        self._info_container = None
        self._progress_container = None
        self._progress_bar = None
        self._progress_label = None
        self._progress_file = None
        self._progress_phase = None
        self._files_table = None
        self._files_pagination_label = None
        self._files_prev_btn = None
        self._files_next_btn = None
        self._files_filter_input = None
        self._changes_container = None
        self._index_btn = None
        self._reindex_btn = None
        self._check_btn = None
        self._is_indexing = False
        self._seen_live_progress = False  # True once daemon reports live indexing
        self._files_page = 0  # current 0-based page index
        self._files_total = 0  # total matching files
        self._files_filter = ""  # current filter string
        self._progress_timer = None
        self._deferred_timer = None

    def destroy(self):
        """Cancel progress polling timer when leaving the page."""
        if self._deferred_timer is not None:
            self._deferred_timer.cancel()
            self._deferred_timer = None
        self._stop_progress_polling()

    def build(self):
        with ui.column().classes("w-full gap-4"):
            ui.label("Indexing").classes("text-2xl font-bold text-white")
            ui.label("Index documents and manage file changes").classes(
                "text-sm text-gray-400 -mt-2"
            )

            # Active RAG info
            self._info_container = ui.column().classes(
                "w-full bg-[#1d232e] rounded-xl border border-[#313949] p-4 gap-2"
            )

            # Action buttons
            with ui.row().classes("gap-2"):
                self._index_btn = ui.button("📄 Incremental Index", on_click=self._do_index).props(
                    "color=primary"
                )
                self._reindex_btn = ui.button("🔄 Full Reindex", on_click=self._do_reindex).props(
                    "color=primary outline"
                )
                self._check_btn = ui.button(
                    "🔍 Check for Changes", on_click=self._check_changes
                ).props("outline color=white")

            # Progress section
            self._progress_container = ui.column().classes(
                "w-full bg-[#1d232e] rounded-xl border border-[#313949] p-4 gap-2"
            )
            self._progress_container.set_visibility(False)

            with self._progress_container:
                self._progress_phase = ui.label("").classes("text-sm font-bold")
                with ui.row().classes("w-full items-center gap-2"):
                    self._progress_bar = ui.linear_progress(value=0).classes("flex-grow")
                    self._progress_label = ui.label("0%").classes(
                        "text-sm text-gray-300 w-16 text-right"
                    )
                self._progress_file = ui.label("").classes("text-sm text-gray-400")

            # File changes section
            self._changes_container = ui.column().classes("w-full gap-2")

            # Indexed files section with filter + pagination
            with ui.column().classes("w-full gap-2"):
                with ui.row().classes("w-full items-center gap-2"):
                    ui.label("Indexed Files").classes("text-lg font-semibold text-white")
                    ui.space()
                    self._files_filter_input = (
                        ui.input(
                            placeholder="Filter files…",
                            on_change=lambda e: self._on_filter_change(e.value),
                        )
                        .props("outlined dark dense clearable")
                        .classes("w-64")
                    )

                self._files_table = (
                    ui.table(
                        columns=[
                            {
                                "name": "num",
                                "label": "#",
                                "field": "num",
                                "align": "right",
                                "sortable": False,
                            },
                            {
                                "name": "file",
                                "label": "File",
                                "field": "file",
                                "align": "left",
                                "sortable": True,
                            },
                            {
                                "name": "chunks",
                                "label": "Chunks",
                                "field": "chunks",
                                "align": "right",
                                "sortable": True,
                            },
                        ],
                        rows=[],
                        row_key="file",
                    )
                    .classes("w-full")
                    .props("dark flat bordered dense")
                )

                with ui.row().classes("w-full items-center justify-between"):
                    self._files_pagination_label = ui.label("").classes("text-sm text-gray-400")
                    with ui.row().classes("gap-1"):
                        self._files_prev_btn = ui.button(
                            "← Prev",
                            on_click=self._files_prev_page,
                        ).props("flat dense no-caps size=sm color=primary")
                        self._files_next_btn = ui.button(
                            "Next →",
                            on_click=self._files_next_page,
                        ).props("flat dense no-caps size=sm color=primary")

            # Show skeleton placeholder, defer data fetch
            with self._info_container:
                _skeleton_line("180px", "22px")  # title
                with ui.row().classes("gap-6"):
                    _skeleton_line("100px", "14px")
                    _skeleton_line("100px", "14px")
                    _skeleton_line("120px", "14px")
                _skeleton_line("160px", "10px")
            self._deferred_timer = ui.timer(0.1, self.refresh, once=True)

    async def refresh(self):
        try:
            status = await run.io_bound(lambda: _safe_call("get_index_status")) or {}
            active = status.get("active_rag")

            self._info_container.clear()
            with self._info_container:
                if active:
                    ui.label(f"Active RAG: {active}").classes("text-lg font-bold text-white")
                    with ui.row().classes("gap-6"):
                        ui.label(f"Files: {status.get('total_files', 0)}").classes(
                            "text-sm text-gray-300"
                        )
                        ui.label(f"Chunks: {status.get('total_chunks', 0)}").classes(
                            "text-sm text-gray-300"
                        )
                        watcher = "Running" if status.get("watcher_running") else "Stopped"
                        ui.label(f"Watcher: {watcher}").classes("text-sm text-gray-300")
                    if status.get("last_indexed"):
                        ui.label(f"Last indexed: {status['last_indexed']}").classes(
                            "text-xs text-gray-500"
                        )
                    errors = status.get("errors", [])
                    if errors:
                        for e in errors[:5]:
                            ui.label(f"⚠ {e}").classes("text-xs text-orange-400")
                else:
                    ui.label("No active RAG selected").classes("text-gray-400")

            # Check if daemon reports a live indexing operation
            live = status.get("indexing")
            if live and live.get("is_indexing") and not self._is_indexing:
                # Another client (or a previous page load) started indexing
                self._is_indexing = True
                self._show_progress_from_status(live)
                self._start_progress_polling()

            # Disable buttons appropriately
            enabled = bool(active) and not self._is_indexing
            self._index_btn.set_enabled(enabled)
            self._reindex_btn.set_enabled(enabled)
            self._check_btn.set_enabled(bool(active))

            # Refresh files table (paginated)
            await self._refresh_files_table()

        except RuntimeError:
            # UI elements deleted (client navigated away / page reload).
            pass
        except Exception as exc:
            logger.warning("Indexing refresh error: %s", exc)

    async def _refresh_files_table(self):
        """Fetch the current page of files from the daemon and update the table."""
        try:
            ps = self._FILES_PAGE_SIZE
            result = await run.io_bound(
                lambda: _safe_call(
                    "list_indexed_files",
                    offset=self._files_page * ps,
                    limit=ps,
                    filter=self._files_filter,
                )
            )
            if result is None:
                return  # collection may be missing during reindex
            files = result.get("files", [])
            total = result.get("total", len(files))
            self._files_total = total
            offset = self._files_page * ps

            self._files_table.rows = [
                {"num": offset + i + 1, "file": f["file"], "chunks": f["chunk_count"]}
                for i, f in enumerate(files)
            ]
            self._files_table.update()

            # Update pagination controls
            if total == 0:
                self._files_pagination_label.set_text("No files")
            else:
                last = min(offset + ps, total)
                page_num = self._files_page + 1
                total_pages = max(1, (total + ps - 1) // ps)
                self._files_pagination_label.set_text(
                    f"Showing {offset + 1}–{last} of {total} files  •  "
                    f"Page {page_num}/{total_pages}"
                )

            self._files_prev_btn.set_enabled(self._files_page > 0)
            self._files_next_btn.set_enabled(offset + ps < total)
        except Exception:
            pass

    async def _on_filter_change(self, value: str):
        """Handle filter input changes — reset to page 0 and refresh."""
        self._files_filter = value or ""
        self._files_page = 0
        await self._refresh_files_table()

    async def _files_prev_page(self):
        if self._files_page > 0:
            self._files_page -= 1
            await self._refresh_files_table()

    async def _files_next_page(self):
        ps = self._FILES_PAGE_SIZE
        if (self._files_page + 1) * ps < self._files_total:
            self._files_page += 1
            await self._refresh_files_table()

    async def _do_index(self):
        await self._run_index(full=False)

    async def _do_reindex(self):
        await self._run_index(full=True)

    def _show_progress_from_status(self, live: dict):
        """Update progress UI elements from a daemon ``indexing`` status dict."""
        self._seen_live_progress = True
        self._progress_container.set_visibility(True)
        processed = live.get("processed_files", 0)
        total = live.get("total_files", 1) or 1
        pct = min(live.get("progress", 0.0), 1.0)
        phase = live.get("status", "indexing")
        current_file = live.get("current_file", "")

        self._progress_bar.value = pct
        self._progress_label.set_text(f"{int(pct * 100)}%")
        self._progress_phase.set_text(f"📄 {phase} — {processed}/{total}")
        self._progress_file.set_text(current_file)

    def _start_progress_polling(self):
        """Start a NiceGUI timer that polls daemon for live indexing status."""
        if self._progress_timer is not None:
            return  # already polling

        async def _poll_progress():
            try:
                status = await run.io_bound(lambda: _safe_call("get_index_status"))
                if status is None:
                    return  # transient RPC failure — retry next tick
                live = status.get("indexing")
                if live and live.get("is_indexing"):
                    self._show_progress_from_status(live)
                elif self._seen_live_progress:
                    # Indexing has finished (we saw it running, now it's gone)
                    self._stop_progress_polling()
                    self._is_indexing = False

                    # Check if daemon reports errors in last_status
                    errors = status.get("errors", [])
                    if errors:
                        self._progress_phase.set_text("⚠️ Done with errors")
                        self._progress_file.set_text(
                            f"{len(errors)} error(s): {errors[0]}"
                            + (" …" if len(errors) > 1 else "")
                        )
                    else:
                        self._progress_phase.set_text("✅ Done")
                        self._progress_file.set_text("Indexing complete")

                    self._progress_bar.value = 1.0
                    self._progress_label.set_text("100%")
                    self._index_btn.props(remove="loading")
                    self._index_btn.set_enabled(True)
                    self._reindex_btn.set_enabled(True)
                    self._deferred_timer = ui.timer(0.1, self.refresh, once=True)
                # else: daemon hasn't started indexing yet — keep polling
            except RuntimeError:
                # UI elements deleted (client navigated away / page reload).
                self._stop_progress_polling()
                self._is_indexing = False
            except Exception as exc:
                # Collection may be temporarily missing during full reindex;
                # just skip this poll cycle — the timer will retry.
                logger.debug("Progress poll error (will retry): %s", exc)

        self._progress_timer = ui.timer(self._PROGRESS_POLL_INTERVAL, _poll_progress)

    def _stop_progress_polling(self):
        """Cancel the progress polling timer."""
        if self._progress_timer is not None:
            self._progress_timer.cancel()
            self._progress_timer = None

    async def _run_index(self, full: bool):
        self._is_indexing = True
        self._index_btn.set_enabled(False)
        self._reindex_btn.set_enabled(False)
        self._index_btn.props("loading")
        self._progress_container.set_visibility(True)
        self._progress_bar.value = 0
        self._progress_label.set_text("0%")
        self._progress_phase.set_text("⏳ Starting…")
        self._progress_file.set_text("")
        self._seen_live_progress = False

        # Start polling the daemon for live progress updates.
        # This runs on the NiceGUI event loop so UI updates push immediately.
        self._start_progress_polling()

        def _do_long_index():
            """Run indexing on a dedicated client so we don't block the
            shared connection used by auto-refresh timers."""
            client = _make_client()
            try:
                return client.index(full=full)
            finally:
                client.close()

        try:
            result = await run.io_bound(_do_long_index)
            files = result.get("processed_files", 0)
            chunks = result.get("total_chunks", 0)
            duration = result.get("duration_seconds", 0)
            errors = result.get("errors", [])

            if errors:
                self._progress_phase.set_text(
                    f"⚠️ Done with {len(errors)} error(s)"
                )
            else:
                self._progress_phase.set_text("✅ Done")
            self._progress_bar.value = 1.0
            self._progress_label.set_text("100%")
            self._progress_file.set_text(
                f"{files} files, {chunks} chunks in {duration:.1f}s"
                + (f" ({len(errors)} errors)" if errors else "")
            )

            # Show phase timing breakdown if available
            phase_keys = [
                "scan_seconds",
                "parse_seconds",
                "chunk_seconds",
                "embed_seconds",
                "upsert_seconds",
                "manifest_seconds",
            ]
            phase_parts = [
                f"{k.replace('_seconds', '')}: {result.get(k, 0):.2f}s"
                for k in phase_keys
                if result.get(k, 0) > 0
            ]
            if phase_parts:
                self._progress_file.set_text(
                    self._progress_file.text + "  |  " + "  ".join(phase_parts)
                )

            # Show errors in a visible block so they can't be missed
            if errors:
                with self._progress_container, ui.expansion(
                    f"⚠ {len(errors)} indexing error(s)",
                    icon="warning",
                ).classes("w-full text-orange-400").props("dense"):
                        for err in errors[:20]:
                            ui.label(err).classes("text-xs text-orange-300 font-mono")
                        if len(errors) > 20:
                            ui.label(f"… and {len(errors) - 20} more").classes(
                                "text-xs text-gray-500"
                            )

            ui.notify(
                f"Indexing complete: {files} files, {chunks} chunks"
                + (f" ({len(errors)} errors)" if errors else ""),
                type="warning" if errors else "positive",
            )
        except RuntimeError:
            # UI elements deleted (client navigated away / page reload).
            # Indexing itself continues in the daemon — nothing to clean up.
            logger.debug("Indexing UI context lost (client disconnected)")
        except Exception as exc:
            try:
                self._progress_phase.set_text("❌ Error")
                self._progress_file.set_text(str(exc))
                ui.notify(f"Indexing failed: {exc}", type="negative", timeout=10000)
            except RuntimeError:
                pass  # UI gone
        finally:
            self._stop_progress_polling()
            self._is_indexing = False
            try:
                self._index_btn.props(remove="loading")
                self._index_btn.set_enabled(True)
                self._reindex_btn.set_enabled(True)
                await self.refresh()
            except RuntimeError:
                pass  # UI gone

    async def _check_changes(self):
        self._check_btn.props("loading")
        self._changes_container.clear()
        try:
            changes = await run.io_bound(lambda: _safe_call("scan_file_changes"))
            new = changes.get("new", [])
            modified = changes.get("modified", [])
            removed = changes.get("removed", [])
            total = len(new) + len(modified) + len(removed)

            with self._changes_container:
                if total == 0:
                    ui.html(
                        '<span class="file-change-badge" style="background:#1e3a20;color:#27ae60">'
                        "✓ No pending changes</span>"
                    )
                else:
                    with ui.row().classes("gap-2 flex-wrap"):
                        if new:
                            ui.html(
                                f'<span class="file-change-badge" style="background:#1e3a20;color:#27ae60">'
                                f"{len(new)} new</span>"
                            )
                        if modified:
                            ui.html(
                                f'<span class="file-change-badge" style="background:#3a2a10;color:#f39c12">'
                                f"{len(modified)} modified</span>"
                            )
                        if removed:
                            ui.html(
                                f'<span class="file-change-badge" style="background:#3a1a18;color:#c0392b">'
                                f"{len(removed)} removed</span>"
                            )

                    with ui.column().classes("gap-0 mt-2 max-h-64 overflow-y-auto"):
                        for f in new[:50]:
                            ui.label(f"  🟢 {f}").classes("text-xs text-green-400 font-mono")
                        for f in modified[:50]:
                            ui.label(f"  🟠 {f}").classes("text-xs text-orange-400 font-mono")
                        for f in removed[:50]:
                            ui.label(f"  🔴 {f}").classes("text-xs text-red-400 font-mono")
        except Exception as exc:
            ui.notify(f"Error checking changes: {exc}", type="negative")
        finally:
            self._check_btn.props(remove="loading")


# ---------------------------------------------------------------------------
# Page: Config
# ---------------------------------------------------------------------------


class ModelsPage:
    """Model management page — browse, download, delete embedding & reranker models."""

    def __init__(self):
        self._container = None
        self._filter_type = "all"
        self._deferred_timer = None

    def destroy(self):
        """Cancel deferred load timer when leaving the page."""
        if self._deferred_timer is not None:
            self._deferred_timer.cancel()
            self._deferred_timer = None

    def build(self):
        with ui.column().classes("w-full gap-4"):
            ui.label("Models").classes("text-2xl font-bold text-white")
            ui.label("Browse, download, and manage embedding & reranker models").classes(
                "text-sm text-gray-400 -mt-2"
            )

            # Filter row
            with ui.row().classes("gap-2 items-center"):
                ui.button("All", on_click=lambda: self._set_filter("all")).props(
                    "flat no-caps size=sm"
                )
                ui.button("Embedding", on_click=lambda: self._set_filter("embedding")).props(
                    "flat no-caps size=sm"
                )
                ui.button("Reranker", on_click=lambda: self._set_filter("reranker")).props(
                    "flat no-caps size=sm"
                )
                ui.space()
                ui.button("↻ Refresh", on_click=self._refresh).props(
                    "flat no-caps size=sm color=primary"
                )

            self._container = ui.column().classes("w-full gap-3")
            with self._container:
                _skeleton_model_cards(3)
            self._deferred_timer = ui.timer(0.1, self._refresh, once=True)

    async def _set_filter(self, type_filter: str):
        self._filter_type = type_filter
        await self._refresh()

    async def _refresh(self):
        if self._container is None:
            return
        try:
            # Show skeleton while loading
            self._container.clear()
            with self._container:
                _skeleton_model_cards(3)
        except RuntimeError:
            return  # UI elements deleted (client navigated away)
        try:
            models = await run.io_bound(lambda: _safe_call("list_models")) or []
            if self._filter_type != "all":
                models = [m for m in models if m.get("type") == self._filter_type]
        except Exception as exc:
            try:
                self._container.clear()
                with self._container:
                    ui.label(f"Error loading models: {exc}").classes("text-red-400")
            except RuntimeError:
                pass  # UI elements deleted
            return

        try:
            self._container.clear()
        except RuntimeError:
            return  # UI elements deleted (client navigated away)
        with self._container:
            if not models:
                ui.label("No models found.").classes("text-gray-400")
                return

            # Group by type
            embedding_models = [m for m in models if m.get("type") == "embedding"]
            reranker_models = [m for m in models if m.get("type") == "reranker"]

            if embedding_models and self._filter_type in ("all", "embedding"):
                ui.label("Embedding Models").classes("text-lg font-semibold text-white mt-2")
                for model in embedding_models:
                    self._build_model_card(model)

            if reranker_models and self._filter_type in ("all", "reranker"):
                ui.label("Reranker Models").classes("text-lg font-semibold text-white mt-4")
                for model in reranker_models:
                    self._build_model_card(model)

    def _build_model_card(self, model: dict):
        status = model.get("status", "available")
        name = model.get("name", "")
        display_name = model.get("display_name", name)
        dims = model.get("dimensions", 0)
        max_tokens = model.get("max_tokens", 0)
        size_mb = model.get("model_size_mb", 0)
        description = model.get("description", "")
        tags = model.get("use_case_tags", [])
        provider = model.get("provider", "local")
        trust_needed = model.get("trust_remote_code", False)
        license_str = model.get("license", "Apache-2.0")
        is_default = model.get("default", False)

        # Status indicator
        if status == "bundled":
            status_color = "green"
            status_icon = "●"
            status_text = "Bundled"
        elif status == "downloaded":
            status_color = "green"
            status_icon = "●"
            status_text = "Downloaded"
        elif status == "api":
            status_color = "blue"
            status_icon = "☁"
            status_text = "API"
        else:
            status_color = "gray"
            status_icon = "○"
            status_text = "Not Downloaded"

        with (
            ui.card().classes(
                "bg-[#1a1f2b] w-full p-4 border border-[#2a3040] hover:border-[#67d3ff44]"
            ),
            ui.row().classes("w-full items-start justify-between"),
        ):
            # Left: info
            with ui.column().classes("gap-1 flex-grow"):
                with ui.row().classes("items-center gap-2"):
                    ui.label(display_name).classes("text-white font-semibold text-base")
                    ui.label(f"{status_icon} {status_text}").classes(
                        f"text-xs text-{status_color}-400"
                    )
                    if is_default:
                        ui.badge("Default", color="primary").props("dense")
                    if trust_needed:
                        ui.badge("trust_remote_code", color="orange").props("dense")

                ui.label(name).classes("text-xs text-gray-500 font-mono")

                if description:
                    ui.label(description).classes("text-sm text-gray-300 mt-1")

                # Tags row
                with ui.row().classes("gap-1 mt-1 flex-wrap"):
                    if model.get("type") == "embedding" and dims:
                        ui.badge(f"{dims}d", color="teal").props("dense outline")
                    if max_tokens:
                        ctx_label = (
                            f"{max_tokens // 1000}K ctx"
                            if max_tokens >= 1000
                            else f"{max_tokens} tokens"
                        )
                        ui.badge(ctx_label, color="teal").props("dense outline")
                    if size_mb and provider == "local":
                        ui.badge(f"~{size_mb} MB", color="grey").props("dense outline")
                    if license_str and license_str != "Apache-2.0":
                        ui.badge(license_str, color="amber").props("dense outline")
                    for tag in tags[:4]:
                        ui.badge(tag, color="grey-7").props("dense outline")

            # Right: actions
            with ui.column().classes("gap-1 items-end min-w-32"):
                if provider == "local":
                    if status in ("bundled", "downloaded"):
                        # Already available
                        ui.label("✓ Ready").classes("text-green-400 text-sm")
                        if not is_default and status != "bundled":
                            ui.button(
                                "🗑 Delete",
                                on_click=lambda n=name: self._confirm_delete(n),
                            ).props("flat dense no-caps color=red size=sm")
                    else:
                        # Not downloaded
                        if trust_needed:
                            ui.button(
                                "⬇ Download (requires trust)",
                                on_click=lambda n=name: self._download_with_trust(n),
                            ).props("dense no-caps color=primary size=sm")
                        else:
                            ui.button(
                                "⬇ Download",
                                on_click=lambda n=name: self._download_model(n),
                            ).props("dense no-caps color=primary size=sm")
                else:
                    # API model
                    ui.label("☁ API-based").classes("text-blue-300 text-xs")
                    ui.label("Set API key in Config").classes("text-gray-500 text-xs")

    async def _download_model(self, model_name: str):
        ui.notify(f"Downloading {model_name}…", type="info")

        def _do_download():
            client = _make_client()
            try:
                return client.download_model(model_name)
            finally:
                client.close()

        try:
            await run.io_bound(_do_download)
            ui.notify(f"✅ {model_name} downloaded!", type="positive")
            await self._refresh()
        except Exception as exc:
            ui.notify(f"Error downloading {model_name}: {exc}", type="negative")

    async def _download_with_trust(self, model_name: str):
        """Download a model that requires trust_remote_code with user consent."""
        with ui.dialog() as dlg, ui.card().classes("bg-[#1d232e] min-w-[400px] max-w-[90vw]"):
            ui.label("⚠️ Trust Remote Code").classes("text-lg font-bold text-amber-400")
            ui.label(
                f"The model '{model_name}' requires executing code from the model author. "
                f"This is needed because the model uses a custom architecture. "
                f"Only proceed if you trust the model source."
            ).classes("text-sm text-gray-300")

            with ui.row().classes("w-full justify-end gap-2 mt-4"):
                ui.button("Cancel", on_click=dlg.close).props("flat color=white")
                ui.button(
                    "Trust & Download",
                    on_click=lambda: self._do_trust_download(dlg, model_name),
                ).props("color=amber")
        dlg.open()

    async def _do_trust_download(self, dlg, model_name: str):
        dlg.close()
        ui.notify(f"Downloading {model_name} (trusted)…", type="info")

        def _do_download():
            client = _make_client()
            try:
                client.trust_model(model_name)
                return client.download_model(model_name, trust_remote_code=True)
            finally:
                client.close()

        try:
            await run.io_bound(_do_download)
            ui.notify(f"✅ {model_name} downloaded!", type="positive")
            await self._refresh()
        except Exception as exc:
            ui.notify(f"Error: {exc}", type="negative")

    def _confirm_delete(self, model_name: str):
        with ui.dialog() as dlg, ui.card().classes("bg-[#1d232e] max-w-[90vw]"):
            ui.label("Delete Model?").classes("text-lg font-bold text-white")
            ui.label(
                f"Delete locally downloaded '{model_name}'? You can re-download it later."
            ).classes("text-sm text-gray-400")

            with ui.row().classes("w-full justify-end gap-2 mt-4"):
                ui.button("Cancel", on_click=dlg.close).props("flat color=white")
                ui.button("Delete", on_click=lambda: self._do_delete(dlg, model_name)).props(
                    "color=red"
                )
        dlg.open()

    async def _do_delete(self, dlg, model_name: str):
        dlg.close()
        try:
            await run.io_bound(lambda: _safe_call("delete_model", model_name))
            ui.notify(f"Deleted {model_name}", type="positive")
            await self._refresh()
        except Exception as exc:
            ui.notify(f"Error: {exc}", type="negative")


class ConfigPage:
    """Settings editor page."""

    def __init__(self):
        self._inputs: dict[str, Any] = {}
        self._form_container = None
        self._status_label = None
        self._deferred_timer = None

    def destroy(self):
        """Cancel deferred load timer when leaving the page."""
        if self._deferred_timer is not None:
            self._deferred_timer.cancel()
            self._deferred_timer = None

    def build(self):
        with ui.column().classes("w-full gap-4"):
            ui.label("Configuration").classes("text-2xl font-bold text-white")
            ui.label("Adjust embedding, search, indexing, and server settings").classes(
                "text-sm text-gray-400 -mt-2"
            )

            self._form_container = ui.column().classes("w-full gap-2")

            # Show skeleton form, defer settings fetch
            with self._form_container:
                _skeleton_config_form()
            self._deferred_timer = ui.timer(0.1, self._build_form, once=True)

            with ui.row().classes("gap-2 mt-2"):
                ui.button("Save Configuration", icon="save", on_click=self._save).props(
                    "color=primary"
                )
                ui.button(
                    "Reset to Defaults", icon="restart_alt", on_click=self._confirm_reset
                ).props("outline color=orange")

            self._status_label = ui.label("").classes("text-sm text-gray-400")

    async def _build_form(self):
        # Show skeleton while loading settings
        try:
            self._form_container.clear()
            with self._form_container:
                _skeleton_config_form()
        except RuntimeError:
            return  # UI elements deleted (client navigated away)
        try:
            settings = await run.io_bound(lambda: _safe_call("get_settings")) or {}
        except Exception as exc:
            try:
                self._form_container.clear()
                with self._form_container:
                    ui.label(f"Error loading settings: {exc}").classes("text-red-400")
            except RuntimeError:
                pass  # UI elements deleted
            return

        self._inputs.clear()
        try:
            self._form_container.clear()
        except RuntimeError:
            return  # UI elements deleted (client navigated away)
        with self._form_container:
            # --- Embedding & Chunking ---
            ui.label("Embedding & Chunking").classes("config-section-title")

            from rag_kb.models import get_embedding_model_names

            self._inputs["embedding_model"] = (
                ui.select(
                    options=get_embedding_model_names(),
                    value=settings.get("embedding_model", "paraphrase-multilingual-MiniLM-L12-v2"),
                    label="Embedding Model",
                )
                .props("outlined dark dense")
                .classes("w-full max-w-md")
            )

            with ui.row().classes("gap-4"):
                self._inputs["chunk_size"] = (
                    ui.number(
                        "Chunk Size",
                        value=settings.get("chunk_size", 1024),
                        min=64,
                        max=8192,
                        step=64,
                    )
                    .props("outlined dark dense")
                    .classes("w-48")
                )

                self._inputs["chunk_overlap"] = (
                    ui.number(
                        "Chunk Overlap",
                        value=settings.get("chunk_overlap", 128),
                        min=0,
                        max=2048,
                        step=16,
                    )
                    .props("outlined dark dense")
                    .classes("w-48")
                )

            # --- Search Quality ---
            ui.label("Search Quality").classes("config-section-title")

            with ui.row().classes("gap-8 items-start flex-wrap"):
                with ui.column().classes("gap-2"):
                    self._inputs["reranking_enabled"] = ui.switch(
                        "Cross-encoder reranking",
                        value=settings.get("reranking_enabled", True),
                    )
                    from rag_kb.models import get_reranker_model_names

                    self._inputs["reranker_model"] = (
                        ui.select(
                            options=get_reranker_model_names(),
                            value=settings.get(
                                "reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"
                            ),
                            label="Reranker Model",
                        )
                        .props("outlined dark dense")
                        .classes("w-80")
                    )

                with ui.column().classes("gap-2"):
                    self._inputs["hybrid_search_enabled"] = ui.switch(
                        "Hybrid search (vector + BM25)",
                        value=settings.get("hybrid_search_enabled", True),
                    )
                    self._inputs["hybrid_search_alpha"] = (
                        ui.slider(
                            min=0,
                            max=1,
                            step=0.05,
                            value=settings.get("hybrid_search_alpha", 0.7),
                        )
                        .classes("w-64")
                        .props("label-always")
                    )
                    ui.label("0 = All BM25 ←→ 1 = All Vector").classes("text-xs text-gray-500")

            with ui.row().classes("gap-4 items-start"):
                self._inputs["min_score_threshold"] = (
                    ui.number(
                        "Min Score Threshold",
                        value=settings.get("min_score_threshold", 0.15),
                        min=0,
                        max=1,
                        step=0.05,
                        format="%.2f",
                    )
                    .props("outlined dark dense")
                    .classes("w-48")
                )

                with ui.column().classes("gap-2"):
                    self._inputs["mmr_enabled"] = ui.switch(
                        "MMR diversity",
                        value=settings.get("mmr_enabled", True),
                    )
                    self._inputs["mmr_lambda"] = (
                        ui.slider(
                            min=0,
                            max=1,
                            step=0.05,
                            value=settings.get("mmr_lambda", 0.7),
                        )
                        .classes("w-64")
                        .props("label-always")
                    )
                    ui.label("0 = Max Diversity ←→ 1 = Max Relevance").classes(
                        "text-xs text-gray-500"
                    )

            # --- Indexing Performance ---
            ui.label("Indexing Performance").classes("config-section-title")

            with ui.row().classes("gap-4"):
                self._inputs["indexing_workers"] = (
                    ui.number(
                        "Parallel Workers",
                        value=settings.get("indexing_workers", 4),
                        min=1,
                        max=32,
                        step=1,
                    )
                    .props("outlined dark dense")
                    .classes("w-48")
                )

                self._inputs["embedding_batch_size"] = (
                    ui.number(
                        "Embedding Batch Size",
                        value=settings.get("embedding_batch_size", 256),
                        min=1,
                        max=4096,
                        step=32,
                    )
                    .props("outlined dark dense")
                    .classes("w-48")
                )

            # --- ChromaDB HNSW ---
            ui.label("ChromaDB HNSW Tuning").classes("config-section-title")

            with ui.row().classes("gap-4"):
                self._inputs["hnsw_ef_construction"] = (
                    ui.number(
                        "EF Construction",
                        value=settings.get("hnsw_ef_construction", 200),
                        min=16,
                        max=1024,
                        step=8,
                    )
                    .props("outlined dark dense")
                    .classes("w-48")
                )

                self._inputs["hnsw_m"] = (
                    ui.number(
                        "M Connections",
                        value=settings.get("hnsw_m", 32),
                        min=4,
                        max=128,
                        step=4,
                    )
                    .props("outlined dark dense")
                    .classes("w-48")
                )

            # --- API Keys ---
            ui.label("API Keys (for API-based models)").classes("config-section-title")
            ui.label(
                "API keys are stored locally in config.yaml. "
                "You can also set OPENAI_API_KEY / VOYAGE_API_KEY environment variables."
            ).classes("text-xs text-gray-500")

            with ui.row().classes("gap-4"):
                self._inputs["openai_api_key"] = (
                    ui.input(
                        "OpenAI API Key",
                        value=settings.get("openai_api_key", ""),
                        password=True,
                        password_toggle_button=True,
                    )
                    .props("outlined dark dense")
                    .classes("w-80")
                )

                self._inputs["voyage_api_key"] = (
                    ui.input(
                        "Voyage AI API Key",
                        value=settings.get("voyage_api_key", ""),
                        password=True,
                        password_toggle_button=True,
                    )
                    .props("outlined dark dense")
                    .classes("w-80")
                )

            # --- MCP Server ---
            ui.label("MCP Server").classes("config-section-title")

            with ui.row().classes("gap-4"):
                self._inputs["host"] = (
                    ui.input(
                        "Host",
                        value=settings.get("host", "127.0.0.1"),
                    )
                    .props("outlined dark dense")
                    .classes("w-48")
                )

                self._inputs["port"] = (
                    ui.number(
                        "Port",
                        value=settings.get("port", 8080),
                        min=1,
                        max=65535,
                        step=1,
                    )
                    .props("outlined dark dense")
                    .classes("w-48")
                )

    def _gather_settings(self) -> dict:
        """Collect current form values into a settings dict."""
        result = {}
        for key, widget in self._inputs.items():
            if hasattr(widget, "value"):
                val = widget.value
                # Convert number-like values
                if (
                    isinstance(val, float)
                    and val == int(val)
                    and key not in ("hybrid_search_alpha", "mmr_lambda", "min_score_threshold")
                ):
                    val = int(val)
                result[key] = val
        return result

    async def _save(self):
        try:
            settings = self._gather_settings()
            await run.io_bound(lambda: _safe_call("save_config", settings))
            self._status_label.set_text("✅ Settings saved")
            ui.notify("Configuration saved", type="positive")
        except Exception as exc:
            self._status_label.set_text(f"❌ Error: {exc}")
            ui.notify(f"Error saving: {exc}", type="negative")

    def _confirm_reset(self):
        with ui.dialog() as dlg, ui.card().classes("bg-[#1d232e] max-w-[90vw]"):
            ui.label("Reset to defaults?").classes("text-lg font-bold text-white")
            ui.label("All settings will be reset to their default values.").classes(
                "text-sm text-gray-400"
            )
            with ui.row().classes("w-full justify-end gap-2 mt-4"):
                ui.button("Cancel", on_click=dlg.close).props("flat color=white")
                ui.button("Reset", on_click=lambda: self._do_reset(dlg)).props("color=orange")
        dlg.open()

    async def _do_reset(self, dlg):
        try:
            # Reset by saving a default settings dict
            from rag_kb.config import AppSettings

            defaults = AppSettings().model_dump()
            defaults.pop("supported_extensions", None)
            await run.io_bound(lambda: _safe_call("save_config", defaults))
            dlg.close()
            await self._build_form()
            ui.notify("Settings reset to defaults", type="positive")
        except Exception as exc:
            ui.notify(f"Error: {exc}", type="negative")


# ---------------------------------------------------------------------------
# Page: Monitoring
# ---------------------------------------------------------------------------


class MonitoringPage:
    """Deep metrics and monitoring dashboard."""

    def __init__(self):
        self._system_container = None
        self._indexing_container = None
        self._embedding_container = None
        self._vector_container = None
        self._search_container = None
        self._refresh_timer = None
        self._deferred_timer = None

    def destroy(self):
        """Cancel auto-refresh timer when leaving the page."""
        if self._deferred_timer is not None:
            self._deferred_timer.cancel()
            self._deferred_timer = None
        if self._refresh_timer is not None:
            self._refresh_timer.cancel()
            self._refresh_timer = None

    def build(self):
        with ui.column().classes("w-full gap-4"):
            ui.label("Monitoring").classes("text-2xl font-bold text-white")
            ui.label("Deep metrics for indexing, embeddings, search & system health").classes(
                "text-sm text-gray-400 -mt-2"
            )

            # System Health section
            with (
                ui.expansion("System Health", icon="monitor_heart")
                .classes("w-full bg-[#1d232e] rounded-xl border border-[#313949]")
                .props("default-opened dark dense")
            ):
                self._system_container = ui.column().classes("w-full gap-2 p-2")
                with self._system_container:
                    _skeleton_metric_cards(5)

            # Indexing Pipeline section
            with (
                ui.expansion("Indexing Pipeline", icon="construction")
                .classes("w-full bg-[#1d232e] rounded-xl border border-[#313949]")
                .props("default-opened dark dense")
            ):
                self._indexing_container = ui.column().classes("w-full gap-2 p-2")
                with self._indexing_container:
                    _skeleton_metric_cards(6)

            # Embedding Performance section
            with (
                ui.expansion("Embedding Performance", icon="memory")
                .classes("w-full bg-[#1d232e] rounded-xl border border-[#313949]")
                .props("dark dense")
            ):
                self._embedding_container = ui.column().classes("w-full gap-2 p-2")

            # Vector Store / ChromaDB section
            with (
                ui.expansion("Vector Store / ChromaDB", icon="storage")
                .classes("w-full bg-[#1d232e] rounded-xl border border-[#313949]")
                .props("dark dense")
            ):
                self._vector_container = ui.column().classes("w-full gap-2 p-2")

            # Search Performance section
            with (
                ui.expansion("Search Performance", icon="search")
                .classes("w-full bg-[#1d232e] rounded-xl border border-[#313949]")
                .props("dark dense")
            ):
                self._search_container = ui.column().classes("w-full gap-2 p-2")

            # Initial async refresh (runs after build returns)
            self._deferred_timer = ui.timer(0.1, self.refresh, once=True)

            # Periodic auto-refresh (15s to reduce load — 5 RPCs per tick)
            self._refresh_timer = ui.timer(15, self.refresh)

    async def refresh(self):
        try:
            data = await run.io_bound(
                lambda: {
                    "dashboard": _safe_call("get_metrics_dashboard"),
                    "vs_details": _safe_call("get_vector_store_details"),
                    "indexing_history": _safe_call("get_indexing_history", limit=20),
                    "search_stats": _safe_call("get_search_stats", limit=50),
                    "embedding_stats": _safe_call("get_embedding_stats", limit=50),
                }
            )
        except RuntimeError:
            return  # UI elements deleted (page navigated away)
        except Exception as exc:
            logger.warning("Monitoring refresh error: %s", exc)
            return

        try:
            self._render_system(data["dashboard"])
            self._render_indexing(data["dashboard"], data["indexing_history"])
            self._render_embedding(data["dashboard"], data["embedding_stats"])
            self._render_vector_store(data["dashboard"], data["vs_details"])
            self._render_search(data["dashboard"], data["search_stats"])
        except RuntimeError:
            pass  # UI elements deleted during render

    def _render_system(self, dashboard: dict):
        self._system_container.clear()
        with self._system_container:
            sys_snap = dashboard.get("system") or {}

            with ui.row().classes("w-full gap-4 flex-wrap"):
                self._metric_card("CPU", f"{sys_snap.get('cpu_percent', 0):.1f}%", "processor")
                self._metric_card(
                    "Memory",
                    f"{sys_snap.get('memory_used_mb', 0):.0f} / {sys_snap.get('memory_total_mb', 0):.0f} MB",
                    "memory",
                )
                self._metric_card("Memory %", f"{sys_snap.get('memory_percent', 0):.1f}%", "memory")
                self._metric_card(
                    "Process RAM",
                    f"{sys_snap.get('process_memory_mb', 0):.1f} MB",
                    "developer_board",
                )
                self._metric_card(
                    "Disk Free",
                    f"{sys_snap.get('disk_free_mb', 0):.0f} MB",
                    "hard_drive",
                )

            with ui.row().classes("w-full gap-4 flex-wrap"):
                uptime = sys_snap.get("daemon_uptime_seconds", 0)
                h, m = divmod(int(uptime), 3600)
                mins = m // 60
                self._metric_card("Daemon Uptime", f"{h}h {mins}m", "schedule")
                self._metric_card(
                    "Active Conns", str(sys_snap.get("active_connections", 0)), "cable"
                )
                self._metric_card(
                    "Total RPC Calls", str(sys_snap.get("total_rpc_calls", 0)), "call_made"
                )

    def _render_indexing(self, dashboard: dict, history: list[dict]):
        self._indexing_container.clear()
        with self._indexing_container:
            agg = dashboard.get("indexing_aggregates") or {}
            last = dashboard.get("last_indexing_run") or {}

            with ui.row().classes("w-full gap-4 flex-wrap"):
                self._metric_card("Total Runs", str(agg.get("total_runs", 0)), "repeat")
                self._metric_card(
                    "Avg Duration",
                    f"{agg.get('avg_duration', 0) or 0:.1f}s",
                    "timer",
                )
                self._metric_card(
                    "Avg Throughput",
                    f"{agg.get('avg_throughput', 0) or 0:.1f} chunks/s",
                    "speed",
                )
                self._metric_card(
                    "Total Errors",
                    str(int(agg.get("total_errors", 0) or 0)),
                    "error",
                )

            if last:
                ui.label("Last Run Phase Breakdown").classes("text-sm font-bold text-gray-300 mt-2")
                phases = [
                    ("Scan", last.get("scan_seconds", 0)),
                    ("Parse", last.get("parse_seconds", 0)),
                    ("Chunk", last.get("chunk_seconds", 0)),
                    ("Embed", last.get("embed_seconds", 0)),
                    ("Upsert", last.get("upsert_seconds", 0)),
                    ("Manifest", last.get("manifest_seconds", 0)),
                ]
                sum(v for _, v in phases) or 1
                # Waterfall bar chart using ECharts
                ui.echart(
                    {
                        "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
                        "grid": {"left": "80px", "right": "40px", "top": "10px", "bottom": "30px"},
                        "xAxis": {"type": "value", "name": "seconds"},
                        "yAxis": {
                            "type": "category",
                            "data": [p[0] for p in phases],
                            "inverse": True,
                        },
                        "series": [
                            {
                                "type": "bar",
                                "data": [round(p[1], 3) for p in phases],
                                "itemStyle": {"color": "#67d3ff"},
                                "label": {
                                    "show": True,
                                    "position": "right",
                                    "formatter": "{c}s",
                                    "color": "#9daec1",
                                },
                            }
                        ],
                    }
                ).classes("w-full h-48")

                with ui.row().classes("gap-6 mt-1"):
                    ui.label(
                        f"Status: {last.get('status', '?')}  |  "
                        f"Files: {last.get('processed_files', 0)}  |  "
                        f"Chunks: {last.get('total_chunks', 0)}  |  "
                        f"Duration: {last.get('duration_seconds', 0):.1f}s  |  "
                        f"Throughput: {last.get('chunks_per_second', 0):.1f} chunks/s"
                    ).classes("text-xs text-gray-400")

            if history:
                ui.label("Indexing Run History").classes("text-sm font-bold text-gray-300 mt-4")
                rows = []
                for h in history[:10]:
                    import datetime

                    ts = h.get("started_at", 0)
                    dt = (
                        datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
                        if ts
                        else "?"
                    )
                    rows.append(
                        {
                            "date": dt,
                            "rag": h.get("rag_name", ""),
                            "status": h.get("status", ""),
                            "files": h.get("processed_files", 0),
                            "chunks": h.get("total_chunks", 0),
                            "duration": f"{h.get('duration_seconds', 0):.1f}s",
                            "errors": h.get("error_count", 0),
                        }
                    )
                ui.table(
                    columns=[
                        {"name": "date", "label": "Date", "field": "date", "align": "left"},
                        {"name": "rag", "label": "RAG", "field": "rag", "align": "left"},
                        {"name": "status", "label": "Status", "field": "status"},
                        {"name": "files", "label": "Files", "field": "files", "align": "right"},
                        {"name": "chunks", "label": "Chunks", "field": "chunks", "align": "right"},
                        {
                            "name": "duration",
                            "label": "Duration",
                            "field": "duration",
                            "align": "right",
                        },
                        {"name": "errors", "label": "Errors", "field": "errors", "align": "right"},
                    ],
                    rows=rows,
                    row_key="date",
                ).classes("w-full").props("dark flat bordered dense")

    def _render_embedding(self, dashboard: dict, stats: list[dict]):
        self._embedding_container.clear()
        with self._embedding_container:
            agg = dashboard.get("embedding_aggregates") or {}

            with ui.row().classes("w-full gap-4 flex-wrap"):
                self._metric_card(
                    "Total Batches", str(agg.get("total_batches", 0)), "batch_prediction"
                )
                self._metric_card(
                    "Avg Batch Time",
                    f"{agg.get('avg_batch_ms', 0) or 0:.1f} ms",
                    "timer",
                )
                self._metric_card(
                    "Avg Throughput",
                    f"{agg.get('avg_throughput', 0) or 0:.1f} chunks/s",
                    "bolt",
                )
                self._metric_card(
                    "Total Embedded",
                    str(int(agg.get("total_texts_embedded", 0) or 0)),
                    "translate",
                )

            if stats:
                # Embedding throughput timeline chart
                times = []
                throughputs = []
                for s in reversed(stats[:30]):
                    import datetime

                    ts = s.get("timestamp", 0)
                    times.append(
                        datetime.datetime.fromtimestamp(ts).strftime("%H:%M:%S") if ts else ""
                    )
                    throughputs.append(round(s.get("chunks_per_second", 0), 1))

                if times:
                    ui.echart(
                        {
                            "tooltip": {"trigger": "axis"},
                            "grid": {
                                "left": "60px",
                                "right": "20px",
                                "top": "20px",
                                "bottom": "30px",
                            },
                            "xAxis": {"type": "category", "data": times},
                            "yAxis": {"type": "value", "name": "chunks/s"},
                            "series": [
                                {
                                    "type": "line",
                                    "data": throughputs,
                                    "smooth": True,
                                    "areaStyle": {"opacity": 0.15},
                                    "itemStyle": {"color": "#67d3ff"},
                                }
                            ],
                        }
                    ).classes("w-full h-48")

    def _render_vector_store(self, dashboard: dict, vs_details: dict):
        self._vector_container.clear()
        with self._vector_container:
            if not vs_details:
                ui.label("No vector store data available").classes("text-gray-400")
                return

            with ui.row().classes("w-full gap-4 flex-wrap"):
                self._metric_card(
                    "Total Chunks", str(vs_details.get("total_chunks", 0)), "data_object"
                )
                self._metric_card(
                    "Total Files", str(vs_details.get("total_files", 0)), "description"
                )
                self._metric_card(
                    "DB Size",
                    f"{vs_details.get('db_size_mb', 0):.1f} MB",
                    "storage",
                )
                self._metric_card(
                    "Avg Chunks/File",
                    f"{vs_details.get('avg_chunks_per_file', 0):.1f}",
                    "stacked_bar_chart",
                )

            hnsw = vs_details.get("hnsw_config", {})
            if hnsw:
                ui.label("HNSW Index Configuration").classes("text-sm font-bold text-gray-300 mt-2")
                with ui.row().classes("gap-6"):
                    ui.label(f"Space: {hnsw.get('space', '?')}").classes("text-sm text-gray-400")
                    ui.label(f"construction_ef: {hnsw.get('construction_ef', '?')}").classes(
                        "text-sm text-gray-400"
                    )
                    ui.label(f"M: {hnsw.get('M', '?')}").classes("text-sm text-gray-400")

            ui.label(f"DB Path: {vs_details.get('db_path', '?')}").classes(
                "text-xs text-gray-500 mt-1 font-mono"
            )

    def _render_search(self, dashboard: dict, stats: list[dict]):
        self._search_container.clear()
        with self._search_container:
            agg = dashboard.get("search_aggregates") or {}

            with ui.row().classes("w-full gap-4 flex-wrap"):
                self._metric_card("Total Queries", str(agg.get("total_queries", 0)), "search")
                self._metric_card(
                    "Avg Latency",
                    f"{agg.get('avg_latency_ms', 0) or 0:.1f} ms",
                    "speed",
                )
                self._metric_card(
                    "Avg Results",
                    f"{agg.get('avg_results', 0) or 0:.1f}",
                    "format_list_numbered",
                )
                self._metric_card(
                    "Avg Top Score",
                    f"{agg.get('avg_top_score', 0) or 0:.3f}",
                    "grade",
                )

            if stats:
                # Latency breakdown chart (last N queries)
                recent = list(reversed(stats[:20]))
                labels = [f"Q{i + 1}" for i in range(len(recent))]
                vec_ms = [round(s.get("vector_search_ms", 0), 1) for s in recent]
                bm25_ms = [round(s.get("bm25_ms", 0), 1) for s in recent]
                rerank_ms = [round(s.get("rerank_ms", 0), 1) for s in recent]
                mmr_ms = [round(s.get("mmr_ms", 0), 1) for s in recent]

                ui.echart(
                    {
                        "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
                        "legend": {
                            "data": ["Vector", "BM25", "Rerank", "MMR"],
                            "textStyle": {"color": "#9daec1"},
                        },
                        "grid": {"left": "60px", "right": "20px", "top": "40px", "bottom": "30px"},
                        "xAxis": {"type": "category", "data": labels},
                        "yAxis": {"type": "value", "name": "ms"},
                        "series": [
                            {
                                "name": "Vector",
                                "type": "bar",
                                "stack": "total",
                                "data": vec_ms,
                                "itemStyle": {"color": "#2f79c6"},
                            },
                            {
                                "name": "BM25",
                                "type": "bar",
                                "stack": "total",
                                "data": bm25_ms,
                                "itemStyle": {"color": "#27ae60"},
                            },
                            {
                                "name": "Rerank",
                                "type": "bar",
                                "stack": "total",
                                "data": rerank_ms,
                                "itemStyle": {"color": "#f39c12"},
                            },
                            {
                                "name": "MMR",
                                "type": "bar",
                                "stack": "total",
                                "data": mmr_ms,
                                "itemStyle": {"color": "#c0392b"},
                            },
                        ],
                    }
                ).classes("w-full h-48")

    @staticmethod
    def _metric_card(label: str, value: str, icon: str = "analytics"):
        with (
            ui.card().classes(
                "bg-[#22262e] border border-[#313949] rounded-lg p-3 min-w-[140px] flex-1"
            ),
            ui.row().classes("items-center gap-2"),
        ):
            ui.icon(icon).classes("text-[#67d3ff] text-lg")
            with ui.column().classes("gap-0"):
                ui.label(value).classes("text-lg font-bold text-[#67d3ff]")
                ui.label(label).classes("text-xs text-gray-400")


# ---------------------------------------------------------------------------
# Main layout
# ---------------------------------------------------------------------------


def _build_app():
    """Build the main NiceGUI application with sidebar navigation."""

    # Inject custom CSS
    ui.add_head_html(f"<style>{_CUSTOM_CSS}</style>")

    # Dark mode
    ui.dark_mode(True)

    # State
    pages: dict[str, Any] = {}
    page_container = None

    page_classes = {
        "dashboard": DashboardPage,
        "search": SearchPage,
        "manage": ManagePage,
        "indexing": IndexingPage,
        "models": ModelsPage,
        "monitoring": MonitoringPage,
        "config": ConfigPage,
    }

    nav_items = [
        ("dashboard", "Dashboard", "dashboard"),
        ("search", "Search", "search"),
        ("library_books", "Manage", "manage"),
        ("description", "Indexing", "indexing"),
        ("smart_toy", "Models", "models"),
        ("monitoring", "Monitoring", "monitoring"),
        ("settings", "Config", "config"),
    ]

    nav_buttons: dict[str, Any] = {}

    def switch_page(page_id: str):
        if page_id not in page_classes:
            return

        # Destroy all existing pages' timers to prevent accumulation
        for page in pages.values():
            page.destroy()

        # Update nav button styles
        for nid, btn in nav_buttons.items():
            if nid == page_id:
                btn.classes(replace="nav-btn nav-active")
            else:
                btn.classes(replace="nav-btn nav-inactive")

        # Build (or rebuild) page content
        page_container.clear()
        with page_container:
            if page_id not in pages:
                pages[page_id] = page_classes[page_id]()
            pages[page_id].build()

    # ---- Header ----
    with (
        ui.header()
        .classes("bg-[#11151c] items-center px-4 shadow-md")
        .props("elevated")
        .style("height: 48px")
    ):
        ui.button(icon="menu", on_click=lambda: drawer.toggle()).props(  # noqa: PLW0108
            "flat round color=white dense"
        )
        ui.label("RAG Knowledge Base").classes("text-white text-base font-bold ml-2")
        ui.space()

        # Persistent indexing indicator — visible from any page
        indexing_badge = ui.button(
            "⏳ Indexing…",
            on_click=lambda: switch_page("indexing"),
        ).props("flat dense no-caps size=sm color=amber").classes("text-xs")
        indexing_badge.set_visibility(False)

        daemon_label = ui.label("● Daemon").classes("text-xs text-green-400")

        async def check_daemon():
            try:
                info = await run.io_bound(lambda: _safe_call("ping"))
                uptime = info.get("uptime", 0)
                daemon_label.set_text(f"● Daemon (up {int(uptime)}s)")
                daemon_label.classes(replace="text-xs text-green-400")
            except Exception:
                daemon_label.set_text("● Daemon offline")
                daemon_label.classes(replace="text-xs text-red-400")

            # Update global indexing badge
            try:
                status = await run.io_bound(lambda: _safe_call("get_index_status"))
                live = (status or {}).get("indexing")
                if live and live.get("is_indexing"):
                    pct = int(min(live.get("progress", 0.0), 1.0) * 100)
                    indexing_badge.set_text(f"⏳ Indexing… {pct}%")
                    indexing_badge.set_visibility(True)
                else:
                    indexing_badge.set_visibility(False)
            except Exception:
                pass

        ui.timer(30, check_daemon)

    # ---- Left Drawer (sidebar navigation) ----
    drawer = ui.left_drawer(value=True, fixed=True, bordered=True)
    with (
        drawer
        .classes("bg-[#11151c] !p-0")
        .props("width=220")
    ):
        ui.label("Offline RAG Manager").classes("text-gray-400 text-xs px-4 pt-4 pb-3")

        for mat_icon, label, page_id in nav_items:
            btn = (
                ui.button(
                    label,
                    icon=mat_icon,
                    on_click=lambda _pid=page_id: switch_page(_pid),
                )
                .props("flat no-caps align=left")
                .classes("nav-btn nav-inactive")
            )
            nav_buttons[page_id] = btn

    # ---- Main content area ----
    page_container = ui.column().classes("w-full p-4 sm:p-6 gap-0 page-content")

    # Start on dashboard.  A deferred check switches to the indexing page
    # if the daemon reports an active indexing operation — avoids a
    # synchronous blocking RPC call during page build which can fail with
    # stale socket errors on Windows and stall the event loop.
    switch_page("dashboard")

    async def _maybe_switch_to_indexing():
        try:
            status = await run.io_bound(lambda: _safe_call("get_index_status"))
            live = (status or {}).get("indexing")
            if live and live.get("is_indexing"):
                switch_page("indexing")
        except Exception:
            pass

    ui.timer(0.3, _maybe_switch_to_indexing, once=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def launch_web_ui(
    host: str = "127.0.0.1",
    port: int = 8501,
    native: bool = False,
    reload: bool = False,
) -> None:
    """Launch the NiceGUI web UI.

    Parameters
    ----------
    host : str
        Bind address.
    port : int
        HTTP port for the web UI.
    native : bool
        If True, open in a native desktop window instead of the browser.
    reload : bool
        Enable auto-reload during development.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for lib in (
        "chromadb",
        "sentence_transformers",
        "httpx",
        "httpcore",
        "urllib3",
        "watchdog",
        "huggingface_hub",
        "transformers",
        "safetensors",
    ):
        logging.getLogger(lib).setLevel(logging.WARNING)

    # Ensure daemon is running before starting the UI server
    try:
        _get_client()
    except Exception as exc:
        logger.exception("Failed to start daemon: %s", exc)  # noqa: TRY401

    # Find a free port if the requested one is already in use
    port = _find_free_port(host, port)

    # Fix Windows asyncio bug: _ProactorBasePipeTransport._call_connection_lost()
    # calls socket.shutdown(SHUT_RDWR) without catching OSError.  When the
    # remote end (browser) has already reset the connection, this raises
    # ConnectionResetError [WinError 10054].  Worse, because the call is in a
    # ``finally`` block, the subsequent socket.close() and cleanup are skipped,
    # causing a resource leak.
    #
    # This is unfixed in CPython through 3.13 and main (3.14-dev).
    # See: https://github.com/python/cpython/issues/87419
    #
    # The monkey-patch below wraps only the shutdown() call in try/except,
    # which is the exact fix CPython should apply.  It can be removed once a
    # CPython release includes the fix.
    if sys.platform == "win32":
        _patch_proactor_connection_lost()

    # ------------------------------------------------------------------
    # Clean shutdown on Ctrl+C / SIGINT / SIGTERM
    # ------------------------------------------------------------------
    def _cleanup():
        """Close the shared DaemonClient (if open) so the socket is released."""
        global _client
        if _client is not None:
            try:
                _client.close()
            except Exception:
                pass
            _client = None

    @app.on_shutdown
    def _on_shutdown():
        _cleanup()

    # On Windows, Ctrl+C inside uvicorn can produce ugly tracebacks.
    # Install a handler that triggers a clean exit instead.
    _original_sigint = signal.getsignal(signal.SIGINT)

    def _sigint_handler(signum, frame):
        logger.info("Ctrl+C received — shutting down…")
        _cleanup()
        # Restore original handler and re-raise so uvicorn can exit.
        signal.signal(signal.SIGINT, _original_sigint)
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _sigint_handler)

    # Register the main page
    @ui.page("/")
    def index():
        _build_app()

    try:
        ui.run(
            host=host,
            port=port,
            title="RAG Knowledge Base",
            dark=True,
            native=native,
            reload=reload,
            show=not native,  # open browser only when not in native mode
            favicon="📚",
        )
    except KeyboardInterrupt:
        pass
    finally:
        _cleanup()


if __name__ == "__main__":
    launch_web_ui()
