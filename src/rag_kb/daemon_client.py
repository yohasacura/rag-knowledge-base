"""Synchronous blocking client for the RAG daemon.

Provides typed Python methods for every JSON-RPC endpoint exposed by the
daemon.  Used by CLI, MCP server and UI as a drop-in replacement for
direct ``RagKnowledgeBaseAPI`` access.

Auto-start
----------
:meth:`DaemonClient.ensure_daemon` checks the PID file and TCP port.  If
the daemon is not reachable it spawns one in the background and waits for
it to become available (with exponential back-off, ~3 s max).
"""

from __future__ import annotations

import logging
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

from rag_kb.config import DATA_DIR
from rag_kb.rpc_protocol import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    RpcError,
    frame_message,
    make_request,
    read_frame_sync,
)

logger = logging.getLogger(__name__)

_PID_FILE = DATA_DIR / "daemon.pid"


class DaemonClient:
    """Synchronous JSON-RPC client for the RAG daemon.

    Usage::

        client = DaemonClient()
        client.ensure_daemon()       # auto-start if needed
        client.connect()
        rags = client.list_rags()
        client.close()

    Or as a context manager::

        with DaemonClient() as client:
            rags = client.list_rags()
    """

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
    ) -> None:
        self._host = host
        self._port = port
        self._sock: socket.socket | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "DaemonClient":
        self.ensure_daemon()
        self.connect()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Open a TCP connection to the daemon."""
        if self._sock is not None:
            return
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(120)  # generous timeout for long-running ops
        sock.connect((self._host, self._port))
        self._sock = sock

    def close(self) -> None:
        """Close the TCP connection."""
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None

    @property
    def connected(self) -> bool:
        return self._sock is not None

    def _ensure_connected(self) -> None:
        """Connect if not already connected."""
        if self._sock is None:
            self.connect()

    # ------------------------------------------------------------------
    # Daemon lifecycle
    # ------------------------------------------------------------------

    def ensure_daemon(self) -> None:
        """Start the daemon if it is not already running.

        1. Check PID file → is process alive?
        2. TCP probe → can we connect?
        3. If not, spawn daemon subprocess and poll until ready.
        """
        if self._probe():
            return

        # Check PID file and whether that process is alive
        if _is_daemon_alive():
            # Process exists but port not open yet — wait a bit
            if self._wait_for_daemon(timeout=5):
                return

        # Spawn a new daemon
        self._spawn_daemon()
        if not self._wait_for_daemon(timeout=10):
            raise RuntimeError(
                f"Daemon failed to start on {self._host}:{self._port} — "
                f"check logs or start manually with: rag-kb-daemon"
            )

    def _probe(self) -> bool:
        """Return True if the daemon is reachable via TCP."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            sock.connect((self._host, self._port))
            sock.close()
            return True
        except OSError:
            return False

    def _wait_for_daemon(self, timeout: float = 10) -> bool:
        """Poll TCP port with exponential back-off until reachable."""
        deadline = time.monotonic() + timeout
        delay = 0.1
        while time.monotonic() < deadline:
            if self._probe():
                return True
            time.sleep(delay)
            delay = min(delay * 1.5, 1.0)
        return False

    def _spawn_daemon(self) -> None:
        """Launch the daemon in a detached subprocess."""
        logger.info("Starting daemon on %s:%d …", self._host, self._port)

        # Ensure log directory exists so the daemon can write its log file
        # even before its own logging setup runs.
        from rag_kb.daemon import LOG_FILE
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable, "-m", "rag_kb.daemon",
            "--host", self._host,
            "--port", str(self._port),
        ]

        # Redirect stdout/stderr to the daemon log file so that early
        # startup messages (or crash tracebacks) are captured.
        log_fh = open(LOG_FILE, "a", encoding="utf-8")

        kwargs: dict[str, Any] = {
            "stdout": log_fh,
            "stderr": log_fh,
            "stdin": subprocess.DEVNULL,
        }

        if sys.platform == "win32":
            # Detach on Windows
            CREATE_NEW_PROCESS_GROUP = 0x00000200
            DETACHED_PROCESS = 0x00000008
            kwargs["creationflags"] = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True

        subprocess.Popen(cmd, **kwargs)

    # ------------------------------------------------------------------
    # Low-level RPC
    # ------------------------------------------------------------------

    def _call(
        self,
        method: str,
        params: dict | None = None,
        on_progress: Callable[[dict], None] | None = None,
        timeout: float | None = None,
    ) -> Any:
        """Send a JSON-RPC request and return the result.

        For methods that stream progress notifications (``index.run``,
        ``config.models.download``), pass *on_progress* to receive them.
        """
        with self._lock:
            self._ensure_connected()
            assert self._sock is not None

            req = make_request(method, params)
            self._sock.sendall(frame_message(req))

            saved_timeout = self._sock.gettimeout()
            if timeout is not None:
                self._sock.settimeout(timeout)

            try:
                while True:
                    msg = read_frame_sync(self._sock)

                    # Progress notification (no "id")
                    if "method" in msg and msg["method"] == "progress":
                        if on_progress is not None:
                            on_progress(msg.get("params", {}))
                        continue

                    # Error response
                    if "error" in msg:
                        err = msg["error"]
                        raise RpcError(err["code"], err["message"], err.get("data"))

                    # Success response
                    return msg.get("result")
            finally:
                # Restore original socket timeout
                if self._sock is not None:
                    try:
                        self._sock.settimeout(saved_timeout)
                    except OSError:
                        pass

    # ------------------------------------------------------------------
    # High-level typed methods
    # ------------------------------------------------------------------

    # --- rag.* ---

    def list_rags(self) -> list[dict]:
        """Return all registered RAGs."""
        return self._call("rag.list")

    def create_rag(
        self,
        name: str,
        folders: list[str] | None = None,
        description: str = "",
        embedding_model: str | None = None,
    ) -> dict:
        """Create a new RAG."""
        return self._call("rag.create", {
            "name": name,
            "folders": folders,
            "description": description,
            "embedding_model": embedding_model,
        })

    def switch_rag(self, name: str) -> dict:
        """Set *name* as the active RAG."""
        return self._call("rag.switch", {"name": name})

    def delete_rag(self, name: str, confirm: bool = True) -> dict:
        """Delete a RAG."""
        return self._call("rag.delete", {"name": name, "confirm": confirm})

    def detach_rag(self, name: str) -> dict:
        """Detach a RAG (make read-only)."""
        return self._call("rag.detach", {"name": name})

    def attach_rag(self, name: str) -> dict:
        """Re-attach a previously detached RAG."""
        return self._call("rag.attach", {"name": name})

    def get_rag(self, name: str) -> dict:
        """Get a specific RAG by name."""
        return self._call("rag.get", {"name": name})

    def update_rag(self, name: str, **kwargs) -> dict:
        """Update mutable fields of a RAG entry."""
        params = {"name": name, **kwargs}
        return self._call("rag.update", params)

    def get_active_name(self) -> str | None:
        """Return the name of the active RAG, or None."""
        rags = self.list_rags()
        for r in rags:
            if r.get("is_active"):
                return r["name"]
        return None

    # --- search.* ---

    def search(
        self,
        query: str,
        top_k: int = 5,
        rag_name: str | None = None,
        min_score: float | None = None,
    ) -> list[dict]:
        """Run a hybrid search."""
        params: dict[str, Any] = {"query": query, "top_k": top_k}
        if rag_name:
            params["rag_name"] = rag_name
        if min_score is not None:
            params["min_score"] = min_score
        return self._call("search.query", params)

    # --- index.* ---

    def index(
        self,
        rag_name: str | None = None,
        full: bool = False,
        workers: int | None = None,
        on_progress: Callable[[dict], None] | None = None,
    ) -> dict:
        """Run the indexing pipeline. Streams progress if *on_progress* given."""
        params: dict[str, Any] = {"full": full}
        if rag_name:
            params["rag_name"] = rag_name
        if workers is not None:
            params["workers"] = workers
        return self._call("index.run", params, on_progress=on_progress,
                          timeout=3600)  # 1 hour — indexing can be very long

    def get_index_status(self, rag_name: str | None = None) -> dict:
        """Return structured status for a RAG."""
        params: dict[str, Any] = {}
        if rag_name:
            params["rag_name"] = rag_name
        return self._call("index.status", params)

    def list_indexed_files(
        self,
        rag_name: str | None = None,
        *,
        offset: int = 0,
        limit: int = 0,
        filter: str = "",
    ) -> dict:
        """List files in the index (paginated).

        Returns a dict with keys: files, total, offset, limit, filter.
        """
        params: dict[str, Any] = {}
        if rag_name:
            params["rag_name"] = rag_name
        if offset:
            params["offset"] = offset
        if limit:
            params["limit"] = limit
        if filter:
            params["filter"] = filter
        return self._call("index.files", params)

    def reindex(
        self,
        rag_name: str | None = None,
        on_progress: Callable[[dict], None] | None = None,
    ) -> dict:
        """Full reindex."""
        params: dict[str, Any] = {}
        if rag_name:
            params["rag_name"] = rag_name
        return self._call("index.reindex", params, on_progress=on_progress,
                          timeout=3600)  # 1 hour — indexing can be very long

    def get_document_content(
        self,
        source_file: str,
        rag_name: str | None = None,
    ) -> list[dict]:
        """Retrieve all indexed chunks from a document."""
        params: dict[str, Any] = {"source_file": source_file}
        if rag_name:
            params["rag_name"] = rag_name
        return self._call("index.document", params)

    def scan_file_changes(self, rag_name: str | None = None) -> dict:
        """Scan source folders for new/modified/removed files."""
        params: dict[str, Any] = {}
        if rag_name:
            params["rag_name"] = rag_name
        return self._call("index.changes", params)

    def cancel_indexing(self) -> dict:
        """Request cancellation of the currently running indexing operation.

        Returns a dict with key ``cancelled`` (bool) indicating whether
        a running indexer was found and signalled.
        """
        return self._call("index.cancel", {})

    def verify_index_consistency(self, rag_name: str | None = None) -> dict:
        """Verify manifest ↔ vector-store consistency.

        Returns a dict with keys: ok, invalidated_files,
        orphan_store_files, orphan_manifest_files, incomplete_indexing.
        """
        params: dict[str, Any] = {}
        if rag_name:
            params["rag_name"] = rag_name
        return self._call("index.verify", params)

    # --- share.* ---

    def export_rag(self, name: str, output_path: str) -> dict:
        """Export a RAG to a .rag file."""
        return self._call("share.export", {"name": name, "output_path": output_path})

    def import_rag(self, path: str, name: str | None = None) -> dict:
        """Import a RAG from a .rag file."""
        params: dict[str, Any] = {"path": path}
        if name:
            params["name"] = name
        return self._call("share.import", params)

    def peek_rag_file(self, path: str) -> dict:
        """Read metadata from a .rag file without importing."""
        return self._call("share.peek", {"path": path})

    # --- config.* ---

    def get_settings(self) -> dict:
        """Return all current settings."""
        return self._call("config.get")

    def update_setting(self, key: str, value: Any) -> dict:
        """Update a single setting."""
        return self._call("config.set", {"key": key, "value": value})

    def save_config(self, settings: dict | None = None) -> dict:
        """Persist settings to disk."""
        params: dict[str, Any] = {}
        if settings:
            params["settings"] = settings
        return self._call("config.save", params)

    def reload_config(self) -> dict:
        """Reload settings from disk."""
        return self._call("config.reload")

    def download_models(
        self,
        output_dir: str | None = None,
        model_name: str | None = None,
    ) -> dict:
        """Download ML models for offline use."""
        params: dict[str, Any] = {}
        if output_dir:
            params["output_dir"] = output_dir
        if model_name:
            params["model_name"] = model_name
        return self._call("config.models.download", params, timeout=3600)

    # --- models.* ---

    def list_models(self, model_type: str | None = None) -> list[dict]:
        """Return all registered models with status."""
        params: dict[str, Any] = {}
        if model_type:
            params["model_type"] = model_type
        return self._call("models.list", params)

    def get_model_info(self, model_name: str) -> dict | None:
        """Return detailed info for a single model."""
        try:
            return self._call("models.info", {"model_name": model_name})
        except Exception:
            return None

    def download_model(
        self,
        model_name: str,
        trust_remote_code: bool | None = None,
    ) -> dict:
        """Download a single model."""
        params: dict[str, Any] = {"model_name": model_name}
        if trust_remote_code is not None:
            params["trust_remote_code"] = trust_remote_code
        return self._call("models.download", params, timeout=3600)

    def delete_model(self, model_name: str) -> dict:
        """Delete a downloaded model."""
        return self._call("models.delete", {"model_name": model_name})

    def trust_model(self, model_name: str) -> dict:
        """Add a model to the trusted list."""
        return self._call("models.trust", {"model_name": model_name})

    # --- watcher.* ---

    def start_watcher(self, rag_name: str | None = None) -> dict:
        """Start the file watcher."""
        params: dict[str, Any] = {}
        if rag_name:
            params["rag_name"] = rag_name
        return self._call("watcher.start", params)

    def stop_watcher(self) -> dict:
        """Stop the file watcher."""
        return self._call("watcher.stop")

    def watcher_status(self) -> dict:
        """Get watcher status."""
        return self._call("watcher.status")

    # --- store.* --- (UI compatibility)

    def get_store_stats(self, rag_name: str | None = None) -> dict:
        """Get store stats for a RAG entry."""
        params: dict[str, Any] = {}
        if rag_name:
            params["rag_name"] = rag_name
        return self._call("store.get", params)

    def close_store(self, db_path: str, force: bool = False) -> dict:
        """Close a cached VectorStore."""
        return self._call("store.close", {"db_path": db_path, "force": force})

    # --- system.* ---

    def ping(self) -> dict:
        """Health check."""
        return self._call("system.ping")

    def shutdown(self) -> dict:
        """Gracefully shut down the daemon."""
        try:
            return self._call("system.shutdown")
        except (ConnectionError, OSError):
            return {"ok": True}  # daemon shut down before we got the response

    def version(self) -> dict:
        """Get daemon version."""
        return self._call("system.version")

    # --- metrics.* ---

    def get_metrics_dashboard(self, rag_name: str | None = None) -> dict:
        """Return aggregated monitoring dashboard summary."""
        params: dict[str, Any] = {}
        if rag_name:
            params["rag_name"] = rag_name
        return self._call("metrics.dashboard", params)

    def get_indexing_history(self, rag_name: str | None = None, limit: int = 50) -> list[dict]:
        """Return recent indexing run history."""
        params: dict[str, Any] = {"limit": limit}
        if rag_name:
            params["rag_name"] = rag_name
        return self._call("metrics.indexing_history", params)

    def get_search_stats(self, rag_name: str | None = None, limit: int = 100) -> list[dict]:
        """Return recent search query stats."""
        params: dict[str, Any] = {"limit": limit}
        if rag_name:
            params["rag_name"] = rag_name
        return self._call("metrics.search_stats", params)

    def get_embedding_stats(self, rag_name: str | None = None, limit: int = 100) -> list[dict]:
        """Return recent embedding batch stats."""
        params: dict[str, Any] = {"limit": limit}
        if rag_name:
            params["rag_name"] = rag_name
        return self._call("metrics.embedding_stats", params)

    def get_system_timeline(self, limit: int = 100) -> list[dict]:
        """Return system resource usage timeline."""
        return self._call("metrics.system_timeline", {"limit": limit})

    def get_vector_store_details(self, rag_name: str | None = None) -> dict:
        """Return detailed vector store / ChromaDB stats."""
        params: dict[str, Any] = {}
        if rag_name:
            params["rag_name"] = rag_name
        return self._call("metrics.vector_store", params)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_daemon_alive() -> bool:
    """Check if the daemon PID file exists and the process is alive."""
    if not _PID_FILE.exists():
        return False
    try:
        pid = int(_PID_FILE.read_text(encoding="utf-8").strip())
    except (ValueError, OSError):
        return False

    # Check if process is alive
    try:
        if sys.platform == "win32":
            # On Windows, os.kill(pid, 0) doesn't work reliably.
            # Use ctypes to check if the process exists.
            import ctypes
            kernel32 = ctypes.windll.kernel32
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if handle:
                kernel32.CloseHandle(handle)
                return True
            return False
        else:
            os.kill(pid, 0)
            return True
    except (OSError, ProcessLookupError):
        return False
