"""Asyncio TCP daemon wrapping RagKnowledgeBaseAPI.

Centralises all data access (ChromaDB, registry.json, file_manifest.db,
config.yaml) in a single long-running daemon process.  CLI, MCP server
and UI become thin clients that communicate over JSON-RPC 2.0 via TCP
sockets on ``127.0.0.1``.

Usage
-----
Start from the command line::

    rag-kb-daemon          # uses defaults
    python -m rag_kb.daemon --port 9527 --idle-timeout 300

The daemon writes a PID file to ``DATA_DIR/daemon.pid`` and shuts itself
down after ``idle_timeout`` seconds without any client connections.
"""

from __future__ import annotations

import asyncio
import logging
import logging.handlers
import os
import signal
import sys
import time
from typing import Any

from rag_kb.config import DATA_DIR, AppSettings
from rag_kb.core import RagKnowledgeBaseAPI
from rag_kb.rpc_protocol import (
    DEFAULT_HOST,
    DEFAULT_IDLE_TIMEOUT,
    DEFAULT_PORT,
    ERR_AUTH_FAILED,
    ERR_FILE_NOT_FOUND,
    ERR_INDEX_ERROR,
    ERR_RAG_ALREADY_EXISTS,
    ERR_RAG_NOT_FOUND,
    INTERNAL_ERROR,
    INVALID_PARAMS,
    METHOD_NOT_FOUND,
    generate_auth_token,
    make_error,
    make_progress,
    make_response,
    read_frame_async,
    remove_auth_token,
    write_frame_async,
)

logger = logging.getLogger(__name__)


class RagDaemon:
    """Asyncio-based TCP daemon wrapping :class:`RagKnowledgeBaseAPI`."""

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        idle_timeout: int = DEFAULT_IDLE_TIMEOUT,
    ) -> None:
        self._host = host
        self._port = port
        self._idle_timeout = idle_timeout

        self._api: RagKnowledgeBaseAPI | None = None
        self._server: asyncio.Server | None = None
        self._idle_handle: asyncio.TimerHandle | None = None
        self._active_connections: int = 0
        self._pid_file = DATA_DIR / "daemon.pid"
        self._start_time: float = 0.0
        self._loop: asyncio.AbstractEventLoop | None = None
        self._auth_token: str = ""
        # Live indexing progress (visible to all clients via index.status)
        self._indexing_state: dict | None = None
        # Shutdown task reference (prevent GC of fire-and-forget task)
        self._shutdown_task: asyncio.Task[None] | None = None
        # Metrics counters
        self._total_rpc_calls: int = 0
        self._metrics_snapshot_timer: asyncio.TimerHandle | None = None
        # Cached index.status result to avoid redundant work during
        # frequent UI polling (every 1-10 s).
        self._status_cache: tuple[float, dict] | None = None
        self._STATUS_CACHE_TTL: float = 3.0  # seconds

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start the daemon: initialise API, listen, write PID file."""
        self._loop = asyncio.get_running_loop()
        self._start_time = time.monotonic()

        logger.info("=" * 60)
        logger.info("Daemon starting (PID %d)", os.getpid())
        logger.info(
            "  host=%s  port=%d  idle_timeout=%d", self._host, self._port, self._idle_timeout
        )
        logger.info("  data_dir=%s", DATA_DIR)
        logger.info("=" * 60)

        # Suppress noisy Windows ProactorEventLoop errors when clients
        # disconnect abruptly (e.g. "forcibly closed by the remote host").
        _orig_handler = self._loop.get_exception_handler()

        def _quiet_exception_handler(
            loop: asyncio.AbstractEventLoop, context: dict[str, Any]
        ) -> None:
            exc = context.get("exception")
            if isinstance(exc, (ConnectionResetError, ConnectionAbortedError, OSError)):
                logger.debug("Suppressed transport error: %s", exc)
                return
            # Fall back to the original / default handler
            if _orig_handler is not None:
                _orig_handler(loop, context)
            else:
                loop.default_exception_handler(context)

        self._loop.set_exception_handler(_quiet_exception_handler)

        # Initialise the core API
        logger.info("Initialising core API…")
        self._api = RagKnowledgeBaseAPI()
        logger.info("Core API initialised.")

        # Pre-open active store + watcher
        active = self._api.get_active_rag()
        if active:
            logger.info(
                "Active RAG: '%s' (model=%s, folders=%s)",
                active.name,
                active.embedding_model,
                active.source_folders,
            )
            self._api.get_store(active)
            self._api.start_watcher()

            # Check for incomplete indexing from a previous crash
            incomplete = self._api.check_incomplete_indexing()
            if incomplete:
                started = incomplete.get("started_at", "unknown")
                logger.warning(
                    "Detected incomplete indexing (started %s). "
                    "Scheduling automatic incremental re-index to recover.",
                    started,
                )
                # Schedule recovery reindex after the server is listening
                self._loop.call_soon(lambda: asyncio.ensure_future(self._recovery_reindex()))

        # WS6: Generate the auth token BEFORE starting the TCP server.
        # If we write the token after the server is listening, a client
        # that auto-starts the daemon could connect before the token file
        # exists, causing auth failures.
        self._auth_token = generate_auth_token()
        logger.info("Auth token generated and written to disk.")

        self._server = await asyncio.start_server(
            self._handle_client,
            self._host,
            self._port,
        )
        self._write_pid_file()
        self._reset_idle_timer()

        addrs = ", ".join(str(s.getsockname()) for s in self._server.sockets)
        logger.info("Daemon listening on %s (PID %d)", addrs, os.getpid())

        # Start periodic metrics snapshot loop
        self._start_metrics_snapshot_loop()

        # Register signal handlers (Unix-only; on Windows we rely on
        # KeyboardInterrupt / the idle timer / system.shutdown RPC).
        if sys.platform != "win32":
            for sig in (signal.SIGTERM, signal.SIGINT):
                self._loop.add_signal_handler(sig, lambda: asyncio.ensure_future(self.stop()))

        try:
            await self._server.serve_forever()
        except asyncio.CancelledError:
            pass
        finally:
            await self._cleanup()

    async def stop(self) -> None:
        """Initiate graceful shutdown."""
        logger.info("Daemon shutting down …")
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

    async def _cleanup(self) -> None:
        if self._idle_handle is not None:
            self._idle_handle.cancel()
        if self._api is not None:
            self._api.shutdown()
            self._api = None
        self._remove_pid_file()
        remove_auth_token()
        logger.info("Daemon stopped.")

    # ------------------------------------------------------------------
    # PID file
    # ------------------------------------------------------------------

    def _write_pid_file(self) -> None:
        self._pid_file.parent.mkdir(parents=True, exist_ok=True)
        self._pid_file.write_text(str(os.getpid()), encoding="utf-8")
        logger.debug("PID file written: %s", self._pid_file)

    def _remove_pid_file(self) -> None:
        try:
            self._pid_file.unlink(missing_ok=True)
            logger.debug("PID file removed.")
        except OSError as exc:
            logger.warning("Could not remove PID file: %s", exc)

    # ------------------------------------------------------------------
    # Idle timer
    # ------------------------------------------------------------------

    def _reset_idle_timer(self) -> None:
        if self._idle_handle is not None:
            self._idle_handle.cancel()
        if self._idle_timeout > 0 and self._loop is not None:
            self._idle_handle = self._loop.call_later(
                self._idle_timeout,
                self._on_idle_timeout,
            )

    def _on_idle_timeout(self) -> None:
        if self._active_connections == 0:
            logger.info("Idle timeout reached (%ds) — shutting down.", self._idle_timeout)
            self._shutdown_task = asyncio.ensure_future(self.stop())
        else:
            # Still have connections — reschedule
            self._reset_idle_timer()

    # ------------------------------------------------------------------
    # Auto-recovery
    # ------------------------------------------------------------------

    async def _recovery_reindex(self) -> None:
        """Run an incremental reindex to recover from a previous crash.

        Executed as a fire-and-forget task shortly after daemon startup.
        """
        assert self._api is not None
        logger.info("Recovery reindex: starting incremental index …")
        loop = asyncio.get_running_loop()
        try:
            state = await loop.run_in_executor(
                None,
                lambda: self._api.index(full=False),
            )
            logger.info(
                "Recovery reindex completed: status=%s files=%d chunks=%d duration=%.1fs",
                state.status,
                state.processed_files,
                state.total_chunks,
                state.duration_seconds,
            )
        except Exception:
            logger.exception("Recovery reindex failed")

    # ------------------------------------------------------------------
    # Client handling
    # ------------------------------------------------------------------

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        peer = writer.get_extra_info("peername")
        logger.info(
            "Client connected: %s (active_connections=%d)", peer, self._active_connections + 1
        )
        self._active_connections += 1
        self._reset_idle_timer()

        try:
            while True:
                try:
                    msg = await read_frame_async(reader)
                except (asyncio.IncompleteReadError, ConnectionError):
                    break
                except Exception as exc:
                    logger.warning("Frame read error from %s: %s", peer, exc)
                    break

                request_id = msg.get("id")
                method = msg.get("method", "")
                params = msg.get("params") or {}

                # ── Auth token validation ────────────────────────────
                msg_token = msg.get("_auth_token", "")
                if self._auth_token and msg_token != self._auth_token:
                    logger.warning("[%s] Auth failed for method %s", peer, method)
                    response = make_error(request_id, ERR_AUTH_FAILED, "Invalid auth token")
                    try:
                        await write_frame_async(writer, response)
                    except ConnectionError:
                        pass
                    break  # disconnect the client

                t0 = time.monotonic()
                self._total_rpc_calls += 1
                # Demote high-frequency polling methods to DEBUG to avoid
                # flooding the log (UI polls these every 1-10 s).
                _is_noisy = method in _NOISY_METHODS
                if _is_noisy:
                    logger.debug("[%s] → %s()", peer, method)
                else:
                    logger.info("[%s] → %s(%s)", peer, method, _summarise_params(params))

                try:
                    result = await self._dispatch(method, params, request_id, writer)
                    elapsed = time.monotonic() - t0
                    response = make_response(request_id, result)
                    if _is_noisy:
                        logger.debug("[%s] ← %s OK (%.3fs)", peer, method, elapsed)
                    else:
                        logger.info("[%s] ← %s OK (%.3fs)", peer, method, elapsed)
                except _AppError as exc:
                    elapsed = time.monotonic() - t0
                    logger.warning(
                        "[%s] ← %s ERROR %d: %s (%.3fs)", peer, method, exc.code, exc, elapsed
                    )
                    response = make_error(request_id, exc.code, str(exc))
                except Exception as exc:
                    elapsed = time.monotonic() - t0
                    logger.exception("[%s] ← %s UNHANDLED ERROR (%.3fs)", peer, method, elapsed)
                    response = make_error(request_id, INTERNAL_ERROR, str(exc))

                try:
                    await write_frame_async(writer, response)
                except ConnectionError:
                    break
        finally:
            self._active_connections -= 1
            self._reset_idle_timer()
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass
            logger.info(
                "Client disconnected: %s (active_connections=%d)", peer, self._active_connections
            )

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    async def _dispatch(
        self,
        method: str,
        params: dict,
        request_id: str | None,
        writer: asyncio.StreamWriter,
    ) -> Any:
        """Route a JSON-RPC method to the appropriate handler."""
        assert self._api is not None

        handler = _METHOD_TABLE.get(method)
        if handler is None:
            raise _AppError(METHOD_NOT_FOUND, f"Unknown method: {method}")

        return await handler(self, params, request_id, writer)

    # ------------------------------------------------------------------
    # RPC method implementations
    # ------------------------------------------------------------------

    # --- rag.* ---

    async def _rag_active_name(self, params: dict, rid: str | None, w) -> Any:
        assert self._api is not None
        return self._api.get_active_name()

    async def _rag_list(self, params: dict, rid: str | None, w) -> Any:
        assert self._api is not None
        rags = self._api.list_rags()
        active_name = self._api.get_active_name()
        return [
            {
                "name": r.name,
                "description": r.description,
                "is_active": r.name == active_name,
                "is_imported": r.is_imported,
                "detached": r.detached,
                "embedding_model": r.embedding_model,
                "file_count": r.file_count,
                "chunk_count": r.chunk_count,
                "source_folders": r.source_folders,
                "created_at": r.created_at,
                "db_path": r.db_path,
            }
            for r in rags
        ]

    async def _rag_create(self, params: dict, rid: str | None, w) -> Any:
        assert self._api is not None
        name = params.get("name")
        if not name:
            raise _AppError(INVALID_PARAMS, "'name' is required")
        logger.info(
            "Creating RAG '%s' (folders=%s, model=%s)",
            name,
            params.get("folders"),
            params.get("embedding_model"),
        )
        entry = self._api.create_rag(
            name=name,
            folders=params.get("folders"),
            description=params.get("description", ""),
            embedding_model=params.get("embedding_model"),
        )
        logger.info("RAG '%s' created at %s", entry.name, entry.db_path)
        return {"name": entry.name, "db_path": entry.db_path}

    async def _rag_switch(self, params: dict, rid: str | None, w) -> Any:
        assert self._api is not None
        name = params.get("name")
        if not name:
            raise _AppError(INVALID_PARAMS, "'name' is required")
        logger.info("Switching active RAG to '%s'", name)
        try:
            entry = self._api.switch_rag(name)
        except KeyError:
            raise _AppError(ERR_RAG_NOT_FOUND, f"RAG '{name}' not found") from None
        logger.info("Active RAG is now '%s'", entry.name)
        return {"name": entry.name, "db_path": entry.db_path}

    async def _rag_delete(self, params: dict, rid: str | None, w) -> Any:
        assert self._api is not None
        name = params.get("name")
        confirm = params.get("confirm", False)
        if not name:
            raise _AppError(INVALID_PARAMS, "'name' is required")
        if not confirm:
            raise _AppError(INVALID_PARAMS, "Deletion requires confirm=true")
        logger.info("Deleting RAG '%s'", name)
        try:
            self._api.delete_rag(name)
        except KeyError:
            raise _AppError(ERR_RAG_NOT_FOUND, f"RAG '{name}' not found") from None
        logger.info("RAG '%s' deleted", name)
        return {"ok": True}

    async def _rag_detach(self, params: dict, rid: str | None, w) -> Any:
        assert self._api is not None
        name = params.get("name")
        if not name:
            raise _AppError(INVALID_PARAMS, "'name' is required")
        try:
            self._api.detach_rag(name)
        except KeyError:
            raise _AppError(ERR_RAG_NOT_FOUND, f"RAG '{name}' not found") from None
        return {"ok": True}

    async def _rag_attach(self, params: dict, rid: str | None, w) -> Any:
        assert self._api is not None
        name = params.get("name")
        if not name:
            raise _AppError(INVALID_PARAMS, "'name' is required")
        try:
            self._api.attach_rag(name)
        except KeyError:
            raise _AppError(ERR_RAG_NOT_FOUND, f"RAG '{name}' not found") from None
        return {"ok": True}

    async def _rag_get(self, params: dict, rid: str | None, w) -> Any:
        assert self._api is not None
        name = params.get("name")
        if not name:
            raise _AppError(INVALID_PARAMS, "'name' is required")
        try:
            entry = self._api.get_rag(name)
        except KeyError:
            raise _AppError(ERR_RAG_NOT_FOUND, f"RAG '{name}' not found") from None
        return {
            "name": entry.name,
            "description": entry.description,
            "db_path": entry.db_path,
            "embedding_model": entry.embedding_model,
            "source_folders": entry.source_folders,
            "created_at": entry.created_at,
            "is_imported": entry.is_imported,
            "detached": entry.detached,
            "file_count": entry.file_count,
            "chunk_count": entry.chunk_count,
        }

    async def _rag_update(self, params: dict, rid: str | None, w) -> Any:
        """Update a RAG entry's mutable fields (source_folders, description)."""
        assert self._api is not None
        name = params.get("name")
        if not name:
            raise _AppError(INVALID_PARAMS, "'name' is required")
        try:
            entry = self._api.get_rag(name)
        except KeyError:
            raise _AppError(ERR_RAG_NOT_FOUND, f"RAG '{name}' not found") from None
        if "source_folders" in params:
            entry.source_folders = params["source_folders"]
        if "description" in params:
            entry.description = params["description"]
        self._api.update_rag(entry)
        return {"ok": True}

    # --- search.* ---

    async def _search_query(self, params: dict, rid: str | None, w) -> Any:
        assert self._api is not None
        query = params.get("query")
        if not query:
            raise _AppError(INVALID_PARAMS, "'query' is required")
        top_k = params.get("top_k", 5)
        rag_name = params.get("rag_name")
        logger.info("Search: query=%r top_k=%d rag=%s", query[:80], top_k, rag_name or "(active)")
        results = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: self._api.search(
                query=query,
                n_results=top_k,
                rag_name=rag_name,
                min_score=params.get("min_score"),
            ),
        )
        logger.info("Search returned %d result(s)", len(results))
        return [
            {
                "text": r.text,
                "source_file": r.source_file,
                "score": r.score,
                "chunk_index": r.chunk_index,
                "metadata": r.metadata,
            }
            for r in results
        ]

    # --- index.* ---

    async def _index_run(self, params: dict, rid: str | None, w: asyncio.StreamWriter) -> Any:
        assert self._api is not None
        rag_name = params.get("rag_name")
        full = params.get("force", False) or params.get("full", False)
        workers = params.get("workers")

        loop = asyncio.get_running_loop()

        def on_progress(state):
            """Send progress notification back to the requesting client
            **and** update the shared ``_indexing_state`` dict so that
            any other client polling ``index.status`` can see live
            progress."""
            # Update shared state (visible via index.status)
            self._indexing_state = {
                "is_indexing": True,
                "status": state.status,
                "processed_files": state.processed_files,
                "total_files": state.total_files,
                "current_file": state.current_file or "",
                "progress": state.progress,
            }

            try:
                notification = make_progress(
                    request_id=rid or "",
                    current=state.processed_files,
                    total=state.total_files,
                    message=state.current_file or state.status,
                )
                # Schedule the write on the event loop from the worker thread
                asyncio.run_coroutine_threadsafe(
                    write_frame_async(w, notification),
                    loop,
                )
            except Exception:
                pass  # Don't let progress errors kill the indexing

        # Mark indexing as started
        logger.info(
            "Indexing started: rag=%s full=%s workers=%s", rag_name or "(active)", full, workers
        )
        self._indexing_state = {
            "is_indexing": True,
            "status": "starting",
            "processed_files": 0,
            "total_files": 0,
            "current_file": "",
            "progress": 0.0,
        }

        try:
            state = await loop.run_in_executor(
                None,
                lambda: self._api.index(
                    rag_name=rag_name,
                    full=full,
                    workers=workers,
                    on_progress=on_progress,
                ),
            )
        except RuntimeError as exc:
            self._indexing_state = None
            logger.exception("Indexing failed")
            raise _AppError(ERR_INDEX_ERROR, str(exc)) from exc
        finally:
            # Clear live state once done
            self._indexing_state = None

        logger.info(
            "Indexing completed: status=%s files=%d chunks=%d errors=%d duration=%.1fs",
            state.status,
            state.processed_files,
            state.total_files,
            len(state.errors),
            state.duration_seconds,
        )
        return {
            "status": state.status,
            "processed_files": state.processed_files,
            "total_files": state.total_files,
            "total_chunks": state.total_chunks,
            "errors": state.errors,
            "duration_seconds": state.duration_seconds,
        }

    async def _index_status(self, params: dict, rid: str | None, w) -> Any:
        assert self._api is not None
        rag_name = params.get("rag_name")

        # When indexing is actively running, skip the expensive
        # store stats call.  The live progress dict already provides
        # everything the UI needs.
        is_indexing = self._indexing_state is not None

        # Return cached result for frequent polling when not indexing
        # and no specific rag_name override is requested.
        now = time.monotonic()
        if not is_indexing and not rag_name and self._status_cache is not None:
            ts, cached = self._status_cache
            if now - ts < self._STATUS_CACHE_TTL:
                # Refresh live indexing overlay (may have just started)
                live = self._indexing_state
                if live is not None:
                    cached = {**cached, "indexing": dict(live)}
                return cached

        loop = asyncio.get_running_loop()
        status = await loop.run_in_executor(
            None,
            lambda: self._api.get_index_status(
                rag_name=rag_name,
                skip_store_stats=is_indexing,
            ),
        )
        result = {
            "active_rag": status.active_rag,
            "is_imported": status.is_imported,
            "total_files": status.total_files,
            "total_chunks": status.total_chunks,
            "watcher_running": status.watcher_running,
            "last_status": status.last_status,
            "last_indexed": status.last_indexed,
            "errors": status.errors,
        }
        # Attach warnings (e.g. incomplete indexing from crash)
        if status.warnings:
            result["warnings"] = status.warnings

        # Cache the base result (without live indexing overlay)
        if not rag_name:
            self._status_cache = (now, result)

        # Attach live indexing progress if an indexing operation is running
        live = self._indexing_state
        if live is not None:
            result["indexing"] = dict(live)
        return result

    async def _index_files(self, params: dict, rid: str | None, w) -> Any:
        assert self._api is not None
        rag_name = params.get("rag_name")
        offset = params.get("offset", 0)
        limit = params.get("limit", 0)
        file_filter = params.get("filter", "")
        result = self._api.list_indexed_files(
            rag_name=rag_name,
            offset=offset,
            limit=limit,
            filter=file_filter,
        )
        return {
            "files": [{"file": f.file, "chunk_count": f.chunk_count} for f in result.files],
            "total": result.total,
            "offset": result.offset,
            "limit": result.limit,
            "filter": result.filter,
        }

    async def _index_reindex(self, params: dict, rid: str | None, w: asyncio.StreamWriter) -> Any:
        # reindex is just index with full=True
        params_copy = dict(params)
        params_copy["full"] = True
        return await self._index_run(params_copy, rid, w)

    async def _index_document(self, params: dict, rid: str | None, w) -> Any:
        assert self._api is not None
        source_file = params.get("source_file")
        if not source_file:
            raise _AppError(INVALID_PARAMS, "'source_file' is required")
        rag_name = params.get("rag_name")
        chunks = self._api.get_document_content(source_file, rag_name=rag_name)
        if not chunks:
            raise _AppError(ERR_FILE_NOT_FOUND, f"No chunks found for '{source_file}'")
        return [{"id": c.id, "text": c.text, "metadata": c.metadata} for c in chunks]

    # --- index.changes ---

    async def _index_changes(self, params: dict, rid: str | None, w) -> Any:
        assert self._api is not None
        rag_name = params.get("rag_name")
        changes = self._api.scan_file_changes(rag_name=rag_name)
        return {
            "new": changes.new,
            "modified": changes.modified,
            "removed": changes.removed,
        }

    # --- index.cancel ---

    async def _index_cancel(self, params: dict, rid: str | None, w) -> Any:
        assert self._api is not None
        logger.info("Indexing cancellation requested")
        cancelled = self._api.cancel_indexing()
        if cancelled and self._indexing_state is not None:
            self._indexing_state["status"] = "cancelling"
        logger.info("Cancellation result: %s", "signalled" if cancelled else "no running indexer")
        return {"cancelled": cancelled}

    async def _index_verify(self, params: dict, rid: str | None, w) -> Any:
        """Verify manifest ↔ vector-store consistency."""
        assert self._api is not None
        rag_name = params.get("rag_name")
        logger.info("Verifying index consistency (rag=%s)", rag_name or "(active)")
        result = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: self._api.verify_index_consistency(rag_name=rag_name),
        )
        logger.info("Verify result: ok=%s", result.get("ok"))
        return result

    # --- share.* ---

    async def _share_export(self, params: dict, rid: str | None, w) -> Any:
        assert self._api is not None
        name = params.get("name")
        output_path = params.get("output_path")
        if not name or not output_path:
            raise _AppError(INVALID_PARAMS, "'name' and 'output_path' are required")
        logger.info("Exporting RAG '%s' → %s", name, output_path)
        try:
            result = self._api.export_rag(name, output_path)
        except KeyError:
            raise _AppError(ERR_RAG_NOT_FOUND, f"RAG '{name}' not found") from None
        logger.info("Export completed: %s", result)
        return {"path": result}

    async def _share_import(self, params: dict, rid: str | None, w) -> Any:
        assert self._api is not None
        path = params.get("path")
        if not path:
            raise _AppError(INVALID_PARAMS, "'path' is required")
        name = params.get("name")
        logger.info("Importing RAG from '%s' (name=%s)", path, name or "(auto)")
        try:
            imported_name = self._api.import_rag(path, new_name=name)
        except ValueError as exc:
            raise _AppError(ERR_RAG_ALREADY_EXISTS, str(exc)) from None
        logger.info("Import completed: '%s'", imported_name)
        return {"name": imported_name}

    async def _share_peek(self, params: dict, rid: str | None, w) -> Any:
        assert self._api is not None
        path = params.get("path")
        if not path:
            raise _AppError(INVALID_PARAMS, "'path' is required")
        return self._api.peek_rag_file(path)

    # --- config.* ---

    async def _config_get(self, params: dict, rid: str | None, w) -> Any:
        assert self._api is not None
        return self._api.get_config().model_dump()

    async def _config_set(self, params: dict, rid: str | None, w) -> Any:
        assert self._api is not None
        key = params.get("key")
        value = params.get("value")
        if not key:
            raise _AppError(INVALID_PARAMS, "'key' is required")
        settings = self._api.get_config()
        if not hasattr(settings, key):
            raise _AppError(INVALID_PARAMS, f"Unknown setting: '{key}'")
        logger.info("Config update: %s = %r", key, value)
        setattr(settings, key, value)
        self._api.save_config(settings)
        return {"ok": True}

    async def _config_save(self, params: dict, rid: str | None, w) -> Any:
        """Accept a full settings dict and persist it."""
        assert self._api is not None
        settings_data = params.get("settings")
        if settings_data:
            settings = AppSettings(**settings_data)
            self._api.save_config(settings)
        else:
            self._api.save_config()
        return {"ok": True}

    async def _config_reload(self, params: dict, rid: str | None, w) -> Any:
        assert self._api is not None
        settings = self._api.reload_config()
        return settings.model_dump()

    async def _config_models_download(self, params: dict, rid: str | None, w) -> Any:
        assert self._api is not None
        loop = asyncio.get_running_loop()
        saved = await loop.run_in_executor(
            None,
            lambda: self._api.download_models(
                output_dir=params.get("output_dir"),
                model_name=params.get("model_name"),
            ),
        )
        return {"paths": [str(p) for p in saved], "count": len(saved)}

    # --- models.* ---

    async def _models_list(self, params: dict, rid: str | None, w) -> Any:
        assert self._api is not None
        model_type = params.get("model_type")
        return self._api.list_models(model_type=model_type)

    async def _models_info(self, params: dict, rid: str | None, w) -> Any:
        assert self._api is not None
        model_name = params.get("model_name")
        if not model_name:
            raise _AppError(INVALID_PARAMS, "'model_name' is required")
        info = self._api.get_model_info(model_name)
        if info is None:
            raise _AppError(INVALID_PARAMS, f"Model '{model_name}' not found")
        return info

    async def _models_download(self, params: dict, rid: str | None, w) -> Any:
        assert self._api is not None
        model_name = params.get("model_name")
        if not model_name:
            raise _AppError(INVALID_PARAMS, "'model_name' is required")
        trust = params.get("trust_remote_code")
        loop = asyncio.get_running_loop()
        path = await loop.run_in_executor(
            None,
            lambda: self._api.download_model(model_name, trust_remote_code=trust),
        )
        return {"path": str(path), "model_name": model_name}

    async def _models_delete(self, params: dict, rid: str | None, w) -> Any:
        assert self._api is not None
        model_name = params.get("model_name")
        if not model_name:
            raise _AppError(INVALID_PARAMS, "'model_name' is required")
        deleted = self._api.delete_model(model_name)
        return {"deleted": deleted, "model_name": model_name}

    async def _models_trust(self, params: dict, rid: str | None, w) -> Any:
        assert self._api is not None
        model_name = params.get("model_name")
        if not model_name:
            raise _AppError(INVALID_PARAMS, "'model_name' is required")
        self._api.trust_model(model_name)
        return {"ok": True, "model_name": model_name}

    # --- watcher.* ---

    async def _watcher_start(self, params: dict, rid: str | None, w) -> Any:
        assert self._api is not None
        rag_name = params.get("rag_name")
        logger.info("Starting file watcher (rag=%s)", rag_name or "(active)")
        self._api.start_watcher(rag_name=rag_name)
        return {"ok": True}

    async def _watcher_stop(self, params: dict, rid: str | None, w) -> Any:
        assert self._api is not None
        logger.info("Stopping file watcher")
        self._api.stop_watcher()
        return {"ok": True}

    async def _watcher_status(self, params: dict, rid: str | None, w) -> Any:
        assert self._api is not None
        running = self._api.is_watcher_running()
        active = self._api.get_active_rag()
        return {
            "running": running,
            "watched_rag": active.name if active else None,
            "watched_folders": active.source_folders if active else [],
        }

    # --- system.* ---

    async def _system_ping(self, params: dict, rid: str | None, w) -> Any:
        uptime = time.monotonic() - self._start_time
        return {"ok": True, "uptime": round(uptime, 1), "pid": os.getpid()}

    async def _system_shutdown(self, params: dict, rid: str | None, w) -> Any:
        logger.info("Shutdown requested via RPC")
        # Schedule shutdown after responding
        asyncio.get_running_loop().call_soon(lambda: asyncio.ensure_future(self.stop()))
        return {"ok": True}

    async def _system_version(self, params: dict, rid: str | None, w) -> Any:
        from rag_kb import __version__

        return {"version": __version__}

    # --- store.* --- (for UI compatibility)

    async def _store_get(self, params: dict, rid: str | None, w) -> Any:
        """Get store stats for a RAG entry.

        Uses ``get_stats_summary()`` when indexing is in progress to avoid
        contention with ChromaDB writes.  Runs in a thread executor to
        keep the event loop responsive.
        """
        assert self._api is not None
        rag_name = params.get("rag_name")
        if rag_name:
            entry = self._api.get_rag(rag_name)
        else:
            entry = self._api.get_active_rag()
        if not entry:
            return {"total_files": 0, "total_chunks": 0}
        store = self._api.get_store(entry)
        if not store:
            return {"total_files": 0, "total_chunks": 0}

        is_indexing = self._indexing_state is not None

        def _get():
            if is_indexing:
                stats = store.get_stats_summary()
            else:
                stats = store.get_stats()
            return {
                "total_files": stats.total_files,
                "total_chunks": stats.total_chunks,
            }

        return await asyncio.get_running_loop().run_in_executor(None, _get)

    async def _store_close(self, params: dict, rid: str | None, w) -> Any:
        assert self._api is not None
        db_path = params.get("db_path")
        if not db_path:
            raise _AppError(INVALID_PARAMS, "'db_path' is required")
        self._api.close_store(db_path, force=params.get("force", False))
        return {"ok": True}

    # --- metrics.* ---

    async def _metrics_dashboard(self, params: dict, rid: str | None, w) -> Any:
        rag_name = params.get("rag_name")
        from rag_kb.metrics import MetricsCollector

        return MetricsCollector.get().get_dashboard(rag_name)

    async def _metrics_indexing_history(self, params: dict, rid: str | None, w) -> Any:
        rag_name = params.get("rag_name")
        limit = params.get("limit", 50)
        from rag_kb.metrics import MetricsCollector

        return MetricsCollector.get().get_indexing_history(rag_name, limit)

    async def _metrics_search_stats(self, params: dict, rid: str | None, w) -> Any:
        rag_name = params.get("rag_name")
        limit = params.get("limit", 100)
        from rag_kb.metrics import MetricsCollector

        return MetricsCollector.get().get_search_stats(rag_name, limit)

    async def _metrics_embedding_stats(self, params: dict, rid: str | None, w) -> Any:
        rag_name = params.get("rag_name")
        limit = params.get("limit", 100)
        from rag_kb.metrics import MetricsCollector

        return MetricsCollector.get().get_embedding_stats(rag_name, limit)

    async def _metrics_system_timeline(self, params: dict, rid: str | None, w) -> Any:
        limit = params.get("limit", 100)
        from rag_kb.metrics import MetricsCollector

        return MetricsCollector.get().get_system_timeline(limit)

    async def _metrics_vector_store(self, params: dict, rid: str | None, w) -> Any:
        """Return detailed vector store stats for the active (or specified) RAG.

        Runs in a thread executor to avoid blocking the event loop —
        even the lightweight ``get_detailed_stats()`` may touch SQLite
        for the file manifest or iterate ChromaDB metadata for large RAGs.
        """
        assert self._api is not None
        rag_name = params.get("rag_name")
        if rag_name:
            entry = self._api.get_rag(rag_name)
        else:
            entry = self._api.get_active_rag()
        if not entry:
            return {}
        store = self._api.get_store(entry)
        if not store:
            return {}

        def _get():
            try:
                return store.get_detailed_stats()
            except Exception:
                # Collection may be temporarily deleted during a full reindex
                return {}

        return await asyncio.get_running_loop().run_in_executor(None, _get)

    # --- Periodic system snapshot ---

    def _start_metrics_snapshot_loop(self) -> None:
        """Start a periodic loop that captures system snapshots."""
        if self._loop is None:
            return

        async def _capture():
            try:
                from rag_kb.metrics import MetricsCollector, capture_system_snapshot

                uptime = time.monotonic() - self._start_time
                snap = capture_system_snapshot(
                    daemon_uptime=uptime,
                    active_connections=self._active_connections,
                    total_rpc_calls=self._total_rpc_calls,
                )
                MetricsCollector.get().record_system_snapshot(snap)
            except Exception:
                logger.debug("System snapshot failed", exc_info=True)

            # Also capture vector store snapshot for the active RAG
            # Skip while indexing is in progress to avoid SQLite lock
            # conflicts with concurrent ChromaDB writes.
            if self._indexing_state is not None:
                logger.debug("Skipping vector store snapshot (indexing in progress)")
            else:
                try:
                    from rag_kb.metrics import MetricsCollector, VectorStoreSnapshot

                    if self._api:
                        entry = self._api.get_active_rag()
                        if entry:
                            store = self._api.get_store(entry)
                            if store:
                                stats = store.get_stats()
                                MetricsCollector.get().record_vector_store_snapshot(
                                    VectorStoreSnapshot(
                                        rag_name=entry.name,
                                        timestamp=time.time(),
                                        total_chunks=stats.total_chunks,
                                        total_files=stats.total_files,
                                        db_size_bytes=stats.db_size_bytes,
                                        avg_chunks_per_file=stats.avg_chunks_per_file,
                                        collection_name="rag_chunks",
                                    )
                                )
                except Exception:
                    logger.debug("Vector store snapshot failed", exc_info=True)

            # Purge old metrics periodically (once per snapshot cycle)
            try:
                from rag_kb.metrics import MetricsCollector

                MetricsCollector.get().purge_old(30)
            except Exception:
                pass

            # Schedule next snapshot
            if self._loop and self._loop.is_running():
                self._metrics_snapshot_timer = self._loop.call_later(
                    60, lambda: asyncio.ensure_future(_capture())
                )

        # First capture after 10s, then every 60s
        if self._loop and self._loop.is_running():
            self._metrics_snapshot_timer = self._loop.call_later(
                10, lambda: asyncio.ensure_future(_capture())
            )


# ---------------------------------------------------------------------------
# Method routing table
# ---------------------------------------------------------------------------

_METHOD_TABLE: dict[str, Any] = {
    # rag.*
    "rag.active_name": RagDaemon._rag_active_name,
    "rag.list": RagDaemon._rag_list,
    "rag.create": RagDaemon._rag_create,
    "rag.switch": RagDaemon._rag_switch,
    "rag.delete": RagDaemon._rag_delete,
    "rag.detach": RagDaemon._rag_detach,
    "rag.attach": RagDaemon._rag_attach,
    "rag.get": RagDaemon._rag_get,
    "rag.update": RagDaemon._rag_update,
    # search.*
    "search.query": RagDaemon._search_query,
    # index.*
    "index.run": RagDaemon._index_run,
    "index.status": RagDaemon._index_status,
    "index.files": RagDaemon._index_files,
    "index.reindex": RagDaemon._index_reindex,
    "index.document": RagDaemon._index_document,
    "index.changes": RagDaemon._index_changes,
    "index.cancel": RagDaemon._index_cancel,
    "index.verify": RagDaemon._index_verify,
    # share.*
    "share.export": RagDaemon._share_export,
    "share.import": RagDaemon._share_import,
    "share.peek": RagDaemon._share_peek,
    # config.*
    "config.get": RagDaemon._config_get,
    "config.set": RagDaemon._config_set,
    "config.save": RagDaemon._config_save,
    "config.reload": RagDaemon._config_reload,
    "config.models.download": RagDaemon._config_models_download,
    # models.*
    "models.list": RagDaemon._models_list,
    "models.info": RagDaemon._models_info,
    "models.download": RagDaemon._models_download,
    "models.delete": RagDaemon._models_delete,
    "models.trust": RagDaemon._models_trust,
    # watcher.*
    "watcher.start": RagDaemon._watcher_start,
    "watcher.stop": RagDaemon._watcher_stop,
    "watcher.status": RagDaemon._watcher_status,
    # system.*
    "system.ping": RagDaemon._system_ping,
    "system.shutdown": RagDaemon._system_shutdown,
    "system.version": RagDaemon._system_version,
    # store.* (UI helpers)
    "store.get": RagDaemon._store_get,
    "store.close": RagDaemon._store_close,
    # metrics.*
    "metrics.dashboard": RagDaemon._metrics_dashboard,
    "metrics.indexing_history": RagDaemon._metrics_indexing_history,
    "metrics.search_stats": RagDaemon._metrics_search_stats,
    "metrics.embedding_stats": RagDaemon._metrics_embedding_stats,
    "metrics.system_timeline": RagDaemon._metrics_system_timeline,
    "metrics.vector_store": RagDaemon._metrics_vector_store,
}


# ---------------------------------------------------------------------------
# Internal error type
# ---------------------------------------------------------------------------


class _AppError(Exception):
    """Exception carrying a JSON-RPC error code."""

    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code


# Methods that are polled frequently (every 1-10 s) by the UI / MCP;
# logged at DEBUG instead of INFO to avoid flooding the log.
_NOISY_METHODS: frozenset[str] = frozenset(
    {
        "index.status",
        "system.ping",
        "watcher.status",
        "rag.list",
        "store.get",
        "config.get",
        "metrics.dashboard",
        "metrics.system_timeline",
        "metrics.vector_store",
    }
)


# ---------------------------------------------------------------------------
# Param summarisation (avoid logging huge blobs)
# ---------------------------------------------------------------------------


def _summarise_params(params: dict) -> str:
    """Return a compact summary of RPC params for log readability."""
    if not params:
        return ""
    parts: list[str] = []
    for k, v in params.items():
        display = v
        if isinstance(v, str) and len(v) > 80:
            display = v[:77] + "…"
        elif isinstance(v, list) and len(v) > 5:
            display = f"[{len(v)} items]"
        parts.append(f"{k}={display!r}")
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

LOG_FILE = DATA_DIR / "daemon.log"
_LOG_MAX_BYTES = 10 * 1024 * 1024  # 10 MB per file
_LOG_BACKUP_COUNT = 3  # keep 3 rotated copies


def setup_daemon_logging(*, verbose: bool = False) -> None:
    """Configure logging for the daemon process.

    Logs go to:
    - **File** (``DATA_DIR/daemon.log``) — always at DEBUG level with
      timestamps, using ``RotatingFileHandler`` (10 MB × 3 backups).
    - **stderr** — at INFO (or DEBUG if *verbose*).
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    # Remove any existing handlers (e.g. from basicConfig)
    root.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    # ---- File handler (always DEBUG) ----
    fh = logging.handlers.RotatingFileHandler(
        str(LOG_FILE),
        maxBytes=_LOG_MAX_BYTES,
        backupCount=_LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    # ---- Console handler ----
    ch = logging.StreamHandler(sys.stderr)
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)
    ch.setFormatter(fmt)
    root.addHandler(ch)

    # Quieten noisy libraries
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
        "PIL",
        "PIL.PngImagePlugin",
        "PIL.Image",
    ):
        logging.getLogger(lib).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    idle_timeout: int = DEFAULT_IDLE_TIMEOUT,
    verbose: bool = False,
) -> None:
    """Start the daemon (blocking).

    Ensures only one daemon runs at a time: if a previous instance
    is alive (stuck or healthy), it is killed before starting.
    """
    setup_daemon_logging(verbose=verbose)

    logger.info("Log file: %s", LOG_FILE)

    # --- Single-instance guard ---
    from rag_kb.daemon_client import _is_daemon_alive, _read_pid, kill_stale_daemon

    stale_pid = _read_pid()
    if stale_pid is not None and stale_pid != os.getpid() and _is_daemon_alive():
        logger.warning(
            "Previous daemon (PID %d) is still running. Killing it …",
            stale_pid,
        )
        if not kill_stale_daemon(graceful_timeout=5.0):
            logger.error(
                "Could not kill previous daemon (PID %d). Aborting.",
                stale_pid,
            )
            sys.exit(1)
        # Small delay to let the OS release the port
        time.sleep(0.5)

    daemon = RagDaemon(host, port, idle_timeout)

    try:
        asyncio.run(daemon.start())
    except KeyboardInterrupt:
        logger.info("Daemon interrupted.")


def _cli_main() -> None:
    """CLI entry point (``rag-kb-daemon``) with argument parsing."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="rag-kb-daemon",
        description="RAG Knowledge Base daemon (JSON-RPC 2.0 over TCP)",
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help=f"Bind address (default: {DEFAULT_HOST})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"TCP port (default: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--idle-timeout",
        type=int,
        default=DEFAULT_IDLE_TIMEOUT,
        help=f"Shutdown after N seconds idle (default: {DEFAULT_IDLE_TIMEOUT}, 0=never)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug-level console output",
    )
    args = parser.parse_args()
    main(host=args.host, port=args.port, idle_timeout=args.idle_timeout, verbose=args.verbose)


if __name__ == "__main__":
    _cli_main()
