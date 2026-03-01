"""Unified service layer for RAG Knowledge Base.

This module provides ``RagKnowledgeBaseAPI`` — the **single** entry-point for
all business logic.  CLI, UI and MCP interfaces are thin adapters that translate
their I/O format (argparse, Qt signals, MCP JSON) into calls to this class.

Design decisions
----------------
* **Stateful class** — holds a store cache, watcher, registry and settings so
  that each interface does not have to re-implement lifecycle management.
* **Progress callback** — long-running operations accept an optional
  ``ProgressCallback`` (message, 0.0–1.0 fraction) so that each interface
  can wire it up in its own way (CLI → print, UI → Qt signal, MCP → no-op).
* **Thread-safe** — a ``threading.Lock`` protects mutable internal state.
"""

from __future__ import annotations

import functools
import logging
import os
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from rag_kb.config import (
    AppSettings,
    RagEntry,
    RagRegistry,
    safe_display_path,
)
from rag_kb.embedder import embed_query
from rag_kb.file_manifest import FileManifest
from rag_kb.indexer import Indexer, IndexingState
from rag_kb.search import (
    bm25_search,
    get_bm25_cache,
    hybrid_fuse_scores,
    mmr_diversify,
    rerank_cross_encoder,
)
from rag_kb.sharing import export_rag as _sharing_export
from rag_kb.sharing import import_rag as _sharing_import
from rag_kb.sharing import peek_rag_file
from rag_kb.vector_store import SearchResult, VectorStore
from rag_kb.watcher import FolderWatcher

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public type aliases
# ---------------------------------------------------------------------------

ProgressCallback = Callable[[str, float], None]
"""Signature: (message, fraction_0_to_1) → None."""


def _locked(method):
    """Decorator: acquires ``self._lock`` for the entire method call.

    Uses ``RLock`` so inner calls to other locked methods (or internal
    helpers that also acquire the lock) are safe — re-entrant.
    """

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return wrapper


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SearchResultItem:
    """A single search hit returned by :pymeth:`RagKnowledgeBaseAPI.search`."""

    text: str
    source_file: str  # display path (relative to source folders)
    score: float  # 0.0 – 1.0
    chunk_index: int
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class IndexStatus:
    """Structured status snapshot for the active RAG."""

    active_rag: str | None = None
    is_imported: bool | None = None
    total_files: int = 0
    total_chunks: int = 0
    watcher_running: bool = False
    last_status: str | None = None
    last_indexed: str | None = None
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class FileInfo:
    """One file in the index."""

    file: str  # display path
    chunk_count: int


@dataclass
class PaginatedFileList:
    """Paginated result from :pymeth:`RagKnowledgeBaseAPI.list_indexed_files`."""

    files: list[FileInfo]
    total: int  # total matching files
    offset: int = 0
    limit: int = 100
    filter: str = ""  # the filter that was applied (if any)


@dataclass
class ChunkInfo:
    """A single stored chunk."""

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FileChanges:
    """Result of scanning source folders against the manifest."""

    new: list[str] = field(default_factory=list)
    modified: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Unified API
# ---------------------------------------------------------------------------


class RagKnowledgeBaseAPI:
    """Single point of management for all RAG operations.

    Instantiate once per process and share across CLI / UI / MCP.
    """

    def __init__(
        self,
        settings: AppSettings | None = None,
        registry: RagRegistry | None = None,
    ) -> None:
        self.settings: AppSettings = settings or AppSettings.load()
        self.registry: RagRegistry = registry or RagRegistry()
        self._stores: dict[str, VectorStore] = {}  # norm(db_path) → store
        self._watcher: FolderWatcher | None = None
        self._last_index_state: IndexingState | None = None
        self._lock = threading.RLock()
        self._current_indexer: Indexer | None = None
        self._indexer_lock = threading.Lock()  # guards _current_indexer only
        self._indexing_rags: set[str] = set()  # RAG names currently being indexed

    # -- internal helpers ---------------------------------------------------

    def _norm_path(self, db_path: str) -> str:
        return os.path.normcase(os.path.normpath(db_path))

    def _get_store(self, entry: RagEntry | None) -> VectorStore | None:
        """Return a cached VectorStore for *entry*, opening one if needed."""
        if entry is None:
            return None
        norm = self._norm_path(entry.db_path)
        with self._lock:
            if norm in self._stores:
                cached = self._stores[norm]
                if cached._client is not None:
                    return cached
                del self._stores[norm]
            store = VectorStore(
                entry.db_path,
                hnsw_ef_construction=self.settings.hnsw_ef_construction,
                hnsw_m=self.settings.hnsw_m,
            )
            self._stores[norm] = store
            return store

    def _close_store(self, db_path: str, *, force: bool = False) -> None:
        norm = self._norm_path(db_path)
        with self._lock:
            store = self._stores.pop(norm, None)
        if store is not None:
            store.close(force=force)

    def _ensure_active(self) -> RagEntry:
        active = self.registry.get_active()
        if active is None:
            raise RuntimeError("No active RAG database. Use switch_rag or create one first.")
        return active

    def _ensure_store(self, entry: RagEntry | None = None) -> VectorStore:
        entry = entry or self._ensure_active()
        store = self._get_store(entry)
        if store is None:
            raise RuntimeError("No active RAG database. Use switch_rag or create one first.")
        return store

    def _start_watcher(self, entry: RagEntry | None) -> FolderWatcher | None:
        if entry is None or entry.is_imported or entry.detached or not entry.source_folders:
            return None
        w = FolderWatcher(entry, self.registry, self.settings)
        w.start()
        return w

    def _sync_watcher(self, entry: RagEntry | None) -> None:
        """Stop old watcher and (re)start one for *entry* if applicable."""
        with self._lock:
            if self._watcher is not None:
                self._watcher.stop()
                self._watcher = None
            self._watcher = self._start_watcher(entry)

    # ======================================================================
    # RAG CRUD
    # ======================================================================

    @_locked
    def create_rag(
        self,
        name: str,
        folders: list[str] | None = None,
        description: str = "",
        embedding_model: str | None = None,
    ) -> RagEntry:
        """Create a new RAG knowledge base."""
        logger.info("Creating RAG '%s' (folders=%s, model=%s)", name, folders, embedding_model)
        entry = self.registry.create_rag(
            name=name,
            description=description,
            folders=folders or [],
            embedding_model=embedding_model or self.settings.embedding_model,
        )
        logger.info("RAG '%s' created at %s", name, entry.db_path)
        return entry

    @_locked
    def list_rags(self) -> list[RagEntry]:
        """Return all registered RAGs."""
        return self.registry.list_rags()

    @_locked
    def get_active_rag(self) -> RagEntry | None:
        """Return the currently active RAG, or None."""
        return self.registry.get_active()

    @_locked
    def get_active_name(self) -> str | None:
        """Return the name of the currently active RAG, or None."""
        return self.registry.get_active_name()

    @_locked
    def get_rag(self, name: str) -> RagEntry:
        """Get a specific RAG by name. Raises KeyError if not found."""
        return self.registry.get_rag(name)

    @_locked
    def switch_rag(self, name: str) -> RagEntry:
        """Set *name* as the active RAG, reopening store & watcher."""
        logger.info("Switching active RAG to '%s'", name)
        self.registry.set_active(name)

        # Close old store and watcher
        self.registry.get_active()  # just set it, so it's name
        # Close all stores for safety — the new active will be re-opened lazily
        with self._lock:
            if self._watcher is not None:
                self._watcher.stop()
                self._watcher = None

        entry = self.registry.get_rag(name)
        # Pre-open the store so callers can use it immediately
        self._get_store(entry)
        self._sync_watcher(entry)
        logger.info("Active RAG is now '%s'", name)
        return entry

    @_locked
    def delete_rag(self, name: str) -> bool:
        """Delete RAG *name*, cleaning up store, watcher, and data.

        Returns True if all data was removed immediately, False if
        file cleanup was deferred.
        """
        logger.info("Deleting RAG '%s'", name)
        entry = self.registry.get_rag(name)  # validate exists

        # WS2: Refuse to delete a RAG that is currently being indexed
        if name in self._indexing_rags:
            raise RuntimeError(
                f"RAG '{name}' is currently being indexed. "
                f"Cancel indexing first before deleting."
            )

        active = self.registry.get_active()
        is_active = active is not None and active.name == name

        if is_active:
            with self._lock:
                if self._watcher is not None:
                    self._watcher.stop()
                    self._watcher = None

        # Always close the store — even for non-active RAGs it may be
        # cached from a previous switch or read operation.
        self._close_store(entry.db_path, force=True)

        # Close any FileManifest SQLite connection(s) pointing into the
        # RAG's data directory so their WAL/SHM files are released.
        from rag_kb.file_manifest import FileManifest as _FM

        _FM.close_for_path(entry.db_path)

        result = self.registry.delete_rag(name)
        logger.info("RAG '%s' deleted (immediate_cleanup=%s)", name, result)

        # If we deleted the active RAG, open the new active if any
        if is_active:
            new_active = self.registry.get_active()
            if new_active:
                self._get_store(new_active)
                self._sync_watcher(new_active)

        return result

    @_locked
    def detach_rag(self, name: str) -> RagEntry:
        """Mark *name* as detached (read-only)."""
        entry = self.registry.get_rag(name)
        entry.detached = True
        self.registry.update_rag(entry)

        # Stop watcher if this is the active RAG
        active = self.registry.get_active()
        if active and active.name == name:
            with self._lock:
                if self._watcher is not None:
                    self._watcher.stop()
                    self._watcher = None
        return entry

    @_locked
    def attach_rag(self, name: str) -> RagEntry:
        """Re-attach a previously detached RAG."""
        entry = self.registry.get_rag(name)
        entry.detached = False
        self.registry.update_rag(entry)

        # Restart watcher if this is the active RAG
        active = self.registry.get_active()
        if active and active.name == name:
            self._sync_watcher(entry)
        return entry

    @_locked
    def update_rag(self, entry: RagEntry) -> None:
        """Persist changes to a RAG entry (e.g. source folders)."""
        self.registry.update_rag(entry)

    # ======================================================================
    # Search
    # ======================================================================

    def search(
        self,
        query: str,
        n_results: int = 5,
        rag_name: str | None = None,
        min_score: float | None = None,
    ) -> list[SearchResultItem]:
        """Run a hybrid search (vector + BM25 + rerank + MMR) on the active
        (or specified) RAG.

        Returns a list of ``SearchResultItem`` ordered by relevance.

        Thread safety
        -------------
        The lock is held only for the short snapshot phase (registry lookup,
        store resolution, settings copy).  The heavy search pipeline (embed,
        vector search, BM25, rerank, MMR) runs **outside** the lock so that
        concurrent searches and indexing are not serialised.
        """
        # --- snapshot under lock -------------------------------------------
        with self._lock:
            if rag_name:
                entry = self.registry.get_rag(rag_name)
            else:
                entry = self._ensure_active()

            store = self._ensure_store(entry)
            settings = self.settings.model_copy()  # snapshot settings

        # --- search pipeline (no lock held) --------------------------------

        if store.count() == 0:
            logger.debug("Search aborted: store is empty for RAG '%s'", entry.name)
            return []

        import time as _time

        _t_search_start = _time.perf_counter()
        _t_vec = 0.0
        _t_bm25 = 0.0
        _t_rerank = 0.0
        _t_mmr = 0.0

        # -- Fetch candidates --
        fetch_k = max(n_results * 4, 20)
        logger.debug(
            "Search: query=%r n_results=%d fetch_k=%d rag=%s",
            query[:80],
            n_results,
            fetch_k,
            entry.name,
        )
        _t0 = _time.perf_counter()
        query_emb = embed_query(query, model_name=entry.embedding_model)
        vec_results = store.search(
            query_emb,
            n_results=fetch_k,
            min_score=settings.min_score_threshold,
            include_embeddings=settings.mmr_enabled,
        )
        _t_vec = (_time.perf_counter() - _t0) * 1000
        if not vec_results:
            return []

        folders = entry.source_folders

        # -- BM25 hybrid fusion --
        if settings.hybrid_search_enabled:
            _t0 = _time.perf_counter()
            try:
                doc_count = store.count()
                bm25_cache = get_bm25_cache()
                bm25_model, all_ids, all_texts, all_metas = bm25_cache.get_or_build(
                    rag_name=entry.name,
                    doc_count=doc_count,
                    fetch_fn=store.get_all_documents,
                )
                if all_texts and bm25_model is not None:
                    bm25_hits = bm25_search(
                        query, all_texts, all_ids, top_k=fetch_k, bm25_index=bm25_model
                    )
                    vec_scores = {
                        r.source_file + "::chunk_" + str(r.chunk_index): r.score
                        for r in vec_results
                    }
                    bm25_scores_map = dict(bm25_hits)
                    fused = hybrid_fuse_scores(
                        vec_scores,
                        bm25_scores_map,
                        alpha=settings.hybrid_search_alpha,
                    )

                    all_results_map = {
                        r.source_file + "::chunk_" + str(r.chunk_index): r for r in vec_results
                    }
                    id_to_idx = {doc_id: i for i, doc_id in enumerate(all_ids)}
                    for hit_id, _ in bm25_hits:
                        if hit_id not in all_results_map and hit_id in id_to_idx:
                            idx = id_to_idx[hit_id]
                            meta = all_metas[idx] if idx < len(all_metas) else {}
                            all_results_map[hit_id] = SearchResult(
                                text=all_texts[idx],
                                source_file=meta.get("source_file", ""),
                                chunk_index=int(meta.get("chunk_index", 0)),
                                score=0.0,
                                metadata=meta,
                            )
                    for key, score in fused.items():
                        if key in all_results_map:
                            all_results_map[key].score = score
                    vec_results = sorted(
                        all_results_map.values(),
                        key=lambda r: r.score,
                        reverse=True,
                    )[:fetch_k]
            except Exception as exc:
                logger.warning("BM25 hybrid search failed, falling back to vector only: %s", exc)
            _t_bm25 = (_time.perf_counter() - _t0) * 1000

        # -- Backfill embeddings for BM25-only hits when MMR is needed --
        if settings.mmr_enabled:
            missing = [r for r in vec_results if r.embedding is None]
            if missing:
                try:
                    ids_needed = [r.source_file + "::chunk_" + str(r.chunk_index) for r in missing]
                    emb_map = store.get_embeddings_by_ids(ids_needed)
                    for r, doc_id in zip(missing, ids_needed, strict=False):
                        if doc_id in emb_map:
                            r.embedding = emb_map[doc_id]
                except Exception as exc:
                    logger.warning("Failed to backfill embeddings for MMR: %s", exc)

        # -- Cross-encoder reranking --
        if settings.reranking_enabled:
            _t0 = _time.perf_counter()
            try:
                texts_for_rerank = [r.text for r in vec_results]
                scores_for_rerank = [r.score for r in vec_results]
                reranked = rerank_cross_encoder(
                    query,
                    texts_for_rerank,
                    scores_for_rerank,
                    model_name=settings.reranker_model,
                )
                reordered = [vec_results[idx] for idx, _ in reranked]
                for res, (_, new_score) in zip(reordered, reranked, strict=False):
                    res.score = new_score
                vec_results = reordered
            except Exception as exc:
                logger.warning("Cross-encoder reranking failed: %s", exc)
            _t_rerank = (_time.perf_counter() - _t0) * 1000

        # -- MMR diversity --
        if settings.mmr_enabled and len(vec_results) > n_results:
            _t0 = _time.perf_counter()
            try:
                doc_embeddings = np.array(
                    [r.embedding for r in vec_results if r.embedding is not None],
                    dtype=np.float32,
                )
                if len(doc_embeddings) == len(vec_results):
                    mmr_indices = mmr_diversify(
                        query_embedding=query_emb,
                        doc_embeddings=doc_embeddings,
                        scores=[r.score for r in vec_results],
                        lambda_mult=settings.mmr_lambda,
                        top_n=n_results,
                    )
                    final_results = [vec_results[i] for i in mmr_indices]
                else:
                    logger.warning("MMR: some embeddings missing, falling back to top-N")
                    final_results = vec_results[:n_results]
            except Exception as exc:
                logger.warning("MMR diversity failed, falling back to top-N: %s", exc)
                final_results = vec_results[:n_results]
            _t_mmr = (_time.perf_counter() - _t0) * 1000
        else:
            final_results = vec_results[:n_results]

        # -- Minimum score threshold --
        effective_min = min_score if min_score is not None else settings.min_score_threshold
        if effective_min > 0:
            final_results = [r for r in final_results if r.score >= effective_min]

        _t_total = (_time.perf_counter() - _t_search_start) * 1000

        # -- Record search metrics (best-effort) --
        try:
            from rag_kb.metrics import MetricsCollector, SearchQueryMetrics

            top_sc = max((r.score for r in final_results), default=0.0)
            min_sc = min((r.score for r in final_results), default=0.0)
            MetricsCollector.get().record_search_query(
                SearchQueryMetrics(
                    rag_name=entry.name,
                    timestamp=_time.time(),
                    query_length=len(query),
                    top_k=n_results,
                    results_returned=len(final_results),
                    total_duration_ms=round(_t_total, 2),
                    vector_search_ms=round(_t_vec, 2),
                    bm25_ms=round(_t_bm25, 2),
                    rerank_ms=round(_t_rerank, 2),
                    mmr_ms=round(_t_mmr, 2),
                    top_score=round(top_sc, 4),
                    min_score=round(min_sc, 4),
                )
            )
        except Exception:
            pass

        # -- Convert to public dataclass --
        logger.debug(
            "Search returning %d results (after min_score=%.3f filter)",
            len(final_results),
            effective_min,
        )
        return [
            SearchResultItem(
                text=r.text,
                source_file=safe_display_path(r.source_file, folders),
                score=round(r.score, 4),
                chunk_index=r.chunk_index,
                metadata=r.metadata,
            )
            for r in final_results
        ]

    # ======================================================================
    # Indexing
    # ======================================================================

    def index(
        self,
        rag_name: str | None = None,
        full: bool = False,
        workers: int | None = None,
        on_progress: Callable[[IndexingState], None] | None = None,
    ) -> IndexingState:
        """Run the indexing pipeline on the active (or specified) RAG.

        Parameters
        ----------
        rag_name : str | None
            RAG to index. Defaults to the active RAG.
        full : bool
            If True, clear all data and rebuild from scratch.
        workers : int | None
            Override number of parallel parsing workers.
        on_progress : callable | None
            Called with an ``IndexingState`` after each progress update.

        Returns
        -------
        IndexingState
            Final state including processed files, chunks, errors, duration.

        Thread safety
        -------------
        Holds the lock only for the short setup / teardown phases.
        The long-running ``Indexer.index()`` call runs **outside** the lock
        so that concurrent reads (search, status, file listing) are not
        blocked for the duration of the indexing pipeline.
        """
        # --- snapshot under lock -------------------------------------------
        with self._lock:
            entry = self.registry.get_rag(rag_name) if rag_name else self._ensure_active()
            logger.info(
                "Starting index pipeline: rag='%s' full=%s workers=%s",
                entry.name,
                full,
                workers,
            )

            if entry.detached:
                raise RuntimeError(
                    f"RAG '{entry.name}' is detached (read-only). Use attach to re-enable indexing."
                )
            if entry.is_imported and not entry.source_folders:
                raise RuntimeError("Cannot index an imported RAG with no source folders.")

            # WS2: Per-RAG indexing guard — reject concurrent index on same RAG
            if entry.name in self._indexing_rags:
                raise RuntimeError(
                    f"RAG '{entry.name}' is already being indexed. "
                    f"Wait for the current indexing to finish or cancel it first."
                )
            self._indexing_rags.add(entry.name)

            settings = self.settings.model_copy()  # always copy — safe snapshot
            if workers is not None:
                settings.indexing_workers = workers

            # WS3: Stop watcher during indexing to avoid the watcher's
            # fallback Indexer from creating a second VectorStore/FileManifest
            # that would conflict with the active indexer.
            if self._watcher is not None:
                self._watcher.stop()
                self._watcher = None

            # Evict cached store before indexing (Indexer creates its own)
            self._close_store(entry.db_path)

        # --- long-running work (no lock held) ------------------------------
        cancel_event = threading.Event()
        indexer = Indexer(
            entry,
            self.registry,
            settings,
            on_progress=on_progress,
            cancel_event=cancel_event,
        )
        with self._indexer_lock:
            self._current_indexer = indexer

        try:
            state = indexer.index(full=full)
        finally:
            with self._indexer_lock:
                self._current_indexer = None
            indexer.close()
            # Always release the per-RAG guard
            with self._lock:
                self._indexing_rags.discard(entry.name)

        logger.info(
            "Index pipeline finished: rag='%s' status=%s files=%d chunks=%d duration=%.1fs errors=%d",
            entry.name,
            state.status,
            state.processed_files,
            state.total_chunks,
            state.duration_seconds,
            len(state.errors),
        )

        # --- bookkeeping under lock ----------------------------------------
        with self._lock:
            self._last_index_state = state
            # Refresh store reference (index may have rebuilt the DB)
            self._close_store(entry.db_path)
            self._get_store(entry)

        # Invalidate BM25 cache — corpus has changed
        get_bm25_cache().invalidate()

        # WS3: Restart watcher after indexing completes
        self._sync_watcher(entry)

        return state

    def cancel_indexing(self) -> bool:
        """Request cancellation of the currently running indexing operation.

        Returns True if a running indexer was found and signalled, False otherwise.
        Cancellation is cooperative: the indexer will stop at the next checkpoint.
        """
        with self._indexer_lock:
            indexer = self._current_indexer
        if indexer is None:
            return False
        indexer.cancel()
        return True

    def check_incomplete_indexing(self, rag_name: str | None = None) -> dict | None:
        """Check whether the given (or active) RAG has an incomplete indexing.

        Returns the lock file info dict if incomplete indexing is detected,
        or None. The stale lock file is removed as a side effect.
        """
        with self._lock:
            if rag_name:
                entry = self.registry.get_rag(rag_name)
            else:
                entry = self.registry.get_active()
        if entry is None:
            return None
        return Indexer.check_incomplete_indexing(entry.db_path)

    # ======================================================================
    # Index status & file listing
    # ======================================================================

    @_locked
    def get_index_status(
        self,
        rag_name: str | None = None,
        *,
        skip_store_stats: bool = False,
    ) -> IndexStatus:
        """Return structured status for the active (or specified) RAG.

        Parameters
        ----------
        skip_store_stats : bool
            When *True*, skip the (expensive) ``store.get_stats()`` call.
            Useful when the caller only needs the live indexing progress
            and doesn't need up-to-date file/chunk counts from the store.
        """
        if rag_name:
            entry = self.registry.get_rag(rag_name)
        else:
            entry = self.registry.get_active()

        status = IndexStatus(
            active_rag=entry.name if entry else None,
            is_imported=entry.is_imported if entry else None,
        )

        if entry and not skip_store_stats:
            # Fast path: read cached stats from file_manifest.db (SQLite)
            # — no ChromaDB queries, ~0ms.
            manifest_db = os.path.join(entry.db_path, "file_manifest.db")
            cached = FileManifest.read_cached_stats(manifest_db)
            if cached is not None:
                status.total_files, status.total_chunks = cached
            else:
                # Fallback for RAGs that haven't been indexed yet
                # after this code was deployed (e.g. imported RAGs).
                store = self._get_store(entry)
                if store:
                    try:
                        stats = store.get_stats_fast()
                        status.total_files = stats.total_files
                        status.total_chunks = stats.total_chunks
                    except Exception:
                        pass

        if self._last_index_state:
            status.last_status = self._last_index_state.status
            status.last_indexed = self._last_index_state.last_indexed
            status.errors = self._last_index_state.errors

        # Check for incomplete indexing (crash recovery)
        if entry:
            incomplete = Indexer.check_incomplete_indexing(entry.db_path)
            if incomplete:
                started = incomplete.get("started_at", "unknown")
                status.warnings.append(
                    f"Previous indexing (started {started}) was interrupted. "
                    f"Run 'reindex' to recover."
                )

        with self._lock:
            status.watcher_running = self._watcher.is_running if self._watcher else False

        return status

    @_locked
    def list_indexed_files(
        self,
        rag_name: str | None = None,
        *,
        offset: int = 0,
        limit: int = 0,
        filter: str = "",
    ) -> PaginatedFileList:
        """List files in the index with their chunk counts.

        Parameters
        ----------
        rag_name : str | None
            RAG to query.  Defaults to the active RAG.
        offset : int
            0-based index of the first file to return.
        limit : int
            Maximum number of files to return.  0 = all (no limit).
        filter : str
            Case-insensitive substring filter on the display path.

        Returns
        -------
        PaginatedFileList
            Paginated file list with total count.
        """
        if rag_name:
            entry = self.registry.get_rag(rag_name)
        else:
            entry = self._ensure_active()

        store = self._ensure_store(entry)
        folders = entry.source_folders

        # Use metadata-only fetch (no texts/embeddings) — much cheaper
        file_chunk_counts = store.list_files_with_counts()

        # Build sorted display-path list
        all_files = sorted(
            (
                FileInfo(
                    file=safe_display_path(src, folders),
                    chunk_count=count,
                )
                for src, count in file_chunk_counts.items()
            ),
            key=lambda f: f.file,
        )

        # Apply filter
        if filter:
            needle = filter.lower()
            all_files = [f for f in all_files if needle in f.file.lower()]

        total = len(all_files)

        # Apply pagination
        if limit > 0:
            page = all_files[offset : offset + limit]
        else:
            page = all_files[offset:] if offset else all_files

        return PaginatedFileList(
            files=page,
            total=total,
            offset=offset,
            limit=limit,
            filter=filter,
        )

    @_locked
    def get_document_content(
        self,
        file_path: str,
        rag_name: str | None = None,
    ) -> list[ChunkInfo]:
        """Retrieve all indexed chunks from a specific document.

        Accepts display paths (as returned by search / list_indexed_files)
        or absolute paths.
        """
        if rag_name:
            entry = self.registry.get_rag(rag_name)
        else:
            entry = self._ensure_active()

        store = self._ensure_store(entry)

        # Try direct lookup (absolute path)
        items = store.get_by_source(file_path)
        if items:
            return [
                ChunkInfo(id=it["id"], text=it["text"], metadata=it.get("metadata", {}))
                for it in items
            ]

        # Reverse-resolve display path → absolute path
        folders = entry.source_folders
        stored_sources = store.get_stats().files
        norm_input = file_path.replace("\\", "/").lower()
        for stored in stored_sources:
            display = safe_display_path(stored, folders)
            if display.replace("\\", "/").lower() == norm_input:
                items = store.get_by_source(stored)
                return [
                    ChunkInfo(id=it["id"], text=it["text"], metadata=it.get("metadata", {}))
                    for it in items
                ]

        return []

    # ======================================================================
    # File change detection
    # ======================================================================

    @_locked
    def scan_file_changes(self, rag_name: str | None = None) -> FileChanges:
        """Scan source folders and compare against the manifest.

        Returns lists of new, modified and removed files.
        """
        if rag_name:
            entry = self.registry.get_rag(rag_name)
        else:
            entry = self._ensure_active()

        from rag_kb.file_manifest import FileManifest

        # Discover current source files
        source_files: list[str] = []
        exts = set(self.settings.supported_extensions)
        for folder in entry.source_folders:
            folder_path = Path(folder)
            if not folder_path.exists():
                continue
            for root, _dirs, filenames in os.walk(folder_path):
                for fname in filenames:
                    fp = Path(root) / fname
                    if fp.suffix.lower() in exts:
                        source_files.append(str(fp))

        source_set = set(source_files)

        manifest_db = os.path.join(entry.db_path, "file_manifest.db")
        if not os.path.exists(manifest_db):
            return FileChanges(new=sorted(source_files))

        manifest = FileManifest(manifest_db)
        indexed_set = manifest.all_paths()

        new_files = sorted(source_set - indexed_set)
        removed_files = sorted(indexed_set - source_set)

        modified_files = [fp for fp in sorted(source_set & indexed_set) if manifest.is_changed(fp)]

        manifest.close()

        return FileChanges(new=new_files, modified=modified_files, removed=removed_files)

    # ======================================================================
    # Index consistency verification
    # ======================================================================

    @_locked
    def verify_index_consistency(self, rag_name: str | None = None) -> dict:
        """Check whether the manifest and vector store are consistent.

        Returns a dict with:
          - ``ok`` (bool): True if everything looks good
          - ``invalidated_files`` (list[str]): manifest entries with empty hash
          - ``orphan_store_files`` (list[str]): files in store but not manifest
          - ``orphan_manifest_files`` (list[str]): files in manifest but not store
          - ``incomplete_indexing`` (bool): whether a lock file was found
        """
        if rag_name:
            entry = self.registry.get_rag(rag_name)
        else:
            entry = self._ensure_active()

        from rag_kb.file_manifest import FileManifest

        store = self._ensure_store(entry)
        manifest_db = os.path.join(entry.db_path, "file_manifest.db")

        result: dict = {
            "ok": True,
            "invalidated_files": [],
            "orphan_store_files": [],
            "orphan_manifest_files": [],
            "incomplete_indexing": False,
        }

        if not os.path.exists(manifest_db):
            return result

        manifest = FileManifest(manifest_db)

        # Check for invalidated rows (empty content_hash)
        for path in manifest.all_paths():
            rec = manifest.get_record(path)
            if rec and not rec.content_hash:
                result["invalidated_files"].append(path)

        # Compare manifest paths vs store paths
        manifest_paths = manifest.all_paths()
        store_files = set(store.list_files_with_counts().keys())

        result["orphan_store_files"] = sorted(store_files - manifest_paths)
        result["orphan_manifest_files"] = sorted(manifest_paths - store_files)

        manifest.close()

        # Check for lock file
        lock_info = Indexer.check_incomplete_indexing(entry.db_path)
        if lock_info is not None:
            result["incomplete_indexing"] = True

        if (
            result["invalidated_files"]
            or result["orphan_store_files"]
            or result["orphan_manifest_files"]
            or result["incomplete_indexing"]
        ):
            result["ok"] = False

        return result

    # ======================================================================
    # Import / Export
    # ======================================================================

    @_locked
    def export_rag(self, rag_name: str, output_path: str) -> str:
        """Export a RAG to a .rag file. Returns the absolute output path."""
        return _sharing_export(self.registry, rag_name, output_path)

    @_locked
    def import_rag(self, file_path: str, new_name: str | None = None) -> str:
        """Import a RAG from a .rag file. Returns the imported name."""
        return _sharing_import(self.registry, file_path, new_name=new_name)

    def peek_rag_file(self, file_path: str) -> dict:
        """Read manifest metadata from a .rag file without importing."""
        return peek_rag_file(file_path)

    # ======================================================================
    # Config management
    # ======================================================================

    @_locked
    def get_config(self) -> AppSettings:
        """Return the current settings."""
        return self.settings

    @_locked
    def save_config(self, settings: AppSettings | None = None) -> None:
        """Persist settings to disk."""
        if settings is not None:
            self.settings = settings
        self.settings.save()

    @_locked
    def reset_config(self) -> AppSettings:
        """Reset to default settings, save, and return them."""
        self.settings = AppSettings()
        self.settings.save()
        return self.settings

    @_locked
    def reload_config(self) -> AppSettings:
        """Reload settings from disk."""
        self.settings = AppSettings.load()
        return self.settings

    # ======================================================================
    # Watcher management
    # ======================================================================

    @property
    def watcher(self) -> FolderWatcher | None:
        with self._lock:
            return self._watcher

    @_locked
    def is_watcher_running(self) -> bool:
        with self._lock:
            return self._watcher is not None and self._watcher.is_running

    @_locked
    def start_watcher(self, rag_name: str | None = None) -> None:
        """Start the file watcher for the given (or active) RAG."""
        entry = self.registry.get_rag(rag_name) if rag_name else self.registry.get_active()
        self._sync_watcher(entry)

    @_locked
    def stop_watcher(self) -> None:
        """Stop the current file watcher if running."""
        with self._lock:
            if self._watcher is not None:
                self._watcher.stop()
                self._watcher = None

    # ======================================================================
    # Model download
    # ======================================================================

    def download_models(
        self,
        output_dir: str | Path | None = None,
        model_name: str | None = None,
    ) -> list[Path]:
        """Download ML models for offline use.

        Parameters
        ----------
        output_dir : path, optional
            Where to save. Defaults to the bundled models directory.
        model_name : str, optional
            A specific model to download. Defaults to all default models.

        Returns
        -------
        list[Path]
            Paths to the saved model directories.
        """
        from rag_kb.models import BUNDLED_MODELS_DIR, DEFAULT_MODELS, download_models

        out = Path(output_dir) if output_dir else BUNDLED_MODELS_DIR
        if model_name:
            models = [{"name": model_name, "type": "embedding", "description": "custom model"}]
        else:
            models = DEFAULT_MODELS
        return download_models(models=models, output_dir=out)

    # ======================================================================
    # Model management  (new — registry-driven)
    # ======================================================================

    def list_models(
        self,
        model_type: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return all registered models with runtime status.

        Parameters
        ----------
        model_type : str | None
            Filter by ``"embedding"`` or ``"reranker"``.  None = all.
        """
        from rag_kb.models import get_all_models_with_status

        all_models = get_all_models_with_status()
        if model_type:
            all_models = [m for m in all_models if m["type"] == model_type]
        return all_models

    def get_model_info(self, model_name: str) -> dict[str, Any] | None:
        """Return detailed info for a single model."""
        from rag_kb.models import get_model_disk_size, get_model_spec, get_model_status

        spec = get_model_spec(model_name)
        if spec is None:
            return None
        d = spec.model_dump()
        status = get_model_status(model_name)
        d["status"] = status.value
        from rag_kb.models import ModelStatus

        d["disk_size_bytes"] = (
            get_model_disk_size(model_name)
            if status
            in (
                ModelStatus.bundled,
                ModelStatus.downloaded,
            )
            else 0
        )
        return d

    def download_model(
        self,
        model_name: str,
        *,
        trust_remote_code: bool | None = None,
        progress: ProgressCallback | None = None,
    ) -> str:
        """Download a single model by name. Returns the saved path."""
        from rag_kb.models import download_model_by_name, get_model_spec

        spec = get_model_spec(model_name)

        # Auto-trust if user previously consented
        if trust_remote_code is None and spec and spec.trust_remote_code:
            trust_remote_code = model_name in self.settings.trusted_models

        if progress:
            size_mb = spec.model_size_mb if spec else 0
            progress(f"Downloading {model_name} (~{size_mb} MB)…", 0.1)

        dest = download_model_by_name(
            model_name,
            trust_remote_code=trust_remote_code,
        )

        if progress:
            progress(f"Downloaded {model_name}", 1.0)

        return str(dest)

    def delete_model(self, model_name: str) -> bool:
        """Delete a downloaded model."""
        from rag_kb.models import delete_downloaded_model

        return delete_downloaded_model(model_name)

    def trust_model(self, model_name: str) -> None:
        """Add a model to the trusted list (user consents to trust_remote_code)."""
        if model_name not in self.settings.trusted_models:
            self.settings.trusted_models.append(model_name)
            self.settings.save()
            logger.info("Model '%s' added to trusted list.", model_name)

    def untrust_model(self, model_name: str) -> None:
        """Remove a model from the trusted list."""
        if model_name in self.settings.trusted_models:
            self.settings.trusted_models.remove(model_name)
            self.settings.save()
            logger.info("Model '%s' removed from trusted list.", model_name)

    def is_model_trusted(self, model_name: str) -> bool:
        """Return True if model is in the trusted list."""
        return model_name in self.settings.trusted_models

    # ======================================================================
    # Store access (for advanced use by UI)
    # ======================================================================

    @_locked
    def get_store(self, entry: RagEntry | None = None) -> VectorStore | None:
        """Return a cached VectorStore for *entry* (or the active RAG).

        Public wrapper — callers should **not** call ``store.close()``.
        """
        entry = entry or self.registry.get_active()
        return self._get_store(entry)

    @_locked
    def close_store(self, db_path: str, *, force: bool = False) -> None:
        """Close and remove a cached store. Public wrapper."""
        self._close_store(db_path, force=force)

    # ======================================================================
    # Lifecycle
    # ======================================================================

    @property
    def last_index_state(self) -> IndexingState | None:
        with self._lock:
            return self._last_index_state

    @last_index_state.setter
    def last_index_state(self, value: IndexingState | None) -> None:
        with self._lock:
            self._last_index_state = value

    @_locked
    def shutdown(self) -> None:
        """Release all resources (stores, watcher). Call on exit."""
        logger.info("Shutting down RagKnowledgeBaseAPI…")
        # Cancel any running indexer
        with self._indexer_lock:
            if self._current_indexer is not None:
                self._current_indexer.cancel()

        with self._lock:
            if self._watcher is not None:
                self._watcher.stop()
                self._watcher = None
        for db_path in list(self._stores.keys()):
            store = self._stores.pop(db_path, None)
            if store is not None:
                store.close(force=True)
        logger.info("RagKnowledgeBaseAPI shut down.")
