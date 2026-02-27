"""Indexing pipeline — scan, parse, chunk, embed, upsert.

Performance-optimised for 10K-100K file scale:
  - **FileManifest** (SQLite + xxhash) with **batch** change detection —
    single SQL query for all paths instead of N individual lookups.
  - **Parallel parsing** via ``ProcessPoolExecutor`` for CPU-bound work
    (regex, HTML, XML) with ``ThreadPoolExecutor`` fallback.
  - **Batched embedding** — collects chunks across files before encoding
    to maximise GPU/CPU utilisation.
  - **Batch vector-store operations** — bulk deletion via ``$in`` queries,
    bulk upserts in 10K slices.
  - **Batch manifest commits** — ``batch_mark_indexed()`` writes all rows
    in a single SQLite transaction instead of N individual commits.
  - **Pipeline parallelism** — parses batch N+1 while embedding batch N.
  - **Auto-tuned workers** — defaults to ``min(cpu_count, 8)`` for an
    optimal balance of I/O concurrency and CPU core utilisation.
  - **Throttled progress notifications** — at most once per 250 ms to avoid
    UI/callback overhead drowning the pipeline for large file counts.
  - Structure-aware chunking with contextual prefixes.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, Future, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from rag_kb.chunker import TextChunk, chunk_text
from rag_kb.config import AppSettings, RagEntry, RagRegistry
from rag_kb.embedder import embed_texts
from rag_kb.file_manifest import FileManifest
from rag_kb.parsers.base import ParsedDocument
from rag_kb.parsers.registry import parse_file
from rag_kb.skip_patterns import is_skipped_dir, is_skipped_file
from rag_kb.vector_store import VectorStore

logger = logging.getLogger(__name__)

# How many files to process in a single memory batch.
# Kept deliberately small so that parsed documents (which can hold MB of
# text + images for large PDFs) don't all accumulate in RAM at once.
_FILE_BATCH_SIZE = 50

# Minimum interval between progress notifications (seconds)
_NOTIFY_INTERVAL = 0.25


class IndexingCancelledError(Exception):
    """Raised when indexing is cancelled cooperatively via cancel event."""


@dataclass
class IndexingState:
    """Observable state of the indexing process."""

    status: str = "idle"  # idle | scanning | parsing | embedding | done | error | cancelled
    progress: float = 0.0  # 0.0 – 1.0
    current_file: str = ""
    total_files: int = 0
    processed_files: int = 0
    total_chunks: int = 0
    errors: list[str] = field(default_factory=list)
    last_indexed: str = ""
    duration_seconds: float = 0.0
    # Per-phase timing (seconds) — populated during indexing
    scan_seconds: float = 0.0
    parse_seconds: float = 0.0
    chunk_seconds: float = 0.0
    embed_seconds: float = 0.0
    upsert_seconds: float = 0.0
    manifest_seconds: float = 0.0
    chunks_per_second: float = 0.0


class Indexer:
    """Orchestrates the full indexing pipeline for a single RAG."""

    def __init__(
        self,
        rag_entry: RagEntry,
        registry: RagRegistry,
        settings: AppSettings | None = None,
        on_progress: Callable[[IndexingState], None] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> None:
        self.rag = rag_entry
        self.registry = registry
        self.settings = settings or AppSettings.load()
        self._on_progress = on_progress
        self._cancel = cancel_event or threading.Event()
        self.state = IndexingState()
        self._store: VectorStore | None = None
        self._manifest: FileManifest | None = None
        self._last_notify_time: float = 0.0  # monotonic timestamp

    @property
    def store(self) -> VectorStore:
        if self._store is None:
            self._store = VectorStore(
                self.rag.db_path,
                hnsw_ef_construction=self.settings.hnsw_ef_construction,
                hnsw_m=self.settings.hnsw_m,
            )
        return self._store

    @property
    def manifest(self) -> FileManifest:
        if self._manifest is None:
            manifest_db = os.path.join(self.rag.db_path, "file_manifest.db")
            self._manifest = FileManifest(manifest_db)
        return self._manifest

    def close(self) -> None:
        """Release resources held by this Indexer.

        Closes the FileManifest SQLite connection and the VectorStore
        (ChromaDB) so that the underlying files are no longer locked.
        Must be called when the Indexer is no longer needed — especially
        important on Windows where open handles prevent directory deletion.
        """
        if self._manifest is not None:
            try:
                self._manifest.close()
            except Exception:  # noqa: BLE001
                pass
            self._manifest = None
        if self._store is not None:
            try:
                self._store.close()
            except Exception:  # noqa: BLE001
                pass
            self._store = None

    # ------------------------------------------------------------------
    # Cancellation helpers
    # ------------------------------------------------------------------

    def cancel(self) -> None:
        """Request cancellation of the current indexing operation."""
        self._cancel.set()

    def _check_cancelled(self) -> None:
        """Raise IndexingCancelledError if cancellation was requested."""
        if self._cancel.is_set():
            self.state.status = "cancelling"
            self._notify(force=True)
            raise IndexingCancelledError("Indexing cancelled by user")

    # ------------------------------------------------------------------
    # Lock file helpers (crash recovery)
    # ------------------------------------------------------------------

    def _lock_file_path(self) -> str:
        return os.path.join(self.rag.db_path, ".indexing_lock")

    def _write_lock_file(self, full: bool) -> None:
        """Write a lock file indicating indexing is in progress."""
        lock_path = self._lock_file_path()
        try:
            os.makedirs(os.path.dirname(lock_path), exist_ok=True)
            with open(lock_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "started_at": datetime.now(timezone.utc).isoformat(),
                        "pid": os.getpid(),
                        "full": full,
                    },
                    f,
                )
        except OSError as exc:
            logger.warning("Could not write indexing lock file: %s", exc)

    def _remove_lock_file(self) -> None:
        """Remove the lock file on successful completion."""
        lock_path = self._lock_file_path()
        try:
            if os.path.exists(lock_path):
                os.remove(lock_path)
        except OSError as exc:
            logger.warning("Could not remove indexing lock file: %s", exc)

    @staticmethod
    def check_incomplete_indexing(db_path: str) -> dict | None:
        """Check if a previous indexing was interrupted.

        Returns the lock file contents as a dict, or None.
        Removes the stale lock file if found.
        """
        lock_path = os.path.join(db_path, ".indexing_lock")
        if not os.path.exists(lock_path):
            return None
        try:
            with open(lock_path, "r", encoding="utf-8") as f:
                info = json.load(f)
            os.remove(lock_path)
            return info
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Could not read indexing lock file: %s", exc)
            try:
                os.remove(lock_path)
            except OSError:
                pass
            return {"started_at": "unknown", "pid": 0, "full": False}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def index(self, full: bool = False) -> IndexingState:
        """Run the indexing pipeline.

        If *full* is True, clears all existing data first (full rebuild).
        Otherwise performs incremental indexing (only changed/new files).

        Raises RuntimeError if the RAG is detached (read-only).
        """
        if self.rag.detached:
            raise RuntimeError(
                f"RAG '{self.rag.name}' is detached (read-only). "
                "Use 'rag-kb attach' to re-enable indexing."
            )

        logger.info(
            "Index start: rag='%s' full=%s model=%s folders=%s",
            self.rag.name, full, self.rag.embedding_model, self.rag.source_folders,
        )
        t0 = time.monotonic()
        t_start_wall = time.time()
        self.state = IndexingState(status="scanning")
        self._notify(force=True)

        # Write lock file for crash recovery
        self._write_lock_file(full)

        try:
            if full:
                logger.info("Full rebuild requested — clearing store and manifest")
                self.store.clear()
                self.manifest.clear()

            self._check_cancelled()

            # 1. Discover files
            t_scan = time.monotonic()
            file_paths = self._discover_files()
            self.state.scan_seconds = round(time.monotonic() - t_scan, 4)
            self.state.total_files = len(file_paths)
            logger.info("Discovered %d source files", len(file_paths))
            self._notify(force=True)

            self._check_cancelled()

            if not file_paths:
                self.state.status = "done"
                self.state.last_indexed = datetime.now(timezone.utc).isoformat()
                self._notify(force=True)
                self._remove_lock_file()
                return self.state

            # 2. Determine which files need (re)indexing using manifest
            files_to_index = self._filter_changed_manifest(file_paths, full)
            logger.info("Files needing (re)indexing: %d / %d", len(files_to_index), len(file_paths))

            # 3. Remove deleted files (batch operations)
            current_paths = {str(p) for p in file_paths}
            known_paths = set(self.manifest.all_paths())
            missing = known_paths - current_paths

            if missing and not current_paths:
                logger.warning(
                    "No source files found but %d files exist in index — "
                    "skipping bulk deletion (source folders may have been removed)",
                    len(missing),
                )
            elif missing:
                missing_list = list(missing)
                self.store.batch_delete_by_sources(missing_list)
                self.manifest.batch_remove(missing_list)
                logger.info("Removed %d deleted file(s) from index", len(missing))

            if not files_to_index and not full:
                self.state.status = "done"
                self.state.progress = 1.0
                self.state.processed_files = self.state.total_files
                self.state.total_chunks = self.store.count()
                self.state.last_indexed = datetime.now(timezone.utc).isoformat()
                self._notify(force=True)
                self._update_registry_stats()
                self._remove_lock_file()
                return self.state

            # 4. Process files in memory-efficient batches
            self.state.status = "parsing"
            self.state.total_files = len(files_to_index)
            self.state.processed_files = 0
            self._notify(force=True)

            workers = self._effective_workers()
            logger.info(
                "Processing %d file(s) in batches of %d (workers=%d)",
                len(files_to_index), _FILE_BATCH_SIZE, workers,
            )

            # ── Pipeline: parse batch N+1 while embedding batch N ──
            # This hides parsing latency behind GPU embedding time
            # (or vice-versa), roughly halving wall-clock time for
            # large workloads.
            parse_pool = ThreadPoolExecutor(max_workers=workers)
            embed_pool = ThreadPoolExecutor(max_workers=1)
            try:
                pending_embed: Future | None = None
                batch_ranges = list(
                    range(0, len(files_to_index), _FILE_BATCH_SIZE)
                )

                for batch_idx, batch_start in enumerate(batch_ranges):
                    self._check_cancelled()
                    batch_files = files_to_index[
                        batch_start : batch_start + _FILE_BATCH_SIZE
                    ]
                    batch_num = batch_idx + 1
                    logger.info(
                        "Processing batch %d/%d (%d files)",
                        batch_num, len(batch_ranges), len(batch_files),
                    )

                    # Phase 1: Parse
                    parsed = self._parse_files(batch_files, workers, parse_pool)

                    # Wait for the previous embed+upsert to finish before
                    # starting the next one (GPU is kept busy continuously).
                    if pending_embed is not None:
                        pending_embed.result()  # raises on error

                    self._check_cancelled()

                    if parsed:
                        # Phase 2-6: embed + upsert (runs in background)
                        pending_embed = embed_pool.submit(
                            self._embed_and_upsert, parsed,
                        )
                    else:
                        pending_embed = None

                # Drain the last embed future
                if pending_embed is not None:
                    pending_embed.result()
            finally:
                parse_pool.shutdown(wait=False)
                embed_pool.shutdown(wait=False)

            # Free memory accumulated during indexing
            self._release_indexing_memory()

            self.state.status = "done"
            self.state.progress = 1.0
            self.state.processed_files = len(files_to_index)
            self.state.total_chunks = self.store.count()
            self.state.last_indexed = datetime.now(timezone.utc).isoformat()
            self.state.duration_seconds = round(time.monotonic() - t0, 2)
            if self.state.duration_seconds > 0 and self.state.total_chunks > 0:
                self.state.chunks_per_second = round(
                    self.state.total_chunks / self.state.duration_seconds, 1
                )
            self._notify(force=True)
            self._update_registry_stats()
            self._remove_lock_file()

            # Record metrics
            self._record_indexing_metrics(t_start_wall, full)

            logger.info(
                "Index done: %d files, %d chunks, %.1fs, %d error(s)",
                self.state.processed_files, self.state.total_chunks,
                self.state.duration_seconds, len(self.state.errors),
            )
            return self.state

        except IndexingCancelledError:
            self.state.status = "cancelled"
            self.state.duration_seconds = round(time.monotonic() - t0, 2)
            self._notify(force=True)
            self._update_registry_stats()
            self._remove_lock_file()
            logger.info("Indexing cancelled after %.1fs", self.state.duration_seconds)
            return self.state

        except Exception as exc:
            self.state.status = "error"
            self.state.errors.append(str(exc))
            self._notify(force=True)
            # Lock file intentionally NOT removed on error —
            # crash recovery will detect it on next startup.
            raise

    def index_single_file_by_path(self, file_path: str) -> None:
        """Index (or re-index) a single file by its path.

        If the RAG is detached the call is silently skipped.
        If the file no longer exists the call is silently skipped.
        """
        if self.rag.detached:
            return
        fp = Path(file_path)
        if fp.exists():
            self._index_single_file(fp)
        else:
            logger.info("File not found, keeping existing chunks: %s", fp)
        self._update_registry_stats()

    def remove_file(self, file_path: str) -> None:
        """Remove all chunks for a file from the index."""
        self.store.delete_by_source(file_path)
        self.manifest.remove(file_path)
        self._update_registry_stats()

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------

    def _process_file_batch(self, file_paths: list[Path], workers: int) -> None:
        """Parse files in parallel, then batch-embed and upsert.

        Optimisations vs. the original implementation:
          - Batch manifest invalidation (1 SQL statement instead of N)
          - Batch vector-store deletion (``$in`` queries)
          - Batch manifest commit (1 transaction instead of N)
          - Throttled progress notifications
          - Embeddings kept as numpy throughout (no list↔numpy roundtrip)
        """
        parsed = self._parse_files(file_paths, workers)
        if parsed:
            self._embed_and_upsert(parsed)

    def _parse_files(
        self,
        file_paths: list[Path],
        workers: int,
        pool: ThreadPoolExecutor | None = None,
    ) -> list[tuple[Path, ParsedDocument]]:
        """Phase 1: parallel parsing.  Returns (filepath, doc) pairs.

        If *pool* is provided it is reused (pipeline mode); otherwise a
        fresh pool is created and shut down when done.
        """
        # Phase 1: Parallel parsing
        logger.debug("Phase 1: Parsing %d files (workers=%d)", len(file_paths), workers)
        t_parse = time.monotonic()
        parsed_results: list[tuple[Path, ParsedDocument]] = []

        def _parse_one(fp: Path) -> tuple[Path, ParsedDocument | None]:
            try:
                doc = parse_file(fp)
                return (fp, doc)
            except Exception as exc:
                msg = f"Error parsing {fp}: {exc}"
                logger.warning(msg)
                self.state.errors.append(msg)
                return (fp, None)

        own_pool = pool is None
        if own_pool:
            pool = ThreadPoolExecutor(max_workers=workers)

        try:
            futures = {pool.submit(_parse_one, fp): fp for fp in file_paths}
            for future in as_completed(futures):
                self._check_cancelled()
                fp, doc = future.result()
                self.state.current_file = str(fp)
                self.state.processed_files += 1
                if self.state.total_files:
                    self.state.progress = min(
                        self.state.processed_files / self.state.total_files, 0.99
                    )
                self._notify()  # throttled
                if doc is not None and not doc.is_empty:
                    parsed_results.append((fp, doc))
        finally:
            if own_pool:
                pool.shutdown(wait=False)

        self.state.parse_seconds += round(time.monotonic() - t_parse, 4)
        return parsed_results

    def _embed_and_upsert(
        self,
        parsed_results: list[tuple[Path, ParsedDocument]],
    ) -> None:
        """Phases 2-6: chunk, embed, upsert, update manifest."""
        if not parsed_results:
            return

        self._check_cancelled()

        # Phase 2: Chunk all parsed documents
        n_parsed = len(parsed_results)
        # Collect source paths *before* we release the ParsedDocuments.
        sources_in_batch = list({str(fp) for fp, _ in parsed_results})

        logger.debug("Phase 2: Chunking %d parsed documents", n_parsed)
        t_chunk = time.monotonic()
        all_chunks: list[TextChunk] = []

        for i, (fp, doc) in enumerate(parsed_results):
            mtime = datetime.fromtimestamp(fp.stat().st_mtime, tz=timezone.utc).isoformat()
            chunks = chunk_text(
                text=doc.text,
                source_file=str(fp),
                chunk_size=self.settings.chunk_size,
                chunk_overlap=self.settings.chunk_overlap,
                metadata={**doc.metadata, "file_modified_at": mtime},
                document_title=doc.title,
                format_hint=doc.format_hint,
            )
            all_chunks.extend(chunks)
            # Release the (potentially large) parsed text immediately
            # so it doesn't linger while we embed/upsert.
            parsed_results[i] = (fp, None)  # type: ignore[assignment]

        del parsed_results  # let GC reclaim ParsedDocument objects
        self.state.chunk_seconds += round(time.monotonic() - t_chunk, 4)

        if not all_chunks:
            logger.debug("Chunking produced 0 chunks — skipping embed/upsert")
            return

        logger.debug("Phase 2 produced %d chunks from %d files", len(all_chunks), n_parsed)

        # Phase 3: Batch-invalidate manifest + batch-delete old chunks
        self.manifest.batch_invalidate(sources_in_batch)
        self.store.batch_delete_by_sources(sources_in_batch)

        try:
            self._check_cancelled()

            # Phase 4: Batch embed all chunks (numpy throughout)
            self.state.status = "embedding"
            self._notify(force=True)

            embed_batch = self.settings.embedding_batch_size
            logger.debug(
                "Phase 4: Embedding %d chunks (model=%s, batch=%d)",
                len(all_chunks), self.rag.embedding_model, embed_batch,
            )

            texts = [c.text for c in all_chunks]

            # Sub-batch for cancellation checks; keep numpy arrays
            import numpy as _np
            t_embed = time.monotonic()
            emb_parts: list[_np.ndarray] = []
            for sub_start in range(0, len(texts), embed_batch):
                self._check_cancelled()
                sub_texts = texts[sub_start : sub_start + embed_batch]
                sub_emb = embed_texts(
                    sub_texts,
                    model_name=self.rag.embedding_model,
                    batch_size=embed_batch,
                    as_numpy=True,
                )
                emb_parts.append(sub_emb)

            embeddings = _np.vstack(emb_parts) if len(emb_parts) > 1 else emb_parts[0]
            self.state.embed_seconds += round(time.monotonic() - t_embed, 4)

            self._check_cancelled()

            # Phase 5: Upsert (numpy embeddings passed directly)
            t_upsert = time.monotonic()
            ids = [c.chunk_id for c in all_chunks]
            metadatas = [
                {
                    "source_file": c.source_file,
                    "chunk_index": str(c.chunk_index),
                    "start_char": str(c.start_char),
                    "end_char": str(c.end_char),
                    "file_modified_at": c.metadata.get("file_modified_at", ""),
                    **{k: v for k, v in c.metadata.items() if k != "file_modified_at"},
                }
                for c in all_chunks
            ]
            self.store.add_documents(ids=ids, texts=texts, embeddings=embeddings, metadatas=metadatas)
            self.state.upsert_seconds += round(time.monotonic() - t_upsert, 4)
            logger.debug("Phase 5: Upserted %d chunks into vector store", len(ids))

            # Phase 6: Batch-update manifest (single transaction)
            t_manifest = time.monotonic()
            file_chunk_counts: dict[str, int] = {}
            for c in all_chunks:
                file_chunk_counts[c.source_file] = file_chunk_counts.get(c.source_file, 0) + 1

            manifest_records = [
                (src, file_chunk_counts.get(src, 0))
                for src in sources_in_batch
            ]
            self.manifest.batch_mark_indexed(manifest_records)
            self.state.manifest_seconds += round(time.monotonic() - t_manifest, 4)

            logger.info(
                "Batch indexed %d files → %d chunks",
                n_parsed,
                len(all_chunks),
            )

        except IndexingCancelledError:
            logger.info(
                "Batch cancelled — %d files will be re-indexed next run",
                len(sources_in_batch),
            )
            raise

    # ------------------------------------------------------------------
    # Single-file indexing (for watcher / manual re-index)
    # ------------------------------------------------------------------

    def _index_single_file(self, fp: Path) -> None:
        """Parse → chunk → embed → upsert one file."""
        source = str(fp)

        # Remove old chunks
        self.store.delete_by_source(source)

        # Parse
        parsed = parse_file(fp)
        if parsed is None or parsed.is_empty:
            return

        # Chunk
        mtime = datetime.fromtimestamp(fp.stat().st_mtime, tz=timezone.utc).isoformat()
        chunks = chunk_text(
            text=parsed.text,
            source_file=source,
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            metadata={**parsed.metadata, "file_modified_at": mtime},
            document_title=parsed.title,
            format_hint=parsed.format_hint,
        )
        if not chunks:
            return

        # Embed
        texts = [c.text for c in chunks]
        embeddings = embed_texts(
            texts,
            model_name=self.rag.embedding_model,
            batch_size=self.settings.embedding_batch_size,
            as_numpy=False,
        )

        # Upsert
        ids = [c.chunk_id for c in chunks]
        metadatas = [
            {
                "source_file": c.source_file,
                "chunk_index": str(c.chunk_index),
                "start_char": str(c.start_char),
                "end_char": str(c.end_char),
                "file_modified_at": c.metadata.get("file_modified_at", ""),
                **{k: v for k, v in c.metadata.items() if k != "file_modified_at"},
            }
            for c in chunks
        ]
        self.store.add_documents(ids=ids, texts=texts, embeddings=embeddings, metadatas=metadatas)
        self.manifest.mark_indexed(source, len(chunks))
        logger.info("Indexed %s → %d chunks", fp.name, len(chunks))

    # ------------------------------------------------------------------
    # File discovery and change detection
    # ------------------------------------------------------------------

    def _discover_files(self) -> list[Path]:
        """Walk source folders and collect supported files.

        Uses ``os.scandir`` internally (via ``os.walk``) and defers
        ``Path`` construction to minimise overhead for large trees.

        Directories matching :data:`skip_patterns.SKIP_DIRS` are pruned
        in-place so ``os.walk`` never descends into them.  Individual
        files are tested against :func:`skip_patterns.is_skipped_file`.
        """
        files: list[str] = []
        exts = set(self.settings.supported_extensions)
        for folder in self.rag.source_folders:
            if not os.path.isdir(folder):
                logger.warning("Source folder does not exist: %s", folder)
                continue
            for root, dirs, filenames in os.walk(folder):
                # Prune skipped directories in-place so os.walk won't descend
                dirs[:] = [d for d in dirs if not is_skipped_dir(d)]
                for fname in filenames:
                    if is_skipped_file(fname):
                        continue
                    # Check extension on raw string to avoid Path overhead
                    dot_idx = fname.rfind(".")
                    if dot_idx != -1 and fname[dot_idx:].lower() in exts:
                        files.append(os.path.join(root, fname))
        files.sort()
        return [Path(f) for f in files]

    def _filter_changed_manifest(
        self, file_paths: list[Path], full: bool
    ) -> list[Path]:
        """Use the SQLite manifest for fast change detection.

        Uses ``batch_filter_changed()`` for a single SQL query instead of
        N individual lookups — critical for 100K-file scale.
        """
        if full:
            return list(file_paths)

        str_paths = [str(fp) for fp in file_paths]
        changed_strs = set(self.manifest.batch_filter_changed(str_paths))
        return [fp for fp in file_paths if str(fp) in changed_strs]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _effective_workers(self) -> int:
        """Return the number of parallel parsing workers to use.

        If the user set ``indexing_workers`` in the config we respect it.
        Otherwise auto-tune to ``min(cpu_count, 8)`` — enough to saturate
        I/O without overwhelming the OS scheduler or GPU contention.
        """
        configured = self.settings.indexing_workers
        if configured and configured > 0:
            return configured
        try:
            cpus = os.cpu_count() or 4
        except Exception:
            cpus = 4
        return min(cpus, 8)

    def _update_registry_stats(self) -> None:
        stats = self.store.get_stats()
        self.rag.file_count = stats.total_files
        self.rag.chunk_count = stats.total_chunks
        self.registry.update_rag(self.rag)

    def _record_indexing_metrics(self, started_at_wall: float, full: bool) -> None:
        """Persist indexing run metrics to the MetricsCollector (best-effort)."""
        try:
            from rag_kb.metrics import IndexingRunMetrics, MetricsCollector

            mc = MetricsCollector.get()
            s = self.state
            mc.record_indexing_run(IndexingRunMetrics(
                rag_name=self.rag.name,
                started_at=started_at_wall,
                duration_seconds=s.duration_seconds,
                total_files=s.total_files,
                processed_files=s.processed_files,
                skipped_files=s.total_files - s.processed_files,
                total_chunks=s.total_chunks,
                error_count=len(s.errors),
                status=s.status,
                scan_seconds=s.scan_seconds,
                parse_seconds=s.parse_seconds,
                chunk_seconds=s.chunk_seconds,
                embed_seconds=s.embed_seconds,
                upsert_seconds=s.upsert_seconds,
                manifest_seconds=s.manifest_seconds,
                chunks_per_second=s.chunks_per_second,
                files_per_second=(
                    round(s.processed_files / s.duration_seconds, 1)
                    if s.duration_seconds > 0 else 0.0
                ),
                is_full_reindex=full,
            ))
        except Exception:
            logger.debug("Failed to record indexing metrics", exc_info=True)

    @staticmethod
    def _release_indexing_memory() -> None:
        """Best-effort memory reclamation after a full indexing run.

        Releases:
        - Unreachable Python objects (cyclic references from PIL, numpy, etc.)
        - PyTorch CUDA tensor cache (prevents stale GPU→CPU spill-over)
        - Surya OCR model weights (reconstituted lazily on next OCR call)
        """
        import gc
        gc.collect()

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

        # Release the Surya OCR singletons — they hold ~1.5 GB of model
        # weights in RAM (+ VRAM) and are only needed during parsing.
        # They will be re-initialised lazily if OCR is needed again.
        try:
            from rag_kb.parsers import image_parser as _ip
            with _ip._surya_lock:
                if _ip._surya_rec_predictor is not None:
                    del _ip._surya_rec_predictor
                    del _ip._surya_det_predictor
                    _ip._surya_rec_predictor = None
                    _ip._surya_det_predictor = None
                    logger.info("Released Surya OCR models to free RAM.")
        except Exception:  # noqa: BLE001
            pass

        gc.collect()

    def _notify(self, *, force: bool = False) -> None:
        """Fire the progress callback, throttled to avoid overhead.

        At 100K files the per-file notification overhead was significant.
        This throttles to at most once per ``_NOTIFY_INTERVAL`` seconds
        unless *force* is True (used for status transitions).
        """
        if not self._on_progress:
            return
        now = time.monotonic()
        if not force and (now - self._last_notify_time) < _NOTIFY_INTERVAL:
            return
        self._last_notify_time = now
        self._on_progress(self.state)
