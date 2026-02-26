"""Indexing pipeline — scan, parse, chunk, embed, upsert.

Improvements for 5-10 GB scale:
  - **FileManifest** (SQLite + xxhash) for O(1) change detection instead
    of per-file ChromaDB queries.
  - **Parallel parsing** via ``concurrent.futures.ThreadPoolExecutor``.
  - **Batched embedding** — collects chunks across files before encoding
    to maximise GPU/CPU utilisation.
  - **Memory-efficient batching** — processes files in configurable
    groups (default 500) to limit peak memory.
  - Structure-aware chunking with contextual prefixes.
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from rag_kb.vector_store import VectorStore

logger = logging.getLogger(__name__)

# How many files to process in a single memory batch
_FILE_BATCH_SIZE = 500


@dataclass
class IndexingState:
    """Observable state of the indexing process."""

    status: str = "idle"  # idle | scanning | parsing | embedding | done | error
    progress: float = 0.0  # 0.0 – 1.0
    current_file: str = ""
    total_files: int = 0
    processed_files: int = 0
    total_chunks: int = 0
    errors: list[str] = field(default_factory=list)
    last_indexed: str = ""
    duration_seconds: float = 0.0


class Indexer:
    """Orchestrates the full indexing pipeline for a single RAG."""

    def __init__(
        self,
        rag_entry: RagEntry,
        registry: RagRegistry,
        settings: AppSettings | None = None,
        on_progress: Callable[[IndexingState], None] | None = None,
    ) -> None:
        self.rag = rag_entry
        self.registry = registry
        self.settings = settings or AppSettings.load()
        self._on_progress = on_progress
        self.state = IndexingState()
        self._store: VectorStore | None = None
        self._manifest: FileManifest | None = None

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

        t0 = time.monotonic()
        self.state = IndexingState(status="scanning")
        self._notify()

        try:
            if full:
                self.store.clear()
                self.manifest.clear()

            # 1. Discover files
            file_paths = self._discover_files()
            self.state.total_files = len(file_paths)
            self._notify()

            if not file_paths:
                self.state.status = "done"
                self.state.last_indexed = datetime.now(timezone.utc).isoformat()
                self._notify()
                return self.state

            # 2. Determine which files need (re)indexing using manifest
            files_to_index = self._filter_changed_manifest(file_paths, full)

            # 3. Remove deleted files
            current_paths = {str(p) for p in file_paths}
            known_paths = set(self.manifest.all_paths())
            missing = known_paths - current_paths

            if missing and not current_paths:
                logger.warning(
                    "No source files found but %d files exist in index — "
                    "skipping bulk deletion (source folders may have been removed)",
                    len(missing),
                )
            else:
                for src in missing:
                    self.store.delete_by_source(src)
                    self.manifest.remove(src)
                    logger.info("Removed deleted file from index: %s", src)

            if not files_to_index and not full:
                self.state.status = "done"
                self.state.progress = 1.0
                self.state.processed_files = self.state.total_files
                self.state.total_chunks = self.store.count()
                self.state.last_indexed = datetime.now(timezone.utc).isoformat()
                self._notify()
                self._update_registry_stats()
                return self.state

            # 4. Process files in memory-efficient batches
            self.state.status = "parsing"
            self.state.total_files = len(files_to_index)
            self._notify()

            workers = self.settings.indexing_workers
            global_processed = 0

            for batch_start in range(0, len(files_to_index), _FILE_BATCH_SIZE):
                batch_files = files_to_index[batch_start : batch_start + _FILE_BATCH_SIZE]
                self._process_file_batch(batch_files, workers)
                global_processed += len(batch_files)
                self.state.processed_files = global_processed
                self.state.progress = global_processed / len(files_to_index)
                self._notify()

            self.state.status = "done"
            self.state.progress = 1.0
            self.state.processed_files = len(files_to_index)
            self.state.total_chunks = self.store.count()
            self.state.last_indexed = datetime.now(timezone.utc).isoformat()
            self.state.duration_seconds = round(time.monotonic() - t0, 2)
            self._notify()
            self._update_registry_stats()
            return self.state

        except Exception as exc:
            self.state.status = "error"
            self.state.errors.append(str(exc))
            self._notify()
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
        """Parse files in parallel, then batch-embed and upsert."""
        # Phase 1: Parallel parsing
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

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_parse_one, fp): fp for fp in file_paths}
            for future in as_completed(futures):
                fp, doc = future.result()
                self.state.current_file = str(fp)
                self._notify()
                if doc is not None and not doc.is_empty:
                    parsed_results.append((fp, doc))

        if not parsed_results:
            return

        # Phase 2: Chunk all parsed documents
        all_chunks: list[TextChunk] = []
        chunk_to_file: list[Path] = []  # track which file each chunk belongs to

        for fp, doc in parsed_results:
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
            chunk_to_file.extend([fp] * len(chunks))

        if not all_chunks:
            return

        # Phase 3: Remove old chunks for these files before upserting
        sources_in_batch = {str(fp) for fp, _ in parsed_results}
        for src in sources_in_batch:
            self.store.delete_by_source(src)

        # Phase 4: Batch embed all chunks at once
        self.state.status = "embedding"
        self._notify()

        texts = [c.text for c in all_chunks]
        embeddings = embed_texts(
            texts,
            model_name=self.rag.embedding_model,
            batch_size=self.settings.embedding_batch_size,
            as_numpy=False,  # ChromaDB needs list-of-lists
        )

        # Phase 5: Upsert
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

        # Phase 6: Update manifest for all successfully indexed files
        file_chunk_counts: dict[str, int] = {}
        for c in all_chunks:
            file_chunk_counts[c.source_file] = file_chunk_counts.get(c.source_file, 0) + 1

        for fp, _ in parsed_results:
            source = str(fp)
            n_chunks = file_chunk_counts.get(source, 0)
            self.manifest.mark_indexed(source, n_chunks)

        logger.info(
            "Batch indexed %d files → %d chunks",
            len(parsed_results),
            len(all_chunks),
        )

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
        """Walk source folders and collect supported files."""
        files: list[Path] = []
        exts = set(self.settings.supported_extensions)
        for folder in self.rag.source_folders:
            folder_path = Path(folder)
            if not folder_path.exists():
                logger.warning("Source folder does not exist: %s", folder)
                continue
            for root, _dirs, filenames in os.walk(folder_path):
                for fname in filenames:
                    fp = Path(root) / fname
                    if fp.suffix.lower() in exts:
                        files.append(fp)
        return sorted(files)

    def _filter_changed_manifest(
        self, file_paths: list[Path], full: bool
    ) -> list[Path]:
        """Use the SQLite manifest for fast change detection.

        Falls back to per-file mtime check → content hash if needed.
        """
        if full:
            return list(file_paths)

        changed: list[Path] = []
        for fp in file_paths:
            if self.manifest.is_changed(str(fp)):
                changed.append(fp)
        return changed

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _update_registry_stats(self) -> None:
        stats = self.store.get_stats()
        self.rag.file_count = stats.total_files
        self.rag.chunk_count = stats.total_chunks
        self.registry.update_rag(self.rag)

    def _notify(self) -> None:
        if self._on_progress:
            self._on_progress(self.state)
