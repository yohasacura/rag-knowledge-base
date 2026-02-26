"""File system watcher for automatic re-indexing using watchdog (Apache 2.0)."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

if TYPE_CHECKING:
    from rag_kb.config import AppSettings, RagEntry, RagRegistry

logger = logging.getLogger(__name__)

# Debounce window — wait this many seconds after last event before re-indexing
_DEBOUNCE_SECONDS = 2.0


class _RagEventHandler(FileSystemEventHandler):
    """Handles file events and triggers re-indexing with debouncing."""

    def __init__(
        self,
        rag_entry: "RagEntry",
        registry: "RagRegistry",
        settings: "AppSettings",
        supported_extensions: set[str],
    ) -> None:
        super().__init__()
        self._rag = rag_entry
        self._registry = registry
        self._settings = settings
        self._exts = supported_extensions
        self._pending: dict[str, float] = {}  # path → timestamp
        self._lock = threading.Lock()
        self._timer: threading.Timer | None = None

    def _is_relevant(self, path: str) -> bool:
        return Path(path).suffix.lower() in self._exts

    def on_created(self, event: FileSystemEvent) -> None:
        if not event.is_directory and self._is_relevant(event.src_path):
            self._schedule(event.src_path)

    def on_modified(self, event: FileSystemEvent) -> None:
        if not event.is_directory and self._is_relevant(event.src_path):
            self._schedule(event.src_path)

    def on_deleted(self, event: FileSystemEvent) -> None:
        if not event.is_directory and self._is_relevant(event.src_path):
            self._schedule(event.src_path)

    def on_moved(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            if hasattr(event, "src_path") and self._is_relevant(event.src_path):
                self._schedule(event.src_path)
            if hasattr(event, "dest_path") and self._is_relevant(event.dest_path):
                self._schedule(event.dest_path)

    def _schedule(self, path: str) -> None:
        with self._lock:
            self._pending[path] = time.monotonic()
            if self._timer:
                self._timer.cancel()
            self._timer = threading.Timer(_DEBOUNCE_SECONDS, self._flush)
            self._timer.daemon = True
            self._timer.start()

    def _flush(self) -> None:
        with self._lock:
            paths = list(self._pending.keys())
            self._pending.clear()

        if not paths:
            return

        logger.info("File changes detected, re-indexing %d file(s)…", len(paths))
        try:
            from rag_kb.indexer import Indexer

            indexer = Indexer(self._rag, self._registry, self._settings)
            for p in paths:
                try:
                    indexer.index_single_file_by_path(p)
                except Exception as exc:
                    logger.warning("Watcher: error re-indexing %s: %s", p, exc)
        except Exception as exc:
            logger.error("Watcher flush error: %s", exc)


class FolderWatcher:
    """Watches source folders of a RAG entry for file changes."""

    def __init__(
        self,
        rag_entry: "RagEntry",
        registry: "RagRegistry",
        settings: "AppSettings",
    ) -> None:
        self._rag = rag_entry
        self._registry = registry
        self._settings = settings
        self._observer: Observer | None = None
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    def start(self) -> None:
        if self._running:
            return
        if self._rag.detached:
            logger.info("RAG '%s' is detached — skipping file watcher", self._rag.name)
            return
        if not self._rag.source_folders:
            logger.info("No source folders to watch for RAG '%s'", self._rag.name)
            return

        exts = set(self._settings.supported_extensions)
        handler = _RagEventHandler(self._rag, self._registry, self._settings, exts)

        self._observer = Observer()
        for folder in self._rag.source_folders:
            fp = Path(folder)
            if fp.exists() and fp.is_dir():
                self._observer.schedule(handler, str(fp), recursive=True)
                logger.info("Watching %s", fp)
            else:
                logger.warning("Folder does not exist, skipping watch: %s", folder)

        self._observer.daemon = True
        self._observer.start()
        self._running = True
        logger.info("File watcher started for RAG '%s'", self._rag.name)

    def stop(self) -> None:
        if self._observer and self._running:
            self._observer.stop()
            self._observer.join(timeout=5)
            self._observer = None
            self._running = False
            logger.info("File watcher stopped")
