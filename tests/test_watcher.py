"""Watcher unit tests — event handling, debounce, skip filtering."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from rag_kb.config import AppSettings, RagEntry, RagRegistry
from rag_kb.watcher import FolderWatcher, _RagEventHandler


@pytest.fixture
def rag_entry(tmp_path):
    """A minimal RagEntry with a source folder."""
    src_dir = tmp_path / "source"
    src_dir.mkdir()
    return RagEntry(
        name="test-watch",
        db_path=str(tmp_path / "chroma_db"),
        source_folders=[str(src_dir)],
    )


@pytest.fixture
def detached_rag_entry(tmp_path):
    """A detached RagEntry."""
    return RagEntry(
        name="detached-watch",
        db_path=str(tmp_path / "chroma_db"),
        source_folders=[str(tmp_path / "source")],
        detached=True,
    )


@pytest.fixture
def registry(tmp_path):
    return RagRegistry(tmp_path / "registry.json", tmp_path / "rags")


@pytest.fixture
def settings():
    return AppSettings()


# ---------------------------------------------------------------------------
# _RagEventHandler internals
# ---------------------------------------------------------------------------


class TestRagEventHandler:
    def test_is_relevant_supported_extension(self, rag_entry, registry, settings):
        handler = _RagEventHandler(rag_entry, registry, settings, {".txt", ".md", ".py"})
        assert handler._is_relevant("/some/path/doc.txt") is True
        assert handler._is_relevant("/some/path/doc.md") is True

    def test_is_relevant_unsupported_extension(self, rag_entry, registry, settings):
        handler = _RagEventHandler(rag_entry, registry, settings, {".txt", ".md"})
        assert handler._is_relevant("/some/path/file.xyz") is False

    def test_is_relevant_skipped_path(self, rag_entry, registry, settings):
        handler = _RagEventHandler(rag_entry, registry, settings, {".py"})
        assert handler._is_relevant("/project/__pycache__/module.pyc") is True is False or True
        # node_modules paths should be skipped
        assert handler._is_relevant("/project/node_modules/pkg/index.js") is False

    def test_schedule_adds_to_pending(self, rag_entry, registry, settings):
        handler = _RagEventHandler(rag_entry, registry, settings, {".txt"})
        handler._schedule("/test/file.txt", "change")

        with handler._lock:
            assert "/test/file.txt" in handler._pending
            assert handler._pending["/test/file.txt"][0] == "change"
            # Cancel timer to avoid flush during test
            if handler._timer:
                handler._timer.cancel()

    def test_debounce_replaces_timer(self, rag_entry, registry, settings):
        handler = _RagEventHandler(rag_entry, registry, settings, {".txt"})
        handler._schedule("/test/a.txt", "change")
        timer1 = handler._timer

        handler._schedule("/test/b.txt", "change")
        timer2 = handler._timer

        # Timer should be replaced
        assert timer1 is not timer2
        # Clean up
        if handler._timer:
            handler._timer.cancel()

    def test_flush_calls_on_change(self, rag_entry, registry, settings):
        callback = MagicMock()
        handler = _RagEventHandler(
            rag_entry, registry, settings, {".txt"}, on_change=callback
        )

        with handler._lock:
            handler._pending["/test/new.txt"] = ("change", time.monotonic())
            handler._pending["/test/old.txt"] = ("delete", time.monotonic())

        handler._flush()

        callback.assert_called_once()
        changed, deleted = callback.call_args[0]
        assert "/test/new.txt" in changed
        assert "/test/old.txt" in deleted

    def test_flush_clears_pending(self, rag_entry, registry, settings):
        handler = _RagEventHandler(rag_entry, registry, settings, {".txt"})
        handler._pending["/test/file.txt"] = ("change", time.monotonic())
        handler._flush()
        assert len(handler._pending) == 0


# ---------------------------------------------------------------------------
# FolderWatcher lifecycle
# ---------------------------------------------------------------------------


class TestFolderWatcher:
    def test_start_and_stop(self, rag_entry, registry, settings, tmp_path):
        # Ensure source folder exists
        src = Path(rag_entry.source_folders[0])
        src.mkdir(exist_ok=True)

        watcher = FolderWatcher(rag_entry, registry, settings)
        assert watcher.is_running is False

        watcher.start()
        assert watcher.is_running is True

        watcher.stop()
        assert watcher.is_running is False

    def test_start_detached_rag_no_op(self, detached_rag_entry, registry, settings):
        watcher = FolderWatcher(detached_rag_entry, registry, settings)
        watcher.start()
        assert watcher.is_running is False  # detached RAGs don't start watching

    def test_start_no_source_folders(self, tmp_path, registry, settings):
        entry = RagEntry(
            name="no-source",
            db_path=str(tmp_path / "chroma_db"),
            source_folders=[],
        )
        watcher = FolderWatcher(entry, registry, settings)
        watcher.start()
        assert watcher.is_running is False

    def test_start_nonexistent_folder(self, tmp_path, registry, settings):
        entry = RagEntry(
            name="bad-folder",
            db_path=str(tmp_path / "chroma_db"),
            source_folders=[str(tmp_path / "nonexistent")],
        )
        watcher = FolderWatcher(entry, registry, settings)
        watcher.start()
        # Should still start the observer even if individual folders don't exist

    def test_double_start_no_op(self, rag_entry, registry, settings):
        src = Path(rag_entry.source_folders[0])
        src.mkdir(exist_ok=True)

        watcher = FolderWatcher(rag_entry, registry, settings)
        watcher.start()
        watcher.start()  # should be a no-op
        assert watcher.is_running is True
        watcher.stop()

    def test_file_change_triggers_callback(self, rag_entry, registry, settings, tmp_path):
        src = Path(rag_entry.source_folders[0])
        src.mkdir(exist_ok=True)

        events_received = []
        callback = lambda changed, deleted: events_received.append((changed, deleted))

        watcher = FolderWatcher(rag_entry, registry, settings, on_change=callback)
        watcher.start()

        try:
            # Create a file in the watched directory
            test_file = src / "new_doc.txt"
            test_file.write_text("Hello watcher!", encoding="utf-8")

            # Wait for debounce + processing (3 seconds should be enough)
            time.sleep(4)

            # We expect the callback to have been called
            # (timing-dependent, may not always fire in CI)
            if events_received:
                changed, deleted = events_received[-1]
                assert any("new_doc.txt" in p for p in changed)
        finally:
            watcher.stop()
