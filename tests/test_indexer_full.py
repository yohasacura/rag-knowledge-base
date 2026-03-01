"""Indexer full tests — discover_files, parse pipeline, single file ops."""

from __future__ import annotations

import os
import threading
from pathlib import Path

import pytest

from rag_kb.config import AppSettings, RagEntry, RagRegistry
from rag_kb.indexer import Indexer, IndexingCancelledError, IndexingState


@pytest.fixture
def rag_env(tmp_path):
    """Set up registry, entry, settings and source docs for indexing tests."""
    reg = RagRegistry(tmp_path / "registry.json", tmp_path / "rags")
    src_dir = tmp_path / "source"
    src_dir.mkdir()

    entry = reg.create_rag("idx-test", folders=[str(src_dir)])
    settings = AppSettings()

    return {
        "registry": reg,
        "entry": entry,
        "settings": settings,
        "src_dir": src_dir,
        "tmp_path": tmp_path,
    }


def _write_sample_docs(src_dir: Path, count: int = 5) -> list[Path]:
    """Create simple text files in src_dir."""
    files = []
    for i in range(count):
        f = src_dir / f"doc{i}.txt"
        f.write_text(f"Document {i} content. This is sample text for indexing.", encoding="utf-8")
        files.append(f)
    return files


# ---------------------------------------------------------------------------
# discover_files
# ---------------------------------------------------------------------------


class TestDiscoverFiles:
    def test_finds_supported_files(self, rag_env):
        src = rag_env["src_dir"]
        _write_sample_docs(src, 3)
        (src / "code.py").write_text("print('hello')", encoding="utf-8")

        indexer = Indexer(rag_env["entry"], rag_env["registry"], rag_env["settings"])
        try:
            files = indexer._discover_files()
            assert len(files) >= 3  # at least the .txt files
        finally:
            indexer.close()

    def test_skips_hidden_directories(self, rag_env):
        src = rag_env["src_dir"]
        _write_sample_docs(src, 1)

        hidden = src / ".git"
        hidden.mkdir()
        (hidden / "config").write_text("git config", encoding="utf-8")

        indexer = Indexer(rag_env["entry"], rag_env["registry"], rag_env["settings"])
        try:
            files = indexer._discover_files()
            assert not any(".git" in str(f) for f in files)
        finally:
            indexer.close()

    def test_recurses_subdirectories(self, rag_env):
        src = rag_env["src_dir"]
        sub = src / "subdir"
        sub.mkdir()
        (sub / "nested.txt").write_text("Nested content.", encoding="utf-8")
        (src / "top.txt").write_text("Top-level.", encoding="utf-8")

        indexer = Indexer(rag_env["entry"], rag_env["registry"], rag_env["settings"])
        try:
            files = indexer._discover_files()
            paths = [str(f) for f in files]
            assert any("nested.txt" in p for p in paths)
            assert any("top.txt" in p for p in paths)
        finally:
            indexer.close()

    def test_empty_source_folder(self, rag_env):
        indexer = Indexer(rag_env["entry"], rag_env["registry"], rag_env["settings"])
        try:
            files = indexer._discover_files()
            assert len(files) == 0
        finally:
            indexer.close()


# ---------------------------------------------------------------------------
# IndexingState
# ---------------------------------------------------------------------------


class TestIndexingState:
    def test_defaults(self):
        state = IndexingState()
        assert state.status == "idle"
        assert state.progress == 0.0
        assert state.total_files == 0
        assert state.errors == []

    def test_mutable(self):
        state = IndexingState()
        state.status = "scanning"
        state.total_files = 10
        assert state.status == "scanning"


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------


class TestIndexerCancellation:
    def test_cancel_before_start(self, rag_env):
        cancel = threading.Event()
        cancel.set()  # pre-cancelled

        indexer = Indexer(
            rag_env["entry"],
            rag_env["registry"],
            rag_env["settings"],
            cancel_event=cancel,
        )
        try:
            _write_sample_docs(rag_env["src_dir"], 3)
            # index() catches IndexingCancelledError internally and returns state
            result = indexer.index(full=True)
            assert result.status == "cancelled"
        finally:
            indexer.close()


# ---------------------------------------------------------------------------
# Full index (slow — requires model loading)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestFullIndex:
    def test_index_and_verify(self, rag_env):
        _write_sample_docs(rag_env["src_dir"], 3)

        progress_states = []

        def on_progress(state: IndexingState):
            progress_states.append(state.status)

        indexer = Indexer(
            rag_env["entry"],
            rag_env["registry"],
            rag_env["settings"],
            on_progress=on_progress,
        )
        try:
            indexer.index(full=True)
            assert indexer.state.status == "done"
            assert indexer.state.total_chunks > 0
            assert indexer.state.processed_files == 3
            assert len(progress_states) > 0
        finally:
            indexer.close()

    def test_incremental_skips_unchanged(self, rag_env):
        _write_sample_docs(rag_env["src_dir"], 2)

        indexer = Indexer(rag_env["entry"], rag_env["registry"], rag_env["settings"])
        try:
            indexer.index(full=True)
            first_chunks = indexer.state.total_chunks
        finally:
            indexer.close()

        # Second index should not produce new chunks (all unchanged)
        indexer2 = Indexer(rag_env["entry"], rag_env["registry"], rag_env["settings"])
        try:
            indexer2.index(full=False)
            assert indexer2.state.status == "done"
            assert indexer2.state.total_chunks == first_chunks
            # No new embedding/upsert work was done
            assert indexer2.state.embed_seconds == 0.0
            assert indexer2.state.upsert_seconds == 0.0
        finally:
            indexer2.close()

    def test_index_single_file(self, rag_env):
        src = rag_env["src_dir"]
        f = src / "single.txt"
        f.write_text("This is a single file to index.", encoding="utf-8")

        indexer = Indexer(rag_env["entry"], rag_env["registry"], rag_env["settings"])
        try:
            indexer.index_single_file_by_path(str(f))
            # Should have created chunks for the single file
            assert indexer.store.count() > 0
        finally:
            indexer.close()

    def test_remove_file(self, rag_env):
        src = rag_env["src_dir"]
        f = src / "to_remove.txt"
        f.write_text("File that will be removed.", encoding="utf-8")

        indexer = Indexer(rag_env["entry"], rag_env["registry"], rag_env["settings"])
        try:
            indexer.index_single_file_by_path(str(f))
            # Now remove
            indexer.remove_file(str(f))
            # Chunks should be gone
            by_source = indexer.store.get_by_source(str(f))
            assert len(by_source) == 0
        finally:
            indexer.close()
