"""Tests for cooperative cancellation of the indexing pipeline."""

from __future__ import annotations

import json
import os
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from rag_kb.indexer import Indexer, IndexingCancelledError

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_rag(tmp_path):
    """Create a minimal RAG entry and source folder with a few text files."""
    source_dir = tmp_path / "docs"
    source_dir.mkdir()

    # Create some test files
    for i in range(5):
        (source_dir / f"file{i}.txt").write_text(
            f"This is test document number {i}. " * 20,
            encoding="utf-8",
        )

    db_path = tmp_path / "db"
    db_path.mkdir()

    # Minimal RagEntry-like object
    rag = MagicMock()
    rag.name = "test-rag"
    rag.db_path = str(db_path)
    rag.source_folders = [str(source_dir)]
    rag.embedding_model = "paraphrase-multilingual-MiniLM-L12-v2"
    rag.detached = False
    rag.is_imported = False
    rag.file_count = 0
    rag.chunk_count = 0

    return rag


@pytest.fixture
def mock_settings():
    """Create a mock AppSettings."""
    settings = MagicMock()
    settings.supported_extensions = [".txt", ".md"]
    settings.chunk_size = 512
    settings.chunk_overlap = 64
    settings.indexing_workers = 2
    settings.embedding_batch_size = 32
    settings.hnsw_ef_construction = 200
    settings.hnsw_m = 32
    return settings


@pytest.fixture
def mock_registry():
    """Create a mock RagRegistry."""
    registry = MagicMock()
    return registry


# ---------------------------------------------------------------------------
# Tests: IndexingCancelledError
# ---------------------------------------------------------------------------


class TestIndexingCancelledError:
    def test_is_exception(self):
        err = IndexingCancelledError("cancelled")
        assert isinstance(err, Exception)
        assert str(err) == "cancelled"


# ---------------------------------------------------------------------------
# Tests: Cancel event basics
# ---------------------------------------------------------------------------


class TestCancelEvent:
    def test_cancel_sets_event(self, tmp_rag, mock_settings, mock_registry):
        cancel_event = threading.Event()
        indexer = Indexer(
            tmp_rag,
            mock_registry,
            mock_settings,
            cancel_event=cancel_event,
        )
        assert not cancel_event.is_set()
        indexer.cancel()
        assert cancel_event.is_set()

    def test_check_cancelled_raises_when_set(self, tmp_rag, mock_settings, mock_registry):
        cancel_event = threading.Event()
        indexer = Indexer(
            tmp_rag,
            mock_registry,
            mock_settings,
            cancel_event=cancel_event,
        )
        # Should not raise when event is not set
        indexer._check_cancelled()

        # Should raise after setting event
        cancel_event.set()
        with pytest.raises(IndexingCancelledError):
            indexer._check_cancelled()

    def test_check_cancelled_emits_cancelling_state(self, tmp_rag, mock_settings, mock_registry):
        """Verify that _check_cancelled updates state to 'cancelling' before raising."""
        cancel_event = threading.Event()
        progress_states = []

        def on_progress(state):
            progress_states.append(state.status)

        indexer = Indexer(
            tmp_rag,
            mock_registry,
            mock_settings,
            on_progress=on_progress,
            cancel_event=cancel_event,
        )

        cancel_event.set()
        with pytest.raises(IndexingCancelledError):
            indexer._check_cancelled()

        assert "cancelling" in progress_states

    def test_cancel_from_another_thread(self, tmp_rag, mock_settings, mock_registry):
        """Verify that cancel_event set from another thread is visible."""
        cancel_event = threading.Event()
        indexer = Indexer(
            tmp_rag,
            mock_registry,
            mock_settings,
            cancel_event=cancel_event,
        )

        # Set from another thread
        def delayed_cancel():
            time.sleep(0.05)
            indexer.cancel()

        t = threading.Thread(target=delayed_cancel)
        t.start()

        # Poll until cancelled
        cancelled = False
        for _ in range(100):
            if cancel_event.is_set():
                cancelled = True
                break
            time.sleep(0.01)

        t.join()
        assert cancelled


# ---------------------------------------------------------------------------
# Tests: Indexer.index() cancellation behaviour
# ---------------------------------------------------------------------------


class TestIndexCancellation:
    @patch("rag_kb.indexer.embed_texts")
    @patch("rag_kb.indexer.VectorStore")
    def test_cancel_before_start_returns_cancelled(
        self,
        MockStore,
        mock_embed,
        tmp_rag,
        mock_settings,
        mock_registry,
    ):
        """If cancel is set before index() starts, it should return cancelled."""
        cancel_event = threading.Event()
        cancel_event.set()  # Pre-cancelled

        indexer = Indexer(
            tmp_rag,
            mock_registry,
            mock_settings,
            cancel_event=cancel_event,
        )
        state = indexer.index(full=False)
        assert state.status == "cancelled"

    @patch("rag_kb.indexer.embed_texts")
    @patch("rag_kb.indexer.VectorStore")
    def test_cancel_produces_cancelled_status(
        self,
        MockStore,
        mock_embed,
        tmp_rag,
        mock_settings,
        mock_registry,
    ):
        """Cancellation during indexing should produce status='cancelled'."""
        cancel_event = threading.Event()
        progress_states = []

        def on_progress(state):
            progress_states.append(state.status)
            # Cancel after the first scanning notification
            if state.status == "scanning":
                cancel_event.set()

        indexer = Indexer(
            tmp_rag,
            mock_registry,
            mock_settings,
            on_progress=on_progress,
            cancel_event=cancel_event,
        )

        state = indexer.index(full=False)
        assert state.status == "cancelled"
        assert "cancelling" in progress_states

    @patch("rag_kb.indexer.embed_texts")
    @patch("rag_kb.indexer.VectorStore")
    def test_cancel_records_duration(
        self,
        MockStore,
        mock_embed,
        tmp_rag,
        mock_settings,
        mock_registry,
    ):
        """Cancellation should still record duration_seconds."""
        cancel_event = threading.Event()
        cancel_event.set()

        indexer = Indexer(
            tmp_rag,
            mock_registry,
            mock_settings,
            cancel_event=cancel_event,
        )
        state = indexer.index(full=False)
        assert state.duration_seconds >= 0.0

    @patch("rag_kb.indexer.embed_texts")
    @patch("rag_kb.indexer.VectorStore")
    def test_cancel_removes_lock_file(
        self,
        MockStore,
        mock_embed,
        tmp_rag,
        mock_settings,
        mock_registry,
    ):
        """Lock file should be cleaned up even on cancellation."""
        cancel_event = threading.Event()
        cancel_event.set()

        indexer = Indexer(
            tmp_rag,
            mock_registry,
            mock_settings,
            cancel_event=cancel_event,
        )
        state = indexer.index(full=False)
        lock_path = os.path.join(tmp_rag.db_path, ".indexing_lock")
        assert not os.path.exists(lock_path), "Lock file should be removed after cancellation"


# ---------------------------------------------------------------------------
# Tests: core.py cancel_indexing()
# ---------------------------------------------------------------------------


class TestCoreCancelIndexing:
    def test_cancel_no_running_indexer(self):
        """cancel_indexing returns False when no indexer is active."""
        from rag_kb.core import RagKnowledgeBaseAPI

        api = MagicMock(spec=RagKnowledgeBaseAPI)
        api._indexer_lock = threading.Lock()
        api._current_indexer = None

        # Call the real method
        result = RagKnowledgeBaseAPI.cancel_indexing(api)
        assert result is False

    def test_cancel_with_running_indexer(self):
        """cancel_indexing returns True and calls indexer.cancel()."""
        from rag_kb.core import RagKnowledgeBaseAPI

        mock_indexer = MagicMock()
        api = MagicMock(spec=RagKnowledgeBaseAPI)
        api._indexer_lock = threading.Lock()
        api._current_indexer = mock_indexer

        result = RagKnowledgeBaseAPI.cancel_indexing(api)
        assert result is True
        mock_indexer.cancel.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: Lock file basics
# ---------------------------------------------------------------------------


class TestLockFile:
    def test_write_and_remove_lock_file(self, tmp_rag, mock_settings, mock_registry):
        indexer = Indexer(tmp_rag, mock_registry, mock_settings)
        lock_path = os.path.join(tmp_rag.db_path, ".indexing_lock")

        indexer._write_lock_file(full=False)
        assert os.path.exists(lock_path)

        with open(lock_path) as f:
            data = json.load(f)
        assert "started_at" in data
        assert "pid" in data
        assert data["full"] is False

        indexer._remove_lock_file()
        assert not os.path.exists(lock_path)

    def test_write_lock_file_full(self, tmp_rag, mock_settings, mock_registry):
        indexer = Indexer(tmp_rag, mock_registry, mock_settings)
        lock_path = os.path.join(tmp_rag.db_path, ".indexing_lock")

        indexer._write_lock_file(full=True)
        with open(lock_path) as f:
            data = json.load(f)
        assert data["full"] is True

    def test_check_incomplete_indexing_no_lock(self, tmp_rag):
        result = Indexer.check_incomplete_indexing(tmp_rag.db_path)
        assert result is None

    def test_check_incomplete_indexing_with_lock(self, tmp_rag, mock_settings, mock_registry):
        indexer = Indexer(tmp_rag, mock_registry, mock_settings)
        indexer._write_lock_file(full=False)

        # Check incomplete indexing — should find and remove the lock
        result = Indexer.check_incomplete_indexing(tmp_rag.db_path)
        assert result is not None
        assert "started_at" in result
        assert "pid" in result

        # Lock file should have been removed
        lock_path = os.path.join(tmp_rag.db_path, ".indexing_lock")
        assert not os.path.exists(lock_path)
