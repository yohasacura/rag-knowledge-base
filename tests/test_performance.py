"""Tests for BM25 caching performance improvements."""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from rag_kb.search import BM25Index, get_bm25_cache


class TestBM25Index:
    """Tests for the BM25Index cache class."""

    @staticmethod
    def _make_fetch_fn(n: int = 10):
        """Create a mock fetch function returning n documents as (ids, texts, metas)."""
        ids = [f"doc_{i}" for i in range(n)]
        texts = [f"This is document number {i} about topic {i}" for i in range(n)]
        metas = [{"source_file": f"file_{i}.txt"} for i in range(n)]
        fn = MagicMock(return_value=(ids, texts, metas))
        return fn

    def test_first_build(self):
        """First call to get_or_build creates the index."""
        idx = BM25Index()
        fetch = self._make_fetch_fn(5)
        bm25, ids, texts, metas = idx.get_or_build("rag1", 5, fetch)
        assert bm25 is not None
        assert len(ids) == 5
        assert len(texts) == 5
        assert len(metas) == 5
        fetch.assert_called_once()

    def test_cache_hit(self):
        """Second call with same params reuses the cached index."""
        idx = BM25Index()
        fetch = self._make_fetch_fn(5)
        result1 = idx.get_or_build("rag1", 5, fetch)
        result2 = idx.get_or_build("rag1", 5, fetch)
        # Same BM25 object should be returned
        assert result1[0] is result2[0]
        # fetch should only be called once
        fetch.assert_called_once()

    def test_cache_miss_on_name_change(self):
        """Changing RAG name invalidates the cache."""
        idx = BM25Index()
        fetch = self._make_fetch_fn(5)
        idx.get_or_build("rag1", 5, fetch)
        idx.get_or_build("rag2", 5, fetch)
        assert fetch.call_count == 2

    def test_cache_miss_on_doc_count_change(self):
        """Changing doc count invalidates the cache."""
        idx = BM25Index()
        fetch = self._make_fetch_fn(5)
        idx.get_or_build("rag1", 5, fetch)
        idx.get_or_build("rag1", 10, fetch)
        assert fetch.call_count == 2

    def test_invalidate(self):
        """Explicit invalidate() forces rebuild on next call."""
        idx = BM25Index()
        fetch = self._make_fetch_fn(5)
        idx.get_or_build("rag1", 5, fetch)
        idx.invalidate()
        idx.get_or_build("rag1", 5, fetch)
        assert fetch.call_count == 2

    def test_stale_after_max_age(self):
        """Index becomes stale after max_age seconds."""
        idx = BM25Index()
        idx.max_age = 0.01  # 10ms
        fetch = self._make_fetch_fn(5)
        idx.get_or_build("rag1", 5, fetch)
        time.sleep(0.02)
        idx.get_or_build("rag1", 5, fetch)
        assert fetch.call_count == 2

    def test_global_cache_singleton(self):
        """get_bm25_cache() returns the same instance."""
        c1 = get_bm25_cache()
        c2 = get_bm25_cache()
        assert c1 is c2

    def test_empty_documents(self):
        """Cache handles empty document set gracefully."""
        idx = BM25Index()
        fetch = MagicMock(return_value=([], [], []))
        bm25, ids, texts, metas = idx.get_or_build("rag1", 0, fetch)
        # BM25 is None when no documents
        assert len(ids) == 0
