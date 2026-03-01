"""VectorStore CRUD integration tests using real ChromaDB on tmp_path."""

from __future__ import annotations

import numpy as np
import pytest

from rag_kb.vector_store import VectorStore, VectorStoreRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dummy_embeddings(n: int, dim: int = 8) -> np.ndarray:
    """Deterministic dummy embeddings for testing."""
    rng = np.random.default_rng(42)
    embs = rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    return embs / (norms + 1e-10)


@pytest.fixture
def store(tmp_path):
    """Create a real ChromaDB store on tmp_path."""
    db_path = str(tmp_path / "chroma_db")
    s = VectorStore(db_path)
    yield s
    s.close(force=True)


@pytest.fixture
def populated_store(store):
    """Store with 5 documents pre-loaded."""
    n = 5
    dim = 384  # typical small model dimension
    embs = _dummy_embeddings(n, dim)
    ids = [f"src/doc{i}.txt::chunk_0" for i in range(n)]
    texts = [f"This is document number {i} with some content." for i in range(n)]
    metadatas = [{"source_file": f"src/doc{i}.txt", "chunk_index": "0"} for i in range(n)]

    store.add_documents(ids, texts, embs, metadatas)
    return store, embs


# ---------------------------------------------------------------------------
# VectorStore init and close
# ---------------------------------------------------------------------------


class TestVectorStoreInit:
    def test_creates_db_directory(self, tmp_path):
        db_path = str(tmp_path / "new_store" / "chroma_db")
        s = VectorStore(db_path)
        assert (tmp_path / "new_store" / "chroma_db").exists()
        s.close(force=True)

    def test_count_empty_store(self, store):
        assert store.count() == 0

    def test_context_manager(self, tmp_path):
        db_path = str(tmp_path / "ctx_store")
        with VectorStore(db_path) as s:
            assert s.count() == 0
        # After exit, client should be None
        assert s._client is None


# ---------------------------------------------------------------------------
# Write operations
# ---------------------------------------------------------------------------


class TestVectorStoreWrite:
    def test_add_documents(self, store):
        embs = _dummy_embeddings(3, 384)
        ids = ["a::chunk_0", "b::chunk_0", "c::chunk_0"]
        texts = ["text a", "text b", "text c"]
        metas = [{"source_file": f} for f in ["a", "b", "c"]]

        store.add_documents(ids, texts, embs, metas)
        assert store.count() == 3

    def test_upsert_updates_existing(self, store):
        embs = _dummy_embeddings(1, 384)
        store.add_documents(["id1"], ["original"], embs, [{"source_file": "f"}])
        assert store.count() == 1

        # Re-add same ID with different text
        store.add_documents(["id1"], ["updated"], embs, [{"source_file": "f"}])
        assert store.count() == 1  # should not duplicate

    def test_add_empty_does_nothing(self, store):
        store.add_documents([], [], np.empty((0, 384)), [])
        assert store.count() == 0

    def test_large_batch_upsert(self, store):
        n = 200
        embs = _dummy_embeddings(n, 384)
        ids = [f"file{i}::chunk_0" for i in range(n)]
        texts = [f"text {i}" for i in range(n)]
        metas = [{"source_file": f"file{i}"} for i in range(n)]

        store.add_documents(ids, texts, embs, metas)
        assert store.count() == n

    def test_delete_by_source(self, populated_store):
        store, _ = populated_store
        initial = store.count()
        deleted = store.delete_by_source("src/doc0.txt")
        assert deleted == 1
        assert store.count() == initial - 1

    def test_batch_delete_by_sources(self, populated_store):
        store, _ = populated_store
        deleted = store.batch_delete_by_sources(["src/doc0.txt", "src/doc1.txt"])
        assert deleted == 2
        assert store.count() == 3

    def test_clear(self, populated_store):
        store, _ = populated_store
        assert store.count() > 0
        store.clear()
        assert store.count() == 0


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


class TestVectorStoreSearch:
    def test_search_returns_results(self, populated_store):
        store, embs = populated_store
        query = embs[0].tolist()
        results = store.search(query, n_results=3)
        assert len(results) > 0
        assert all(r.text for r in results)

    def test_search_scores_descending(self, populated_store):
        store, embs = populated_store
        results = store.search(embs[0].tolist(), n_results=5)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_empty_store(self, store):
        query = [0.0] * 384
        results = store.search(query, n_results=5)
        assert results == []

    def test_search_with_min_score(self, populated_store):
        store, embs = populated_store
        results = store.search(embs[0].tolist(), n_results=5, min_score=0.99)
        # Only very similar results should pass
        for r in results:
            assert r.score >= 0.99

    def test_search_include_embeddings(self, populated_store):
        store, embs = populated_store
        results = store.search(embs[0].tolist(), n_results=3, include_embeddings=True)
        assert len(results) > 0
        assert results[0].embedding is not None


# ---------------------------------------------------------------------------
# Read operations
# ---------------------------------------------------------------------------


class TestVectorStoreRead:
    def test_get_all_documents(self, populated_store):
        store, _ = populated_store
        ids, texts, metas = store.get_all_documents()
        assert len(ids) == 5
        assert len(texts) == 5
        assert len(metas) == 5

    def test_get_by_source(self, populated_store):
        store, _ = populated_store
        items = store.get_by_source("src/doc0.txt")
        assert len(items) == 1
        assert items[0]["text"]

    def test_get_by_source_not_found(self, populated_store):
        store, _ = populated_store
        items = store.get_by_source("nonexistent.txt")
        assert items == []

    def test_list_sources(self, populated_store):
        store, _ = populated_store
        sources = store.list_sources()
        assert len(sources) == 5
        assert "src/doc0.txt" in sources

    def test_list_files_with_counts(self, populated_store):
        store, _ = populated_store
        counts = store.list_files_with_counts()
        assert len(counts) == 5
        assert all(c == 1 for c in counts.values())

    def test_get_stats(self, populated_store):
        store, _ = populated_store
        stats = store.get_stats()
        assert stats.total_chunks == 5
        assert stats.total_files == 5
        assert stats.avg_chunks_per_file == 1.0

    def test_get_stats_empty(self, store):
        stats = store.get_stats()
        assert stats.total_chunks == 0
        assert stats.total_files == 0


# ---------------------------------------------------------------------------
# _sanitise_metadatas
# ---------------------------------------------------------------------------


class TestSanitiseMetadatas:
    def test_none_values_replaced(self):
        result = VectorStore._sanitise_metadatas([{"key": None}])
        assert result == [{"key": ""}]

    def test_string_values_unchanged(self):
        result = VectorStore._sanitise_metadatas([{"key": "value"}])
        assert result == [{"key": "value"}]

    def test_non_scalar_converted(self):
        result = VectorStore._sanitise_metadatas([{"key": [1, 2, 3]}])
        assert result == [{"key": "[1, 2, 3]"}]

    def test_int_and_float_preserved(self):
        result = VectorStore._sanitise_metadatas([{"i": 42, "f": 3.14}])
        assert result == [{"i": 42, "f": 3.14}]

    def test_bool_preserved(self):
        result = VectorStore._sanitise_metadatas([{"flag": True}])
        assert result == [{"flag": True}]


# ---------------------------------------------------------------------------
# VectorStoreRegistry
# ---------------------------------------------------------------------------


class TestVectorStoreRegistryExtended:
    def test_register_and_get(self, tmp_path):
        reg = VectorStoreRegistry()
        s = VectorStore(str(tmp_path / "db"))
        reg.register(s)
        assert reg.get(str(tmp_path / "db")) is s
        s.close(force=True)
        reg.close_all()

    def test_unregister(self, tmp_path):
        reg = VectorStoreRegistry()
        s = VectorStore(str(tmp_path / "db"))
        reg.register(s)
        reg.unregister(str(tmp_path / "db"))
        assert reg.get(str(tmp_path / "db")) is None
        s.close(force=True)

    def test_close_for_path(self, tmp_path):
        reg = VectorStoreRegistry()
        s = VectorStore(str(tmp_path / "db"))
        reg.register(s)
        reg.close_for_path(str(tmp_path / "db"))
        assert reg.get(str(tmp_path / "db")) is None

    def test_close_all(self, tmp_path):
        reg = VectorStoreRegistry()
        s1 = VectorStore(str(tmp_path / "db1"))
        s2 = VectorStore(str(tmp_path / "db2"))
        reg.register(s1)
        reg.register(s2)
        reg.close_all()
        assert reg.get(str(tmp_path / "db1")) is None
        assert reg.get(str(tmp_path / "db2")) is None

    def test_get_nonexistent(self):
        reg = VectorStoreRegistry()
        assert reg.get("/nonexistent/path") is None
