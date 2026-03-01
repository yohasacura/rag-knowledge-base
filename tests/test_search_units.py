"""Unit tests for search module utilities: BM25, hybrid fusion, reranking, MMR."""

from __future__ import annotations

import numpy as np
import pytest

from rag_kb.search import (
    BM25Index,
    bm25_search,
    get_bm25_cache,
    hybrid_fuse_scores,
    mmr_diversify,
    rerank_cross_encoder,
)


# ---------------------------------------------------------------------------
# bm25_search
# ---------------------------------------------------------------------------


class TestBM25Search:
    def test_basic_bm25_ranking(self):
        corpus = [
            "the cat sat on the mat",
            "the dog played in the park",
            "quantum computing with qubits and entanglement",
        ]
        results = bm25_search("cat mat", corpus)
        assert len(results) > 0
        # The cat/mat document should score highest
        top_id, top_score = results[0]
        assert top_id == "0"
        assert top_score > 0

    def test_empty_query_returns_empty(self):
        results = bm25_search("", ["doc one", "doc two"])
        assert results == []

    def test_empty_corpus_returns_empty(self):
        results = bm25_search("query", [])
        assert results == []

    def test_whitespace_query_returns_empty(self):
        results = bm25_search("   ", ["doc one"])
        assert results == []

    def test_custom_doc_ids(self):
        corpus = [
            "apple banana fruit juice",
            "cherry date grape melon",
            "orange pear kiwi strawberry",
            "mango avocado passion fruit",
            "watermelon blueberry cantaloupe",
            "dragonfruit elderberry fig",
            "guava honeydew jackfruit",
            "lemon lime nectarine",
        ]
        ids = [f"doc-{i}" for i in range(len(corpus))]
        results = bm25_search("apple", corpus, doc_ids=ids)
        assert len(results) > 0
        assert results[0][0] == "doc-0"  # apple is in first document

    def test_top_k_limits_results(self):
        corpus = [f"document number {i} with some text" for i in range(20)]
        results = bm25_search("document text", corpus, top_k=5)
        assert len(results) <= 5

    def test_pre_built_index(self):
        from rank_bm25 import BM25Okapi

        corpus = [
            "hello world from python land",
            "goodbye world from java realm",
            "greetings again from typescript kingdom",
            "farewell from ruby domain",
            "welcome to the c sharp universe",
            "salutations from go territory",
            "howdy from rust country",
            "hey there from swift region",
        ]
        tokenized = [doc.lower().split() for doc in corpus]
        bm25 = BM25Okapi(tokenized)

        results = bm25_search("hello", corpus, bm25_index=bm25)
        assert len(results) > 0


# ---------------------------------------------------------------------------
# hybrid_fuse_scores
# ---------------------------------------------------------------------------


class TestHybridFuseScores:
    def test_basic_fusion(self):
        vec = {"a": 0.9, "b": 0.5}
        bm25 = {"a": 0.3, "b": 0.8}
        fused = hybrid_fuse_scores(vec, bm25, alpha=0.7)
        assert "a" in fused
        assert "b" in fused

    def test_alpha_one_pure_vector(self):
        vec = {"a": 1.0, "b": 0.0}
        bm25 = {"a": 0.0, "b": 1.0}
        fused = hybrid_fuse_scores(vec, bm25, alpha=1.0)
        assert fused["a"] > fused["b"]

    def test_alpha_zero_pure_bm25(self):
        vec = {"a": 1.0, "b": 0.0}
        bm25 = {"a": 0.0, "b": 1.0}
        fused = hybrid_fuse_scores(vec, bm25, alpha=0.0)
        assert fused["b"] > fused["a"]

    def test_disjoint_keys(self):
        vec = {"a": 0.8}
        bm25 = {"b": 0.7}
        fused = hybrid_fuse_scores(vec, bm25, alpha=0.5)
        assert "a" in fused
        assert "b" in fused

    def test_empty_scores_returns_empty(self):
        assert hybrid_fuse_scores({}, {}) == {}

    def test_single_entry_both(self):
        vec = {"x": 0.5}
        bm25 = {"x": 0.5}
        fused = hybrid_fuse_scores(vec, bm25, alpha=0.5)
        # With identical scores, normalised result should be non-negative
        assert fused["x"] >= 0.0


# ---------------------------------------------------------------------------
# rerank_cross_encoder
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestRerankCrossEncoder:
    def test_reranking_changes_order(self):
        query = "What is quantum computing?"
        texts = [
            "The weather is sunny today.",
            "Quantum computing uses qubits for computation.",
            "I like pizza and pasta.",
        ]
        scores = [0.9, 0.5, 0.7]  # intentionally mis-ranked

        reranked = rerank_cross_encoder(query, texts, scores)
        assert len(reranked) == 3
        # The quantum text should be ranked first
        top_idx = reranked[0][0]
        assert top_idx == 1  # index of "Quantum computing..."

    def test_empty_texts_returns_empty(self):
        result = rerank_cross_encoder("query", [], [])
        assert result == []

    def test_top_n_limits_output(self):
        texts = ["a", "b", "c", "d", "e"]
        scores = [0.5] * 5
        result = rerank_cross_encoder("test", texts, scores, top_n=2)
        assert len(result) == 2

    def test_scores_are_normalized_01(self):
        texts = ["some text", "other text"]
        scores = [0.5, 0.3]
        result = rerank_cross_encoder("query", texts, scores)
        for _, score in result:
            assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# mmr_diversify
# ---------------------------------------------------------------------------


class TestMMRDiversify:
    def _make_embeddings(self, n, dim=4):
        """Create n random unit embeddings."""
        rng = np.random.default_rng(42)
        embs = rng.standard_normal((n, dim)).astype(np.float32)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        return embs / (norms + 1e-10)

    def test_basic_mmr(self):
        embs = self._make_embeddings(5)
        query = embs[0]
        scores = [0.9, 0.8, 0.7, 0.6, 0.5]
        selected = mmr_diversify(query, embs, scores, top_n=3)
        assert len(selected) == 3
        assert all(0 <= idx < 5 for idx in selected)

    def test_top_n_exceeds_available(self):
        embs = self._make_embeddings(3)
        query = embs[0]
        scores = [0.9, 0.8, 0.7]
        selected = mmr_diversify(query, embs, scores, top_n=10)
        # Should return all 3
        assert len(selected) == 3

    def test_empty_scores(self):
        embs = np.empty((0, 4), dtype=np.float32)
        selected = mmr_diversify(np.zeros(4), embs, [], top_n=5)
        assert selected == []

    def test_lambda_zero_max_diversity(self):
        embs = self._make_embeddings(5)
        query = embs[0]
        scores = [0.9, 0.8, 0.7, 0.6, 0.5]
        selected = mmr_diversify(query, embs, scores, lambda_mult=0.0, top_n=3)
        assert len(selected) == 3

    def test_lambda_one_max_relevance(self):
        embs = self._make_embeddings(5)
        query = embs[0]
        scores = [0.9, 0.8, 0.7, 0.6, 0.5]
        selected = mmr_diversify(query, embs, scores, lambda_mult=1.0, top_n=3)
        assert len(selected) == 3

    def test_single_document(self):
        embs = self._make_embeddings(1)
        selected = mmr_diversify(embs[0], embs, [1.0], top_n=1)
        assert selected == [0]

    def test_identical_embeddings(self):
        """All identical embeddings — MMR should still select top_n."""
        embs = np.ones((5, 4), dtype=np.float32)
        scores = [0.9, 0.8, 0.7, 0.6, 0.5]
        selected = mmr_diversify(embs[0], embs, scores, top_n=3)
        assert len(selected) == 3


# ---------------------------------------------------------------------------
# BM25Index cache
# ---------------------------------------------------------------------------


class TestBM25IndexExtended:
    def test_get_bm25_cache_returns_singleton(self):
        c1 = get_bm25_cache()
        c2 = get_bm25_cache()
        assert c1 is c2

    def test_cache_rebuilds_on_rag_change(self):
        cache = BM25Index()
        docs = (["id1"], ["text one"], [{}])
        cache.get_or_build("rag-a", 1, lambda: docs)
        assert cache._rag_name == "rag-a"

        cache.get_or_build("rag-b", 1, lambda: docs)
        assert cache._rag_name == "rag-b"

    def test_cache_rebuilds_on_count_change(self):
        cache = BM25Index()
        call_count = 0

        def fetch():
            nonlocal call_count
            call_count += 1
            return ["id1"], ["text"], [{}]

        cache.get_or_build("r", 1, fetch)
        cache.get_or_build("r", 1, fetch)  # should hit cache
        cache.get_or_build("r", 2, fetch)  # should miss (count changed)
        assert call_count == 2  # built twice, not three times

    def test_invalidate(self):
        cache = BM25Index()
        cache.get_or_build("r", 1, lambda: (["id"], ["txt"], [{}]))
        assert cache._bm25 is not None
        cache.invalidate()
        assert cache._bm25 is None
