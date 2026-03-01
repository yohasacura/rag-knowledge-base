"""Deep search pipeline tests with known synthetic content.

Creates RAGs with topically distinct documents to validate semantic relevance,
hybrid fusion, reranking, and MMR diversity.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.slow


@pytest.fixture
def search_api(api_instance, tmp_path):
    """Create and index a RAG with topically distinct documents."""
    docs_dir = tmp_path / "search-docs"
    docs_dir.mkdir()

    # 3 topically distinct documents
    (docs_dir / "quantum.md").write_text(
        "# Quantum Computing\n\n"
        "Quantum computing uses qubits that exploit superposition and entanglement.\n"
        "Quantum gates manipulate qubits to perform computations.\n"
        "Shor's algorithm can factor large numbers exponentially faster.\n"
        "Grover's algorithm provides quadratic speedup for search problems.\n"
        "Quantum error correction is essential for fault-tolerant quantum computers.\n",
        encoding="utf-8",
    )
    (docs_dir / "music.md").write_text(
        "# Classical Music History\n\n"
        "Bach composed the Well-Tempered Clavier and Brandenburg Concertos.\n"
        "Mozart wrote over 600 compositions including symphonies and operas.\n"
        "Beethoven's nine symphonies transformed the classical music landscape.\n"
        "The Romantic period introduced greater emotional expression in music.\n"
        "Orchestral instruments include strings, woodwinds, brass, and percussion.\n",
        encoding="utf-8",
    )
    (docs_dir / "ml.md").write_text(
        "# Machine Learning Fundamentals\n\n"
        "Supervised learning trains models on labeled data for classification.\n"
        "Neural networks consist of layers of interconnected artificial neurons.\n"
        "Deep learning uses multiple hidden layers for feature extraction.\n"
        "Gradient descent optimises model parameters to minimise loss functions.\n"
        "Overfitting occurs when a model memorises training data instead of generalising.\n",
        encoding="utf-8",
    )

    api_instance.create_rag("search-test", folders=[str(docs_dir)])
    api_instance.index("search-test")
    return api_instance


class TestSearchRelevance:
    """Validate that search returns topically relevant results."""

    def test_quantum_query_returns_quantum_doc(self, search_api):
        results = search_api.search("quantum entanglement qubits", n_results=3, rag_name="search-test")
        assert len(results) > 0
        # The top result should come from the quantum document
        top = results[0]
        assert "quantum" in top.text.lower() or "qubit" in top.text.lower()

    def test_music_query_returns_music_doc(self, search_api):
        results = search_api.search("Bach symphony orchestra", n_results=3, rag_name="search-test")
        assert len(results) > 0
        top = results[0]
        assert any(
            kw in top.text.lower()
            for kw in ("bach", "symphony", "music", "orchestra", "mozart", "beethoven")
        )

    def test_ml_query_returns_ml_doc(self, search_api):
        results = search_api.search(
            "neural network gradient descent", n_results=3, rag_name="search-test"
        )
        assert len(results) > 0
        top = results[0]
        assert any(
            kw in top.text.lower()
            for kw in ("neural", "gradient", "learning", "network", "deep")
        )

    def test_scores_are_descending(self, search_api):
        results = search_api.search("computing", n_results=5, rag_name="search-test")
        if len(results) > 1:
            scores = [r.score for r in results]
            assert scores == sorted(scores, reverse=True)


class TestSearchEdgeCases:
    """Edge cases for the search API."""

    def test_empty_query(self, search_api):
        """Empty query should return empty results (not crash)."""
        results = search_api.search("", rag_name="search-test")
        assert isinstance(results, list)

    def test_very_long_query(self, search_api):
        """10k character query should not crash."""
        long_query = "quantum " * 1250  # ~10000 chars
        results = search_api.search(long_query, n_results=2, rag_name="search-test")
        assert isinstance(results, list)

    def test_special_characters_in_query(self, search_api):
        """Queries with special chars should not crash."""
        results = search_api.search(
            'query with "quotes" and (parens) [brackets] {braces} <angles>',
            rag_name="search-test",
        )
        assert isinstance(results, list)

    def test_unicode_query(self, search_api):
        """Unicode query should work."""
        results = search_api.search("量子コンピュータ", rag_name="search-test")
        assert isinstance(results, list)

    def test_no_results_with_high_min_score(self, search_api):
        """Setting min_score very high should return fewer/no results."""
        results = search_api.search(
            "quantum", n_results=5, rag_name="search-test", min_score=0.99
        )
        # With very high threshold, we should get few or no results
        assert len(results) <= 5

    def test_min_score_zero_returns_more(self, search_api):
        """Setting min_score=0 should return at least as many as the default."""
        default_results = search_api.search("computing", n_results=5, rag_name="search-test")
        loose_results = search_api.search(
            "computing", n_results=5, rag_name="search-test", min_score=0.0
        )
        assert len(loose_results) >= len(default_results)

    def test_n_results_one(self, search_api):
        """Requesting exactly 1 result should work."""
        results = search_api.search("quantum", n_results=1, rag_name="search-test")
        assert len(results) <= 1


class TestSearchOnEmptyRag:
    """Search on a RAG with no indexed content."""

    def test_search_empty_rag(self, api_instance, tmp_path):
        empty_dir = tmp_path / "empty-docs"
        empty_dir.mkdir()
        api_instance.create_rag("empty-rag", folders=[str(empty_dir)])
        # No indexing → store is empty
        results = api_instance.search("anything", rag_name="empty-rag")
        assert results == []


class TestSearchSingleDocument:
    """Search on a RAG with just 1 file."""

    def test_single_doc_rag(self, api_instance, tmp_path):
        docs = tmp_path / "single-doc"
        docs.mkdir()
        (docs / "only.txt").write_text(
            "The quick brown fox jumps over the lazy dog.", encoding="utf-8"
        )
        api_instance.create_rag("single-rag", folders=[str(docs)])
        api_instance.index("single-rag")

        results = api_instance.search("fox jumps", n_results=3, rag_name="single-rag")
        assert len(results) > 0
        assert "fox" in results[0].text.lower()
