"""Advanced search utilities — re-ranking, hybrid search, MMR diversity.

All heavy models are loaded lazily and cached as singletons so that the
first query pays the loading cost and subsequent queries are instant.
"""

from __future__ import annotations

import logging
import os
import warnings
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy cross-encoder singleton
# ---------------------------------------------------------------------------

_reranker_cache: dict[str, Any] = {}


def _get_reranker(model_name: str) -> Any:
    """Load (or return cached) CrossEncoder model."""
    if model_name not in _reranker_cache:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for re-ranking: "
                "pip install sentence-transformers"
            ) from exc

        # Prefer a bundled / pre-downloaded model directory
        from rag_kb.models import get_model_path, get_model_spec
        resolved = get_model_path(model_name)

        # Respect trust_remote_code from the registry
        spec = get_model_spec(model_name)
        trust = spec.trust_remote_code if spec else False

        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        from rag_kb.device import detect_device
        device = detect_device()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _reranker_cache[model_name] = CrossEncoder(
                resolved, trust_remote_code=trust, device=device,
            )
        logger.info("Cross-encoder '%s' loaded (device=%s).", model_name, device)
    return _reranker_cache[model_name]


# ---------------------------------------------------------------------------
# Re-ranking
# ---------------------------------------------------------------------------

def rerank_cross_encoder(
    query: str,
    texts: list[str],
    scores: list[float],
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_n: int | None = None,
) -> list[tuple[int, float]]:
    """Re-rank *texts* against *query* using a cross-encoder.

    Returns a list of ``(original_index, new_score)`` tuples sorted by
    descending relevance.  *top_n* limits the returned list length.
    """
    if not texts:
        return []

    logger.debug("Reranking %d texts with model '%s'", len(texts), model_name)
    reranker = _get_reranker(model_name)
    pairs = [[query, t] for t in texts]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ce_scores = reranker.predict(pairs, show_progress_bar=False)

    # Normalise cross-encoder scores to 0-1 via sigmoid
    ce_scores_arr = np.array(ce_scores, dtype=np.float64)
    ce_scores_norm = 1.0 / (1.0 + np.exp(-ce_scores_arr))

    indexed = list(enumerate(ce_scores_norm))
    indexed.sort(key=lambda x: x[1], reverse=True)

    if top_n is not None:
        indexed = indexed[:top_n]

    return [(idx, float(score)) for idx, score in indexed]


# ---------------------------------------------------------------------------
# Maximal Marginal Relevance (MMR)
# ---------------------------------------------------------------------------

def mmr_diversify(
    query_embedding: np.ndarray | list[float],
    doc_embeddings: np.ndarray,
    scores: list[float],
    lambda_mult: float = 0.7,
    top_n: int = 5,
) -> list[int]:
    """Select a diverse subset of documents using MMR.

    Parameters
    ----------
    query_embedding : array-like, shape (dim,)
    doc_embeddings  : array, shape (n_docs, dim)
    scores          : relevance scores for each doc (higher = better)
    lambda_mult     : 0 = max diversity, 1 = max relevance
    top_n           : number of documents to select

    Returns
    -------
    List of original indices in the selection order.
    """
    if len(scores) == 0:
        return []
    if len(scores) <= top_n:
        return list(range(len(scores)))

    q = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
    docs = np.array(doc_embeddings, dtype=np.float32)

    # Normalise for cosine similarity
    q_norm = q / (np.linalg.norm(q) + 1e-10)
    docs_norm = docs / (np.linalg.norm(docs, axis=1, keepdims=True) + 1e-10)

    # Similarity to query
    sim_to_query = (docs_norm @ q_norm.T).flatten()

    selected: list[int] = []
    candidates = list(range(len(scores)))

    for _ in range(min(top_n, len(candidates))):
        if not candidates:
            break

        best_idx = -1
        best_score = -float("inf")

        for c in candidates:
            relevance = lambda_mult * sim_to_query[c]

            # Max similarity to already-selected
            if selected:
                sim_to_selected = (docs_norm[c] @ docs_norm[selected].T).max()
            else:
                sim_to_selected = 0.0

            diversity_penalty = (1.0 - lambda_mult) * sim_to_selected
            mmr_score = relevance - diversity_penalty

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = c

        if best_idx >= 0:
            selected.append(best_idx)
            candidates.remove(best_idx)

    return selected


# ---------------------------------------------------------------------------
# BM25 keyword search
# ---------------------------------------------------------------------------

def bm25_search(
    query: str,
    corpus_texts: list[str],
    doc_ids: list[str] | None = None,
    top_k: int = 20,
) -> list[tuple[str, float]]:
    """Run BM25 keyword search over *corpus_texts*.

    Parameters
    ----------
    query : str
        Search query.
    corpus_texts : list[str]
        The documents to search over.
    doc_ids : list[str] | None
        Optional document identifiers.  If provided, results use these IDs;
        otherwise the list index (as string) is returned.
    top_k : int
        Maximum number of results.

    Returns a list of ``(doc_id, bm25_score)`` sorted by descending score.
    """
    if not corpus_texts or not query.strip():
        return []

    try:
        from rank_bm25 import BM25Okapi
    except ImportError as exc:
        raise ImportError("rank-bm25 is required for hybrid search: pip install rank-bm25") from exc

    tokenized_corpus = [doc.lower().split() for doc in corpus_texts]
    bm25 = BM25Okapi(tokenized_corpus)

    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    # Sort descending
    indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    ids = doc_ids if doc_ids else [str(i) for i in range(len(corpus_texts))]
    return [(ids[idx], float(s)) for idx, s in indexed[:top_k] if s > 0]


# ---------------------------------------------------------------------------
# Hybrid score fusion
# ---------------------------------------------------------------------------

def hybrid_fuse_scores(
    vector_scores: dict[str, float],
    bm25_scores: dict[str, float],
    alpha: float = 0.7,
) -> dict[str, float]:
    """Fuse vector and BM25 scores using weighted combination.

    *alpha* = 1.0 → pure vector search, *alpha* = 0.0 → pure BM25.
    Keys are chunk IDs (or any common identifier).

    Both score dicts are normalised to [0, 1] before fusion.
    """
    all_keys = set(vector_scores) | set(bm25_scores)
    if not all_keys:
        return {}

    # Normalise vector scores to [0, 1]
    v_vals = list(vector_scores.values()) if vector_scores else [0.0]
    v_min, v_max = min(v_vals), max(v_vals)
    v_range = v_max - v_min if v_max > v_min else 1.0

    # Normalise BM25 scores to [0, 1]
    b_vals = list(bm25_scores.values()) if bm25_scores else [0.0]
    b_min, b_max = min(b_vals), max(b_vals)
    b_range = b_max - b_min if b_max > b_min else 1.0

    fused: dict[str, float] = {}
    for key in all_keys:
        v = (vector_scores.get(key, v_min) - v_min) / v_range
        b = (bm25_scores.get(key, b_min) - b_min) / b_range
        fused[key] = alpha * v + (1 - alpha) * b

    return fused
