"""Advanced search utilities — re-ranking, hybrid search, MMR diversity.

All heavy models are loaded lazily and cached as singletons so that the
first query pays the loading cost and subsequent queries are instant.
"""

from __future__ import annotations

import logging
import os
import threading
import time
import warnings
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BM25 index cache — avoids reconstructing on every query
# ---------------------------------------------------------------------------


class BM25Index:
    """Cached BM25 index that invalidates when the corpus changes.

    Keyed by ``(rag_name, doc_count)`` — if the document count changes
    the index is rebuilt.  A full `collection.count()` call is cheap.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._rag_name: str | None = None
        self._doc_count: int = 0
        self._bm25: Any = None
        self._ids: list[str] = []
        self._texts: list[str] = []
        self._metas: list[dict[str, str]] = []
        self._built_at: float = 0.0
        # Maximum age in seconds before forcing a rebuild
        self.max_age: float = 300.0  # 5 minutes

    def _is_stale(self, rag_name: str, doc_count: int) -> bool:
        """Check if the cached index needs rebuilding."""
        if self._bm25 is None:
            return True
        if self._rag_name != rag_name:
            return True
        if self._doc_count != doc_count:
            return True
        return time.monotonic() - self._built_at > self.max_age

    def get_or_build(
        self,
        rag_name: str,
        doc_count: int,
        fetch_fn,
    ) -> tuple[Any, list[str], list[str], list[dict[str, str]]]:
        """Return the BM25 model and corpus data, rebuilding if stale.

        Parameters
        ----------
        rag_name : str
            Active RAG name (cache key).
        doc_count : int
            Current collection document count (invalidation check).
        fetch_fn : callable
            ``() -> (ids, texts, metas)`` to load the full corpus.

        Returns
        -------
        ``(bm25, ids, texts, metas)``
        """
        with self._lock:
            if self._is_stale(rag_name, doc_count):
                logger.debug(
                    "BM25 index cache miss (rag=%s, count=%d→%d) — rebuilding",
                    rag_name,
                    self._doc_count,
                    doc_count,
                )
                ids, texts, metas = fetch_fn()
                self._build(rag_name, doc_count, ids, texts, metas)
            # Return a snapshot so callers hold stable references
            return self._bm25, self._ids, self._texts, self._metas

    def _build(
        self,
        rag_name: str,
        doc_count: int,
        ids: list[str],
        texts: list[str],
        metas: list[dict[str, str]],
    ) -> None:
        from rank_bm25 import BM25Okapi

        tokenized = [doc.lower().split() for doc in texts]
        self._bm25 = BM25Okapi(tokenized) if tokenized else None
        self._ids = ids
        self._texts = texts
        self._metas = metas
        self._rag_name = rag_name
        self._doc_count = doc_count
        self._built_at = time.monotonic()
        logger.debug("BM25 index built: %d documents", len(ids))

    def invalidate(self) -> None:
        """Force cache invalidation (e.g. after indexing)."""
        with self._lock:
            self._bm25 = None
            self._rag_name = None
            self._doc_count = 0


# Global singleton
_bm25_cache = BM25Index()


def get_bm25_cache() -> BM25Index:
    """Return the global BM25 index cache singleton."""
    return _bm25_cache


# ---------------------------------------------------------------------------
# Lazy cross-encoder singleton
# ---------------------------------------------------------------------------

_reranker_cache: dict[str, Any] = {}
_reranker_lock = threading.Lock()
_MAX_RERANKER_CACHE_SIZE = 3


def _get_reranker(model_name: str) -> Any:
    """Load (or return cached) CrossEncoder model.

    Uses double-checked locking: fast path returns under lock,
    heavy model loading happens outside the lock, then the result
    is inserted under the lock (re-checking to handle the race
    where another thread loaded the same model concurrently).
    """
    # Fast path — model already cached
    with _reranker_lock:
        if model_name in _reranker_cache:
            return _reranker_cache[model_name]

    # Slow path — load model outside lock to avoid blocking searches
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required for re-ranking: "
            "pip install sentence-transformers"
        ) from exc

    from rag_kb.models import get_model_path, get_model_spec

    resolved = get_model_path(model_name)

    spec = get_model_spec(model_name)
    trust = spec.trust_remote_code if spec else False

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    from rag_kb.device import detect_device

    device = detect_device()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = CrossEncoder(
            resolved,
            trust_remote_code=trust,
            device=device,
        )

    # Re-acquire lock for cache insertion (double-check)
    with _reranker_lock:
        if model_name in _reranker_cache:
            # Another thread beat us — discard our copy, return theirs
            return _reranker_cache[model_name]
        # Evict oldest entry if cache is full
        if len(_reranker_cache) >= _MAX_RERANKER_CACHE_SIZE:
            oldest_key = next(iter(_reranker_cache))
            logger.info(
                "Evicting reranker '%s' from cache (max=%d)",
                oldest_key,
                _MAX_RERANKER_CACHE_SIZE,
            )
            del _reranker_cache[oldest_key]
        _reranker_cache[model_name] = model

    logger.info("Cross-encoder '%s' loaded (device=%s).", model_name, device)
    return model


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
    *,
    bm25_index: Any | None = None,
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
    bm25_index : BM25Okapi | None
        Pre-built BM25 index.  If provided, skips index construction.

    Returns a list of ``(doc_id, bm25_score)`` sorted by descending score.
    """
    if not corpus_texts or not query.strip():
        return []

    if bm25_index is None:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise ImportError(
                "rank-bm25 is required for hybrid search: pip install rank-bm25"
            ) from exc

        tokenized_corpus = [doc.lower().split() for doc in corpus_texts]
        bm25_index = BM25Okapi(tokenized_corpus)

    tokenized_query = query.lower().split()
    scores = bm25_index.get_scores(tokenized_query)

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
