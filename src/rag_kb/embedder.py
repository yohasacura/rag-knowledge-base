"""Embedding engine — thin facade over pluggable backends.

The public functions (``embed_texts``, ``embed_query``, …) are kept
backward-compatible so that callers (indexer, core, search) do not need
changes.  Internally, the heavy lifting is delegated to an
``EmbeddingBackend`` resolved from the model registry.

Features
--------
- Automatic backend selection (local sentence-transformers, OpenAI, Voyage AI)
- Query / document prefix injection for instruction-tuned models
- ``trust_remote_code`` respected per-model via registry
- Normalised cosine-ready embeddings
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from rag_kb.embedding_backends import (
    EmbeddingBackend,
    get_embedding_backend,
    clear_backend_cache,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API — drop-in replacements for the old functions
# ---------------------------------------------------------------------------

def _backend(
    model_name: str,
    *,
    api_key: str | None = None,
    trust_remote_code: bool | None = None,
) -> EmbeddingBackend:
    """Return (cached) backend for *model_name*."""
    return get_embedding_backend(
        model_name,
        api_key=api_key,
        trust_remote_code=trust_remote_code,
    )


def embed_texts(
    texts: list[str],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 256,
    show_progress: bool = False,
    as_numpy: bool = True,
    *,
    api_key: str | None = None,
    trust_remote_code: bool | None = None,
) -> np.ndarray | list[list[float]]:
    """Embed a list of texts and return their vector representations.

    Parameters
    ----------
    as_numpy : bool
        If *True* (default), return a 2-D numpy array.  Otherwise return a
        list of lists (legacy behaviour for direct ChromaDB upsert).
    api_key : str | None
        API key override for API-based models.
    trust_remote_code : bool | None
        Override for trust_remote_code. If None, uses the registry value.
    """
    if not texts:
        return np.empty((0, 0)) if as_numpy else []

    backend = _backend(model_name, api_key=api_key, trust_remote_code=trust_remote_code)

    t0 = time.perf_counter()
    embeddings = backend.embed_texts(
        texts,
        batch_size=batch_size,
        show_progress=show_progress,
        is_query=False,
    )
    duration_ms = (time.perf_counter() - t0) * 1000

    # Record embedding batch metrics (best-effort, import lazily)
    try:
        from rag_kb.metrics import EmbeddingBatchMetrics, MetricsCollector
        dim = embeddings.shape[1] if embeddings.ndim == 2 else 0
        cps = len(texts) / (duration_ms / 1000) if duration_ms > 0 else 0.0
        device = getattr(backend, '_model', None)
        device_str = ""
        if device is not None:
            device_str = str(getattr(device, 'device', ''))
        # Determine backend type
        backend_type = type(backend).__name__
        MetricsCollector.get().record_embedding_batch(EmbeddingBatchMetrics(
            rag_name=model_name,   # rag_name set to model; caller can override
            timestamp=time.time(),
            backend=backend_type,
            model_name=model_name,
            batch_size=len(texts),
            dimension=dim,
            duration_ms=round(duration_ms, 2),
            chunks_per_second=round(cps, 1),
            device=device_str,
        ))
    except Exception:
        pass  # Never let metrics recording break embedding

    if as_numpy:
        return embeddings
    return [e.tolist() for e in embeddings]


def embed_query(
    text: str,
    model_name: str = "all-MiniLM-L6-v2",
    *,
    api_key: str | None = None,
    trust_remote_code: bool | None = None,
) -> list[float]:
    """Embed a single query string."""
    backend = _backend(model_name, api_key=api_key, trust_remote_code=trust_remote_code)
    return backend.embed_single(text, is_query=True).tolist()


def embed_query_numpy(
    text: str,
    model_name: str = "all-MiniLM-L6-v2",
    *,
    api_key: str | None = None,
    trust_remote_code: bool | None = None,
) -> np.ndarray:
    """Embed a single query string and return as numpy 1-D array."""
    backend = _backend(model_name, api_key=api_key, trust_remote_code=trust_remote_code)
    return backend.embed_single(text, is_query=True)


def get_embedding_dimension(model_name: str = "all-MiniLM-L6-v2") -> int:
    """Return the dimensionality of the embedding model.

    For registry-known models this returns the catalogued dimension without
    loading the model.  Falls back to actually loading for unknown models.
    """
    from rag_kb.models import get_model_spec
    spec = get_model_spec(model_name)
    if spec and spec.dimensions > 0:
        return spec.dimensions
    # Fallback — load the model
    backend = _backend(model_name)
    return backend.get_dimension()
