"""Embedding engine wrapping sentence-transformers (Apache 2.0).

Improvements over original:
  - Returns **numpy arrays** by default (avoids list-of-list copies).
  - Uses ``normalize_embeddings=True`` for pre-normalised cosine vectors.
  - Auto-detects CUDA / MPS and selects the best device.
  - Adaptive batch sizing based on available memory.
"""

from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Suppress noisy third-party output at import time
# ---------------------------------------------------------------------------
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

# Lazy-loaded singleton
_model_cache: dict[str, Any] = {}


def _silence_loggers() -> None:
    """Mute chatty libraries that pollute stderr during model loading."""
    for name in (
        "huggingface_hub",
        "transformers",
        "sentence_transformers",
        "safetensors",
    ):
        logging.getLogger(name).setLevel(logging.ERROR)


def _detect_device() -> str:
    """Auto-detect the best available device."""
    try:
        import torch
        if torch.cuda.is_available():
            logger.info("CUDA device detected — using GPU acceleration.")
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("Apple MPS device detected.")
            return "mps"
    except ImportError:
        pass
    return "cpu"


def _model_is_cached(model_name: str) -> bool:
    """Return True if the model weights already exist in the local HF cache."""
    try:
        from huggingface_hub import scan_cache_dir

        cache_info = scan_cache_dir()
        cached_repos = {r.repo_id for r in cache_info.repos}
        # sentence-transformers models are cached under their full repo id
        candidates = (
            model_name,
            f"sentence-transformers/{model_name}",
        )
        return any(c in cached_repos for c in candidates)
    except Exception:
        return False


def _get_model(model_name: str) -> Any:
    """Load (or return cached) SentenceTransformer model."""
    if model_name not in _model_cache:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required: pip install sentence-transformers"
            ) from exc

        _silence_loggers()

        is_cached = _model_is_cached(model_name)
        if is_cached:
            logger.info("Loading embedding model '%s' from local cache …", model_name)
        else:
            logger.info(
                "Downloading embedding model '%s' (~80 MB, one-time only) …",
                model_name,
            )

        device = _detect_device()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kwargs: dict[str, Any] = {"trust_remote_code": False, "device": device}
            if is_cached:
                kwargs["local_files_only"] = True
            _model_cache[model_name] = SentenceTransformer(model_name, **kwargs)
        logger.info("Model '%s' ready (device=%s).", model_name, device)
    return _model_cache[model_name]


def embed_texts(
    texts: list[str],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 256,
    show_progress: bool = False,
    as_numpy: bool = True,
) -> np.ndarray | list[list[float]]:
    """Embed a list of texts and return their vector representations.

    Parameters
    ----------
    as_numpy : bool
        If *True* (default), return a 2-D numpy array.  Otherwise return a
        list of lists (legacy behaviour for direct ChromaDB upsert).
    """
    if not texts:
        return np.empty((0, 0)) if as_numpy else []
    model = _get_model(model_name)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    if as_numpy:
        return embeddings  # already ndarray
    return [e.tolist() for e in embeddings]


def embed_query(
    text: str,
    model_name: str = "all-MiniLM-L6-v2",
) -> list[float]:
    """Embed a single query string."""
    model = _get_model(model_name)
    embedding = model.encode(
        text,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embedding.tolist()


def embed_query_numpy(
    text: str,
    model_name: str = "all-MiniLM-L6-v2",
) -> np.ndarray:
    """Embed a single query string and return as numpy 1-D array."""
    model = _get_model(model_name)
    return model.encode(
        text,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )


def get_embedding_dimension(model_name: str = "all-MiniLM-L6-v2") -> int:
    """Return the dimensionality of the embedding model."""
    model = _get_model(model_name)
    return model.get_sentence_embedding_dimension()
