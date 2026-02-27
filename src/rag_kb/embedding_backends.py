"""Pluggable embedding backends — local sentence-transformers and API providers.

Each backend exposes the same interface so that the rest of the application
(embedder, indexer, search) does not need to know which provider is active.

Supported backends
------------------

- **SentenceTransformerBackend** — runs locally via ``sentence-transformers``.
- **OpenAIEmbeddingBackend** — uses the OpenAI Embeddings API.
- **VoyageEmbeddingBackend** — uses the Voyage AI Embeddings API.
"""

from __future__ import annotations

import logging
import os
import warnings
from abc import ABC, abstractmethod
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class EmbeddingBackend(ABC):
    """Protocol / base class for all embedding backends."""

    @abstractmethod
    def embed_texts(
        self,
        texts: list[str],
        *,
        batch_size: int = 256,
        show_progress: bool = False,
        is_query: bool = False,
    ) -> np.ndarray:
        """Embed a list of texts, returning (N, D) ndarray."""
        ...

    @abstractmethod
    def embed_single(self, text: str, *, is_query: bool = False) -> np.ndarray:
        """Embed a single text, returning (D,) ndarray."""
        ...

    @abstractmethod
    def get_dimension(self) -> int:
        """Return the dimensionality of embeddings produced by this backend."""
        ...


# ---------------------------------------------------------------------------
# Local: sentence-transformers
# ---------------------------------------------------------------------------

class SentenceTransformerBackend(EmbeddingBackend):
    """Backend wrapping a local ``SentenceTransformer`` model."""

    def __init__(
        self,
        model_name: str,
        *,
        trust_remote_code: bool = False,
        device: str | None = None,
        query_prefix: str | None = None,
        document_prefix: str | None = None,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required: pip install sentence-transformers"
            ) from exc

        from rag_kb.models import get_model_path

        self._raw_name = model_name
        resolved = get_model_path(model_name)

        # Silence noisy loggers
        for name in ("huggingface_hub", "transformers",
                      "sentence_transformers", "safetensors"):
            logging.getLogger(name).setLevel(logging.ERROR)

        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

        device = device or _detect_device()
        logger.info("Loading embedding model '%s' (device=%s) …", model_name, device)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model = SentenceTransformer(
                resolved,
                trust_remote_code=trust_remote_code,
                device=device,
            )
        logger.info("Model '%s' ready.", model_name)

        self._query_prefix = query_prefix or ""
        self._document_prefix = document_prefix or ""

    # -- EmbeddingBackend interface -----------------------------------------

    def embed_texts(
        self,
        texts: list[str],
        *,
        batch_size: int = 256,
        show_progress: bool = False,
        is_query: bool = False,
    ) -> np.ndarray:
        if not texts:
            return np.empty((0, self.get_dimension()))
        prefix = self._query_prefix if is_query else self._document_prefix
        if prefix:
            texts = [prefix + t for t in texts]
        return self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    def embed_single(self, text: str, *, is_query: bool = False) -> np.ndarray:
        prefix = self._query_prefix if is_query else self._document_prefix
        if prefix:
            text = prefix + text
        return self._model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    def get_dimension(self) -> int:
        dim = self._model.get_sentence_embedding_dimension()
        if dim is None:
            raise RuntimeError(f"Cannot determine dimension for model '{self._raw_name}'")
        return int(dim)


# ---------------------------------------------------------------------------
# OpenAI API
# ---------------------------------------------------------------------------

class OpenAIEmbeddingBackend(EmbeddingBackend):
    """Backend using the OpenAI Embeddings API."""

    def __init__(
        self,
        model_name: str,
        *,
        api_key: str | None = None,
        dimensions: int = 1536,
    ) -> None:
        try:
            import openai  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "openai is required for OpenAI embeddings: pip install openai"
            ) from exc

        self._api_model = model_name.removeprefix("openai/")
        self._dimensions = dimensions
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not self._api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY or provide it "
                "in the settings."
            )
        logger.info("OpenAI embedding backend initialised (%s, dim=%d).",
                     self._api_model, dimensions)

    def _get_client(self) -> Any:
        import openai
        return openai.OpenAI(api_key=self._api_key)

    def embed_texts(
        self,
        texts: list[str],
        *,
        batch_size: int = 256,
        show_progress: bool = False,
        is_query: bool = False,
    ) -> np.ndarray:
        if not texts:
            return np.empty((0, self._dimensions))

        client = self._get_client()
        all_embeddings: list[list[float]] = []

        # Process in batches (OpenAI allows up to 2048 per call)
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = client.embeddings.create(
                model=self._api_model,
                input=batch,
                dimensions=self._dimensions,
            )
            # Sort by index to maintain order
            sorted_data = sorted(response.data, key=lambda d: d.index)
            all_embeddings.extend([d.embedding for d in sorted_data])

        arr = np.array(all_embeddings, dtype=np.float32)
        # Normalise to unit length for cosine compatibility
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms

    def embed_single(self, text: str, *, is_query: bool = False) -> np.ndarray:
        result = self.embed_texts([text], is_query=is_query)
        return result[0]

    def get_dimension(self) -> int:
        return self._dimensions


# ---------------------------------------------------------------------------
# Voyage AI API
# ---------------------------------------------------------------------------

class VoyageEmbeddingBackend(EmbeddingBackend):
    """Backend using the Voyage AI Embeddings API."""

    def __init__(
        self,
        model_name: str,
        *,
        api_key: str | None = None,
        dimensions: int = 1024,
    ) -> None:
        try:
            import voyageai  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "voyageai is required for Voyage embeddings: pip install voyageai"
            ) from exc

        self._api_model = model_name.removeprefix("voyage-ai/")
        self._dimensions = dimensions
        self._api_key = api_key or os.environ.get("VOYAGE_API_KEY", "")
        if not self._api_key:
            raise ValueError(
                "Voyage AI API key is required. Set VOYAGE_API_KEY or provide "
                "it in the settings."
            )
        logger.info("Voyage AI embedding backend initialised (%s, dim=%d).",
                     self._api_model, dimensions)

    def _get_client(self) -> Any:
        import voyageai
        return voyageai.Client(api_key=self._api_key)

    def embed_texts(
        self,
        texts: list[str],
        *,
        batch_size: int = 128,
        show_progress: bool = False,
        is_query: bool = False,
    ) -> np.ndarray:
        if not texts:
            return np.empty((0, self._dimensions))

        client = self._get_client()
        all_embeddings: list[list[float]] = []
        input_type = "query" if is_query else "document"

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            result = client.embed(
                batch,
                model=self._api_model,
                input_type=input_type,
            )
            all_embeddings.extend(result.embeddings)

        arr = np.array(all_embeddings, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms

    def embed_single(self, text: str, *, is_query: bool = False) -> np.ndarray:
        result = self.embed_texts([text], is_query=is_query)
        return result[0]

    def get_dimension(self) -> int:
        return self._dimensions


# ---------------------------------------------------------------------------
# Device detection (shared by local backends)
# ---------------------------------------------------------------------------

def _detect_device() -> str:
    """Auto-detect the best available compute device.

    Delegates to the centralised helper in :mod:`rag_kb.device`.
    """
    from rag_kb.device import detect_device
    return detect_device()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

# Singleton cache — one backend per (model_name, provider)
_backend_cache: dict[str, EmbeddingBackend] = {}


def get_embedding_backend(
    model_name: str,
    *,
    api_key: str | None = None,
    trust_remote_code: bool | None = None,
    force_reload: bool = False,
) -> EmbeddingBackend:
    """Return the appropriate ``EmbeddingBackend`` for *model_name*.

    Looks up the model registry to determine the provider and configuration.
    Backends are cached as singletons keyed by model name.
    """
    if model_name in _backend_cache and not force_reload:
        return _backend_cache[model_name]

    from rag_kb.models import get_model_spec, ModelProvider

    spec = get_model_spec(model_name)
    provider = spec.provider if spec else ModelProvider.local

    backend: EmbeddingBackend

    if provider == ModelProvider.openai:
        dims = spec.dimensions if spec else 1536
        backend = OpenAIEmbeddingBackend(
            model_name, api_key=api_key, dimensions=dims,
        )
    elif provider == ModelProvider.voyage:
        dims = spec.dimensions if spec else 1024
        backend = VoyageEmbeddingBackend(
            model_name, api_key=api_key, dimensions=dims,
        )
    else:
        # Local sentence-transformers
        trust = trust_remote_code if trust_remote_code is not None else (
            spec.trust_remote_code if spec else False
        )
        q_prefix = spec.query_prefix if spec else None
        d_prefix = spec.document_prefix if spec else None

        backend = SentenceTransformerBackend(
            model_name,
            trust_remote_code=trust,
            query_prefix=q_prefix,
            document_prefix=d_prefix,
        )

    _backend_cache[model_name] = backend
    return backend


def clear_backend_cache() -> None:
    """Remove all cached backends (used on model switch)."""
    _backend_cache.clear()
