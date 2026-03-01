"""Model registry and management for embedding and reranker models.

This module provides:
  1. A curated **model registry** with metadata for every supported model.
  2. Pre-downloading models for offline/bundled distribution.
  3. Resolving model paths — preferring a bundled ``models/`` directory
     (inside a PyInstaller bundle or repo) over the HuggingFace cache.
  4. Checking download status, disk usage, and managing local model files.

Models are categorised into two types:

- **Embedding** models produce dense vectors for documents and queries.
- **Reranker** (cross-encoder) models re-score query–document pairs for
  higher precision.

Providers:

- ``local`` — runs via sentence-transformers, fully offline after download.
- ``openai`` — uses the OpenAI Embeddings API (requires API key).
- ``voyage`` — uses the Voyage AI Embeddings API (requires API key).
"""

from __future__ import annotations

import logging
import sys
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ModelType(str, Enum):
    """Model purpose."""

    embedding = "embedding"
    reranker = "reranker"


class ModelProvider(str, Enum):
    """How the model is served."""

    local = "local"  # sentence-transformers, runs on-device
    openai = "openai"  # OpenAI Embeddings API
    voyage = "voyage"  # Voyage AI Embeddings API


class ModelStatus(str, Enum):
    """Runtime availability of a model."""

    bundled = "bundled"  # shipped with the application
    downloaded = "downloaded"  # present in HF cache or bundled dir
    available = "available"  # known but not yet downloaded
    api = "api"  # API-based, no local download needed


# ---------------------------------------------------------------------------
# Model specification
# ---------------------------------------------------------------------------


class ModelSpec(BaseModel):
    """Rich metadata for a supported model."""

    name: str = Field(description="HuggingFace repo ID or API model identifier")
    display_name: str = Field(description="Human-friendly label")
    type: ModelType
    provider: ModelProvider = ModelProvider.local

    # Technical specs
    dimensions: int = Field(description="Output embedding dimensions")
    max_tokens: int = Field(description="Maximum input sequence length (tokens)")
    model_size_mb: int = Field(0, description="Approximate download size in MB (0 for API)")

    # Documentation
    description: str = Field("", description="When to use this model (2-3 sentences)")
    use_case_tags: list[str] = Field(
        default_factory=list, description="E.g. multilingual, long-context, lightweight"
    )
    license: str = Field("Apache-2.0", description="SPDX license identifier")

    # Loading behaviour
    trust_remote_code: bool = Field(
        False, description="Whether the model requires trust_remote_code=True"
    )
    requires_api_key: bool = Field(False, description="True for API-based models")
    api_key_env_var: str | None = Field(None, description="Environment variable for the API key")

    # Recommended settings
    recommended_chunk_size: int = Field(1024, description="Optimal chunk size (chars)")
    recommended_chunk_overlap: int = Field(128, description="Optimal chunk overlap (chars)")
    recommended_batch_size: int = Field(256, description="Optimal embedding batch size")

    # Prefix handling (for instruction-tuned models)
    query_prefix: str | None = Field(
        None, description="Prefix prepended to queries, e.g. 'search_query: '"
    )
    document_prefix: str | None = Field(None, description="Prefix prepended to documents")

    # Matryoshka / MRL support
    matryoshka_dims: list[int] | None = Field(
        None, description="Supported Matryoshka dimension choices"
    )

    # Flags
    default: bool = Field(False, description="Bundled/default model")


# ---------------------------------------------------------------------------
# Curated embedding model catalog
# ---------------------------------------------------------------------------

EMBEDDING_MODELS: list[ModelSpec] = [
    # -- Lightweight / fast -----------------------------------------------
    ModelSpec(
        name="paraphrase-multilingual-MiniLM-L12-v2",
        display_name="Paraphrase Multilingual MiniLM L12",
        type=ModelType.embedding,
        dimensions=384,
        max_tokens=512,
        model_size_mb=120,
        description=(
            "Default multilingual embedding model. Compact and fast, supports "
            "50+ languages with good quality. Best for general-purpose multilingual "
            "RAG on CPU."
        ),
        use_case_tags=["multilingual", "lightweight", "cpu-friendly"],
        recommended_chunk_size=800,
        recommended_chunk_overlap=100,
        default=True,
    ),
    ModelSpec(
        name="all-MiniLM-L6-v2",
        display_name="All MiniLM L6 v2",
        type=ModelType.embedding,
        dimensions=384,
        max_tokens=512,
        model_size_mb=80,
        description=(
            "Fastest English embedding model. Very small footprint, ideal for "
            "quick experiments and constrained environments. Slightly lower quality "
            "than larger models."
        ),
        use_case_tags=["english", "lightweight", "cpu-friendly", "fast"],
        recommended_chunk_size=800,
        recommended_chunk_overlap=100,
    ),
    ModelSpec(
        name="all-MiniLM-L12-v2",
        display_name="All MiniLM L12 v2",
        type=ModelType.embedding,
        dimensions=384,
        max_tokens=512,
        model_size_mb=120,
        description=(
            "Balanced English model — more accurate than L6, still lightweight. "
            "Good default for English-only workloads on CPU."
        ),
        use_case_tags=["english", "lightweight", "cpu-friendly"],
        recommended_chunk_size=800,
        recommended_chunk_overlap=100,
    ),
    # -- Balanced quality -------------------------------------------------
    ModelSpec(
        name="all-mpnet-base-v2",
        display_name="All MPNet Base v2",
        type=ModelType.embedding,
        dimensions=768,
        max_tokens=512,
        model_size_mb=420,
        description=(
            "Higher-quality English model based on MPNet. Produces richer 768-d "
            "embeddings. Good balance between accuracy and speed for medium-sized "
            "corpora."
        ),
        use_case_tags=["english", "balanced"],
        recommended_chunk_size=800,
        recommended_chunk_overlap=100,
    ),
    ModelSpec(
        name="BAAI/bge-base-en-v1.5",
        display_name="BGE Base EN v1.5",
        type=ModelType.embedding,
        dimensions=768,
        max_tokens=512,
        model_size_mb=440,
        description=(
            "Strong English baseline from BAAI. Well-tuned for retrieval tasks "
            "with 768-d embeddings. Excellent quality-to-size ratio."
        ),
        use_case_tags=["english", "balanced", "retrieval-tuned"],
        license="MIT",
        recommended_chunk_size=800,
        recommended_chunk_overlap=100,
    ),
    ModelSpec(
        name="BAAI/bge-large-en-v1.5",
        display_name="BGE Large EN v1.5",
        type=ModelType.embedding,
        dimensions=1024,
        max_tokens=512,
        model_size_mb=1300,
        description=(
            "High-quality English model from BAAI with 1024-d embeddings. "
            "Top-tier retrieval accuracy for English. Requires more RAM/VRAM."
        ),
        use_case_tags=["english", "high-accuracy", "gpu-recommended"],
        license="MIT",
        recommended_chunk_size=800,
        recommended_chunk_overlap=100,
        recommended_batch_size=128,
    ),
    # -- Top-tier / long-context ------------------------------------------
    ModelSpec(
        name="BAAI/bge-m3",
        display_name="BGE-M3 (Multi-Functionality)",
        type=ModelType.embedding,
        dimensions=1024,
        max_tokens=8192,
        model_size_mb=2200,
        description=(
            "Top-tier multilingual model supporting 100+ languages with 8K context. "
            "Excels at dense retrieval. Best open-source choice for multilingual "
            "RAG with long documents."
        ),
        use_case_tags=["multilingual", "long-context", "high-accuracy", "gpu-recommended"],
        license="MIT",
        trust_remote_code=False,
        recommended_chunk_size=2048,
        recommended_chunk_overlap=256,
        recommended_batch_size=64,
    ),
    ModelSpec(
        name="nomic-ai/nomic-embed-text-v1.5",
        display_name="Nomic Embed Text v1.5",
        type=ModelType.embedding,
        dimensions=768,
        max_tokens=8192,
        model_size_mb=550,
        description=(
            "High-performance open-source model with 8K context and Matryoshka "
            "support. Requires task prefixes ('search_query:', 'search_document:'). "
            "Great quality-to-size ratio for long context."
        ),
        use_case_tags=["english", "long-context", "matryoshka"],
        trust_remote_code=True,
        query_prefix="search_query: ",
        document_prefix="search_document: ",
        matryoshka_dims=[64, 128, 256, 512, 768],
        recommended_chunk_size=2048,
        recommended_chunk_overlap=256,
        recommended_batch_size=128,
    ),
    ModelSpec(
        name="jinaai/jina-embeddings-v3",
        display_name="Jina Embeddings v3",
        type=ModelType.embedding,
        dimensions=1024,
        max_tokens=8192,
        model_size_mb=1200,
        description=(
            "Versatile multilingual model with 8K context, task-specific LoRA "
            "adapters, and Matryoshka support. Supports 30 languages. "
            "Note: CC-BY-NC-4.0 license (non-commercial)."
        ),
        use_case_tags=["multilingual", "long-context", "matryoshka", "non-commercial"],
        license="CC-BY-NC-4.0",
        trust_remote_code=True,
        matryoshka_dims=[32, 64, 128, 256, 512, 768, 1024],
        recommended_chunk_size=2048,
        recommended_chunk_overlap=256,
        recommended_batch_size=64,
    ),
    ModelSpec(
        name="Snowflake/snowflake-arctic-embed-l-v2.0",
        display_name="Arctic Embed L v2.0",
        type=ModelType.embedding,
        dimensions=1024,
        max_tokens=8192,
        model_size_mb=1300,
        description=(
            "High-quality open-source embedding model from Snowflake, optimized "
            "for retrieval with strong compression. 8K context with excellent "
            "accuracy."
        ),
        use_case_tags=["english", "long-context", "high-accuracy", "gpu-recommended"],
        trust_remote_code=True,
        query_prefix="Represent this sentence for searching relevant passages: ",
        recommended_chunk_size=2048,
        recommended_chunk_overlap=256,
        recommended_batch_size=64,
    ),
    ModelSpec(
        name="Alibaba-NLP/gte-Qwen2-1.5B-instruct",
        display_name="GTE Qwen2 1.5B Instruct",
        type=ModelType.embedding,
        dimensions=1536,
        max_tokens=32000,
        model_size_mb=6600,
        description=(
            "Very large LLM-based embedding model with 32K context for ultra-long "
            "documents. Exceptional quality but requires significant GPU memory "
            "(~8 GB VRAM). Best for long technical documents."
        ),
        use_case_tags=["multilingual", "long-context", "high-accuracy", "gpu-required"],
        trust_remote_code=True,
        recommended_chunk_size=4096,
        recommended_chunk_overlap=512,
        recommended_batch_size=16,
    ),
    # -- Older / compatibility --------------------------------------------
    ModelSpec(
        name="paraphrase-MiniLM-L6-v2",
        display_name="Paraphrase MiniLM L6 v2",
        type=ModelType.embedding,
        dimensions=384,
        max_tokens=512,
        model_size_mb=80,
        description=(
            "Small paraphrase-tuned model. Fast and lightweight, good for "
            "semantic similarity but not specifically tuned for retrieval."
        ),
        use_case_tags=["english", "lightweight", "cpu-friendly"],
        recommended_chunk_size=800,
        recommended_chunk_overlap=100,
    ),
    ModelSpec(
        name="multi-qa-MiniLM-L6-cos-v1",
        display_name="Multi-QA MiniLM L6",
        type=ModelType.embedding,
        dimensions=384,
        max_tokens=512,
        model_size_mb=80,
        description=(
            "Tuned specifically for question-answering retrieval. Good for "
            "FAQ-style knowledge bases with short documents."
        ),
        use_case_tags=["english", "lightweight", "qa-tuned"],
        recommended_chunk_size=800,
        recommended_chunk_overlap=100,
    ),
    ModelSpec(
        name="multi-qa-mpnet-base-cos-v1",
        display_name="Multi-QA MPNet Base",
        type=ModelType.embedding,
        dimensions=768,
        max_tokens=512,
        model_size_mb=420,
        description=(
            "Larger QA-tuned model with 768-d output. Better accuracy than "
            "the MiniLM variant for QA retrieval tasks."
        ),
        use_case_tags=["english", "balanced", "qa-tuned"],
        recommended_chunk_size=800,
        recommended_chunk_overlap=100,
    ),
    # -- API-based models -------------------------------------------------
    ModelSpec(
        name="openai/text-embedding-3-small",
        display_name="OpenAI Embedding 3 Small",
        type=ModelType.embedding,
        provider=ModelProvider.openai,
        dimensions=1536,
        max_tokens=8192,
        model_size_mb=0,
        description=(
            "OpenAI's efficient embedding model. Good quality with low cost. "
            "Supports Matryoshka dimension reduction. Requires API key and "
            "internet connection."
        ),
        use_case_tags=["multilingual", "long-context", "api", "matryoshka"],
        requires_api_key=True,
        api_key_env_var="OPENAI_API_KEY",
        matryoshka_dims=[256, 512, 1024, 1536],
        recommended_chunk_size=2048,
        recommended_chunk_overlap=256,
    ),
    ModelSpec(
        name="openai/text-embedding-3-large",
        display_name="OpenAI Embedding 3 Large",
        type=ModelType.embedding,
        provider=ModelProvider.openai,
        dimensions=3072,
        max_tokens=8192,
        model_size_mb=0,
        description=(
            "OpenAI's highest-quality embedding model. Excellent retrieval "
            "accuracy with Matryoshka support for flexible dimensions. "
            "Higher cost than the small variant."
        ),
        use_case_tags=["multilingual", "long-context", "api", "matryoshka", "high-accuracy"],
        requires_api_key=True,
        api_key_env_var="OPENAI_API_KEY",
        matryoshka_dims=[256, 512, 1024, 1536, 3072],
        recommended_chunk_size=2048,
        recommended_chunk_overlap=256,
    ),
    ModelSpec(
        name="voyage-ai/voyage-3",
        display_name="Voyage AI voyage-3",
        type=ModelType.embedding,
        provider=ModelProvider.voyage,
        dimensions=1024,
        max_tokens=32000,
        model_size_mb=0,
        description=(
            "Voyage AI's latest embedding model optimized for RAG. "
            "32K context with strong multilingual capabilities. "
            "Requires API key and internet connection."
        ),
        use_case_tags=["multilingual", "long-context", "api", "high-accuracy"],
        requires_api_key=True,
        api_key_env_var="VOYAGE_API_KEY",
        recommended_chunk_size=4096,
        recommended_chunk_overlap=512,
    ),
]


# ---------------------------------------------------------------------------
# Curated reranker model catalog
# ---------------------------------------------------------------------------

RERANKER_MODELS: list[ModelSpec] = [
    ModelSpec(
        name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        display_name="MS MARCO MiniLM L6 (Fast)",
        type=ModelType.reranker,
        dimensions=0,  # rerankers output scores, not embeddings
        max_tokens=512,
        model_size_mb=80,
        description=(
            "Default reranker — fast and lightweight cross-encoder trained on "
            "MS MARCO. Best for low-latency scenarios on CPU."
        ),
        use_case_tags=["english", "lightweight", "cpu-friendly", "fast"],
        recommended_chunk_size=800,
        recommended_chunk_overlap=100,
        default=True,
    ),
    ModelSpec(
        name="cross-encoder/ms-marco-MiniLM-L-12-v2",
        display_name="MS MARCO MiniLM L12",
        type=ModelType.reranker,
        dimensions=0,
        max_tokens=512,
        model_size_mb=130,
        description=(
            "More accurate version of the MiniLM reranker with 12 layers. "
            "Slight latency increase for meaningfully better re-ranking."
        ),
        use_case_tags=["english", "balanced"],
    ),
    ModelSpec(
        name="cross-encoder/ms-marco-TinyBERT-L-2-v2",
        display_name="MS MARCO TinyBERT L2 (Ultra-fast)",
        type=ModelType.reranker,
        dimensions=0,
        max_tokens=512,
        model_size_mb=18,
        description=(
            "Ultra-lightweight reranker with only 2 layers. Minimal latency "
            "impact, suitable for high-throughput scenarios."
        ),
        use_case_tags=["english", "lightweight", "cpu-friendly", "fast"],
    ),
    ModelSpec(
        name="cross-encoder/stsb-TinyBERT-L-4",
        display_name="STS-B TinyBERT L4",
        type=ModelType.reranker,
        dimensions=0,
        max_tokens=512,
        model_size_mb=55,
        description=(
            "Semantic similarity reranker trained on STS Benchmark. "
            "Better for semantic matching tasks than passage retrieval."
        ),
        use_case_tags=["english", "lightweight", "semantic-similarity"],
    ),
    ModelSpec(
        name="BAAI/bge-reranker-v2-m3",
        display_name="BGE Reranker v2 M3 (Multilingual)",
        type=ModelType.reranker,
        dimensions=0,
        max_tokens=8192,
        model_size_mb=1100,
        description=(
            "Top-tier multilingual reranker from BAAI with 8K context. "
            "Best choice for multilingual or long-document scenarios. "
            "Requires significant memory."
        ),
        use_case_tags=["multilingual", "long-context", "high-accuracy", "gpu-recommended"],
        license="MIT",
        trust_remote_code=False,
        recommended_batch_size=32,
    ),
    ModelSpec(
        name="BAAI/bge-reranker-base",
        display_name="BGE Reranker Base",
        type=ModelType.reranker,
        dimensions=0,
        max_tokens=512,
        model_size_mb=440,
        description=(
            "Solid English reranker from BAAI. Good accuracy with moderate "
            "resource usage. Strong baseline for English workloads."
        ),
        use_case_tags=["english", "balanced"],
        license="MIT",
    ),
    ModelSpec(
        name="jinaai/jina-reranker-v2-base-multilingual",
        display_name="Jina Reranker v2 Multilingual",
        type=ModelType.reranker,
        dimensions=0,
        max_tokens=8192,
        model_size_mb=560,
        description=(
            "Multilingual reranker from Jina AI with 8K context. "
            "Good balance of quality and size for multilingual scenarios. "
            "Note: CC-BY-NC-4.0 license (non-commercial)."
        ),
        use_case_tags=["multilingual", "long-context", "non-commercial"],
        license="CC-BY-NC-4.0",
        trust_remote_code=True,
    ),
    ModelSpec(
        name="mixedbread-ai/mxbai-rerank-base-v1",
        display_name="mxbai Rerank Base v1",
        type=ModelType.reranker,
        dimensions=0,
        max_tokens=512,
        model_size_mb=440,
        description=(
            "High-quality reranker from Mixedbread AI. Efficient and accurate "
            "for English retrieval tasks."
        ),
        use_case_tags=["english", "balanced"],
    ),
]


# ---------------------------------------------------------------------------
# Combined index for fast lookup
# ---------------------------------------------------------------------------

_ALL_MODELS: dict[str, ModelSpec] = {}


def _rebuild_index() -> None:
    """(Re)build the name -> spec lookup."""
    global _ALL_MODELS
    _ALL_MODELS = {m.name: m for m in EMBEDDING_MODELS + RERANKER_MODELS}


_rebuild_index()


# ---------------------------------------------------------------------------
# Public query helpers
# ---------------------------------------------------------------------------


def get_model_spec(name: str) -> ModelSpec | None:
    """Return the ``ModelSpec`` for *name*, or ``None`` if unknown."""
    return _ALL_MODELS.get(name)


def get_all_embedding_models() -> list[ModelSpec]:
    """Return all known embedding model specs."""
    return list(EMBEDDING_MODELS)


def get_all_reranker_models() -> list[ModelSpec]:
    """Return all known reranker model specs."""
    return list(RERANKER_MODELS)


def get_embedding_model_names() -> list[str]:
    """Return a plain list of embedding model names (for UI dropdowns)."""
    return [m.name for m in EMBEDDING_MODELS]


def get_reranker_model_names() -> list[str]:
    """Return a plain list of reranker model names (for UI dropdowns)."""
    return [m.name for m in RERANKER_MODELS]


# ---------------------------------------------------------------------------
# Where bundled models live
# ---------------------------------------------------------------------------


def _get_bundled_models_dir() -> Path:
    """Return the ``models/`` directory, whether running from source or a
    PyInstaller bundle."""
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        base = Path(sys._MEIPASS)
    else:
        base = Path(__file__).resolve().parent.parent.parent  # src/rag_kb -> repo root
    return base / "models"


BUNDLED_MODELS_DIR = _get_bundled_models_dir()


def get_model_path(model_name: str) -> str:
    """Return a local path for *model_name* if bundled, else the original name.

    When a matching sub-directory exists under ``models/``, return its path
    so that ``SentenceTransformer(path)`` / ``CrossEncoder(path)`` loads
    entirely offline without hitting the network.

    The directory is looked up by converting ``/`` in the model name to ``--``
    (the HuggingFace cache convention) or using the bare name.
    """
    if not BUNDLED_MODELS_DIR.exists():
        return model_name

    # Try exact match first
    candidate = BUNDLED_MODELS_DIR / model_name
    if candidate.is_dir():
        logger.info("Using bundled model at %s", candidate)
        return str(candidate)

    # Try with '/' -> '--' replacement
    safe_name = model_name.replace("/", "--")
    candidate = BUNDLED_MODELS_DIR / safe_name
    if candidate.is_dir():
        logger.info("Using bundled model at %s", candidate)
        return str(candidate)

    # Try with "sentence-transformers/" prefix
    st_name = f"sentence-transformers--{model_name}"
    candidate = BUNDLED_MODELS_DIR / st_name
    if candidate.is_dir():
        logger.info("Using bundled model at %s", candidate)
        return str(candidate)

    return model_name


# ---------------------------------------------------------------------------
# Model availability checks
# ---------------------------------------------------------------------------


def _is_bundled(model_name: str) -> bool:
    """Return True if the model exists in the bundled models/ directory."""
    if not BUNDLED_MODELS_DIR.exists():
        return False
    safe_name = model_name.replace("/", "--")
    for candidate_name in (model_name, safe_name, f"sentence-transformers--{model_name}"):
        if (BUNDLED_MODELS_DIR / candidate_name).is_dir():
            return True
    return False


def _is_in_hf_cache(model_name: str) -> bool:
    """Return True if the model exists in the HuggingFace cache."""
    try:
        from huggingface_hub import scan_cache_dir

        cache_info = scan_cache_dir()
        cached_repos = {r.repo_id for r in cache_info.repos}
        candidates = (
            model_name,
            f"sentence-transformers/{model_name}",
        )
        return any(c in cached_repos for c in candidates)
    except Exception:
        return False


def is_model_available_locally(model_name: str) -> bool:
    """Return True if *model_name* is ready to use without downloading."""
    return _is_bundled(model_name) or _is_in_hf_cache(model_name)


def get_model_status(model_name: str) -> ModelStatus:
    """Determine the current availability of a model."""
    spec = get_model_spec(model_name)

    # API models are always "available" (no download needed)
    if spec and spec.provider != ModelProvider.local:
        return ModelStatus.api

    if _is_bundled(model_name):
        return ModelStatus.bundled

    if _is_in_hf_cache(model_name):
        return ModelStatus.downloaded

    return ModelStatus.available


def get_model_disk_size(model_name: str) -> int:
    """Return actual disk size in bytes for a locally stored model, or 0."""
    # Check bundled dir
    safe_name = model_name.replace("/", "--")
    for candidate_name in (model_name, safe_name, f"sentence-transformers--{model_name}"):
        candidate = BUNDLED_MODELS_DIR / candidate_name
        if candidate.is_dir():
            return sum(f.stat().st_size for f in candidate.rglob("*") if f.is_file())

    # Check HF cache
    try:
        from huggingface_hub import scan_cache_dir

        cache_info = scan_cache_dir()
        for repo in cache_info.repos:
            if repo.repo_id in (model_name, f"sentence-transformers/{model_name}"):
                return repo.size_on_disk
    except Exception:
        pass

    return 0


def get_all_models_with_status() -> list[dict[str, Any]]:
    """Return all models with their current status — used by UI and CLI.

    Each dict contains the full ``ModelSpec`` fields plus ``status`` and
    ``disk_size_bytes``.
    """
    result = []
    for spec in EMBEDDING_MODELS + RERANKER_MODELS:
        status = get_model_status(spec.name)
        d = spec.model_dump()
        d["status"] = status.value
        d["disk_size_bytes"] = (
            get_model_disk_size(spec.name)
            if status
            in (
                ModelStatus.bundled,
                ModelStatus.downloaded,
            )
            else 0
        )
        result.append(d)
    return result


# ---------------------------------------------------------------------------
# Model download / delete
# ---------------------------------------------------------------------------

# Legacy constant — kept for backward compatibility
DEFAULT_MODELS: list[dict[str, str]] = [
    {
        "name": "paraphrase-multilingual-MiniLM-L12-v2",
        "type": "embedding",
        "description": "Default multilingual embedding model (~120 MB)",
    },
    {
        "name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "type": "reranker",
        "description": "Default cross-encoder reranker (~80 MB)",
    },
]


def download_models(
    models: list[dict[str, str]] | None = None,
    output_dir: Path | None = None,
) -> list[Path]:
    """Download models into *output_dir* for offline bundling.

    Each model is saved into a sub-directory named after the model
    (with ``/`` replaced by ``--``).

    Returns the list of directories created.
    """
    try:
        from sentence_transformers import CrossEncoder, SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required: pip install sentence-transformers"
        ) from exc

    if models is None:
        models = DEFAULT_MODELS
    if output_dir is None:
        output_dir = BUNDLED_MODELS_DIR

    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    for spec_dict in models:
        name = spec_dict["name"]
        model_type = spec_dict.get("type", "embedding")
        safe_name = name.replace("/", "--")
        dest = output_dir / safe_name

        if dest.exists():
            logger.info("Model '%s' already downloaded at %s — skipping.", name, dest)
            saved.append(dest)
            continue

        logger.info("Downloading '%s' (%s) …", name, spec_dict.get("description", model_type))

        # Determine trust_remote_code from registry
        reg_spec = get_model_spec(name)
        trust = reg_spec.trust_remote_code if reg_spec else False

        if model_type == "reranker":
            model = CrossEncoder(name, trust_remote_code=trust)
        else:
            model = SentenceTransformer(name, trust_remote_code=trust)

        model.save(str(dest))
        logger.info("Saved '%s' → %s", name, dest)
        saved.append(dest)

    return saved


def download_model_by_name(
    model_name: str,
    output_dir: Path | None = None,
    trust_remote_code: bool | None = None,
) -> Path:
    """Download a single model by its registry name.

    Returns the path to the saved model directory.

    Parameters
    ----------
    trust_remote_code : bool | None
        Override for trust_remote_code. If None, uses the registry value.
    """
    spec = get_model_spec(model_name)

    if spec and spec.provider != ModelProvider.local:
        raise ValueError(
            f"Model '{model_name}' is API-based ({spec.provider.value}) and "
            f"does not need to be downloaded."
        )

    model_type = spec.type.value if spec else "embedding"
    trust = (
        trust_remote_code
        if trust_remote_code is not None
        else (spec.trust_remote_code if spec else False)
    )
    desc = spec.description if spec else "custom model"

    try:
        from sentence_transformers import CrossEncoder, SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required: pip install sentence-transformers"
        ) from exc

    out = output_dir or BUNDLED_MODELS_DIR
    out.mkdir(parents=True, exist_ok=True)
    safe_name = model_name.replace("/", "--")
    dest = out / safe_name

    if dest.exists():
        logger.info("Model '%s' already exists at %s", model_name, dest)
        return dest

    logger.info("Downloading '%s' (%s) …", model_name, desc)
    if model_type == "reranker":
        m = CrossEncoder(model_name, trust_remote_code=trust)
    else:
        m = SentenceTransformer(model_name, trust_remote_code=trust)
    m.save(str(dest))
    logger.info("Saved '%s' → %s", model_name, dest)
    return dest


def _delete_from_hf_cache(model_name: str) -> bool:
    """Remove *model_name* from the HuggingFace hub cache.

    Returns True if anything was deleted.
    """
    try:
        from huggingface_hub import scan_cache_dir

        cache_info = scan_cache_dir()
        candidates = (model_name, f"sentence-transformers/{model_name}")
        commit_hashes: set[str] = set()
        for repo in cache_info.repos:
            if repo.repo_id in candidates:
                for rev in repo.revisions:
                    commit_hashes.add(rev.commit_hash)
        if not commit_hashes:
            return False
        delete_strategy = cache_info.delete_revisions(*commit_hashes)
        delete_strategy.execute()
        logger.info(
            "Deleted %d revision(s) of '%s' from HuggingFace cache (freed %s)",
            len(commit_hashes),
            model_name,
            delete_strategy.expected_freed_size_str,
        )
        return True
    except Exception as exc:
        logger.debug("Could not clean HF cache for '%s': %s", model_name, exc)
        return False


def delete_downloaded_model(model_name: str) -> bool:
    """Remove a downloaded model from the bundled directory **and** the
    HuggingFace cache so that status correctly shows ``available``.

    Returns True if a model was deleted from at least one location,
    False if not found anywhere.
    Does NOT delete bundled (default) models.
    """
    spec = get_model_spec(model_name)
    if spec and spec.default:
        raise ValueError(
            f"Cannot delete default model '{model_name}'. "
            f"Default models are bundled with the application."
        )

    deleted_bundled = False
    safe_name = model_name.replace("/", "--")
    for candidate_name in (model_name, safe_name, f"sentence-transformers--{model_name}"):
        candidate = BUNDLED_MODELS_DIR / candidate_name
        if candidate.is_dir():
            import shutil

            shutil.rmtree(candidate)
            logger.info("Deleted model '%s' from %s", model_name, candidate)
            deleted_bundled = True
            break

    deleted_cache = _delete_from_hf_cache(model_name)

    return deleted_bundled or deleted_cache


# ---------------------------------------------------------------------------
# Allow running as:  python -m rag_kb.models
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Download models for offline bundling")
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output directory (default: models/ in repo root)",
    )
    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List all available models with status",
    )
    args = parser.parse_args()

    if args.list:
        print(f"\n{'Name':<50} {'Type':<10} {'Dim':<6} {'Ctx':<7} {'Size':<8} {'Status'}")
        print("─" * 100)
        for info in get_all_models_with_status():
            size = f"{info['model_size_mb']}MB" if info["model_size_mb"] else "API"
            print(
                f"{info['name']:<50} {info['type']:<10} "
                f"{info['dimensions']:<6} {info['max_tokens']:<7} "
                f"{size:<8} {info['status']}"
            )
    else:
        out = Path(args.output) if args.output else None
        paths = download_models(output_dir=out)
        for p in paths:
            size_mb = sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / (1024 * 1024)
            print(f"  {p.name}  ({size_mb:.1f} MB)")
        print(f"\n{len(paths)} model(s) ready.")
