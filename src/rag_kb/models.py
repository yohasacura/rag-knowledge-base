"""Bundled model management for self-contained packaging.

This module handles:
  1. Pre-downloading models for offline/bundled distribution.
  2. Resolving model paths — preferring a bundled ``models/`` directory
     (inside a PyInstaller bundle or repo) over the HuggingFace cache.

Models used by the application
------------------------------
- **Embedding**: ``paraphrase-multilingual-MiniLM-L12-v2``
  (configurable via ``AppSettings.embedding_model``)
- **Reranker** : ``cross-encoder/ms-marco-MiniLM-L-6-v2``
  (configurable via ``AppSettings.reranker_model``)
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Where bundled models live
# ---------------------------------------------------------------------------

def _get_bundled_models_dir() -> Path:
    """Return the ``models/`` directory, whether running from source or a
    PyInstaller bundle."""
    # PyInstaller sets sys._MEIPASS to the temp extraction folder
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        base = Path(sys._MEIPASS)
    else:
        # Running from source — look relative to repo root
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

    # Try exact match first (e.g. "paraphrase-multilingual-MiniLM-L12-v2")
    candidate = BUNDLED_MODELS_DIR / model_name
    if candidate.is_dir():
        logger.info("Using bundled model at %s", candidate)
        return str(candidate)

    # Try with '/' → '--' replacement (e.g. "cross-encoder/ms-marco-MiniLM-L-6-v2"
    #   → "cross-encoder--ms-marco-MiniLM-L-6-v2")
    safe_name = model_name.replace("/", "--")
    candidate = BUNDLED_MODELS_DIR / safe_name
    if candidate.is_dir():
        logger.info("Using bundled model at %s", candidate)
        return str(candidate)

    # Try with "sentence-transformers/" prefix (HF repo convention)
    st_name = f"sentence-transformers--{model_name}"
    candidate = BUNDLED_MODELS_DIR / st_name
    if candidate.is_dir():
        logger.info("Using bundled model at %s", candidate)
        return str(candidate)

    return model_name


# ---------------------------------------------------------------------------
# Pre-download helper  (used by build scripts / CLI)
# ---------------------------------------------------------------------------

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
        from sentence_transformers import SentenceTransformer, CrossEncoder
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

    for spec in models:
        name = spec["name"]
        model_type = spec.get("type", "embedding")
        safe_name = name.replace("/", "--")
        dest = output_dir / safe_name

        if dest.exists():
            logger.info("Model '%s' already downloaded at %s — skipping.", name, dest)
            saved.append(dest)
            continue

        logger.info("Downloading '%s' (%s) …", name, spec.get("description", model_type))

        if model_type == "reranker":
            model = CrossEncoder(name, trust_remote_code=False)
        else:
            model = SentenceTransformer(name, trust_remote_code=False)

        model.save(str(dest))
        logger.info("Saved '%s' → %s", name, dest)
        saved.append(dest)

    return saved


# ---------------------------------------------------------------------------
# Allow running as:  python -m rag_kb.models
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Download models for offline bundling")
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output directory (default: models/ in repo root)",
    )
    args = parser.parse_args()

    out = Path(args.output) if args.output else None
    paths = download_models(output_dir=out)
    for p in paths:
        size_mb = sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / (1024 * 1024)
        print(f"  {p.name}  ({size_mb:.1f} MB)")
    print(f"\n{len(paths)} model(s) ready.")
