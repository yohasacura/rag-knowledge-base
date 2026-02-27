"""Centralized GPU / device detection for all backends.

Every component that needs to decide between GPU and CPU should call the
helpers here so that detection logic lives in one place and logging is
consistent.
"""

from __future__ import annotations

import logging
from functools import lru_cache

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def detect_device() -> str:
    """Auto-detect the best available compute device.

    Returns ``"cuda"`` when an NVIDIA GPU with CUDA is available,
    ``"mps"`` on Apple Silicon with Metal Performance Shaders, or
    ``"cpu"`` as the fallback.
    """
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            logger.info("CUDA device detected (%s) — using GPU acceleration.", name)
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("Apple MPS device detected — using GPU acceleration.")
            return "mps"
    except ImportError:
        logger.debug("PyTorch not installed — falling back to CPU.")
    return "cpu"


def is_cuda_available() -> bool:
    """Return *True* if a CUDA-capable GPU is available."""
    return detect_device() == "cuda"


@lru_cache(maxsize=1)
def onnxruntime_has_cuda() -> bool:
    """Return *True* if the installed ``onnxruntime`` package exposes CUDA.

    This is the case when ``onnxruntime-gpu`` (or the full
    ``onnxruntime-directml`` / ``onnxruntime-cuda``) has been installed
    *and* the CUDA toolkit is visible to it at runtime.
    """
    try:
        import onnxruntime as ort

        providers = ort.get_available_providers()
        if "CUDAExecutionProvider" in providers:
            logger.info(
                "ONNX Runtime CUDA provider available — "
                "RapidOCR / ONNX models will use GPU."
            )
            return True
    except ImportError:
        pass
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not query ONNX Runtime providers: %s", exc)
    return False
