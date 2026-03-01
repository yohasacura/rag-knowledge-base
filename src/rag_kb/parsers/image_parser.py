"""Image file parser — extracts metadata, OCR text, and optional captions.

Supports common image formats via Pillow (MIT-like HPND license).

OCR is handled by one of four backends (auto-detected, in priority order):
  1. surya-ocr    (GPL-3.0) — PyTorch-native, GPU-accelerated (CUDA/MPS),
     90+ languages, auto-downloads models.  Used as the **primary**
     engine when available.
  2. rapidocr v3.x  (Apache 2.0) — self-contained, PP-OCRv5 ONNX models
     auto-downloaded, native Latin-script language support (EN, DE, FR …).
     Installed by default as a core dependency.
  3. pytesseract  (Apache 2.0) — requires system Tesseract install
  4. easyocr      (Apache 2.0) — pure Python + PyTorch, auto-downloads models

The default install uses Surya OCR (GPU) with RapidOCR (CPU) as the
primary fallback so OCR works out of the box on all platforms.

A hook point (ImageCaptioner protocol) is provided so that a local
vision model (e.g. moondream, BLIP-2, Florence-2) can be registered
later for describing diagrams, screenshots, and other non-text images.
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from pathlib import Path
from typing import Protocol, runtime_checkable

from rag_kb.parsers.base import ParsedDocument

logger = logging.getLogger(__name__)

# ── Supported image extensions ──────────────────────────────────────────

_IMAGE_EXTENSIONS: list[str] = [
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".tiff",
    ".tif",
    ".webp",
    ".ico",
    ".heic",
    ".heif",
]

# ── Vision-model captioning hook ────────────────────────────────────────


@runtime_checkable
class ImageCaptioner(Protocol):
    """Protocol for optional vision-model image captioning.

    Implement this protocol and register an instance via
    ``ImageParser.set_captioner()`` to enable Tier-2 image understanding
    (diagram / screenshot / photo descriptions).
    """

    @abstractmethod
    def caption(self, image_path: Path) -> str:
        """Return a short textual description of the image."""
        ...


# ── OCR backend auto-detection ──────────────────────────────────────────
#
# Priority order:
#   1. RapidOCR v3 — self-contained (ONNX models auto-downloaded), Apache 2.0
#      Configured with LangRec.LATIN for German/English/French/… support
#   2. pytesseract — lightweight wrapper, needs system Tesseract install
#   3. EasyOCR  — pure Python + PyTorch, auto-downloads models
#
# The default `pip install` pulls in rapidocr so OCR works
# out of the box with zero system dependencies.

import platform as _platform  # noqa: E402
import shutil as _shutil  # noqa: E402

# Well-known Tesseract install locations per platform
_TESSERACT_PATHS: list[str] = []
if _platform.system() == "Windows":
    import os as _os

    _TESSERACT_PATHS = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        _os.path.join(
            _os.environ.get("LOCALAPPDATA", ""), "Programs", "Tesseract-OCR", "tesseract.exe"
        ),
    ]
elif _platform.system() == "Darwin":
    _TESSERACT_PATHS = ["/usr/local/bin/tesseract", "/opt/homebrew/bin/tesseract"]

_tesseract_configured = False


def _configure_tesseract(pytesseract_module) -> None:
    """Auto-detect the Tesseract binary if it's not already on PATH."""
    global _tesseract_configured
    if _tesseract_configured:
        return
    _tesseract_configured = True

    if _shutil.which("tesseract"):
        return

    for candidate in _TESSERACT_PATHS:
        if Path(candidate).is_file():
            pytesseract_module.pytesseract.tesseract_cmd = candidate
            logger.debug("Tesseract found at %s", candidate)
            return

    logger.debug("Tesseract binary not found in well-known locations")


# Cache heavy OCR engine instances — protected by locks for thread-safety
import threading as _threading  # noqa: E402

_surya_lock = _threading.Lock()
_surya_rec_predictor = None
_surya_det_predictor = None
_surya_disabled = False  # Set True after unrecoverable CUDA error
_rapidocr_engine = None
_easyocr_reader = None


def _get_surya_predictors():
    """Lazy-init shared Surya OCR predictors (thread-safe).

    Surya is PyTorch-native and automatically uses the best available
    device (CUDA GPU → MPS → CPU) through the standard PyTorch device
    selection.  Models are downloaded from HuggingFace on first use.

    Raises ``RuntimeError`` if Surya has been disabled after a CUDA error.

    Returns ``(recognition_predictor, detection_predictor)``.
    """
    global _surya_rec_predictor, _surya_det_predictor
    if _surya_disabled:
        raise RuntimeError("Surya OCR disabled after CUDA error")
    with _surya_lock:
        if _surya_disabled:
            raise RuntimeError("Surya OCR disabled after CUDA error")
        if _surya_rec_predictor is None:
            from surya.detection import DetectionPredictor
            from surya.foundation import FoundationPredictor
            from surya.recognition import RecognitionPredictor

            from rag_kb.device import detect_device

            device = detect_device()
            logger.info("Initialising Surya OCR (device=%s) …", device)
            foundation = FoundationPredictor(device=device)
            _surya_rec_predictor = RecognitionPredictor(foundation)
            _surya_det_predictor = DetectionPredictor(device=device)
            logger.info("Surya OCR ready (device=%s).", device)
    return _surya_rec_predictor, _surya_det_predictor


def _disable_surya(reason: str) -> None:
    """Permanently disable Surya for this process after a CUDA error.

    Once a CUDA device-side assert fires the entire CUDA context is
    poisoned — all subsequent CUDA calls will fail.  Rather than
    retrying (and logging hundreds of identical errors), we fall
    through immediately to the CPU-based RapidOCR backend.
    """
    global _surya_disabled, _surya_rec_predictor, _surya_det_predictor
    if _surya_disabled:
        return
    _surya_disabled = True
    _surya_rec_predictor = None
    _surya_det_predictor = None
    logger.warning(
        "Surya OCR disabled for this session due to CUDA error: %s  "
        "— falling back to RapidOCR (CPU). Restart the daemon to retry GPU.",
        reason,
    )


def _get_rapidocr_engine():
    """Lazy-init a shared RapidOCR v3 engine with Latin-script support.

    Uses PP-OCRv5 models with ``LangRec.LATIN`` so that German umlauts
    (ä ö ü ß), French accents, and other Latin-script characters are
    recognised correctly alongside English.  Models are auto-downloaded
    on first use and cached locally by the ``rapidocr`` package.

    GPU note: RapidOCR's torch backend does not support PP-OCRv5 + Latin
    for the recognition stage, so we always use ONNX Runtime here.  If
    ``onnxruntime-gpu`` is installed, the CUDAExecutionProvider is
    requested automatically.  Otherwise ONNX Runtime runs on CPU (which
    is already fast for OCR workloads).
    """
    global _rapidocr_engine
    if _rapidocr_engine is None:
        from rapidocr import LangRec, OCRVersion, RapidOCR

        from rag_kb.device import onnxruntime_has_cuda

        params: dict = {
            "Rec.lang_type": LangRec.LATIN,
            "Rec.ocr_version": OCRVersion.PPOCRV5,
        }

        # If onnxruntime-gpu is installed, prefer CUDA with CPU fallback.
        if onnxruntime_has_cuda():
            for key in (
                "Det.onnxruntime.providers",
                "Cls.onnxruntime.providers",
                "Rec.onnxruntime.providers",
            ):
                params[key] = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            logger.info("RapidOCR will use ONNX Runtime with CUDA provider.")
        else:
            logger.debug("RapidOCR will use ONNX Runtime on CPU.")

        _rapidocr_engine = RapidOCR(params=params)
    return _rapidocr_engine


def _get_easyocr_reader():
    """Lazy-init a shared EasyOCR reader (English by default).

    Automatically enables GPU when a CUDA device is available.
    """
    global _easyocr_reader
    if _easyocr_reader is None:
        import easyocr

        from rag_kb.device import is_cuda_available

        use_gpu = is_cuda_available()
        logger.info("EasyOCR will use %s.", "GPU" if use_gpu else "CPU")
        _easyocr_reader = easyocr.Reader(["en"], gpu=use_gpu)
    return _easyocr_reader


def _ocr_extract(image_path: Path) -> tuple[str, str]:
    """Try to extract text from an image file via OCR.

    Thin wrapper around :func:`ocr_image` that opens the file first.

    Returns ``(text, engine_name)`` or ``("", "none")`` if no engine
    is available.
    """
    try:
        from PIL import Image as _PILImage

        with _PILImage.open(image_path) as raw_img:
            img = raw_img.convert("RGB")
    except Exception as exc:
        logger.debug("Cannot open image %s: %s", image_path, exc)
        return "", "none"
    return ocr_image(img, label=str(image_path))


def _is_cuda_error(exc: Exception) -> bool:
    """Return True if *exc* is a CUDA device-side assert or similar fatal error."""
    msg = str(exc).lower()
    return any(kw in msg for kw in ("cuda error", "device-side assert", "cublas", "cudnn"))


def ocr_image(
    image: PIL.Image.Image,  # noqa: F821 — forward ref to avoid top-level import
    *,
    label: str = "<embedded image>",
) -> tuple[str, str]:
    """Run OCR on an **in-memory PIL Image**.

    This is the public entry-point that other parsers (PDF, DOCX, PPTX)
    should call to OCR embedded images while preserving document order.

    Parameters
    ----------
    image:
        A PIL ``Image`` already opened / decoded.
    label:
        Human-readable identifier for logging (e.g. ``"page-3/img-1"``).

    Returns
    -------
    ``(text, engine_name)`` or ``("", "none")`` when no engine succeeds.
    """
    import numpy as np

    image = image.convert("RGB")

    # --- 1. Surya OCR (PyTorch, GPU-accelerated, 90+ languages) ---------
    if not _surya_disabled:
        try:
            rec_pred, det_pred = _get_surya_predictors()
            # Serialise all Surya CUDA calls — PyTorch CUDA is not
            # thread-safe and concurrent access causes device-side asserts.
            with _surya_lock:
                predictions = rec_pred([image], det_predictor=det_pred)
            if predictions:
                lines = [ln.text for ln in predictions[0].text_lines if ln.text]
                text = "\n".join(lines)
                return text.strip(), "surya"
            return "", "surya"

        except ImportError:
            pass
        except Exception as exc:
            if _is_cuda_error(exc):
                _disable_surya(str(exc))
            else:
                logger.debug("Surya OCR failed for %s: %s", label, exc)

    # --- 2. RapidOCR v3 (accepts numpy array) ---------------------------
    try:
        engine = _get_rapidocr_engine()
        arr = np.array(image)
        result = engine(arr)
        if result and result.txts:
            text = "\n".join(result.txts)
            return text.strip(), "rapidocr"
        return "", "rapidocr"

    except ImportError:
        pass
    except Exception as exc:
        logger.debug("RapidOCR failed for %s: %s", label, exc)

    # --- 3. pytesseract (accepts PIL Image directly) --------------------
    try:
        import pytesseract

        _configure_tesseract(pytesseract)
        text = pytesseract.image_to_string(image)
        return text.strip(), "tesseract"

    except ImportError:
        pass
    except Exception as exc:
        logger.debug("pytesseract failed for %s: %s", label, exc)

    # --- 4. EasyOCR (accepts numpy array) -------------------------------
    try:
        reader = _get_easyocr_reader()
        arr = np.array(image)
        results = reader.readtext(arr, detail=0)
        text = "\n".join(results)
        return text.strip(), "easyocr"

    except ImportError:
        pass
    except Exception as exc:
        logger.debug("easyocr failed for %s: %s", label, exc)

    return "", "none"


def ocr_images_batch(
    images: list[PIL.Image.Image],  # noqa: F821
    *,
    labels: list[str] | None = None,
) -> list[tuple[str, str]]:
    """Batch-OCR multiple images — significantly faster on GPU.

    Surya OCR (the primary engine) accepts a list of images and
    processes them as a single GPU batch, amortising kernel-launch
    overhead and maximising tensor-core utilisation.

    Falls back to per-image :func:`ocr_image` for non-Surya backends.

    Returns a list of ``(text, engine_name)`` in the same order as *images*.
    """
    if not images:
        return []

    if labels is None:
        labels = [f"img-{i}" for i in range(len(images))]

    rgb_images = [img.convert("RGB") for img in images]

    # --- Try Surya batch (GPU) ------------------------------------------
    if not _surya_disabled:
        try:
            rec_pred, det_pred = _get_surya_predictors()
            with _surya_lock:
                predictions = rec_pred(rgb_images, det_predictor=det_pred)
            results: list[tuple[str, str]] = []
            for pred in predictions:
                lines = [ln.text for ln in pred.text_lines if ln.text]
                results.append(("\n".join(lines).strip(), "surya"))
            return results
        except ImportError:
            pass
        except Exception as exc:
            if _is_cuda_error(exc):
                _disable_surya(str(exc))
            else:
                logger.debug("Surya batch OCR failed: %s", exc)

    # --- Fallback: per-image with non-Surya engines ---------------------
    return [ocr_image(img, label=lbl) for img, lbl in zip(rgb_images, labels, strict=False)]


# ── Metadata extraction ────────────────────────────────────────────────


def _extract_metadata(image_path: Path) -> dict[str, str]:
    """Extract image metadata using Pillow."""
    meta: dict[str, str] = {"format": "image"}

    try:
        from PIL import Image
        from PIL.ExifTags import TAGS

        with Image.open(image_path) as img:
            meta["image_format"] = img.format or image_path.suffix.lstrip(".")
            meta["width"] = str(img.width)
            meta["height"] = str(img.height)
            meta["mode"] = img.mode  # RGB, RGBA, L, etc.

            # Extract EXIF if present
            exif_data = img.getexif()
            if exif_data:
                interesting_tags = {
                    "Make",
                    "Model",
                    "DateTime",
                    "DateTimeOriginal",
                    "Software",
                    "ImageDescription",
                    "Artist",
                    "Copyright",
                    "ExifImageWidth",
                    "ExifImageHeight",
                }
                for tag_id, value in exif_data.items():
                    tag_name = TAGS.get(tag_id, str(tag_id))
                    if tag_name in interesting_tags:
                        meta[f"exif_{tag_name}"] = str(value)

    except ImportError:
        logger.warning("Pillow is required for image metadata: pip install Pillow")
    except Exception as exc:
        logger.debug("Metadata extraction failed for %s: %s", image_path, exc)

    return meta


# ── Main parser ─────────────────────────────────────────────────────────


class ImageParser:
    """Parse image files — metadata + OCR text + optional vision-model caption.

    Gracefully degrades:
    - With OCR backend:  metadata + extracted text
    - Without OCR:       metadata only (dimensions, format, EXIF)
    - With captioner:    all of the above + AI-generated description
    """

    supported_extensions: list[str] = _IMAGE_EXTENSIONS

    # Class-level captioner — set once, shared across calls
    _captioner: ImageCaptioner | None = None

    @classmethod
    def set_captioner(cls, captioner: ImageCaptioner) -> None:
        """Register a vision-model captioner for Tier-2 image understanding.

        Example usage::

            from rag_kb.parsers.image_parser import ImageParser, ImageCaptioner

            class MoondreamCaptioner:
                def caption(self, image_path: Path) -> str:
                    # ... load moondream2 and generate caption ...
                    return description

            ImageParser.set_captioner(MoondreamCaptioner())
        """
        cls._captioner = captioner
        logger.info("Image captioner registered: %s", type(captioner).__name__)

    def parse(self, file_path: Path) -> ParsedDocument:
        # 1. Metadata (always)
        metadata = _extract_metadata(file_path)

        sections: list[str] = []

        # 2. OCR text extraction (if an engine is available)
        ocr_text, engine = _ocr_extract(file_path)
        metadata["ocr_engine"] = engine

        if ocr_text:
            metadata["ocr_chars"] = str(len(ocr_text))
            sections.append(f"[OCR Text]\n{ocr_text}")

        # 3. Vision-model caption (if a captioner is registered)
        if self._captioner is not None:
            try:
                caption = self._captioner.caption(file_path)
                if caption and caption.strip():
                    metadata["has_caption"] = "true"
                    sections.append(f"[Image Description]\n{caption.strip()}")
            except Exception as exc:
                logger.warning("Captioner failed for %s: %s", file_path, exc)

        # 4. EXIF description (some images have text baked into metadata)
        exif_desc = metadata.get("exif_ImageDescription", "")
        if exif_desc:
            sections.append(f"[EXIF Description]\n{exif_desc}")

        # 5. Build final text
        if sections:
            text = "\n\n".join(sections)
        else:
            # No OCR text and no caption — produce a minimal summary
            dims = f"{metadata.get('width', '?')}x{metadata.get('height', '?')}"
            fmt = metadata.get("image_format", file_path.suffix)
            text = f"[Image: {file_path.name} — {fmt}, {dims}]"
            metadata["text_extraction"] = "metadata_only"

        return ParsedDocument(
            text=text,
            source_path=str(file_path),
            metadata=metadata,
        )
