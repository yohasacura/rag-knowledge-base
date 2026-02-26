"""Image file parser — extracts metadata, OCR text, and optional captions.

Supports common image formats via Pillow (MIT-like HPND license).

OCR is handled by one of three backends (auto-detected, in priority order):
  1. rapidocr v3.x  (Apache 2.0) — **self-contained**, PP-OCRv5 ONNX models
     auto-downloaded, native Latin-script language support (EN, DE, FR …).
     Installed by default as a core dependency.
  2. pytesseract  (Apache 2.0) — requires system Tesseract install
  3. easyocr      (Apache 2.0) — pure Python + PyTorch, auto-downloads models

The default install uses RapidOCR so OCR works out of the box on all platforms
with zero configuration.

A hook point (ImageCaptioner protocol) is provided so that a local
vision model (e.g. moondream, BLIP-2, Florence-2) can be registered
later for describing diagrams, screenshots, and other non-text images.
"""

from __future__ import annotations

import logging
import re
from abc import abstractmethod
from pathlib import Path
from typing import Protocol, runtime_checkable

from rag_kb.parsers.base import DocumentParser, ParsedDocument

logger = logging.getLogger(__name__)

# ── Supported image extensions ──────────────────────────────────────────

_IMAGE_EXTENSIONS: list[str] = [
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".tif",
    ".webp", ".ico", ".heic", ".heif",
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

import platform as _platform
import shutil as _shutil

# Well-known Tesseract install locations per platform
_TESSERACT_PATHS: list[str] = []
if _platform.system() == "Windows":
    import os as _os
    _TESSERACT_PATHS = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        _os.path.join(_os.environ.get("LOCALAPPDATA", ""), "Programs", "Tesseract-OCR", "tesseract.exe"),
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


# Cache heavy OCR engine instances
_rapidocr_engine = None
_easyocr_reader = None


def _get_rapidocr_engine():
    """Lazy-init a shared RapidOCR v3 engine with Latin-script support.

    Uses PP-OCRv5 models with ``LangRec.LATIN`` so that German umlauts
    (ä ö ü ß), French accents, and other Latin-script characters are
    recognised correctly alongside English.  Models are auto-downloaded
    on first use and cached locally by the ``rapidocr`` package.
    """
    global _rapidocr_engine
    if _rapidocr_engine is None:
        from rapidocr import RapidOCR, LangRec, OCRVersion
        _rapidocr_engine = RapidOCR(
            params={
                "Rec.lang_type": LangRec.LATIN,
                "Rec.ocr_version": OCRVersion.PPOCRV5,
            }
        )
    return _rapidocr_engine


def _get_easyocr_reader():
    """Lazy-init a shared EasyOCR reader (English by default)."""
    global _easyocr_reader
    if _easyocr_reader is None:
        import easyocr
        _easyocr_reader = easyocr.Reader(["en"], gpu=False)
    return _easyocr_reader


def _ocr_extract(image_path: Path) -> tuple[str, str]:
    """Try to extract text from an image via OCR.

    Returns ``(text, engine_name)`` or ``("", "none")`` if no engine
    is available.
    """

    # --- 1. RapidOCR v3 (self-contained, Latin-script, PP-OCRv5 models) ---
    try:
        engine = _get_rapidocr_engine()
        result = engine(str(image_path))
        if result and result.txts:
            text = "\n".join(result.txts)
            return text.strip(), "rapidocr"
        return "", "rapidocr"  # engine ran but found no text

    except ImportError:
        pass  # rapidocr not installed
    except Exception as exc:
        logger.debug("RapidOCR failed for %s: %s", image_path, exc)

    # --- 2. pytesseract (needs system Tesseract) ---
    try:
        import pytesseract  # Apache 2.0
        from PIL import Image

        _configure_tesseract(pytesseract)
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img)
        return text.strip(), "tesseract"

    except ImportError:
        pass  # pytesseract not installed
    except Exception as exc:
        # Tesseract binary not found / other runtime error
        logger.debug("pytesseract failed for %s: %s", image_path, exc)

    # --- 3. EasyOCR (pure Python + PyTorch, auto-downloads models) ---
    try:
        reader = _get_easyocr_reader()
        results = reader.readtext(str(image_path), detail=0)
        text = "\n".join(results)
        return text.strip(), "easyocr"

    except ImportError:
        pass
    except Exception as exc:
        logger.debug("easyocr failed for %s: %s", image_path, exc)

    return "", "none"


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
                    "Make", "Model", "DateTime", "DateTimeOriginal",
                    "Software", "ImageDescription", "Artist",
                    "Copyright", "ExifImageWidth", "ExifImageHeight",
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
