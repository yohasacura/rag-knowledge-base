"""PDF file parser using pypdf (BSD license).

Extracts both text *and* embedded images (via OCR) from each page,
preserving document order so that image-based content (scanned pages,
diagrams with labels, charts) is interleaved with the surrounding text.

Performance
-----------
Images are collected across **all** pages first, then OCR'd in a single
GPU batch via :func:`~rag_kb.parsers.image_parser.ocr_images_batch`.
This amortises CUDA kernel-launch overhead and keeps the GPU busy with
a large tensor batch instead of hundreds of tiny single-image calls.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass
from pathlib import Path

from rag_kb.parsers.base import ParsedDocument

logger = logging.getLogger(__name__)

# Maximum images to batch-OCR in one GPU call (limits VRAM usage)
_OCR_BATCH_SIZE = 32

# If a page already has more than this many characters of extracted text,
# skip OCR for that page's images — they are almost certainly decorative
# (logos, charts with text already in the page stream, etc.).
# Set to 0 to always OCR all images.
_SKIP_OCR_TEXT_THRESHOLD = 200

# Maximum total images to collect per PDF for OCR.  Large PDFs can contain
# thousands of embedded images (icons, bullets, decorative graphics) that
# blow up RAM when decoded as bitmaps.  Once we hit this limit we stop
# collecting and rely on the text extractable from remaining pages.
_MAX_IMAGES_PER_PDF = 200


@dataclass
class _PendingImage:
    """Tracks an image awaiting batch OCR."""

    page_idx: int  # 0-based page index
    img_idx: int  # index within the page's image list
    pil_image: object  # PIL.Image.Image


class PdfParser:
    """Parse PDF files using pypdf."""

    supported_extensions: list[str] = [".pdf"]

    def parse(self, file_path: Path) -> ParsedDocument:
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise ImportError("pypdf is required for PDF parsing: pip install pypdf") from exc

        reader = PdfReader(str(file_path))

        # ── Phase 1: extract text + collect images ──────────────────
        page_texts: list[str] = []  # per-page extracted text
        pending_images: list[_PendingImage] = []  # images awaiting OCR

        for i, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            page_texts.append(page_text)

            # Skip image OCR on pages with plenty of extractable text —
            # images on such pages are almost always decorative (logos,
            # inline figures whose labels are already in the text stream).
            if _SKIP_OCR_TEXT_THRESHOLD and len(page_text.strip()) >= _SKIP_OCR_TEXT_THRESHOLD:
                continue

            # Stop collecting images once we hit the per-PDF cap.
            if len(pending_images) >= _MAX_IMAGES_PER_PDF:
                continue

            # Collect images for batch OCR
            try:
                self._collect_page_images(page, i, pending_images)
            except Exception as exc:
                logger.debug("Image extraction failed on page %d of %s: %s", i + 1, file_path, exc)

        # Save page count + title before releasing the reader.
        n_pages = len(reader.pages)
        title = ""
        try:
            info = reader.metadata
            if info and info.title:
                title = str(info.title).strip()
        except Exception:
            pass
        if not title:
            title = file_path.stem.replace("-", " ").replace("_", " ").title()
        del reader  # release the PDF file buffer (~file-size of RAM)

        # ── Phase 2: batch-OCR all collected images ─────────────────
        # Maps (page_idx, img_idx) → OCR text
        ocr_results: dict[tuple[int, int], str] = {}
        total_images_ocrd = 0

        if pending_images:
            ocr_results, total_images_ocrd = self._batch_ocr(pending_images)
        del pending_images  # release any remaining PIL image references

        # ── Phase 3: assemble final document ────────────────────────
        pages: list[str] = []
        for i, page_text in enumerate(page_texts):
            page_parts: list[str] = []

            if page_text.strip():
                page_parts.append(page_text)

            # Append OCR results for this page's images (in order)
            page_img_indices = sorted(
                (key for key in ocr_results if key[0] == i),
                key=lambda k: k[1],
            )
            for key in page_img_indices:
                label = f"page-{i + 1}/img-{key[1]}"
                page_parts.append(f"[Image OCR — {label}]\n{ocr_results[key]}")

            if page_parts:
                pages.append(f"[PAGE {i + 1}]\n" + "\n\n".join(page_parts))

        text = "\n\n".join(pages)

        metadata: dict[str, str] = {
            "format": "pdf",
            "page_count": str(n_pages),
        }
        if total_images_ocrd:
            metadata["images_ocrd"] = str(total_images_ocrd)

        return ParsedDocument(
            text=text,
            source_path=str(file_path),
            metadata=metadata,
            title=title,
            format_hint="pdf",
        )

    # ------------------------------------------------------------------
    # Image collection (phase 1)
    # ------------------------------------------------------------------

    @staticmethod
    def _collect_page_images(
        page,
        page_idx: int,
        pending: list[_PendingImage],
    ) -> None:
        """Extract PIL images from *page* and append to *pending*."""
        from PIL import Image as PILImage

        try:
            images = page.images
        except Exception:
            return

        for idx, img_file in enumerate(images):
            try:
                pil_img = PILImage.open(io.BytesIO(img_file.data)).convert("RGB")

                # Skip tiny images (icons, bullets, decorative)
                if pil_img.width < 50 or pil_img.height < 50:
                    continue

                pending.append(
                    _PendingImage(
                        page_idx=page_idx,
                        img_idx=idx,
                        pil_image=pil_img,
                    )
                )
            except Exception as exc:
                logger.debug("Cannot open image %d on page %d: %s", idx, page_idx + 1, exc)

    # ------------------------------------------------------------------
    # Batch OCR (phase 2)
    # ------------------------------------------------------------------

    @staticmethod
    def _batch_ocr(
        pending: list[_PendingImage],
    ) -> tuple[dict[tuple[int, int], str], int]:
        """OCR all pending images in GPU-friendly batches.

        Returns ``(results_dict, total_images_with_text)``.
        """
        from rag_kb.parsers.image_parser import ocr_images_batch

        results: dict[tuple[int, int], str] = {}
        count = 0

        for batch_start in range(0, len(pending), _OCR_BATCH_SIZE):
            batch = pending[batch_start : batch_start + _OCR_BATCH_SIZE]
            images = [p.pil_image for p in batch]
            labels = [f"page-{p.page_idx + 1}/img-{p.img_idx}" for p in batch]

            ocr_out = ocr_images_batch(images, labels=labels)

            for pend, (text, _engine) in zip(batch, ocr_out, strict=False):
                # Release the PIL image immediately — it can be tens of MB
                # of uncompressed bitmap and we no longer need it.
                pend.pil_image = None  # type: ignore[assignment]
                if text:
                    results[(pend.page_idx, pend.img_idx)] = text
                    count += 1

        logger.debug("Batch OCR: %d/%d images produced text", count, len(pending))
        return results, count
