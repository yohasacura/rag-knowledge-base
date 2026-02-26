"""PDF file parser using pypdf (BSD license)."""

from __future__ import annotations

import logging
from pathlib import Path

from rag_kb.parsers.base import DocumentParser, ParsedDocument

logger = logging.getLogger(__name__)


class PdfParser:
    """Parse PDF files using pypdf."""

    supported_extensions: list[str] = [".pdf"]

    def parse(self, file_path: Path) -> ParsedDocument:
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise ImportError("pypdf is required for PDF parsing: pip install pypdf") from exc

        reader = PdfReader(str(file_path))
        pages: list[str] = []
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            if page_text.strip():
                pages.append(f"[PAGE {i + 1}]\n{page_text}")

        text = "\n\n".join(pages)

        # Try to extract title from PDF metadata
        title = ""
        try:
            info = reader.metadata
            if info and info.title:
                title = str(info.title).strip()
        except Exception:
            pass
        if not title:
            title = file_path.stem.replace("-", " ").replace("_", " ").title()

        return ParsedDocument(
            text=text,
            source_path=str(file_path),
            metadata={
                "format": "pdf",
                "page_count": str(len(reader.pages)),
            },
            title=title,
            format_hint="pdf",
        )
