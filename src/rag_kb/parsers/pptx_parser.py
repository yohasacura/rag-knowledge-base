"""PPTX file parser using python-pptx (MIT license)."""

from __future__ import annotations

import logging
from pathlib import Path

from rag_kb.parsers.base import DocumentParser, ParsedDocument

logger = logging.getLogger(__name__)


class PptxParser:
    """Parse Microsoft PowerPoint .pptx files."""

    supported_extensions: list[str] = [".pptx"]

    def parse(self, file_path: Path) -> ParsedDocument:
        try:
            from pptx import Presentation
        except ImportError as exc:
            raise ImportError("python-pptx is required for PPTX parsing: pip install python-pptx") from exc

        prs = Presentation(str(file_path))
        slides_text: list[str] = []

        for slide_num, slide in enumerate(prs.slides, 1):
            slide_parts: list[str] = [f"[Slide {slide_num}]"]
            for shape in slide.shapes:
                if shape.has_text_frame:
                    for paragraph in shape.text_frame.paragraphs:
                        text = paragraph.text.strip()
                        if text:
                            slide_parts.append(text)
                if shape.has_table:
                    for row in shape.table.rows:
                        row_text = " | ".join(
                            cell.text.strip() for cell in row.cells if cell.text.strip()
                        )
                        if row_text:
                            slide_parts.append(row_text)
            if len(slide_parts) > 1:  # more than just the slide marker
                slides_text.append("\n".join(slide_parts))

        text = "\n\n".join(slides_text)
        return ParsedDocument(
            text=text,
            source_path=str(file_path),
            metadata={
                "format": "pptx",
                "slide_count": str(len(prs.slides)),
            },
        )
