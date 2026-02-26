"""DOCX file parser using python-docx (MIT license)."""

from __future__ import annotations

import logging
from pathlib import Path

from rag_kb.parsers.base import DocumentParser, ParsedDocument

logger = logging.getLogger(__name__)


class DocxParser:
    """Parse Microsoft Word .docx files."""

    supported_extensions: list[str] = [".docx"]

    def parse(self, file_path: Path) -> ParsedDocument:
        try:
            from docx import Document
        except ImportError as exc:
            raise ImportError("python-docx is required for DOCX parsing: pip install python-docx") from exc

        doc = Document(str(file_path))
        paragraphs: list[str] = []

        for para in doc.paragraphs:
            text = para.text.strip()
            if text:
                paragraphs.append(text)

        # Also extract text from tables
        for table in doc.tables:
            for row in table.rows:
                row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                if row_text:
                    paragraphs.append(row_text)

        text = "\n\n".join(paragraphs)
        return ParsedDocument(
            text=text,
            source_path=str(file_path),
            metadata={
                "format": "docx",
                "paragraph_count": str(len(doc.paragraphs)),
            },
        )
