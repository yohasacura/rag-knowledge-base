"""OpenDocument format parsers (ODT, ODS, ODP) using zipfile + XML extraction."""

from __future__ import annotations

import logging
import re
import zipfile
from pathlib import Path

from rag_kb.parsers.base import DocumentParser, ParsedDocument

logger = logging.getLogger(__name__)


class OdtParser:
    """Parse OpenDocument Text (.odt) files."""

    supported_extensions: list[str] = [".odt"]

    def parse(self, file_path: Path) -> ParsedDocument:
        text = _extract_odf_text(file_path)
        return ParsedDocument(
            text=text,
            source_path=str(file_path),
            metadata={"format": "odt"},
        )


class OdsParser:
    """Parse OpenDocument Spreadsheet (.ods) files."""

    supported_extensions: list[str] = [".ods"]

    def parse(self, file_path: Path) -> ParsedDocument:
        text = _extract_odf_text(file_path)
        return ParsedDocument(
            text=text,
            source_path=str(file_path),
            metadata={"format": "ods"},
        )


class OdpParser:
    """Parse OpenDocument Presentation (.odp) files."""

    supported_extensions: list[str] = [".odp"]

    def parse(self, file_path: Path) -> ParsedDocument:
        text = _extract_odf_text(file_path)
        return ParsedDocument(
            text=text,
            source_path=str(file_path),
            metadata={"format": "odp"},
        )


def _extract_odf_text(file_path: Path) -> str:
    """Extract text from an OpenDocument format file (ODF).

    ODF files are ZIP archives containing content.xml with the document content.
    We parse the XML and extract all text:p and text:h elements.
    """
    try:
        with zipfile.ZipFile(str(file_path), "r") as zf:
            if "content.xml" not in zf.namelist():
                logger.warning("No content.xml found in %s", file_path)
                return ""
            content_xml = zf.read("content.xml").decode("utf-8", errors="replace")
    except zipfile.BadZipFile:
        logger.warning("Invalid ZIP/ODF file: %s", file_path)
        return ""

    # Try lxml first for namespace-aware parsing
    try:
        from lxml import etree

        root = etree.fromstring(content_xml.encode("utf-8"))
        ns = {
            "text": "urn:oasis:names:tc:opendocument:xmlns:text:1.0",
            "table": "urn:oasis:names:tc:opendocument:xmlns:table:1.0",
        }
        paragraphs: list[str] = []

        # Extract text paragraphs and headings
        for elem in root.iter(
            "{urn:oasis:names:tc:opendocument:xmlns:text:1.0}p",
            "{urn:oasis:names:tc:opendocument:xmlns:text:1.0}h",
        ):
            text = "".join(elem.itertext()).strip()
            if text:
                paragraphs.append(text)

        # Extract table cells
        for cell in root.iter(
            "{urn:oasis:names:tc:opendocument:xmlns:table:1.0}table-cell"
        ):
            cell_text = "".join(cell.itertext()).strip()
            if cell_text:
                paragraphs.append(cell_text)

        return "\n\n".join(paragraphs)

    except Exception:
        # Fallback: crude XML tag stripping
        text = re.sub(r"<[^>]+>", " ", content_xml)
        text = re.sub(r"\s+", " ", text).strip()
        return text
