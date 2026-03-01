"""EPUB ebook parser — extracts text from XHTML content files inside the EPUB archive."""

from __future__ import annotations

import logging
import re
import zipfile
from pathlib import Path

from rag_kb.parsers.base import ParsedDocument

logger = logging.getLogger(__name__)

_RE_HTML_TAGS = re.compile(r"<[^>]+>")
_RE_WHITESPACE = re.compile(r"\s+")
_RE_MULTI_NEWLINES = re.compile(r"\n{3,}")


class EpubParser:
    """Parse EPUB ebook files, extracting text content from all chapters."""

    supported_extensions: list[str] = [".epub"]

    def parse(self, file_path: Path) -> ParsedDocument:
        try:
            with zipfile.ZipFile(str(file_path), "r") as zf:
                text = _extract_epub_text(zf)
        except zipfile.BadZipFile:
            logger.warning("Invalid EPUB/ZIP file: %s", file_path)
            return ParsedDocument(text="", source_path=str(file_path), metadata={"format": "epub"})

        return ParsedDocument(
            text=text,
            source_path=str(file_path),
            metadata={"format": "epub"},
        )


def _extract_epub_text(zf: zipfile.ZipFile) -> str:
    """Extract text from all XHTML/HTML content files in the EPUB archive."""
    content_files = [
        name
        for name in zf.namelist()
        if name.endswith((".xhtml", ".html", ".htm", ".xml"))
        and "META-INF" not in name
        and "mimetype" not in name
    ]

    # Sort to maintain chapter order (heuristic, often alphabetical = chapter order)
    content_files.sort()

    chapters: list[str] = []

    for fname in content_files:
        try:
            raw = zf.read(fname).decode("utf-8", errors="replace")
            text = _html_to_text(raw)
            if text.strip():
                chapters.append(text.strip())
        except Exception as exc:
            logger.debug("Skipping EPUB entry %s: %s", fname, exc)

    return "\n\n".join(chapters)


def _html_to_text(html: str) -> str:
    """Convert HTML/XHTML to plain text."""
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "lxml")
        for tag in soup(["script", "style"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
    except Exception:
        # Fallback: strip tags
        text = _RE_HTML_TAGS.sub(" ", html)
        text = _RE_WHITESPACE.sub(" ", text).strip()

    text = _RE_MULTI_NEWLINES.sub("\n\n", text)
    return text
