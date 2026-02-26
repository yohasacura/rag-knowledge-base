"""Markdown file parser.

Keeps headings intact so the structure-aware chunker can split by sections.
Only strips formatting noise (images, inline markup, HTML tags).
"""

from __future__ import annotations

import re
from pathlib import Path

from rag_kb.parsers.base import DocumentParser, ParsedDocument


class MarkdownParser:
    """Parse Markdown files, preserving headings for structure-aware chunking."""

    supported_extensions: list[str] = [".md", ".markdown"]

    def parse(self, file_path: Path) -> ParsedDocument:
        text = file_path.read_text(encoding="utf-8", errors="replace")
        title = _extract_title(text, file_path)
        cleaned = _clean_markdown(text)
        return ParsedDocument(
            text=cleaned,
            source_path=str(file_path),
            metadata={"format": "markdown"},
            title=title,
            format_hint="markdown",
        )


def _extract_title(text: str, file_path: Path) -> str:
    """Extract document title from first H1 heading or filename."""
    m = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    if m:
        return m.group(1).strip()
    return file_path.stem.replace("-", " ").replace("_", " ").title()


def _clean_markdown(text: str) -> str:
    """Light cleanup of markdown syntax — keeps headings for structural splitting."""
    # Remove images
    text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", text)
    # Convert links to just their text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # Remove emphasis markers but keep text
    text = re.sub(r"\*{1,3}(.+?)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,3}(.+?)_{1,3}", r"\1", text)
    # Remove code fence markers (keep content)
    text = re.sub(r"```[a-zA-Z]*\n?", "", text)
    # Remove inline code backticks
    text = re.sub(r"`(.+?)`", r"\1", text)
    # NOTE: Headings are KEPT for structure-aware chunking
    # Remove horizontal rules
    text = re.sub(r"^[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
