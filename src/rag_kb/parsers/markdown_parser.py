"""Markdown file parser.

Keeps headings intact so the structure-aware chunker can split by sections.
Only strips formatting noise (images, inline markup, HTML tags).
"""

from __future__ import annotations

import re
from pathlib import Path

from rag_kb.parsers.base import DocumentParser, ParsedDocument

# Pre-compiled regex patterns (avoid re-compilation per file)
_RE_TITLE = re.compile(r"^#\s+(.+)$", re.MULTILINE)
_RE_IMAGES = re.compile(r"!\[([^\]]*)\]\([^)]+\)")
_RE_LINKS = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_RE_BOLD_STAR = re.compile(r"\*{1,3}(.+?)\*{1,3}")
_RE_BOLD_UNDER = re.compile(r"_{1,3}(.+?)_{1,3}")
_RE_CODE_FENCE = re.compile(r"```[a-zA-Z]*\n?")
_RE_INLINE_CODE = re.compile(r"`(.+?)`")
_RE_HRULE = re.compile(r"^[-*_]{3,}\s*$", re.MULTILINE)
_RE_HTML_TAGS = re.compile(r"<[^>]+>")
_RE_MULTI_NEWLINES = re.compile(r"\n{3,}")


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
    m = _RE_TITLE.search(text)
    if m:
        return m.group(1).strip()
    return file_path.stem.replace("-", " ").replace("_", " ").title()


def _clean_markdown(text: str) -> str:
    """Light cleanup of markdown syntax — keeps headings for structural splitting."""
    text = _RE_IMAGES.sub(r"\1", text)
    text = _RE_LINKS.sub(r"\1", text)
    text = _RE_BOLD_STAR.sub(r"\1", text)
    text = _RE_BOLD_UNDER.sub(r"\1", text)
    text = _RE_CODE_FENCE.sub("", text)
    text = _RE_INLINE_CODE.sub(r"\1", text)
    # NOTE: Headings are KEPT for structure-aware chunking
    text = _RE_HRULE.sub("", text)
    text = _RE_HTML_TAGS.sub("", text)
    text = _RE_MULTI_NEWLINES.sub("\n\n", text)
    return text.strip()
