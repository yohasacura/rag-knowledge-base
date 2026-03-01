"""reStructuredText file parser with markup stripping."""

from __future__ import annotations

import re
from pathlib import Path

from rag_kb.parsers.base import ParsedDocument

# Pre-compiled regex patterns for RST markup stripping
_RE_DIRECTIVES = re.compile(r"^\.\.\s[a-zA-Z0-9_-]+::\s*.*$", re.MULTILINE)
_RE_DIRECTIVE_OPTS = re.compile(r"^\s+:[a-zA-Z0-9_-]+:.*$", re.MULTILINE)
_RE_ROLES = re.compile(r":[a-zA-Z0-9_-]+:`([^`]+)`")
_RE_INLINE_LITERAL = re.compile(r"``(.+?)``")
_RE_BOLD = re.compile(r"\*\*(.+?)\*\*")
_RE_EMPHASIS = re.compile(r"\*(.+?)\*")
_RE_SUBST_REF = re.compile(r"\|([^|]+)\|")
_RE_FOOTNOTE_REF = re.compile(r"\[#?[^\]]*\]_")
_RE_HYPERLINK_REF = re.compile(r"`([^<]+)\s*<[^>]+>`_")
_RE_ANON_HYPERLINK = re.compile(r"`([^`]+)`__?")
_RE_SECTION_LINES = re.compile(r"^[=\-~`'^\"#\*\+\.]{3,}\s*$", re.MULTILINE)
_RE_COMMENT_LINES = re.compile(r"^\.\.\s+[^:].*$", re.MULTILINE)
_RE_TARGET_DEFS = re.compile(r"^\.\.\s_[^:]+:\s*.*$", re.MULTILINE)
_RE_MULTI_NEWLINES = re.compile(r"\n{3,}")


class RstParser:
    """Parse reStructuredText files, stripping markup for cleaner embeddings."""

    supported_extensions: list[str] = [".rst", ".rest"]

    def parse(self, file_path: Path) -> ParsedDocument:
        text = file_path.read_text(encoding="utf-8", errors="replace")
        cleaned = _strip_rst(text)
        return ParsedDocument(
            text=cleaned,
            source_path=str(file_path),
            metadata={"format": "rst"},
        )


def _strip_rst(text: str) -> str:
    """Strip reStructuredText markup for better embeddings."""
    text = _RE_DIRECTIVES.sub("", text)
    text = _RE_DIRECTIVE_OPTS.sub("", text)
    text = _RE_ROLES.sub(r"\1", text)
    text = _RE_INLINE_LITERAL.sub(r"\1", text)
    text = _RE_BOLD.sub(r"\1", text)
    text = _RE_EMPHASIS.sub(r"\1", text)
    text = _RE_SUBST_REF.sub(r"\1", text)
    text = _RE_FOOTNOTE_REF.sub("", text)
    text = _RE_HYPERLINK_REF.sub(r"\1", text)
    text = _RE_ANON_HYPERLINK.sub(r"\1", text)
    text = _RE_SECTION_LINES.sub("", text)
    text = _RE_COMMENT_LINES.sub("", text)
    text = _RE_TARGET_DEFS.sub("", text)
    text = _RE_MULTI_NEWLINES.sub("\n\n", text)
    return text.strip()
