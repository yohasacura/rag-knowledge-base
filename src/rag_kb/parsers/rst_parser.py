"""reStructuredText file parser with markup stripping."""

from __future__ import annotations

import re
from pathlib import Path

from rag_kb.parsers.base import DocumentParser, ParsedDocument


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
    # Remove directive blocks (.. directive::) but keep their content
    text = re.sub(r"^\.\. [a-zA-Z0-9_-]+::\s*.*$", "", text, flags=re.MULTILINE)
    # Remove directive options (:option: value)
    text = re.sub(r"^\s+:[a-zA-Z0-9_-]+:.*$", "", text, flags=re.MULTILINE)
    # Remove role markers :role:`text` → text
    text = re.sub(r":[a-zA-Z0-9_-]+:`([^`]+)`", r"\1", text)
    # Remove inline literal ``text`` → text
    text = re.sub(r"``(.+?)``", r"\1", text)
    # Remove emphasis markers
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    # Remove substitution references |text|
    text = re.sub(r"\|([^|]+)\|", r"\1", text)
    # Remove footnote/citation references [#name]_ or [name]_
    text = re.sub(r"\[#?[^\]]*\]_", "", text)
    # Remove hyperlink references `text <url>`_ → text
    text = re.sub(r"`([^<]+)\s*<[^>]+>`_", r"\1", text)
    # Remove anonymous hyperlink `text`__
    text = re.sub(r"`([^`]+)`__?", r"\1", text)
    # Remove section underline/overline characters
    text = re.sub(r"^[=\-~`'^\"#\*\+\.]{3,}\s*$", "", text, flags=re.MULTILINE)
    # Remove comment lines (.. comment)
    text = re.sub(r"^\.\.\s+[^:].*$", "", text, flags=re.MULTILINE)
    # Remove target definitions (.. _name:)
    text = re.sub(r"^\.\. _[^:]+:\s*.*$", "", text, flags=re.MULTILINE)
    # Collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
