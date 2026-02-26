"""XML file parser using lxml / stdlib xml."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from rag_kb.parsers.base import DocumentParser, ParsedDocument

logger = logging.getLogger(__name__)


class XmlParser:
    """Parse XML files, extracting element text content."""

    supported_extensions: list[str] = [".xml", ".xsl", ".xslt", ".xsd", ".svg", ".rss", ".atom"]

    def parse(self, file_path: Path) -> ParsedDocument:
        raw = file_path.read_text(encoding="utf-8", errors="replace")

        try:
            from lxml import etree

            root = etree.fromstring(raw.encode("utf-8"))
            lines = _walk_lxml(root)
        except Exception:
            # Fallback: strip XML tags
            lines = _strip_xml_tags(raw)

        text = "\n".join(lines)
        # Collapse multiple blank lines
        text = re.sub(r"\n{3,}", "\n\n", text).strip()

        return ParsedDocument(
            text=text,
            source_path=str(file_path),
            metadata={"format": "xml"},
        )


def _walk_lxml(element, depth: int = 0) -> list[str]:
    """Walk an lxml element tree, extracting text in a structured way."""
    lines: list[str] = []
    # Strip namespace from tag name for readability
    tag = _local_name(element.tag) if isinstance(element.tag, str) else ""

    # Element's own text
    text = (element.text or "").strip()
    if text and tag:
        indent = "  " * min(depth, 6)
        lines.append(f"{indent}{tag}: {text}")
    elif text:
        lines.append(text)

    # Recurse children
    for child in element:
        lines.extend(_walk_lxml(child, depth + 1))

    # Tail text (text after closing tag)
    tail = (element.tail or "").strip()
    if tail:
        lines.append(tail)

    return lines


def _local_name(tag: str) -> str:
    """Remove namespace prefix from XML tag name."""
    if "}" in tag:
        return tag.split("}", 1)[1]
    return tag


def _strip_xml_tags(raw: str) -> list[str]:
    """Fallback: strip XML tags and return plain text lines."""
    text = re.sub(r"<[^>]+>", " ", raw)
    text = re.sub(r"\s+", " ", text).strip()
    return [line.strip() for line in text.split(".") if line.strip()]
