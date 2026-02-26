"""RTF (Rich Text Format) file parser using striprtf."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from rag_kb.parsers.base import DocumentParser, ParsedDocument

logger = logging.getLogger(__name__)


class RtfParser:
    """Parse RTF files, converting to plain text."""

    supported_extensions: list[str] = [".rtf"]

    def parse(self, file_path: Path) -> ParsedDocument:
        try:
            from striprtf.striprtf import rtf_to_text
        except ImportError as exc:
            raise ImportError(
                "striprtf is required for RTF parsing: pip install striprtf"
            ) from exc

        raw = file_path.read_text(encoding="utf-8", errors="replace")

        try:
            text = rtf_to_text(raw)
        except Exception as exc:
            logger.warning("RTF conversion failed for %s: %s", file_path, exc)
            # Fallback: crude RTF tag stripping
            text = re.sub(r"[{\\][^{}]*[}]?", "", raw)
            text = re.sub(r"\\[a-z]+\d*\s?", "", text)

        # Clean up whitespace
        text = re.sub(r"\n{3,}", "\n\n", text).strip()

        return ParsedDocument(
            text=text,
            source_path=str(file_path),
            metadata={"format": "rtf"},
        )
