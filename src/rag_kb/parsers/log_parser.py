"""Log file parser with structured line extraction."""

from __future__ import annotations

import re
from pathlib import Path

from rag_kb.parsers.base import DocumentParser, ParsedDocument


class LogParser:
    """Parse log files, preserving structure and extracting metadata."""

    supported_extensions: list[str] = [".log"]

    def parse(self, file_path: Path) -> ParsedDocument:
        text = file_path.read_text(encoding="utf-8", errors="replace")

        # Count severity levels for metadata
        line_count = text.count("\n") + 1
        error_count = len(re.findall(r"\b(?:ERROR|FATAL|CRITICAL)\b", text, re.IGNORECASE))
        warning_count = len(re.findall(r"\bWARN(?:ING)?\b", text, re.IGNORECASE))

        return ParsedDocument(
            text=text,
            source_path=str(file_path),
            metadata={
                "format": "log",
                "line_count": str(line_count),
                "error_count": str(error_count),
                "warning_count": str(warning_count),
            },
        )
