"""Plain text file parser."""

from __future__ import annotations

from pathlib import Path

from rag_kb.parsers.base import DocumentParser, ParsedDocument


class TxtParser:
    """Parse plain text files."""

    supported_extensions: list[str] = [".txt", ".text"]

    def parse(self, file_path: Path) -> ParsedDocument:
        text = file_path.read_text(encoding="utf-8", errors="replace")
        return ParsedDocument(
            text=text,
            source_path=str(file_path),
            metadata={"format": "text"},
        )
