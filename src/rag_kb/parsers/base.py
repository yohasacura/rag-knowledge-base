"""Base protocol for document parsers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable


@dataclass
class ParsedDocument:
    """Result of parsing a single file."""

    text: str
    source_path: str
    metadata: dict[str, str] = field(default_factory=dict)
    title: str = ""
    format_hint: str = ""  # "markdown", "code", "pdf", or ""

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


@runtime_checkable
class DocumentParser(Protocol):
    """Protocol that all document parsers must implement."""

    supported_extensions: list[str]

    def parse(self, file_path: Path) -> ParsedDocument:
        """Parse a file and return its text content."""
        ...
