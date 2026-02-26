"""CSV / TSV file parser with structured table extraction."""

from __future__ import annotations

import csv
import io
import logging
from pathlib import Path

from rag_kb.parsers.base import DocumentParser, ParsedDocument

logger = logging.getLogger(__name__)


class CsvParser:
    """Parse CSV and TSV files, presenting data as readable tables."""

    supported_extensions: list[str] = [".csv", ".tsv"]

    def parse(self, file_path: Path) -> ParsedDocument:
        raw = file_path.read_text(encoding="utf-8", errors="replace")

        # Auto-detect dialect
        try:
            dialect = csv.Sniffer().sniff(raw[:8192])
        except csv.Error:
            dialect = csv.excel  # fallback

        # Force tab delimiter for .tsv
        if file_path.suffix.lower() == ".tsv":
            dialect.delimiter = "\t"

        reader = csv.reader(io.StringIO(raw), dialect)
        rows: list[list[str]] = []
        for row in reader:
            rows.append([cell.strip() for cell in row])

        if not rows:
            return ParsedDocument(text="", source_path=str(file_path), metadata={"format": "csv"})

        # Build markdown-style table
        lines: list[str] = []
        header = rows[0]
        lines.append(" | ".join(header))
        lines.append(" | ".join("---" for _ in header))
        for row in rows[1:]:
            # Pad or truncate to match header width
            padded = row + [""] * max(0, len(header) - len(row))
            lines.append(" | ".join(padded[: len(header)]))

        text = "\n".join(lines)

        return ParsedDocument(
            text=text,
            source_path=str(file_path),
            metadata={
                "format": "csv",
                "row_count": str(len(rows) - 1),
                "column_count": str(len(header)),
            },
        )
