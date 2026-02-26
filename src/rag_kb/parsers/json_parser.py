"""JSON file parser with structured key-value flattening."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from rag_kb.parsers.base import DocumentParser, ParsedDocument

logger = logging.getLogger(__name__)

# Maximum depth we recurse into nested structures
_MAX_DEPTH = 10


class JsonParser:
    """Parse JSON files, flattening nested structures into readable text."""

    supported_extensions: list[str] = [".json", ".jsonl"]

    def parse(self, file_path: Path) -> ParsedDocument:
        raw = file_path.read_text(encoding="utf-8", errors="replace")

        # Handle JSON Lines (one JSON object per line)
        if file_path.suffix.lower() == ".jsonl":
            return self._parse_jsonl(raw, file_path)

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning("Invalid JSON in %s: %s — falling back to raw text", file_path, exc)
            return ParsedDocument(
                text=raw,
                source_path=str(file_path),
                metadata={"format": "json", "parse_error": str(exc)},
            )

        lines = _flatten(data)
        text = "\n".join(lines)

        return ParsedDocument(
            text=text,
            source_path=str(file_path),
            metadata={
                "format": "json",
                "type": type(data).__name__,
            },
        )

    def _parse_jsonl(self, raw: str, file_path: Path) -> ParsedDocument:
        """Parse JSON Lines format — one JSON object per line."""
        records: list[str] = []
        for i, line in enumerate(raw.strip().splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                flat = _flatten(obj, prefix=f"record_{i}")
                records.extend(flat)
                records.append("")  # blank separator
            except json.JSONDecodeError:
                records.append(f"record_{i}: {line}")

        return ParsedDocument(
            text="\n".join(records),
            source_path=str(file_path),
            metadata={"format": "jsonl", "line_count": str(len(records))},
        )


def _flatten(data: object, prefix: str = "", depth: int = 0) -> list[str]:
    """Recursively flatten JSON data into 'key: value' lines."""
    if depth > _MAX_DEPTH:
        return [f"{prefix}: ..."]

    lines: list[str] = []

    if isinstance(data, dict):
        for key, value in data.items():
            full_key = f"{prefix}.{key}" if prefix else str(key)
            lines.extend(_flatten(value, full_key, depth + 1))
    elif isinstance(data, list):
        if all(isinstance(item, (str, int, float, bool, type(None))) for item in data):
            val = ", ".join(str(v) for v in data)
            lines.append(f"{prefix}: [{val}]")
        else:
            for i, item in enumerate(data):
                lines.extend(_flatten(item, f"{prefix}[{i}]", depth + 1))
    else:
        val = str(data) if data is not None else ""
        if prefix:
            lines.append(f"{prefix}: {val}")
        else:
            lines.append(val)

    return lines
