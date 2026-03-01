"""YAML file parser with structured extraction."""

from __future__ import annotations

import logging
from pathlib import Path

from rag_kb.parsers.base import ParsedDocument

logger = logging.getLogger(__name__)

_MAX_DEPTH = 10


class YamlParser:
    """Parse YAML files, flattening nested structures into readable text."""

    supported_extensions: list[str] = [".yaml", ".yml"]

    def parse(self, file_path: Path) -> ParsedDocument:
        try:
            import yaml
        except ImportError as exc:
            raise ImportError("pyyaml is required for YAML parsing: pip install pyyaml") from exc

        raw = file_path.read_text(encoding="utf-8", errors="replace")

        try:
            # Support multi-document YAML
            docs = list(yaml.safe_load_all(raw))
        except yaml.YAMLError as exc:
            logger.warning("Invalid YAML in %s: %s — falling back to raw text", file_path, exc)
            return ParsedDocument(
                text=raw,
                source_path=str(file_path),
                metadata={"format": "yaml", "parse_error": str(exc)},
            )

        all_lines: list[str] = []
        for doc_idx, data in enumerate(docs):
            if data is None:
                continue
            if len(docs) > 1:
                all_lines.append(f"[Document {doc_idx + 1}]")
            all_lines.extend(_flatten(data))
            all_lines.append("")

        text = "\n".join(all_lines).strip()

        return ParsedDocument(
            text=text,
            source_path=str(file_path),
            metadata={
                "format": "yaml",
                "document_count": str(len(docs)),
            },
        )


def _flatten(data: object, prefix: str = "", depth: int = 0) -> list[str]:
    """Recursively flatten YAML data into 'key: value' lines."""
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
