"""Structure-aware text chunker with contextual prefixes.

Key improvements over the original character-based splitter:
  - **Structure-aware**: splits Markdown by headings, code by function/class
    boundaries, and PDFs by page markers *before* applying recursive splitting.
  - **Contextual prefix**: optionally prepends document title and section heading
    to each chunk so that the embedding captures document context (Anthropic's
    "contextual retrieval" technique).
  - **Larger default overlap** (128 chars) to preserve context at boundaries.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# Pre-compiled patterns (avoid re-compilation per file)
_RE_MD_HEADING = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
_RE_CODE_PYTHON = re.compile(r"^((?:async\s+)?def\s+\w+|class\s+\w+)", re.MULTILINE)
_RE_CODE_JSTS = re.compile(r"^(?:export\s+)?(?:function\s+\w+|class\s+\w+|const\s+\w+\s*=)", re.MULTILINE)
_RE_PDF_PAGE = re.compile(r"\[PAGE\s+(\d+)\]")


@dataclass
class TextChunk:
    """A chunk of text with positional and source metadata."""

    text: str
    source_file: str
    chunk_index: int
    start_char: int
    end_char: int
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def chunk_id(self) -> str:
        """Deterministic ID for this chunk."""
        return f"{self.source_file}::chunk_{self.chunk_index}"


# ---------------------------------------------------------------------------
# Separators (ordered from coarsest to finest)
# ---------------------------------------------------------------------------

_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


# ---------------------------------------------------------------------------
# Structure-aware section splitting
# ---------------------------------------------------------------------------

@dataclass
class Section:
    """A structural section extracted from a document."""
    heading: str
    text: str
    start_char: int  # offset in the original document


def split_by_structure(
    text: str,
    format_hint: str = "",
) -> list[Section]:
    """Split *text* into structural sections based on document format.

    Parameters
    ----------
    text : str
        Full document text.
    format_hint : str
        One of ``"markdown"``, ``"code"``, ``"pdf"``, or ``""`` (default).

    Returns
    -------
    List of Section objects with heading and text.
    """
    if format_hint == "markdown":
        return _split_markdown_sections(text)
    elif format_hint == "code":
        return _split_code_sections(text)
    elif format_hint == "pdf":
        return _split_pdf_sections(text)
    else:
        return [Section(heading="", text=text, start_char=0)]


def _split_markdown_sections(text: str) -> list[Section]:
    """Split Markdown into sections by headings (## or ###)."""
    matches = list(_RE_MD_HEADING.finditer(text))

    if not matches:
        return [Section(heading="", text=text, start_char=0)]

    sections: list[Section] = []

    # Text before the first heading
    if matches[0].start() > 0:
        pre_text = text[: matches[0].start()].strip()
        if pre_text:
            sections.append(Section(heading="", text=pre_text, start_char=0))

    heading_stack: list[tuple[int, str]] = []  # (level, heading_text)

    for i, m in enumerate(matches):
        level = len(m.group(1))
        heading_text = m.group(2).strip()

        # Maintain heading hierarchy
        while heading_stack and heading_stack[-1][0] >= level:
            heading_stack.pop()
        heading_stack.append((level, heading_text))

        # Build hierarchical heading string
        full_heading = " > ".join(h for _, h in heading_stack)

        # Section text runs from end of heading line to start of next heading
        sec_start = m.end()
        sec_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sec_text = text[sec_start:sec_end].strip()

        if sec_text:
            sections.append(Section(
                heading=full_heading,
                text=sec_text,
                start_char=sec_start,
            ))

    return sections if sections else [Section(heading="", text=text, start_char=0)]


def _split_code_sections(text: str) -> list[Section]:
    """Split code into sections by top-level function/class definitions."""
    patterns = [_RE_CODE_PYTHON, _RE_CODE_JSTS]

    matches: list[re.Match] = []
    for pat in patterns:
        matches.extend(pat.finditer(text))

    if not matches:
        return [Section(heading="", text=text, start_char=0)]

    matches.sort(key=lambda m: m.start())
    unique_matches: list[re.Match] = []
    for m in matches:
        if not unique_matches or m.start() > unique_matches[-1].start() + 10:
            unique_matches.append(m)

    sections: list[Section] = []

    # Preamble before first definition
    if unique_matches[0].start() > 0:
        pre_text = text[: unique_matches[0].start()].strip()
        if pre_text:
            sections.append(Section(heading="(module preamble)", text=pre_text, start_char=0))

    for i, m in enumerate(unique_matches):
        heading = m.group(0).strip()
        sec_start = m.start()
        sec_end = unique_matches[i + 1].start() if i + 1 < len(unique_matches) else len(text)
        sec_text = text[sec_start:sec_end].strip()

        if sec_text:
            sections.append(Section(
                heading=heading,
                text=sec_text,
                start_char=sec_start,
            ))

    return sections if sections else [Section(heading="", text=text, start_char=0)]


def _split_pdf_sections(text: str) -> list[Section]:
    """Split PDF text by page markers inserted by the PDF parser."""
    matches = list(_RE_PDF_PAGE.finditer(text))

    if not matches:
        return [Section(heading="", text=text, start_char=0)]

    sections: list[Section] = []

    if matches[0].start() > 0:
        pre_text = text[: matches[0].start()].strip()
        if pre_text:
            sections.append(Section(heading="", text=pre_text, start_char=0))

    for i, m in enumerate(matches):
        page_num = m.group(1)
        heading = f"Page {page_num}"
        sec_start = m.end()
        sec_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sec_text = text[sec_start:sec_end].strip()

        if sec_text:
            sections.append(Section(
                heading=heading,
                text=sec_text,
                start_char=sec_start,
            ))

    return sections if sections else [Section(heading="", text=text, start_char=0)]


# ---------------------------------------------------------------------------
# Main chunking API
# ---------------------------------------------------------------------------

def chunk_text(
    text: str,
    source_file: str,
    chunk_size: int = 1024,
    chunk_overlap: int = 128,
    metadata: dict[str, str] | None = None,
    document_title: str = "",
    format_hint: str = "",
) -> list[TextChunk]:
    """Split *text* into overlapping chunks with structure-aware splitting.

    Flow:
      1. Split into structural sections (by headings / functions / pages).
      2. Apply recursive character splitting within each section.
      3. Add contextual prefix (title + heading) for better embeddings.
      4. Build overlapping chunks with metadata.
    """
    if not text.strip():
        return []

    extra_meta = metadata or {}
    sections = split_by_structure(text, format_hint)

    all_chunks: list[TextChunk] = []
    global_idx = 0

    for section in sections:
        sec_text = section.text
        if not sec_text.strip():
            continue

        raw_pieces = _recursive_split(sec_text, chunk_size, list(_SEPARATORS))

        pos = 0
        for piece in raw_pieces:
            if not piece.strip():
                pos += len(piece)
                continue

            start = sec_text.find(piece, pos)
            if start == -1 or start > pos + len(piece):
                start = pos
            end = start + len(piece)

            # Prepend overlap from previous text
            overlap_start = max(0, start - chunk_overlap)
            if overlap_start < start and global_idx > 0:
                overlapped_text = sec_text[overlap_start:end]
            else:
                overlapped_text = piece

            # Build contextual prefix for the embedding
            prefix_parts: list[str] = []
            if document_title:
                prefix_parts.append(f"Title: {document_title}")
            if section.heading:
                prefix_parts.append(f"Section: {section.heading}")
            prefix = "\n".join(prefix_parts)
            if prefix:
                display_text = f"{prefix}\n\n{overlapped_text.strip()}"
            else:
                display_text = overlapped_text.strip()

            if display_text.strip():
                chunk_meta = dict(extra_meta)
                if section.heading:
                    chunk_meta["section_heading"] = section.heading
                if document_title:
                    chunk_meta["document_title"] = document_title

                all_chunks.append(
                    TextChunk(
                        text=display_text,
                        source_file=source_file,
                        chunk_index=global_idx,
                        start_char=section.start_char + (overlap_start if overlap_start < start else start),
                        end_char=section.start_char + end,
                        metadata=chunk_meta,
                    )
                )
                global_idx += 1

            pos = end

    return [c for c in all_chunks if c.text.strip()]


# ---------------------------------------------------------------------------
# Recursive splitting
# ---------------------------------------------------------------------------

def _recursive_split(text: str, max_size: int, separators: list[str]) -> list[str]:
    """Recursively split *text* on successively finer separators."""
    if len(text) <= max_size:
        return [text]

    if not separators:
        return [text[i: i + max_size] for i in range(0, len(text), max_size)]

    sep = separators[0]
    rest = separators[1:]

    if sep == "":
        return [text[i: i + max_size] for i in range(0, len(text), max_size)]

    parts = text.split(sep)
    result: list[str] = []
    current = ""

    for part in parts:
        candidate = f"{current}{sep}{part}" if current else part
        if len(candidate) <= max_size:
            current = candidate
        else:
            if current:
                result.append(current)
            if len(part) > max_size:
                result.extend(_recursive_split(part, max_size, rest))
            else:
                current = part
                continue
            current = ""

    if current:
        result.append(current)

    return result
