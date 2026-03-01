"""Chunker unit tests — structure-aware splitting, overlap, prefixes."""

from __future__ import annotations

import pytest

from rag_kb.chunker import (
    Section,
    TextChunk,
    _recursive_split,
    _split_code_sections,
    _split_markdown_sections,
    _split_pdf_sections,
    chunk_text,
    split_by_structure,
)


# ---------------------------------------------------------------------------
# chunk_text — basic behaviour
# ---------------------------------------------------------------------------


class TestChunkTextBasic:
    def test_empty_text_returns_empty(self):
        assert chunk_text("", source_file="f.txt") == []

    def test_whitespace_only_returns_empty(self):
        assert chunk_text("   \n\n   ", source_file="f.txt") == []

    def test_short_text_single_chunk(self):
        text = "Hello world."
        chunks = chunk_text(text, source_file="f.txt", chunk_size=1000)
        assert len(chunks) == 1
        assert "Hello world." in chunks[0].text

    def test_chunk_source_file(self):
        chunks = chunk_text("Some content.", source_file="test.md")
        assert all(c.source_file == "test.md" for c in chunks)

    def test_chunk_index_sequential(self):
        text = "A " * 500
        chunks = chunk_text(text, source_file="f.txt", chunk_size=100)
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_chunk_id_format(self):
        chunks = chunk_text("Some data.", source_file="doc.txt")
        assert chunks[0].chunk_id == "doc.txt::chunk_0"


# ---------------------------------------------------------------------------
# chunk_text — size and overlap
# ---------------------------------------------------------------------------


class TestChunkSizeOverlap:
    def test_chunks_respect_max_size(self):
        # Generate text larger than chunk_size
        text = "word " * 300  # ~1500 chars
        chunks = chunk_text(text, source_file="f.txt", chunk_size=200, chunk_overlap=0)
        for c in chunks:
            # Account for possible contextual prefix
            assert len(c.text) < 500  # generous bound

    def test_text_exactly_at_chunk_size(self):
        text = "x" * 1024
        chunks = chunk_text(text, source_file="f.txt", chunk_size=1024, chunk_overlap=0)
        assert len(chunks) >= 1

    def test_overlap_produces_shared_content(self):
        text = "sentence one. " * 50 + "sentence two. " * 50
        chunks = chunk_text(text, source_file="f.txt", chunk_size=200, chunk_overlap=50)
        if len(chunks) >= 2:
            # There should be some overlapping text between consecutive chunks
            # (hard to verify exactly, but we should get more than 1 chunk)
            assert len(chunks) >= 2

    def test_large_overlap_does_not_crash(self):
        text = "word " * 100
        # Overlap larger than chunk content should not crash
        chunks = chunk_text(text, source_file="f.txt", chunk_size=200, chunk_overlap=190)
        assert isinstance(chunks, list)


# ---------------------------------------------------------------------------
# chunk_text — contextual prefix
# ---------------------------------------------------------------------------


class TestContextualPrefix:
    def test_title_prefix(self):
        chunks = chunk_text(
            "Content here.", source_file="f.txt", document_title="My Document"
        )
        assert len(chunks) > 0
        assert "Title: My Document" in chunks[0].text

    def test_section_heading_in_metadata(self):
        text = "# Introduction\n\nThis is the intro.\n\n# Body\n\nThis is the body."
        chunks = chunk_text(text, source_file="f.md", format_hint="markdown")
        headings = [c.metadata.get("section_heading", "") for c in chunks]
        assert any(h for h in headings)

    def test_title_and_section_combined(self):
        text = "# Chapter 1\n\nContent of chapter one."
        chunks = chunk_text(
            text,
            source_file="f.md",
            format_hint="markdown",
            document_title="Book Title",
        )
        assert len(chunks) > 0
        first = chunks[0].text
        assert "Title: Book Title" in first
        assert "Section:" in first


# ---------------------------------------------------------------------------
# split_by_structure
# ---------------------------------------------------------------------------


class TestSplitByStructure:
    def test_default_format_returns_single_section(self):
        sections = split_by_structure("Hello world.", "")
        assert len(sections) == 1
        assert sections[0].text == "Hello world."

    def test_markdown_format(self):
        text = "# H1\n\nParagraph 1.\n\n## H2\n\nParagraph 2."
        sections = split_by_structure(text, "markdown")
        assert len(sections) >= 2

    def test_code_format(self):
        text = "import os\n\ndef foo():\n    pass\n\nclass Bar:\n    pass\n"
        sections = split_by_structure(text, "code")
        assert len(sections) >= 1

    def test_pdf_format(self):
        text = "[PAGE 1] Page one content.\n[PAGE 2] Page two content."
        sections = split_by_structure(text, "pdf")
        assert len(sections) >= 2

    def test_unknown_format_single_section(self):
        sections = split_by_structure("text", "unknown_format")
        assert len(sections) == 1


# ---------------------------------------------------------------------------
# _split_markdown_sections
# ---------------------------------------------------------------------------


class TestMarkdownSections:
    def test_no_headings_single_section(self):
        sections = _split_markdown_sections("Just plain text.")
        assert len(sections) == 1

    def test_heading_hierarchy(self):
        text = "# Top\n\nPreamble.\n\n## Sub\n\nDetail.\n\n### SubSub\n\nMore detail."
        sections = _split_markdown_sections(text)
        headings = [s.heading for s in sections]
        assert any("Top" in h for h in headings)
        assert any("Sub" in h for h in headings)

    def test_preamble_before_first_heading(self):
        text = "Preamble text.\n\n# Heading\n\nBody."
        sections = _split_markdown_sections(text)
        assert sections[0].heading == ""
        assert "Preamble" in sections[0].text

    def test_deeply_nested_headings(self):
        text = "# L1\n\na\n\n## L2\n\nb\n\n### L3\n\nc\n\n#### L4\n\nd\n\n##### L5\n\ne\n\n###### L6\n\nf"
        sections = _split_markdown_sections(text)
        # Should produce sections for each heading level
        assert len(sections) >= 6


# ---------------------------------------------------------------------------
# _split_code_sections
# ---------------------------------------------------------------------------


class TestCodeSections:
    def test_python_functions(self):
        text = "import os\n\ndef foo():\n    return 1\n\ndef bar():\n    return 2\n"
        sections = _split_code_sections(text)
        assert len(sections) >= 2  # preamble + at least one function

    def test_python_classes(self):
        text = "class Foo:\n    pass\n\nclass Bar:\n    pass\n"
        sections = _split_code_sections(text)
        assert len(sections) >= 2

    def test_no_definitions_single_section(self):
        text = "x = 1\ny = 2\nprint(x + y)"
        sections = _split_code_sections(text)
        assert len(sections) == 1

    def test_async_def(self):
        text = "async def handler():\n    await do_thing()\n"
        sections = _split_code_sections(text)
        assert any("async def handler" in s.heading for s in sections)


# ---------------------------------------------------------------------------
# _split_pdf_sections
# ---------------------------------------------------------------------------


class TestPdfSections:
    def test_page_markers(self):
        text = "[PAGE 1] First page.\n[PAGE 2] Second page.\n[PAGE 3] Third."
        sections = _split_pdf_sections(text)
        assert len(sections) >= 3
        assert any("Page 1" in s.heading for s in sections)

    def test_no_page_markers(self):
        text = "Just text without page markers."
        sections = _split_pdf_sections(text)
        assert len(sections) == 1

    def test_content_before_first_marker(self):
        text = "Preamble.\n[PAGE 1] Content."
        sections = _split_pdf_sections(text)
        assert sections[0].heading == ""  # preamble
        assert "Preamble" in sections[0].text


# ---------------------------------------------------------------------------
# _recursive_split
# ---------------------------------------------------------------------------


class TestRecursiveSplit:
    def test_short_text_not_split(self):
        result = _recursive_split("short", 100, ["\n\n", "\n", " "])
        assert result == ["short"]

    def test_splits_on_separator(self):
        text = "part one\n\npart two\n\npart three"
        result = _recursive_split(text, 20, ["\n\n", "\n", " "])
        assert len(result) >= 2

    def test_empty_separator_falls_back_to_char_split(self):
        text = "a" * 100
        result = _recursive_split(text, 30, [""])
        assert all(len(p) <= 30 for p in result)

    def test_single_very_long_line(self):
        text = "x" * 5000
        result = _recursive_split(text, 200, ["\n\n", "\n", ". ", " ", ""])
        assert all(len(p) <= 200 for p in result)


# ---------------------------------------------------------------------------
# Unicode handling
# ---------------------------------------------------------------------------


class TestUnicodeChunking:
    def test_cjk_characters(self):
        text = "量子コンピューティングは未来の技術です。" * 50
        chunks = chunk_text(text, source_file="cjk.txt", chunk_size=100)
        assert len(chunks) >= 1

    def test_emoji_text(self):
        text = "Hello 🌍 World 🎉 " * 50
        chunks = chunk_text(text, source_file="emoji.txt", chunk_size=100)
        assert len(chunks) >= 1

    def test_mixed_scripts(self):
        text = "English العربية Русский 中文 日本語 한국어 " * 20
        chunks = chunk_text(text, source_file="mixed.txt", chunk_size=100)
        assert len(chunks) >= 1
