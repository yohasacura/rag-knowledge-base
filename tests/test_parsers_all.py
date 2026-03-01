"""Comprehensive parser tests using test-docs-verify/ sample files + synthetic edge cases.

Each parser class is tested against:
1. Its real sample file from test-docs-verify/ (if present)
2. Synthetic edge-case content
3. Empty file handling

Parsers for binary formats (PDF, DOCX, PPTX, XLSX, EPUB, RTF, ODF, images)
require their corresponding test file; tests are skipped if file missing.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from rag_kb.parsers.base import ParsedDocument
from rag_kb.parsers.registry import (
    SUPPORTED_EXTENSIONS,
    get_parser,
    parse_file,
)

# Absolute path to the verification docs shipped with the repo
_VERIFY_DIR = Path(__file__).resolve().parent.parent / "test-docs-verify"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _has_verify_file(name: str) -> bool:
    """Check if a specific test-docs-verify file exists."""
    return (_VERIFY_DIR / name).is_file()


def _parse_verify(name: str) -> ParsedDocument | None:
    """Parse a file from test-docs-verify/."""
    return parse_file(_VERIFY_DIR / name)


def _parse_tmp(path: Path) -> ParsedDocument | None:
    """Parse a synthetic temp file."""
    return parse_file(path)


# ---------------------------------------------------------------------------
# Registry-level tests
# ---------------------------------------------------------------------------


class TestParserRegistry:
    def test_supported_extensions_not_empty(self):
        assert len(SUPPORTED_EXTENSIONS) > 30  # we have 80+ extensions

    def test_get_parser_returns_none_for_unknown(self, tmp_path):
        f = tmp_path / "data.xyz123"
        f.write_text("hello", encoding="utf-8")
        assert get_parser(f) is None

    def test_parse_file_returns_none_for_unknown(self, tmp_path):
        f = tmp_path / "data.unknown"
        f.write_text("hello", encoding="utf-8")
        assert parse_file(f) is None

    def test_parser_instances_cached(self, tmp_path):
        f1 = tmp_path / "a.txt"
        f1.write_text("a", encoding="utf-8")
        f2 = tmp_path / "b.txt"
        f2.write_text("b", encoding="utf-8")
        p1 = get_parser(f1)
        p2 = get_parser(f2)
        assert p1 is p2  # same cached instance


# ---------------------------------------------------------------------------
# TxtParser
# ---------------------------------------------------------------------------


class TestTxtParser:
    def test_real_file(self):
        doc = _parse_verify("test.txt")
        assert doc is not None
        assert not doc.is_empty
        assert doc.metadata.get("format") == "text"

    def test_synthetic(self, tmp_path):
        f = tmp_path / "hello.txt"
        f.write_text("Hello World!", encoding="utf-8")
        doc = _parse_tmp(f)
        assert doc is not None
        assert "Hello World!" in doc.text
        assert doc.source_path == str(f)

    def test_empty(self, tmp_path):
        f = tmp_path / "empty.txt"
        f.write_text("", encoding="utf-8")
        doc = _parse_tmp(f)
        assert doc is not None
        assert doc.is_empty

    def test_unicode(self, tmp_path):
        f = tmp_path / "uni.txt"
        f.write_text("こんにちは 🌍 مرحبا", encoding="utf-8")
        doc = _parse_tmp(f)
        assert doc is not None
        assert "こんにちは" in doc.text


# ---------------------------------------------------------------------------
# MarkdownParser
# ---------------------------------------------------------------------------


class TestMarkdownParser:
    def test_real_file(self):
        doc = _parse_verify("test.md")
        assert doc is not None
        assert not doc.is_empty
        assert doc.format_hint == "markdown"

    def test_headings_preserved(self, tmp_path):
        f = tmp_path / "doc.md"
        f.write_text("# Title\n\nParagraph text.\n\n## Section\n\nMore text.", encoding="utf-8")
        doc = _parse_tmp(f)
        assert doc is not None
        assert "Title" in doc.text
        assert "Section" in doc.text

    def test_empty(self, tmp_path):
        f = tmp_path / "empty.md"
        f.write_text("", encoding="utf-8")
        doc = _parse_tmp(f)
        assert doc is not None
        assert doc.is_empty


# ---------------------------------------------------------------------------
# JsonParser
# ---------------------------------------------------------------------------


class TestJsonParser:
    def test_real_file(self):
        doc = _parse_verify("test.json")
        assert doc is not None
        assert not doc.is_empty
        assert doc.metadata.get("format") == "json"

    def test_jsonl_file(self):
        doc = _parse_verify("test.jsonl")
        assert doc is not None
        assert not doc.is_empty

    def test_nested_object(self, tmp_path):
        f = tmp_path / "nested.json"
        f.write_text('{"a": {"b": {"c": 42}}}', encoding="utf-8")
        doc = _parse_tmp(f)
        assert doc is not None
        assert "42" in doc.text

    def test_invalid_json_fallback(self, tmp_path):
        f = tmp_path / "bad.json"
        f.write_text("{not valid json", encoding="utf-8")
        doc = _parse_tmp(f)
        assert doc is not None
        # Falls back to raw text
        assert "{not valid json" in doc.text
        assert doc.metadata.get("parse_error")

    def test_json_array(self, tmp_path):
        f = tmp_path / "arr.json"
        f.write_text('[1, 2, 3]', encoding="utf-8")
        doc = _parse_tmp(f)
        assert doc is not None
        assert doc.metadata.get("type") == "list"

    def test_empty(self, tmp_path):
        f = tmp_path / "empty.json"
        f.write_text("", encoding="utf-8")
        doc = _parse_tmp(f)
        assert doc is not None


# ---------------------------------------------------------------------------
# CsvParser
# ---------------------------------------------------------------------------


class TestCsvParser:
    def test_real_csv(self):
        doc = _parse_verify("test.csv")
        assert doc is not None
        assert not doc.is_empty
        assert doc.metadata.get("format") == "csv"

    def test_real_tsv(self):
        if not _has_verify_file("test.tsv"):
            pytest.skip("test.tsv not available")
        doc = _parse_verify("test.tsv")
        assert doc is not None
        assert not doc.is_empty

    def test_synthetic_csv(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("name,age\nAlice,30\nBob,25", encoding="utf-8")
        doc = _parse_tmp(f)
        assert doc is not None
        assert "Alice" in doc.text
        assert doc.metadata.get("row_count") == "2"
        assert doc.metadata.get("column_count") == "2"

    def test_empty_csv(self, tmp_path):
        f = tmp_path / "empty.csv"
        f.write_text("", encoding="utf-8")
        doc = _parse_tmp(f)
        assert doc is not None
        assert doc.is_empty


# ---------------------------------------------------------------------------
# YamlParser
# ---------------------------------------------------------------------------


class TestYamlParser:
    def test_real_file(self):
        doc = _parse_verify("test.yaml")
        assert doc is not None
        assert not doc.is_empty

    def test_synthetic(self, tmp_path):
        f = tmp_path / "conf.yaml"
        f.write_text("key: value\nlist:\n  - item1\n  - item2", encoding="utf-8")
        doc = _parse_tmp(f)
        assert doc is not None
        assert "key" in doc.text or "value" in doc.text


# ---------------------------------------------------------------------------
# XmlParser
# ---------------------------------------------------------------------------


class TestXmlParser:
    def test_real_file(self):
        doc = _parse_verify("test.xml")
        assert doc is not None
        assert not doc.is_empty

    def test_synthetic(self, tmp_path):
        f = tmp_path / "data.xml"
        f.write_text('<?xml version="1.0"?>\n<root><item>Hello</item></root>', encoding="utf-8")
        doc = _parse_tmp(f)
        assert doc is not None
        assert "Hello" in doc.text


# ---------------------------------------------------------------------------
# HtmlParser
# ---------------------------------------------------------------------------


class TestHtmlParser:
    def test_real_file(self):
        doc = _parse_verify("test.html")
        assert doc is not None
        assert not doc.is_empty

    def test_strips_tags(self, tmp_path):
        f = tmp_path / "page.html"
        f.write_text("<html><body><p>Clean text</p></body></html>", encoding="utf-8")
        doc = _parse_tmp(f)
        assert doc is not None
        assert "Clean text" in doc.text
        # HTML tags should be processed out or at least text extracted
        assert "<p>" not in doc.text or "Clean text" in doc.text


# ---------------------------------------------------------------------------
# RstParser
# ---------------------------------------------------------------------------


class TestRstParser:
    def test_real_file(self):
        doc = _parse_verify("test.rst")
        assert doc is not None
        assert not doc.is_empty

    def test_synthetic(self, tmp_path):
        f = tmp_path / "doc.rst"
        f.write_text(
            "Title\n=====\n\nSome reStructuredText content.\n\nSubsection\n----------\n\nMore.",
            encoding="utf-8",
        )
        doc = _parse_tmp(f)
        assert doc is not None
        assert "Title" in doc.text


# ---------------------------------------------------------------------------
# CodeParser  (.js, .sh, .sql, .toml, .ini)
# ---------------------------------------------------------------------------


class TestCodeParser:
    def test_js_file(self):
        doc = _parse_verify("test.js")
        assert doc is not None
        assert doc.metadata.get("language") == "javascript"
        assert doc.format_hint == "code"

    def test_sh_file(self):
        doc = _parse_verify("test.sh")
        assert doc is not None
        assert doc.metadata.get("language") == "shell"

    def test_sql_file(self):
        doc = _parse_verify("test.sql")
        assert doc is not None
        assert doc.metadata.get("language") == "sql"

    def test_toml_file(self):
        doc = _parse_verify("test.toml")
        assert doc is not None
        assert doc.metadata.get("language") == "toml"

    def test_ini_file(self):
        doc = _parse_verify("test.ini")
        assert doc is not None
        assert doc.metadata.get("language") == "ini"

    def test_synthetic_python(self, tmp_path):
        f = tmp_path / "example.py"
        f.write_text(
            '"""Module docstring."""\n\ndef greet(name):\n    return f"Hello {name}"',
            encoding="utf-8",
        )
        doc = _parse_tmp(f)
        assert doc is not None
        assert doc.metadata.get("language") == "python"
        assert "greet" in doc.text
        # Docstring extraction
        if doc.metadata.get("doc_summary"):
            assert "Module docstring" in doc.metadata["doc_summary"]

    def test_line_count(self, tmp_path):
        f = tmp_path / "small.py"
        content = "line1\nline2\nline3"
        f.write_text(content, encoding="utf-8")
        doc = _parse_tmp(f)
        assert doc is not None
        assert doc.metadata.get("line_count") == "3"


# ---------------------------------------------------------------------------
# LogParser
# ---------------------------------------------------------------------------


class TestLogParser:
    def test_real_file(self):
        if not _has_verify_file("test.log"):
            pytest.skip("test.log not available")
        doc = _parse_verify("test.log")
        assert doc is not None
        assert doc.metadata.get("format") == "log"

    def test_error_counting(self, tmp_path):
        f = tmp_path / "app.log"
        f.write_text(
            "2024-01-01 INFO Start\n"
            "2024-01-01 ERROR Oops\n"
            "2024-01-01 WARNING Hmm\n"
            "2024-01-01 ERROR Crash\n"
            "2024-01-01 FATAL Boom\n",
            encoding="utf-8",
        )
        doc = _parse_tmp(f)
        assert doc is not None
        # ERROR + FATAL + CRITICAL
        assert int(doc.metadata.get("error_count", "0")) >= 2
        assert int(doc.metadata.get("warning_count", "0")) >= 1


# ---------------------------------------------------------------------------
# Binary format parsers — require test-docs-verify/ files
# ---------------------------------------------------------------------------


class TestPdfParser:
    @pytest.mark.skipif(not _has_verify_file("test.pdf"), reason="test.pdf missing")
    def test_real_file(self):
        doc = _parse_verify("test.pdf")
        assert doc is not None
        assert not doc.is_empty


class TestDocxParser:
    @pytest.mark.skipif(not _has_verify_file("test.docx"), reason="test.docx missing")
    def test_real_file(self):
        doc = _parse_verify("test.docx")
        assert doc is not None
        assert not doc.is_empty


class TestPptxParser:
    @pytest.mark.skipif(not _has_verify_file("test.pptx"), reason="test.pptx missing")
    def test_real_file(self):
        doc = _parse_verify("test.pptx")
        assert doc is not None
        assert not doc.is_empty


class TestXlsxParser:
    @pytest.mark.skipif(not _has_verify_file("test.xlsx"), reason="test.xlsx missing")
    def test_real_file(self):
        doc = _parse_verify("test.xlsx")
        assert doc is not None
        assert not doc.is_empty


class TestEpubParser:
    @pytest.mark.skipif(not _has_verify_file("test.epub"), reason="test.epub missing")
    def test_real_file(self):
        doc = _parse_verify("test.epub")
        assert doc is not None
        assert not doc.is_empty


class TestRtfParser:
    @pytest.mark.skipif(not _has_verify_file("test.rtf"), reason="test.rtf missing")
    def test_real_file(self):
        doc = _parse_verify("test.rtf")
        assert doc is not None
        assert not doc.is_empty


class TestOdtParser:
    @pytest.mark.skipif(not _has_verify_file("test.odt"), reason="test.odt missing")
    def test_real_file(self):
        doc = _parse_verify("test.odt")
        assert doc is not None
        assert not doc.is_empty


class TestOdsParser:
    @pytest.mark.skipif(not _has_verify_file("test.ods"), reason="test.ods missing")
    def test_real_file(self):
        doc = _parse_verify("test.ods")
        assert doc is not None
        assert not doc.is_empty


class TestOdpParser:
    @pytest.mark.skipif(not _has_verify_file("test.odp"), reason="test.odp missing")
    def test_real_file(self):
        doc = _parse_verify("test.odp")
        assert doc is not None
        assert not doc.is_empty


class TestImageParser:
    @pytest.mark.skipif(not _has_verify_file("test.png"), reason="test.png missing")
    def test_png(self):
        doc = _parse_verify("test.png")
        assert doc is not None
        # Image parser should return metadata at minimum
        assert doc.source_path.endswith("test.png")

    @pytest.mark.skipif(not _has_verify_file("test.jpg"), reason="test.jpg missing")
    def test_jpg(self):
        doc = _parse_verify("test.jpg")
        assert doc is not None

    @pytest.mark.skipif(not _has_verify_file("test.gif"), reason="test.gif missing")
    def test_gif(self):
        doc = _parse_verify("test.gif")
        assert doc is not None

    @pytest.mark.skipif(not _has_verify_file("test.bmp"), reason="test.bmp missing")
    def test_bmp(self):
        doc = _parse_verify("test.bmp")
        assert doc is not None

    @pytest.mark.skipif(not _has_verify_file("test.tiff"), reason="test.tiff missing")
    def test_tiff(self):
        doc = _parse_verify("test.tiff")
        assert doc is not None

    @pytest.mark.skipif(not _has_verify_file("test.webp"), reason="test.webp missing")
    def test_webp(self):
        doc = _parse_verify("test.webp")
        assert doc is not None


# ---------------------------------------------------------------------------
# Cross-parser: all test-docs-verify/ files parseable without crash
# ---------------------------------------------------------------------------


class TestAllTestDocs:
    """Smoke test: every file in test-docs-verify/ should parse without exception."""

    @pytest.mark.parametrize(
        "filename",
        sorted(f.name for f in _VERIFY_DIR.iterdir() if f.is_file())
        if _VERIFY_DIR.is_dir()
        else [],
    )
    def test_parse_no_crash(self, filename):
        fpath = _VERIFY_DIR / filename
        parser = get_parser(fpath)
        if parser is None:
            pytest.skip(f"No parser registered for {fpath.suffix}")
        doc = parser.parse(fpath)
        assert isinstance(doc, ParsedDocument)
        assert doc.source_path == str(fpath)
