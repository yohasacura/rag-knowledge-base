"""Parser registry — maps file extensions to parser implementations."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from rag_kb.parsers.base import DocumentParser, ParsedDocument
from rag_kb.parsers.markdown_parser import MarkdownParser
from rag_kb.parsers.txt_parser import TxtParser
from rag_kb.parsers.pdf_parser import PdfParser
from rag_kb.parsers.docx_parser import DocxParser
from rag_kb.parsers.pptx_parser import PptxParser
from rag_kb.parsers.html_parser import HtmlParser
from rag_kb.parsers.xlsx_parser import XlsxParser
from rag_kb.parsers.csv_parser import CsvParser
from rag_kb.parsers.json_parser import JsonParser
from rag_kb.parsers.yaml_parser import YamlParser
from rag_kb.parsers.xml_parser import XmlParser
from rag_kb.parsers.rst_parser import RstParser
from rag_kb.parsers.rtf_parser import RtfParser
from rag_kb.parsers.odf_parser import OdtParser, OdsParser, OdpParser
from rag_kb.parsers.epub_parser import EpubParser
from rag_kb.parsers.code_parser import CodeParser
from rag_kb.parsers.log_parser import LogParser
from rag_kb.parsers.image_parser import ImageParser

# Instantiate parsers once — order matters: more specific parsers first,
# TxtParser last as a catch-all for remaining plain text formats.
_PARSERS: list[DocumentParser] = [
    MarkdownParser(),
    PdfParser(),
    DocxParser(),
    PptxParser(),
    XlsxParser(),
    HtmlParser(),
    CsvParser(),
    JsonParser(),
    YamlParser(),
    XmlParser(),
    RstParser(),
    RtfParser(),
    OdtParser(),
    OdsParser(),
    OdpParser(),
    EpubParser(),
    ImageParser(),
    CodeParser(),
    LogParser(),
    TxtParser(),
]

# Build extension → parser lookup
_EXT_MAP: dict[str, DocumentParser] = {}
for _parser in _PARSERS:
    for _ext in _parser.supported_extensions:
        _EXT_MAP[_ext.lower()] = _parser

SUPPORTED_EXTENSIONS: list[str] = sorted(_EXT_MAP.keys())


def get_parser(file_path: Path) -> DocumentParser | None:
    """Return the appropriate parser for a file, or None if unsupported."""
    ext = file_path.suffix.lower()
    return _EXT_MAP.get(ext)


def parse_file(file_path: Path) -> ParsedDocument | None:
    """Parse a file using the appropriate parser. Returns None if unsupported."""
    parser = get_parser(file_path)
    if parser is None:
        return None
    return parser.parse(file_path)
