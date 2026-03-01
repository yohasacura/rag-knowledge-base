"""Parser registry — maps file extensions to parser implementations.

Parsers are registered lazily: only the extension→module mapping is built
at import time.  The actual parser class is imported and instantiated on
first use, avoiding heavy dependency imports (pypdf, python-docx, etc.)
until a file of that type is actually encountered.
"""

from __future__ import annotations

import importlib
import logging
from pathlib import Path

from rag_kb.parsers.base import DocumentParser, ParsedDocument

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy parser registry
# ---------------------------------------------------------------------------

# Extension → (module_path, class_name) mapping.
# Parser classes are only instantiated on first use.
_PARSER_SPECS: list[tuple[str, str]] = [
    ("rag_kb.parsers.markdown_parser", "MarkdownParser"),
    ("rag_kb.parsers.pdf_parser", "PdfParser"),
    ("rag_kb.parsers.docx_parser", "DocxParser"),
    ("rag_kb.parsers.pptx_parser", "PptxParser"),
    ("rag_kb.parsers.xlsx_parser", "XlsxParser"),
    ("rag_kb.parsers.html_parser", "HtmlParser"),
    ("rag_kb.parsers.csv_parser", "CsvParser"),
    ("rag_kb.parsers.json_parser", "JsonParser"),
    ("rag_kb.parsers.yaml_parser", "YamlParser"),
    ("rag_kb.parsers.xml_parser", "XmlParser"),
    ("rag_kb.parsers.rst_parser", "RstParser"),
    ("rag_kb.parsers.rtf_parser", "RtfParser"),
    ("rag_kb.parsers.odf_parser", "OdtParser"),
    ("rag_kb.parsers.odf_parser", "OdsParser"),
    ("rag_kb.parsers.odf_parser", "OdpParser"),
    ("rag_kb.parsers.epub_parser", "EpubParser"),
    ("rag_kb.parsers.image_parser", "ImageParser"),
    ("rag_kb.parsers.code_parser", "CodeParser"),
    ("rag_kb.parsers.log_parser", "LogParser"),
    ("rag_kb.parsers.txt_parser", "TxtParser"),
]

# Extension → (module_path, class_name) lookup — built eagerly from
# the parser classes' supported_extensions without importing them.
# We import each module *once* just to read extensions, but the heavy
# dependencies (pypdf, etc.) are deferred to the parse() call.
_EXT_MAP: dict[str, tuple[str, str]] = {}
_PARSER_CACHE: dict[str, DocumentParser] = {}  # class_name → instance


def _build_ext_map() -> None:
    """Build the extension → parser spec mapping.

    Imports each parser module to read its ``supported_extensions`` class
    attribute.  This is done once at first access.
    """
    for module_path, class_name in _PARSER_SPECS:
        try:
            mod = importlib.import_module(module_path)
            cls = getattr(mod, class_name)
            for ext in cls.supported_extensions:
                _EXT_MAP[ext.lower()] = (module_path, class_name)
        except Exception as exc:
            logger.debug("Could not load parser %s.%s: %s", module_path, class_name, exc)


_build_ext_map()

SUPPORTED_EXTENSIONS: list[str] = sorted(_EXT_MAP.keys())


def _get_parser_instance(module_path: str, class_name: str) -> DocumentParser:
    """Lazily instantiate a parser, caching by class name."""
    if class_name not in _PARSER_CACHE:
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
        _PARSER_CACHE[class_name] = cls()
    return _PARSER_CACHE[class_name]


def get_parser(file_path: Path) -> DocumentParser | None:
    """Return the appropriate parser for a file, or None if unsupported."""
    ext = file_path.suffix.lower()
    spec = _EXT_MAP.get(ext)
    if spec is None:
        return None
    module_path, class_name = spec
    return _get_parser_instance(module_path, class_name)


def parse_file(file_path: Path) -> ParsedDocument | None:
    """Parse a file using the appropriate parser. Returns None if unsupported."""
    parser = get_parser(file_path)
    if parser is None:
        return None
    return parser.parse(file_path)
