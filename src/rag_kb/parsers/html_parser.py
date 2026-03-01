"""HTML file parser using BeautifulSoup (MIT) + lxml (BSD)."""

from __future__ import annotations

import logging
import re
from pathlib import Path

from rag_kb.parsers.base import ParsedDocument

logger = logging.getLogger(__name__)

_RE_MULTI_NEWLINES = re.compile(r"\n{3,}")


class HtmlParser:
    """Parse HTML files, extracting meaningful text content."""

    supported_extensions: list[str] = [".html", ".htm"]

    def parse(self, file_path: Path) -> ParsedDocument:
        try:
            from bs4 import BeautifulSoup
        except ImportError as exc:
            raise ImportError(
                "beautifulsoup4 is required for HTML parsing: pip install beautifulsoup4 lxml"
            ) from exc

        raw = file_path.read_text(encoding="utf-8", errors="replace")
        soup = BeautifulSoup(raw, "lxml")

        # Remove script and style elements
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        # Get the title
        title = ""
        if soup.title and soup.title.string:
            title = soup.title.string.strip()

        # Extract text
        text = soup.get_text(separator="\n", strip=True)

        # Collapse multiple newlines
        text = _RE_MULTI_NEWLINES.sub("\n\n", text)

        if title:
            text = f"{title}\n\n{text}"

        return ParsedDocument(
            text=text,
            source_path=str(file_path),
            metadata={
                "format": "html",
                "title": title,
            },
        )
