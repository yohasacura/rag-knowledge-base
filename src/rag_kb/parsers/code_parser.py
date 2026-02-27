"""Source code file parser — reads code with language-aware metadata."""

from __future__ import annotations

import re
from pathlib import Path

from rag_kb.parsers.base import DocumentParser, ParsedDocument

# Pre-compiled regex patterns for doc-comment extraction
_RE_PY_DOCSTRING_DQ = re.compile(r'"""(.*?)"""', re.DOTALL)
_RE_PY_DOCSTRING_SQ = re.compile(r"'''(.*?)'''", re.DOTALL)
_RE_C_BLOCK_COMMENT = re.compile(r"/\*\*?(.*?)\*/", re.DOTALL)
_RE_C_COMMENT_STARS = re.compile(r"^\s*\*\s?", re.MULTILINE)

# Extension → language name mapping
_LANG_MAP: dict[str, str] = {
    ".py": "python",
    ".pyw": "python",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".jsx": "javascript-react",
    ".ts": "typescript",
    ".tsx": "typescript-react",
    ".java": "java",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".scala": "scala",
    ".c": "c",
    ".h": "c-header",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".cc": "cpp",
    ".hpp": "cpp-header",
    ".hxx": "cpp-header",
    ".cs": "csharp",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".m": "objective-c",
    ".mm": "objective-cpp",
    ".r": "r",
    ".R": "r",
    ".lua": "lua",
    ".pl": "perl",
    ".pm": "perl",
    ".sh": "shell",
    ".bash": "shell",
    ".zsh": "shell",
    ".fish": "shell",
    ".ps1": "powershell",
    ".psm1": "powershell",
    ".bat": "batch",
    ".cmd": "batch",
    ".sql": "sql",
    ".dart": "dart",
    ".ex": "elixir",
    ".exs": "elixir",
    ".erl": "erlang",
    ".hrl": "erlang",
    ".hs": "haskell",
    ".ml": "ocaml",
    ".mli": "ocaml",
    ".fs": "fsharp",
    ".fsx": "fsharp",
    ".clj": "clojure",
    ".cljs": "clojure",
    ".groovy": "groovy",
    ".gradle": "groovy",
    ".vim": "vim",
    ".el": "emacs-lisp",
    ".zig": "zig",
    ".v": "v",
    ".nim": "nim",
    ".tf": "terraform",
    ".hcl": "hcl",
    ".proto": "protobuf",
    ".graphql": "graphql",
    ".gql": "graphql",
    ".vue": "vue",
    ".svelte": "svelte",
    # Config / data formats treated as code
    ".toml": "toml",
    ".ini": "ini",
    ".cfg": "ini",
    ".conf": "config",
    ".env": "dotenv",
    ".properties": "properties",
    ".makefile": "makefile",
    ".cmake": "cmake",
    ".dockerfile": "dockerfile",
}

# Special filenames (no extension) → language
_NAME_MAP: dict[str, str] = {
    "makefile": "makefile",
    "dockerfile": "dockerfile",
    "jenkinsfile": "groovy",
    "vagrantfile": "ruby",
    "gemfile": "ruby",
    "rakefile": "ruby",
    "cmakelists.txt": "cmake",
}


class CodeParser:
    """Parse source code files, preserving content with language metadata."""

    supported_extensions: list[str] = sorted(_LANG_MAP.keys())

    def parse(self, file_path: Path) -> ParsedDocument:
        text = file_path.read_text(encoding="utf-8", errors="replace")

        ext = file_path.suffix.lower()
        lang = _LANG_MAP.get(ext, "")
        if not lang:
            lang = _NAME_MAP.get(file_path.name.lower(), "unknown")

        # Extract doc-comments and module-level docstrings for extra context
        doc_summary = _extract_doc_summary(text, lang)

        metadata: dict[str, str] = {
            "format": "code",
            "language": lang,
            "line_count": str(text.count("\n") + 1),
        }
        if doc_summary:
            metadata["doc_summary"] = doc_summary

        # Build title from filename
        code_title = f"{file_path.name} ({lang})"

        return ParsedDocument(
            text=text,
            source_path=str(file_path),
            metadata=metadata,
            title=code_title,
            format_hint="code",
        )


def _extract_doc_summary(text: str, lang: str) -> str:
    """Extract a short documentation summary from the top of a source file."""
    lines = text.split("\n", 30)[:30]  # first 30 lines only
    header = "\n".join(lines)

    # Python docstrings
    if lang == "python":
        m = _RE_PY_DOCSTRING_DQ.search(header)
        if not m:
            m = _RE_PY_DOCSTRING_SQ.search(header)
        if m:
            return m.group(1).strip()[:200]

    # C-style block comments (Java, JS, TS, C, C++, Go, Rust, etc.)
    if lang in (
        "javascript", "typescript", "java", "c", "cpp", "csharp",
        "go", "rust", "kotlin", "scala", "swift", "dart", "php",
    ):
        m = _RE_C_BLOCK_COMMENT.search(header)
        if m:
            comment = m.group(1)
            # Strip leading * from each line
            comment = _RE_C_COMMENT_STARS.sub("", comment)
            return comment.strip()[:200]

    # Hash-style comments (shell, Ruby, Perl, Python, R)
    comment_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#") and not stripped.startswith("#!"):
            comment_lines.append(stripped.lstrip("# "))
        elif comment_lines:
            break
    if comment_lines:
        return " ".join(comment_lines)[:200]

    return ""
