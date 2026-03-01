"""Skip-patterns tests — directory, file, and path-level filtering."""

from __future__ import annotations

import pytest

from rag_kb.skip_patterns import (
    SKIP_DIRS,
    SKIP_FILES,
    SKIP_SUFFIXES,
    is_skipped_dir,
    is_skipped_file,
    is_skipped_path,
)


# ---------------------------------------------------------------------------
# is_skipped_dir
# ---------------------------------------------------------------------------


class TestIsSkippedDir:
    @pytest.mark.parametrize(
        "dirname",
        [
            ".git",
            ".hg",
            "node_modules",
            "__pycache__",
            "venv",
            ".venv",
            ".idea",
            ".vscode",
            "dist",
            "build",
            ".npm",
            ".yarn",
            ".tox",
            ".pytest_cache",
            ".mypy_cache",
        ],
    )
    def test_known_dirs_skipped(self, dirname):
        assert is_skipped_dir(dirname) is True

    def test_case_insensitive(self):
        assert is_skipped_dir("Node_Modules") is True
        assert is_skipped_dir(".GIT") is True
        assert is_skipped_dir("__PYCACHE__") is True

    def test_egg_info_wildcard(self):
        assert is_skipped_dir("mypackage.egg-info") is True
        assert is_skipped_dir("SomeLib.egg-info") is True

    def test_normal_dirs_pass(self):
        assert is_skipped_dir("src") is False
        assert is_skipped_dir("docs") is False
        assert is_skipped_dir("tests") is False
        assert is_skipped_dir("my_project") is False

    def test_empty_string(self):
        assert is_skipped_dir("") is False


# ---------------------------------------------------------------------------
# is_skipped_file
# ---------------------------------------------------------------------------


class TestIsSkippedFile:
    @pytest.mark.parametrize(
        "filename",
        [
            ".DS_Store",
            "Thumbs.db",
            "desktop.ini",
            ".gitignore",
            "package-lock.json",
            "yarn.lock",
            "poetry.lock",
            ".env",
        ],
    )
    def test_known_files_skipped(self, filename):
        assert is_skipped_file(filename) is True

    @pytest.mark.parametrize(
        "filename",
        [
            "module.pyc",
            "library.so",
            "app.exe",
            "bundle.min.js",
            "style.min.css",
            "data.sqlite",
            "secret.pem",
            "backup.tmp",
            "archive.zip",
            "font.woff2",
            "video.mp4",
        ],
    )
    def test_suffix_patterns(self, filename):
        assert is_skipped_file(filename) is True

    def test_normal_files_pass(self):
        assert is_skipped_file("main.py") is False
        assert is_skipped_file("README.md") is False
        assert is_skipped_file("data.json") is False
        assert is_skipped_file("styles.css") is False
        assert is_skipped_file("index.html") is False

    def test_case_insensitive_exact(self):
        assert is_skipped_file(".DS_STORE") is True
        assert is_skipped_file("THUMBS.DB") is True

    def test_empty_string(self):
        assert is_skipped_file("") is False


# ---------------------------------------------------------------------------
# is_skipped_path
# ---------------------------------------------------------------------------


class TestIsSkippedPath:
    def test_with_unix_paths(self):
        assert is_skipped_path("/project/node_modules/react/index.js") is True
        assert is_skipped_path("/project/.git/config") is True
        assert is_skipped_path("/project/__pycache__/module.pyc") is True

    def test_with_windows_backslashes(self):
        assert is_skipped_path("C:\\project\\node_modules\\react\\index.js") is True
        assert is_skipped_path("C:\\project\\.git\\config") is True

    def test_skip_file_in_normal_dir(self):
        assert is_skipped_path("/project/src/.DS_Store") is True
        assert is_skipped_path("/project/src/package-lock.json") is True

    def test_skip_suffix_in_path(self):
        assert is_skipped_path("/project/src/module.pyc") is True
        assert is_skipped_path("/project/dist/app.min.js") is True

    def test_normal_path_passes(self):
        assert is_skipped_path("/project/src/main.py") is False
        assert is_skipped_path("/project/docs/README.md") is False

    def test_empty_path(self):
        assert is_skipped_path("") is False


# ---------------------------------------------------------------------------
# Data integrity
# ---------------------------------------------------------------------------


class TestDataIntegrity:
    def test_skip_dirs_is_frozenset(self):
        assert isinstance(SKIP_DIRS, frozenset)

    def test_skip_files_is_frozenset(self):
        assert isinstance(SKIP_FILES, frozenset)

    def test_skip_suffixes_is_tuple(self):
        assert isinstance(SKIP_SUFFIXES, tuple)

    def test_dirs_are_lowercase(self):
        for d in SKIP_DIRS:
            assert d == d.lower(), f"SKIP_DIRS entry not lowercase: {d}"

    def test_files_are_lowercase(self):
        for f in SKIP_FILES:
            # Allow special chars like Icon\r
            if "\r" not in f:
                assert f == f.lower(), f"SKIP_FILES entry not lowercase: {f}"

    def test_suffixes_start_with_dot_or_tilde(self):
        for s in SKIP_SUFFIXES:
            assert s.startswith(".") or s == "~", f"Unexpected suffix: {s}"
