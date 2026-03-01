"""Tests for security hardening — ZIP imports and path traversal protection."""

from __future__ import annotations

import io
import json
import zipfile

import pytest

from rag_kb.sharing import (
    MAX_INDIVIDUAL_FILE_SIZE,
    MAX_MEMBER_COUNT,
    MAX_TOTAL_UNCOMPRESSED_SIZE,
    _check_path_traversal,
    _validate_zip_safety,
)


class TestValidateZipSafety:
    """Tests for _validate_zip_safety()."""

    def _make_zip(self, members: dict[str, bytes | str]) -> zipfile.ZipFile:
        """Create an in-memory ZIP with given members."""
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for name, data in members.items():
                if isinstance(data, str):
                    data = data.encode("utf-8")
                zf.writestr(name, data)
        buf.seek(0)
        return zipfile.ZipFile(buf, "r")

    def test_valid_zip_passes(self):
        """A normal ZIP with a manifest and chroma files passes validation."""
        manifest = json.dumps({"name": "test", "embedding_model": "all-MiniLM-L6-v2"})
        zf = self._make_zip({
            "manifest.json": manifest,
            "chroma_db/data.bin": b"\x00" * 100,
        })
        # Should not raise
        _validate_zip_safety(zf)

    def test_rejects_path_traversal_dotdot(self):
        """ZIP members with ../ in the path are rejected."""
        zf = self._make_zip({"../../../etc/passwd": "pwned"})
        with pytest.raises(ValueError, match="path traversal"):
            _validate_zip_safety(zf)

    def test_rejects_absolute_path(self):
        """ZIP members with absolute paths are rejected."""
        zf = self._make_zip({"/etc/passwd": "pwned"})
        with pytest.raises(ValueError, match="path traversal"):
            _validate_zip_safety(zf)

    def test_rejects_too_many_members(self):
        """ZIP with more members than MAX_MEMBER_COUNT is rejected."""
        members = {f"file_{i}.txt": "x" for i in range(MAX_MEMBER_COUNT + 1)}
        zf = self._make_zip(members)
        with pytest.raises(ValueError, match="zip-bomb|Too many"):
            _validate_zip_safety(zf)

    def test_rejects_oversized_total(self):
        """ZIP whose total uncompressed size exceeds the limit is rejected."""
        from unittest.mock import patch as mock_patch

        zf = self._make_zip({"file.bin": b"x"})
        # Patch the constant to a tiny value so even 1 byte exceeds it
        with mock_patch("rag_kb.sharing.MAX_TOTAL_UNCOMPRESSED_SIZE", 0):
            with pytest.raises(ValueError, match="exceeds.*limit"):
                _validate_zip_safety(zf)

    def test_rejects_oversized_individual_file(self):
        """ZIP with a single file exceeding individual limit is rejected."""
        from unittest.mock import patch as mock_patch

        zf = self._make_zip({"big.bin": b"x" * 100})
        with mock_patch("rag_kb.sharing.MAX_INDIVIDUAL_FILE_SIZE", 1):
            with pytest.raises(ValueError, match="GB"):
                _validate_zip_safety(zf)


class TestCheckPathTraversal:
    """Tests for _check_path_traversal()."""

    def test_normal_relative_path(self, tmp_path):
        """A simple relative path within target_dir is accepted."""
        target = tmp_path / "chroma_db"
        target.mkdir()
        # Should not raise
        _check_path_traversal("data/file.bin", target)

    def test_dotdot_traversal(self, tmp_path):
        """A path with ../ that escapes the target is rejected."""
        target = tmp_path / "chroma_db"
        target.mkdir()
        with pytest.raises(ValueError, match="traversal"):
            _check_path_traversal("../../evil.txt", target)

    def test_deeply_nested_is_ok(self, tmp_path):
        """A deeply nested but valid path is accepted."""
        target = tmp_path / "chroma_db"
        target.mkdir()
        _check_path_traversal("a/b/c/d/e/file.bin", target)
