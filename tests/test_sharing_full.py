"""Full sharing tests — export/import round-trip, peek, security hardening."""

from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path

import pytest

from rag_kb.config import RagRegistry
from rag_kb.sharing import (
    MAX_MEMBER_COUNT,
    MANIFEST_NAME,
    _check_path_traversal,
    _validate_manifest,
    _validate_zip_safety,
    export_rag,
    import_rag,
    peek_rag_file,
)


@pytest.fixture
def registry(tmp_path):
    return RagRegistry(tmp_path / "registry.json", tmp_path / "rags")


@pytest.fixture
def sample_rag(registry, tmp_path):
    """Create a RAG with a fake chroma_db directory."""
    entry = registry.create_rag("sample", description="Test RAG")
    db_path = Path(entry.db_path)
    db_path.mkdir(parents=True, exist_ok=True)
    # Create some fake chroma files
    (db_path / "index.bin").write_bytes(b"\x00" * 100)
    (db_path / "metadata.json").write_text('{"key": "value"}', encoding="utf-8")
    sub = db_path / "segments"
    sub.mkdir()
    (sub / "data.bin").write_bytes(b"\x01" * 200)
    # Update entry counts
    entry.file_count = 5
    entry.chunk_count = 50
    registry.update_rag(entry)
    return entry


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


class TestExportRag:
    def test_basic_export(self, registry, sample_rag, tmp_path):
        output = str(tmp_path / "export" / "sample.rag")
        result = export_rag(registry, "sample", output)
        assert Path(result).exists()
        assert result.endswith(".rag")

    def test_export_auto_suffix(self, registry, sample_rag, tmp_path):
        output = str(tmp_path / "export" / "noext")
        result = export_rag(registry, "sample", output)
        assert result.endswith(".rag")

    def test_exported_zip_has_manifest(self, registry, sample_rag, tmp_path):
        output = str(tmp_path / "exported.rag")
        export_rag(registry, "sample", output)
        with zipfile.ZipFile(output, "r") as zf:
            assert MANIFEST_NAME in zf.namelist()
            manifest = json.loads(zf.read(MANIFEST_NAME))
            assert manifest["name"] == "sample"
            assert manifest["file_count"] == 5

    def test_exported_zip_has_chroma_files(self, registry, sample_rag, tmp_path):
        output = str(tmp_path / "exported.rag")
        export_rag(registry, "sample", output)
        with zipfile.ZipFile(output, "r") as zf:
            names = zf.namelist()
            assert any("chroma_db/index.bin" in n for n in names)
            assert any("chroma_db/metadata.json" in n for n in names)

    def test_export_nonexistent_rag(self, registry, tmp_path):
        with pytest.raises(KeyError):
            export_rag(registry, "nonexistent", str(tmp_path / "out.rag"))


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------


class TestImportRag:
    def test_basic_import(self, registry, sample_rag, tmp_path):
        output = str(tmp_path / "exported.rag")
        export_rag(registry, "sample", output)

        # Import under a different name
        name = import_rag(registry, output, new_name="imported",
                          rags_dir=tmp_path / "rags")
        assert name == "imported"
        entry = registry.get_rag("imported")
        assert entry.is_imported is True

    def test_import_auto_rename_on_collision(self, registry, sample_rag, tmp_path):
        output = str(tmp_path / "exported.rag")
        export_rag(registry, "sample", output)

        # Import without specifying name — will collide with "sample"
        name = import_rag(registry, output, rags_dir=tmp_path / "rags")
        assert name != "sample"  # should be "sample-2" or similar
        assert name in [r.name for r in registry.list_rags()]

    def test_import_nonexistent_file(self, registry, tmp_path):
        with pytest.raises(FileNotFoundError):
            import_rag(registry, str(tmp_path / "ghost.rag"),
                       rags_dir=tmp_path / "rags")


# ---------------------------------------------------------------------------
# Peek
# ---------------------------------------------------------------------------


class TestPeekRagFile:
    def test_peek(self, registry, sample_rag, tmp_path):
        output = str(tmp_path / "peek_test.rag")
        export_rag(registry, "sample", output)

        info = peek_rag_file(output)
        assert info["name"] == "sample"
        assert "file_size_mb" in info

    def test_peek_nonexistent(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            peek_rag_file(str(tmp_path / "nope.rag"))


# ---------------------------------------------------------------------------
# ZIP safety validation
# ---------------------------------------------------------------------------


class TestZipSafety:
    def test_path_traversal_in_members(self, tmp_path):
        """ZIP with ../../../etc/passwd members should be rejected."""
        bad_zip = tmp_path / "bad.rag"
        with zipfile.ZipFile(str(bad_zip), "w") as zf:
            zf.writestr(MANIFEST_NAME, '{"name":"x","embedding_model":"m"}')
            zf.writestr("../../../etc/passwd", "pwned")

        with zipfile.ZipFile(str(bad_zip), "r") as zf:
            with pytest.raises(ValueError, match="path traversal"):
                _validate_zip_safety(zf)

    def test_path_traversal_check_function(self, tmp_path):
        target = tmp_path / "safe"
        target.mkdir()
        with pytest.raises(ValueError, match="[Pp]ath traversal"):
            _check_path_traversal("../../etc/passwd", target)

    def test_safe_path_ok(self, tmp_path):
        target = tmp_path / "safe"
        target.mkdir()
        _check_path_traversal("subdir/file.txt", target)  # should not raise


# ---------------------------------------------------------------------------
# Manifest validation
# ---------------------------------------------------------------------------


class TestManifestValidation:
    def test_valid_manifest(self):
        _validate_manifest({"name": "test", "embedding_model": "model"})  # OK

    def test_missing_name(self):
        with pytest.raises(ValueError, match="name"):
            _validate_manifest({"embedding_model": "model"})

    def test_missing_model(self):
        with pytest.raises(ValueError, match="embedding_model"):
            _validate_manifest({"name": "test"})
