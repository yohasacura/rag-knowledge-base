"""Extended FileManifest tests — batch operations, concurrency, edge cases."""

from __future__ import annotations

import threading

import pytest

from rag_kb.file_manifest import FileManifest


@pytest.fixture
def manifest(tmp_path):
    """Create a FileManifest backed by a temp SQLite DB."""
    db = tmp_path / "file_manifest.db"
    m = FileManifest(str(db))
    yield m
    m.close()


@pytest.fixture
def populated_manifest(manifest, tmp_path):
    """Manifest with 5 pre-indexed files."""
    files = []
    for i in range(5):
        f = tmp_path / f"doc{i}.txt"
        f.write_text(f"Content of document {i}", encoding="utf-8")
        manifest.mark_indexed(str(f), chunk_count=1)
        files.append(str(f))
    return manifest, files


# ---------------------------------------------------------------------------
# batch_filter_changed
# ---------------------------------------------------------------------------


class TestBatchFilterChanged:
    def test_all_unchanged(self, populated_manifest):
        manifest, files = populated_manifest
        changed = manifest.batch_filter_changed(files)
        assert changed == []

    def test_mix_changed_unchanged(self, populated_manifest, tmp_path):
        manifest, files = populated_manifest
        # Modify one file
        f0 = tmp_path / "doc0.txt"
        f0.write_text("Modified content!", encoding="utf-8")

        changed = manifest.batch_filter_changed(files)
        assert str(f0) in changed
        assert len(changed) == 1

    def test_new_file_detected(self, populated_manifest, tmp_path):
        manifest, files = populated_manifest
        new_f = tmp_path / "new.txt"
        new_f.write_text("Brand new file.", encoding="utf-8")

        all_files = files + [str(new_f)]
        changed = manifest.batch_filter_changed(all_files)
        assert str(new_f) in changed


# ---------------------------------------------------------------------------
# batch_mark_indexed
# ---------------------------------------------------------------------------


class TestBatchMarkIndexed:
    def test_batch_mark(self, manifest, tmp_path):
        files = []
        for i in range(10):
            f = tmp_path / f"batch{i}.txt"
            f.write_text(f"Content {i}", encoding="utf-8")
            files.append(str(f))

        manifest.batch_mark_indexed([(f, 1) for f in files])
        assert manifest.count() == 10

        # None should be marked as changed now
        for f in files:
            assert not manifest.is_changed(f)


# ---------------------------------------------------------------------------
# batch_invalidate
# ---------------------------------------------------------------------------


class TestBatchInvalidate:
    def test_batch_invalidate(self, populated_manifest):
        manifest, files = populated_manifest
        manifest.batch_invalidate(files[:3])

        # Invalidated files should be detected as changed
        for f in files[:3]:
            assert manifest.is_changed(f)

        # Remaining should be unchanged
        for f in files[3:]:
            assert not manifest.is_changed(f)


# ---------------------------------------------------------------------------
# batch_remove
# ---------------------------------------------------------------------------


class TestBatchRemove:
    def test_batch_remove(self, populated_manifest):
        manifest, files = populated_manifest
        manifest.batch_remove(files[:2])
        assert manifest.count() == 3
        remaining = manifest.all_paths()
        assert files[0] not in remaining
        assert files[1] not in remaining


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestFileManifestEdgeCases:
    def test_is_changed_nonexistent_file(self, manifest):
        """is_changed on a file that doesn't exist in manifest should return True."""
        assert manifest.is_changed("/nonexistent/file.txt") is True

    def test_file_content_unchanged_after_mark(self, manifest, tmp_path):
        """After mark_indexed, same content should not be changed."""
        f = tmp_path / "stable.txt"
        f.write_text("Stable content.", encoding="utf-8")
        manifest.mark_indexed(str(f), chunk_count=1)
        assert manifest.is_changed(str(f)) is False

    def test_file_content_modified(self, manifest, tmp_path):
        """Modified file content should be detected as changed."""
        f = tmp_path / "modify.txt"
        f.write_text("Original.", encoding="utf-8")
        manifest.mark_indexed(str(f), chunk_count=1)

        f.write_text("Modified!", encoding="utf-8")
        assert manifest.is_changed(str(f)) is True

    def test_file_touched_same_content(self, manifest, tmp_path):
        """Re-writing the same content should NOT be detected as changed (xxHash-based)."""
        f = tmp_path / "touch.txt"
        f.write_text("Same content.", encoding="utf-8")
        manifest.mark_indexed(str(f), chunk_count=1)

        # Re-write same content (different mtime)
        f.write_text("Same content.", encoding="utf-8")
        assert manifest.is_changed(str(f)) is False

    def test_clear(self, populated_manifest):
        manifest, _ = populated_manifest
        manifest.clear()
        assert manifest.count() == 0

    def test_all_paths(self, populated_manifest):
        manifest, files = populated_manifest
        paths = manifest.all_paths()
        assert isinstance(paths, set)
        assert len(paths) == 5

    def test_get_record(self, populated_manifest):
        manifest, files = populated_manifest
        rec = manifest.get_record(files[0])
        assert rec is not None
        assert rec.content_hash  # should have a hash

    def test_get_record_nonexistent(self, manifest):
        rec = manifest.get_record("/no/such/file.txt")
        assert rec is None


# ---------------------------------------------------------------------------
# Concurrent access
# ---------------------------------------------------------------------------


class TestFileManifestConcurrency:
    def test_concurrent_mark_and_check(self, manifest, tmp_path):
        """Two threads marking and checking the same manifest."""
        errors = []
        files = []
        for i in range(20):
            f = tmp_path / f"conc{i}.txt"
            f.write_text(f"Content {i}", encoding="utf-8")
            files.append(str(f))

        def writer():
            try:
                for f in files[:10]:
                    manifest.mark_indexed(f, chunk_count=1)
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for f in files[10:]:
                    manifest.is_changed(f)
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=writer)
        t2 = threading.Thread(target=reader)
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        assert errors == [], f"Concurrent access errors: {errors}"
