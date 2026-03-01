"""End-to-end lifecycle test: create → index → search → export → import → delete.

Uses real ChromaDB, real embedding models, and real parsers on synthetic
documents.  No mocks — this exercises the full production code path.
"""

from __future__ import annotations

import zipfile

import pytest

pytestmark = pytest.mark.slow


# ---------------------------------------------------------------------------
# Full lifecycle
# ---------------------------------------------------------------------------


class TestE2ELifecycle:
    """Complete lifecycle test for the RAG Knowledge Base API."""

    def test_create_rag(self, api_instance, sample_docs_dir):
        """Create a new RAG and verify it appears in the registry."""
        entry = api_instance.create_rag(
            name="e2e-test",
            folders=[str(sample_docs_dir)],
            description="End-to-end test RAG",
        )
        assert entry.name == "e2e-test"
        assert entry.description == "End-to-end test RAG"
        assert str(sample_docs_dir) in entry.source_folders

        rags = api_instance.list_rags()
        assert any(r.name == "e2e-test" for r in rags)

        active = api_instance.get_active_rag()
        assert active is not None
        assert active.name == "e2e-test"

    def test_index_documents(self, api_instance, sample_docs_dir):
        """Index documents and verify progress + results."""
        api_instance.create_rag("e2e-idx", folders=[str(sample_docs_dir)])

        progress_states = []

        def on_progress(state):
            progress_states.append(state.status)

        result = api_instance.index("e2e-idx", on_progress=on_progress)

        assert result.status == "done"
        assert result.processed_files > 0
        assert result.total_chunks > 0
        assert result.duration_seconds >= 0
        assert len(result.errors) == 0

        # Verify progress callbacks fired
        assert len(progress_states) > 0

    def test_get_index_status(self, api_instance, sample_docs_dir):
        """Verify index status reflects indexed content."""
        api_instance.create_rag("e2e-status", folders=[str(sample_docs_dir)])
        api_instance.index("e2e-status")

        status = api_instance.get_index_status("e2e-status")
        assert status.active_rag == "e2e-status"
        assert status.total_files > 0
        assert status.total_chunks > 0

    def test_list_indexed_files(self, api_instance, sample_docs_dir):
        """List indexed files with pagination and filtering."""
        api_instance.create_rag("e2e-files", folders=[str(sample_docs_dir)])
        api_instance.index("e2e-files")

        # Full list
        result = api_instance.list_indexed_files("e2e-files")
        assert result.total > 0
        assert len(result.files) > 0
        assert all(f.chunk_count > 0 for f in result.files)

        # Pagination
        page = api_instance.list_indexed_files("e2e-files", offset=0, limit=2)
        assert len(page.files) <= 2
        assert page.total == result.total

        # Filter
        filtered = api_instance.list_indexed_files("e2e-files", filter=".md")
        assert all(".md" in f.file.lower() for f in filtered.files)

    def test_search(self, api_instance, sample_docs_dir):
        """Search indexed documents and verify result structure."""
        api_instance.create_rag("e2e-search", folders=[str(sample_docs_dir)])
        api_instance.index("e2e-search")

        results = api_instance.search("quantum computing", n_results=3, rag_name="e2e-search")
        assert len(results) > 0

        # Verify result structure
        first = results[0]
        assert first.text
        assert first.source_file
        assert isinstance(first.score, float)
        assert first.score > 0

        # Scores should be in descending order
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_get_document_content(self, api_instance, sample_docs_dir):
        """Retrieve chunks for a specific document."""
        api_instance.create_rag("e2e-doc", folders=[str(sample_docs_dir)])
        api_instance.index("e2e-doc")

        files = api_instance.list_indexed_files("e2e-doc")
        assert files.total > 0

        first_file = files.files[0].file
        chunks = api_instance.get_document_content(first_file, rag_name="e2e-doc")
        assert len(chunks) > 0
        assert all(c.text for c in chunks)

    def test_scan_file_changes(self, api_instance, sample_docs_dir):
        """Scan for file changes after adding a new document."""
        api_instance.create_rag("e2e-scan", folders=[str(sample_docs_dir)])
        api_instance.index("e2e-scan")

        # Add a new file
        new_file = sample_docs_dir / "new_doc.txt"
        new_file.write_text("This is a newly added document.", encoding="utf-8")

        changes = api_instance.scan_file_changes("e2e-scan")
        assert len(changes.new) > 0
        assert any("new_doc" in p for p in changes.new)

    def test_incremental_index(self, api_instance, sample_docs_dir):
        """Incremental index only processes changed files."""
        api_instance.create_rag("e2e-incr", folders=[str(sample_docs_dir)])
        result1 = api_instance.index("e2e-incr")
        first_count = result1.processed_files

        # Add a new file
        (sample_docs_dir / "extra.txt").write_text("Extra content here.", encoding="utf-8")

        result2 = api_instance.index("e2e-incr")
        # Only the new file should be processed
        assert result2.processed_files < first_count or result2.processed_files == 1

    def test_full_reindex(self, api_instance, sample_docs_dir):
        """Full reindex re-processes all files."""
        api_instance.create_rag("e2e-full", folders=[str(sample_docs_dir)])
        api_instance.index("e2e-full")

        result = api_instance.index("e2e-full", full=True)
        assert result.status == "done"
        assert result.processed_files > 0

    def test_verify_consistency(self, api_instance, sample_docs_dir):
        """Verify index consistency returns clean result after indexing."""
        api_instance.create_rag("e2e-verify", folders=[str(sample_docs_dir)])
        api_instance.index("e2e-verify")

        result = api_instance.verify_index_consistency("e2e-verify")
        assert result["ok"] is True
        assert len(result["invalidated_files"]) == 0
        assert len(result["orphan_store_files"]) == 0
        assert len(result["orphan_manifest_files"]) == 0

    def test_export_and_peek(self, api_instance, sample_docs_dir, tmp_path):
        """Export a RAG to .rag file and peek at metadata."""
        api_instance.create_rag("e2e-export", folders=[str(sample_docs_dir)])
        api_instance.index("e2e-export")

        export_path = tmp_path / "export.rag"
        result_path = api_instance.export_rag("e2e-export", str(export_path))
        assert export_path.exists() or (tmp_path / result_path).exists()

        # Verify it's a valid ZIP
        actual_path = export_path if export_path.exists() else tmp_path / result_path
        assert zipfile.is_zipfile(actual_path)

        # Peek metadata
        meta = api_instance.peek_rag_file(str(actual_path))
        assert meta["name"] == "e2e-export"

    def test_import_rag(self, api_instance, sample_docs_dir, tmp_path):
        """Import a RAG from a .rag file."""
        api_instance.create_rag("e2e-imp-src", folders=[str(sample_docs_dir)])
        api_instance.index("e2e-imp-src")

        export_path = tmp_path / "imp.rag"
        api_instance.export_rag("e2e-imp-src", str(export_path))

        imported_name = api_instance.import_rag(str(export_path), new_name="imported-rag")
        assert imported_name

        rags = api_instance.list_rags()
        names = [r.name for r in rags]
        assert "imported-rag" in names or imported_name in names

    def test_switch_rag(self, api_instance, sample_docs_dir):
        """Switch between RAGs."""
        api_instance.create_rag("rag-a", folders=[str(sample_docs_dir)])
        api_instance.create_rag("rag-b", folders=[str(sample_docs_dir)])

        api_instance.switch_rag("rag-b")
        assert api_instance.get_active_name() == "rag-b"

        api_instance.switch_rag("rag-a")
        assert api_instance.get_active_name() == "rag-a"

    def test_detach_and_attach(self, api_instance, sample_docs_dir):
        """Detach makes RAG read-only; attach re-enables indexing."""
        api_instance.create_rag("e2e-detach", folders=[str(sample_docs_dir)])
        api_instance.index("e2e-detach")

        # Detach → search still works, index fails
        api_instance.detach_rag("e2e-detach")
        entry = api_instance.get_rag("e2e-detach")
        assert entry.detached is True

        results = api_instance.search("test", rag_name="e2e-detach")
        # Search on detached RAG should still work (read-only)
        assert isinstance(results, list)

        with pytest.raises(RuntimeError, match="detached"):
            api_instance.index("e2e-detach")

        # Re-attach
        api_instance.attach_rag("e2e-detach")
        entry = api_instance.get_rag("e2e-detach")
        assert entry.detached is False

    def test_delete_rag(self, api_instance, sample_docs_dir):
        """Delete a RAG and verify it's removed."""
        api_instance.create_rag("e2e-del", folders=[str(sample_docs_dir)])
        api_instance.index("e2e-del")

        api_instance.delete_rag("e2e-del")

        rags = api_instance.list_rags()
        assert not any(r.name == "e2e-del" for r in rags)

    def test_shutdown(self, api_instance):
        """Shutdown releases all resources."""
        api_instance.create_rag("e2e-shutdown", folders=[])
        api_instance.shutdown()
        # Double shutdown should not crash
        api_instance.shutdown()


# ---------------------------------------------------------------------------
# Config management
# ---------------------------------------------------------------------------


class TestConfigManagement:
    """Config ops through the API."""

    def test_get_config(self, api_instance):
        config = api_instance.get_config()
        assert config.chunk_size > 0
        assert config.embedding_model

    def test_save_and_reload_config(self, api_instance):
        config = api_instance.get_config()
        config.chunk_size = 512
        api_instance.save_config(config)

        reloaded = api_instance.reload_config()
        assert reloaded.chunk_size == 512

    def test_reset_config(self, api_instance):
        config = api_instance.get_config()
        config.chunk_size = 999
        api_instance.save_config(config)

        default = api_instance.reset_config()
        assert default.chunk_size == 1024  # default value
