"""Edge case tests — designed to probe unhandled or boundary scenarios.

Some of these are marked xfail to document known limitations or race
conditions that should be addressed in future work.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from rag_kb.config import AppSettings, RagRegistry, _validate_rag_name


# ---------------------------------------------------------------------------
# Concurrent registry operations
# ---------------------------------------------------------------------------


class TestConcurrentRegistryAccess:
    def test_concurrent_create_different_names(self, tmp_path):
        """Multiple threads creating different RAGs concurrently."""
        reg = RagRegistry(tmp_path / "registry.json", tmp_path / "rags")
        errors = []
        created = []

        def create_rag(name: str):
            try:
                reg.create_rag(name)
                created.append(name)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=create_rag, args=(f"rag-{i}",))
            for i in range(10)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        # All should succeed (different names)
        assert len(errors) == 0
        assert len(created) == 10

    def test_concurrent_create_same_name(self, tmp_path):
        """Two threads creating the same RAG — one should fail."""
        reg = RagRegistry(tmp_path / "registry.json", tmp_path / "rags")
        results = {"success": 0, "error": 0}
        lock = threading.Lock()

        def create_rag():
            try:
                reg.create_rag("conflict")
                with lock:
                    results["success"] += 1
            except ValueError:
                with lock:
                    results["error"] += 1

        t1 = threading.Thread(target=create_rag)
        t2 = threading.Thread(target=create_rag)
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)

        assert results["success"] == 1
        assert results["error"] == 1


# ---------------------------------------------------------------------------
# RAG name edge cases  
# ---------------------------------------------------------------------------


class TestRagNameEdgeCases:
    def test_very_long_name_exactly_128(self, tmp_path):
        reg = RagRegistry(tmp_path / "registry.json", tmp_path / "rags")
        name = "a" * 128
        reg.create_rag(name)
        assert reg.get_rag(name).name == name

    def test_name_with_unicode(self, tmp_path):
        reg = RagRegistry(tmp_path / "registry.json", tmp_path / "rags")
        reg.create_rag("知識ベース")
        entry = reg.get_rag("知識ベース")
        assert entry.name == "知識ベース"

    def test_name_with_spaces(self, tmp_path):
        reg = RagRegistry(tmp_path / "registry.json", tmp_path / "rags")
        reg.create_rag("my rag base")
        assert reg.get_rag("my rag base").name == "my rag base"

    def test_name_with_dots(self, tmp_path):
        reg = RagRegistry(tmp_path / "registry.json", tmp_path / "rags")
        reg.create_rag("v1.0.0")
        assert reg.get_rag("v1.0.0").name == "v1.0.0"

    def test_name_with_hyphens_underscores(self, tmp_path):
        reg = RagRegistry(tmp_path / "registry.json", tmp_path / "rags")
        reg.create_rag("my-rag_v2")
        assert reg.get_rag("my-rag_v2").name == "my-rag_v2"


# ---------------------------------------------------------------------------
# Settings edge cases
# ---------------------------------------------------------------------------


class TestSettingsEdgeCases:
    def test_zero_chunk_overlap(self, tmp_path):
        s = AppSettings(chunk_size=512, chunk_overlap=0)
        cfg = tmp_path / "config.yaml"
        s.save(cfg)
        loaded = AppSettings.load(cfg)
        assert loaded.chunk_overlap == 0

    def test_very_small_chunk_size(self, tmp_path):
        s = AppSettings(chunk_size=10, chunk_overlap=2)
        cfg = tmp_path / "config.yaml"
        s.save(cfg)
        loaded = AppSettings.load(cfg)
        assert loaded.chunk_size == 10

    def test_alpha_boundary_values(self):
        s0 = AppSettings(hybrid_search_alpha=0.0)
        assert s0.hybrid_search_alpha == 0.0
        s1 = AppSettings(hybrid_search_alpha=1.0)
        assert s1.hybrid_search_alpha == 1.0


# ---------------------------------------------------------------------------
# Validate rag name extreme inputs
# ---------------------------------------------------------------------------


class TestValidateRagNameEdge:
    def test_single_char_ok(self):
        _validate_rag_name("a")  # Should not raise

    def test_all_forbidden_chars(self):
        for ch in '<>:"/\\|?*':
            with pytest.raises(ValueError):
                _validate_rag_name(f"name{ch}")

    def test_tab_is_ok(self):
        """Tab character is NOT in the forbidden list — should be allowed."""
        _validate_rag_name("name\twith\ttabs")


# ---------------------------------------------------------------------------
# Empty/missing states
# ---------------------------------------------------------------------------


class TestEmptyStates:
    def test_empty_registry_operations(self, tmp_path):
        reg = RagRegistry(tmp_path / "registry.json", tmp_path / "rags")
        assert reg.list_rags() == []
        assert reg.get_active_name() is None
        assert reg.get_active() is None

    def test_delete_last_rag_clears_active(self, tmp_path):
        reg = RagRegistry(tmp_path / "registry.json", tmp_path / "rags")
        reg.create_rag("only-one")
        assert reg.get_active_name() == "only-one"
        reg.delete_rag("only-one")
        assert reg.get_active_name() is None

    def test_settings_default_extensions_match_parsers(self):
        from rag_kb.parsers.registry import SUPPORTED_EXTENSIONS

        s = AppSettings()
        assert set(s.supported_extensions) == set(SUPPORTED_EXTENSIONS)


# ---------------------------------------------------------------------------
# File handling edge cases
# ---------------------------------------------------------------------------


class TestFileEdgeCases:
    def test_binary_file_parsing_no_crash(self, tmp_path):
        """Attempting to parse a binary file should not crash."""
        from rag_kb.parsers.registry import parse_file

        f = tmp_path / "binary.txt"
        f.write_bytes(b"\x00\x01\x02\xff\xfe\xfd" * 100)
        # TxtParser uses errors="replace", so should not crash
        doc = parse_file(f)
        assert doc is not None

    def test_very_long_filename(self, tmp_path):
        """A file with a very long name should still be parseable."""
        from rag_kb.parsers.registry import parse_file

        name = "a" * 200 + ".txt"
        try:
            f = tmp_path / name
            f.write_text("Content.", encoding="utf-8")
            doc = parse_file(f)
            assert doc is not None
        except OSError:
            pytest.skip("OS does not support filename this long")

    def test_file_with_bom(self, tmp_path):
        """UTF-8 BOM should not cause parsing issues."""
        from rag_kb.parsers.registry import parse_file

        f = tmp_path / "bom.txt"
        f.write_bytes(b"\xef\xbb\xbf" + "Hello BOM".encode("utf-8"))
        doc = parse_file(f)
        assert doc is not None
        assert "Hello" in doc.text


# ---------------------------------------------------------------------------
# Detached RAG behavior
# ---------------------------------------------------------------------------


class TestDetachedRag:
    def test_detached_flag_persists(self, tmp_path):
        reg = RagRegistry(tmp_path / "registry.json", tmp_path / "rags")
        entry = reg.create_rag("det-test")
        entry.detached = True
        reg.update_rag(entry)

        reg2 = RagRegistry(tmp_path / "registry.json", tmp_path / "rags")
        loaded = reg2.get_rag("det-test")
        assert loaded.detached is True


# ---------------------------------------------------------------------------
# Safe display path edge cases
# ---------------------------------------------------------------------------


import sys


class TestSafeDisplayPathEdge:
    @pytest.mark.skipif(sys.platform != "win32", reason="Windows-only path semantics")
    def test_windows_paths(self):
        from rag_kb.config import safe_display_path

        # Windows-style paths
        result = safe_display_path(
            "C:\\Users\\test\\docs\\file.txt",
            ["C:\\Users\\test\\docs"],
        )
        assert result == "file.txt"

    def test_path_with_spaces(self):
        from rag_kb.config import safe_display_path

        result = safe_display_path(
            "/data/my docs/file name.txt",
            ["/data/my docs"],
        )
        assert result == "file name.txt"
