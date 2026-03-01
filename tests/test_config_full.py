"""Full config & registry tests — AppSettings, RagRegistry, validation, helpers."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from rag_kb.config import (
    AppSettings,
    RagRegistry,
    _validate_rag_name,
    safe_display_path,
)


# ---------------------------------------------------------------------------
# AppSettings
# ---------------------------------------------------------------------------


class TestAppSettings:
    def test_defaults(self):
        s = AppSettings()
        assert s.embedding_model == "paraphrase-multilingual-MiniLM-L12-v2"
        assert s.chunk_size == 1024
        assert s.chunk_overlap == 128
        assert s.reranking_enabled is True
        assert s.hybrid_search_enabled is True
        assert 0 < s.hybrid_search_alpha <= 1
        assert s.min_score_threshold >= 0

    def test_save_load_roundtrip(self, tmp_path):
        cfg = tmp_path / "config.yaml"
        s = AppSettings(chunk_size=512, chunk_overlap=64)
        s.save(cfg)
        assert cfg.exists()

        loaded = AppSettings.load(cfg)
        assert loaded.chunk_size == 512
        assert loaded.chunk_overlap == 64
        # supported_extensions should be populated from parser registry, not from file
        assert len(loaded.supported_extensions) > 0

    def test_load_creates_default_if_missing(self, tmp_path):
        cfg = tmp_path / "nonexistent" / "config.yaml"
        loaded = AppSettings.load(cfg)
        assert cfg.exists()
        assert loaded.embedding_model == "paraphrase-multilingual-MiniLM-L12-v2"

    def test_save_strips_extensions(self, tmp_path):
        cfg = tmp_path / "config.yaml"
        s = AppSettings()
        s.save(cfg)
        raw = cfg.read_text(encoding="utf-8")
        assert "supported_extensions" not in raw

    def test_api_key_env_override(self, monkeypatch):
        s = AppSettings(openai_api_key="stored-key")
        assert s.resolve_openai_api_key() == "stored-key"

        monkeypatch.setenv("RAG_KB_OPENAI_API_KEY", "env-key")
        assert s.resolve_openai_api_key() == "env-key"

    def test_voyage_key_env_override(self, monkeypatch):
        s = AppSettings(voyage_api_key="stored")
        assert s.resolve_voyage_api_key() == "stored"

        monkeypatch.setenv("RAG_KB_VOYAGE_API_KEY", "env-voyage")
        assert s.resolve_voyage_api_key() == "env-voyage"

    def test_save_omits_api_keys_from_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("RAG_KB_OPENAI_API_KEY", "secret")
        cfg = tmp_path / "config.yaml"
        s = AppSettings(openai_api_key="should-not-appear")
        s.save(cfg)
        raw = cfg.read_text(encoding="utf-8")
        assert "should-not-appear" not in raw

    def test_custom_fields_persist(self, tmp_path):
        cfg = tmp_path / "config.yaml"
        s = AppSettings(
            hnsw_ef_construction=400,
            hnsw_m=64,
            embedding_batch_size=512,
        )
        s.save(cfg)
        loaded = AppSettings.load(cfg)
        assert loaded.hnsw_ef_construction == 400
        assert loaded.hnsw_m == 64
        assert loaded.embedding_batch_size == 512


# ---------------------------------------------------------------------------
# RagRegistry
# ---------------------------------------------------------------------------


class TestRagRegistry:
    def test_create_and_list(self, tmp_path):
        reg = RagRegistry(tmp_path / "registry.json", tmp_path / "rags")
        reg.create_rag("first", description="Test RAG")
        rags = reg.list_rags()
        assert len(rags) == 1
        assert rags[0].name == "first"

    def test_active_set_on_first_create(self, tmp_path):
        reg = RagRegistry(tmp_path / "registry.json", tmp_path / "rags")
        reg.create_rag("alpha")
        assert reg.get_active_name() == "alpha"

    def test_create_duplicate_raises(self, tmp_path):
        reg = RagRegistry(tmp_path / "registry.json", tmp_path / "rags")
        reg.create_rag("dup")
        with pytest.raises(ValueError, match="already exists"):
            reg.create_rag("dup")

    def test_get_rag(self, tmp_path):
        reg = RagRegistry(tmp_path / "registry.json", tmp_path / "rags")
        reg.create_rag("myrag")
        entry = reg.get_rag("myrag")
        assert entry.name == "myrag"

    def test_get_nonexistent_raises(self, tmp_path):
        reg = RagRegistry(tmp_path / "registry.json", tmp_path / "rags")
        with pytest.raises(KeyError, match="does not exist"):
            reg.get_rag("no-such")

    def test_delete_rag(self, tmp_path):
        reg = RagRegistry(tmp_path / "registry.json", tmp_path / "rags")
        reg.create_rag("todelete")
        reg.delete_rag("todelete")
        assert len(reg.list_rags()) == 0
        assert reg.get_active_name() is None

    def test_delete_nonexistent_raises(self, tmp_path):
        reg = RagRegistry(tmp_path / "registry.json", tmp_path / "rags")
        with pytest.raises(KeyError):
            reg.delete_rag("ghost")

    def test_set_active(self, tmp_path):
        reg = RagRegistry(tmp_path / "registry.json", tmp_path / "rags")
        reg.create_rag("a")
        reg.create_rag("b")
        reg.set_active("b")
        assert reg.get_active_name() == "b"

    def test_update_rag(self, tmp_path):
        reg = RagRegistry(tmp_path / "registry.json", tmp_path / "rags")
        reg.create_rag("up")
        entry = reg.get_rag("up")
        entry.description = "Updated description"
        entry.file_count = 42
        reg.update_rag(entry)

        # Verify persistence
        reg2 = RagRegistry(tmp_path / "registry.json", tmp_path / "rags")
        assert reg2.get_rag("up").description == "Updated description"
        assert reg2.get_rag("up").file_count == 42

    def test_persistence_across_instances(self, tmp_path):
        path = tmp_path / "registry.json"
        rags_dir = tmp_path / "rags"
        reg1 = RagRegistry(path, rags_dir)
        reg1.create_rag("persist1")
        reg1.create_rag("persist2")
        reg1.set_active("persist2")

        reg2 = RagRegistry(path, rags_dir)
        names = [r.name for r in reg2.list_rags()]
        assert "persist1" in names
        assert "persist2" in names
        assert reg2.get_active_name() == "persist2"

    def test_active_falls_back_after_delete(self, tmp_path):
        reg = RagRegistry(tmp_path / "registry.json", tmp_path / "rags")
        reg.create_rag("a")
        reg.create_rag("b")
        reg.set_active("a")
        reg.delete_rag("a")
        # Should fall back to remaining rag
        assert reg.get_active_name() == "b"


# ---------------------------------------------------------------------------
# _validate_rag_name
# ---------------------------------------------------------------------------


class TestValidateRagName:
    def test_valid_name(self):
        _validate_rag_name("my-rag-2024")  # should not raise

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            _validate_rag_name("")

    def test_whitespace_only_raises(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            _validate_rag_name("   ")

    def test_forbidden_chars(self):
        for ch in '<>:"/\\|?*':
            with pytest.raises(ValueError, match="forbidden"):
                _validate_rag_name(f"bad{ch}name")

    def test_max_length(self):
        _validate_rag_name("a" * 128)  # exactly 128 should be fine
        with pytest.raises(ValueError, match="128"):
            _validate_rag_name("a" * 129)

    def test_unicode_name_ok(self):
        _validate_rag_name("知識ベース")  # unicode should be fine

    def test_name_with_spaces(self):
        _validate_rag_name("my rag name")  # spaces are allowed


# ---------------------------------------------------------------------------
# safe_display_path
# ---------------------------------------------------------------------------


class TestSafeDisplayPath:
    def test_relative_to_source(self):
        result = safe_display_path("/data/docs/file.txt", ["/data/docs"])
        assert result == "file.txt"

    def test_nested_relative(self):
        result = safe_display_path("/data/docs/sub/file.txt", ["/data/docs"])
        expected = str(Path("sub") / "file.txt")
        assert result == expected

    def test_no_matching_folder(self):
        result = safe_display_path("/other/path/file.txt", ["/data/docs"])
        assert result == "file.txt"  # falls back to just filename

    def test_no_source_folders(self):
        result = safe_display_path("/any/path/file.txt")
        assert result == "file.txt"

    def test_empty_source_folders(self):
        result = safe_display_path("/any/path/file.txt", [])
        assert result == "file.txt"
