"""Model registry tests — catalog, specs, paths, status."""

from __future__ import annotations

import pytest

from rag_kb.models import (
    EMBEDDING_MODELS,
    RERANKER_MODELS,
    ModelProvider,
    ModelSpec,
    ModelStatus,
    ModelType,
    get_all_embedding_models,
    get_all_reranker_models,
    get_embedding_model_names,
    get_model_path,
    get_model_spec,
    get_model_status,
    get_reranker_model_names,
)


# ---------------------------------------------------------------------------
# Catalog integrity
# ---------------------------------------------------------------------------


class TestCatalog:
    def test_embedding_models_not_empty(self):
        assert len(EMBEDDING_MODELS) > 5

    def test_reranker_models_not_empty(self):
        assert len(RERANKER_MODELS) >= 3

    def test_all_embeddings_have_required_fields(self):
        for m in EMBEDDING_MODELS:
            assert m.name, f"Missing name on model {m}"
            assert m.display_name
            assert m.type == ModelType.embedding
            assert m.dimensions > 0
            assert m.max_tokens > 0

    def test_all_rerankers_have_required_fields(self):
        for m in RERANKER_MODELS:
            assert m.name
            assert m.display_name
            assert m.type == ModelType.reranker

    def test_exactly_one_default(self):
        defaults = [m for m in EMBEDDING_MODELS if m.default]
        assert len(defaults) == 1
        assert defaults[0].name == "paraphrase-multilingual-MiniLM-L12-v2"

    def test_no_duplicate_names(self):
        all_names = [m.name for m in EMBEDDING_MODELS + RERANKER_MODELS]
        assert len(all_names) == len(set(all_names)), "Duplicate model names found"


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------


class TestLookup:
    def test_get_model_spec_known(self):
        spec = get_model_spec("paraphrase-multilingual-MiniLM-L12-v2")
        assert spec is not None
        assert spec.dimensions == 384

    def test_get_model_spec_unknown(self):
        assert get_model_spec("nonexistent-model-xyz") is None

    def test_get_all_embedding_models(self):
        models = get_all_embedding_models()
        assert all(m.type == ModelType.embedding for m in models)

    def test_get_all_reranker_models(self):
        models = get_all_reranker_models()
        assert all(m.type == ModelType.reranker for m in models)

    def test_embedding_model_names(self):
        names = get_embedding_model_names()
        assert "paraphrase-multilingual-MiniLM-L12-v2" in names

    def test_reranker_model_names(self):
        names = get_reranker_model_names()
        assert len(names) >= 3


# ---------------------------------------------------------------------------
# Model path resolution
# ---------------------------------------------------------------------------


class TestModelPath:
    def test_non_bundled_returns_name(self):
        # When models/ dir doesn't contain the model, returns model name as-is
        result = get_model_path("paraphrase-multilingual-MiniLM-L12-v2")
        # Either returns the repo name or a local path depending on bundling
        assert isinstance(result, str)
        assert len(result) > 0

    def test_unknown_model_returns_name(self):
        result = get_model_path("totally-fake-model")
        assert result == "totally-fake-model"

    def test_bundled_model_returns_path(self, tmp_path, monkeypatch):
        """If a matching dir exists under BUNDLED_MODELS_DIR, returns path."""
        import rag_kb.models as models_mod

        bundled = tmp_path / "models"
        model_dir = bundled / "test-model"
        model_dir.mkdir(parents=True)

        monkeypatch.setattr(models_mod, "BUNDLED_MODELS_DIR", bundled)
        result = get_model_path("test-model")
        assert result == str(model_dir)


# ---------------------------------------------------------------------------
# Model status
# ---------------------------------------------------------------------------


class TestModelStatus:
    def test_api_model_status(self):
        # Find an API model
        api_models = [m for m in EMBEDDING_MODELS if m.provider != ModelProvider.local]
        if not api_models:
            pytest.skip("No API models in catalog")
        status = get_model_status(api_models[0].name)
        assert status == ModelStatus.api

    def test_local_model_status(self):
        status = get_model_status("paraphrase-multilingual-MiniLM-L12-v2")
        # Will be either downloaded or available
        assert status in (
            ModelStatus.downloaded,
            ModelStatus.available,
            ModelStatus.bundled,
        )


# ---------------------------------------------------------------------------
# Trust remote code
# ---------------------------------------------------------------------------


class TestTrustRemoteCode:
    def test_default_model_no_trust(self):
        spec = get_model_spec("paraphrase-multilingual-MiniLM-L12-v2")
        assert spec is not None
        assert spec.trust_remote_code is False

    def test_models_requiring_trust_flagged(self):
        """All models with trust_remote_code=True should be identifiable."""
        trust_models = [
            m for m in EMBEDDING_MODELS + RERANKER_MODELS if m.trust_remote_code
        ]
        # These exist — verify the flag is proper type
        for m in trust_models:
            assert isinstance(m.trust_remote_code, bool)


# ---------------------------------------------------------------------------
# API key models
# ---------------------------------------------------------------------------


class TestApiKeyModels:
    def test_api_models_have_env_var(self):
        api_models = [
            m
            for m in EMBEDDING_MODELS
            if m.requires_api_key
        ]
        for m in api_models:
            assert m.api_key_env_var, f"API model {m.name} missing api_key_env_var"

    def test_api_models_zero_size(self):
        for m in EMBEDDING_MODELS:
            if m.provider != ModelProvider.local:
                assert m.model_size_mb == 0, f"API model {m.name} should have 0 size"
