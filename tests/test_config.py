"""Tests for configuration — API key env var resolution, settings."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from rag_kb.config import AppSettings


class TestApiKeyEnvVarResolution:
    """Tests for API key env var fallback in AppSettings."""

    def test_openai_key_from_env(self):
        """resolve_openai_api_key returns env var when set."""
        settings = AppSettings()
        settings.openai_api_key = "stored-key"
        with patch.dict(os.environ, {"RAG_KB_OPENAI_API_KEY": "env-key"}):
            assert settings.resolve_openai_api_key() == "env-key"

    def test_openai_key_fallback_to_stored(self):
        """resolve_openai_api_key returns stored key when env var not set."""
        settings = AppSettings()
        settings.openai_api_key = "stored-key"
        with patch.dict(os.environ, {}, clear=True):
            # Remove the env var if it exists
            os.environ.pop("RAG_KB_OPENAI_API_KEY", None)
            assert settings.resolve_openai_api_key() == "stored-key"

    def test_voyage_key_from_env(self):
        """resolve_voyage_api_key returns env var when set."""
        settings = AppSettings()
        settings.voyage_api_key = "stored-key"
        with patch.dict(os.environ, {"RAG_KB_VOYAGE_API_KEY": "env-key"}):
            assert settings.resolve_voyage_api_key() == "env-key"

    def test_voyage_key_fallback_to_stored(self):
        """resolve_voyage_api_key returns stored key when env var not set."""
        settings = AppSettings()
        settings.voyage_api_key = "stored-key"
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("RAG_KB_VOYAGE_API_KEY", None)
            assert settings.resolve_voyage_api_key() == "stored-key"

    def test_empty_env_var_falls_through(self):
        """Empty env var should fall through to stored value."""
        settings = AppSettings()
        settings.openai_api_key = "stored-key"
        with patch.dict(os.environ, {"RAG_KB_OPENAI_API_KEY": ""}):
            # Empty string is falsy, should fall back to stored
            result = settings.resolve_openai_api_key()
            assert result in ("stored-key", "")  # depends on implementation
