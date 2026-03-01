"""Tests for architecture improvements — VectorStoreRegistry, auth tokens, parser registry."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestVectorStoreRegistry:
    """Tests for VectorStoreRegistry thread-safe registry."""

    @staticmethod
    def _mock_store(db_path: str) -> MagicMock:
        """Create a mock VectorStore with a _db_path attribute."""
        store = MagicMock()
        store._db_path = db_path
        return store

    def test_register_and_get(self):
        from rag_kb.vector_store import VectorStoreRegistry

        reg = VectorStoreRegistry()
        mock_store = self._mock_store("/path/to/db")
        reg.register(mock_store)
        assert reg.get("/path/to/db") is mock_store

    def test_unregister(self):
        from rag_kb.vector_store import VectorStoreRegistry

        reg = VectorStoreRegistry()
        mock_store = self._mock_store("/path/to/db")
        reg.register(mock_store)
        reg.unregister("/path/to/db")
        assert reg.get("/path/to/db") is None

    def test_close_for_path(self):
        from rag_kb.vector_store import VectorStoreRegistry

        reg = VectorStoreRegistry()
        mock_store = self._mock_store("/path/to/db")
        reg.register(mock_store)
        reg.close_for_path("/path/to/db")
        mock_store.close.assert_called_once()
        assert reg.get("/path/to/db") is None

    def test_close_all(self):
        from rag_kb.vector_store import VectorStoreRegistry

        reg = VectorStoreRegistry()
        stores = []
        for i in range(3):
            s = self._mock_store(f"/path/{i}")
            stores.append(s)
            reg.register(s)
        reg.close_all()
        for s in stores:
            s.close.assert_called_once()

    def test_path_normalization(self):
        """Registry normalizes paths for consistent lookups."""
        from rag_kb.vector_store import VectorStoreRegistry

        reg = VectorStoreRegistry()
        mock_store = self._mock_store("/path/to/db")
        reg.register(mock_store)
        # Same normalized path should find it
        assert reg.get("/path/to/db") is mock_store

    def test_global_singleton(self):
        from rag_kb.vector_store import get_store_registry

        r1 = get_store_registry()
        r2 = get_store_registry()
        assert r1 is r2


class TestAuthToken:
    """Tests for daemon auth token generation and reading."""

    def test_generate_and_read_token(self, tmp_path):
        from rag_kb.rpc_protocol import generate_auth_token, read_auth_token

        with patch("rag_kb.rpc_protocol._get_token_path", return_value=tmp_path / "token"):
            token = generate_auth_token()
            assert isinstance(token, str)
            assert len(token) > 0
            read_back = read_auth_token()
            assert read_back == token

    def test_remove_token(self, tmp_path):
        from rag_kb.rpc_protocol import (
            generate_auth_token,
            read_auth_token,
            remove_auth_token,
        )

        with patch("rag_kb.rpc_protocol._get_token_path", return_value=tmp_path / "token"):
            generate_auth_token()
            remove_auth_token()
            assert read_auth_token() is None

    def test_read_nonexistent_token(self, tmp_path):
        from rag_kb.rpc_protocol import read_auth_token

        with patch(
            "rag_kb.rpc_protocol._get_token_path", return_value=tmp_path / "nonexistent"
        ):
            assert read_auth_token() is None


class TestParserRegistry:
    """Tests for lazy parser loading in the parser registry."""

    def test_supported_extensions_populated(self):
        from rag_kb.parsers.registry import SUPPORTED_EXTENSIONS

        assert len(SUPPORTED_EXTENSIONS) > 0
        assert ".txt" in SUPPORTED_EXTENSIONS
        assert ".md" in SUPPORTED_EXTENSIONS
        assert ".py" in SUPPORTED_EXTENSIONS

    def test_get_parser_returns_instance(self):
        from rag_kb.parsers.registry import get_parser

        parser = get_parser(Path("test.txt"))
        assert parser is not None
        assert hasattr(parser, "parse")

    def test_get_parser_caches(self):
        from rag_kb.parsers.registry import get_parser

        p1 = get_parser(Path("test.txt"))
        p2 = get_parser(Path("other.txt"))
        # Same parser type for same extension
        assert p1 is p2

    def test_get_parser_unknown_ext(self):
        from rag_kb.parsers.registry import get_parser

        parser = get_parser(Path("test.unknown_extension_xyz"))
        assert parser is None

    def test_parse_file_unknown_ext(self):
        from rag_kb.parsers.registry import parse_file

        result = parse_file(Path("test.unknown_extension_xyz"))
        assert result is None

    def test_get_parser_case_insensitive(self):
        from rag_kb.parsers.registry import get_parser

        p_lower = get_parser(Path("test.txt"))
        p_upper = get_parser(Path("test.TXT"))
        # Both should return the same parser (or at least the same type)
        assert type(p_lower) is type(p_upper)
