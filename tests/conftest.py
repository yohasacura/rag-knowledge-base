"""Shared fixtures for the RAG Knowledge Base test suite."""

from __future__ import annotations

import shutil
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# pytest custom options
# ---------------------------------------------------------------------------


def pytest_addoption(parser):
    """Register --runslow CLI flag."""
    parser.addoption("--runslow", action="store_true", default=False, help="run slow tests")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


# ---------------------------------------------------------------------------
# Original mock fixtures (unchanged for existing tests)
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_rag_entry(tmp_path):
    """Create a minimal mock RagEntry with temp paths."""
    source_dir = tmp_path / "docs"
    source_dir.mkdir()
    db_path = tmp_path / "db"
    db_path.mkdir()

    rag = MagicMock()
    rag.name = "test-rag"
    rag.db_path = str(db_path)
    rag.source_folders = [str(source_dir)]
    rag.embedding_model = "all-MiniLM-L6-v2"
    rag.detached = False
    rag.is_imported = False
    rag.file_count = 0
    rag.chunk_count = 0
    return rag


@pytest.fixture
def mock_settings():
    """Create a minimal mock AppSettings."""
    settings = MagicMock()
    settings.supported_extensions = [".txt", ".md"]
    settings.chunk_size = 1000
    settings.chunk_overlap = 200
    settings.embedding_model = "all-MiniLM-L6-v2"
    settings.openai_api_key = ""
    settings.voyage_api_key = ""
    settings.trusted_models = []
    return settings


@pytest.fixture
def mock_registry():
    """Create a minimal mock RagRegistry."""
    registry = MagicMock()
    registry.get_active_name.return_value = "test-rag"
    return registry


# ---------------------------------------------------------------------------
# Isolated data directory fixtures (for integration / E2E tests)
# ---------------------------------------------------------------------------

_WORKSPACE_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Create an isolated data directory structure for tests."""
    data_dir = tmp_path / "rag-kb-data"
    data_dir.mkdir()
    (data_dir / "rags").mkdir()
    return data_dir


@pytest.fixture
def isolated_settings(tmp_data_dir):
    """Real AppSettings persisted to the temp data dir."""
    from rag_kb.config import AppSettings

    config_path = tmp_data_dir / "config.yaml"
    settings = AppSettings()
    settings.save(config_path)
    return AppSettings.load(config_path)


@pytest.fixture
def isolated_registry(tmp_data_dir):
    """Real RagRegistry backed by a temp directory."""
    from rag_kb.config import RagRegistry

    return RagRegistry(
        registry_path=tmp_data_dir / "registry.json",
        rags_dir=tmp_data_dir / "rags",
    )


@pytest.fixture
def api_instance(isolated_settings, isolated_registry):
    """A fully wired RagKnowledgeBaseAPI that uses temp dirs (no daemon)."""
    from rag_kb.core import RagKnowledgeBaseAPI

    api = RagKnowledgeBaseAPI(settings=isolated_settings, registry=isolated_registry)
    yield api
    api.shutdown()


@pytest.fixture
def sample_docs_dir(tmp_path):
    """Create a temp directory with a handful of small synthetic documents."""
    docs = tmp_path / "sample-docs"
    docs.mkdir()

    (docs / "readme.md").write_text(
        "# Project Overview\n\n"
        "This is a test project about quantum computing and entanglement.\n\n"
        "## Key Concepts\n\n"
        "Quantum entanglement is a phenomenon where particles become correlated.\n"
        "Superposition allows qubits to be in multiple states simultaneously.\n",
        encoding="utf-8",
    )
    (docs / "notes.txt").write_text(
        "Classical music has a rich history spanning several centuries.\n"
        "Bach, Mozart, and Beethoven are considered the greatest composers.\n"
        "The symphony as a form was perfected during the Classical period.\n",
        encoding="utf-8",
    )
    (docs / "data.json").write_text(
        '{"topic": "machine learning", '
        '"description": "ML is a subset of artificial intelligence that '
        'enables systems to learn from data.", '
        '"algorithms": ["random forest", "neural networks", "SVM"]}',
        encoding="utf-8",
    )
    (docs / "example.py").write_text(
        '"""A sample Python module for testing."""\n\n\n'
        "def fibonacci(n: int) -> int:\n"
        '    """Calculate the nth Fibonacci number."""\n'
        "    if n <= 1:\n"
        "        return n\n"
        "    return fibonacci(n - 1) + fibonacci(n - 2)\n\n\n"
        "class Calculator:\n"
        '    """Simple calculator class."""\n\n'
        "    def add(self, a: float, b: float) -> float:\n"
        "        return a + b\n",
        encoding="utf-8",
    )
    (docs / "info.csv").write_text(
        "name,age,city\nAlice,30,London\nBob,25,Paris\nCharlie,35,Tokyo\n",
        encoding="utf-8",
    )
    (docs / "page.html").write_text(
        "<html><head><title>Test Page</title></head>"
        "<body><h1>Welcome</h1><p>This is an HTML test document.</p>"
        "<script>alert('ignored')</script></body></html>",
        encoding="utf-8",
    )
    return docs


@pytest.fixture
def all_format_docs_dir(tmp_path):
    """Copy all test documents from tests/fixtures/ into a temp directory."""
    src = _WORKSPACE_ROOT / "tests" / "fixtures"
    if not src.exists():
        pytest.skip("tests/fixtures/ not found")
    dst = tmp_path / "all-format-docs"
    shutil.copytree(src, dst)
    return dst
