# CLAUDE.md

## Project Overview

**RAG Knowledge Base** — a 100% offline Retrieval-Augmented Generation (RAG) storage and MCP server for querying local document knowledge bases. Distributed as a Python package via PyPI (`rag-knowledge-base`). Provides CLI, MCP server (stdio/HTTP), Web UI, and Python API interfaces.

**Repo:** https://github.com/yohasacura/rag-knowledge-base
**Version:** 1.0.0 | **License:** MIT | **Python:** 3.10–3.13

## Quick Reference

```bash
# Install from source (editable, with dev deps)
pip install -e ".[dev]"

# Lint & format
ruff check src/        # lint (0 violations expected)
ruff format src/       # auto-format

# Run tests
pytest tests/ -v

# CLI entry point
rag-kb <command>

# MCP server (stdio)
rag-kb serve

# MCP server (HTTP)
rag-kb serve --http --port 8080

# Web UI
rag-kb ui
```

## Project Structure

```
src/rag_kb/              # Main package
  cli.py                 # CLI entry point (rag-kb command)
  core.py                # RagKnowledgeBaseAPI — unified service layer
  daemon.py              # Asyncio TCP daemon (JSON-RPC 2.0) with auth tokens
  daemon_client.py       # Synchronous JSON-RPC client with auto-start
  config.py              # AppSettings (Pydantic) + RAG registry + env var API keys
  indexer.py             # Indexing pipeline: scan → parse → chunk → embed → upsert
  search.py              # BM25 (cached), hybrid fusion, cross-encoder reranking, MMR
  chunker.py             # Structure-aware text chunking
  embedder.py            # Embedding facade (pluggable backends)
  embedding_backends.py  # Backends: local (sentence-transformers), OpenAI, Voyage
  vector_store.py        # ChromaDB wrapper with tunable HNSW + VectorStoreRegistry
  file_manifest.py       # SQLite-backed fast change detection (xxHash)
  mcp_server.py          # MCP server implementation (FastMCP, 22 tools)
  web_ui.py              # NiceGUI-based web dashboard (FastAPI + Vue/Quasar)
  models.py              # Model registry & management
  metrics.py             # Persistent metrics collection (SQLite)
  sharing.py             # Export/import RAG as .rag files (ZIP, security-hardened)
  watcher.py             # File-system watcher for auto re-indexing
  device.py              # GPU/device auto-detection (CUDA, MPS, CPU)
  skip_patterns.py       # Gitignore-like file exclusions
  rpc_protocol.py        # JSON-RPC 2.0 protocol utilities + auth token helpers
  parsers/               # 20+ document parsers (lazily loaded)
    base.py              # DocumentParser protocol
    registry.py          # Lazy extension → parser dispatch

tests/
  conftest.py                # Shared pytest fixtures
  test_architecture.py       # VectorStoreRegistry, auth tokens, parser registry tests
  test_cancellation.py       # Cooperative cancellation tests
  test_chunker.py            # Structure-aware chunking tests
  test_concurrency.py        # Thread-safety and concurrent access tests
  test_config.py             # API key env var resolution tests
  test_config_full.py        # Full config management tests
  test_crash_recovery.py     # Crash recovery & manifest invalidation tests
  test_daemon_integration.py # Daemon JSON-RPC integration tests
  test_device.py             # GPU/device detection tests
  test_e2e_lifecycle.py      # End-to-end RAG lifecycle tests
  test_e2e_search_pipeline.py # End-to-end search pipeline tests
  test_edge_cases.py         # Edge case and error handling tests
  test_embedder.py           # Embedding facade tests
  test_file_manifest_ext.py  # File manifest extended tests
  test_indexer_full.py       # Full indexing pipeline tests
  test_metrics.py            # Metrics collection tests
  test_models.py             # Model registry & management tests
  test_parsers_all.py        # All parser format tests
  test_performance.py        # BM25 cache tests
  test_rpc_protocol.py       # JSON-RPC protocol tests
  test_search_units.py       # Search unit tests (BM25, fusion, reranking, MMR)
  test_security.py           # ZIP import hardening & path traversal tests
  test_sharing_full.py       # Export/import full tests
  test_skip_patterns.py      # Gitignore-like skip pattern tests
  test_vector_store_full.py  # VectorStore full tests
  test_watcher.py            # File watcher tests
```

## Architecture

### Daemon-Client Pattern
All data access goes through a single asyncio TCP daemon (`RagDaemon`) on `127.0.0.1:9527`. CLI, MCP server, and Web UI are thin clients communicating via JSON-RPC 2.0 over TCP (4-byte length-prefix framing). `DaemonClient` auto-spawns the daemon if not running. All requests are authenticated via a file-based auth token. The daemon exposes 42 RPC methods across 8 namespaces (`rag.*`, `search.*`, `index.*`, `share.*`, `config.*`, `models.*`, `watcher.*`, `system.*`, `store.*`, `metrics.*`). Idle timeout: 300s. Log rotation: 10 MB × 3 backups.

### Core Service Layer
`RagKnowledgeBaseAPI` (core.py) is the single entry point for all business logic. Thread-safe via `threading.RLock` (acquired by `@_locked` decorator). Supports progress callbacks for long-running operations.

### Search Pipeline
```
Query → Embed → Vector search (ChromaDB) + BM25 keyword search (cached 5min TTL)
      → Hybrid fusion (alpha=0.7 default)
      → Min-score filtering (0.15 default)
      → Cross-encoder reranking (ms-marco-MiniLM-L-6-v2, bounded cache)
      → MMR diversification → Results
```

### Indexing Pipeline
```
Scan → Filter changed (FileManifest) → Parse (parallel, lazy-loaded parsers)
     → Chunk (structure-aware) → Embed (batched, 512/batch)
     → Upsert (ChromaDB, 10K/batch) → Update manifest → Invalidate BM25 cache
```

### Key Patterns
- **Pluggable embedding backends** — `EmbeddingBackend` ABC with `embed_texts()`, `embed_single()`, `get_dimension()`
- **Parser protocol** — `DocumentParser` with `supported_extensions` and `parse(path) → ParsedDocument`
- **Lazy parser loading** — parser modules imported on first use to avoid loading heavy deps (pypdf, docx, etc.)
- **VectorStoreRegistry** — thread-safe registry replaces `gc.get_objects()` for store lifecycle management
- **BM25 cache** — `BM25Index` singleton with rag-name/doc-count/TTL invalidation
- **Auth tokens** — cryptographic token written to disk, validated on every daemon RPC call
- **API key env vars** — `RAG_KB_OPENAI_API_KEY` / `RAG_KB_VOYAGE_API_KEY` env vars override stored keys
- **trust_remote_code gate** — factory rejects untrusted models requesting remote code execution
- **Structure-aware chunking** — preserves heading hierarchy, code function/class boundaries, PDF page markers
- **Contextual prefixes** — chunks prepended with document title + section path for better embedding context
- **Incremental indexing** — xxHash content hashing via FileManifest; unchanged files skipped
- **MMR diversification** — Maximal Marginal Relevance for result diversity (lambda=0.7 default)
- **ZIP import hardening** — path traversal protection, size/count limits, zip-bomb detection
- **Persistent metrics** — SQLite-backed metrics for indexing, search, embedding, and vector store stats
- **OCR pipeline** — 4 backends: Surya OCR (GPU), RapidOCR (CPU/ONNX), Tesseract, EasyOCR with automatic fallback

## Build System

- **Build backend:** Hatchling (PEP 517/518)
- **Config:** `pyproject.toml`
- **Entry point:** `rag-kb = "rag_kb.cli:main"`
- **Package layout:** `src/rag_kb/`

## Tooling

| Tool | Config | Purpose |
|------|--------|---------|
| Ruff >=0.8.0 | `[tool.ruff]` in pyproject.toml | Linting (18 rule categories) + formatting |
| mypy >=1.8.0 | `[tool.mypy]` in pyproject.toml | Type checking (warn_return_any, etc.) |
| pytest | `[tool.pytest.ini_options]` | Testing with `-x -v` defaults |
| pytest-cov | `[tool.coverage]` | Coverage reporting (90% target) |

## Key Dependencies

| Category | Package | Purpose |
|----------|---------|---------|
| Vector store | chromadb | Persistent HNSW index |
| Embeddings | sentence-transformers | Local embedding models |
| MCP | mcp[cli] | Model Context Protocol server |
| Web UI | nicegui | NiceGUI dashboard (FastAPI + Vue) |
| Search | rank-bm25 | BM25 keyword matching |
| Validation | pydantic | Settings and data models |
| CLI | rich | Terminal formatting |
| File watch | watchdog | Auto re-indexing |
| Hashing | xxhash | Fast content change detection |
| Dev | ruff, mypy, pytest, pytest-cov | Lint, type check, test |

## Configuration

- **App settings:** `config.yaml` (platform-dependent location)
  - Windows: `%LOCALAPPDATA%\rag-kb\config.yaml`
  - macOS: `~/Library/Application Support/rag-kb/config.yaml`
  - Linux: `~/.local/share/rag-kb/config.yaml`
- **RAG registry:** `registry.json` (same parent directory)
- **File manifest:** `file_manifest.db` per RAG (SQLite)
- **Auth token:** `daemon.token` in app data dir (auto-generated)
- **API keys (env vars):** `RAG_KB_OPENAI_API_KEY`, `RAG_KB_VOYAGE_API_KEY`

## Conventions

- Type hints throughout (Python 3.10+ syntax with `X | None`)
- Pydantic v2 for settings and data validation
- Ruff-enforced style (100-char lines, 18 lint rule categories)
- Docstrings in reST/Google style
- `rich` for CLI output formatting
- Thread safety via `threading.RLock` in core API
- `raise ... from err` / `from None` in all except clauses (B904)
- `logging.exception` for error-level logs in except blocks (TRY400)
- `Path` API preferred over `os.path` (PTH rules)
- Cooperative cancellation for long-running operations
- Crash recovery with lock file detection and manifest invalidation
