# CLAUDE.md

## Project Overview

**RAG Knowledge Base** — a 100% offline Retrieval-Augmented Generation (RAG) storage and MCP server for querying local document knowledge bases. Distributed as a Python package via PyPI (`rag-knowledge-base`). Provides CLI, MCP server (stdio/HTTP), Web UI, and Python API interfaces.

**Repo:** https://github.com/yohasacura/rag-knowledge-base
**Version:** 1.0.0 | **License:** MIT | **Python:** 3.10–3.13

## Quick Reference

```bash
# Install from source (editable, with dev deps)
pip install -e ".[dev]"

# Run tests
pytest tests/

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
  daemon.py              # Asyncio TCP daemon (JSON-RPC 2.0)
  daemon_client.py       # Synchronous JSON-RPC client with auto-start
  config.py              # AppSettings (Pydantic) + RAG registry management
  indexer.py             # Indexing pipeline: scan → parse → chunk → embed → upsert
  search.py              # BM25, hybrid fusion, cross-encoder reranking, MMR
  chunker.py             # Structure-aware text chunking
  embedder.py            # Embedding facade (pluggable backends)
  embedding_backends.py  # Backends: local (sentence-transformers), OpenAI, Voyage
  vector_store.py        # ChromaDB wrapper with tunable HNSW
  file_manifest.py       # SQLite-backed fast change detection (xxHash)
  mcp_server.py          # MCP server implementation (FastMCP)
  web_ui.py              # NiceGUI-based web dashboard
  models.py              # Model registry & management
  metrics.py             # Persistent metrics collection (SQLite)
  sharing.py             # Export/import RAG as .rag files (ZIP archives)
  watcher.py             # File-system watcher for auto re-indexing
  device.py              # GPU/device auto-detection (CUDA, MPS, CPU)
  skip_patterns.py       # Gitignore-like file exclusions
  rpc_protocol.py        # JSON-RPC 2.0 protocol utilities
  parsers/               # 20+ document parsers (markdown, pdf, docx, code, images, etc.)
    base.py              # DocumentParser protocol
    registry.py          # Extension → parser dispatch

tests/
  test_cancellation.py   # Cooperative cancellation tests
  test_crash_recovery.py # Crash recovery & manifest invalidation tests
```

## Architecture

### Daemon-Client Pattern
All data access goes through a single asyncio TCP daemon (`RagDaemon`) on `127.0.0.1:9527`. CLI, MCP server, and Web UI are thin clients communicating via JSON-RPC 2.0 over TCP. `DaemonClient` auto-spawns the daemon if not running.

### Core Service Layer
`RagKnowledgeBaseAPI` (core.py) is the single entry point for all business logic. Thread-safe via `threading.RLock` (acquired by `@_locked` decorator). Supports progress callbacks for long-running operations.

### Search Pipeline
```
Query → Embed → Vector search (ChromaDB) + BM25 keyword search
      → Hybrid fusion (alpha=0.7 default)
      → Min-score filtering (0.15 default)
      → Cross-encoder reranking (ms-marco-MiniLM-L-6-v2)
      → MMR diversification → Results
```

### Indexing Pipeline
```
Scan → Filter changed (FileManifest) → Parse (parallel) → Chunk (structure-aware)
     → Embed (batched, 256/batch) → Upsert (ChromaDB, 10K/batch) → Update manifest
```

### Key Patterns
- **Pluggable embedding backends** — `EmbeddingBackend` ABC with `embed_texts()`, `embed_single()`, `get_dimension()`
- **Parser protocol** — `DocumentParser` with `supported_extensions` and `parse(path) → ParsedDocument`
- **Structure-aware chunking** — preserves heading hierarchy, code function/class boundaries, PDF page markers
- **Contextual prefixes** — chunks prepended with document title + section path for better embedding context
- **Incremental indexing** — xxHash content hashing via FileManifest; unchanged files skipped

## Build System

- **Build backend:** Hatchling (PEP 517/518)
- **Config:** `pyproject.toml`
- **Entry point:** `rag-kb = "rag_kb.cli:main"`
- **Package layout:** `src/rag_kb/`

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

## Configuration

- **App settings:** `config.yaml` (platform-dependent location)
  - Windows: `%LOCALAPPDATA%\rag-kb\config.yaml`
  - macOS: `~/Library/Application Support/rag-kb/config.yaml`
  - Linux: `~/.local/share/rag-kb/config.yaml`
- **RAG registry:** `registry.json` (same parent directory)
- **File manifest:** `file_manifest.db` per RAG (SQLite)

## Conventions

- Type hints throughout (Python 3.10+ syntax with `X | None`)
- Pydantic v2 for settings and data validation
- Docstrings in reST/Google style
- `rich` for CLI output formatting
- Thread safety via `threading.RLock` in core API
- Cooperative cancellation for long-running operations
- Crash recovery with lock file detection and manifest invalidation
