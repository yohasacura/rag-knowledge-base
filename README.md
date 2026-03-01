# RAG Knowledge Base

[![PyPI version](https://img.shields.io/pypi/v/rag-knowledge-base.svg)](https://pypi.org/project/rag-knowledge-base/)
[![Python](https://img.shields.io/pypi/pyversions/rag-knowledge-base.svg)](https://pypi.org/project/rag-knowledge-base/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

100% offline RAG storage and MCP server for querying local document knowledge bases. Designed for high-quality retrieval over large corpora (5–10 GB+) with GPU acceleration, hybrid search, and cross-encoder reranking.

## Features

### Core
- **Fully offline** — no internet required after initial setup
- **30+ file formats** — Markdown, PDF, DOCX, PPTX, XLSX, CSV, JSON, YAML, XML, HTML, RST, RTF, ODT/ODS/ODP, EPUB, images (OCR), source code, logs, and plain text
- **Multiple RAG databases** — create, switch, and manage independent knowledge bases
- **Shareable** — export a RAG as a single `.rag` file; import on another machine
- **MCP server** — query from VS Code Copilot, Claude Desktop, or any MCP client
- **Web UI** — NiceGUI dashboard for search, status, and management
- **File watching** — automatic re-indexing on file changes
- **Cross-platform** — Windows, macOS, and Linux

### Search Quality
- **Hybrid search** — combines vector similarity (cosine) with BM25 keyword matching via configurable score fusion
- **Cross-encoder reranking** — second-stage reranking with `cross-encoder/ms-marco-MiniLM-L-6-v2` for precision
- **Structure-aware chunking** — preserves Markdown heading hierarchy, PDF page boundaries, and code function/class boundaries
- **Contextual prefixes** — each chunk carries its document title and section path for better embedding quality
- **Min-score filtering** — automatically filters low-relevance noise from results
- **MMR diversification** — Maximal Marginal Relevance reduces redundancy in results (configurable lambda)

### Indexing Performance
- **GPU acceleration** — auto-detects CUDA and Apple MPS; falls back to CPU
- **Parallel file parsing** — multi-threaded document parsing with configurable worker count
- **Batched embedding** — encodes chunks in large batches (default 512) for optimal throughput
- **Incremental indexing** — xxHash-based file manifest tracks changes at O(1) cost; unchanged files are skipped instantly
- **Tuned HNSW** — configurable `ef_construction` and `M` parameters for ChromaDB's vector index

### Reliability
- **Cancellation** — cooperative cancellation of long-running indexing; partial work is safely cleaned up and re-indexed on the next run
- **Crash recovery** — lock file detection and manifest invalidation; the daemon auto-runs an incremental reindex on startup if a previous crash is detected
- **Index verification** — `rag-kb verify` checks manifest ↔ vector store consistency and reports orphaned/invalidated files

## Quick Start

### Install from PyPI

```bash
pip install rag-knowledge-base
```

This installs everything needed to index, search, and serve via MCP — including parsers for PDF, DOCX, PPTX, XLSX, and RTF.

#### Optional extras

| Extra | Install command | What it adds |
|-------|----------------|-------------|
| `web` | `pip install rag-knowledge-base[web]` | NiceGUI web dashboard |
| `ocr` | `pip install rag-knowledge-base[ocr]` | Image OCR (Surya, RapidOCR) |
| `api` | `pip install rag-knowledge-base[api]` | OpenAI / Voyage embedding APIs |
| `monitoring` | `pip install rag-knowledge-base[monitoring]` | System metrics (psutil) |
| `all` | `pip install rag-knowledge-base[all]` | All of the above |

### Install from source

```bash
pip install -e .
```

> **GPU support:** Install PyTorch with CUDA before installing this package:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cu126
> ```

### Create and Index a RAG

```bash
# Create a knowledge base pointing at your document folders
rag-kb create my-docs \
  --folders /path/to/docs /path/to/more/docs \
  --description "My documentation"

# Set as active
rag-kb use my-docs

# Index (auto-detects GPU, uses 14 parallel workers by default)
rag-kb index

# Re-run index any time — only changed files are processed
rag-kb index
```

### Query via MCP (VS Code Copilot)

```bash
# Start MCP server (stdio mode for VS Code)
rag-kb serve
```

Add to your `.vscode/mcp.json`:

```json
{
  "servers": {
    "rag-knowledge-base": {
      "command": "rag-kb",
      "args": ["serve"]
    }
  }
}
```

### Query via MCP (HTTP)

```bash
# Start MCP server on HTTP
rag-kb serve --http --port 8080
```

### Web Dashboard

```bash
# Launch NiceGUI dashboard
rag-kb ui
```

### Share a RAG

```bash
# Export
rag-kb export my-docs --output my-docs.rag

# Import (on another machine)
rag-kb import my-docs.rag --name received-docs
```

## Supported File Formats

| Category | Extensions |
|----------|------------|
| Documents | `.md`, `.markdown`, `.txt`, `.text`, `.pdf`, `.docx`, `.pptx`, `.rst`, `.rest`, `.rtf`, `.epub` |
| Spreadsheets | `.xlsx`, `.xls`, `.csv`, `.tsv` |
| Data | `.json`, `.jsonl`, `.yaml`, `.yml`, `.xml`, `.xsl`, `.xslt`, `.xsd`, `.svg`, `.rss`, `.atom` |
| Web | `.html`, `.htm` |
| OpenDocument | `.odt`, `.ods`, `.odp` |
| Images (OCR) | `.png`, `.jpg`, `.jpeg`, `.gif`, `.bmp`, `.tiff`, `.tif`, `.webp`, `.ico`, `.heic`, `.heif` |
| Code | `.py`, `.js`, `.mjs`, `.cjs`, `.jsx`, `.ts`, `.tsx`, `.java`, `.c`, `.h`, `.cpp`, `.cxx`, `.cc`, `.hpp`, `.hxx`, `.cs`, `.go`, `.rs`, `.rb`, `.php`, `.swift`, `.kt`, `.kts`, `.scala`, `.m`, `.mm`, `.r`, `.R`, `.lua`, `.pl`, `.pm`, `.sh`, `.bash`, `.zsh`, `.fish`, `.ps1`, `.psm1`, `.bat`, `.cmd`, `.sql`, `.dart`, `.ex`, `.exs`, `.erl`, `.hrl`, `.hs`, `.ml`, `.mli`, `.fs`, `.fsx`, `.clj`, `.cljs`, `.groovy`, `.gradle`, `.vim`, `.el`, `.zig`, `.v`, `.nim`, `.tf`, `.hcl`, `.proto`, `.graphql`, `.gql`, `.vue`, `.svelte`, `.toml`, `.ini`, `.cfg`, `.conf`, `.env`, `.properties`, `.makefile`, `.cmake`, `.dockerfile` |
| Logs | `.log` |

## CLI Reference

| Command | Description |
|---------|-------------|
| `rag-kb create NAME --folders ...` | Create a new RAG database |
| `rag-kb list` | List all RAG databases |
| `rag-kb use NAME` | Set the active RAG |
| `rag-kb delete NAME [-y]` | Delete a RAG database |
| `rag-kb detach NAME` | Detach a RAG from source files (read-only mode) |
| `rag-kb attach NAME` | Re-attach a detached RAG to its source files |
| `rag-kb index [--rag NAME] [--workers N]` | Index documents (parallel, incremental) |
| `rag-kb cancel-index` | Cancel a running indexing operation |
| `rag-kb verify [--rag NAME]` | Verify index consistency (manifest vs vector store) |
| `rag-kb search QUERY [-n N]` | Search the active RAG |
| `rag-kb status [--rag NAME]` | Show index status and statistics |
| `rag-kb files [--rag NAME]` | List indexed files |
| `rag-kb export NAME --output FILE` | Export RAG to a `.rag` file |
| `rag-kb import FILE [--name NAME]` | Import RAG from a `.rag` file |
| `rag-kb serve [--http] [--port N]` | Start MCP server |
| `rag-kb ui [--port N]` | Launch web dashboard |
| `rag-kb config` | Show current configuration |
| `rag-kb download-models [--output DIR]` | Pre-download ML models for offline/bundled use |
| `rag-kb models list\|info\|download\|delete` | Manage embedding/reranker models |
| `rag-kb daemon status\|stop\|restart\|logs` | Manage the background daemon |
| `rag-kb stats [CATEGORY]` | Show detailed metrics and monitoring statistics |
| `rag-kb monitor [--interval N]` | Live monitoring dashboard (like htop for RAG) |

Use `-v` for verbose output: `rag-kb -v index`

## MCP Tools

When connected via MCP, the following tools are available:

| Tool | Description |
|------|-------------|
| `search_knowledge_base` | Hybrid search with optional cross-encoder reranking |
| `get_document_content` | Retrieve all chunks from a specific document |
| `list_indexed_files` | List all indexed files with metadata |
| `get_index_status` | Current indexing state and statistics |
| `reindex` | Trigger re-indexing of the active RAG |
| `cancel_indexing` | Cancel a running indexing operation |
| `verify_index_consistency` | Check manifest ↔ vector store consistency |
| `list_rags` | List all available RAG databases |
| `switch_rag` | Switch the active RAG database |
| `create_rag` | Create a new RAG knowledge base |
| `delete_rag` | Permanently delete a RAG database |
| `export_rag` | Export a RAG to a shareable file |
| `import_rag` | Import a RAG from a shared file |
| `detach_rag` | Detach/re-attach a RAG from source files |
| `list_models` | List available embedding/reranker models |
| `get_model_info` | Get detailed info about a model |
| `get_monitoring_metrics` | Get comprehensive monitoring dashboard |
| `get_indexing_history` | Get recent indexing run history |
| `get_search_stats` | Get recent search query performance stats |
| `get_embedding_stats` | Get recent embedding batch performance stats |
| `get_vector_store_details` | Get detailed vector store / ChromaDB health info |

## Configuration

Config file location (platform-dependent):
- **Windows:** `%LOCALAPPDATA%\rag-kb\config.yaml`
- **macOS:** `~/Library/Application Support/rag-kb/config.yaml`
- **Linux:** `~/.local/share/rag-kb/config.yaml`

```yaml
# Embedding
embedding_model: paraphrase-multilingual-MiniLM-L12-v2
chunk_size: 1024
chunk_overlap: 128

# Search quality
reranking_enabled: true
reranker_model: cross-encoder/ms-marco-MiniLM-L-6-v2
hybrid_search_enabled: true
hybrid_search_alpha: 0.7        # 0.0 = pure BM25, 1.0 = pure vector
min_score_threshold: 0.15
mmr_enabled: true
mmr_lambda: 0.7                 # 0.0 = max diversity, 1.0 = max relevance

# Indexing performance
indexing_workers: 14             # parallel file-parsing threads
embedding_batch_size: 512        # texts per encode() call

# ChromaDB HNSW tuning
hnsw_ef_construction: 256
hnsw_m: 48

# File types to index
supported_extensions:
  - .md
  - .txt
  - .pdf
  - .docx
  - .pptx
  - .xlsx
  - .csv
  - .json
  - .yaml
  - .xml
  - .html
  - .htm
  - .rst
  - .rtf
  - .odt
  - .epub
  - .png
  - .jpg
  # ... see config for full list

# Server
host: 127.0.0.1
port: 8080
```

## Daemon Management

The daemon is a background process that manages all data access. It starts automatically on first CLI use and shuts down after an idle timeout.

```bash
# Check daemon status
rag-kb daemon status

# View daemon logs
rag-kb daemon logs

# Follow logs in real-time (like tail -f)
rag-kb daemon logs -f

# Show last 100 lines
rag-kb daemon logs -n 100

# Restart the daemon
rag-kb daemon restart

# Stop the daemon
rag-kb daemon stop
```

### Crash Recovery

If indexing is interrupted (crash, power loss, kill), the daemon automatically detects and recovers on the next startup:

1. **Lock file detection** — a `.indexing_lock` file in the RAG database marks active indexing
2. **Manifest invalidation** — partially processed files are pre-marked so `is_changed()` returns `True`
3. **Auto-recovery** — the daemon runs an incremental reindex on startup if a previous crash is detected

You can also manually check index consistency:

```bash
# Verify manifest ↔ vector store consistency
rag-kb verify

# Repair by re-indexing
rag-kb index
```

## Architecture

```
Documents  ──►  Parsers (30+ formats)
                    │
                    ▼
             Structure-aware Chunker
             (headings / pages / functions)
                    │
                    ▼
             Embedding Model (CUDA / MPS / CPU)
             + Contextual Prefixes
                    │
                    ▼
             ChromaDB (HNSW cosine index)
                    │
          ┌─────────┼─────────┐
          ▼         ▼         ▼
       MCP Server  Web UI   CLI
          │
    ┌─────┼─────┐
    ▼     ▼     ▼
  Vector  BM25  Hybrid Fusion
    │           │
    ▼           ▼
  Cross-encoder Reranking
    │
    ▼
  Results
```

## License

MIT License — see [LICENSE](LICENSE) for details.

All dependencies use MIT, Apache 2.0, or BSD licenses.
