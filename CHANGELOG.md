# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.1] - 2025-07-19

### Added

- **27 new test files** covering all parsers, search pipeline, config, sharing,
  e2e lifecycle, RPC protocol, BM25 cache, device detection, metrics, security,
  edge cases, concurrency, crash recovery, watcher, skip patterns, file manifest,
  and vector store
- Persistent metrics collection (SQLite-backed) in `metrics.py`
- MMR diversification and bounded cross-encoder cache in search pipeline
- `VectorStoreRegistry` for thread-safe store lifecycle management
- ZIP import hardening: path traversal protection, size/count limits, zip-bomb detection
- Comprehensive gitignore-like file exclusions in `skip_patterns.py`
- `trust_remote_code` gate and backend caching for embedding backends
- 3 new CLI commands: `download-models`, `stats`, `monitor` (22 total)
- Batch operations and concurrency support in file manifest
- `py.typed` marker for PEP 561 typed package support
- CHANGELOG.md
- CI workflow for automated testing on pull requests

### Changed

- **Web UI migrated from Gradio to NiceGUI** (FastAPI + Vue/Quasar)
- Daemon expanded to 42 RPC methods across 10 namespaces
- Config defaults tuned: workers 14, batch 512, hnsw_ef 256
- Parsers now lazy-loaded: 20 parsers supporting 90+ file extensions
- Upgraded Development Status classifier from Beta to Production/Stable
- Expanded PyPI classifiers and keywords for better discoverability

### Fixed

- Exception handling for store statistics retrieval in daemon
- Input sanitization in `SentenceTransformerBackend` for tokeniser-safe strings
- Indexer now removes chunks of non-existent files and validates chunk texts
- `_NoOpEmbeddingFunction` in VectorStore prevents default embedding issues
- Metadata sanitization in VectorStore for ChromaDB compatibility
- Watcher handles file change events with explicit event types
- Thread safety in web UI `_safe_call` method to prevent race conditions
- CLAUDE.md and README.md synchronized with actual codebase state

## [1.0.0] - 2025-07-13

### Added

- Initial release: 100% offline RAG storage and MCP server
- CLI with 19 commands (`rag-kb`)
- MCP server (stdio and HTTP modes) with 22 tools
- NiceGUI-based web dashboard
- Asyncio TCP daemon with JSON-RPC 2.0 protocol and auth tokens
- ChromaDB vector store with tunable HNSW parameters
- Hybrid search: vector + BM25 keyword search with fusion and reranking
- Structure-aware text chunking with contextual prefixes
- Pluggable embedding backends: local (sentence-transformers), OpenAI, Voyage
- 20 document parsers supporting 90+ file extensions
- OCR pipeline with 4 backends: Surya OCR, RapidOCR, Tesseract, EasyOCR
- Incremental indexing with xxHash content change detection
- Export/import RAG as portable `.rag` files
- File-system watcher for auto re-indexing
- GPU/device auto-detection (CUDA, MPS, CPU)

[1.0.1]: https://github.com/yohasacura/rag-knowledge-base/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/yohasacura/rag-knowledge-base/releases/tag/v1.0.0
