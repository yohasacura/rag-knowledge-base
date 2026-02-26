"""MCP Server — exposes RAG knowledge base tools via Model Context Protocol.

Improvements:
  - Hybrid search (vector + BM25 keyword fusion).
  - Cross-encoder reranking for higher precision.
  - MMR diversity filtering to avoid near-duplicate results.
  - Advanced search tool with full control over search parameters.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from mcp.server.fastmcp import Context, FastMCP

from rag_kb.config import AppSettings, RagEntry, RagRegistry, safe_display_path
from rag_kb.embedder import embed_query
from rag_kb.indexer import Indexer, IndexingState
from rag_kb.search import bm25_search, hybrid_fuse_scores, mmr_diversify, rerank_cross_encoder
from rag_kb.sharing import export_rag, import_rag, peek_rag_file
from rag_kb.vector_store import VectorStore
from rag_kb.watcher import FolderWatcher

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Application context shared across all tool calls
# ---------------------------------------------------------------------------

@dataclass
class AppContext:
    settings: AppSettings
    registry: RagRegistry
    store: VectorStore | None
    watcher: FolderWatcher | None
    active_rag: RagEntry | None
    last_index_state: IndexingState | None = None


def _open_store(entry: RagEntry | None, settings: AppSettings | None = None) -> VectorStore | None:
    if entry is None:
        return None
    s = settings or AppSettings.load()
    return VectorStore(
        entry.db_path,
        hnsw_ef_construction=s.hnsw_ef_construction,
        hnsw_m=s.hnsw_m,
    )


def _start_watcher(entry: RagEntry | None, registry: RagRegistry, settings: AppSettings) -> FolderWatcher | None:
    if entry is None or entry.is_imported or entry.detached or not entry.source_folders:
        return None
    w = FolderWatcher(entry, registry, settings)
    w.start()
    return w


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    settings = AppSettings.load()
    registry = RagRegistry()
    active = registry.get_active()
    store = _open_store(active, settings)
    watcher = _start_watcher(active, registry, settings)

    ctx = AppContext(
        settings=settings,
        registry=registry,
        store=store,
        watcher=watcher,
        active_rag=active,
    )
    try:
        yield ctx
    finally:
        if watcher:
            watcher.stop()
        if store is not None:
            store.close(force=True)


# ---------------------------------------------------------------------------
# FastMCP server
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "RAG Knowledge Base",
    instructions=(
        "This server provides semantic search over locally indexed documents.\n"
        "Typical workflows:\n"
        "- Search: use search_knowledge_base, then get_document_content with the returned source_file to read full documents.\n"
        "- Browse: use list_indexed_files to see all files, then get_document_content for any file.\n"
        "- Manage: use list_rags / switch_rag to work with multiple knowledge bases.\n"
        "- Create: use create_rag to create a new knowledge base, then reindex to build the index.\n"
        "- Share: use export_rag / import_rag to share knowledge bases as portable .rag files.\n"
        "- Cleanup: use delete_rag (with confirm=True) to permanently remove a RAG."
    ),
    lifespan=app_lifespan,
)


# ---------------------------------------------------------------------------
# Helper to get context
# ---------------------------------------------------------------------------

def _ctx(ctx) -> AppContext:
    """Extract AppContext from the request context."""
    return ctx.request_context.lifespan_context


def _ensure_store(app: AppContext) -> VectorStore:
    if app.store is None:
        raise RuntimeError("No active RAG database. Use switch_rag or create one first.")
    return app.store


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@mcp.tool()
def search_knowledge_base(query: str, n_results: int = 5, ctx: Context = None) -> list[dict[str, Any]]:
    """Search the active RAG knowledge base with a natural language query.

    Uses hybrid search (vector + keyword), cross-encoder reranking, and MMR diversity
    filtering for best results. Returns the top n_results (default 5) document chunks
    after deduplication.

    Each result contains: text, source_file (display path), score (0-1), chunk_index.
    The returned source_file values can be passed directly to get_document_content
    to retrieve all chunks from that file.
    """
    app = _ctx(ctx)
    store = _ensure_store(app)
    assert app.active_rag is not None
    settings = app.settings

    # Fetch more candidates for reranking
    fetch_k = max(n_results * 4, 20)
    query_emb = embed_query(query, model_name=app.active_rag.embedding_model)
    vec_results = store.search(
        query_emb,
        n_results=fetch_k,
        min_score=settings.min_score_threshold,
        include_embeddings=settings.mmr_enabled,
    )

    if not vec_results:
        return []

    folders = app.active_rag.source_folders

    # Hybrid search with BM25
    if settings.hybrid_search_enabled:
        try:
            all_ids, all_texts, all_metas = store.get_all_documents()
            if all_texts:
                bm25_hits = bm25_search(query, all_texts, all_ids, top_k=fetch_k)
                # Build score maps
                vec_scores = {r.source_file + "::chunk_" + str(r.chunk_index): r.score for r in vec_results}
                bm25_scores = {hit_id: score for hit_id, score in bm25_hits}
                fused = hybrid_fuse_scores(vec_scores, bm25_scores, alpha=settings.hybrid_search_alpha)

                # Rebuild results list with fused scores
                all_results_map = {r.source_file + "::chunk_" + str(r.chunk_index): r for r in vec_results}
                # Add BM25-only hits from store data
                id_to_idx = {doc_id: i for i, doc_id in enumerate(all_ids)}
                for hit_id, _ in bm25_hits:
                    if hit_id not in all_results_map and hit_id in id_to_idx:
                        idx = id_to_idx[hit_id]
                        meta = all_metas[idx] if idx < len(all_metas) else {}
                        from rag_kb.vector_store import SearchResult
                        all_results_map[hit_id] = SearchResult(
                            text=all_texts[idx],
                            source_file=meta.get("source_file", ""),
                            chunk_index=int(meta.get("chunk_index", 0)),
                            score=0.0,
                            metadata=meta,
                        )

                # Apply fused scores
                for key, score in fused.items():
                    if key in all_results_map:
                        all_results_map[key].score = score

                vec_results = sorted(all_results_map.values(), key=lambda r: r.score, reverse=True)[:fetch_k]
        except Exception as exc:
            logger.warning("BM25 hybrid search failed, falling back to vector only: %s", exc)

    # Backfill embeddings for BM25-only hits when MMR is needed
    if settings.mmr_enabled:
        missing = [r for r in vec_results if r.embedding is None]
        if missing:
            try:
                ids_needed = [
                    r.source_file + "::chunk_" + str(r.chunk_index)
                    for r in missing
                ]
                emb_map = store.get_embeddings_by_ids(ids_needed)
                for r, doc_id in zip(missing, ids_needed):
                    if doc_id in emb_map:
                        r.embedding = emb_map[doc_id]
            except Exception as exc:
                logger.warning("Failed to backfill embeddings for MMR: %s", exc)

    # Reranking
    if settings.reranking_enabled:
        try:
            texts_for_rerank = [r.text for r in vec_results]
            scores_for_rerank = [r.score for r in vec_results]
            reranked = rerank_cross_encoder(
                query, texts_for_rerank, scores_for_rerank,
                model_name=settings.reranker_model,
            )
            reordered = [vec_results[idx] for idx, _ in reranked]
            for res, (_, new_score) in zip(reordered, reranked):
                res.score = new_score
            vec_results = reordered
        except Exception as exc:
            logger.warning("Cross-encoder reranking failed: %s", exc)

    # MMR diversity
    if settings.mmr_enabled and len(vec_results) > n_results:
        try:
            import numpy as np
            doc_embeddings = np.array(
                [r.embedding for r in vec_results if r.embedding is not None],
                dtype=np.float32,
            )
            if len(doc_embeddings) == len(vec_results):
                mmr_indices = mmr_diversify(
                    query_embedding=query_emb,
                    doc_embeddings=doc_embeddings,
                    scores=[r.score for r in vec_results],
                    lambda_mult=settings.mmr_lambda,
                    top_n=n_results,
                )
                final_results = [vec_results[i] for i in mmr_indices]
            else:
                logger.warning("MMR: some embeddings missing, falling back to top-N")
                final_results = vec_results[:n_results]
        except Exception as exc:
            logger.warning("MMR diversity failed, falling back to top-N: %s", exc)
            final_results = vec_results[:n_results]
    else:
        final_results = vec_results[:n_results]

    # Apply minimum score threshold after all score transformations
    min_thr = settings.min_score_threshold
    if min_thr > 0:
        final_results = [r for r in final_results if r.score >= min_thr]

    return [
        {
            "text": r.text,
            "source_file": safe_display_path(r.source_file, folders),
            "score": round(r.score, 4),
            "chunk_index": r.chunk_index,
        }
        for r in final_results
    ]


@mcp.tool()
def get_document_content(file_path: str, ctx: Context = None) -> list[dict[str, Any]]:
    """Retrieve all indexed chunks from a specific document.

    Accepts the display path returned by search_knowledge_base or list_indexed_files
    (e.g. 'src/rag_kb/search.py'). Also accepts absolute paths.
    """
    app = _ctx(ctx)
    store = _ensure_store(app)

    # Try direct lookup first (absolute path)
    items = store.get_by_source(file_path)
    if items:
        return items

    # Reverse-resolve: map display paths back to stored absolute paths
    folders = app.active_rag.source_folders if app.active_rag else []
    stored_sources = store.get_stats().files
    # Normalise the input for comparison (backslash/forward-slash agnostic)
    norm_input = file_path.replace("\\", "/").lower()
    for stored in stored_sources:
        display = safe_display_path(stored, folders)
        if display.replace("\\", "/").lower() == norm_input:
            return store.get_by_source(stored)

    return []


@mcp.tool()
def list_indexed_files(ctx: Context = None) -> list[dict[str, Any]]:
    """List all files currently indexed in the active RAG database.

    Returns file display paths and chunk counts. The file paths can be
    passed directly to get_document_content to retrieve their chunks.
    """
    app = _ctx(ctx)
    store = _ensure_store(app)
    active = app.active_rag
    folders = active.source_folders if active else []

    # Build chunk counts from metadata in one query instead of O(n) queries
    all_ids, _all_texts, all_metas = store.get_all_documents()
    file_chunk_counts: dict[str, int] = {}
    for meta in all_metas:
        src = (meta or {}).get("source_file", "")
        if src:
            file_chunk_counts[src] = file_chunk_counts.get(src, 0) + 1

    return [
        {
            "file": safe_display_path(src, folders),
            "chunk_count": count,
        }
        for src, count in sorted(file_chunk_counts.items())
    ]


@mcp.tool()
def get_index_status(ctx: Context = None) -> dict[str, Any]:
    """Get the current indexing status and statistics for the active RAG.

    Returns: active_rag name, is_imported flag, total_files, total_chunks,
    watcher_running status, and (if available) last indexing status/timestamp/errors.
    """
    app = _ctx(ctx)
    active = app.active_rag
    result: dict[str, Any] = {
        "active_rag": active.name if active else None,
        "is_imported": active.is_imported if active else None,
    }
    if app.store:
        stats = app.store.get_stats()
        result["total_files"] = stats.total_files
        result["total_chunks"] = stats.total_chunks
    if app.last_index_state:
        result["last_status"] = app.last_index_state.status
        result["last_indexed"] = app.last_index_state.last_indexed
        result["errors"] = app.last_index_state.errors
    result["watcher_running"] = app.watcher.is_running if app.watcher else False
    return result


@mcp.tool()
def reindex(full: bool = False, ctx: Context = None) -> dict[str, Any]:
    """Trigger re-indexing of the active RAG.

    Set full=True for a complete rebuild from scratch.
    With full=False (default), only changed/new files are re-indexed.

    Returns: status, files_processed, total_chunks, duration_seconds, and errors.
    """
    app = _ctx(ctx)
    if app.active_rag is None:
        raise RuntimeError("No active RAG database. Use switch_rag or create one first.")
    if app.active_rag.detached:
        raise RuntimeError(f"RAG '{app.active_rag.name}' is detached (read-only). Use detach_rag(detach=False) to re-enable.")
    if app.active_rag.is_imported and not app.active_rag.source_folders:
        raise RuntimeError("Cannot reindex an imported RAG with no source folders.")

    indexer = Indexer(app.active_rag, app.registry, app.settings)
    state = indexer.index(full=full)
    app.last_index_state = state
    app.store = _open_store(app.active_rag, app.settings)  # refresh store reference
    return {
        "status": state.status,
        "files_processed": state.processed_files,
        "total_chunks": state.total_chunks,
        "duration_seconds": state.duration_seconds,
        "errors": state.errors,
    }


@mcp.tool()
def list_rags(ctx: Context = None) -> list[dict[str, Any]]:
    """List all available RAG databases (local and imported).

    Returns for each RAG: name, description, is_active, is_imported, detached,
    embedding_model, file_count, chunk_count, source_folders, and created_at.
    """
    app = _ctx(ctx)
    active_name = app.registry.get_active_name()
    rags = app.registry.list_rags()
    return [
        {
            "name": r.name,
            "description": r.description,
            "is_active": r.name == active_name,
            "is_imported": r.is_imported,
            "detached": r.detached,
            "embedding_model": r.embedding_model,
            "file_count": r.file_count,
            "chunk_count": r.chunk_count,
            "source_folders": r.source_folders,
            "created_at": r.created_at,
        }
        for r in rags
    ]


@mcp.tool()
def switch_rag(name: str, ctx: Context = None) -> dict[str, str]:
    """Switch the active RAG database to the specified name.

    Closes the current store and file watcher, then opens the new RAG.
    The file watcher is automatically restarted for local (non-imported, non-detached) RAGs.
    """
    app = _ctx(ctx)
    app.registry.set_active(name)

    # Stop old watcher and close old store
    if app.watcher:
        app.watcher.stop()
    if app.store is not None:
        app.store.close()

    # Load new RAG
    entry = app.registry.get_rag(name)
    app.active_rag = entry
    app.store = _open_store(entry, app.settings)
    app.watcher = _start_watcher(entry, app.registry, app.settings)

    return {"active_rag": name, "status": "switched"}


@mcp.tool()
def export_rag_tool(name: str, output_path: str, ctx: Context = None) -> dict[str, str]:
    """Export a RAG database to a shareable .rag file.

    output_path: Full file path for the export (e.g. 'C:/exports/my_rag.rag').
    The '.rag' extension is added automatically if missing.
    """
    app = _ctx(ctx)
    result_path = export_rag(app.registry, name, output_path)
    return {"exported_to": result_path, "rag_name": name}


@mcp.tool()
def import_rag_tool(file_path: str, name: str | None = None, ctx: Context = None) -> dict[str, Any]:
    """Import a RAG database from a .rag file.

    Optionally specify a custom name. If a RAG with the same name already exists,
    a numeric suffix is added automatically (e.g. 'MyRAG' becomes 'MyRAG-2').

    Returns: imported_name, embedding_model, file_count, chunk_count, original_name.
    """
    app = _ctx(ctx)
    # Peek first to show info
    info = peek_rag_file(file_path)
    imported_name = import_rag(app.registry, file_path, new_name=name)
    return {
        "imported_name": imported_name,
        "embedding_model": info.get("embedding_model"),
        "file_count": info.get("file_count", 0),
        "chunk_count": info.get("chunk_count", 0),
        "original_name": info.get("name"),
    }


@mcp.tool()
def detach_rag(name: str | None = None, detach: bool = True, ctx: Context = None) -> dict[str, str]:
    """Detach (or re-attach) a RAG from its source files.

    A detached RAG is read-only: indexing and file watching are disabled,
    so you can safely delete the original source files without losing the
    indexed knowledge base. Set detach=False to re-attach.

    If name is omitted, operates on the currently active RAG.
    """
    app = _ctx(ctx)
    rag_name = name or (app.active_rag.name if app.active_rag else None)
    if not rag_name:
        raise RuntimeError("No RAG specified and no active RAG set.")
    entry = app.registry.get_rag(rag_name)
    entry.detached = detach
    app.registry.update_rag(entry)

    # Stop watcher if detaching the active RAG
    if detach and app.active_rag and app.active_rag.name == rag_name and app.watcher:
        app.watcher.stop()
        app.watcher = None

    action = "detached" if detach else "re-attached"
    return {"rag": rag_name, "status": action}


@mcp.tool()
def create_rag(
    name: str,
    source_folders: list[str],
    description: str = "",
    embedding_model: str | None = None,
    ctx: Context = None,
) -> dict[str, Any]:
    """Create a new RAG knowledge base.

    name: A unique name for the RAG (alphanumeric, hyphens, underscores).
    source_folders: List of absolute folder paths to index documents from.
    description: Optional human-readable description.
    embedding_model: Optional embedding model name (defaults to the global setting).

    After creating, use reindex() to build the search index.
    """
    app = _ctx(ctx)
    entry = app.registry.create_rag(
        name=name,
        description=description,
        folders=source_folders,
        embedding_model=embedding_model,
    )
    return {
        "name": entry.name,
        "description": entry.description,
        "embedding_model": entry.embedding_model,
        "source_folders": entry.source_folders,
        "status": "created",
    }


@mcp.tool()
def delete_rag(name: str, confirm: bool = False, ctx: Context = None) -> dict[str, str]:
    """Permanently delete a RAG database and all its indexed data.

    This action is irreversible. You must pass confirm=True to proceed.
    If the deleted RAG is the active one, the server will switch to the
    next available RAG automatically.
    """
    app = _ctx(ctx)
    if not confirm:
        raise ValueError(
            "Deletion requires explicit confirmation. "
            "Call delete_rag(name='...', confirm=True) to proceed."
        )

    # If we're deleting the currently active RAG, close its store and
    # watcher first so the files are released before deletion.
    is_active = app.active_rag and app.active_rag.name == name
    if is_active:
        if app.watcher:
            app.watcher.stop()
            app.watcher = None
        if app.store is not None:
            app.store.close(force=True)
            app.store = None
        app.active_rag = None
        import gc; gc.collect()

    app.registry.delete_rag(name)

    # If we just deleted the active RAG, switch to the new active (if any)
    if is_active:
        new_active = app.registry.get_active()
        if new_active:
            app.active_rag = new_active
            app.store = _open_store(new_active, app.settings)
            app.watcher = _start_watcher(new_active, app.registry, app.settings)

    return {"rag": name, "status": "deleted"}


# ---------------------------------------------------------------------------
# Entry point helpers
# ---------------------------------------------------------------------------

def run_stdio() -> None:
    """Run the MCP server with stdio transport."""
    mcp.run(transport="stdio")


def run_http(host: str = "127.0.0.1", port: int = 8080) -> None:
    """Run the MCP server with streamable HTTP transport."""
    mcp.settings.host = host
    mcp.settings.port = port
    mcp.run(transport="streamable-http")
