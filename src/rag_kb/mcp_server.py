"""MCP Server — exposes RAG knowledge base tools via Model Context Protocol.

Thin adapter — all business logic is accessed via the daemon process
through ``DaemonClient``.  The daemon is auto-started on first use.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from mcp.server.fastmcp import Context, FastMCP

from rag_kb.daemon_client import DaemonClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan — create and yield the DaemonClient
# ---------------------------------------------------------------------------


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[DaemonClient]:
    client = DaemonClient()
    client.ensure_daemon()
    client.connect()

    try:
        yield client
    finally:
        client.close()


# ---------------------------------------------------------------------------
# FastMCP server
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "RAG Knowledge Base",
    instructions=(
        "This server provides semantic search over locally indexed documents.\n"
        "Typical workflows:\n"
        "- Search: use search_knowledge_base, then get_document_content with the returned source_file to read full documents.\n"
        "- Browse: use list_indexed_files to paginate through files, then get_document_content for any file.\n"
        "- Manage: use list_rags / switch_rag to work with multiple knowledge bases.\n"
        "- Create: use create_rag to create a new knowledge base, then reindex to build the index.\n"
        "- Share: use export_rag / import_rag to share knowledge bases as portable .rag files.\n"
        "- Cleanup: use delete_rag (with confirm=True) to permanently remove a RAG."
    ),
    lifespan=app_lifespan,
)


# ---------------------------------------------------------------------------
# Helper to get the API from request context
# ---------------------------------------------------------------------------


def _client(ctx) -> DaemonClient:
    """Extract the DaemonClient from the request context."""
    return ctx.request_context.lifespan_context


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool()
def search_knowledge_base(
    query: str, n_results: int = 5, ctx: Context = None
) -> list[dict[str, Any]]:
    """Search the active RAG knowledge base with a natural language query.

    Uses hybrid search (vector + keyword), cross-encoder reranking, and MMR diversity
    filtering for best results. Returns the top n_results (default 5) document chunks
    after deduplication.

    Each result contains: text, source_file (display path), score (0-1), chunk_index.
    The returned source_file values can be passed directly to get_document_content
    to retrieve all chunks from that file.
    """
    client = _client(ctx)
    return client.search(query=query, top_k=n_results)


@mcp.tool()
def get_document_content(file_path: str, ctx: Context = None) -> list[dict[str, Any]]:
    """Retrieve all indexed chunks from a specific document.

    Accepts the display path returned by search_knowledge_base or list_indexed_files
    (e.g. 'src/rag_kb/search.py'). Also accepts absolute paths.
    """
    client = _client(ctx)
    return client.get_document_content(file_path)


@mcp.tool()
def list_indexed_files(
    offset: int = 0,
    limit: int = 100,
    filter: str = "",
    ctx: Context = None,
) -> dict[str, Any]:
    """List files currently indexed in the active RAG database (paginated).

    Returns file display paths and chunk counts. The file paths can be
    passed directly to get_document_content to retrieve their chunks.

    Parameters:
      offset: 0-based index of the first file to return (default 0).
      limit: Maximum number of files per page (default 100, 0 = all).
      filter: Case-insensitive substring filter on file paths.

    Returns a dict with:
      files: list of {file, chunk_count} entries for the current page.
      total: total number of matching files.
      offset: the offset that was applied.
      limit: the limit that was applied.
    """
    client = _client(ctx)
    return client.list_indexed_files(offset=offset, limit=limit, filter=filter)


@mcp.tool()
def get_index_status(ctx: Context = None) -> dict[str, Any]:
    """Get the current indexing status and statistics for the active RAG.

    Returns: active_rag name, is_imported flag, total_files, total_chunks,
    watcher_running status, and (if available) last indexing status/timestamp/errors.
    """
    client = _client(ctx)
    return client.get_index_status()


@mcp.tool()
def reindex(full: bool = False, ctx: Context = None) -> dict[str, Any]:
    """Trigger re-indexing of the active RAG.

    Set full=True for a complete rebuild from scratch.
    With full=False (default), only changed/new files are re-indexed.

    Returns: status, files_processed, total_chunks, duration_seconds, and errors.
    """
    client = _client(ctx)
    result = client.index(full=full)
    return {
        "status": result.get("status"),
        "files_processed": result.get("processed_files"),
        "total_chunks": result.get("total_chunks"),
        "duration_seconds": result.get("duration_seconds"),
        "errors": result.get("errors", []),
    }


@mcp.tool()
def cancel_indexing(ctx: Context = None) -> dict[str, Any]:
    """Cancel a currently running indexing or re-indexing operation.

    Cancellation is cooperative — the indexer will stop at the next safe
    checkpoint. Files that were partially processed will be automatically
    re-indexed on the next run.

    Returns: cancelled (bool) indicating whether a running indexer was found.
    """
    client = _client(ctx)
    return client.cancel_indexing()


@mcp.tool()
def verify_index_consistency(ctx: Context = None) -> dict[str, Any]:
    """Verify that the manifest and vector store are consistent.

    Checks for:
    - Invalidated files (marked for re-indexing due to crash or cancellation)
    - Orphan store files (in vector store but not in manifest)
    - Orphan manifest files (in manifest but not in vector store)
    - Incomplete indexing (lock file from a previous crash)

    Returns a dict with:
      ok: True if everything is consistent, False otherwise.
      invalidated_files: list of files with empty content hash.
      orphan_store_files: list of files in store but not in manifest.
      orphan_manifest_files: list of files in manifest but not in store.
      incomplete_indexing: True if a lock file was found.
    """
    client = _client(ctx)
    return client.verify_index_consistency()


@mcp.tool()
def list_rags(ctx: Context = None) -> list[dict[str, Any]]:
    """List all available RAG databases (local and imported).

    Returns for each RAG: name, description, is_active, is_imported, detached,
    embedding_model, file_count, chunk_count, source_folders, and created_at.
    """
    client = _client(ctx)
    return client.list_rags()


@mcp.tool()
def switch_rag(name: str, ctx: Context = None) -> dict[str, str]:
    """Switch the active RAG database to the specified name.

    Closes the current store and file watcher, then opens the new RAG.
    The file watcher is automatically restarted for local (non-imported, non-detached) RAGs.
    """
    client = _client(ctx)
    client.switch_rag(name)
    return {"active_rag": name, "status": "switched"}


@mcp.tool()
def export_rag_tool(name: str, output_path: str, ctx: Context = None) -> dict[str, str]:
    """Export a RAG database to a shareable .rag file.

    output_path: Full file path for the export (e.g. 'C:/exports/my_rag.rag').
    The '.rag' extension is added automatically if missing.
    """
    client = _client(ctx)
    result = client.export_rag(name, output_path)
    return {"exported_to": result.get("path", ""), "rag_name": name}


@mcp.tool()
def import_rag_tool(file_path: str, name: str | None = None, ctx: Context = None) -> dict[str, Any]:
    """Import a RAG database from a .rag file.

    Optionally specify a custom name. If a RAG with the same name already exists,
    a numeric suffix is added automatically (e.g. 'MyRAG' becomes 'MyRAG-2').

    Returns: imported_name, embedding_model, file_count, chunk_count, original_name.
    """
    client = _client(ctx)
    info = client.peek_rag_file(file_path)
    result = client.import_rag(file_path, name=name)
    return {
        "imported_name": result.get("name"),
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
    client = _client(ctx)
    rag_name = name or client.get_active_name()
    if not rag_name:
        raise RuntimeError("No RAG specified and no active RAG set.")

    if detach:
        client.detach_rag(rag_name)
    else:
        client.attach_rag(rag_name)

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
    client = _client(ctx)
    result = client.create_rag(
        name=name,
        folders=source_folders,
        description=description,
        embedding_model=embedding_model,
    )
    return {
        "name": result.get("name"),
        "db_path": result.get("db_path"),
        "status": "created",
    }


@mcp.tool()
def delete_rag(name: str, confirm: bool = False, ctx: Context = None) -> dict[str, str]:
    """Permanently delete a RAG database and all its indexed data.

    This action is irreversible. You must pass confirm=True to proceed.
    If the deleted RAG is the active one, the server will switch to the
    next available RAG automatically.
    """
    if not confirm:
        raise ValueError(
            "Deletion requires explicit confirmation. "
            "Call delete_rag(name='...', confirm=True) to proceed."
        )
    client = _client(ctx)
    client.delete_rag(name)
    return {"rag": name, "status": "deleted"}


# ---------------------------------------------------------------------------
# Model management tools
# ---------------------------------------------------------------------------


@mcp.tool()
def list_models(
    model_type: str | None = None,
    ctx: Context = None,
) -> list[dict[str, Any]]:
    """List all available embedding and reranker models with their status.

    Returns model name, type, dimensions, max_tokens, provider, status
    (bundled/downloaded/available/api), and description for each model.

    Optionally filter by model_type: 'embedding' or 'reranker'.
    """
    client = _client(ctx)
    return client.list_models(model_type=model_type)


@mcp.tool()
def get_model_info(model_name: str, ctx: Context = None) -> dict[str, Any]:
    """Get detailed information about a specific model.

    Returns all metadata including dimensions, max_tokens, description,
    use_case_tags, license, trust requirements, and download status.
    """
    client = _client(ctx)
    info = client.get_model_info(model_name)
    if info is None:
        raise ValueError(f"Model '{model_name}' not found in registry.")
    return info


# ---------------------------------------------------------------------------
# Monitoring tools
# ---------------------------------------------------------------------------


@mcp.tool()
def get_monitoring_metrics(ctx: Context = None) -> dict[str, Any]:
    """Get a comprehensive monitoring dashboard with system health, indexing,
    search, and embedding aggregate metrics.

    Returns system snapshot (CPU, memory, disk, uptime, connections),
    indexing aggregates (total runs, avg duration, throughput, errors),
    search aggregates (total queries, avg latency, avg top score),
    embedding aggregates (total batches, avg time, throughput),
    the last indexing run details, and vector store health info.
    """
    client = _client(ctx)
    return client.get_metrics_dashboard()


@mcp.tool()
def get_indexing_history(limit: int = 10, ctx: Context = None) -> list[dict[str, Any]]:
    """Get recent indexing run history with per-phase timing breakdown.

    Each entry includes: rag_name, status, processed_files, total_chunks,
    duration_seconds, scan/parse/chunk/embed/upsert/manifest seconds,
    chunks_per_second, error_count, started_at.
    """
    client = _client(ctx)
    return client.get_indexing_history(limit=limit)


@mcp.tool()
def get_search_stats(limit: int = 20, ctx: Context = None) -> list[dict[str, Any]]:
    """Get recent search query performance stats.

    Each entry includes: query_text, result_count, total_ms,
    vector_search_ms, bm25_ms, rerank_ms, mmr_ms, top_score, timestamp.
    """
    client = _client(ctx)
    return client.get_search_stats(limit=limit)


@mcp.tool()
def get_embedding_stats(limit: int = 20, ctx: Context = None) -> list[dict[str, Any]]:
    """Get recent embedding batch performance stats.

    Each entry includes: backend_type, model_name, batch_size,
    dimension, duration_ms, chunks_per_second, device, timestamp.
    """
    client = _client(ctx)
    return client.get_embedding_stats(limit=limit)


@mcp.tool()
def get_vector_store_details(ctx: Context = None) -> dict[str, Any]:
    """Get detailed vector store / ChromaDB health information.

    Returns total_chunks, total_files, db_size_mb, avg_chunks_per_file,
    hnsw_config (space, construction_ef, M), db_path, and collection info.
    """
    client = _client(ctx)
    return client.get_vector_store_details()


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
