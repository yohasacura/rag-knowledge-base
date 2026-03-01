"""Persistent metrics collection for RAG knowledge base operations.

Provides a thread-safe ``MetricsCollector`` singleton backed by SQLite
for durable storage. Records timing and throughput data for:

- Indexing pipeline phases (scan → parse → chunk → embed → upsert)
- Embedding performance (batches, throughput, latency)
- Vector store / ChromaDB health (size, counts, HNSW config)
- Search performance (latency breakdown per stage)
- System snapshots (CPU, memory, disk, daemon uptime, connections)

Used by the daemon process. Thin clients (CLI, MCP, WebUI) access
metrics via JSON-RPC endpoints.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag_kb.config import DATA_DIR

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes for structured metric events
# ---------------------------------------------------------------------------


@dataclass
class IndexingRunMetrics:
    """Summary of a single indexing run."""

    rag_name: str
    started_at: float  # time.time()
    duration_seconds: float = 0.0
    total_files: int = 0
    processed_files: int = 0
    skipped_files: int = 0
    total_chunks: int = 0
    error_count: int = 0
    status: str = "completed"  # completed | cancelled | error
    # Phase timings (seconds)
    scan_seconds: float = 0.0
    parse_seconds: float = 0.0
    chunk_seconds: float = 0.0
    embed_seconds: float = 0.0
    upsert_seconds: float = 0.0
    manifest_seconds: float = 0.0
    # Throughput
    chunks_per_second: float = 0.0
    files_per_second: float = 0.0
    is_full_reindex: bool = False


@dataclass
class EmbeddingBatchMetrics:
    """Metrics for a single embedding batch."""

    rag_name: str
    timestamp: float  # time.time()
    backend: str = ""  # sentence_transformer | openai | voyage
    model_name: str = ""
    batch_size: int = 0
    dimension: int = 0
    duration_ms: float = 0.0
    chunks_per_second: float = 0.0
    device: str = ""  # cpu | cuda | mps


@dataclass
class SearchQueryMetrics:
    """Metrics for a single search query."""

    rag_name: str
    timestamp: float
    query_length: int = 0
    top_k: int = 5
    results_returned: int = 0
    total_duration_ms: float = 0.0
    # Per-phase breakdown (ms)
    vector_search_ms: float = 0.0
    bm25_ms: float = 0.0
    fusion_ms: float = 0.0
    rerank_ms: float = 0.0
    mmr_ms: float = 0.0
    top_score: float = 0.0
    min_score: float = 0.0


@dataclass
class VectorStoreSnapshot:
    """Point-in-time snapshot of vector store health."""

    rag_name: str
    timestamp: float
    total_chunks: int = 0
    total_files: int = 0
    db_size_bytes: int = 0
    avg_chunks_per_file: float = 0.0
    collection_name: str = ""


@dataclass
class SystemSnapshot:
    """Point-in-time snapshot of system resource usage."""

    timestamp: float
    cpu_percent: float = 0.0
    memory_used_mb: float = 0.0
    memory_total_mb: float = 0.0
    memory_percent: float = 0.0
    disk_used_mb: float = 0.0
    disk_free_mb: float = 0.0
    daemon_uptime_seconds: float = 0.0
    active_connections: int = 0
    total_rpc_calls: int = 0
    process_memory_mb: float = 0.0


# ---------------------------------------------------------------------------
# SQLite-backed metrics store
# ---------------------------------------------------------------------------

_DB_FILE = DATA_DIR / "metrics.db"

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS indexing_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    rag_name TEXT NOT NULL,
    started_at REAL NOT NULL,
    duration_seconds REAL DEFAULT 0,
    total_files INTEGER DEFAULT 0,
    processed_files INTEGER DEFAULT 0,
    skipped_files INTEGER DEFAULT 0,
    total_chunks INTEGER DEFAULT 0,
    error_count INTEGER DEFAULT 0,
    status TEXT DEFAULT 'completed',
    scan_seconds REAL DEFAULT 0,
    parse_seconds REAL DEFAULT 0,
    chunk_seconds REAL DEFAULT 0,
    embed_seconds REAL DEFAULT 0,
    upsert_seconds REAL DEFAULT 0,
    manifest_seconds REAL DEFAULT 0,
    chunks_per_second REAL DEFAULT 0,
    files_per_second REAL DEFAULT 0,
    is_full_reindex INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS embedding_batches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    rag_name TEXT NOT NULL,
    timestamp REAL NOT NULL,
    backend TEXT DEFAULT '',
    model_name TEXT DEFAULT '',
    batch_size INTEGER DEFAULT 0,
    dimension INTEGER DEFAULT 0,
    duration_ms REAL DEFAULT 0,
    chunks_per_second REAL DEFAULT 0,
    device TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS search_queries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    rag_name TEXT NOT NULL,
    timestamp REAL NOT NULL,
    query_length INTEGER DEFAULT 0,
    top_k INTEGER DEFAULT 5,
    results_returned INTEGER DEFAULT 0,
    total_duration_ms REAL DEFAULT 0,
    vector_search_ms REAL DEFAULT 0,
    bm25_ms REAL DEFAULT 0,
    fusion_ms REAL DEFAULT 0,
    rerank_ms REAL DEFAULT 0,
    mmr_ms REAL DEFAULT 0,
    top_score REAL DEFAULT 0,
    min_score REAL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS vector_store_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    rag_name TEXT NOT NULL,
    timestamp REAL NOT NULL,
    total_chunks INTEGER DEFAULT 0,
    total_files INTEGER DEFAULT 0,
    db_size_bytes INTEGER DEFAULT 0,
    avg_chunks_per_file REAL DEFAULT 0,
    collection_name TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS system_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    cpu_percent REAL DEFAULT 0,
    memory_used_mb REAL DEFAULT 0,
    memory_total_mb REAL DEFAULT 0,
    memory_percent REAL DEFAULT 0,
    disk_used_mb REAL DEFAULT 0,
    disk_free_mb REAL DEFAULT 0,
    daemon_uptime_seconds REAL DEFAULT 0,
    active_connections INTEGER DEFAULT 0,
    total_rpc_calls INTEGER DEFAULT 0,
    process_memory_mb REAL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_indexing_runs_rag ON indexing_runs(rag_name, started_at);
CREATE INDEX IF NOT EXISTS idx_embedding_batches_rag ON embedding_batches(rag_name, timestamp);
CREATE INDEX IF NOT EXISTS idx_search_queries_rag ON search_queries(rag_name, timestamp);
CREATE INDEX IF NOT EXISTS idx_vector_store_snapshots_rag ON vector_store_snapshots(rag_name, timestamp);
CREATE INDEX IF NOT EXISTS idx_system_snapshots_ts ON system_snapshots(timestamp);
"""


class MetricsStore:
    """SQLite-backed persistent metrics storage.

    Thread-safe — each call acquires a short-lived connection from a
    shared serialized connection (SQLite WAL mode for read concurrency).
    """

    def __init__(self, db_path: Path | None = None):
        self._db_path = str(db_path or _DB_FILE)
        self._lock = threading.Lock()
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=5)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            conn = self._connect()
            try:
                conn.executescript(_SCHEMA_SQL)
                conn.commit()
            finally:
                conn.close()

    # ---- Writes ----

    def record_indexing_run(self, m: IndexingRunMetrics) -> None:
        sql = """
        INSERT INTO indexing_runs (
            rag_name, started_at, duration_seconds, total_files,
            processed_files, skipped_files, total_chunks, error_count,
            status, scan_seconds, parse_seconds, chunk_seconds,
            embed_seconds, upsert_seconds, manifest_seconds,
            chunks_per_second, files_per_second, is_full_reindex
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    sql,
                    (
                        m.rag_name,
                        m.started_at,
                        m.duration_seconds,
                        m.total_files,
                        m.processed_files,
                        m.skipped_files,
                        m.total_chunks,
                        m.error_count,
                        m.status,
                        m.scan_seconds,
                        m.parse_seconds,
                        m.chunk_seconds,
                        m.embed_seconds,
                        m.upsert_seconds,
                        m.manifest_seconds,
                        m.chunks_per_second,
                        m.files_per_second,
                        int(m.is_full_reindex),
                    ),
                )
                conn.commit()
            finally:
                conn.close()

    def record_embedding_batch(self, m: EmbeddingBatchMetrics) -> None:
        sql = """
        INSERT INTO embedding_batches (
            rag_name, timestamp, backend, model_name, batch_size,
            dimension, duration_ms, chunks_per_second, device
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    sql,
                    (
                        m.rag_name,
                        m.timestamp,
                        m.backend,
                        m.model_name,
                        m.batch_size,
                        m.dimension,
                        m.duration_ms,
                        m.chunks_per_second,
                        m.device,
                    ),
                )
                conn.commit()
            finally:
                conn.close()

    def record_search_query(self, m: SearchQueryMetrics) -> None:
        sql = """
        INSERT INTO search_queries (
            rag_name, timestamp, query_length, top_k, results_returned,
            total_duration_ms, vector_search_ms, bm25_ms, fusion_ms,
            rerank_ms, mmr_ms, top_score, min_score
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    sql,
                    (
                        m.rag_name,
                        m.timestamp,
                        m.query_length,
                        m.top_k,
                        m.results_returned,
                        m.total_duration_ms,
                        m.vector_search_ms,
                        m.bm25_ms,
                        m.fusion_ms,
                        m.rerank_ms,
                        m.mmr_ms,
                        m.top_score,
                        m.min_score,
                    ),
                )
                conn.commit()
            finally:
                conn.close()

    def record_vector_store_snapshot(self, m: VectorStoreSnapshot) -> None:
        sql = """
        INSERT INTO vector_store_snapshots (
            rag_name, timestamp, total_chunks, total_files,
            db_size_bytes, avg_chunks_per_file, collection_name
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    sql,
                    (
                        m.rag_name,
                        m.timestamp,
                        m.total_chunks,
                        m.total_files,
                        m.db_size_bytes,
                        m.avg_chunks_per_file,
                        m.collection_name,
                    ),
                )
                conn.commit()
            finally:
                conn.close()

    def record_system_snapshot(self, m: SystemSnapshot) -> None:
        sql = """
        INSERT INTO system_snapshots (
            timestamp, cpu_percent, memory_used_mb, memory_total_mb,
            memory_percent, disk_used_mb, disk_free_mb,
            daemon_uptime_seconds, active_connections, total_rpc_calls,
            process_memory_mb
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    sql,
                    (
                        m.timestamp,
                        m.cpu_percent,
                        m.memory_used_mb,
                        m.memory_total_mb,
                        m.memory_percent,
                        m.disk_used_mb,
                        m.disk_free_mb,
                        m.daemon_uptime_seconds,
                        m.active_connections,
                        m.total_rpc_calls,
                        m.process_memory_mb,
                    ),
                )
                conn.commit()
            finally:
                conn.close()

    # ---- Reads ----

    def _fetch_all(self, sql: str, params: tuple = ()) -> list[dict]:
        with self._lock:
            conn = self._connect()
            try:
                rows = conn.execute(sql, params).fetchall()
                return [dict(r) for r in rows]
            finally:
                conn.close()

    def _fetch_one(self, sql: str, params: tuple = ()) -> dict | None:
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(sql, params).fetchone()
                return dict(row) if row else None
            finally:
                conn.close()

    def get_indexing_history(self, rag_name: str | None = None, limit: int = 50) -> list[dict]:
        if rag_name:
            return self._fetch_all(
                "SELECT * FROM indexing_runs WHERE rag_name = ? ORDER BY started_at DESC LIMIT ?",
                (rag_name, limit),
            )
        return self._fetch_all(
            "SELECT * FROM indexing_runs ORDER BY started_at DESC LIMIT ?",
            (limit,),
        )

    def get_embedding_stats(self, rag_name: str | None = None, limit: int = 100) -> list[dict]:
        if rag_name:
            return self._fetch_all(
                "SELECT * FROM embedding_batches WHERE rag_name = ? "
                "ORDER BY timestamp DESC LIMIT ?",
                (rag_name, limit),
            )
        return self._fetch_all(
            "SELECT * FROM embedding_batches ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        )

    def get_search_stats(self, rag_name: str | None = None, limit: int = 100) -> list[dict]:
        if rag_name:
            return self._fetch_all(
                "SELECT * FROM search_queries WHERE rag_name = ? ORDER BY timestamp DESC LIMIT ?",
                (rag_name, limit),
            )
        return self._fetch_all(
            "SELECT * FROM search_queries ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        )

    def get_vector_store_history(self, rag_name: str | None = None, limit: int = 50) -> list[dict]:
        if rag_name:
            return self._fetch_all(
                "SELECT * FROM vector_store_snapshots WHERE rag_name = ? "
                "ORDER BY timestamp DESC LIMIT ?",
                (rag_name, limit),
            )
        return self._fetch_all(
            "SELECT * FROM vector_store_snapshots ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        )

    def get_system_timeline(self, limit: int = 100) -> list[dict]:
        return self._fetch_all(
            "SELECT * FROM system_snapshots ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        )

    def get_dashboard_summary(self, rag_name: str | None = None) -> dict:
        """Aggregate summary suitable for a monitoring dashboard."""
        summary: dict[str, Any] = {}

        # Last indexing run
        if rag_name:
            last_run = self._fetch_one(
                "SELECT * FROM indexing_runs WHERE rag_name = ? ORDER BY started_at DESC LIMIT 1",
                (rag_name,),
            )
        else:
            last_run = self._fetch_one(
                "SELECT * FROM indexing_runs ORDER BY started_at DESC LIMIT 1",
            )
        summary["last_indexing_run"] = last_run

        # Indexing aggregates
        if rag_name:
            agg = self._fetch_one(
                "SELECT COUNT(*) as total_runs, "
                "AVG(duration_seconds) as avg_duration, "
                "AVG(chunks_per_second) as avg_throughput, "
                "SUM(total_chunks) as total_chunks_indexed, "
                "SUM(error_count) as total_errors "
                "FROM indexing_runs WHERE rag_name = ?",
                (rag_name,),
            )
        else:
            agg = self._fetch_one(
                "SELECT COUNT(*) as total_runs, "
                "AVG(duration_seconds) as avg_duration, "
                "AVG(chunks_per_second) as avg_throughput, "
                "SUM(total_chunks) as total_chunks_indexed, "
                "SUM(error_count) as total_errors "
                "FROM indexing_runs",
            )
        summary["indexing_aggregates"] = agg

        # Embedding aggregates
        if rag_name:
            emb_agg = self._fetch_one(
                "SELECT COUNT(*) as total_batches, "
                "AVG(duration_ms) as avg_batch_ms, "
                "AVG(chunks_per_second) as avg_throughput, "
                "SUM(batch_size) as total_texts_embedded "
                "FROM embedding_batches WHERE rag_name = ?",
                (rag_name,),
            )
        else:
            emb_agg = self._fetch_one(
                "SELECT COUNT(*) as total_batches, "
                "AVG(duration_ms) as avg_batch_ms, "
                "AVG(chunks_per_second) as avg_throughput, "
                "SUM(batch_size) as total_texts_embedded "
                "FROM embedding_batches",
            )
        summary["embedding_aggregates"] = emb_agg

        # Search aggregates
        if rag_name:
            search_agg = self._fetch_one(
                "SELECT COUNT(*) as total_queries, "
                "AVG(total_duration_ms) as avg_latency_ms, "
                "AVG(results_returned) as avg_results, "
                "AVG(top_score) as avg_top_score "
                "FROM search_queries WHERE rag_name = ?",
                (rag_name,),
            )
        else:
            search_agg = self._fetch_one(
                "SELECT COUNT(*) as total_queries, "
                "AVG(total_duration_ms) as avg_latency_ms, "
                "AVG(results_returned) as avg_results, "
                "AVG(top_score) as avg_top_score "
                "FROM search_queries",
            )
        summary["search_aggregates"] = search_agg

        # Latest vector store snapshot
        if rag_name:
            vs = self._fetch_one(
                "SELECT * FROM vector_store_snapshots WHERE rag_name = ? "
                "ORDER BY timestamp DESC LIMIT 1",
                (rag_name,),
            )
        else:
            vs = self._fetch_one(
                "SELECT * FROM vector_store_snapshots ORDER BY timestamp DESC LIMIT 1",
            )
        summary["vector_store"] = vs

        # Latest system snapshot
        sys_snap = self._fetch_one(
            "SELECT * FROM system_snapshots ORDER BY timestamp DESC LIMIT 1",
        )
        summary["system"] = sys_snap

        return summary

    # ---- Maintenance ----

    def purge_old(self, retention_days: int = 30) -> int:
        """Delete records older than *retention_days*. Returns rows deleted."""
        cutoff = time.time() - (retention_days * 86400)
        tables_cols = [
            ("indexing_runs", "started_at"),
            ("embedding_batches", "timestamp"),
            ("search_queries", "timestamp"),
            ("vector_store_snapshots", "timestamp"),
            ("system_snapshots", "timestamp"),
        ]
        total = 0
        with self._lock:
            conn = self._connect()
            try:
                for table, col in tables_cols:
                    cur = conn.execute(f"DELETE FROM {table} WHERE {col} < ?", (cutoff,))
                    total += cur.rowcount
                conn.commit()
            finally:
                conn.close()
        if total:
            logger.info("Purged %d old metric rows (retention=%dd)", total, retention_days)
        return total


# ---------------------------------------------------------------------------
# MetricsCollector — singleton facade used by instrumented code
# ---------------------------------------------------------------------------


class MetricsCollector:
    """Thread-safe singleton that instruments code writes to.

    Usage::

        mc = MetricsCollector.get()
        mc.record_indexing_run(IndexingRunMetrics(...))
    """

    _instance: MetricsCollector | None = None
    _lock = threading.Lock()

    def __init__(self, store: MetricsStore | None = None):
        self._store = store or MetricsStore()

    @classmethod
    def get(cls, store: MetricsStore | None = None) -> MetricsCollector:
        """Return the global MetricsCollector, creating it if needed."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(store)
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (for testing)."""
        with cls._lock:
            cls._instance = None

    @property
    def store(self) -> MetricsStore:
        return self._store

    # ---- Recording shortcuts ----

    def record_indexing_run(self, m: IndexingRunMetrics) -> None:
        try:
            self._store.record_indexing_run(m)
        except Exception:
            logger.debug("Failed to record indexing run metric", exc_info=True)

    def record_embedding_batch(self, m: EmbeddingBatchMetrics) -> None:
        try:
            self._store.record_embedding_batch(m)
        except Exception:
            logger.debug("Failed to record embedding batch metric", exc_info=True)

    def record_search_query(self, m: SearchQueryMetrics) -> None:
        try:
            self._store.record_search_query(m)
        except Exception:
            logger.debug("Failed to record search query metric", exc_info=True)

    def record_vector_store_snapshot(self, m: VectorStoreSnapshot) -> None:
        try:
            self._store.record_vector_store_snapshot(m)
        except Exception:
            logger.debug("Failed to record vector store snapshot", exc_info=True)

    def record_system_snapshot(self, m: SystemSnapshot) -> None:
        try:
            self._store.record_system_snapshot(m)
        except Exception:
            logger.debug("Failed to record system snapshot", exc_info=True)

    # ---- Query shortcuts ----

    def get_dashboard(self, rag_name: str | None = None) -> dict:
        return self._store.get_dashboard_summary(rag_name)

    def get_indexing_history(self, rag_name: str | None = None, limit: int = 50) -> list[dict]:
        return self._store.get_indexing_history(rag_name, limit)

    def get_embedding_stats(self, rag_name: str | None = None, limit: int = 100) -> list[dict]:
        return self._store.get_embedding_stats(rag_name, limit)

    def get_search_stats(self, rag_name: str | None = None, limit: int = 100) -> list[dict]:
        return self._store.get_search_stats(rag_name, limit)

    def get_vector_store_history(self, rag_name: str | None = None, limit: int = 50) -> list[dict]:
        return self._store.get_vector_store_history(rag_name, limit)

    def get_system_timeline(self, limit: int = 100) -> list[dict]:
        return self._store.get_system_timeline(limit)

    def purge_old(self, retention_days: int = 30) -> int:
        return self._store.purge_old(retention_days)


# ---------------------------------------------------------------------------
# System snapshot helper (uses psutil when available)
# ---------------------------------------------------------------------------


def capture_system_snapshot(
    daemon_uptime: float = 0.0,
    active_connections: int = 0,
    total_rpc_calls: int = 0,
) -> SystemSnapshot:
    """Capture current system resource usage.

    Uses *psutil* if available; falls back to zero values gracefully.
    """
    snap = SystemSnapshot(
        timestamp=time.time(),
        daemon_uptime_seconds=daemon_uptime,
        active_connections=active_connections,
        total_rpc_calls=total_rpc_calls,
    )

    try:
        import psutil

        # System-wide
        snap.cpu_percent = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory()
        snap.memory_used_mb = mem.used / (1024 * 1024)
        snap.memory_total_mb = mem.total / (1024 * 1024)
        snap.memory_percent = mem.percent

        # Data directory disk
        disk = psutil.disk_usage(str(DATA_DIR))
        snap.disk_used_mb = disk.used / (1024 * 1024)
        snap.disk_free_mb = disk.free / (1024 * 1024)

        # Process-level memory
        proc = psutil.Process(os.getpid())
        snap.process_memory_mb = proc.memory_info().rss / (1024 * 1024)

    except ImportError:
        logger.debug("psutil not installed — system metrics unavailable")
    except Exception:
        logger.debug("Failed to capture system metrics", exc_info=True)

    return snap
