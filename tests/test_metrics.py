"""Metrics store tests — record/read round-trips, dashboard, purge, singleton."""

from __future__ import annotations

import threading
import time

import pytest

from rag_kb.metrics import (
    EmbeddingBatchMetrics,
    IndexingRunMetrics,
    MetricsCollector,
    MetricsStore,
    SearchQueryMetrics,
    SystemSnapshot,
    VectorStoreSnapshot,
)


@pytest.fixture
def store(tmp_path):
    """Fresh MetricsStore backed by temp SQLite DB."""
    return MetricsStore(db_path=tmp_path / "metrics.db")


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset MetricsCollector singleton between tests."""
    MetricsCollector.reset()
    yield
    MetricsCollector.reset()


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class TestMetricsSchema:
    def test_creates_db_file(self, tmp_path):
        db = tmp_path / "new_metrics.db"
        MetricsStore(db_path=db)
        assert db.exists()

    def test_creates_parent_dirs(self, tmp_path):
        db = tmp_path / "sub" / "deep" / "metrics.db"
        MetricsStore(db_path=db)
        assert db.exists()


# ---------------------------------------------------------------------------
# Indexing runs
# ---------------------------------------------------------------------------


class TestIndexingRuns:
    def test_record_and_read(self, store):
        m = IndexingRunMetrics(
            rag_name="test-rag",
            started_at=time.time(),
            duration_seconds=5.2,
            total_files=10,
            processed_files=8,
            skipped_files=2,
            total_chunks=100,
            error_count=0,
            status="completed",
        )
        store.record_indexing_run(m)

        rows = store.get_indexing_history("test-rag")
        assert len(rows) == 1
        assert rows[0]["rag_name"] == "test-rag"
        assert rows[0]["total_files"] == 10
        assert rows[0]["status"] == "completed"

    def test_multiple_runs(self, store):
        for i in range(5):
            store.record_indexing_run(
                IndexingRunMetrics(
                    rag_name="multi",
                    started_at=time.time() + i,
                    total_chunks=i * 10,
                )
            )
        rows = store.get_indexing_history("multi")
        assert len(rows) == 5

    def test_filter_by_rag_name(self, store):
        store.record_indexing_run(
            IndexingRunMetrics(rag_name="rag-a", started_at=time.time())
        )
        store.record_indexing_run(
            IndexingRunMetrics(rag_name="rag-b", started_at=time.time())
        )
        assert len(store.get_indexing_history("rag-a")) == 1
        assert len(store.get_indexing_history()) == 2  # all


# ---------------------------------------------------------------------------
# Embedding batches
# ---------------------------------------------------------------------------


class TestEmbeddingBatches:
    def test_record_and_read(self, store):
        m = EmbeddingBatchMetrics(
            rag_name="test-rag",
            timestamp=time.time(),
            backend="sentence_transformer",
            model_name="all-MiniLM-L6-v2",
            batch_size=64,
            dimension=384,
            duration_ms=120.5,
            chunks_per_second=531.1,
            device="cpu",
        )
        store.record_embedding_batch(m)
        rows = store.get_embedding_stats("test-rag")
        assert len(rows) == 1
        assert rows[0]["backend"] == "sentence_transformer"
        assert rows[0]["dimension"] == 384


# ---------------------------------------------------------------------------
# Search queries
# ---------------------------------------------------------------------------


class TestSearchQueries:
    def test_record_and_read(self, store):
        m = SearchQueryMetrics(
            rag_name="test-rag",
            timestamp=time.time(),
            query_length=25,
            top_k=5,
            results_returned=3,
            total_duration_ms=45.0,
            vector_search_ms=20.0,
            bm25_ms=10.0,
            top_score=0.85,
        )
        store.record_search_query(m)
        rows = store.get_search_stats("test-rag")
        assert len(rows) == 1
        assert rows[0]["query_length"] == 25
        assert rows[0]["top_score"] == pytest.approx(0.85)


# ---------------------------------------------------------------------------
# Vector store snapshots
# ---------------------------------------------------------------------------


class TestVectorStoreSnapshots:
    def test_record_and_read(self, store):
        m = VectorStoreSnapshot(
            rag_name="test-rag",
            timestamp=time.time(),
            total_chunks=500,
            total_files=20,
            db_size_bytes=1024000,
        )
        store.record_vector_store_snapshot(m)
        rows = store.get_vector_store_history("test-rag")
        assert len(rows) == 1
        assert rows[0]["total_chunks"] == 500


# ---------------------------------------------------------------------------
# System snapshots
# ---------------------------------------------------------------------------


class TestSystemSnapshots:
    def test_record_and_read(self, store):
        m = SystemSnapshot(
            timestamp=time.time(),
            cpu_percent=42.5,
            memory_used_mb=4096.0,
            memory_total_mb=16384.0,
        )
        store.record_system_snapshot(m)
        rows = store.get_system_timeline()
        assert len(rows) == 1
        assert rows[0]["cpu_percent"] == pytest.approx(42.5)


# ---------------------------------------------------------------------------
# Dashboard summary
# ---------------------------------------------------------------------------


class TestDashboardSummary:
    def test_empty_database(self, store):
        summary = store.get_dashboard_summary()
        assert summary["last_indexing_run"] is None
        assert summary["system"] is None

    def test_with_data(self, store):
        store.record_indexing_run(
            IndexingRunMetrics(
                rag_name="dash",
                started_at=time.time(),
                total_chunks=50,
            )
        )
        store.record_search_query(
            SearchQueryMetrics(
                rag_name="dash",
                timestamp=time.time(),
                results_returned=3,
            )
        )
        summary = store.get_dashboard_summary("dash")
        assert summary["last_indexing_run"] is not None
        assert summary["indexing_aggregates"]["total_runs"] == 1
        assert summary["search_aggregates"]["total_queries"] == 1


# ---------------------------------------------------------------------------
# Purge
# ---------------------------------------------------------------------------


class TestPurge:
    def test_purge_old_records(self, store):
        # Insert old records (timestamp 60 days ago)
        old_ts = time.time() - 60 * 86400
        store.record_indexing_run(
            IndexingRunMetrics(rag_name="old", started_at=old_ts)
        )
        store.record_search_query(
            SearchQueryMetrics(rag_name="old", timestamp=old_ts)
        )
        # Insert recent records
        store.record_indexing_run(
            IndexingRunMetrics(rag_name="new", started_at=time.time())
        )

        deleted = store.purge_old(retention_days=30)
        assert deleted >= 2  # old indexing + old search

        # Recent should survive
        assert len(store.get_indexing_history("new")) == 1
        assert len(store.get_indexing_history("old")) == 0


# ---------------------------------------------------------------------------
# MetricsCollector singleton
# ---------------------------------------------------------------------------


class TestMetricsCollector:
    def test_singleton(self, store):
        mc1 = MetricsCollector.get(store)
        mc2 = MetricsCollector.get()
        assert mc1 is mc2

    def test_reset(self, store):
        mc1 = MetricsCollector.get(store)
        MetricsCollector.reset()
        mc2 = MetricsCollector.get(store)
        assert mc1 is not mc2

    def test_record_via_collector(self, store):
        mc = MetricsCollector.get(store)
        mc.record_indexing_run(
            IndexingRunMetrics(rag_name="coll", started_at=time.time())
        )
        assert len(mc.get_indexing_history("coll")) == 1

    def test_dashboard_via_collector(self, store):
        mc = MetricsCollector.get(store)
        dashboard = mc.get_dashboard()
        assert isinstance(dashboard, dict)


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestMetricsThreadSafety:
    def test_concurrent_writes(self, store):
        errors = []

        def writer(name: str, count: int):
            try:
                for i in range(count):
                    store.record_search_query(
                        SearchQueryMetrics(
                            rag_name=name,
                            timestamp=time.time(),
                            results_returned=i,
                        )
                    )
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer, args=("t1", 20)),
            threading.Thread(target=writer, args=("t2", 20)),
            threading.Thread(target=writer, args=("t3", 20)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15)

        assert errors == []
        all_rows = store.get_search_stats()
        assert len(all_rows) == 60
