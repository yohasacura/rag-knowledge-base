"""Vector store layer wrapping ChromaDB (Apache 2.0).

Improvements:
  - Tunable **HNSW parameters** (``ef_construction``, ``M``) for better recall.
  - Accepts numpy arrays for embeddings (avoids list conversion overhead).
  - ``get_all_documents()`` for BM25 keyword index construction.
  - Score threshold filtering in ``search()``.
  - Larger default upsert batch (10 000).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single search hit."""

    text: str
    source_file: str
    chunk_index: int
    score: float  # cosine similarity (higher = more similar)
    metadata: dict[str, str] = field(default_factory=dict)
    embedding: list[float] | None = None


@dataclass
class StoreStats:
    """Aggregated statistics about the vector store."""

    total_chunks: int = 0
    total_files: int = 0
    files: list[str] = field(default_factory=list)
    db_size_bytes: int = 0
    avg_chunks_per_file: float = 0.0


class _NoOpEmbeddingFunction:
    """No-op embedding function to prevent ChromaDB from using its default.

    We always provide pre-computed embeddings in ``upsert()`` calls, so the
    collection must **not** run its own embedding pipeline.  Without this,
    ChromaDB assigns a ``DefaultEmbeddingFunction`` whose tokeniser can
    raise ``TextEncodeInput must be Union[…]`` on malformed document text.

    Implements the minimal interface expected by ChromaDB >= 1.x so that
    ``get_or_create_collection()`` validation passes.
    """

    @staticmethod
    def name() -> str:
        return "noop"

    def __call__(self, input: list[str]) -> list[list[float]]:
        # Should never actually be called — we always pass embeddings.
        raise NotImplementedError(
            "NoOpEmbeddingFunction should never be called; "
            "provide embeddings explicitly in upsert/add."
        )

    # ChromaDB >= 1.x may call these during collection validation
    @staticmethod
    def default_space() -> str:
        return "cosine"

    @staticmethod
    def supported_spaces() -> list[str]:
        return ["cosine", "l2", "ip"]

    @staticmethod
    def get_config() -> dict:
        return {}

    @staticmethod
    def build_from_config(config: dict) -> "_NoOpEmbeddingFunction":
        return _NoOpEmbeddingFunction()

    @staticmethod
    def validate_config(config: dict) -> None:
        pass

    def validate_config_update(self, old_config: dict, new_config: dict) -> None:
        pass

    @staticmethod
    def is_legacy() -> bool:
        return False


class VectorStore:
    """Abstraction over a ChromaDB persistent collection."""

    COLLECTION_NAME = "rag_chunks"

    def __init__(
        self,
        db_path: str,
        hnsw_ef_construction: int = 200,
        hnsw_m: int = 32,
    ) -> None:
        try:
            import chromadb
            from chromadb.config import (
                DEFAULT_DATABASE,
                DEFAULT_TENANT,
                Settings,
            )
        except ImportError as exc:
            raise ImportError("chromadb is required: pip install chromadb") from exc

        self._db_path = db_path
        Path(db_path).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False),
            tenant=DEFAULT_TENANT,
            database=DEFAULT_DATABASE,
        )
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={
                "hnsw:space": "cosine",
                "hnsw:construction_ef": hnsw_ef_construction,
                "hnsw:M": hnsw_m,
            },
        )
        logger.debug("VectorStore opened at %s (%d chunks)", db_path, self._collection.count())

    def close(self, *, force: bool = False) -> None:
        """Release the ChromaDB client references.

        Parameters
        ----------
        force : bool
            When *True* (e.g. right before deleting the database folder),
            stop the ChromaDB system, delete the Rust bindings that hold
            memory-mapped HNSW files, and clear the singleton cache so the
            path can be safely deleted on Windows.
        """
        if force and self._client is not None:
            try:
                # Delete the Rust bindings first — this releases the native
                # memory-mapped .bin file handles on Windows.
                server = getattr(self._client, "_server", None)
                if server is not None and hasattr(server, "bindings"):
                    del server.bindings

                # Stop the system (stops remaining components)
                system = getattr(self._client, "_system", None)
                if system is not None:
                    try:
                        system.stop()
                    except Exception:  # noqa: BLE001
                        pass

                # Clear the singleton cache so the stopped system is not
                # reused and the path can be opened fresh later.
                if hasattr(self._client, "clear_system_cache"):
                    self._client.clear_system_cache()
            except Exception:  # noqa: BLE001
                logger.debug("Ignoring error while stopping ChromaDB system", exc_info=True)
        self._collection = None  # type: ignore[assignment]
        self._client = None      # type: ignore[assignment]

    def __enter__(self) -> "VectorStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    @staticmethod
    def _sanitise_metadatas(
        metadatas: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        """Ensure every metadata value is a ChromaDB-compatible type.

        ChromaDB only accepts str, int, float, and bool.  ``None`` values
        and other types (bytes, list, etc.) cause internal errors.  This
        method converts or drops invalid entries so the upsert never fails
        due to metadata.
        """
        clean: list[dict[str, str]] = []
        for meta in metadatas:
            fixed: dict[str, str] = {}
            for k, v in meta.items():
                if v is None:
                    fixed[k] = ""
                elif isinstance(v, (str, int, float, bool)):
                    fixed[k] = v
                else:
                    fixed[k] = str(v)
            clean.append(fixed)
        return clean

    def add_documents(
        self,
        ids: list[str],
        texts: list[str],
        embeddings: list[list[float]] | np.ndarray,
        metadatas: list[dict[str, str]],
    ) -> None:
        """Add (or update) documents in the collection.

        *embeddings* can be a 2-D numpy array or a list of lists.
        """
        if not ids:
            return

        # Sanitise metadata — ChromaDB rejects None and non-scalar types
        metadatas = self._sanitise_metadatas(metadatas)

        # ChromaDB accepts numpy ndarrays directly (>= 0.4); avoid the
        # expensive .tolist() conversion that allocates millions of
        # Python float objects for large batches.
        emb: list[list[float]] | np.ndarray = embeddings

        batch = 10_000
        for i in range(0, len(ids), batch):
            self._collection.upsert(
                ids=ids[i : i + batch],
                documents=texts[i : i + batch],
                embeddings=emb[i : i + batch],
                metadatas=metadatas[i : i + batch],
            )
        logger.debug("Upserted %d chunks", len(ids))

    def delete_by_source(self, source_file: str) -> int:
        """Delete all chunks belonging to *source_file*. Return count deleted."""
        results = self._collection.get(where={"source_file": source_file})
        chunk_ids = results["ids"]
        if chunk_ids:
            self._collection.delete(ids=chunk_ids)
        logger.debug("Deleted %d chunks for %s", len(chunk_ids), source_file)
        return len(chunk_ids)

    def batch_delete_by_sources(self, source_files: list[str]) -> int:
        """Delete all chunks belonging to any of the given *source_files*.

        Far more efficient than calling ``delete_by_source()`` N times:
        uses a single ``$in`` query for smaller batches or a scan-and-filter
        for larger ones, minimising ChromaDB round-trips.

        Returns the total number of chunks deleted.
        """
        if not source_files:
            return 0

        total_deleted = 0
        # ChromaDB $in operator has a practical limit; process in groups of 100
        batch = 100
        for i in range(0, len(source_files), batch):
            sub = source_files[i : i + batch]
            try:
                if len(sub) == 1:
                    where_clause: dict = {"source_file": sub[0]}
                else:
                    where_clause = {"source_file": {"$in": sub}}
                results = self._collection.get(where=where_clause)
                chunk_ids = results["ids"]
                if chunk_ids:
                    # ChromaDB delete also has limits; sub-batch if needed
                    del_batch = 10_000
                    for j in range(0, len(chunk_ids), del_batch):
                        self._collection.delete(ids=chunk_ids[j : j + del_batch])
                    total_deleted += len(chunk_ids)
            except Exception as exc:
                logger.warning("batch_delete_by_sources failed for sub-batch: %s", exc)
                # Fallback: delete one by one
                for src in sub:
                    total_deleted += self.delete_by_source(src)
        logger.debug("Batch-deleted %d chunks for %d source files",
                     total_deleted, len(source_files))
        return total_deleted

    def get_embeddings_by_ids(self, ids: list[str]) -> dict[str, list[float]]:
        """Return stored embeddings for the given document *ids*.

        Returns a mapping ``{id: embedding}`` for each ID found.
        """
        if not ids:
            return {}
        results = self._collection.get(ids=ids, include=["embeddings"])
        out: dict[str, list[float]] = {}
        if len(results.get("ids") or []) > 0 and results.get("embeddings") is not None and len(results["embeddings"]) > 0:
            for doc_id, emb in zip(results["ids"], results["embeddings"]):
                if emb is not None:
                    out[doc_id] = emb
        return out

    def clear(self) -> None:
        """Remove all documents."""
        self._client.delete_collection(self.COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def search(
        self,
        query_embedding: list[float],
        n_results: int = 5,
        where: dict[str, Any] | None = None,
        min_score: float = 0.0,
        include_embeddings: bool = False,
    ) -> list[SearchResult]:
        """Return the *n_results* most similar chunks.

        Chunks with cosine similarity below *min_score* are filtered out.
        If *include_embeddings* is True, each result includes its stored embedding.
        """
        total = self._collection.count()
        if total == 0:
            return []
        include = ["documents", "metadatas", "distances"]
        if include_embeddings:
            include.append("embeddings")
        kwargs: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": min(n_results, total),
            "include": include,
        }
        if where:
            kwargs["where"] = where

        results = self._collection.query(**kwargs)

        hits: list[SearchResult] = []
        if not results["ids"] or not results["ids"][0]:
            return hits

        embeddings_list = results.get("embeddings", [[]])[0] if include_embeddings else []

        for idx, doc_id in enumerate(results["ids"][0]):
            meta = (results["metadatas"][0][idx] if results["metadatas"] else {}) or {}
            dist = results["distances"][0][idx] if results["distances"] else 0.0
            similarity = round(1.0 - dist, 4)  # cosine distance → similarity
            if similarity < min_score:
                continue
            text = results["documents"][0][idx] if results["documents"] else ""
            emb = embeddings_list[idx] if include_embeddings and idx < len(embeddings_list) else None
            hits.append(
                SearchResult(
                    text=text,
                    source_file=meta.get("source_file", ""),
                    chunk_index=int(meta.get("chunk_index", 0)),
                    score=similarity,
                    metadata=meta,
                    embedding=emb,
                )
            )
        return hits

    def get_all_documents(self) -> tuple[list[str], list[str], list[dict[str, str]]]:
        """Return *all* (ids, texts, metadatas) — used for BM25 index building.

        Warning: pulls the entire collection into memory.  Fine for collections
        under ~500 k chunks.
        """
        total = self._collection.count()
        if total == 0:
            return [], [], []

        results = self._collection.get(include=["documents", "metadatas"])
        ids = results["ids"]
        texts = results["documents"] or [""] * len(ids)
        metas = results["metadatas"] or [{}] * len(ids)
        return ids, texts, metas  # type: ignore[return-value]

    def get_by_source(self, source_file: str) -> list[dict[str, Any]]:
        """Return all chunks for a given source file."""
        results = self._collection.get(
            where={"source_file": source_file},
            include=["documents", "metadatas"],
        )
        items: list[dict[str, Any]] = []
        for i, doc_id in enumerate(results["ids"]):
            items.append({
                "id": doc_id,
                "text": results["documents"][i] if results["documents"] else "",
                "metadata": results["metadatas"][i] if results["metadatas"] else {},
            })
        return sorted(items, key=lambda x: int(x.get("metadata", {}).get("chunk_index", 0)))

    def list_sources(self) -> list[str]:
        """Return distinct source file paths."""
        results = self._collection.get(include=["metadatas"])
        sources: set[str] = set()
        if results["metadatas"]:
            for meta in results["metadatas"]:
                src = (meta or {}).get("source_file", "")
                if src:
                    sources.add(src)
        return sorted(sources)

    def list_files_with_counts(self) -> dict[str, int]:
        """Return ``{source_file: chunk_count}`` for every file in the store.

        Only fetches metadata (no texts or embeddings), so this is
        significantly cheaper than ``get_all_documents()`` for large collections.
        """
        results = self._collection.get(include=["metadatas"])
        counts: dict[str, int] = {}
        if results["metadatas"]:
            for meta in results["metadatas"]:
                src = (meta or {}).get("source_file", "")
                if src:
                    counts[src] = counts.get(src, 0) + 1
        return counts

    def get_stats(self) -> StoreStats:
        """Return aggregate statistics."""
        sources = self.list_sources()
        total_chunks = self._collection.count()
        total_files = len(sources)
        db_size = self._get_db_size()
        avg_cpf = round(total_chunks / total_files, 1) if total_files else 0.0
        return StoreStats(
            total_chunks=total_chunks,
            total_files=total_files,
            files=sources,
            db_size_bytes=db_size,
            avg_chunks_per_file=avg_cpf,
        )

    def get_detailed_stats(self) -> dict[str, Any]:
        """Return extended stats including HNSW config and DB size."""
        stats = self.get_stats()
        hnsw_config = {}
        try:
            meta = self._collection.metadata or {}
            hnsw_config = {
                "space": meta.get("hnsw:space", "cosine"),
                "construction_ef": meta.get("hnsw:construction_ef", "?"),
                "M": meta.get("hnsw:M", "?"),
            }
        except Exception:
            pass
        return {
            "total_chunks": stats.total_chunks,
            "total_files": stats.total_files,
            "db_size_bytes": stats.db_size_bytes,
            "db_size_mb": round(stats.db_size_bytes / (1024 * 1024), 2),
            "avg_chunks_per_file": stats.avg_chunks_per_file,
            "collection_name": self.COLLECTION_NAME,
            "db_path": self._db_path,
            "hnsw_config": hnsw_config,
        }

    def _get_db_size(self) -> int:
        """Return total size of the ChromaDB directory in bytes."""
        try:
            return sum(
                f.stat().st_size
                for f in Path(self._db_path).rglob("*")
                if f.is_file()
            )
        except Exception:
            return 0

    def count(self) -> int:
        return self._collection.count()
