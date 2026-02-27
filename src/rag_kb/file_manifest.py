"""File manifest — fast change-detection backed by SQLite.

Stores file path, mtime, content hash and chunk count for every indexed file.
This replaces per-file ChromaDB queries during incremental indexing, turning
change detection from O(N × query) to O(N × stat).
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import NamedTuple

import xxhash

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class FileRecord(NamedTuple):
    """A single row in the manifest."""

    path: str
    mtime_ns: int
    content_hash: str
    chunk_count: int
    last_indexed: str


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


class FileManifest:
    """SQLite-backed index of every file known to a RAG.

    Typical location: ``<rag_dir>/file_manifest.db``
    """

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS files (
        path         TEXT PRIMARY KEY,
        mtime_ns     INTEGER NOT NULL,
        content_hash TEXT    NOT NULL,
        chunk_count  INTEGER NOT NULL DEFAULT 0,
        last_indexed TEXT    NOT NULL DEFAULT ''
    );
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(self._SCHEMA)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def is_changed(self, path: str | Path) -> bool:
        """Return *True* if *path* is new or modified since last indexing.

        Check order (fast → slow):
          1. Not in manifest → changed
          2. content_hash is empty (invalidated) → changed
          3. mtime_ns differs → compute hash → compare
          4. Hash matches    → unchanged
        """
        str_path = str(path)
        p = Path(path)
        row = self._get(str_path)
        if row is None:
            return True  # new file

        if not row.content_hash:
            return True  # invalidated — needs re-indexing

        try:
            current_mtime_ns = p.stat().st_mtime_ns
        except OSError:
            return True  # can't stat → treat as changed

        if current_mtime_ns == row.mtime_ns:
            return False  # fast path — nothing changed

        # mtime changed — verify with content hash
        current_hash = self._hash_file(p)
        return current_hash != row.content_hash

    def batch_filter_changed(self, paths: list[str]) -> list[str]:
        """Return the subset of *paths* that are new or modified.

        Optimised for 10K-100K files: fetches all known records in a
        single SQLite query instead of N individual lookups.
        Automatically chunks for SQLite variable-count limits.
        """
        if not paths:
            return []

        # Fetch all known records in batched queries
        # (SQLite variable limit is 32766 in modern versions, use 5000 to be safe)
        known: dict[str, tuple[int, str]] = {}
        _SQL_BATCH = 5000
        for i in range(0, len(paths), _SQL_BATCH):
            sub = paths[i : i + _SQL_BATCH]
            placeholders = ",".join("?" for _ in sub)
            cur = self._conn.execute(
                f"SELECT path, mtime_ns, content_hash FROM files WHERE path IN ({placeholders})",
                sub,
            )
            for row in cur:
                known[row[0]] = (row[1], row[2])

        changed: list[str] = []
        for p_str in paths:
            rec = known.get(p_str)
            if rec is None:
                changed.append(p_str)  # new file
                continue

            mtime_ns_stored, content_hash = rec
            if not content_hash:
                changed.append(p_str)  # invalidated
                continue

            try:
                current_mtime_ns = Path(p_str).stat().st_mtime_ns
            except OSError:
                changed.append(p_str)
                continue

            if current_mtime_ns == mtime_ns_stored:
                continue  # unchanged (fast path)

            # mtime changed — verify content hash
            current_hash = self._hash_file(p_str)
            if current_hash != content_hash:
                changed.append(p_str)

        return changed

    def mark_indexed(
        self,
        path: str | Path,
        chunk_count: int,
        last_indexed: str = "",
    ) -> None:
        """Record that *path* has been successfully indexed."""
        str_path = str(path)
        p = Path(path)
        try:
            mtime_ns = p.stat().st_mtime_ns
        except OSError:
            mtime_ns = 0
        content_hash = self._hash_file(p)
        self._conn.execute(
            """
            INSERT INTO files (path, mtime_ns, content_hash, chunk_count, last_indexed)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
                mtime_ns     = excluded.mtime_ns,
                content_hash = excluded.content_hash,
                chunk_count  = excluded.chunk_count,
                last_indexed = excluded.last_indexed
            """,
            (str_path, mtime_ns, content_hash, chunk_count, last_indexed),
        )
        self._conn.commit()

    def batch_mark_indexed(
        self,
        records: list[tuple[str, int]],
        last_indexed: str = "",
    ) -> None:
        """Batch-record that multiple files have been indexed.

        *records* is a list of ``(path, chunk_count)`` tuples.
        All rows are committed in a **single transaction** — orders of
        magnitude faster than N individual commits for large batches.
        """
        if not records:
            return
        rows: list[tuple[str, int, str, int, str]] = []
        for path_str, chunk_count in records:
            p = Path(path_str)
            try:
                mtime_ns = p.stat().st_mtime_ns
            except OSError:
                mtime_ns = 0
            content_hash = self._hash_file(p)
            rows.append((path_str, mtime_ns, content_hash, chunk_count, last_indexed))

        self._conn.executemany(
            """
            INSERT INTO files (path, mtime_ns, content_hash, chunk_count, last_indexed)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
                mtime_ns     = excluded.mtime_ns,
                content_hash = excluded.content_hash,
                chunk_count  = excluded.chunk_count,
                last_indexed = excluded.last_indexed
            """,
            rows,
        )
        self._conn.commit()

    def remove(self, path: str) -> None:
        """Remove a file from the manifest."""
        self._conn.execute("DELETE FROM files WHERE path = ?", (path,))
        self._conn.commit()

    def batch_remove(self, paths: list[str]) -> None:
        """Remove multiple files in a single transaction."""
        if not paths:
            return
        _SQL_BATCH = 5000
        for i in range(0, len(paths), _SQL_BATCH):
            sub = paths[i : i + _SQL_BATCH]
            placeholders = ",".join("?" for _ in sub)
            self._conn.execute(
                f"DELETE FROM files WHERE path IN ({placeholders})", sub
            )
        self._conn.commit()

    def invalidate(self, path: str | Path) -> None:
        """Mark *path* as needing re-indexing (set content_hash to NULL).

        This is called **before** deleting old chunks so that a crash
        between deletion and re-insertion is recovered on the next run
        (``is_changed()`` will return True for invalidated entries).
        """
        str_path = str(path)
        self._conn.execute(
            "UPDATE files SET content_hash = '' WHERE path = ?",
            (str_path,),
        )
        self._conn.commit()

    def batch_invalidate(self, paths: list[str]) -> None:
        """Invalidate multiple files in a single transaction."""
        if not paths:
            return
        _SQL_BATCH = 5000
        for i in range(0, len(paths), _SQL_BATCH):
            sub = paths[i : i + _SQL_BATCH]
            placeholders = ",".join("?" for _ in sub)
            self._conn.execute(
                f"UPDATE files SET content_hash = '' WHERE path IN ({placeholders})",
                sub,
            )
        self._conn.commit()

    def all_paths(self) -> set[str]:
        """Return the set of all indexed file paths."""
        cur = self._conn.execute("SELECT path FROM files")
        return {row[0] for row in cur.fetchall()}

    def get_record(self, path: str) -> FileRecord | None:
        """Return the manifest record for *path*, or None."""
        return self._get(path)

    def count(self) -> int:
        """Return total number of files in the manifest."""
        cur = self._conn.execute("SELECT COUNT(*) FROM files")
        return cur.fetchone()[0]

    def clear(self) -> None:
        """Remove all entries (used during full rebuild)."""
        self._conn.execute("DELETE FROM files")
        self._conn.commit()

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._conn.close()

    # ------------------------------------------------------------------
    # Migration helper
    # ------------------------------------------------------------------

    def populate_from_store_metadata(
        self,
        source_files: list[str],
        store_get_by_source,
    ) -> int:
        """One-time migration: populate manifest from existing ChromaDB metadata.

        For each *source_file* that exists on disk and is not already in the
        manifest, read the ``file_modified_at`` value stored in ChromaDB and
        insert a manifest row.

        Returns the number of files migrated.
        """
        migrated = 0
        existing = self.all_paths()
        for src in source_files:
            if src in existing:
                continue
            fp = Path(src)
            if not fp.exists():
                continue
            # Get chunk count from store
            items = store_get_by_source(src)
            chunk_count = len(items)
            stored_mtime = ""
            if items:
                stored_mtime = items[0].get("metadata", {}).get("file_modified_at", "")
            try:
                mtime_ns = fp.stat().st_mtime_ns
            except OSError:
                continue
            content_hash = self._hash_file(fp)
            self._conn.execute(
                """
                INSERT OR IGNORE INTO files (path, mtime_ns, content_hash, chunk_count, last_indexed)
                VALUES (?, ?, ?, ?, ?)
                """,
                (src, mtime_ns, content_hash, chunk_count, stored_mtime),
            )
            migrated += 1
        self._conn.commit()
        if migrated:
            logger.info("Migrated %d files to file manifest", migrated)
        return migrated

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get(self, path: str) -> FileRecord | None:
        cur = self._conn.execute(
            "SELECT path, mtime_ns, content_hash, chunk_count, last_indexed FROM files WHERE path = ?",
            (path,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return FileRecord(*row)

    @staticmethod
    def _hash_file(path: str | Path, block_size: int = 1 << 20) -> str:
        """Compute xxhash-64 of file contents in streaming fashion."""
        h = xxhash.xxh64()
        try:
            with open(path, "rb") as f:
                while True:
                    chunk = f.read(block_size)
                    if not chunk:
                        break
                    h.update(chunk)
        except OSError:
            return ""
        return h.hexdigest()
