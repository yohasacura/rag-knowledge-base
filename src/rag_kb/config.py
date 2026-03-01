"""Configuration management and RAG registry."""

from __future__ import annotations

import gc
import json
import logging
import os
import platform
import shutil
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


def _close_chroma_for_path(db_path: str) -> None:
    """Close any in-process ChromaDB client pointing at *db_path*.

    Uses the VectorStoreRegistry for clean lifecycle management.
    Also closes FileManifest SQLite connections under the given path.
    """
    try:
        from rag_kb.vector_store import get_store_registry

        registry = get_store_registry()
        registry.close_for_path(db_path)
    except Exception:
        pass

    try:
        from rag_kb.file_manifest import FileManifest

        norm = os.path.normcase(os.path.normpath(db_path))
        # Best-effort: walk GC only for FileManifest (lightweight, few instances)
        for obj in gc.get_objects():
            try:
                if isinstance(obj, FileManifest):
                    manifest_path = os.path.normcase(os.path.normpath(getattr(obj, "_db_path", "")))
                    if manifest_path.startswith(norm):
                        obj.close()
            except Exception:
                pass
    except Exception:
        pass


def _rmtree_with_retries(path: Path, retries: int = 8, delay: float = 0.3) -> None:
    """Remove a directory tree with retries for transient Windows locks.

    Before retrying, the helper attempts to close any in-process ChromaDB
    clients and FileManifest SQLite connections that may still hold files open.
    """
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            shutil.rmtree(path)
            return
        except (PermissionError, OSError) as exc:
            last_exc = exc
            # Try to close ChromaDB + FileManifest connections holding locks.
            # Pass *both* the chroma_db dir (for ChromaDB/VectorStore) and
            # the parent rag dir (so FileManifest paths that live under it
            # are also matched).
            chroma_dir = path / "chroma_db"
            if chroma_dir.exists():
                _close_chroma_for_path(str(chroma_dir))
            # Also sweep with the parent path so FileManifest instances
            # whose db_path is <rag_dir>/chroma_db/file_manifest.db match.
            _close_chroma_for_path(str(path))
            gc.collect()
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))  # progressive back-off
                continue
            raise
    if last_exc:
        raise last_exc


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def _default_data_dir() -> Path:
    """Return the platform-appropriate data directory."""
    system = platform.system()
    if system == "Windows":
        base = Path.home() / "AppData" / "Local"
    elif system == "Darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(
            __import__("os").environ.get("XDG_DATA_HOME", str(Path.home() / ".local" / "share"))
        )
    return base / "rag-kb"


DATA_DIR: Path = _default_data_dir()
CONFIG_PATH: Path = DATA_DIR / "config.yaml"
REGISTRY_PATH: Path = DATA_DIR / "registry.json"
RAGS_DIR: Path = DATA_DIR / "rags"

# ---------------------------------------------------------------------------
# Global settings (loaded from config.yaml)
# ---------------------------------------------------------------------------

# Import the definitive extension list built from all registered parsers
# so that file discovery automatically covers every format the app can handle.
from rag_kb.parsers.registry import SUPPORTED_EXTENSIONS as _ALL_PARSER_EXTENSIONS  # noqa: E402

_DEFAULT_EXTENSIONS: list[str] = list(_ALL_PARSER_EXTENSIONS)


class AppSettings(BaseModel):
    """Application-wide settings persisted in config.yaml."""

    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    chunk_size: int = 1024
    chunk_overlap: int = 128
    supported_extensions: list[str] = Field(default_factory=lambda: list(_DEFAULT_EXTENSIONS))
    host: str = "127.0.0.1"
    port: int = 8080

    # Search quality settings
    reranking_enabled: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    hybrid_search_enabled: bool = True
    hybrid_search_alpha: float = 0.7  # 0 = all BM25, 1 = all vector
    min_score_threshold: float = 0.15
    mmr_enabled: bool = True
    mmr_lambda: float = 0.7  # 0 = max diversity, 1 = max relevance

    # Indexing performance settings
    indexing_workers: int = 14  # parallel parsing workers (0 = auto)
    embedding_batch_size: int = 512  # texts per encode() call

    # ChromaDB HNSW tuning
    hnsw_ef_construction: int = 256
    hnsw_m: int = 48

    # Model management
    trusted_models: list[str] = Field(
        default_factory=list,
        description="Models the user has consented to run with trust_remote_code=True",
    )

    # API keys — prefer environment variables over config.yaml values.
    # Set RAG_KB_OPENAI_API_KEY / RAG_KB_VOYAGE_API_KEY in your environment.
    openai_api_key: str = ""
    voyage_api_key: str = ""

    # ------------------------------------------------------------------

    def resolve_openai_api_key(self) -> str:
        """Return the OpenAI API key, preferring env var over stored value."""
        return os.environ.get("RAG_KB_OPENAI_API_KEY", "") or self.openai_api_key

    def resolve_voyage_api_key(self) -> str:
        """Return the Voyage API key, preferring env var over stored value."""
        return os.environ.get("RAG_KB_VOYAGE_API_KEY", "") or self.voyage_api_key

    # ------------------------------------------------------------------

    def save(self, path: Path | None = None) -> None:
        path = path or CONFIG_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        # Don't persist supported_extensions — always derived from parsers
        data = self.model_dump()
        data.pop("supported_extensions", None)
        # Don't persist API keys if they come from environment variables
        if os.environ.get("RAG_KB_OPENAI_API_KEY"):
            data.pop("openai_api_key", None)
        if os.environ.get("RAG_KB_VOYAGE_API_KEY"):
            data.pop("voyage_api_key", None)
        # Atomic write: write to temp file then os.replace()
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp", prefix=".config_")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                yaml.safe_dump(data, fh, default_flow_style=False, sort_keys=False)
            Path(tmp).replace(path)
        except BaseException:
            try:
                Path(tmp).unlink(missing_ok=True)
            except OSError:
                pass
            raise

    @classmethod
    def load(cls, path: Path | None = None) -> AppSettings:
        path = path or CONFIG_PATH
        if path.exists():
            with open(path, encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
            # Remove stale saved extensions — always use parser registry
            data.pop("supported_extensions", None)
            return cls(**data)
        settings = cls()
        settings.save(path)
        return settings


# ---------------------------------------------------------------------------
# RAG entry (one per knowledge-base)
# ---------------------------------------------------------------------------


class RagEntry(BaseModel):
    """Metadata for a single RAG database."""

    name: str
    description: str = ""
    db_path: str = ""  # absolute path to chroma_db dir
    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    embedding_dimension: int = 0  # 0 = auto-detect from model registry
    source_folders: list[str] = Field(default_factory=list)
    created_at: str = ""
    is_imported: bool = False
    imported_from: str = ""
    file_count: int = 0
    chunk_count: int = 0
    detached: bool = False  # True → read-only, indexing/watcher disabled


# ---------------------------------------------------------------------------
# RAG Registry  (registry.json)
# ---------------------------------------------------------------------------


class RagRegistry:
    """Manages the collection of RAG databases."""

    def __init__(self, registry_path: Path | None = None, rags_dir: Path | None = None):
        self._path = registry_path or REGISTRY_PATH
        self._rags_dir = rags_dir or RAGS_DIR
        self._rags: dict[str, RagEntry] = {}
        self._active: str | None = None
        self._file_lock = threading.RLock()
        self._load()

    # -- persistence --------------------------------------------------------

    def _load(self) -> None:
        if self._path.exists():
            with open(self._path, encoding="utf-8") as fh:
                data: dict[str, Any] = json.load(fh)
            self._active = data.get("active")
            for name, entry_data in data.get("rags", {}).items():
                self._rags[name] = RagEntry(**entry_data)
            self._pending_cleanups: list[str] = data.get("pending_cleanups", [])
        else:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._pending_cleanups = []
            self._save()
        # Run deferred cleanup of previously locked directories
        self._run_pending_cleanups()

    def _save(self) -> None:
        with self._file_lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            data: dict[str, Any] = {
                "active": self._active,
                "rags": {n: e.model_dump() for n, e in self._rags.items()},
            }
            if self._pending_cleanups:
                data["pending_cleanups"] = self._pending_cleanups
            # Atomic write: write to temp file then os.replace() so that
            # a crash mid-write cannot corrupt the registry.
            fd, tmp = tempfile.mkstemp(
                dir=str(self._path.parent), suffix=".tmp", prefix=".registry_"
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    json.dump(data, fh, indent=2, ensure_ascii=False)
                Path(tmp).replace(self._path)
            except BaseException:
                # Clean up the temp file on any failure
                try:
                    Path(tmp).unlink(missing_ok=True)
                except OSError:
                    pass
                raise

    def _add_pending_cleanup(self, dir_path: str) -> None:
        if dir_path not in self._pending_cleanups:
            self._pending_cleanups.append(dir_path)
            self._save()

    def _run_pending_cleanups(self) -> None:
        """Attempt to delete directories that could not be removed earlier."""
        if not self._pending_cleanups:
            return
        remaining: list[str] = []
        for dir_path in self._pending_cleanups:
            p = Path(dir_path)
            if not p.exists():
                logger.debug("Pending cleanup path already gone: %s", p)
                continue
            try:
                # Close any lingering handles before retrying deletion
                _close_chroma_for_path(str(p))
                chroma_dir = p / "chroma_db"
                if chroma_dir.exists():
                    _close_chroma_for_path(str(chroma_dir))
                gc.collect()
                _rmtree_with_retries(p)
                logger.info("Deferred cleanup succeeded: %s", p)
            except (PermissionError, OSError) as exc:
                logger.debug("Deferred cleanup still failing for %s: %s", p, exc)
                remaining.append(dir_path)
        self._pending_cleanups = remaining
        self._save()

    # -- CRUD ---------------------------------------------------------------

    def create_rag(
        self,
        name: str,
        description: str = "",
        folders: list[str] | None = None,
        embedding_model: str | None = None,
    ) -> RagEntry:
        with self._file_lock:
            if name in self._rags:
                raise ValueError(f"RAG '{name}' already exists")
            _validate_rag_name(name)

            rag_dir = self._rags_dir / name / "chroma_db"
            rag_dir.mkdir(parents=True, exist_ok=True)

            settings = AppSettings.load()

            entry = RagEntry(
                name=name,
                description=description,
                db_path=str(rag_dir),
                embedding_model=embedding_model or settings.embedding_model,
                source_folders=[str(Path(f).resolve()) for f in (folders or [])],
                created_at=datetime.now(timezone.utc).isoformat(),
            )
            self._rags[name] = entry
            if self._active is None:
                self._active = name
            self._save()
            logger.info("Created RAG '%s' → %s", name, rag_dir)
            return entry

    def list_rags(self) -> list[RagEntry]:
        with self._file_lock:
            return list(self._rags.values())

    def get_rag(self, name: str) -> RagEntry:
        with self._file_lock:
            if name not in self._rags:
                raise KeyError(f"RAG '{name}' does not exist")
            return self._rags[name]

    def delete_rag(self, name: str) -> bool:
        """Delete a RAG from the registry and attempt to remove its data.

        Returns ``True`` if all data was removed immediately, ``False`` if
        file cleanup was deferred (e.g. another process holds the files open).
        """
        with self._file_lock:
            if name not in self._rags:
                raise KeyError(f"RAG '{name}' does not exist")
            rag_dir = self._rags_dir / name

            # Phase 1: Remove the registry entry and persist immediately
            # so the RAG disappears from the UI even if file deletion fails.
            del self._rags[name]
            if self._active == name:
                self._active = next(iter(self._rags), None)
            self._save()
            logger.info("Deleted RAG '%s' from registry", name)

        # Phase 2: Try to delete the data directory.  If the files are
        # locked (common on Windows with ChromaDB memory-mapped files),
        # record the path for deferred cleanup on next startup / exit.
        # Done OUTSIDE the lock — filesystem ops can be slow.
        if rag_dir.exists():
            try:
                _rmtree_with_retries(rag_dir)
                logger.info("Removed data directory for '%s'", name)
            except (PermissionError, OSError) as exc:
                logger.warning(
                    "Could not remove data directory %s now (%s); scheduled for deferred cleanup.",
                    rag_dir,
                    exc,
                )
                self._add_pending_cleanup(str(rag_dir))
                return False
        return True

    def update_rag(self, entry: RagEntry) -> None:
        with self._file_lock:
            self._rags[entry.name] = entry
            self._save()

    # -- active RAG ---------------------------------------------------------

    def set_active(self, name: str) -> None:
        with self._file_lock:
            if name not in self._rags:
                raise KeyError(f"RAG '{name}' does not exist")
            self._active = name
            self._save()

    def get_active_name(self) -> str | None:
        with self._file_lock:
            return self._active

    def get_active(self) -> RagEntry | None:
        with self._file_lock:
            if self._active and self._active in self._rags:
                return self._rags[self._active]
            return None

    # -- import helper ------------------------------------------------------

    def register_imported_rag(
        self,
        name: str,
        description: str,
        db_path: str,
        embedding_model: str,
        imported_from: str,
        file_count: int = 0,
        chunk_count: int = 0,
    ) -> RagEntry:
        with self._file_lock:
            if name in self._rags:
                raise ValueError(f"RAG '{name}' already exists — use a different name")
            _validate_rag_name(name)

            entry = RagEntry(
                name=name,
                description=description,
                db_path=db_path,
                embedding_model=embedding_model,
                source_folders=[],
                created_at=datetime.now(timezone.utc).isoformat(),
                is_imported=True,
                imported_from=imported_from,
                file_count=file_count,
                chunk_count=chunk_count,
            )
            self._rags[name] = entry
            if self._active is None:
                self._active = name
            self._save()
            logger.info("Registered imported RAG '%s' from %s", name, imported_from)
            return entry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate_rag_name(name: str) -> None:
    if not name or not name.strip():
        raise ValueError("RAG name cannot be empty")
    forbidden = set('<>:"/\\|?*')
    if any(c in forbidden for c in name):
        raise ValueError(f"RAG name contains forbidden characters: {forbidden}")
    if len(name) > 128:
        raise ValueError("RAG name must be ≤128 characters")


def safe_display_path(
    full_path: str,
    source_folders: list[str] | None = None,
) -> str:
    """Convert an absolute file path to a safe display string.

    Returns a path relative to the matching source folder so that the user's
    directory structure is not exposed.  Falls back to just the filename if no
    source folder matches.
    """
    fp = Path(full_path)
    for folder in source_folders or []:
        try:
            rel = fp.relative_to(folder)
            return str(rel)
        except ValueError:
            continue
    # No matching source folder – just the file name
    return fp.name
