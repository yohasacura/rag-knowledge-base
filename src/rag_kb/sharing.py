"""Export / Import RAG databases as shareable .rag files (ZIP archives)."""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from rag_kb import __version__
from rag_kb.config import RAGS_DIR, RagRegistry

logger = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.json"

# Safety limits for ZIP import to prevent zip-bombs and resource exhaustion
MAX_TOTAL_UNCOMPRESSED_SIZE: int = 10 * 1024 * 1024 * 1024  # 10 GB
MAX_INDIVIDUAL_FILE_SIZE: int = 2 * 1024 * 1024 * 1024  # 2 GB
MAX_MEMBER_COUNT: int = 100_000


def export_rag(
    registry: RagRegistry,
    rag_name: str,
    output_path: str,
) -> str:
    """Export a RAG database to a .rag file.

    Returns the absolute path to the created file.
    """
    entry = registry.get_rag(rag_name)
    db_path = Path(entry.db_path)

    if not db_path.exists():
        raise FileNotFoundError(f"ChromaDB directory not found: {db_path}")

    output = Path(output_path)
    if output.suffix.lower() != ".rag":
        output = output.with_suffix(".rag")
    output.parent.mkdir(parents=True, exist_ok=True)

    manifest = {
        "format_version": 1,
        "tool_version": __version__,
        "name": entry.name,
        "description": entry.description,
        "embedding_model": entry.embedding_model,
        "file_count": entry.file_count,
        "chunk_count": entry.chunk_count,
        "created_at": entry.created_at,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "source_folders": entry.source_folders,  # informational only
    }

    with ZipFile(str(output), "w", ZIP_DEFLATED) as zf:
        # Write manifest
        zf.writestr(MANIFEST_NAME, json.dumps(manifest, indent=2, ensure_ascii=False))

        # Add all files from the ChromaDB directory
        for file_path in sorted(db_path.rglob("*")):
            if file_path.is_file():
                arc_name = f"chroma_db/{file_path.relative_to(db_path)}"
                zf.write(str(file_path), arc_name)

    size_mb = output.stat().st_size / (1024 * 1024)
    logger.info(
        "Exported RAG '%s' → %s (%.1f MB, %d files, %d chunks)",
        rag_name,
        output,
        size_mb,
        entry.file_count,
        entry.chunk_count,
    )
    return str(output.resolve())


def import_rag(
    registry: RagRegistry,
    file_path: str,
    new_name: str | None = None,
    rags_dir: Path | None = None,
) -> str:
    """Import a RAG database from a .rag file.

    Returns the name of the imported RAG.
    """
    src = Path(file_path)
    if not src.exists():
        raise FileNotFoundError(f"File not found: {src}")

    rags_directory = rags_dir or RAGS_DIR

    with ZipFile(str(src), "r") as zf:
        # Read and validate manifest
        if MANIFEST_NAME not in zf.namelist():
            raise ValueError(f"Invalid .rag file: missing {MANIFEST_NAME}")

        # ── Safety checks ─────────────────────────────────────────────
        _validate_zip_safety(zf)

        manifest = json.loads(zf.read(MANIFEST_NAME))
        _validate_manifest(manifest)

        name = new_name or manifest.get("name", src.stem)

        # Ensure unique name
        existing = {r.name for r in registry.list_rags()}
        if name in existing:
            base = name
            counter = 2
            while name in existing:
                name = f"{base}-{counter}"
                counter += 1
            logger.info("Name '%s' already exists, using '%s' instead", base, name)

        # Extract chroma_db to a temp dir first, then move
        rag_dir = rags_directory / name / "chroma_db"
        rag_dir.mkdir(parents=True, exist_ok=True)

        chroma_members = [
            m for m in zf.namelist() if m.startswith("chroma_db/") and not m.endswith("/")
        ]
        if not chroma_members:
            raise ValueError("Invalid .rag file: no chroma_db/ contents found")

        for member in chroma_members:
            # Strip the chroma_db/ prefix and write to target
            relative = member[len("chroma_db/") :]
            if not relative:
                continue

            # ── Path traversal protection ─────────────────────────────
            _check_path_traversal(relative, rag_dir)
            target = rag_dir / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src_f, open(target, "wb") as dst_f:
                shutil.copyfileobj(src_f, dst_f)

    # Register in the registry
    registry.register_imported_rag(
        name=name,
        description=manifest.get("description", ""),
        db_path=str(rag_dir),
        embedding_model=manifest.get("embedding_model", "all-MiniLM-L6-v2"),
        imported_from=str(src.resolve()),
        file_count=manifest.get("file_count", 0),
        chunk_count=manifest.get("chunk_count", 0),
    )

    logger.info(
        "Imported RAG '%s' from %s (model: %s, %d files, %d chunks)",
        name,
        src.name,
        manifest.get("embedding_model"),
        manifest.get("file_count", 0),
        manifest.get("chunk_count", 0),
    )
    return name


def peek_rag_file(file_path: str) -> dict:
    """Read and return the manifest from a .rag file without importing."""
    src = Path(file_path)
    if not src.exists():
        raise FileNotFoundError(f"File not found: {src}")

    with ZipFile(str(src), "r") as zf:
        if MANIFEST_NAME not in zf.namelist():
            raise ValueError(f"Invalid .rag file: missing {MANIFEST_NAME}")
        _validate_zip_safety(zf)
        manifest = json.loads(zf.read(MANIFEST_NAME))

    manifest["file_size_mb"] = round(src.stat().st_size / (1024 * 1024), 2)
    return manifest


def _validate_zip_safety(zf: ZipFile) -> None:
    """Validate ZIP archive against resource exhaustion and malicious content.

    Raises ``ValueError`` if the archive exceeds safety limits.
    """
    members = zf.infolist()

    # 1. Member count limit
    if len(members) > MAX_MEMBER_COUNT:
        raise ValueError(
            f"ZIP archive contains {len(members)} members "
            f"(limit: {MAX_MEMBER_COUNT}). Possible zip-bomb."
        )

    # 2. Total uncompressed size + individual file size limits
    total_size = 0
    for info in members:
        if info.file_size > MAX_INDIVIDUAL_FILE_SIZE:
            raise ValueError(
                f"ZIP member '{info.filename}' has uncompressed size "
                f"{info.file_size / (1024**3):.1f} GB "
                f"(limit: {MAX_INDIVIDUAL_FILE_SIZE / (1024**3):.0f} GB)"
            )
        total_size += info.file_size

    if total_size > MAX_TOTAL_UNCOMPRESSED_SIZE:
        raise ValueError(
            f"ZIP total uncompressed size {total_size / (1024**3):.1f} GB "
            f"exceeds limit of {MAX_TOTAL_UNCOMPRESSED_SIZE / (1024**3):.0f} GB. "
            f"Possible zip-bomb."
        )

    # 3. Path traversal detection on all members
    for info in members:
        name = info.filename
        if name.startswith("/") or ".." in name.split("/"):
            raise ValueError(
                f"ZIP member '{name}' contains path traversal component — refusing to extract."
            )


def _check_path_traversal(relative: str, target_dir: Path) -> None:
    """Ensure resolved extraction path stays within *target_dir*.

    Raises ``ValueError`` on path traversal attempts.
    """
    resolved = (target_dir / relative).resolve()
    target_resolved = target_dir.resolve()
    if not str(resolved).startswith(str(target_resolved)):
        raise ValueError(
            f"Path traversal detected: member resolves to '{resolved}' "
            f"which is outside target directory '{target_resolved}'"
        )


def _validate_manifest(manifest: dict) -> None:
    """Basic validation of manifest fields."""
    required = ["name", "embedding_model"]
    for key in required:
        if key not in manifest:
            raise ValueError(f"Invalid manifest: missing required field '{key}'")
    fmt_version = manifest.get("format_version", 0)
    if fmt_version > 1:
        logger.warning(
            "This .rag file uses format version %d; this tool supports version 1. "
            "Import may not work correctly.",
            fmt_version,
        )
