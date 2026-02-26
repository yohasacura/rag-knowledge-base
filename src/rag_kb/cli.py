"""Command-line interface for rag-kb."""

from __future__ import annotations

import argparse
import logging
import sys

from rich.console import Console
from rich.table import Table
from rich.logging import RichHandler

console = Console()


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
    )
    # Quieten noisy libraries
    for lib in (
        "chromadb", "sentence_transformers", "httpx", "httpcore",
        "urllib3", "watchdog", "huggingface_hub", "transformers",
        "safetensors",
    ):
        logging.getLogger(lib).setLevel(logging.WARNING)


# -----------------------------------------------------------------------
# Sub-commands
# -----------------------------------------------------------------------

def cmd_create(args: argparse.Namespace) -> None:
    from rag_kb.config import AppSettings, RagRegistry

    settings = AppSettings.load()
    registry = RagRegistry()
    entry = registry.create_rag(
        name=args.name,
        description=args.description or "",
        folders=args.folders or [],
        embedding_model=args.model or settings.embedding_model,
    )
    console.print(f"[green]✔[/green] Created RAG [bold]{entry.name}[/bold]")
    if entry.source_folders:
        console.print(f"  Source folders: {', '.join(entry.source_folders)}")
    console.print(f"  Database: {entry.db_path}")


def cmd_list(args: argparse.Namespace) -> None:
    from rag_kb.config import RagRegistry

    registry = RagRegistry()
    rags = registry.list_rags()
    active = registry.get_active_name()

    if not rags:
        console.print("[yellow]No RAG databases found.[/yellow] Use [bold]rag-kb create[/bold] to create one.")
        return

    table = Table(title="RAG Databases")
    table.add_column("", width=3)
    table.add_column("Name", style="bold")
    table.add_column("Type")
    table.add_column("Model")
    table.add_column("Files", justify="right")
    table.add_column("Chunks", justify="right")
    table.add_column("Status")
    table.add_column("Description")

    for r in rags:
        marker = "→" if r.name == active else ""
        rtype = "📥 Imported" if r.is_imported else "📁 Local"
        status = "🔒 Detached" if r.detached else ""
        table.add_row(
            marker, r.name, rtype, r.embedding_model,
            str(r.file_count), str(r.chunk_count), status, r.description,
        )
    console.print(table)


def cmd_use(args: argparse.Namespace) -> None:
    from rag_kb.config import RagRegistry

    registry = RagRegistry()
    registry.set_active(args.name)
    console.print(f"[green]✔[/green] Active RAG → [bold]{args.name}[/bold]")


def cmd_delete(args: argparse.Namespace) -> None:
    from rag_kb.config import RagRegistry

    registry = RagRegistry()
    if not args.yes:
        if not console.input(f"Delete RAG '{args.name}'? [y/N] ").strip().lower().startswith("y"):
            console.print("Cancelled.")
            return
    registry.delete_rag(args.name)
    console.print(f"[green]✔[/green] Deleted RAG [bold]{args.name}[/bold]")


def cmd_detach(args: argparse.Namespace) -> None:
    from rag_kb.config import RagRegistry

    registry = RagRegistry()
    entry = registry.get_rag(args.name)
    entry.detached = True
    registry.update_rag(entry)
    console.print(f"[green]✔[/green] Detached RAG [bold]{args.name}[/bold]")
    console.print("  Indexing and file watching are now disabled.")
    console.print("  You can safely delete source files — the knowledge base is preserved.")
    console.print(f"  To re-attach later: [bold]rag-kb attach {args.name}[/bold]")


def cmd_attach(args: argparse.Namespace) -> None:
    from rag_kb.config import RagRegistry

    registry = RagRegistry()
    entry = registry.get_rag(args.name)
    entry.detached = False
    registry.update_rag(entry)
    console.print(f"[green]✔[/green] Re-attached RAG [bold]{args.name}[/bold]")
    console.print("  Indexing and file watching are re-enabled.")


def cmd_index(args: argparse.Namespace) -> None:
    from rag_kb.config import AppSettings, RagRegistry
    from rag_kb.indexer import Indexer

    settings = AppSettings.load()
    registry = RagRegistry()

    # Override workers if specified
    if hasattr(args, "workers") and args.workers:
        settings.indexing_workers = args.workers

    rag_name = args.rag or registry.get_active_name()
    if not rag_name:
        console.print("[red]No RAG specified and no active RAG set.[/red]")
        sys.exit(1)

    entry = registry.get_rag(rag_name)
    if entry.detached:
        console.print(f"[red]RAG '{rag_name}' is detached (read-only).[/red] Use [bold]rag-kb attach {rag_name}[/bold] to re-enable indexing.")
        sys.exit(1)
    if entry.is_imported and not entry.source_folders:
        console.print("[red]Cannot index an imported RAG with no source folders.[/red]")
        sys.exit(1)

    console.print(f"Indexing RAG [bold]{rag_name}[/bold] …")

    def _progress(state):
        if state.current_file:
            console.print(f"  [{state.processed_files}/{state.total_files}] {state.current_file}", highlight=False)

    indexer = Indexer(entry, registry, settings, on_progress=_progress)
    state = indexer.index(full=args.full)

    console.print(
        f"[green]✔[/green] Done: {state.processed_files} files, "
        f"{state.total_chunks} chunks, {state.duration_seconds}s"
    )
    if state.errors:
        console.print(f"[yellow]⚠ {len(state.errors)} error(s):[/yellow]")
        for e in state.errors:
            console.print(f"  - {e}")


def cmd_export(args: argparse.Namespace) -> None:
    from rag_kb.config import RagRegistry
    from rag_kb.sharing import export_rag

    registry = RagRegistry()
    result = export_rag(registry, args.name, args.output)
    console.print(f"[green]✔[/green] Exported [bold]{args.name}[/bold] → {result}")


def cmd_import(args: argparse.Namespace) -> None:
    from rag_kb.config import RagRegistry
    from rag_kb.sharing import import_rag, peek_rag_file

    registry = RagRegistry()

    # Show info about the file
    info = peek_rag_file(args.file)
    console.print(f"  Name: {info.get('name')}")
    console.print(f"  Model: {info.get('embedding_model')}")
    console.print(f"  Files: {info.get('file_count', '?')}, Chunks: {info.get('chunk_count', '?')}")
    console.print(f"  Size: {info.get('file_size_mb', '?')} MB")

    imported_name = import_rag(registry, args.file, new_name=args.name)
    console.print(f"[green]✔[/green] Imported as [bold]{imported_name}[/bold]")


def cmd_serve(args: argparse.Namespace) -> None:
    from rag_kb.mcp_server import run_http, run_stdio

    if args.http:
        console.print(f"Starting MCP server (HTTP) on {args.host}:{args.port} …")
        run_http(host=args.host, port=args.port)
    else:
        # stdio mode — suppress console output that would interfere with protocol
        logging.getLogger().handlers.clear()
        run_stdio()


def cmd_ui(args: argparse.Namespace) -> None:
    from rag_kb.ui import launch_ui

    console.print("Launching desktop UI …")
    launch_ui()


def cmd_config(args: argparse.Namespace) -> None:
    from rag_kb.config import AppSettings, CONFIG_PATH, DATA_DIR

    settings = AppSettings.load()
    console.print(f"[bold]Data directory:[/bold] {DATA_DIR}")
    console.print(f"[bold]Config file:[/bold] {CONFIG_PATH}")
    console.print()

    import yaml
    console.print(yaml.safe_dump(settings.model_dump(), default_flow_style=False))


def cmd_download_models(args: argparse.Namespace) -> None:
    from pathlib import Path
    from rag_kb.models import download_models, DEFAULT_MODELS, BUNDLED_MODELS_DIR

    output = Path(args.output) if args.output else BUNDLED_MODELS_DIR
    console.print(f"Downloading models to [bold]{output}[/bold] …")

    if args.model:
        models = [{"name": args.model, "type": "embedding", "description": "custom model"}]
    else:
        models = DEFAULT_MODELS
        console.print(f"  Models: {', '.join(m['name'] for m in models)}")

    saved = download_models(models=models, output_dir=output)
    for p in saved:
        console.print(f"  [green]✔[/green] {p.name} ({_dir_size_mb(p):.1f} MB)")
    console.print(f"[green]✔[/green] Done — {len(saved)} model(s) saved.")


def _dir_size_mb(path: "Path") -> float:
    """Return total size of a directory in megabytes."""
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / (1024 * 1024)


# -----------------------------------------------------------------------
# Argument parser
# -----------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-kb",
        description="RAG Knowledge Base — offline RAG storage and MCP server",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")

    sub = parser.add_subparsers(dest="command", help="Available commands")

    # create
    p = sub.add_parser("create", help="Create a new RAG database")
    p.add_argument("name", help="Name for the RAG")
    p.add_argument("--folders", nargs="+", metavar="PATH", help="Source folders to index")
    p.add_argument("--description", "-d", default="", help="Optional description")
    p.add_argument("--model", default=None, help="Embedding model name")

    # list
    sub.add_parser("list", help="List all RAG databases")

    # use
    p = sub.add_parser("use", help="Set the active RAG database")
    p.add_argument("name", help="RAG name to activate")

    # delete
    p = sub.add_parser("delete", help="Delete a RAG database")
    p.add_argument("name", help="RAG name to delete")
    p.add_argument("-y", "--yes", action="store_true", help="Skip confirmation")

    # detach
    p = sub.add_parser("detach", help="Detach a RAG from source files (read-only mode)")
    p.add_argument("name", help="RAG name to detach")

    # attach
    p = sub.add_parser("attach", help="Re-attach a detached RAG to its source files")
    p.add_argument("name", help="RAG name to re-attach")

    # index
    p = sub.add_parser("index", help="Index documents into a RAG")
    p.add_argument("--rag", default=None, help="RAG name (defaults to active)")
    p.add_argument("--full", action="store_true", help="Full rebuild (clear and re-index)")
    p.add_argument("--workers", type=int, default=None, help="Number of parallel parsing workers (default: from config)")

    # export
    p = sub.add_parser("export", help="Export a RAG to a .rag file")
    p.add_argument("name", help="RAG name to export")
    p.add_argument("--output", "-o", required=True, help="Output file path")

    # import
    p = sub.add_parser("import", help="Import a RAG from a .rag file")
    p.add_argument("file", help="Path to the .rag file")
    p.add_argument("--name", default=None, help="Custom name for the imported RAG")

    # serve
    p = sub.add_parser("serve", help="Start MCP server")
    p.add_argument("--http", action="store_true", help="Use HTTP transport instead of stdio")
    p.add_argument("--host", default="127.0.0.1", help="Host (default: 127.0.0.1)")
    p.add_argument("--port", type=int, default=8080, help="Port (default: 8080)")

    # ui
    sub.add_parser("ui", help="Launch desktop UI")

    # config
    sub.add_parser("config", help="Show current configuration")

    # download-models
    p = sub.add_parser("download-models", help="Pre-download ML models for offline/bundled use")
    p.add_argument("--output", "-o", default=None, help="Output directory (default: models/ in project root)")
    p.add_argument("--model", default=None, help="Specific model name to download (default: all default models)")

    return parser


_CMD_MAP = {
    "create": cmd_create,
    "list": cmd_list,
    "use": cmd_use,
    "delete": cmd_delete,
    "detach": cmd_detach,
    "attach": cmd_attach,
    "index": cmd_index,
    "export": cmd_export,
    "import": cmd_import,
    "serve": cmd_serve,
    "ui": cmd_ui,
    "config": cmd_config,
    "download-models": cmd_download_models,
}


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "serve" and not getattr(args, "http", False):
        # stdio mode: minimal logging
        _setup_logging(verbose=False)
    else:
        _setup_logging(verbose=getattr(args, "verbose", False))

    if not args.command:
        parser.print_help()
        sys.exit(0)

    handler = _CMD_MAP.get(args.command)
    if handler:
        try:
            handler(args)
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted.[/yellow]")
            sys.exit(130)
        except Exception as exc:
            console.print(f"[red]Error:[/red] {exc}")
            if getattr(args, "verbose", False):
                console.print_exception()
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
