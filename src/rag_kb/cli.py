"""Command-line interface for rag-kb.

Thin adapter — all business logic is accessed via the daemon process
through ``DaemonClient``.  The daemon is auto-started on first use.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

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
        "chromadb",
        "sentence_transformers",
        "httpx",
        "httpcore",
        "urllib3",
        "watchdog",
        "huggingface_hub",
        "transformers",
        "safetensors",
    ):
        logging.getLogger(lib).setLevel(logging.WARNING)


def _client():
    """Lazy-create a DaemonClient, ensuring the daemon is running."""
    from rag_kb.daemon_client import DaemonClient

    client = DaemonClient()
    client.ensure_daemon()
    client.connect()
    return client


# -----------------------------------------------------------------------
# Sub-commands
# -----------------------------------------------------------------------


def cmd_create(args: argparse.Namespace) -> None:
    client = _client()
    result = client.create_rag(
        name=args.name,
        folders=args.folders or [],
        description=args.description or "",
        embedding_model=args.model,
    )
    console.print(f"[green]✔[/green] Created RAG [bold]{result['name']}[/bold]")
    console.print(f"  Database: {result.get('db_path', '')}")


def cmd_list(args: argparse.Namespace) -> None:
    client = _client()
    rags = client.list_rags()
    active_name = None
    for r in rags:
        if r.get("is_active"):
            active_name = r["name"]
            break

    if not rags:
        console.print(
            "[yellow]No RAG databases found.[/yellow] Use [bold]rag-kb create[/bold] to create one."
        )
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
        marker = "→" if r["name"] == active_name else ""
        rtype = "📥 Imported" if r.get("is_imported") else "📁 Local"
        status = "🔒 Detached" if r.get("detached") else ""
        table.add_row(
            marker,
            r["name"],
            rtype,
            r.get("embedding_model", ""),
            str(r.get("file_count", 0)),
            str(r.get("chunk_count", 0)),
            status,
            r.get("description", ""),
        )
    console.print(table)


def cmd_use(args: argparse.Namespace) -> None:
    client = _client()
    client.switch_rag(args.name)
    console.print(f"[green]✔[/green] Active RAG → [bold]{args.name}[/bold]")


def cmd_delete(args: argparse.Namespace) -> None:
    client = _client()
    if not args.yes and not console.input(
        f"Delete RAG '{args.name}'? [y/N] "
    ).strip().lower().startswith("y"):
        console.print("Cancelled.")
        return
    client.delete_rag(args.name)
    console.print(f"[green]✔[/green] Deleted RAG [bold]{args.name}[/bold]")


def cmd_detach(args: argparse.Namespace) -> None:
    client = _client()
    client.detach_rag(args.name)
    console.print(f"[green]✔[/green] Detached RAG [bold]{args.name}[/bold]")
    console.print("  Indexing and file watching are now disabled.")
    console.print("  You can safely delete source files — the knowledge base is preserved.")
    console.print(f"  To re-attach later: [bold]rag-kb attach {args.name}[/bold]")


def cmd_attach(args: argparse.Namespace) -> None:
    client = _client()
    client.attach_rag(args.name)
    console.print(f"[green]✔[/green] Re-attached RAG [bold]{args.name}[/bold]")
    console.print("  Indexing and file watching are re-enabled.")


def cmd_index(args: argparse.Namespace) -> None:
    client = _client()
    rag_name = args.rag or client.get_active_name()
    if not rag_name:
        console.print("[red]No RAG specified and no active RAG set.[/red]")
        sys.exit(1)

    console.print(f"Indexing RAG [bold]{rag_name}[/bold] …")

    def _progress(p):
        msg = p.get("message", "")
        current = p.get("current", 0)
        total = p.get("total", 0)
        if msg:
            console.print(f"  [{current}/{total}] {msg}", highlight=False)

    result = client.index(
        rag_name=rag_name,
        full=args.full,
        workers=args.workers,
        on_progress=_progress,
    )

    status = result.get("status", "unknown")
    if status == "cancelled":
        console.print(
            f"[yellow]⚠ Indexing cancelled after "
            f"{result.get('duration_seconds', 0)}s[/yellow]\n"
            f"  Partially indexed files will be re-indexed on the next run."
        )
    else:
        console.print(
            f"[green]✔[/green] Done: {result.get('processed_files', 0)} files, "
            f"{result.get('total_chunks', 0)} chunks, {result.get('duration_seconds', 0)}s"
        )
    errors = result.get("errors", [])
    if errors:
        console.print(f"[yellow]⚠ {len(errors)} error(s):[/yellow]")
        for e in errors:
            console.print(f"  - {e}")


def cmd_cancel_index(args: argparse.Namespace) -> None:
    """Cancel a running indexing operation."""
    client = _client()
    result = client.cancel_indexing()
    if result.get("cancelled"):
        console.print(
            "[green]✔[/green] Cancellation requested — indexing will stop at the next checkpoint."
        )
    else:
        console.print("[yellow]No indexing operation is currently running.[/yellow]")


def cmd_verify(args: argparse.Namespace) -> None:
    """Verify index consistency between manifest and vector store."""
    client = _client()
    result = client.verify_index_consistency(rag_name=args.rag)

    if result.get("ok"):
        console.print("[green]✔[/green] Index is consistent.")
        return

    console.print("[yellow]⚠ Index inconsistencies detected:[/yellow]")

    inv = result.get("invalidated_files", [])
    if inv:
        console.print(f"\n  [bold]Invalidated files[/bold] ({len(inv)}):")
        console.print("  Files marked for re-indexing (empty content hash).")
        for f in inv[:20]:
            console.print(f"    - {f}")
        if len(inv) > 20:
            console.print(f"    … and {len(inv) - 20} more")

    orphan_store = result.get("orphan_store_files", [])
    if orphan_store:
        console.print(f"\n  [bold]Orphan store files[/bold] ({len(orphan_store)}):")
        console.print("  In vector store but not in manifest.")
        for f in orphan_store[:20]:
            console.print(f"    - {f}")
        if len(orphan_store) > 20:
            console.print(f"    … and {len(orphan_store) - 20} more")

    orphan_manifest = result.get("orphan_manifest_files", [])
    if orphan_manifest:
        console.print(f"\n  [bold]Orphan manifest files[/bold] ({len(orphan_manifest)}):")
        console.print("  In manifest but not in vector store.")
        for f in orphan_manifest[:20]:
            console.print(f"    - {f}")
        if len(orphan_manifest) > 20:
            console.print(f"    … and {len(orphan_manifest) - 20} more")

    if result.get("incomplete_indexing"):
        console.print(
            "\n  [bold]Incomplete indexing[/bold]: lock file detected from a previous crash."
        )

    console.print(
        "\n  Run [bold]rag-kb index[/bold] or [bold]rag-kb index --full[/bold] to repair."
    )


def cmd_search(args: argparse.Namespace) -> None:
    client = _client()
    results = client.search(
        query=args.query,
        top_k=args.top,
        rag_name=args.rag,
    )

    if not results:
        console.print("[yellow]No results found.[/yellow]")
        return

    for i, r in enumerate(results, 1):
        console.print(f"\n[bold cyan]─── Result {i} ───[/bold cyan]")
        console.print(
            f"[bold]Source:[/bold] {r['source_file']}  [dim](chunk {r['chunk_index']}, score {r['score']})[/dim]"
        )
        console.print(r["text"])


def cmd_status(args: argparse.Namespace) -> None:
    client = _client()
    status = client.get_index_status(rag_name=args.rag)

    if not status.get("active_rag"):
        console.print("[yellow]No active RAG.[/yellow]")
        return

    console.print(f"[bold]Active RAG:[/bold]   {status['active_rag']}")
    console.print(f"[bold]Files:[/bold]        {status.get('total_files', 0)}")
    console.print(f"[bold]Chunks:[/bold]       {status.get('total_chunks', 0)}")
    console.print(
        f"[bold]Watcher:[/bold]      {'running' if status.get('watcher_running') else 'stopped'}"
    )
    if status.get("last_indexed"):
        console.print(f"[bold]Last indexed:[/bold] {status['last_indexed']}")
    errors = status.get("errors", [])
    if errors:
        console.print(f"[yellow]Errors: {len(errors)}[/yellow]")
        for e in errors:
            console.print(f"  - {e}")


def cmd_files(args: argparse.Namespace) -> None:
    client = _client()
    page = getattr(args, "page", 1)
    limit = getattr(args, "limit", 50)
    file_filter = getattr(args, "filter", "") or ""
    offset = (page - 1) * limit

    result = client.list_indexed_files(
        rag_name=args.rag,
        offset=offset,
        limit=limit,
        filter=file_filter,
    )

    files = result.get("files", [])
    total = result.get("total", len(files))

    if not files:
        if file_filter:
            console.print(f"[yellow]No files matching '{file_filter}'.[/yellow]")
        else:
            console.print("[yellow]No files indexed.[/yellow]")
        return

    title = "Indexed Files"
    if file_filter:
        title += f" (filter: '{file_filter}')"

    table = Table(title=title)
    table.add_column("#", justify="right", style="dim")
    table.add_column("File", style="bold")
    table.add_column("Chunks", justify="right")

    for i, f in enumerate(files, start=offset + 1):
        table.add_row(str(i), f["file"], str(f["chunk_count"]))
    console.print(table)

    # Pagination footer
    last = min(offset + limit, total)
    console.print(
        f"[dim]Showing {offset + 1}–{last} of {total} file(s)"
        + (f"  •  Page {page}/{(total + limit - 1) // limit}" if total > limit else "")
        + "[/dim]"
    )


def cmd_export(args: argparse.Namespace) -> None:
    client = _client()
    result = client.export_rag(args.name, args.output)
    console.print(f"[green]✔[/green] Exported [bold]{args.name}[/bold] → {result.get('path', '')}")


def cmd_import(args: argparse.Namespace) -> None:
    client = _client()

    # Show info about the file
    info = client.peek_rag_file(args.file)
    console.print(f"  Name: {info.get('name')}")
    console.print(f"  Model: {info.get('embedding_model')}")
    console.print(f"  Files: {info.get('file_count', '?')}, Chunks: {info.get('chunk_count', '?')}")
    console.print(f"  Size: {info.get('file_size_mb', '?')} MB")

    result = client.import_rag(args.file, name=args.name)
    console.print(f"[green]✔[/green] Imported as [bold]{result.get('name', '')}[/bold]")


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
    from rag_kb.web_ui import launch_web_ui

    mode = "native window" if args.native else f"http://{args.host}:{args.port}"
    console.print(f"Launching web UI ({mode}) …")
    launch_web_ui(
        host=args.host,
        port=args.port,
        native=args.native,
    )


def cmd_config(args: argparse.Namespace) -> None:
    from rag_kb.config import CONFIG_PATH, DATA_DIR

    client = _client()
    settings = client.get_settings()
    console.print(f"[bold]Data directory:[/bold] {DATA_DIR}")
    console.print(f"[bold]Config file:[/bold] {CONFIG_PATH}")
    console.print()

    import yaml

    console.print(yaml.safe_dump(settings, default_flow_style=False))


def cmd_download_models(args: argparse.Namespace) -> None:
    client = _client()
    console.print("Downloading models …")

    result = client.download_models(
        output_dir=args.output,
        model_name=args.model,
    )

    paths = result.get("paths", [])
    for p in paths:
        console.print(f"  [green]✔[/green] {Path(p).name}")
    console.print(f"[green]✔[/green] Done — {result.get('count', len(paths))} model(s) saved.")


def cmd_models(args: argparse.Namespace) -> None:
    """Handle the models subcommand: list, info, download, delete."""
    action = getattr(args, "models_action", None)
    if not action:
        console.print("Usage: rag-kb models {list|info|download|delete}")
        return

    client = _client()

    if action == "list":
        models = client.list_models(model_type=getattr(args, "type", None))
        if not models:
            console.print("[yellow]No models found.[/yellow]")
            return

        table = Table(title="Available Models", show_lines=False, pad_edge=False)
        table.add_column("Name", style="bold")
        table.add_column("Type", no_wrap=True)
        table.add_column("Dim/Ctx", no_wrap=True, justify="right")
        table.add_column("Size", justify="right", no_wrap=True)
        table.add_column("Status", no_wrap=True)

        for m in models:
            size = f"{m['model_size_mb']}MB" if m.get("model_size_mb") else "API"
            dim = m.get("dimensions", 0)
            dim_ctx = f"{dim or '—'}/{m.get('max_tokens', '?')}"
            status = m.get("status", "available")
            provider = m.get("provider", "local")
            status_style = {
                "bundled": "[green]bundled[/green]",
                "downloaded": "[green]downloaded[/green]",
                "api": f"[blue]api ({provider})[/blue]",
                "available": "[dim]available[/dim]",
            }.get(status, status)
            table.add_row(
                m["name"],
                m["type"],
                dim_ctx,
                size,
                status_style,
            )
        console.print(table)

    elif action == "info":
        model_name = args.model_name
        info = client.get_model_info(model_name)
        if info is None:
            console.print(f"[red]Model '{model_name}' not found in registry.[/red]")
            return

        console.print(f"\n[bold]{info.get('display_name', model_name)}[/bold]")
        console.print(f"  Name: {info['name']}")
        console.print(f"  Type: {info['type']}")
        console.print(f"  Provider: {info.get('provider', 'local')}")
        console.print(f"  Dimensions: {info.get('dimensions', '?')}")
        console.print(f"  Max tokens: {info.get('max_tokens', '?')}")
        console.print(f"  Size: ~{info.get('model_size_mb', 0)} MB")
        console.print(f"  Status: {info.get('status', 'unknown')}")
        console.print(f"  License: {info.get('license', 'unknown')}")
        if info.get("trust_remote_code"):
            console.print("  [yellow]⚠ Requires trust_remote_code=True[/yellow]")
        if info.get("description"):
            console.print(f"\n  {info['description']}")
        if info.get("use_case_tags"):
            console.print(f"  Tags: {', '.join(info['use_case_tags'])}")

    elif action == "download":
        model_name = args.model_name
        trust = getattr(args, "trust", False)
        console.print(f"Downloading [bold]{model_name}[/bold] …")
        try:
            result = client.download_model(model_name, trust_remote_code=trust)
            console.print(f"[green]✔[/green] Downloaded to {result.get('path', '')}")
        except Exception as exc:
            console.print(f"[red]Error:[/red] {exc}")

    elif action == "delete":
        model_name = args.model_name
        if not getattr(args, "yes", False) and (
            not console.input(f"Delete model '{model_name}'? [y/N] ")
            .strip()
            .lower()
            .startswith("y")
        ):
            console.print("Cancelled.")
            return
        try:
            result = client.delete_model(model_name)
            if result.get("deleted"):
                console.print(f"[green]✔[/green] Deleted {model_name}")
            else:
                console.print("[yellow]Model not found locally.[/yellow]")
        except Exception as exc:
            console.print(f"[red]Error:[/red] {exc}")


def cmd_daemon(args: argparse.Namespace) -> None:
    """Handle daemon subcommands: status, stop, restart, logs."""
    action = getattr(args, "action", None)
    if not action:
        console.print("Usage: rag-kb daemon {status|stop|restart|logs}")
        return

    from rag_kb.daemon_client import DaemonClient

    if action == "logs":
        _cmd_daemon_logs(args)
        return

    client = DaemonClient()

    if action == "status":
        if client._probe():
            client.connect()
            info = client.ping()
            console.print(
                f"[green]●[/green] Daemon is running (PID {info.get('pid', '?')}, uptime {info.get('uptime', '?')}s)"
            )
            client.close()
        else:
            console.print("[red]●[/red] Daemon is not running.")

    elif action == "stop":
        if client._probe():
            client.connect()
            client.shutdown()
            client.close()
            # Wait briefly for graceful shutdown
            import time

            time.sleep(1)
            # If still alive, force-kill
            from rag_kb.daemon_client import _is_daemon_alive, kill_stale_daemon

            if _is_daemon_alive():
                kill_stale_daemon(graceful_timeout=3.0)
            console.print("[green]✔[/green] Daemon stopped.")
        else:
            # PID alive but not responding? Kill it.
            from rag_kb.daemon_client import _is_daemon_alive, kill_stale_daemon

            if _is_daemon_alive():
                kill_stale_daemon(graceful_timeout=3.0)
                console.print("[green]✔[/green] Stale daemon killed.")
            else:
                console.print("[yellow]Daemon is not running.[/yellow]")

    elif action == "restart":
        if client._probe():
            client.connect()
            client.shutdown()
            client.close()
            console.print("Daemon stopped. Restarting …")
            import time

            time.sleep(1)
        # Kill any stale daemon that didn't shut down cleanly
        from rag_kb.daemon_client import _is_daemon_alive, kill_stale_daemon

        if _is_daemon_alive():
            kill_stale_daemon(graceful_timeout=3.0)
            import time

            time.sleep(0.5)
        client.ensure_daemon()
        console.print("[green]✔[/green] Daemon restarted.")


def _cmd_daemon_logs(args: argparse.Namespace) -> None:
    """Tail the daemon log file, similar to ``docker logs -t -f``."""
    import time as _time

    from rag_kb.config import DATA_DIR

    log_file = DATA_DIR / "daemon.log"

    follow = getattr(args, "follow", False)
    tail_lines = getattr(args, "tail", 50)

    if not log_file.exists():
        console.print(f"[yellow]Log file not found:[/yellow] {log_file}")
        console.print("The daemon has not been started yet or logs have been cleared.")
        return

    if follow:
        console.print(f"[dim]Following {log_file}  (Ctrl+C to stop)[/dim]")
    # -- Print last N lines --
    try:
        with open(log_file, encoding="utf-8", errors="replace") as fh:
            if tail_lines == 0:
                # Print ALL lines
                for line in fh:
                    sys.stdout.write(line)
            else:
                # Efficient tail: read the whole file and take last N
                # For very large files we could seek backwards, but
                # 10 MB log files are handled fine this way.
                lines = fh.readlines()
                for line in lines[-tail_lines:]:
                    sys.stdout.write(line)
                fh.tell()
    except OSError as exc:
        console.print(f"[red]Error reading log file:[/red] {exc}")
        return

    if not follow:
        return

    # -- Follow mode (like tail -f) --
    try:
        with open(log_file, encoding="utf-8", errors="replace") as fh:
            # Seek to end
            fh.seek(0, 2)
            while True:
                line = fh.readline()
                if line:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                else:
                    # Check if file was rotated (size shrank)
                    try:
                        current_size = log_file.stat().st_size
                        if current_size < fh.tell():
                            # File was rotated — reopen from start
                            console.print("[dim]--- log rotated ---[/dim]")
                            fh.seek(0)
                    except OSError:
                        pass
                    _time.sleep(0.3)
    except KeyboardInterrupt:
        console.print("\n[dim]Log following stopped.[/dim]")


def _dir_size_mb(path: Path) -> float:
    """Return total size of a directory in megabytes."""
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return total / (1024 * 1024)


# -----------------------------------------------------------------------
# Monitoring / stats commands
# -----------------------------------------------------------------------


def cmd_stats(args: argparse.Namespace) -> None:
    """Show comprehensive metrics and monitoring statistics."""
    client = _client()
    category = getattr(args, "category", "all")

    if category in ("all", "dashboard"):
        dashboard = client.get_metrics_dashboard()

        # System section
        sys_snap = dashboard.get("system") or {}
        if sys_snap:
            t = Table(title="System Health", show_header=False)
            t.add_column("Metric", style="bold cyan")
            t.add_column("Value", style="white")
            t.add_row("CPU", f"{sys_snap.get('cpu_percent', 0):.1f}%")
            t.add_row(
                "Memory",
                f"{sys_snap.get('memory_used_mb', 0):.0f} / {sys_snap.get('memory_total_mb', 0):.0f} MB ({sys_snap.get('memory_percent', 0):.1f}%)",
            )
            t.add_row("Process RAM", f"{sys_snap.get('process_memory_mb', 0):.1f} MB")
            t.add_row("Disk Free", f"{sys_snap.get('disk_free_mb', 0):.0f} MB")
            uptime = sys_snap.get("daemon_uptime_seconds", 0)
            h, rem = divmod(int(uptime), 3600)
            m = rem // 60
            t.add_row("Daemon Uptime", f"{h}h {m}m")
            t.add_row("Active Connections", str(sys_snap.get("active_connections", 0)))
            t.add_row("Total RPC Calls", str(sys_snap.get("total_rpc_calls", 0)))
            console.print(t)

        # Indexing aggregates
        idx = dashboard.get("indexing_aggregates") or {}
        t = Table(title="Indexing Aggregates", show_header=False)
        t.add_column("Metric", style="bold yellow")
        t.add_column("Value", style="white")
        t.add_row("Total Runs", str(idx.get("total_runs", 0)))
        t.add_row("Avg Duration", f"{idx.get('avg_duration', 0) or 0:.1f}s")
        t.add_row("Avg Throughput", f"{idx.get('avg_throughput', 0) or 0:.1f} chunks/s")
        t.add_row("Total Errors", str(int(idx.get("total_errors", 0) or 0)))
        console.print(t)

        # Search aggregates
        srch = dashboard.get("search_aggregates") or {}
        t = Table(title="Search Aggregates", show_header=False)
        t.add_column("Metric", style="bold green")
        t.add_column("Value", style="white")
        t.add_row("Total Queries", str(srch.get("total_queries", 0)))
        t.add_row("Avg Latency", f"{srch.get('avg_latency_ms', 0) or 0:.1f} ms")
        t.add_row("Avg Results", f"{srch.get('avg_results', 0) or 0:.1f}")
        t.add_row("Avg Top Score", f"{srch.get('avg_top_score', 0) or 0:.3f}")
        console.print(t)

        # Embedding aggregates
        emb = dashboard.get("embedding_aggregates") or {}
        t = Table(title="Embedding Aggregates", show_header=False)
        t.add_column("Metric", style="bold magenta")
        t.add_column("Value", style="white")
        t.add_row("Total Batches", str(emb.get("total_batches", 0)))
        t.add_row("Avg Batch Time", f"{emb.get('avg_batch_ms', 0) or 0:.1f} ms")
        t.add_row("Avg Throughput", f"{emb.get('avg_throughput', 0) or 0:.1f} chunks/s")
        t.add_row("Total Embedded", str(int(emb.get("total_texts_embedded", 0) or 0)))
        console.print(t)

    if category in ("all", "indexing"):
        history = client.get_indexing_history(limit=getattr(args, "limit", 10))
        if history:
            t = Table(title="Indexing History")
            t.add_column("Date", style="dim")
            t.add_column("RAG")
            t.add_column("Status")
            t.add_column("Files", justify="right")
            t.add_column("Chunks", justify="right")
            t.add_column("Duration", justify="right")
            t.add_column("Scan", justify="right", style="dim")
            t.add_column("Parse", justify="right", style="dim")
            t.add_column("Embed", justify="right", style="dim")
            t.add_column("Upsert", justify="right", style="dim")
            t.add_column("Errors", justify="right", style="red")
            for h in history:
                import datetime

                ts = h.get("started_at", 0)
                dt = datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M") if ts else "?"
                color = "green" if h.get("status") == "completed" else "red"
                t.add_row(
                    dt,
                    h.get("rag_name", ""),
                    f"[{color}]{h.get('status', '')}[/{color}]",
                    str(h.get("processed_files", 0)),
                    str(h.get("total_chunks", 0)),
                    f"{h.get('duration_seconds', 0):.1f}s",
                    f"{h.get('scan_seconds', 0):.2f}s",
                    f"{h.get('parse_seconds', 0):.2f}s",
                    f"{h.get('embed_seconds', 0):.2f}s",
                    f"{h.get('upsert_seconds', 0):.2f}s",
                    str(h.get("error_count", 0)),
                )
            console.print(t)

    if category in ("all", "search"):
        search_stats = client.get_search_stats(limit=getattr(args, "limit", 20))
        if search_stats:
            t = Table(title="Recent Search Queries")
            t.add_column("Time", style="dim")
            t.add_column("Query")
            t.add_column("Results", justify="right")
            t.add_column("Total ms", justify="right")
            t.add_column("Vec ms", justify="right", style="dim")
            t.add_column("BM25 ms", justify="right", style="dim")
            t.add_column("Rerank ms", justify="right", style="dim")
            t.add_column("Top Score", justify="right")
            for s in search_stats:
                import datetime

                ts = s.get("timestamp", 0)
                dt = datetime.datetime.fromtimestamp(ts).strftime("%H:%M:%S") if ts else "?"
                t.add_row(
                    dt,
                    (s.get("query_text", "")[:40] + "…")
                    if len(s.get("query_text", "")) > 40
                    else s.get("query_text", ""),
                    str(s.get("result_count", 0)),
                    f"{s.get('total_ms', 0):.0f}",
                    f"{s.get('vector_search_ms', 0):.0f}",
                    f"{s.get('bm25_ms', 0):.0f}",
                    f"{s.get('rerank_ms', 0):.0f}",
                    f"{s.get('top_score', 0):.3f}",
                )
            console.print(t)

    if category in ("all", "embedding"):
        emb_stats = client.get_embedding_stats(limit=getattr(args, "limit", 20))
        if emb_stats:
            t = Table(title="Recent Embedding Batches")
            t.add_column("Time", style="dim")
            t.add_column("Backend")
            t.add_column("Model")
            t.add_column("Batch Size", justify="right")
            t.add_column("Duration ms", justify="right")
            t.add_column("Chunks/s", justify="right")
            t.add_column("Device", style="dim")
            for e in emb_stats:
                import datetime

                ts = e.get("timestamp", 0)
                dt = datetime.datetime.fromtimestamp(ts).strftime("%H:%M:%S") if ts else "?"
                t.add_row(
                    dt,
                    e.get("backend_type", ""),
                    (e.get("model_name", "")[:30]),
                    str(e.get("batch_size", 0)),
                    f"{e.get('duration_ms', 0):.0f}",
                    f"{e.get('chunks_per_second', 0):.1f}",
                    e.get("device", ""),
                )
            console.print(t)

    if category in ("all", "vector"):
        vs = client.get_vector_store_details()
        if vs:
            t = Table(title="Vector Store / ChromaDB", show_header=False)
            t.add_column("Metric", style="bold blue")
            t.add_column("Value", style="white")
            t.add_row("Total Chunks", str(vs.get("total_chunks", 0)))
            t.add_row("Total Files", str(vs.get("total_files", 0)))
            t.add_row("DB Size", f"{vs.get('db_size_mb', 0):.2f} MB")
            t.add_row("Avg Chunks/File", f"{vs.get('avg_chunks_per_file', 0):.1f}")
            hnsw = vs.get("hnsw_config", {})
            if hnsw:
                t.add_row("HNSW Space", hnsw.get("space", "?"))
                t.add_row("construction_ef", str(hnsw.get("construction_ef", "?")))
                t.add_row("M", str(hnsw.get("M", "?")))
            t.add_row("DB Path", vs.get("db_path", "?"))
            console.print(t)

    client.close()


def cmd_monitor(args: argparse.Namespace) -> None:
    """Live monitor: continuously display key metrics (like top/htop)."""
    import time as _time

    client = _client()
    interval = getattr(args, "interval", 5)

    try:
        from rich.layout import Layout
        from rich.live import Live
        from rich.panel import Panel

        def _build_display():
            dashboard = client.get_metrics_dashboard()
            sys_snap = dashboard.get("system") or {}

            # System panel
            sys_lines = []
            sys_lines.append(
                f"CPU: {sys_snap.get('cpu_percent', 0):.1f}%  |  "
                f"Memory: {sys_snap.get('memory_used_mb', 0):.0f}/{sys_snap.get('memory_total_mb', 0):.0f} MB ({sys_snap.get('memory_percent', 0):.1f}%)  |  "
                f"Process: {sys_snap.get('process_memory_mb', 0):.1f} MB"
            )
            uptime = sys_snap.get("daemon_uptime_seconds", 0)
            h, rem = divmod(int(uptime), 3600)
            m = rem // 60
            sys_lines.append(
                f"Uptime: {h}h {m}m  |  "
                f"Connections: {sys_snap.get('active_connections', 0)}  |  "
                f"RPC Calls: {sys_snap.get('total_rpc_calls', 0)}"
            )
            sys_panel = Panel("\n".join(sys_lines), title="System", border_style="cyan")

            # Indexing panel
            idx = dashboard.get("indexing_aggregates") or {}
            last = dashboard.get("last_indexing_run") or {}
            idx_lines = [
                f"Runs: {idx.get('total_runs', 0)}  |  "
                f"Avg: {idx.get('avg_duration', 0) or 0:.1f}s  |  "
                f"Throughput: {idx.get('avg_throughput', 0) or 0:.1f} chunks/s",
            ]
            if last:
                idx_lines.append(
                    f"Last: {last.get('status', '?')} — {last.get('processed_files', 0)} files, "
                    f"{last.get('total_chunks', 0)} chunks in {last.get('duration_seconds', 0):.1f}s"
                )
            idx_panel = Panel("\n".join(idx_lines), title="Indexing", border_style="yellow")

            # Search panel
            srch = dashboard.get("search_aggregates") or {}
            srch_lines = [
                f"Queries: {srch.get('total_queries', 0)}  |  "
                f"Avg Latency: {srch.get('avg_latency_ms', 0) or 0:.1f} ms  |  "
                f"Avg Top Score: {srch.get('avg_top_score', 0) or 0:.3f}",
            ]
            srch_panel = Panel("\n".join(srch_lines), title="Search", border_style="green")

            # Embedding panel
            emb = dashboard.get("embedding_aggregates") or {}
            emb_lines = [
                f"Batches: {emb.get('total_batches', 0)}  |  "
                f"Avg: {emb.get('avg_batch_ms', 0) or 0:.1f} ms  |  "
                f"Throughput: {emb.get('avg_throughput', 0) or 0:.1f} chunks/s",
            ]
            emb_panel = Panel("\n".join(emb_lines), title="Embeddings", border_style="magenta")

            layout = Layout()
            layout.split_column(
                Layout(sys_panel, name="system"),
                Layout(name="middle"),
                Layout(name="bottom"),
            )
            layout["middle"].split_row(
                Layout(idx_panel, name="indexing"),
                Layout(emb_panel, name="embedding"),
            )
            layout["bottom"].split_row(
                Layout(srch_panel, name="search"),
            )
            return layout

        console.print(f"[dim]Live monitoring (refreshing every {interval}s, Ctrl+C to stop)[/dim]")
        with Live(_build_display(), console=console, refresh_per_second=1) as live:
            while True:
                _time.sleep(interval)
                try:
                    live.update(_build_display())
                except Exception:
                    pass

    except KeyboardInterrupt:
        console.print("\n[dim]Monitor stopped.[/dim]")
    finally:
        client.close()


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
    p.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel parsing workers (default: from config)",
    )

    # cancel-index
    sub.add_parser("cancel-index", help="Cancel a running indexing operation")

    # verify
    p = sub.add_parser("verify", help="Verify index consistency (manifest vs vector store)")
    p.add_argument("--rag", default=None, help="RAG name (defaults to active)")

    # search (NEW — feature parity)
    p = sub.add_parser("search", help="Search the active RAG")
    p.add_argument("query", help="Search query")
    p.add_argument("--rag", default=None, help="RAG name (defaults to active)")
    p.add_argument("--top", "-n", type=int, default=5, help="Number of results (default: 5)")

    # status (NEW — feature parity)
    p = sub.add_parser("status", help="Show index status for a RAG")
    p.add_argument("--rag", default=None, help="RAG name (defaults to active)")

    # files (NEW — feature parity)
    p = sub.add_parser("files", help="List indexed files")
    p.add_argument("--rag", default=None, help="RAG name (defaults to active)")
    p.add_argument("--page", type=int, default=1, help="Page number (default: 1)")
    p.add_argument("--limit", "-n", type=int, default=50, help="Files per page (default: 50)")
    p.add_argument("--filter", "-f", default="", help="Filter files by name substring")

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
    p = sub.add_parser("ui", help="Launch web UI")
    p.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    p.add_argument("--port", type=int, default=8501, help="HTTP port (default: 8501)")
    p.add_argument("--native", action="store_true", help="Open in a native desktop window")

    # config
    sub.add_parser("config", help="Show current configuration")

    # download-models
    p = sub.add_parser("download-models", help="Pre-download ML models for offline/bundled use")
    p.add_argument(
        "--output", "-o", default=None, help="Output directory (default: models/ in project root)"
    )
    p.add_argument(
        "--model",
        default=None,
        help="Specific model name to download (default: all default models)",
    )

    # models
    p = sub.add_parser("models", help="Manage embedding and reranker models")
    models_sub = p.add_subparsers(dest="models_action", help="Model management commands")

    mp = models_sub.add_parser("list", help="List all available models with status")
    mp.add_argument(
        "--type", choices=["embedding", "reranker"], default=None, help="Filter by model type"
    )

    mp = models_sub.add_parser("info", help="Show detailed info about a model")
    mp.add_argument("model_name", help="Model name (e.g. BAAI/bge-m3)")

    mp = models_sub.add_parser("download", help="Download a model for local use")
    mp.add_argument("model_name", help="Model name to download")
    mp.add_argument(
        "--trust", action="store_true", help="Trust remote code (required for some models)"
    )

    mp = models_sub.add_parser("delete", help="Delete a downloaded model")
    mp.add_argument("model_name", help="Model name to delete")
    mp.add_argument("-y", "--yes", action="store_true", help="Skip confirmation")

    # daemon
    p = sub.add_parser("daemon", help="Manage the background daemon")
    daemon_sub = p.add_subparsers(dest="action", help="Daemon commands")

    daemon_sub.add_parser("status", help="Show daemon status")
    daemon_sub.add_parser("stop", help="Stop the daemon")
    daemon_sub.add_parser("restart", help="Restart the daemon")

    lp = daemon_sub.add_parser("logs", help="Show daemon logs (like docker logs)")
    lp.add_argument("-f", "--follow", action="store_true", help="Follow log output (like tail -f)")
    lp.add_argument(
        "-n",
        "--tail",
        type=int,
        default=50,
        help="Number of lines to show from the end (default: 50, 0 = all)",
    )

    # stats
    p = sub.add_parser("stats", help="Show detailed metrics and monitoring statistics")
    p.add_argument(
        "category",
        nargs="?",
        default="all",
        choices=["all", "dashboard", "indexing", "search", "embedding", "vector"],
        help="Which category to show (default: all)",
    )
    p.add_argument(
        "--limit", "-n", type=int, default=10, help="Number of history items (default: 10)"
    )

    # monitor
    p = sub.add_parser("monitor", help="Live monitoring dashboard (like htop for RAG)")
    p.add_argument(
        "--interval", "-i", type=int, default=5, help="Refresh interval in seconds (default: 5)"
    )

    return parser


_CMD_MAP = {
    "create": cmd_create,
    "list": cmd_list,
    "use": cmd_use,
    "delete": cmd_delete,
    "detach": cmd_detach,
    "attach": cmd_attach,
    "index": cmd_index,
    "cancel-index": cmd_cancel_index,
    "verify": cmd_verify,
    "search": cmd_search,
    "status": cmd_status,
    "files": cmd_files,
    "export": cmd_export,
    "import": cmd_import,
    "serve": cmd_serve,
    "ui": cmd_ui,
    "config": cmd_config,
    "download-models": cmd_download_models,
    "models": cmd_models,
    "daemon": cmd_daemon,
    "stats": cmd_stats,
    "monitor": cmd_monitor,
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
