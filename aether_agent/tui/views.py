"""Rich rendering helpers for the Aether Agent terminal interface."""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table


def create_console() -> Console:
    return Console()


def render_banner(console: Console, model: str, session_id: str) -> None:
    """Render startup welcome banner."""
    body = (
        f"[bold cyan]Aether Agent[/bold cyan]  |  Model: [green]{model}[/green]  |  Session: [yellow]{session_id}[/yellow]\n"
        "[dim]Commands: /model, /skills, /memory, /cron, /gateway, /clear, /exit, /help[/dim]"
    )
    console.print(Panel(body, border_style="cyan", expand=False))


def render_assistant_reply(console: Console, content: str) -> None:
    """Render assistant Markdown response inside a styled panel."""
    console.print(
        Panel(
            Markdown(content or "(empty response)"),
            title="Aether",
            title_align="left",
            border_style="bright_blue",
        )
    )


def render_tool_event(console: Console, event: dict[str, Any]) -> None:
    """Render concise tool start/end status line."""
    phase = event.get("phase")
    name = event.get("name", "tool")
    if phase == "start":
        console.print(f"[dim]→ Running tool [bold]{name}[/bold]...[/dim]")
    elif phase == "end":
        status = event.get("status", "unknown")
        color = "green" if status == "success" else "red"
        console.print(f"[dim]← Tool [bold]{name}[/bold] [{color}]{status}[/{color}][/dim]")


def render_skills_table(console: Console, skills: list[dict[str, Any]]) -> None:
    """Display loaded skills in a Rich table."""
    if not skills:
        console.print("[yellow]No skills loaded.[/yellow]")
        return
    table = Table(title="Loaded Skills")
    table.add_column("Name", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Description")
    for s in skills:
        table.add_row(s["name"], s.get("status", "active"), s.get("description", ""))
    console.print(table)


def render_memory_results(console: Console, results: list[dict[str, Any]]) -> None:
    """Display FTS5 search hits in a Rich table."""
    if not results:
        console.print("[yellow]No matching memory entries found.[/yellow]")
        return
    table = Table(title="Memory Recall (FTS5)")
    table.add_column("Session", style="cyan")
    table.add_column("Role", style="magenta")
    table.add_column("Content")
    for r in results:
        snippet = (r.get("content") or "").replace("\n", " ")[:120]
        table.add_row(r.get("session_id", ""), r.get("role", ""), snippet)
    console.print(table)
