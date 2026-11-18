"""Rich rendering helpers and Hermes-inspired modern TUI views for Ivy."""

from __future__ import annotations

import os
import shutil
from typing import Any

from rich import box
from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from ivy import __version__
from ivy.tui.theme import IVY_ASCII_LOGO, IVY_HERO_ART, ThemeConfig, get_theme

_TOOL_GROUPS: dict[str, str] = {
    "terminal": "terminal",
    "read_file": "files",
    "write_file": "files",
    "list_directory": "files",
    "patch_file": "files",
    "search_files": "files",
    "code_runner": "sandbox",
    "web_fetch": "web",
    "web_search": "web",
    "read_skill": "skills",
}


def create_console() -> Console:
    return Console()


def _group_tools(tool_names: list[str]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for name in sorted(tool_names):
        group = _TOOL_GROUPS.get(name, "other")
        grouped.setdefault(group, []).append(name)
    return grouped


def render_banner(
    console: Console,
    model: str,
    session_id: str,
    cwd: str | None = None,
    tool_names: list[str] | None = None,
    skills: list[dict[str, Any]] | None = None,
    theme: ThemeConfig | None = None,
) -> None:
    """Render Hermes-inspired split welcome banner with hero emblem, tools, and skills."""
    t = theme or get_theme()
    resolved_cwd = cwd or os.getcwd()
    tools = list(tool_names or [])
    loaded_skills = list(skills or [])

    palette = {
        "title": t.banner_title,
        "accent": t.banner_accent,
        "border": t.banner_border,
        "dim": t.banner_dim,
    }

    left_lines = [
        "",
        IVY_HERO_ART.format(**palette),
        "",
        f"[bold {t.banner_accent}]{escape(model)}[/] [dim]·[/] [dim]Ivy v{__version__}[/]",
        f"[dim]{escape(resolved_cwd)}[/]",
        f"[dim]Session:[/] [{t.ui_label}]{escape(session_id)}[/]",
    ]

    right_lines: list[str] = [f"[bold {t.banner_accent}]Available Tools[/]"]
    if tools:
        for group, names in _group_tools(tools).items():
            joined = ", ".join(f"[{t.banner_text}]{escape(n)}[/]" for n in names)
            right_lines.append(f"[dim]{group}:[/] {joined}")
    else:
        right_lines.append("[dim]No tools registered[/]")

    right_lines.extend(["", f"[bold {t.banner_accent}]Available Skills[/]"])
    if loaded_skills:
        skill_badges = ", ".join(
            f"[{t.banner_text}]{escape(s['name'])}[/]" for s in loaded_skills[:12]
        )
        if len(loaded_skills) > 12:
            skill_badges += f" [dim]+{len(loaded_skills) - 12} more[/]"
        right_lines.append(skill_badges)
    else:
        right_lines.append("[dim]No skills installed[/]")

    right_lines.extend(
        [
            "",
            f"[dim]{len(tools)} tools · {len(loaded_skills)} skills · /help for commands · /theme to switch skin[/]",
        ]
    )

    grid = Table.grid(padding=(0, 3))
    grid.add_column("left", justify="left")
    grid.add_column("right", justify="left")
    grid.add_row("\n".join(left_lines), "\n".join(right_lines))

    outer = Panel(
        grid,
        title=f"[bold {t.banner_title}]{t.glyph} Ivy v{__version__}[/]",
        border_style=t.banner_border,
        box=box.ROUNDED,
        padding=(0, 2),
    )

    term_size = shutil.get_terminal_size((100, 30))
    console.print()
    if term_size.columns >= 75 and term_size.lines >= 28:
        console.print(IVY_ASCII_LOGO.format(**palette))
        console.print()
    console.print(outer)


def render_assistant_reply(
    console: Console, content: str, theme: ThemeConfig | None = None
) -> None:
    """Render assistant Markdown response framed with modern accent bars."""
    t = theme or get_theme()
    rule = f"[{t.ui_accent}]{'─' * 44}[/]"
    console.print()
    console.print(rule)
    console.print(f"[bold {t.ui_accent}]{t.glyph} Ivy[/]")
    console.print(Markdown(content or "(empty response)"))
    console.print(rule)


def render_tool_event(
    console: Console, event: dict[str, Any], theme: ThemeConfig | None = None
) -> None:
    """Render concise Hermes-style tool activity row."""
    t = theme or get_theme()
    phase = event.get("phase")
    name = event.get("name", "tool")
    if phase == "start":
        args = event.get("arguments") or {}
        preview = ""
        if isinstance(args, dict) and args:
            first_val = str(next(iter(args.values()))).replace("\n", " ")
            if len(first_val) > 48:
                first_val = first_val[:45] + "..."
            preview = f" [dim]{escape(first_val)}[/]"
        console.print(f"  [dim]┊[/] [{t.ui_label}]⚙ {escape(name)}[/]{preview}")
    elif phase == "end":
        status = event.get("status", "unknown")
        if status == "success":
            console.print(f"  [dim]┊[/] [{t.ui_ok}]✓ {escape(name)}[/]")
        elif status == "rejected":
            console.print(f"  [dim]┊[/] [{t.ui_warn}]⊘ {escape(name)} (rejected)[/]")
        else:
            console.print(f"  [dim]┊[/] [{t.ui_error}]✗ {escape(name)} ({escape(status)})[/]")


def render_error(
    console: Console, exc: Exception | str, theme: ThemeConfig | None = None
) -> None:
    """Render a clean, actionable error panel without raw stack noise."""
    t = theme or get_theme()
    msg = str(exc)
    hint = ""
    lower = msg.lower()
    if "401" in lower or "api key" in lower or "authentication" in lower or "missing-api-key" in lower:
        hint = (
            "\n\n[dim]Tip: Set your API key in [bold].env[/bold] "
            "(OPENAI_API_KEY / ANTHROPIC_API_KEY / GEMINI_API_KEY / OPENROUTER_API_KEY) "
            "or switch to a local model with [bold]/model ollama/qwen2.5[/bold].[/dim]"
        )
    console.print(
        Panel(
            f"[{t.ui_error}]{escape(msg)}[/]{hint}",
            title=f"[bold {t.ui_error}]Turn Error[/]",
            border_style=t.ui_error,
            box=box.ROUNDED,
            expand=False,
        )
    )


def render_help_table(
    console: Console, theme: ThemeConfig | None = None
) -> None:
    """Render available slash commands in a rounded Rich table."""
    t = theme or get_theme()
    table = Table(
        title=f"[{t.banner_title}]{t.glyph} Available Commands[/]",
        box=box.ROUNDED,
        border_style=t.banner_border,
    )
    table.add_column("Command", style=f"bold {t.ui_label}")
    table.add_column("Description")
    rows = [
        ("/setup [base] [key] [model]", "Configure API endpoint/key & save to .env"),
        ("/model [name]", "Show active model or switch & save to .env"),
        ("/session [list|<name>]", "List saved sessions or switch active session"),
        ("/new [name]", "Start a new fresh session"),
        ("/skills list", "List all discovered skills"),
        ("/skills inspect <name>", "Inspect instructions of a skill"),
        ("/skills remove <name>", "Delete a skill"),
        ("/skills reflect", "Extract a reusable draft skill from current session"),
        ("/memory <query>", "Search conversation history via SQLite FTS5"),
        ("/cron list", "Show scheduled background cron jobs"),
        ("/gateway status", "Show Telegram/Discord gateway configuration"),
        ("/theme [name]", "Switch & save TUI theme (ivy, hermes, ocean)"),
        ("/clear", "Clear current session message history"),
        ("/exit", "Exit Ivy"),
    ]
    for cmd, desc in rows:
        table.add_row(cmd, desc)
    console.print(table)


def render_skills_table(
    console: Console, skills: list[dict[str, Any]], theme: ThemeConfig | None = None
) -> None:
    """Display loaded skills in a Rich table."""
    t = theme or get_theme()
    if not skills:
        console.print(f"[{t.ui_warn}]No skills loaded.[/]")
        return
    table = Table(
        title=f"[{t.banner_title}]Loaded Skills[/]",
        box=box.ROUNDED,
        border_style=t.banner_border,
    )
    table.add_column("Name", style=f"bold {t.ui_label}")
    table.add_column("Status", style=t.ui_ok)
    table.add_column("Description")
    for s in skills:
        table.add_row(s["name"], s.get("status", "active"), s.get("description", ""))
    console.print(table)


def render_memory_results(
    console: Console, results: list[dict[str, Any]], theme: ThemeConfig | None = None
) -> None:
    """Display FTS5 search hits in a Rich table."""
    t = theme or get_theme()
    if not results:
        console.print(f"[{t.ui_warn}]No matching memory entries found.[/]")
        return
    table = Table(
        title=f"[{t.banner_title}]Memory Recall (FTS5)[/]",
        box=box.ROUNDED,
        border_style=t.banner_border,
    )
    table.add_column("Session", style=t.ui_label)
    table.add_column("Role", style=t.ui_accent)
    table.add_column("Content")
    for r in results:
        snippet = (r.get("content") or "").replace("\n", " ")[:120]
        table.add_row(r.get("session_id", ""), r.get("role", ""), snippet)
    console.print(table)


def render_sessions_table(
    console: Console,
    sessions: list[dict[str, Any]],
    current_session_id: str,
    theme: ThemeConfig | None = None,
) -> None:
    """Display saved sessions in a Rich table, marking the active session."""
    t = theme or get_theme()
    if not sessions:
        console.print(
            f"[{t.ui_warn}]No saved sessions yet.[/] Active session: [{t.ui_label}]{current_session_id}[/]"
        )
        return
    table = Table(
        title=f"[{t.banner_title}]Saved Sessions[/]",
        box=box.ROUNDED,
        border_style=t.banner_border,
    )
    table.add_column("", width=2)
    table.add_column("Session ID", style=f"bold {t.ui_label}")
    table.add_column("Platform", style=t.ui_accent)
    table.add_column("Messages", justify="right")
    table.add_column("Last Updated", style="dim")
    for s in sessions:
        sid = s.get("session_id", "")
        marker = f"[{t.ui_ok}]*[/]" if sid == current_session_id else ""
        updated = str(s.get("updated_at") or "")[:19].replace("T", " ")
        table.add_row(
            marker,
            sid,
            str(s.get("platform") or ""),
            str(s.get("message_count", 0)),
            updated,
        )
    console.print(table)
