"""Rich rendering helpers and Hermes-inspired modern TUI views for Ivy."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

from rich import box
from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from ivy import __version__
from ivy.core.types import ChatMessage, Role
from ivy.tui.helpers import format_relative_time, format_token_count, get_git_branch
from ivy.tui.theme import IVY_ASCII_LOGO, IVY_HERO_ART, ThemeConfig, get_theme

_TOOL_GROUPS: dict[str, str] = {
    "terminal": "terminal",
    "process": "terminal",
    "read_file": "files",
    "write_file": "files",
    "list_directory": "files",
    "patch_file": "files",
    "search_files": "files",
    "code_runner": "sandbox",
    "web_fetch": "web",
    "web_search": "web",
    "read_skill": "skills",
    "skill_manage": "skills",
    "todo_list": "planning",
    "clarify": "planning",
    "delegate_task": "agents",
    "remember_preference": "memory",
    "memory_note": "memory",
    "session_search": "memory",
    "cron_manage": "automation",
}


def create_console() -> Console:
    return Console()


def _group_tools(tool_names: list[str]) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for name in sorted(tool_names):
        group = "mcp" if name.startswith("mcp_") else _TOOL_GROUPS.get(name, "other")
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
    session_title: str | None = None,
    git_branch: str | None = None,
) -> None:
    """Render Hermes-inspired split welcome banner with hero emblem, tools, and skills."""
    t = theme or get_theme()
    resolved_cwd = cwd or os.getcwd()
    branch = git_branch if git_branch is not None else get_git_branch(resolved_cwd)
    cwd_display = f"{resolved_cwd} ({branch})" if branch else resolved_cwd
    session_display = (
        f"{session_id} · {session_title}" if session_title else session_id
    )
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
        f"[dim]{escape(cwd_display)}[/]",
        f"[dim]Session:[/] [{t.ui_label}]{escape(session_display)}[/]",
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
            f"[dim]{len(tools)} tools · {len(loaded_skills)} skills · /help cmds · !sh · @file · Ctrl+G editor[/]",
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


def _render_inline_diff(console: Console, diff_text: str, theme: ThemeConfig, max_lines: int = 20) -> None:
    lines = diff_text.splitlines()
    shown = lines[:max_lines]
    for ln in shown:
        if ln.startswith("+++") or ln.startswith("---"):
            console.print(f"    [dim]{escape(ln)}[/]")
        elif ln.startswith("@@"):
            console.print(f"    [{theme.ui_accent}]{escape(ln)}[/]")
        elif ln.startswith("+"):
            console.print(f"    [{theme.diff_add}]{escape(ln)}[/]")
        elif ln.startswith("-"):
            console.print(f"    [{theme.diff_del}]{escape(ln)}[/]")
        else:
            console.print(f"    [dim]{escape(ln)}[/]")
    if len(lines) > max_lines:
        console.print(f"    [dim]… {len(lines) - max_lines} more diff lines (/diff for full view)[/dim]")


def render_tool_event(
    console: Console,
    event: dict[str, Any],
    theme: ThemeConfig | None = None,
    display_mode: str = "compact",
) -> None:
    """Render Hermes-style tool activity row with duration, inline diffs, and verbosity modes."""
    if display_mode == "focus":
        return

    t = theme or get_theme()
    phase = event.get("phase")
    name = event.get("name", "tool")
    if phase == "start":
        args = event.get("arguments") or {}
        preview = ""
        if isinstance(args, dict) and args:
            first_val = str(next(iter(args.values()))).replace("\n", " ")
            if len(first_val) > 56:
                first_val = first_val[:53] + "..."
            preview = f" [dim]{escape(first_val)}[/]"
        console.print(f"  [dim]┊[/] [{t.ui_label}]⚙ {escape(name)}[/]{preview}")
    elif phase == "end":
        status = event.get("status", "unknown")
        dur = event.get("duration_s")
        dur_str = f" [dim]({dur:.2f}s)[/dim]" if isinstance(dur, (int, float)) else ""
        meta = event.get("metadata") if isinstance(event.get("metadata"), dict) else {}
        diff_badge = ""
        if meta and meta.get("diff"):
            diff_badge = (
                f" [{t.diff_add}]+{meta.get('added', 0)}[/] "
                f"[{t.diff_del}]-{meta.get('removed', 0)}[/]"
            )

        if status == "success":
            console.print(f"  [dim]┊[/] [{t.ui_ok}]✓ {escape(name)}[/]{dur_str}{diff_badge}")
        elif status == "rejected":
            console.print(f"  [dim]┊[/] [{t.ui_warn}]⊘ {escape(name)} (rejected)[/]{dur_str}")
        else:
            console.print(f"  [dim]┊[/] [{t.ui_error}]✗ {escape(name)} ({escape(status)})[/]{dur_str}")

        if meta and meta.get("diff"):
            _render_inline_diff(console, str(meta["diff"]), t)
        elif name == "todo_list" and meta and isinstance(meta.get("todos"), list):
            markers = {"completed": f"[{t.ui_ok}]✓[/]", "in_progress": f"[{t.ui_accent}]▸[/]", "pending": "[dim]·[/dim]", "cancelled": "[dim]~[/dim]"}
            for item in meta["todos"][:8]:
                m = markers.get(item.get("status", "pending"), "·")
                console.print(f"    {m} [dim]#{escape(str(item.get('id', '')))}[/] {escape(str(item.get('content', '')))}")

        if display_mode == "verbose":
            detail = (event.get("error") or event.get("output") or "").strip()
            if detail:
                clipped = detail[:600] + ("\n… [truncated]" if len(detail) > 600 else "")
                for ln in clipped.splitlines()[:12]:
                    console.print(f"    [dim]│ {escape(ln)}[/]")
        elif status != "success" and event.get("error"):
            err_line = str(event["error"]).strip().splitlines()[0][:140]
            console.print(f"    [{t.ui_error}]│ {escape(err_line)}[/]")


def render_turn_footer(
    console: Console,
    duration_s: float,
    tool_count: int = 0,
    tokens: int = 0,
    timestamp_str: str | None = None,
    theme: ThemeConfig | None = None,
) -> None:
    """Print subtle per-turn metrics footer beneath assistant response."""
    parts = [f"{duration_s:.1f}s"]
    if tool_count > 0:
        parts.append(f"{tool_count} tool{'s' if tool_count != 1 else ''}")
    if tokens > 0:
        parts.append(f"{format_token_count(tokens)} tok")
    if timestamp_str:
        parts.append(timestamp_str)
    console.print(f"[dim]─ {' · '.join(parts)}[/dim]")


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
        ("/session | /resume", "Interactive session switcher (Ctrl+N new, Del remove)"),
        ("/new [name]", "Start a fresh session (alias: /reset)"),
        ("/fork [name] | /rewind [N]", "Branch session or roll back to an earlier user prompt"),
        ("/rollback [list|diff|restore]", "Inspect diff or restore pre-edit file checkpoints"),
        ("/title [name] | /history [N]", "Set session title or show turns with timestamps"),
        ("/undo | /retry", "Remove last turn or undo & re-send last prompt"),
        ("/save [md|json] | /copy [N]", "Export session or copy assistant reply to clipboard"),
        ("/status | /context | /usage", "Inspect session state, token meter & API usage"),
        ("/compact [focus] | /btw <q>", "Summarize older turns or ask off-record side question"),
        ("/diff [--stat] | /doctor [--fix]", "Inspect workspace diff or diagnose Ivy install"),
        ("/init | /review | /goal", "Generate AGENTS.md, audit diff via subagent, or verification loop"),
        ("/reasoning [mode] | /verbose [mode]", "Control <think> display or tool detail (compact/verbose/focus)"),
        ("/focus [on|off] | /yolo", "Distraction-free view or session auto-approve toggle"),
        ("/todo [clear] | /profile [list|set]", "View session task checklist or long-term preferences"),
        ("/cron ... | /mcp [list|reload]", "Manage persisted cron jobs or .mcp.json stdio servers"),
        ("/skills list|inspect|remove|reflect", "Manage and synthesize skills"),
        ("/memory <query>", "Search conversation history via SQLite FTS5"),
        ("/setup [base] [key] [model]", "Configure API endpoint/key & save to .env"),
        ("/model [name] | /theme [name]", "Pick/switch active model or TUI skin"),
        ("/palette (Ctrl+P) | /prompt (Ctrl+G)", "Fuzzy command palette or compose in $EDITOR"),
        ("Ctrl+S  |  !<cmd>  |  @file/@diff", "Stash/restore draft · local shell (0 tok) · attach context"),
        ("/clear | /exit", "Clear current session or exit Ivy"),
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
    """Display saved sessions in a Rich table, marking active session and quick # index."""
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
    table.add_column("#", justify="right", style="dim", width=3)
    table.add_column("", width=2)
    table.add_column("Session ID", style=f"bold {t.ui_label}")
    table.add_column("Title")
    table.add_column("Msgs", justify="right")
    table.add_column("Last Updated", style="dim")
    for idx, s in enumerate(sessions, start=1):
        sid = s.get("session_id", "")
        marker = f"[{t.ui_ok}]●[/]" if sid == current_session_id else ""
        updated = format_relative_time(s.get("updated_at")) or str(s.get("updated_at") or "")[:16]
        title = str(s.get("title") or "").strip()[:38]
        table.add_row(
            str(idx),
            marker,
            sid,
            escape(title) if title else "[dim](new session)[/dim]",
            str(s.get("message_count", 0)),
            updated,
        )
    console.print(table)
    console.print("[dim]Tip: /session <id> or /session <#> to switch · /new [name] to start fresh[/dim]")


def render_context_panel(
    console: Console,
    breakdown: dict[str, Any],
    model: str,
    theme: ThemeConfig | None = None,
) -> None:
    """Render visual 20-cell context meter and token allocation breakdown."""
    t = theme or get_theme()
    pct = float(breakdown.get("percent", 0.0))
    used = int(breakdown.get("used_tokens", 0))
    max_tok = int(breakdown.get("max_tokens", 128_000))
    cells = 20
    filled = max(0, min(cells, round((pct / 100.0) * cells)))
    bar_color = (
        t.ctx_ok if pct < 50 else t.ctx_warn if pct < 80 else t.ctx_high if pct < 95 else t.ctx_crit
    )
    bar = f"[{bar_color}]{'█' * filled}[/][dim]{'░' * (cells - filled)}[/]"
    prefix = "~" if breakdown.get("estimated", True) else ""

    table = Table(
        title=f"[{t.banner_title}]Context Window · {escape(model)}[/]",
        box=box.ROUNDED,
        border_style=t.banner_border,
    )
    table.add_column("Category", style=f"bold {t.ui_label}")
    table.add_column("Tokens", justify="right")
    table.add_column("Share", justify="right", style="dim")

    for label, key in (
        ("System prompt", "system_tokens"),
        ("Tool schemas", "tool_tokens"),
        ("Conversation history", "history_tokens"),
        ("Free remaining", "free_tokens"),
    ):
        val = int(breakdown.get(key, 0))
        share = (val / max_tok * 100.0) if max_tok else 0.0
        table.add_row(label, f"{prefix}{format_token_count(val)}", f"{share:.1f}%")

    console.print(f"\n  {bar}  [bold {bar_color}]{pct:.1f}%[/] ({prefix}{format_token_count(used)} / {format_token_count(max_tok)} tok)")
    console.print(table)


def render_usage_table(
    console: Console,
    usage_stats: dict[str, int],
    model: str,
    compressions: int = 0,
    theme: ThemeConfig | None = None,
) -> None:
    """Render session API token usage summary."""
    t = theme or get_theme()
    table = Table(
        title=f"[{t.banner_title}]Session Usage · {escape(model)}[/]",
        box=box.ROUNDED,
        border_style=t.banner_border,
    )
    table.add_column("Metric", style=f"bold {t.ui_label}")
    table.add_column("Value", justify="right")
    table.add_row("API calls", str(usage_stats.get("api_calls", 0)))
    table.add_row("Prompt tokens", format_token_count(usage_stats.get("prompt_tokens", 0)))
    table.add_row("Completion tokens", format_token_count(usage_stats.get("completion_tokens", 0)))
    table.add_row("Total tokens", format_token_count(usage_stats.get("total_tokens", 0)))
    table.add_row("Compactions", str(compressions))
    console.print(table)


def render_status_panel(
    console: Console,
    status_info: dict[str, Any],
    theme: ThemeConfig | None = None,
) -> None:
    """Render comprehensive active session status panel."""
    t = theme or get_theme()
    lines = [
        f"[dim]Session:[/]     [{t.ui_label}]{escape(str(status_info.get('session_id', '')))}[/]"
        + (f"  [dim]({escape(str(status_info['title']))})[/dim]" if status_info.get("title") else ""),
        f"[dim]Model:[/]       [bold]{escape(str(status_info.get('model', '')))}[/bold] ({escape(str(status_info.get('provider', 'openai')))})",
        f"[dim]Working Dir:[/] {escape(str(status_info.get('cwd', '')))}",
        f"[dim]Context:[/]     ~{format_token_count(int(status_info.get('ctx_used', 0)))} / {format_token_count(int(status_info.get('ctx_max', 128000)))} ({status_info.get('ctx_pct', 0):.1f}%) · {status_info.get('message_count', 0)} messages",
        f"[dim]Display:[/]     {escape(str(status_info.get('display_mode', 'compact')))} · theme={escape(t.name)}",
        f"[dim]Approvals:[/]   {'[bold red]YOLO (session auto-approve)[/]' if status_info.get('yolo') else 'interactive guard'}",
        f"[dim]Session Tok:[/] {format_token_count(int(status_info.get('total_tokens', 0)))} across {status_info.get('api_calls', 0)} API calls",
    ]
    console.print(
        Panel(
            "\n".join(lines),
            title=f"[bold {t.banner_title}]{t.glyph} Ivy Status[/]",
            border_style=t.banner_border,
            box=box.ROUNDED,
            expand=False,
        )
    )


def render_history_view(
    console: Console,
    messages: list[ChatMessage],
    limit: int = 20,
    theme: ThemeConfig | None = None,
) -> None:
    """Display formatted recent messages with timestamps."""
    t = theme or get_theme()
    slice_msgs = messages[-limit:] if limit > 0 else messages
    if not slice_msgs:
        console.print(f"[{t.ui_warn}]No messages in this session yet.[/]")
        return
    for msg in slice_msgs:
        ts = msg.timestamp.strftime("%H:%M")
        body = (msg.content or "").strip()
        if msg.role == Role.USER:
            console.print(f"[dim][{ts}][/dim] [bold {t.ui_accent}]❯ You:[/] {escape(body)}")
        elif msg.role == Role.ASSISTANT:
            preview = body[:240] + ("…" if len(body) > 240 else "")
            if msg.tool_calls and not preview:
                preview = "[tool calls: " + ", ".join(tc.function_name for tc in msg.tool_calls) + "]"
            console.print(f"[dim][{ts}][/dim] [bold {t.ui_label}]{t.glyph} Ivy:[/] {escape(preview)}")
        elif msg.role == Role.TOOL:
            first = body.splitlines()[0][:100] if body else ""
            console.print(f"  [dim][{ts}] ⚙ {escape(msg.name or 'tool')}: {escape(first)}[/dim]")


def render_diff_output(
    console: Console, diff_text: str, theme: ThemeConfig | None = None
) -> None:
    """Render workspace git diff cleanly with semantic colors."""
    t = theme or get_theme()
    if not diff_text.strip():
        console.print("[dim]Working tree has no uncommitted diff.[/dim]")
        return
    lines = diff_text.splitlines()
    for ln in lines[:120]:
        if ln.startswith("diff --git") or ln.startswith("index "):
            console.print(f"[bold {t.ui_label}]{escape(ln)}[/]")
        elif ln.startswith("+++") or ln.startswith("---"):
            console.print(f"[dim]{escape(ln)}[/]")
        elif ln.startswith("@@"):
            console.print(f"[{t.ui_accent}]{escape(ln)}[/]")
        elif ln.startswith("+"):
            console.print(f"[{t.diff_add}]{escape(ln)}[/]")
        elif ln.startswith("-"):
            console.print(f"[{t.diff_del}]{escape(ln)}[/]")
        else:
            console.print(escape(ln))
    if len(lines) > 120:
        console.print(f"[dim]… truncated ({len(lines) - 120} more lines)[/dim]")


def render_todo_panel(
    console: Console,
    todos: list[dict[str, str]],
    theme: ThemeConfig | None = None,
) -> None:
    """Display session task checklist in a compact Rich Panel."""
    t = theme or get_theme()
    if not todos:
        console.print("[dim]Task checklist is empty.[/dim]")
        return
    markers = {
        "completed": f"[{t.ui_ok}]✓[/]",
        "in_progress": f"[{t.ui_accent}]▸[/]",
        "pending": "[dim]·[/dim]",
        "cancelled": "[dim]~[/dim]",
    }
    lines = [
        f"{markers.get(item.get('status', 'pending'), '·')} "
        f"[bold]#{escape(item.get('id', ''))}[/bold] {escape(item.get('content', ''))} "
        f"[dim]({escape(item.get('status', 'pending'))})[/dim]"
        for item in todos
    ]
    console.print(
        Panel(
            "\n".join(lines),
            title=f"[bold {t.banner_title}]{t.glyph} Task Checklist[/]",
            border_style=t.banner_border,
            box=box.ROUNDED,
            expand=False,
        )
    )


def render_profile_table(
    console: Console,
    profile: dict[str, str],
    theme: ThemeConfig | None = None,
) -> None:
    """Display persistent user profile preferences."""
    t = theme or get_theme()
    if not profile:
        console.print("[dim]No saved user preferences yet (/profile set <key> <value>).[/dim]")
        return
    table = Table(
        title=f"[{t.banner_title}]User Profile Memory[/]",
        box=box.ROUNDED,
        border_style=t.banner_border,
    )
    table.add_column("Key", style=f"bold {t.ui_label}")
    table.add_column("Value")
    for k, v in sorted(profile.items()):
        table.add_row(escape(k), escape(v))
    console.print(table)


def _format_checkpoint_path(raw_path: str, max_len: int = 52) -> str:
    """Shorten checkpoint paths relative to cwd while keeping the trailing filename intact."""
    shown = str(raw_path)
    try:
        p = Path(shown)
        if p.is_absolute():
            shown = str(p.resolve().relative_to(Path.cwd().resolve()))
    except (OSError, ValueError):
        pass
    if len(shown) <= max_len:
        return shown
    tail_len = max(12, max_len - 17)
    return f"{shown[:16]}…{shown[-tail_len:]}"


def render_checkpoints_table(
    console: Console,
    checkpoints: list[Any],
    theme: ThemeConfig | None = None,
) -> None:
    """Display pre-edit file checkpoints for /rollback."""
    t = theme or get_theme()
    if not checkpoints:
        console.print("[dim]No file checkpoints recorded in this session yet.[/dim]")
        return
    table = Table(
        title=f"[{t.banner_title}]File Checkpoints (/rollback restore <id>)[/]",
        box=box.ROUNDED,
        border_style=t.banner_border,
    )
    table.add_column("ID", justify="right", style=f"bold {t.ui_label}", width=4)
    table.add_column("Tool", style=t.ui_accent)
    table.add_column("Action")
    table.add_column("Path")
    table.add_column("When", style="dim")
    for cp in checkpoints[:20]:
        kind = "modified" if cp.existed_before else "created"
        table.add_row(
            str(cp.id),
            cp.tool_name,
            kind,
            escape(_format_checkpoint_path(cp.display_path)),
            format_relative_time(cp.timestamp),
        )
    console.print(table)


def render_mcp_table(
    console: Console,
    servers: list[dict[str, Any]],
    theme: ThemeConfig | None = None,
) -> None:
    """Display connected/errored stdio MCP servers."""
    t = theme or get_theme()
    if not servers:
        console.print("[dim]No MCP servers loaded (add `.mcp.json` and run `/mcp reload`).[/dim]")
        return
    table = Table(
        title=f"[{t.banner_title}]MCP Servers (.mcp.json)[/]",
        box=box.ROUNDED,
        border_style=t.banner_border,
    )
    table.add_column("Server", style=f"bold {t.ui_label}")
    table.add_column("Status")
    table.add_column("Tools", justify="right")
    table.add_column("Note", style="dim")
    for s in servers:
        st = s.get("status", "")
        st_fmt = f"[{t.ui_ok}]{st}[/]" if st == "connected" else f"[{t.ui_error}]{st}[/]"
        table.add_row(
            escape(str(s.get("name", ""))),
            st_fmt,
            str(s.get("tools", 0)),
            escape(str(s.get("error", ""))),
        )
    console.print(table)


def render_processes_table(
    console: Console,
    processes: list[dict[str, Any]],
    theme: ThemeConfig | None = None,
) -> None:
    """Display tracked background shell processes for /ps."""
    t = theme or get_theme()
    if not processes:
        console.print("[dim]No background processes started in this session.[/dim]")
        return
    table = Table(
        title=f"[{t.banner_title}]Background Processes (/ps log <id> | /ps stop <id|all>)[/]",
        box=box.ROUNDED,
        border_style=t.banner_border,
    )
    table.add_column("ID", style=f"bold {t.ui_label}")
    table.add_column("PID", justify="right")
    table.add_column("Status")
    table.add_column("Elapsed", justify="right", style="dim")
    table.add_column("Command")
    for p in processes:
        st = str(p.get("status", ""))
        st_fmt = f"[{t.ui_ok}]{st}[/]" if st == "running" else f"[dim]{st} ({p.get('exit_code')})[/]"
        cmd_raw = str(p.get("command", "")).strip().replace("\n", " ")
        cmd_short = (cmd_raw[:54] + "…") if len(cmd_raw) > 55 else cmd_raw
        table.add_row(
            escape(str(p.get("process_id", ""))),
            str(p.get("pid") or "—"),
            st_fmt,
            f"{p.get('elapsed_s', 0)}s",
            escape(cmd_short),
        )
    console.print(table)
