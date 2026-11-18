"""Prompt-toolkit input configuration, bottom status bar, @mentions, and slash autocomplete."""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.history import FileHistory, History, InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style

from ivy.tui.helpers import format_token_count, list_workspace_files
from ivy.tui.theme import ThemeConfig, get_theme, list_themes

if TYPE_CHECKING:
    from ivy.skills.engine import SkillEngine

SLASH_COMMANDS = [
    "/setup",
    "/doctor",
    "/model",
    "/session",
    "/resume",
    "/new",
    "/fork",
    "/rewind",
    "/rollback",
    "/title",
    "/undo",
    "/retry",
    "/history",
    "/save",
    "/copy",
    "/status",
    "/context",
    "/usage",
    "/compact",
    "/btw",
    "/diff",
    "/init",
    "/review",
    "/goal",
    "/reasoning",
    "/verbose",
    "/focus",
    "/yolo",
    "/todo",
    "/profile",
    "/mcp",
    "/prompt",
    "/palette",
    "/skills",
    "/gateway",
    "/cron",
    "/memory",
    "/theme",
    "/clear",
    "/help",
    "/exit",
]

SLASH_COMMAND_META: dict[str, str] = {
    "/setup": "Configure API base URL, API key, and default model (.env)",
    "/doctor": "Diagnose runtime, endpoint, DB, skills & tools (--fix)",
    "/model": "Switch or inspect active LLM model (persisted to .env)",
    "/session": "List or pick saved sessions interactively",
    "/resume": "Alias for /session — resume saved session",
    "/new": "Start a new fresh session",
    "/fork": "Fork current session into a new branch",
    "/rewind": "Pick an earlier user turn to roll back / re-run / fork",
    "/rollback": "Inspect diff or restore pre-edit file checkpoints",
    "/title": "Set or view current session title",
    "/undo": "Undo last user turn + assistant reply",
    "/retry": "Undo last turn and re-send last prompt",
    "/history": "Display recent conversation turns with timestamps",
    "/save": "Export conversation transcript (md or json)",
    "/copy": "Copy last (or Nth-last) assistant reply to clipboard",
    "/status": "Show session, context window, model & approval state",
    "/context": "Inspect token breakdown across prompt, tools & history",
    "/usage": "Show session API calls and token totals",
    "/compact": "Summarize older turns via LLM to save context",
    "/btw": "Ask a quick side question without polluting session history",
    "/diff": "Inspect git diff in current workspace",
    "/init": "Generate/update AGENTS.md project guide in current workspace",
    "/review": "Run isolated subagent code review on recent changes",
    "/goal": "Autonomous execution loop with optional verification --gate",
    "/reasoning": "Control <think> display (collapsed | stream | hide | show)",
    "/verbose": "Set tool output mode (compact | verbose | focus)",
    "/focus": "Toggle distraction-free view (hides tool lines)",
    "/yolo": "Toggle session dangerous-command auto-approval",
    "/todo": "Inspect or clear current session task checklist",
    "/profile": "View or edit persistent cross-session user preferences",
    "/mcp": "Inspect or reload .mcp.json stdio MCP servers",
    "/prompt": "Compose multiline prompt in $EDITOR (Ctrl+G)",
    "/palette": "Open interactive command palette (Ctrl+P)",
    "/skills": "List, inspect, remove, or reflect skills",
    "/gateway": "Show Telegram/Discord gateway status",
    "/cron": "Manage scheduled cron tasks (list|add|pause|resume|remove)",
    "/memory": "Search past sessions via SQLite FTS5",
    "/theme": "Switch TUI skin (ivy, hermes, ocean)",
    "/clear": "Clear current session history",
    "/help": "Show command reference",
    "/exit": "Exit Ivy",
}

_SUBCOMMAND_META: dict[str, dict[str, str]] = {
    "/session": {
        "list": "List saved sessions",
    },
    "/resume": {
        "list": "List saved sessions",
    },
    "/doctor": {
        "--fix": "Create missing .env / skills directory automatically",
        "--no-live": "Skip HTTP endpoint connectivity probe",
    },
    "/rollback": {
        "list": "List pre-edit file checkpoints",
        "diff": "Show unified diff for checkpoint",
        "restore": "Restore file from pre-edit checkpoint",
    },
    "/mcp": {
        "list": "Show configured/connected MCP servers",
        "reload": "Reload .mcp.json and connect servers",
    },
    "/reasoning": {
        "collapsed": "Hide <think> tags and show compact duration/line count",
        "stream": "Show <think> text dimmed during streaming",
        "hide": "Suppress <think> summary completely",
        "show": "Inspect the last hidden reasoning block",
    },
    "/todo": {
        "list": "Show active task checklist",
        "clear": "Clear checklist for this session",
    },
    "/profile": {
        "list": "List saved cross-session preferences",
        "set": "Set preference: /profile set <key> <value>",
        "clear": "Clear all saved preferences",
    },
    "/verbose": {
        "compact": "Concise tool lines + inline diffs (default)",
        "verbose": "Show tool stdout/stderr snippets",
        "focus": "Hide tool chatter completely",
    },
    "/focus": {
        "on": "Hide tool lines during responses",
        "off": "Show tool lines (compact)",
        "status": "Show current focus state",
    },
    "/diff": {
        "--stat": "Show diff file summary only",
        "staged": "Show staged git diff",
    },
    "/save": {
        "md": "Export transcript as Markdown",
        "json": "Export transcript as JSON",
    },
    "/skills": {
        "list": "List all loaded skills",
        "inspect": "Inspect a skill's instructions",
        "remove": "Delete a skill",
        "reflect": "Synthesize a skill from current session",
    },
    "/gateway": {
        "start": "Start Telegram/Discord daemon",
        "status": "Show gateway configuration status",
    },
    "/cron": {
        "list": "List scheduled cron tasks",
        "add": "Add cron job: /cron add <name> \"<expr>\" <prompt>",
        "pause": "Pause a cron job by name",
        "resume": "Resume a paused cron job",
        "remove": "Delete a cron job by name",
    },
}


class SlashCompleter(Completer):
    """Context-aware nested completer for /commands and @file/@diff mentions."""

    def __init__(
        self,
        skill_engine: SkillEngine | None = None,
        cwd: str | None = None,
        session_provider: Callable[[], list[dict[str, Any]]] | None = None,
    ) -> None:
        self.words = list(SLASH_COMMANDS)
        self.skill_engine = skill_engine
        self.cwd = cwd
        self.session_provider = session_provider
        self._file_cache: list[str] = []
        self._file_cache_ts: float = 0.0

    def _workspace_files(self) -> list[str]:
        now = time.monotonic()
        if not self._file_cache or (now - self._file_cache_ts) > 5.0:
            self._file_cache = list_workspace_files(self.cwd, limit=200)
            self._file_cache_ts = now
        return self._file_cache

    def _complete_at_mentions(self, before_cursor: str) -> Iterable[Completion]:
        """Suggest @diff, @staged, and workspace files when typing an @mention."""
        word = before_cursor.rsplit(None, 1)[-1] if before_cursor.strip() else ""
        if not word.startswith("@"):
            return
        query = word[1:].lower()
        built_ins = [
            ("@diff", "Attach current git diff"),
            ("@staged", "Attach staged git diff"),
        ]
        for token, desc in built_ins:
            if token[1:].startswith(query):
                yield Completion(
                    token,
                    start_position=-len(word),
                    display_meta=desc,
                )

        matched: list[str] = []
        for rel in self._workspace_files():
            low = rel.lower()
            if not query or query in low:
                matched.append(rel)
        matched.sort(key=lambda p: (0 if p.lower().startswith(query) else 1, len(p), p))
        for rel in matched[:15]:
            yield Completion(
                f"@{rel}",
                start_position=-len(word),
                display_meta="workspace file",
            )

    def get_completions(
        self, document: Document, complete_event: CompleteEvent
    ) -> Iterable[Completion]:
        before = document.text_before_cursor
        stripped = before.lstrip()

        if not stripped.startswith("/"):
            yield from self._complete_at_mentions(before)
            return

        ends_with_space = before.endswith(" ")
        tokens = stripped.split()

        # Level 1: top-level `/command`
        if len(tokens) == 1 and not ends_with_space:
            prefix = tokens[0].lower()
            for cmd in self.words:
                if cmd.startswith(prefix):
                    yield Completion(
                        cmd,
                        start_position=-len(tokens[0]),
                        display_meta=SLASH_COMMAND_META.get(cmd, ""),
                    )
            return

        cmd = tokens[0].lower()
        partial = "" if ends_with_space else tokens[-1]
        arg_pos = 1 if ends_with_space else len(tokens) - 1

        # Level 2: `/theme <name>`
        if cmd == "/theme" and arg_pos == 1:
            for th in list_themes():
                if th.name.startswith(partial.lower()):
                    yield Completion(
                        th.name,
                        start_position=-len(partial),
                        display_meta=th.description,
                    )
            return

        # Level 2: `/session <id>` or `list`
        if cmd in ("/session", "/resume") and arg_pos == 1:
            if "list".startswith(partial.lower()):
                yield Completion("list", start_position=-len(partial), display_meta="List saved sessions")
            if self.session_provider is not None:
                for sess in self.session_provider():
                    sid = str(sess.get("session_id") or "")
                    if not sid or not sid.lower().startswith(partial.lower()):
                        continue
                    title = (sess.get("title") or "").strip()
                    msgs = sess.get("message_count", 0)
                    meta = f"{title} ({msgs} msgs)" if title else f"{msgs} msgs"
                    yield Completion(sid, start_position=-len(partial), display_meta=meta)
            return

        # Level 2: static subcommands
        if cmd in _SUBCOMMAND_META and arg_pos == 1:
            for sub, meta in _SUBCOMMAND_META[cmd].items():
                if sub.startswith(partial.lower()):
                    yield Completion(
                        sub,
                        start_position=-len(partial),
                        display_meta=meta,
                    )
            return

        # Level 3: `/skills inspect|remove <skill-name>`
        if (
            cmd == "/skills"
            and len(tokens) >= 2
            and tokens[1].lower() in ("inspect", "remove")
            and arg_pos == 2
            and self.skill_engine is not None
        ):
            for skill in self.skill_engine.list_skills():
                name = skill["name"]
                if name.lower().startswith(partial.lower()):
                    yield Completion(
                        name,
                        start_position=-len(partial),
                        display_meta=skill.get("description", ""),
                    )


def build_slash_completer(
    skill_engine: SkillEngine | None = None,
    cwd: str | None = None,
    session_provider: Callable[[], list[dict[str, Any]]] | None = None,
) -> SlashCompleter:
    """Create nested autocomplete completer for slash commands and @mentions."""
    return SlashCompleter(skill_engine=skill_engine, cwd=cwd, session_provider=session_provider)


def build_prompt_style(theme: ThemeConfig | None = None) -> Style:
    """Build prompt_toolkit style matching the active TUI theme (noreverse footer)."""
    t = theme or get_theme()
    return Style.from_dict(
        {
            "prompt": f"bold {t.ui_accent}",
            "bottom-toolbar": f"noreverse bg:default {t.status_bar_text}",
            "bottom-toolbar.text": f"noreverse bg:default {t.status_bar_text}",
            "bottom-toolbar.rule": f"noreverse bg:default {t.status_bar_dim}",
            "bottom-toolbar.strong": f"noreverse bg:default bold {t.status_bar_strong}",
            "bottom-toolbar.accent": f"noreverse bg:default {t.ui_accent}",
            "bottom-toolbar.dim": f"noreverse bg:default {t.status_bar_dim}",
            "bottom-toolbar.ctx-ok": f"noreverse bg:default {t.ctx_ok}",
            "bottom-toolbar.ctx-warn": f"noreverse bg:default {t.ctx_warn}",
            "bottom-toolbar.ctx-high": f"noreverse bg:default {t.ctx_high}",
            "bottom-toolbar.ctx-crit": f"noreverse bg:default bold {t.ctx_crit}",
            "bottom-toolbar.yolo": f"noreverse bg:default bold {t.ui_error}",
            "bottom-toolbar.focus": f"noreverse bg:default bold {t.ui_accent}",
            "completion-menu": f"noreverse bg:{t.completion_bg} {t.status_bar_text}",
            "completion-menu.completion.current": (
                f"noreverse bg:{t.completion_current_bg} bold {t.status_bar_strong}"
            ),
            "completion-menu.meta.completion": f"noreverse bg:{t.completion_bg} {t.status_bar_dim}",
            "completion-menu.meta.completion.current": (
                f"noreverse bg:{t.completion_current_bg} {t.status_bar_text}"
            ),
        }
    )


def build_prompt_key_bindings() -> KeyBindings:
    """Build multiline, $EDITOR, palette (Ctrl+P), and stash (Ctrl+S) keybindings."""
    kb = KeyBindings()

    @kb.add("escape", "enter")
    @kb.add("c-j")
    def _insert_newline(event) -> None:
        event.current_buffer.insert_text("\n")

    @kb.add("c-g")
    def _open_external_editor(event) -> None:
        buf = event.current_buffer
        event.app.exit(result=f"__IVY_EDITOR__:{buf.text}")

    @kb.add("c-p")
    def _open_palette(event) -> None:
        event.app.exit(result="__IVY_PALETTE__")

    @kb.add("c-s")
    def _stash_draft(event) -> None:
        buf = event.current_buffer
        event.app.exit(result=f"__IVY_STASH__:{buf.text}")

    return kb


def _ctx_style_class(percent: float) -> str:
    if percent < 50:
        return "class:bottom-toolbar.ctx-ok"
    if percent < 80:
        return "class:bottom-toolbar.ctx-warn"
    if percent < 95:
        return "class:bottom-toolbar.ctx-high"
    return "class:bottom-toolbar.ctx-crit"


def build_bottom_toolbar(
    model: str,
    session_id: str,
    tool_count: int = 0,
    skill_count: int = 0,
    theme: ThemeConfig | None = None,
    width: int | None = None,
    ctx_percent: float | None = None,
    ctx_estimated: bool = True,
    total_tokens: int = 0,
    last_duration_s: float | None = None,
    yolo: bool = False,
    focus_mode: bool = False,
    session_title: str | None = None,
    git_branch: str | None = None,
) -> FormattedText:
    """Render Hermes-style responsive two-row footer with live context meter & badges."""
    t = theme or get_theme()
    cols = max(40, width or shutil.get_terminal_size((100, 24)).columns)

    session_label = f"{session_id} ({session_title[:18]})" if session_title else session_id

    left_parts: list[tuple[str, str]] = [
        ("class:bottom-toolbar.strong", f" {t.glyph} Ivy"),
        ("class:bottom-toolbar.dim", " · "),
        ("class:bottom-toolbar.text", model),
        ("class:bottom-toolbar.dim", " · "),
        ("class:bottom-toolbar.text", session_label),
    ]

    if git_branch and cols >= 90:
        left_parts.extend([
            ("class:bottom-toolbar.dim", f" ({git_branch[:16]})"),
        ])

    if ctx_percent is not None:
        pct = max(0.0, min(100.0, float(ctx_percent)))
        cells = 8
        filled = max(0, min(cells, round((pct / 100.0) * cells)))
        bar = f"{'■' * filled}{'·' * (cells - filled)}"
        prefix = "~" if ctx_estimated else ""
        left_parts.extend([
            ("class:bottom-toolbar.dim", " · "),
            (_ctx_style_class(pct), f"[{bar}] {prefix}{pct:.0f}%"),
        ])
    else:
        left_parts.append(
            ("class:bottom-toolbar.dim", f" · {tool_count} tools · {skill_count} skills")
        )

    if total_tokens > 0 and cols >= 85:
        left_parts.append(("class:bottom-toolbar.dim", f" · {format_token_count(total_tokens)} tok"))

    if last_duration_s is not None and cols >= 95:
        left_parts.append(("class:bottom-toolbar.dim", f" · {last_duration_s:.1f}s"))

    if yolo:
        left_parts.extend([("class:bottom-toolbar.dim", " · "), ("class:bottom-toolbar.yolo", "YOLO")])

    if focus_mode:
        left_parts.extend([("class:bottom-toolbar.dim", " · "), ("class:bottom-toolbar.focus", "FOCUS")])

    right_parts: list[tuple[str, str]] = [
        ("class:bottom-toolbar.strong", "!sh"),
        ("class:bottom-toolbar.dim", " · "),
        ("class:bottom-toolbar.strong", "@file"),
        ("class:bottom-toolbar.dim", " · "),
        ("class:bottom-toolbar.strong", "/help"),
        ("class:bottom-toolbar.dim", " · "),
        ("class:bottom-toolbar.strong", "Ctrl+C"),
        ("class:bottom-toolbar.dim", " cancel "),
    ]

    left_len = sum(len(txt) for _, txt in left_parts)
    right_len = sum(len(txt) for _, txt in right_parts)

    fragments: list[tuple[str, str]] = [
        ("class:bottom-toolbar.rule", "─" * cols),
        ("", "\n"),
    ]
    if left_len + right_len + 2 <= cols:
        pad = " " * (cols - left_len - right_len)
        fragments.extend(left_parts)
        fragments.append(("class:bottom-toolbar.dim", pad))
        fragments.extend(right_parts)
    else:
        fragments.extend(left_parts)

    return FormattedText(fragments)


def _resolve_history(history_path: str | Path | None) -> History:
    if not history_path:
        return InMemoryHistory()
    try:
        path = Path(history_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return FileHistory(str(path))
    except OSError:
        return InMemoryHistory()


def create_prompt_session(
    theme: ThemeConfig | None = None,
    bottom_toolbar: Callable[[], FormattedText] | None = None,
    skill_engine: SkillEngine | None = None,
    history_path: str | Path | None = None,
    cwd: str | None = None,
    session_provider: Callable[[], list[dict[str, Any]]] | None = None,
) -> PromptSession[str]:
    """Create interactive prompt_toolkit session with FileHistory, bindings & completion."""
    return PromptSession(
        history=_resolve_history(history_path),
        auto_suggest=AutoSuggestFromHistory(),
        completer=build_slash_completer(
            skill_engine=skill_engine, cwd=cwd, session_provider=session_provider
        ),
        complete_while_typing=True,
        key_bindings=build_prompt_key_bindings(),
        style=build_prompt_style(theme),
        bottom_toolbar=bottom_toolbar,
    )
