"""Prompt-toolkit input configuration, bottom status bar, and nested slash autocomplete."""

from __future__ import annotations

import shutil
from typing import TYPE_CHECKING, Callable, Iterable

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.styles import Style

from ivy.tui.theme import ThemeConfig, get_theme, list_themes

if TYPE_CHECKING:
    from ivy.skills.engine import SkillEngine

SLASH_COMMANDS = [
    "/setup",
    "/model",
    "/session",
    "/new",
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
    "/model": "Switch or inspect active LLM model (persisted to .env)",
    "/session": "List saved sessions or switch active session",
    "/new": "Start a new fresh session",
    "/skills": "List, inspect, remove, or reflect skills",
    "/gateway": "Show Telegram/Discord gateway status",
    "/cron": "List scheduled cron tasks",
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
    },
}


class SlashCompleter(Completer):
    """Context-aware nested completer for Ivy slash commands and sub-arguments."""

    def __init__(self, skill_engine: SkillEngine | None = None) -> None:
        self.words = list(SLASH_COMMANDS)
        self.skill_engine = skill_engine

    def get_completions(
        self, document: Document, complete_event: CompleteEvent
    ) -> Iterable[Completion]:
        text = document.text_before_cursor.lstrip()
        if not text.startswith("/"):
            return

        ends_with_space = document.text_before_cursor.endswith(" ")
        tokens = text.split()

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

        # Level 2: `/skills|gateway|cron <subcommand>`
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
) -> SlashCompleter:
    """Create nested autocomplete completer for slash commands."""
    return SlashCompleter(skill_engine=skill_engine)


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


def build_bottom_toolbar(
    model: str,
    session_id: str,
    tool_count: int = 0,
    skill_count: int = 0,
    theme: ThemeConfig | None = None,
    width: int | None = None,
) -> FormattedText:
    """Render Hermes-style two-row clean footer (top rule + split left/right status)."""
    t = theme or get_theme()
    cols = max(40, width or shutil.get_terminal_size((100, 24)).columns)

    left_parts: list[tuple[str, str]] = [
        ("class:bottom-toolbar.strong", f" {t.glyph} Ivy"),
        ("class:bottom-toolbar.dim", " · "),
        ("class:bottom-toolbar.text", model),
        ("class:bottom-toolbar.dim", " · "),
        ("class:bottom-toolbar.text", session_id),
        ("class:bottom-toolbar.dim", f" · {tool_count} tools · {skill_count} skills"),
    ]

    right_parts: list[tuple[str, str]] = [
        ("class:bottom-toolbar.strong", "/setup"),
        ("class:bottom-toolbar.dim", " key · "),
        ("class:bottom-toolbar.strong", "/help"),
        ("class:bottom-toolbar.dim", " cmds · "),
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


def create_prompt_session(
    theme: ThemeConfig | None = None,
    bottom_toolbar: Callable[[], FormattedText] | None = None,
    skill_engine: SkillEngine | None = None,
) -> PromptSession[str]:
    """Create an interactive prompt_toolkit session with themed chrome and completion."""
    return PromptSession(
        history=InMemoryHistory(),
        completer=build_slash_completer(skill_engine=skill_engine),
        complete_while_typing=True,
        style=build_prompt_style(theme),
        bottom_toolbar=bottom_toolbar,
    )
