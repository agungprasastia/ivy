"""CLI parser, interactive TUI loop, and gateway entrypoint (`ivy`)."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import platform
import shlex
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable, Sequence
from uuid import uuid4

from prompt_toolkit.formatted_text import FormattedText, HTML
from rich import box as rich_box
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.markup import escape as rich_escape
from rich.panel import Panel as RichPanel

from ivy.config import IvyConfig, load_config, update_env_file
from ivy.core.agent import IvyAgent
from ivy.core.llm import LLMClient
from ivy.core.subagent import SubagentRunner
from ivy.core.types import ChatMessage, Role
from ivy.cron.scheduler import CronScheduler
from ivy.cron.task import ScheduledTask
from ivy.gateway.server import GatewayServer
from ivy.memory.context import ContextWindowManager, compute_context_breakdown
from ivy.memory.store import MemoryStore
from ivy.skills.engine import SkillEngine
from ivy.skills.generator import SkillGenerator
from ivy.skills.tool import ReadSkillTool
from ivy.tools.approval import SafetyApprovalPolicy
from ivy.tools.checkpoint import FileCheckpointStore
from ivy.tools.clarify import ClarifyTool
from ivy.tools.cron_tool import CronManageTool
from ivy.tools.delegate import DelegateTaskTool
from ivy.tools.files import FileTools
from ivy.tools.mcp import MCPManager
from ivy.tools.profile import RememberPreferenceTool
from ivy.tools.registry import ToolRegistry
from ivy.tools.runner import CodeRunnerTool
from ivy.tools.session_search import SessionSearchTool
from ivy.tools.skill_manager import SkillManagerTool
from ivy.tools.terminal import TerminalTool
from ivy.tools.todo import TodoListTool, TodoStore
from ivy.tools.web import WebFetchTool, WebSearchTool
from ivy.tui.doctor import render_doctor_report, run_doctor_checks
from ivy.tui.helpers import (
    copy_to_clipboard,
    expand_at_mentions,
    export_session_json,
    export_session_markdown,
    extract_thinking_blocks,
    format_relative_time,
    format_token_count,
    get_git_branch,
    resolve_slash_command,
    run_git_diff,
)
from ivy.tui.picker import PickerOption, run_interactive_picker
from ivy.tui.prompt import (
    SLASH_COMMAND_META,
    SLASH_COMMANDS,
    build_bottom_toolbar,
    build_prompt_style,
    create_prompt_session,
)
from ivy.tui.theme import ThemeConfig, get_theme, list_themes
from ivy.tui.views import (
    create_console,
    render_assistant_reply,
    render_banner,
    render_checkpoints_table,
    render_context_panel,
    render_diff_output,
    render_error,
    render_help_table,
    render_history_view,
    render_mcp_table,
    render_memory_results,
    render_profile_table,
    render_sessions_table,
    render_skills_table,
    render_status_panel,
    render_todo_panel,
    render_tool_event,
    render_turn_footer,
    render_usage_table,
)

_NOISY_HTTP_LOGGERS = (
    "httpx",
    "httpx2",
    "httpcore",
    "httpcore2",
    "openai",
    "anthropic",
)


def configure_logging(verbose: bool = False, log_level: str = "INFO") -> None:
    """Configure root logging and silence noisy HTTP SDK logs unless verbose."""
    level = logging.DEBUG if verbose else getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(level=level)
    if not verbose:
        for logger_name in _NOISY_HTTP_LOGGERS:
            logging.getLogger(logger_name).setLevel(logging.WARNING)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for `ivy`."""
    parser = argparse.ArgumentParser(
        prog="ivy",
        description="Ivy — Autonomous self-improving Python AI agent",
    )
    parser.add_argument("-p", "--prompt", type=str, default=None, help="Run a single prompt non-interactively and exit")
    parser.add_argument("-c", "--continue", dest="continue_latest", action="store_true", help="Resume the most recently updated session")
    parser.add_argument("--model", type=str, default=None, help="Override default LLM model")
    parser.add_argument("--session", type=str, default="cli:main", help="Session ID to resume")
    parser.add_argument("--theme", type=str, default=None, help="TUI theme (ivy, hermes, ocean)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    subparsers = parser.add_subparsers(dest="command")

    gw_parser = subparsers.add_parser("gateway", help="Manage multi-platform messaging gateway")
    gw_sub = gw_parser.add_subparsers(dest="gateway_action")
    gw_sub.add_parser("start", help="Start the async messaging gateway & cron daemon")
    gw_sub.add_parser("status", help="Show configured gateway adapters")

    model_parser = subparsers.add_parser("model", help="Show or test configured model")
    model_parser.add_argument("model_name", nargs="?", default=None, help="Model name")

    skills_parser = subparsers.add_parser("skills", help="Inspect and manage learned skills")
    skills_sub = skills_parser.add_subparsers(dest="skills_action")
    skills_sub.add_parser("list", help="List available skills")
    inspect_p = skills_sub.add_parser("inspect", help="Inspect a skill's instructions")
    inspect_p.add_argument("name", help="Skill name")
    remove_p = skills_sub.add_parser("remove", help="Remove a skill")
    remove_p.add_argument("name", help="Skill name")

    cron_parser = subparsers.add_parser("cron", help="Inspect scheduled cron tasks")
    cron_sub = cron_parser.add_subparsers(dest="cron_action")
    cron_sub.add_parser("list", help="List scheduled tasks")

    doc_parser = subparsers.add_parser("doctor", help="Diagnose environment, endpoint, DB & tools")
    doc_parser.add_argument("--fix", action="store_true", help="Auto-create missing .env / skills dir")
    doc_parser.add_argument("--no-live", dest="no_live", action="store_true", help="Skip HTTP endpoint probe")

    return parser


def build_default_tool_registry(
    config: IvyConfig,
    interactive: bool = True,
    unattended: bool = False,
    skill_engine: SkillEngine | None = None,
    approval_callback: Callable[[str], bool | Awaitable[bool]] | None = None,
    approval_policy: SafetyApprovalPolicy | None = None,
    memory_store: MemoryStore | None = None,
    llm_client: LLMClient | None = None,
    todo_store: TodoStore | None = None,
    checkpoint_store: FileCheckpointStore | None = None,
    cron_scheduler: CronScheduler | None = None,
    session_id_getter: Callable[[], str] | None = None,
    clarify_callback: Callable[[str, list[str], bool], Any] | None = None,
) -> ToolRegistry:
    """Create and populate the default ToolRegistry."""
    registry = ToolRegistry()
    policy = approval_policy or SafetyApprovalPolicy(
        auto_approve=config.auto_approve_safety,
        interactive=interactive,
        unattended=unattended,
        approval_callback=approval_callback,
    )
    registry.register(TerminalTool(approval_policy=policy))
    FileTools(
        checkpoint_store=checkpoint_store,
        session_id_getter=session_id_getter,
    ).register_all(registry)
    registry.register(CodeRunnerTool(timeout=config.code_runner_timeout))
    registry.register(WebFetchTool())
    registry.register(WebSearchTool())
    registry.register(TodoListTool(store=todo_store, session_id_getter=session_id_getter))
    if interactive or clarify_callback is not None:
        registry.register(ClarifyTool(clarify_callback=clarify_callback))
    if memory_store is not None:
        registry.register(RememberPreferenceTool(memory_store=memory_store))
        registry.register(SessionSearchTool(memory_store=memory_store))
    if skill_engine is not None:
        registry.register(ReadSkillTool(skill_engine=skill_engine))
        registry.register(SkillManagerTool(skill_engine=skill_engine))
    if cron_scheduler is not None:
        registry.register(
            CronManageTool(cron_scheduler=cron_scheduler, memory_store=memory_store)
        )
    if llm_client is not None:
        runner = SubagentRunner(
            llm_client=llm_client,
            tool_registry=registry,
            memory_store=memory_store,
            max_iterations=config.subagent_max_iterations,
        )
        registry.register(
            DelegateTaskTool(runner=runner, session_id_getter=session_id_getter)
        )
    return registry


class IvyCLIApp:
    """Coordinates interactive TUI session, streaming, persistence, and slash commands."""

    def __init__(
        self,
        config: IvyConfig,
        llm_client: LLMClient,
        memory_store: MemoryStore,
        skill_engine: SkillEngine,
        agent: IvyAgent,
        skill_generator: SkillGenerator | None = None,
        cron_scheduler: CronScheduler | None = None,
        session_id: str = "cli:main",
        console: Console | None = None,
        theme_name: str | None = None,
        env_path: str | Path | None = None,
        approval_policy: SafetyApprovalPolicy | None = None,
        history_path: str | Path | None = None,
    ) -> None:
        self.config = config
        self.llm_client = llm_client
        self.memory_store = memory_store
        self.skill_engine = skill_engine
        self.agent = agent
        self.skill_generator = skill_generator
        self.cron_scheduler = cron_scheduler or CronScheduler()
        self.session_id = session_id
        self.console = console or create_console()
        self.theme: ThemeConfig = get_theme(theme_name or config.default_theme)
        self.env_path = env_path
        self.approval_policy = approval_policy or self._discover_approval_policy()
        self.history_path = history_path
        self.display_mode: str = "verbose" if getattr(config, "verbose", False) else "compact"
        self.reasoning_mode: str = "collapsed"
        self.todo_store: TodoStore = self._discover_todo_store() or TodoStore()
        self.checkpoint_store: FileCheckpointStore = self._discover_checkpoint_store() or FileCheckpointStore()
        self.mcp_manager: MCPManager = MCPManager()
        self._active_status: Any = None
        self._prompt_session: Any = None
        self._session_title: str | None = None
        self._git_branch: str | None = get_git_branch()
        self._last_reasoning: str = ""
        self._prompt_stash: list[str] = []
        self._pending_prefill: str = ""
        self._usage_stats: dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "api_calls": 0,
        }
        self._compressions: int = 0
        self._ctx_percent: float | None = None
        self._last_duration_s: float | None = None
        self._cached_sessions: list[dict[str, Any]] = []

    def _discover_todo_store(self) -> TodoStore | None:
        registry = getattr(self.agent, "tool_registry", None)
        if registry is None or not hasattr(registry, "get"):
            return None
        try:
            tool = registry.get("todo_list")
            return getattr(tool, "store", None)
        except Exception:
            return None

    def _discover_checkpoint_store(self) -> FileCheckpointStore | None:
        registry = getattr(self.agent, "tool_registry", None)
        if registry is None or not hasattr(registry, "get"):
            return None
        try:
            wf = registry.get("write_file")
            cp = getattr(wf, "checkpoint_store", None)
            if isinstance(cp, FileCheckpointStore):
                return cp
            pf = registry.get("patch_file")
            cp2 = getattr(pf, "checkpoint_store", None)
            return cp2 if isinstance(cp2, FileCheckpointStore) else None
        except Exception:
            return None

    def _discover_approval_policy(self) -> SafetyApprovalPolicy | None:
        registry = getattr(self.agent, "tool_registry", None)
        if registry is None or not hasattr(registry, "get"):
            return None
        try:
            term = registry.get("terminal")
            return getattr(term, "approval_policy", None)
        except Exception:
            return None

    def _tool_names(self) -> list[str]:
        registry = getattr(self.agent, "tool_registry", None)
        if registry is None or not hasattr(registry, "list_tools"):
            return []
        try:
            return [t.name for t in registry.list_tools()]
        except Exception:
            return []

    def _tool_schemas(self) -> list[dict[str, Any]]:
        registry = getattr(self.agent, "tool_registry", None)
        if registry is None or not hasattr(registry, "get_schemas"):
            return []
        try:
            schemas = registry.get_schemas()
            return schemas if isinstance(schemas, list) else []
        except Exception:
            return []

    def _is_yolo(self) -> bool:
        return bool(self.approval_policy and self.approval_policy.auto_approve)

    def _render_toolbar(self) -> FormattedText:
        return build_bottom_toolbar(
            model=self.llm_client.model,
            session_id=self.session_id,
            tool_count=len(self._tool_names()),
            skill_count=len(self.skill_engine.list_skills()),
            theme=self.theme,
            ctx_percent=self._ctx_percent,
            total_tokens=self._usage_stats["total_tokens"],
            last_duration_s=self._last_duration_s,
            yolo=self._is_yolo(),
            focus_mode=(self.display_mode == "focus"),
            session_title=self._session_title,
            git_branch=self._git_branch,
        )

    async def _refresh_session_metadata(self) -> None:
        try:
            self._session_title = await self.memory_store.get_session_title(self.session_id)
            self._cached_sessions = await self.memory_store.list_sessions()
            history = await self.memory_store.get_messages(self.session_id, limit=100)
            breakdown = compute_context_breakdown(
                history,
                model=self.llm_client.model,
                tool_schemas=self._tool_schemas(),
            )
            self._ctx_percent = float(breakdown["percent"])
        except Exception:
            pass

    def _can_open_picker(self) -> bool:
        from io import StringIO

        if not (sys.stdin.isatty() and sys.stdout.isatty()):
            return False
        if isinstance(getattr(self.console, "file", None), StringIO):
            return False
        return True

    async def _run_session_picker(self) -> None:
        while True:
            sessions = await self.memory_store.list_sessions()
            self._cached_sessions = sessions
            if not sessions:
                await self.memory_store.ensure_session(self.session_id)
                sessions = await self.memory_store.list_sessions()
                self._cached_sessions = sessions

            options: list[PickerOption] = []
            for s in sessions:
                sid = str(s.get("session_id") or "")
                title = (s.get("title") or "").strip() or "(new session)"
                msgs = int(s.get("message_count", 0))
                rel = format_relative_time(s.get("updated_at"))
                options.append(
                    PickerOption(
                        value=sid,
                        label=title,
                        meta=f"{sid} · {msgs}m · {rel}" if rel else f"{sid} · {msgs}m",
                    )
                )

            res = await run_interactive_picker(
                title=f"Switch Session ({len(options)})",
                options=options,
                current_value=self.session_id,
                theme=self.theme,
                allow_new=True,
                allow_delete=True,
            )
            if res.action == "cancel":
                return
            if res.action == "new":
                self.session_id = f"cli:{uuid4().hex[:8]}"
                self._session_title = None
                await self.memory_store.ensure_session(self.session_id)
                await self._refresh_session_metadata()
                self.console.print(
                    f"Started new session: [{self.theme.ui_label}]{self.session_id}[/]"
                )
                return
            if res.action == "delete" and res.value:
                deleted_sid = res.value
                await self.memory_store.delete_session(deleted_sid)
                self.console.print(f"[dim]Deleted session {deleted_sid}.[/dim]")
                if deleted_sid == self.session_id:
                    remaining = await self.memory_store.list_sessions()
                    if remaining:
                        self.session_id = str(remaining[0]["session_id"])
                    else:
                        self.session_id = f"cli:{uuid4().hex[:8]}"
                        await self.memory_store.ensure_session(self.session_id)
                    await self._refresh_session_metadata()
                continue
            if res.action == "select" and res.value:
                if res.value != self.session_id:
                    await self._switch_session(res.value)
                else:
                    self.console.print(f"[dim]Already on session {self.session_id}.[/dim]")
                return

    async def _run_theme_picker(self) -> None:
        options = [
            PickerOption(value=t.name, label=t.name, meta=t.description)
            for t in list_themes()
        ]
        res = await run_interactive_picker(
            title="Select TUI Theme",
            options=options,
            current_value=self.theme.name,
            theme=self.theme,
        )
        if res.action == "select" and res.value:
            self.theme = get_theme(res.value)
            self._persist_env({"IVY_THEME": self.theme.name})
            if self._prompt_session is not None:
                self._prompt_session.style = build_prompt_style(self.theme)
            self.console.print(
                f"Switched theme to: [bold {self.theme.ui_accent}]{self.theme.name}[/] — {self.theme.description}"
            )

    async def _run_model_picker(self) -> None:
        presets = [
            self.llm_client.model,
            "gpt-4o-mini",
            "gpt-4o",
            "claude-sonnet-4-5",
            "gemini/gemini-2.5-flash",
            "ollama/qwen2.5",
        ]
        unique = list(dict.fromkeys(m for m in presets if m))
        options = [PickerOption(value=m, label=m) for m in unique]
        res = await run_interactive_picker(
            title="Select or Type Model",
            options=options,
            current_value=self.llm_client.model,
            theme=self.theme,
            allow_custom=True,
        )
        if res.action == "select" and res.value:
            self.llm_client.set_model(res.value)
            self._persist_env({"DEFAULT_MODEL": self.llm_client.model})
            await self._refresh_session_metadata()
            self.console.print(f"Switched model to: [{self.theme.ui_ok}]{self.llm_client.model}[/]")

    def _persist_env(self, updates: dict[str, str]) -> None:
        if self.env_path is not None:
            update_env_file(updates, env_path=self.env_path)

    async def _interactive_approval(self, command: str) -> Any:
        """Pause active spinner and present rich multi-option safety approval prompt."""
        from rich import box as rich_box
        from rich.markup import escape as rich_escape
        from rich.panel import Panel as RichPanel

        status = self._active_status
        if status is not None:
            status.stop()
        try:
            self.console.print()
            self.console.print(
                RichPanel(
                    f"[bold]{rich_escape(command)}[/bold]\n\n"
                    "[dim][y] allow once · [s] allow cmd this session · [a] allow all (/yolo) · [n] deny · [t] deny & tell Ivy[/dim]",
                    title=f"[bold {self.theme.ui_warn}]⚠ Safety Approval[/]",
                    border_style=self.theme.ui_warn,
                    box=rich_box.ROUNDED,
                    expand=False,
                )
            )
            answer = await asyncio.to_thread(
                lambda: input("Choice [y/s/a/N/t]: ").strip().lower()
            )
            if answer in ("y", "yes"):
                return True
            if answer in ("s", "session"):
                return "session"
            if answer in ("a", "always", "yolo"):
                return "always"
            if answer in ("t", "tell"):
                feedback = await asyncio.to_thread(
                    lambda: input("Tell Ivy what to do instead: ").strip()
                )
                return (False, feedback or "Use a safer alternative.")
            return False
        except (EOFError, KeyboardInterrupt):
            return False
        finally:
            if status is not None:
                status.start()

    async def _interactive_clarify(
        self, question: str, choices: list[str], multi_select: bool
    ) -> Any:
        """Ask user a structured clarifying question via picker or prompt fallback."""
        status = self._active_status
        if status is not None:
            status.stop()
        try:
            if choices and not multi_select and self._can_open_picker():
                opts = [PickerOption(value=c, label=c) for c in choices]
                opts.append(PickerOption(value="__OTHER__", label="Other (type custom answer)"))
                res = await run_interactive_picker(
                    title=question[:60],
                    options=opts,
                    theme=self.theme,
                )
                if res.action == "select" and res.value and res.value != "__OTHER__":
                    return res.value
                if res.action == "cancel":
                    return ""
                answer = await asyncio.to_thread(lambda: input(f"  {question}: ").strip())
                return answer

            self.console.print(f"\n[bold {self.theme.ui_accent}]? {rich_escape(question)}[/]")
            for idx, ch in enumerate(choices, start=1):
                self.console.print(f"  [dim]{idx}.[/] {rich_escape(ch)}")
            prompt_lbl = (
                "Enter numbers/choices (comma-separated) or text: "
                if multi_select and choices
                else "Choice # or answer: "
            )
            raw = await asyncio.to_thread(lambda: input(prompt_lbl).strip())
            if multi_select:
                picked: list[str] = []
                for part in raw.split(","):
                    tok = part.strip()
                    if tok.isdigit() and 1 <= int(tok) <= len(choices):
                        picked.append(choices[int(tok) - 1])
                    elif tok:
                        picked.append(tok)
                return picked
            if raw.isdigit() and 1 <= int(raw) <= len(choices):
                return choices[int(raw) - 1]
            return raw
        except (EOFError, KeyboardInterrupt):
            return [] if multi_select else ""
        finally:
            if status is not None:
                status.start()

    async def _run_command_palette(self) -> bool:
        """Open Ctrl+P fuzzy command palette and execute the chosen command."""
        options = [
            PickerOption(
                value=cmd,
                label=cmd,
                meta=SLASH_COMMAND_META.get(cmd, ""),
            )
            for cmd in SLASH_COMMANDS
        ]
        if not self._can_open_picker():
            render_help_table(self.console, theme=self.theme)
            return True
        res = await run_interactive_picker(
            title="Command Palette",
            options=options,
            theme=self.theme,
        )
        if res.action == "select" and res.value:
            return await self.handle_slash_command(res.value)
        return True

    async def _handle_btw(self, question: str) -> None:
        """Ask a quick side question against snapshot history without mutating MemoryStore."""
        if not question:
            self.console.print(f"[{self.theme.ui_warn}]Usage: /btw <side question>[/]")
            return
        history = await self.memory_store.get_messages(self.session_id, limit=30)
        messages = [
            ChatMessage(
                role=Role.SYSTEM,
                content="Answer the user's side question concisely using session context without invoking tools.",
            ),
            *history,
            ChatMessage(role=Role.USER, content=question),
        ]
        status = self.console.status(
            f"[{self.theme.ui_accent}]{self.theme.glyph} Quick answer (/btw)...[/]",
            spinner="dots",
        )
        try:
            with status:
                reply = await self.llm_client.chat_completion(messages=messages, stream=False)
            raw_text = (
                reply.content
                if isinstance(reply, ChatMessage)
                else "".join([c async for c in reply])
            ) or ""
            visible, _ = extract_thinking_blocks(raw_text)
            self.console.print(
                RichPanel(
                    Markdown(visible or raw_text or "(empty)"),
                    title=f"[bold {self.theme.ui_label}]BTW (off-record)[/]",
                    border_style=self.theme.banner_dim,
                    box=rich_box.ROUNDED,
                    expand=False,
                )
            )
        except Exception as exc:
            render_error(self.console, exc, theme=self.theme)

    async def _handle_rewind(self, args: list[str]) -> None:
        """Roll back session to a chosen earlier user prompt for editing or re-running."""
        turns = await self.memory_store.get_user_turns(self.session_id)
        if not turns:
            self.console.print(f"[{self.theme.ui_warn}]No user turns in this session to rewind.[/]")
            return

        target: dict[str, Any] | None = None
        if args and args[0].isdigit():
            idx = int(args[0]) - 1
            if 0 <= idx < len(turns):
                target = turns[idx]
        elif self._can_open_picker():
            options = [
                PickerOption(
                    value=str(t["id"]),
                    label=" ".join((t["content"] or "").split())[:64],
                    meta=f"turn #{i} · {format_relative_time(t.get('created_at'))}",
                )
                for i, t in enumerate(turns, start=1)
            ]
            res = await run_interactive_picker(
                title=f"Rewind Session ({len(turns)} turns)",
                options=options,
                current_value=str(turns[-1]["id"]),
                theme=self.theme,
            )
            if res.action != "select" or not res.value:
                return
            by_id = {str(t["id"]): t for t in turns}
            target = by_id.get(res.value)
        else:
            for i, t in enumerate(turns, start=1):
                snippet = " ".join((t["content"] or "").split())[:72]
                self.console.print(f"  [dim]{i}.[/] {rich_escape(snippet)}")
            self.console.print("[dim]Run /rewind <#> to roll back to turn #.[/dim]")
            return

        if target is None:
            self.console.print(f"[{self.theme.ui_warn}]Turn not found.[/]")
            return

        prompt_text, deleted = await self.memory_store.rewind_to_message_id(
            self.session_id, int(target["id"])
        )
        await self._refresh_session_metadata()
        self._pending_prefill = prompt_text or ""
        self.console.print(
            f"[{self.theme.ui_ok}]Rewound {deleted} messages[/] — prompt loaded for edit/resend."
        )

    async def _handle_rollback(self, args: list[str]) -> None:
        """Inspect, diff, or restore pre-edit FileCheckpoint snapshots."""
        action = args[0].lower() if args else "list"
        if action == "list":
            cps = self.checkpoint_store.list_checkpoints(session_id=self.session_id)
            if not cps:
                cps = self.checkpoint_store.list_checkpoints(session_id=None)
            render_checkpoints_table(self.console, cps, theme=self.theme)
            return

        cp = None
        if len(args) >= 2 and args[1].isdigit():
            cp = self.checkpoint_store.get(int(args[1]))
        else:
            cp = self.checkpoint_store.latest(session_id=self.session_id) or self.checkpoint_store.latest()

        if cp is None:
            self.console.print(f"[{self.theme.ui_warn}]No matching file checkpoint found.[/]")
            return

        if action == "diff":
            diff_text = self.checkpoint_store.diff_checkpoint(cp.id)
            render_diff_output(self.console, diff_text, theme=self.theme)
            return

        if action == "restore":
            ok, msg = self.checkpoint_store.restore_checkpoint(cp.id)
            color = self.theme.ui_ok if ok else self.theme.ui_error
            self.console.print(f"[{color}]{rich_escape(msg)}[/]")
            return

        self.console.print(f"[{self.theme.ui_warn}]Usage: /rollback [list | diff [id] | restore [id]][/]")

    async def _handle_init(self, args: list[str]) -> None:
        """Ask Ivy to inspect the repository and write/update AGENTS.md."""
        notes = " ".join(args).strip()
        extra = f" Extra focus notes: {notes}." if notes else ""
        prompt = (
            "Inspect this workspace (directory layout, manifests like pyproject.toml/package.json, "
            "run/test commands, and conventions) and write or update a concise, factual `AGENTS.md` file "
            "at the repository root capturing architecture, build/test commands, and coding rules."
            f"{extra}"
        )
        await self._run_turn_with_live_stream(prompt)

    async def _handle_review(self, args: list[str]) -> None:
        """Audit uncommitted git diff using an isolated SubagentRunner or direct LLM critique."""
        focus = " ".join(args).strip()
        diff_text = run_git_diff(staged=False) or run_git_diff(staged=True)
        if not diff_text or diff_text.startswith("git "):
            self.console.print("[dim]No uncommitted git diff to review (run `/diff` to verify).[/dim]")
            return

        focus_line = f"Focus specifically on: {focus}.\n" if focus else ""
        review_prompt = (
            "Perform a concise, rigorous code review of the uncommitted git diff below. "
            "Highlight real bugs, regressions, security issues, missing edge-case guards, or missing tests.\n"
            f"{focus_line}\n```diff\n{diff_text[:10000]}\n```"
        )
        status = self.console.status(
            f"[{self.theme.ui_accent}]{self.theme.glyph} Auditing changes (/review)...[/]",
            spinner="dots",
        )
        try:
            with status:
                reg = getattr(self.agent, "tool_registry", None)
                if isinstance(reg, ToolRegistry):
                    read_tools = [
                        t.name for t in reg.list_tools() if getattr(t, "concurrent_safe", False)
                    ]
                    reviewer = SubagentRunner(
                        llm_client=self.llm_client,
                        tool_registry=reg,
                        memory_store=self.memory_store,
                        max_iterations=min(4, self.config.subagent_max_iterations),
                    )
                    report = await reviewer.run(
                        task_prompt=review_prompt,
                        parent_session_id=self.session_id,
                        allowed_tools=read_tools,
                    )
                else:
                    reply = await self.llm_client.chat_completion(
                        messages=[
                            ChatMessage(role=Role.SYSTEM, content="You are an expert code reviewer."),
                            ChatMessage(role=Role.USER, content=review_prompt),
                        ],
                        stream=False,
                    )
                    report = reply.content if isinstance(reply, ChatMessage) else ""
            visible, _ = extract_thinking_blocks(report or "")
            self.console.print(
                RichPanel(
                    Markdown(visible or report or "No issues found."),
                    title=f"[bold {self.theme.banner_title}]{self.theme.glyph} Code Review[/]",
                    border_style=self.theme.banner_border,
                    box=rich_box.ROUNDED,
                    expand=False,
                )
            )
        except Exception as exc:
            render_error(self.console, exc, theme=self.theme)

    async def _handle_goal(self, raw_args: str) -> None:
        """Autonomous /goal loop with optional --gate \"<shell_cmd>\" and --max-turns N."""
        try:
            tokens = shlex.split(raw_args)
        except ValueError:
            tokens = raw_args.split()

        gate_cmd: str | None = None
        max_turns = 5
        obj_parts: list[str] = []
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if tok == "--gate" and i + 1 < len(tokens):
                gate_cmd = tokens[i + 1]
                i += 2
            elif tok == "--max-turns" and i + 1 < len(tokens):
                try:
                    max_turns = max(1, min(15, int(tokens[i + 1])))
                except ValueError:
                    pass
                i += 2
            else:
                obj_parts.append(tok)
                i += 1

        objective = " ".join(obj_parts).strip()
        if not objective:
            self.console.print(
                f"[{self.theme.ui_warn}]Usage: /goal <objective> [--gate \"<command>\"] [--max-turns 5][/]"
            )
            return

        current_prompt = objective
        for idx in range(1, max_turns + 1):
            self.console.print(
                f"\n[bold {self.theme.ui_accent}]⊙ Goal iteration {idx}/{max_turns}[/]"
                + (f" [dim](gate: {rich_escape(gate_cmd)})[/dim]" if gate_cmd else "")
            )
            await self._run_turn_with_live_stream(current_prompt)
            if not gate_cmd:
                return

            proc = await asyncio.create_subprocess_shell(
                gate_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=60.0)
            except asyncio.TimeoutError:
                self.console.print(f"[{self.theme.ui_error}]✗ Gate command timed out (60s).[/]")
                return

            if proc.returncode == 0:
                self.console.print(
                    f"[bold {self.theme.ui_ok}]✓ Verification gate passed ({rich_escape(gate_cmd)})![/]"
                )
                return

            combined = (
                stdout_b.decode("utf-8", errors="replace")
                + "\n"
                + stderr_b.decode("utf-8", errors="replace")
            ).strip()
            self.console.print(
                f"[{self.theme.ui_warn}]✗ Gate failed (exit {proc.returncode}); feeding output to next iteration...[/]"
            )
            current_prompt = (
                f"Verification gate `{gate_cmd}` failed with exit code {proc.returncode}.\n"
                f"Output:\n```\n{combined[-1500:]}\n```\n\n"
                f"Fix the remaining issues to accomplish the goal: {objective}"
            )

        self.console.print(
            f"[{self.theme.ui_error}]Goal reached max turns ({max_turns}) with gate still failing.[/]"
        )

    async def _handle_mcp(self, args: list[str]) -> None:
        action = args[0].lower() if args else "list"
        reg = getattr(self.agent, "tool_registry", None)
        if action == "reload":
            if not isinstance(reg, ToolRegistry):
                self.console.print(f"[{self.theme.ui_warn}]ToolRegistry unavailable for MCP reload.[/]")
                return
            loaded = await self.mcp_manager.reload(reg)
            render_mcp_table(self.console, self.mcp_manager.list_servers(), theme=self.theme)
            self.console.print(f"[dim]Registered {len(loaded)} MCP tool(s).[/dim]")
            return
        render_mcp_table(self.console, self.mcp_manager.list_servers(), theme=self.theme)

    async def _handle_cron_command(self, raw_line: str, args: list[str]) -> None:
        action = args[0].lower() if args else "list"
        if action == "list":
            tasks = self.cron_scheduler.list_tasks()
            if not tasks:
                self.console.print("[dim]No scheduled cron tasks registered.[/dim]")
                return
            for t in tasks:
                state = f"[{self.theme.ui_ok}]active[/]" if t.enabled else "[dim]paused[/dim]"
                self.console.print(
                    f"- [bold]{rich_escape(t.name)}[/bold] [{state}] `{t.cron_expr}` "
                    f"(next: {t.next_run}) — {rich_escape((t.prompt or '')[:60])}"
                )
            return

        if action == "add":
            try:
                tokens = shlex.split(raw_line)
            except ValueError as exc:
                self.console.print(f"[{self.theme.ui_error}]Invalid quotes:[/] {exc}")
                return
            if len(tokens) < 5:
                self.console.print(
                    f"[{self.theme.ui_warn}]Usage: /cron add <name> \"<cron_expr>\" <prompt>[/]"
                )
                return
            name, expr = tokens[2], tokens[3]
            prompt = " ".join(tokens[4:]).strip()
            try:
                task = ScheduledTask(name=name, cron_expr=expr, prompt=prompt, enabled=True)
            except ValueError as exc:
                self.console.print(f"[{self.theme.ui_error}]Invalid cron expression:[/] {exc}")
                return
            self.cron_scheduler.add_task(task)
            await self.memory_store.save_cron_job(name, expr, prompt, enabled=True)
            self.console.print(f"[{self.theme.ui_ok}]Added cron job[/] `{name}` (`{expr}`).")
            return

        if action in ("pause", "resume", "remove") and len(args) >= 2:
            name = args[1]
            if action == "pause":
                ok = self.cron_scheduler.pause_task(name)
                if ok:
                    await self.memory_store.set_cron_job_enabled(name, False)
                msg = f"Paused '{name}'" if ok else f"Cron job '{name}' not found"
            elif action == "resume":
                ok = self.cron_scheduler.resume_task(name)
                if ok:
                    await self.memory_store.set_cron_job_enabled(name, True)
                msg = f"Resumed '{name}'" if ok else f"Cron job '{name}' not found"
            else:
                ok = self.cron_scheduler.remove_task(name)
                await self.memory_store.delete_cron_job(name)
                msg = f"Removed '{name}'" if ok else f"Cron job '{name}' not found"
            self.console.print(msg)
            return

        self.console.print(
            f"[{self.theme.ui_warn}]Usage: /cron [list | add <name> \"<expr>\" <prompt> | pause <name> | resume <name> | remove <name>][/]"
        )

    async def _handle_setup(self, args: list[str]) -> None:
        """Configure API base URL, API key, and default model, persisting to .env."""
        if len(args) >= 2:
            api_base = args[0].strip()
            api_key = args[1].strip()
            model = args[2].strip() if len(args) >= 3 else self.llm_client.model
        else:
            default_base = self.config.effective_api_base or "http://localhost:20128/v1"
            self.console.print(
                f"[bold {self.theme.ui_accent}]{self.theme.glyph} Ivy Endpoint Setup[/] "
                "[dim](press Enter to keep current value)[/dim]"
            )
            api_base = await asyncio.to_thread(
                lambda: input(f"  API Base URL [{default_base}]: ").strip() or default_base
            )
            api_key = await asyncio.to_thread(
                lambda: input("  API Key: ").strip()
            )
            model = await asyncio.to_thread(
                lambda: input(f"  Default Model [{self.llm_client.model}]: ").strip() or self.llm_client.model
            )

        updates: dict[str, str] = {"DEFAULT_MODEL": model}
        if api_base:
            updates["IVY_API_BASE"] = api_base
        if api_key:
            updates["OPENAI_API_KEY"] = api_key

        self.llm_client.reconfigure(
            model=model,
            api_base=api_base or None,
            openai_api_key=api_key or None,
        )
        self._persist_env(updates)
        self.console.print(
            f"[{self.theme.ui_ok}]Saved endpoint configuration[/] "
            f"(model=[bold]{self.llm_client.model}[/bold], base=[dim]{self.llm_client.base_url or 'default'}[/dim])"
        )

    async def _compose_in_editor(self, initial_text: str = "") -> str | None:
        """Open $VISUAL/$EDITOR (or notepad on Windows) with a markdown draft."""
        editor = os.environ.get("VISUAL") or os.environ.get("EDITOR")
        if not editor:
            editor = "notepad" if platform.system() == "Windows" else "vi"
        tmp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".md",
                prefix="ivy_prompt_",
                delete=False,
                encoding="utf-8",
            ) as tf:
                tf.write(initial_text)
                tmp_path = tf.name
            subprocess.run([editor, tmp_path], check=False)
            content = Path(tmp_path).read_text(encoding="utf-8", errors="replace").strip()
            return content or None
        except Exception as exc:
            self.console.print(f"[{self.theme.ui_error}]Failed to open editor ({editor}):[/] {exc}")
            return None
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    async def _run_bang_command(self, command: str) -> None:
        """Run a local shell command cleanly via !cmd with safety guard & zero LLM tokens."""
        if not command:
            self.console.print(f"[{self.theme.ui_warn}]Usage: !<shell command>[/]")
            return
        if self.approval_policy is not None:
            approved, feedback = await self.approval_policy.evaluate_approval(command)
            if not approved:
                msg = feedback or "Command rejected by safety policy."
                self.console.print(f"[{self.theme.ui_warn}]⊘ {msg}[/]")
                return
        t0 = time.monotonic()
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=60.0)
            out = stdout_b.decode("utf-8", errors="replace").strip()
            err = stderr_b.decode("utf-8", errors="replace").strip()
            dur = time.monotonic() - t0
            if out:
                self.console.print(out, markup=False)
            if err:
                self.console.print(f"[{self.theme.ui_error}]{err}[/]")
            self.console.print(f"[dim]$ exit {proc.returncode} · {dur:.2f}s (0 tokens)[/dim]")
        except asyncio.TimeoutError:
            self.console.print(f"[{self.theme.ui_error}]Shell command timed out (60s).[/]")
        except Exception as exc:
            self.console.print(f"[{self.theme.ui_error}]Shell error:[/] {exc}")

    async def _switch_session(self, target: str) -> None:
        """Switch session by index (#N from /session list) or explicit session ID."""
        resolved = target.strip()
        if resolved.isdigit():
            idx = int(resolved) - 1
            if not self._cached_sessions:
                self._cached_sessions = await self.memory_store.list_sessions()
            if 0 <= idx < len(self._cached_sessions):
                resolved = str(self._cached_sessions[idx]["session_id"])
            else:
                self.console.print(f"[{self.theme.ui_warn}]Session #{target} not found.[/]")
                return
        self.session_id = resolved
        await self._refresh_session_metadata()
        self.console.print(
            f"Switched session to: [{self.theme.ui_label}]{self.session_id}[/]"
            + (f" [dim]({self._session_title})[/dim]" if self._session_title else "")
        )

    async def handle_slash_command(self, raw_line: str) -> bool:
        """Handle a slash command (exact, alias, unique prefix, typo hint). Returns False to exit."""
        parts = raw_line.strip().split()
        raw_cmd = parts[0].lower()
        args = parts[1:]

        resolved_cmd, suggestions, mode = resolve_slash_command(raw_cmd, SLASH_COMMANDS)
        if resolved_cmd is None:
            if mode == "ambiguous":
                opts = ", ".join(suggestions)
                self.console.print(f"[{self.theme.ui_warn}]Ambiguous command '{raw_cmd}':[/] {opts}")
            elif suggestions:
                opts = ", ".join(suggestions)
                self.console.print(f"[{self.theme.ui_warn}]Unknown command:[/] {raw_cmd} (did you mean {opts}?)")
            else:
                self.console.print(f"[{self.theme.ui_warn}]Unknown command:[/] {raw_cmd} (try /help)")
            return True

        cmd = resolved_cmd

        if cmd == "/exit":
            self.console.print(f"[dim]Goodbye! {self.theme.glyph}[/dim]")
            return False

        if cmd == "/help":
            render_help_table(self.console, theme=self.theme)
            return True

        if cmd == "/setup":
            await self._handle_setup(args)
            return True

        if cmd == "/theme":
            if not args:
                if self._can_open_picker():
                    await self._run_theme_picker()
                else:
                    names = ", ".join(t.name for t in list_themes())
                    self.console.print(
                        f"Active theme: [bold {self.theme.ui_accent}]{self.theme.name}[/] (available: {names})"
                    )
            else:
                self.theme = get_theme(args[0])
                self._persist_env({"IVY_THEME": self.theme.name})
                if self._prompt_session is not None:
                    self._prompt_session.style = build_prompt_style(self.theme)
                self.console.print(
                    f"Switched theme to: [bold {self.theme.ui_accent}]{self.theme.name}[/] — {self.theme.description}"
                )
            return True

        if cmd == "/model":
            if not args:
                if self._can_open_picker():
                    await self._run_model_picker()
                else:
                    self.console.print(f"Active model: [{self.theme.ui_ok}]{self.llm_client.model}[/]")
            else:
                self.llm_client.set_model(args[0])
                self._persist_env({"DEFAULT_MODEL": self.llm_client.model})
                await self._refresh_session_metadata()
                self.console.print(f"Switched model to: [{self.theme.ui_ok}]{self.llm_client.model}[/]")
            return True

        if cmd == "/clear":
            await self.memory_store.clear_session(self.session_id)
            await self._refresh_session_metadata()
            self.console.print(
                f"Cleared history for session [{self.theme.ui_label}]{self.session_id}[/]."
            )
            return True

        if cmd == "/session":
            if not args:
                if self._can_open_picker():
                    await self._run_session_picker()
                else:
                    sessions = await self.memory_store.list_sessions()
                    self._cached_sessions = sessions
                    render_sessions_table(
                        self.console, sessions, self.session_id, theme=self.theme
                    )
            elif args[0].lower() == "list":
                sessions = await self.memory_store.list_sessions()
                self._cached_sessions = sessions
                render_sessions_table(
                    self.console, sessions, self.session_id, theme=self.theme
                )
            else:
                await self._switch_session(args[0])
            return True

        if cmd == "/new":
            if args:
                name = args[0].strip()
                self.session_id = name if ":" in name else f"cli:{name}"
            else:
                self.session_id = f"cli:{uuid4().hex[:8]}"
            self._session_title = None
            await self.memory_store.ensure_session(self.session_id)
            await self._refresh_session_metadata()
            self.console.print(
                f"Started new session: [{self.theme.ui_label}]{self.session_id}[/]"
            )
            return True

        if cmd == "/fork":
            dst_name = args[0].strip() if args else f"cli:{uuid4().hex[:8]}"
            if ":" not in dst_name:
                dst_name = f"cli:{dst_name}"
            copied = await self.memory_store.fork_session(self.session_id, dst_name)
            self.session_id = dst_name
            await self._refresh_session_metadata()
            self.console.print(
                f"[{self.theme.ui_ok}]Forked session →[/] [{self.theme.ui_label}]{self.session_id}[/] "
                f"[dim]({copied} messages copied)[/dim]"
            )
            return True

        if cmd == "/title":
            if not args:
                cur = await self.memory_store.get_session_title(self.session_id)
                self.console.print(
                    f"Session [{self.theme.ui_label}]{self.session_id}[/] title: "
                    + (f"[bold]{cur}[/bold]" if cur else "[dim](untitled)[/dim]")
                )
            else:
                new_title = " ".join(args).strip()
                await self.memory_store.set_session_title(self.session_id, new_title)
                self._session_title = new_title
                self.console.print(f"Updated session title: [bold {self.theme.ui_label}]{new_title}[/]")
            return True

        if cmd == "/undo":
            undone_prompt, deleted = await self.memory_store.delete_last_turn(self.session_id)
            if deleted == 0:
                self.console.print(f"[{self.theme.ui_warn}]Nothing to undo in this session.[/]")
            else:
                await self._refresh_session_metadata()
                preview = (undone_prompt or "").strip().replace("\n", " ")[:60]
                self.console.print(
                    f"[{self.theme.ui_ok}]Undid last turn[/] (removed {deleted} messages: [dim]{preview}[/dim])"
                )
                cps = self.checkpoint_store.list_checkpoints(session_id=self.session_id)
                if cps:
                    self.console.print(
                        f"[dim]Tip: {len(cps)} file checkpoint(s) exist — run `/rollback list` or `/rollback restore` to revert disk edits.[/dim]"
                    )
            return True

        if cmd == "/rollback":
            await self._handle_rollback(args)
            return True

        if cmd == "/retry":
            undone_prompt, deleted = await self.memory_store.delete_last_turn(self.session_id)
            if not undone_prompt:
                self.console.print(f"[{self.theme.ui_warn}]No previous user prompt to retry.[/]")
                return True
            self.console.print(f"[dim]Retrying prompt ({deleted} msgs rolled back)...[/dim]")
            await self._run_turn_with_live_stream(undone_prompt)
            return True

        if cmd == "/history":
            limit = int(args[0]) if args and args[0].isdigit() else 20
            messages = await self.memory_store.get_messages(self.session_id, limit=max(1, limit))
            render_history_view(self.console, messages, limit=limit, theme=self.theme)
            return True

        if cmd == "/save":
            raw_arg_str = raw_line.strip()[len(parts[0]):].strip()
            fmt = "md"
            out_path: str | None = None
            if raw_arg_str:
                first_tok, *rest_toks = raw_arg_str.split(maxsplit=1)
                if first_tok.lower() in ("md", "markdown", "json"):
                    fmt = "json" if first_tok.lower() == "json" else "md"
                    out_path = rest_toks[0].strip().strip("\"'") if rest_toks else None
                else:
                    out_path = raw_arg_str.strip("\"'")
                    if out_path.lower().endswith(".json"):
                        fmt = "json"
            messages = await self.memory_store.get_messages(self.session_id, limit=1000)
            if not out_path:
                safe_sid = self.session_id.replace(":", "_")
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                out_path = f"ivy_{safe_sid}_{stamp}.{fmt}"
            content = (
                export_session_json(self.session_id, self._session_title, messages)
                if fmt == "json"
                else export_session_markdown(self.session_id, self._session_title, messages)
            )
            dest = Path(out_path)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content, encoding="utf-8")
            self.console.print(f"[{self.theme.ui_ok}]Saved transcript:[/] {dest} ({len(messages)} msgs)")
            return True

        if cmd == "/copy":
            nth = int(args[0]) if args and args[0].isdigit() and int(args[0]) >= 1 else 1
            messages = await self.memory_store.get_messages(self.session_id, limit=200)
            replies = [
                m.content
                for m in messages
                if m.role.value == "assistant" and (m.content or "").strip()
            ]
            if not replies or nth > len(replies):
                self.console.print(f"[{self.theme.ui_warn}]No matching assistant response to copy.[/]")
                return True
            target_text = replies[-nth] or ""
            ok, method = copy_to_clipboard(target_text)
            if ok:
                self.console.print(
                    f"[{self.theme.ui_ok}]Copied reply[/] ({len(target_text)} chars via {method})."
                )
            else:
                self.console.print(f"[{self.theme.ui_error}]Clipboard copy failed:[/] {method}")
            return True

        if cmd == "/context":
            history = await self.memory_store.get_messages(self.session_id, limit=200)
            breakdown = compute_context_breakdown(
                history,
                model=self.llm_client.model,
                tool_schemas=self._tool_schemas(),
            )
            self._ctx_percent = float(breakdown["percent"])
            render_context_panel(self.console, breakdown, self.llm_client.model, theme=self.theme)
            return True

        if cmd == "/usage":
            render_usage_table(
                self.console,
                self._usage_stats,
                self.llm_client.model,
                compressions=self._compressions,
                theme=self.theme,
            )
            return True

        if cmd == "/compact":
            focus = " ".join(args).strip() or None
            history = await self.memory_store.get_messages(self.session_id, limit=500)
            mgr = ContextWindowManager()
            compacted, stats = await mgr.compact_with_llm(
                history, llm_client=self.llm_client, keep_recent=6, focus=focus
            )
            if stats["removed_messages"] == 0:
                self.console.print("[dim]Session history is already compact enough.[/dim]")
            else:
                await self.memory_store.replace_session_messages(self.session_id, compacted)
                self._compressions += 1
                await self._refresh_session_metadata()
                self.console.print(
                    f"[{self.theme.ui_ok}]Compacted {stats['removed_messages']} messages[/] "
                    f"(saved ~{format_token_count(stats['saved_tokens'])} tokens)"
                )
            return True

        if cmd == "/status":
            history = await self.memory_store.get_messages(self.session_id, limit=200)
            breakdown = compute_context_breakdown(
                history,
                model=self.llm_client.model,
                tool_schemas=self._tool_schemas(),
            )
            render_status_panel(
                self.console,
                {
                    "session_id": self.session_id,
                    "title": self._session_title,
                    "model": self.llm_client.model,
                    "provider": getattr(self.llm_client, "provider", "openai"),
                    "cwd": os.getcwd(),
                    "ctx_used": breakdown["used_tokens"],
                    "ctx_max": breakdown["max_tokens"],
                    "ctx_pct": breakdown["percent"],
                    "message_count": len(history),
                    "display_mode": self.display_mode,
                    "yolo": self._is_yolo(),
                    "total_tokens": self._usage_stats["total_tokens"],
                    "api_calls": self._usage_stats["api_calls"],
                },
                theme=self.theme,
            )
            return True

        if cmd == "/diff":
            staged = any(a.lower() == "staged" for a in args)
            stat = any(a.lower() == "--stat" for a in args)
            diff_out = run_git_diff(staged=staged, stat=stat)
            render_diff_output(self.console, diff_out, theme=self.theme)
            return True

        if cmd == "/verbose":
            modes = ("compact", "verbose", "focus")
            if args and args[0].lower() in modes:
                self.display_mode = args[0].lower()
            else:
                idx = modes.index(self.display_mode) if self.display_mode in modes else 0
                self.display_mode = modes[(idx + 1) % len(modes)]
            self.console.print(f"Display mode: [bold {self.theme.ui_accent}]{self.display_mode}[/]")
            return True

        if cmd == "/focus":
            sub = args[0].lower() if args else ("off" if self.display_mode == "focus" else "on")
            if sub == "on":
                self.display_mode = "focus"
            elif sub == "off":
                self.display_mode = "compact"
            self.console.print(
                f"Focus mode: [{'bold ' + self.theme.ui_accent if self.display_mode == 'focus' else 'dim'}]"
                f"{'ON (tool lines hidden)' if self.display_mode == 'focus' else 'OFF'}[/]"
            )
            return True

        if cmd == "/yolo":
            if self.approval_policy is None:
                self.console.print(f"[{self.theme.ui_warn}]No approval policy attached.[/]")
            else:
                self.approval_policy.auto_approve = not self.approval_policy.auto_approve
                state = (
                    "[bold red]ON — auto-approving dangerous commands this session[/]"
                    if self.approval_policy.auto_approve
                    else f"[{self.theme.ui_ok}]OFF — interactive safety guard active[/]"
                )
                self.console.print(f"YOLO mode: {state}")
            return True

        if cmd == "/prompt":
            initial = " ".join(args)
            composed = await self._compose_in_editor(initial)
            if composed:
                await self._run_turn_with_live_stream(composed)
            else:
                self.console.print("[dim]Editor buffer empty — nothing sent.[/dim]")
            return True

        if cmd == "/memory":
            query = " ".join(args).strip()
            if not query:
                self.console.print(f"[{self.theme.ui_warn}]Usage: /memory <search query>[/]")
            else:
                hits = await self.memory_store.search_fts(query)
                render_memory_results(self.console, hits, theme=self.theme)
            return True

        if cmd == "/skills":
            action = args[0].lower() if args else "list"
            if action == "list":
                render_skills_table(self.console, self.skill_engine.list_skills(), theme=self.theme)
            elif action == "inspect" and len(args) >= 2:
                skill = self.skill_engine.inspect_skill(args[1])
                if skill is None:
                    self.console.print(f"[{self.theme.ui_error}]Skill not found:[/] {args[1]}")
                else:
                    self.console.print(
                        f"[bold {self.theme.ui_label}]{skill.name}[/] ({skill.status})"
                    )
                    self.console.print(skill.instructions)
            elif action == "remove" and len(args) >= 2:
                removed = self.skill_engine.remove_skill(args[1])
                msg = f"Removed skill '{args[1]}'" if removed else f"Skill '{args[1]}' not found"
                self.console.print(msg)
            elif action == "reflect":
                if self.skill_generator is None:
                    self.console.print(f"[{self.theme.ui_warn}]Skill reflection is not configured.[/]")
                else:
                    history = await self.memory_store.get_messages(self.session_id)
                    created = await self.skill_generator.reflect_and_create(history)
                    if created is not None:
                        self.console.print(f"[{self.theme.ui_ok}]Created skill:[/] {created.name}")
                    else:
                        self.console.print("[dim]No reusable skill extracted from this session.[/dim]")
            else:
                self.console.print(
                    f"[{self.theme.ui_warn}]Usage: /skills [list | inspect <name> | remove <name> | reflect][/]"
                )
            return True

        if cmd == "/cron":
            await self._handle_cron_command(raw_line, args)
            return True

        if cmd == "/doctor":
            fix = any(a == "--fix" for a in args)
            live_probe = not any(a == "--no-live" for a in args)
            findings = await run_doctor_checks(
                config=self.config,
                llm_client=self.llm_client,
                memory_store=self.memory_store,
                skill_engine=self.skill_engine,
                live_probe=live_probe,
                fix=fix,
                env_path=self.env_path or ".env",
            )
            render_doctor_report(self.console, findings, theme=self.theme)
            return True

        if cmd == "/reasoning":
            arg = args[0].lower() if args else ""
            if arg == "show":
                if not self._last_reasoning:
                    self.console.print("[dim]No captured <think> reasoning block for the last turn.[/dim]")
                else:
                    self.console.print(
                        RichPanel(
                            rich_escape(self._last_reasoning),
                            title=f"[bold {self.theme.ui_label}]Last Reasoning Block[/]",
                            border_style=self.theme.banner_dim,
                            box=rich_box.ROUNDED,
                            expand=False,
                        )
                    )
            elif arg in ("collapsed", "stream", "hide"):
                self.reasoning_mode = arg
                self.console.print(f"Reasoning mode: [bold {self.theme.ui_accent}]{arg}[/]")
            else:
                self.console.print(
                    f"Reasoning mode: [bold {self.theme.ui_accent}]{self.reasoning_mode}[/] "
                    "(use: collapsed | stream | hide | show)"
                )
            return True

        if cmd == "/todo":
            sub = args[0].lower() if args else "list"
            if sub == "clear":
                self.todo_store.clear(self.session_id)
                self.console.print("[dim]Cleared task checklist for this session.[/dim]")
            else:
                render_todo_panel(self.console, self.todo_store.get(self.session_id), theme=self.theme)
            return True

        if cmd == "/profile":
            sub = args[0].lower() if args else "list"
            if sub == "set" and len(args) >= 3:
                key = args[1].strip()
                val = " ".join(args[2:]).strip()
                await self.memory_store.set_user_preference(key, val)
                self.console.print(f"[{self.theme.ui_ok}]Saved preference:[/] {key}={val}")
            elif sub in ("del", "rm", "delete") and len(args) >= 2:
                removed = await self.memory_store.delete_user_preference(args[1])
                self.console.print(f"Removed '{args[1]}'" if removed else f"Key '{args[1]}' not found")
            elif sub == "clear":
                count = await self.memory_store.clear_user_profile()
                self.console.print(f"[dim]Cleared {count} profile preferences.[/dim]")
            else:
                prof = await self.memory_store.get_user_profile()
                render_profile_table(self.console, prof, theme=self.theme)
            return True

        if cmd == "/btw":
            question = raw_line.strip()[len(parts[0]):].strip()
            await self._handle_btw(question)
            return True

        if cmd == "/rewind":
            await self._handle_rewind(args)
            return True

        if cmd == "/palette":
            return await self._run_command_palette()

        if cmd == "/init":
            await self._handle_init(args)
            return True

        if cmd == "/review":
            await self._handle_review(args)
            return True

        if cmd == "/goal":
            raw_goal_args = raw_line.strip()[len(parts[0]):].strip()
            await self._handle_goal(raw_goal_args)
            return True

        if cmd == "/mcp":
            await self._handle_mcp(args)
            return True

        if cmd == "/gateway":
            tg_state = "configured" if self.config.telegram_token else "not set"
            dc_state = "configured" if self.config.discord_token else "not set"
            self.console.print(f"Gateway tokens — Telegram: {tg_state}, Discord: {dc_state}")
            return True

        return True

    async def _run_turn_with_live_stream(self, prompt_text: str) -> None:
        """Execute one turn with @mention expansion, spinner, live Markdown streaming, and footer."""
        effective_prompt, attached = expand_at_mentions(prompt_text)
        if attached:
            self.console.print(f"[dim]📎 Attached: {', '.join(attached)}[/dim]")

        # Auto-title new untitled sessions on the first user turn (zero LLM cost)
        if not self._session_title:
            auto_title = " ".join(prompt_text.strip().split())[:48]
            if auto_title:
                self._session_title = auto_title
                try:
                    await self.memory_store.set_session_title(self.session_id, auto_title)
                except Exception:
                    pass

        t0 = time.monotonic()
        tool_events = 0
        hidden_tool_events = 0
        status = self.console.status(
            f"[{self.theme.ui_accent}]{self.theme.glyph} Thinking...[/]",
            spinner="dots",
        )
        self._active_status = status
        live: Live | None = None
        chunks: list[str] = []
        rule = f"[{self.theme.ui_accent}]{'─' * 44}[/]"

        def _stop_live() -> None:
            nonlocal live
            if live is not None:
                live.stop()
                live = None

        def _on_token(chunk: str) -> None:
            nonlocal live
            chunks.append(chunk)
            raw_stream = "".join(chunks)
            if self.reasoning_mode in ("collapsed", "hide"):
                visible, thinking = extract_thinking_blocks(raw_stream)
                if not visible:
                    if thinking and live is None:
                        status.update(
                            f"[{self.theme.ui_accent}]{self.theme.glyph} Reasoning ({len(thinking)} chars)...[/]"
                        )
                    return
                display_text = visible
            else:
                visible, thinking = extract_thinking_blocks(raw_stream)
                display_text = (
                    f"> *Thinking:* {thinking.replace(chr(10), ' ')}\n\n{visible}".strip()
                    if thinking
                    else visible
                )

            if live is None:
                status.stop()
                self.console.print()
                self.console.print(rule)
                self.console.print(f"[bold {self.theme.ui_accent}]{self.theme.glyph} Ivy[/]")
                live = Live(
                    Markdown(display_text),
                    console=self.console,
                    refresh_per_second=12,
                    vertical_overflow="visible",
                )
                live.start()
            else:
                live.update(Markdown(display_text))

        def _on_tool(ev: dict[str, Any]) -> None:
            nonlocal tool_events, hidden_tool_events
            if ev.get("phase") == "end":
                tool_events += 1
            if self.display_mode == "focus":
                if ev.get("phase") == "end":
                    hidden_tool_events += 1
                return
            if ev.get("phase") == "start":
                status.update(
                    f"[{self.theme.ui_accent}]{self.theme.glyph} Running {ev.get('name', 'tool')}...[/]"
                )
            elif ev.get("phase") == "end":
                status.update(f"[{self.theme.ui_accent}]{self.theme.glyph} Thinking...[/]")
            if live is not None:
                _stop_live()
                chunks.clear()
                status.start()
            render_tool_event(self.console, ev, theme=self.theme, display_mode=self.display_mode)

        try:
            with status:
                reply = await self.agent.run_turn(
                    prompt=effective_prompt,
                    session_id=self.session_id,
                    stream=True,
                    stream_callback=_on_token,
                    tool_callback=_on_tool,
                )
            elapsed = time.monotonic() - t0
            self._last_duration_s = elapsed

            turn_tokens = 0
            meta = getattr(reply, "metadata", None)
            if isinstance(meta, dict):
                tu = meta.get("turn_usage")
                if isinstance(tu, dict):
                    self._usage_stats["prompt_tokens"] += int(tu.get("prompt_tokens", 0) or 0)
                    self._usage_stats["completion_tokens"] += int(tu.get("completion_tokens", 0) or 0)
                    self._usage_stats["total_tokens"] += int(tu.get("total_tokens", 0) or 0)
                    self._usage_stats["api_calls"] += int(tu.get("api_calls", 0) or 0)
                    turn_tokens = int(tu.get("total_tokens", 0) or 0)

            raw_final = reply.content or "".join(chunks)
            visible_final, thought_final = extract_thinking_blocks(raw_final)
            if thought_final:
                self._last_reasoning = thought_final
            final_render = (
                raw_final
                if self.reasoning_mode == "stream"
                else (visible_final or raw_final)
            )
            if live is not None:
                live.update(Markdown(final_render))
                _stop_live()
                if thought_final and self.reasoning_mode == "collapsed":
                    n_lines = len(thought_final.splitlines())
                    self.console.print(
                        f"[dim]∴ Thought ({n_lines} line{'s' if n_lines != 1 else ''} · /reasoning show)[/dim]"
                    )
                self.console.print(rule)
            else:
                if thought_final and self.reasoning_mode == "collapsed":
                    n_lines = len(thought_final.splitlines())
                    self.console.print(
                        f"[dim]∴ Thought ({n_lines} line{'s' if n_lines != 1 else ''} · /reasoning show)[/dim]"
                    )
                render_assistant_reply(self.console, final_render, theme=self.theme)

            if hidden_tool_events > 0:
                self.console.print(
                    f"[dim]⋯ {hidden_tool_events} tool call{'s' if hidden_tool_events != 1 else ''} hidden (/focus off to show)[/dim]"
                )
            render_turn_footer(
                self.console,
                duration_s=elapsed,
                tool_count=tool_events,
                tokens=turn_tokens,
                timestamp_str=datetime.now().strftime("%H:%M"),
                theme=self.theme,
            )
            await self._refresh_session_metadata()
        except (asyncio.CancelledError, KeyboardInterrupt):
            _stop_live()
            self.console.print(f"[{self.theme.ui_warn}]Turn cancelled.[/]")
        except Exception as exc:
            _stop_live()
            render_error(self.console, exc, theme=self.theme)
        finally:
            _stop_live()
            self._active_status = None

    def _handle_stash_trigger(self, draft: str) -> None:
        """Stash a non-empty draft or pop the latest stashed draft into _pending_prefill."""
        cleaned = draft.strip()
        if cleaned:
            self._prompt_stash.append(draft)
            self.console.print(
                f"[dim]📌 Stashed draft ({len(self._prompt_stash)} in stash · press Ctrl+S on empty line to restore).[/dim]"
            )
        elif self._prompt_stash:
            self._pending_prefill = self._prompt_stash.pop()
            self.console.print(
                f"[dim]📌 Restored stashed draft ({len(self._prompt_stash)} remaining).[/dim]"
            )
        else:
            self.console.print("[dim]Prompt stash is empty.[/dim]")

    async def _maybe_offer_first_run_setup(self) -> None:
        """Offer quick /setup when launching interactively with missing API credentials."""
        if not self._can_open_picker():
            return
        provider = getattr(self.llm_client, "provider", "openai")
        if provider in ("ollama", "vllm"):
            return
        api_key = self.llm_client._resolve_api_key(provider)
        if (api_key and api_key != "missing-api-key") or self.config.effective_api_base:
            return
        self.console.print(
            f"\n[{self.theme.ui_warn}]No API key or custom endpoint configured yet.[/]"
        )
        ans = await asyncio.to_thread(
            lambda: input("Run /setup now? [Y/n]: ").strip().lower()
        )
        if ans in ("", "y", "yes"):
            await self._handle_setup([])

    async def run_interactive(self) -> None:
        """Start interactive prompt_toolkit REPL with Hermes-inspired chrome."""
        await self._refresh_session_metadata()
        render_banner(
            self.console,
            model=self.llm_client.model,
            session_id=self.session_id,
            tool_names=self._tool_names(),
            skills=self.skill_engine.list_skills(),
            theme=self.theme,
            session_title=self._session_title,
            git_branch=self._git_branch,
        )
        await self._maybe_offer_first_run_setup()
        prompt_session = create_prompt_session(
            theme=self.theme,
            bottom_toolbar=self._render_toolbar,
            skill_engine=self.skill_engine,
            history_path=self.history_path,
            session_provider=lambda: self._cached_sessions,
        )
        self._prompt_session = prompt_session

        while True:
            prefill = self._pending_prefill
            self._pending_prefill = ""
            try:
                prompt_msg = HTML(f"\n<prompt>{self.theme.prompt_symbol} </prompt>")
                if prefill:
                    user_input = await prompt_session.prompt_async(prompt_msg, default=prefill)
                else:
                    user_input = await prompt_session.prompt_async(prompt_msg)
            except (EOFError, KeyboardInterrupt):
                self.console.print(f"\n[dim]Exiting Ivy. {self.theme.glyph}[/dim]")
                break

            if user_input == "__IVY_PALETTE__":
                keep = await self._run_command_palette()
                if not keep:
                    break
                continue

            if user_input.startswith("__IVY_STASH__:"):
                self._handle_stash_trigger(user_input[len("__IVY_STASH__:"):])
                continue

            if user_input.startswith("__IVY_EDITOR__:"):
                draft = user_input[len("__IVY_EDITOR__:"):]
                composed = await self._compose_in_editor(draft)
                if not composed:
                    continue
                user_input = composed

            text = user_input.strip()
            if not text:
                continue

            if text.startswith("!"):
                await self._run_bang_command(text[1:].strip())
                continue

            if text.startswith("/"):
                keep_running = await self.handle_slash_command(text)
                if not keep_running:
                    break
                continue

            await self._run_turn_with_live_stream(text)


async def _run_async(args: argparse.Namespace) -> int:
    config = load_config()
    if args.model:
        config.default_model = args.model
    if args.verbose:
        config.verbose = True
    configure_logging(verbose=config.verbose, log_level=config.log_level)

    store = MemoryStore(db_path=config.db_path)
    await store.initialize()
    skills = SkillEngine(skills_dir=config.skills_dir)
    await skills.load_skills()
    llm = LLMClient(config=config)
    skill_gen = SkillGenerator(llm_client=llm, skill_engine=skills)
    console = create_console()

    try:
        if args.command == "skills":
            action = args.skills_action or "list"
            if action == "list":
                render_skills_table(console, skills.list_skills())
            elif action == "inspect":
                skill = skills.inspect_skill(args.name)
                if skill is None:
                    console.print(f"[red]Skill not found:[/red] {args.name}")
                    return 1
                console.print(f"[bold cyan]{skill.name}[/bold cyan]\n{skill.instructions}")
            elif action == "remove":
                if not skills.remove_skill(args.name):
                    console.print(f"[red]Skill not found:[/red] {args.name}")
                    return 1
                console.print(f"Removed skill '{args.name}'.")
            return 0

        if args.command == "model":
            console.print(args.model_name or config.default_model)
            return 0

        if args.command == "doctor":
            findings = await run_doctor_checks(
                config=config,
                llm_client=llm,
                memory_store=store,
                skill_engine=skills,
                live_probe=not getattr(args, "no_live", False),
                fix=getattr(args, "fix", False),
                env_path=".env",
            )
            return render_doctor_report(console, findings, theme=get_theme(config.default_theme))

        if args.command == "gateway":
            action = args.gateway_action or "status"
            gw_policy = SafetyApprovalPolicy(
                auto_approve=config.auto_approve_safety,
                interactive=False,
                unattended=False,
            )
            registry = build_default_tool_registry(
                config,
                interactive=False,
                unattended=False,
                skill_engine=skills,
                approval_policy=gw_policy,
                memory_store=store,
                llm_client=llm,
            )
            agent = IvyAgent(
                llm_client=llm,
                tool_registry=registry,
                memory_store=store,
                skill_engine=skills,
                skill_generator=skill_gen,
                config=config,
            )
            cron = CronScheduler(
                agent=agent,
                memory_store=store,
                approval_policy=gw_policy,
            )
            await cron.load_from_store()
            registry.register(CronManageTool(cron_scheduler=cron, memory_store=store))
            gw = GatewayServer(
                agent=agent,
                config=config,
                cron_scheduler=cron,
                approval_policy=gw_policy,
            )
            cron.dispatcher = gw.dispatch
            gw.configure_default_adapters()

            if action == "status":
                console.print(f"Allowed users: {len(config.allowed_users)}")
                return 0

            await gw.start()
            console.print("[green]Ivy Gateway daemon started. Press Ctrl+C to stop.[/green]")
            try:
                while True:
                    await asyncio.sleep(1.0)
            except (KeyboardInterrupt, asyncio.CancelledError):
                await gw.stop()
            return 0

        cli_policy = SafetyApprovalPolicy(
            auto_approve=config.auto_approve_safety,
            interactive=True,
            unattended=False,
        )
        todo_store = TodoStore()
        checkpoint_store = FileCheckpointStore()
        target_session = args.session
        if getattr(args, "continue_latest", False) and target_session == "cli:main":
            latest = await store.get_latest_session_id()
            if latest:
                target_session = latest

        session_holder = {"id": target_session}
        registry = build_default_tool_registry(
            config,
            interactive=True,
            unattended=False,
            skill_engine=skills,
            approval_policy=cli_policy,
            memory_store=store,
            llm_client=llm,
            todo_store=todo_store,
            checkpoint_store=checkpoint_store,
            session_id_getter=lambda: session_holder["id"],
        )
        agent = IvyAgent(
            llm_client=llm,
            tool_registry=registry,
            memory_store=store,
            skill_engine=skills,
            skill_generator=skill_gen,
            config=config,
        )
        cli_cron = CronScheduler(
            agent=agent,
            memory_store=store,
            approval_policy=cli_policy,
        )
        await cli_cron.load_from_store()
        registry.register(CronManageTool(cron_scheduler=cli_cron, memory_store=store))

        app = IvyCLIApp(
            config=config,
            llm_client=llm,
            memory_store=store,
            skill_engine=skills,
            agent=agent,
            skill_generator=skill_gen,
            cron_scheduler=cli_cron,
            session_id=target_session,
            console=console,
            theme_name=args.theme or config.default_theme,
            env_path=".env",
            approval_policy=cli_policy,
            history_path=Path.home() / ".ivy" / "history",
        )
        app.todo_store = todo_store
        app.checkpoint_store = checkpoint_store
        for tool_name in ("todo_list", "delegate_task", "write_file", "patch_file"):
            t_obj = registry.get(tool_name)
            if t_obj is not None and hasattr(t_obj, "session_id_getter"):
                setattr(t_obj, "session_id_getter", lambda: app.session_id)
        clarify_tool = registry.get("clarify")
        if isinstance(clarify_tool, ClarifyTool):
            clarify_tool.clarify_callback = app._interactive_clarify
        cli_policy.approval_callback = app._interactive_approval
        if Path(".mcp.json").is_file():
            await app.mcp_manager.reload(registry)

        try:
            if args.command == "cron":
                await app.handle_slash_command("/cron list")
                return 0

            if args.prompt:
                await app._run_turn_with_live_stream(args.prompt)
                return 0

            await app.run_interactive()
            return 0
        finally:
            await app.mcp_manager.close_all()
    finally:
        await llm.close()
        await store.close()


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for `ivy`."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return asyncio.run(_run_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
