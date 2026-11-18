"""CLI parser, interactive TUI loop, and gateway entrypoint (`ivy`)."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Awaitable, Callable, Sequence

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
from ivy.cron.scheduler import CronScheduler
from ivy.gateway.server import GatewayServer
from ivy.memory.context import compute_context_breakdown
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
from ivy.tools.process_registry import BackgroundProcessManager, ProcessManageTool
from ivy.tools.profile import RememberPreferenceTool
from ivy.tools.registry import ToolRegistry
from ivy.tools.runner import CodeRunnerTool
from ivy.tools.session_search import SessionSearchTool
from ivy.tools.skill_manager import SkillManagerTool
from ivy.tools.terminal import TerminalTool
from ivy.tools.todo import TodoListTool, TodoStore
from ivy.tools.web import WebFetchTool, WebSearchTool
from ivy.tui.commands import SlashCommandMixin
from ivy.tui.doctor import render_doctor_report, run_doctor_checks
from ivy.tui.helpers import (
    copy_to_clipboard,  # noqa: F401 - exposed for _app_ns() & test monkeypatches
    expand_at_mentions,
    extract_thinking_blocks,
    get_git_branch,
    run_git_diff,  # noqa: F401 - exposed for _app_ns() & test monkeypatches
)
from ivy.tui.picker import PickerOption, run_interactive_picker
from ivy.tui.prompt import (
    build_bottom_toolbar,
    create_prompt_session,
)
from ivy.tui.theme import ThemeConfig, get_theme
from ivy.tui.views import (
    create_console,
    render_assistant_reply,
    render_banner,
    render_error,
    render_skills_table,
    render_tool_event,
    render_turn_footer,
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
    process_manager: BackgroundProcessManager | None = None,
) -> ToolRegistry:
    """Create and populate the default ToolRegistry."""
    registry = ToolRegistry()
    policy = approval_policy or SafetyApprovalPolicy(
        auto_approve=config.auto_approve_safety,
        interactive=interactive,
        unattended=unattended,
        approval_callback=approval_callback,
    )
    registry.register(TerminalTool(approval_policy=policy, process_manager=process_manager))
    if process_manager is not None:
        registry.register(ProcessManageTool(manager=process_manager))
    workspace_root = Path.cwd().resolve() if config.restrict_workspace else None
    FileTools(
        checkpoint_store=checkpoint_store,
        session_id_getter=session_id_getter,
        workspace_root=workspace_root,
        require_read_before_overwrite=True,
    ).register_all(registry)
    registry.register(CodeRunnerTool(timeout=config.code_runner_timeout))
    registry.register(WebFetchTool(allow_private_urls=config.allow_private_urls))
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


class IvyCLIApp(SlashCommandMixin):
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
        process_manager: BackgroundProcessManager | None = None,
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
        self._process_manager: BackgroundProcessManager | None = process_manager
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

    def _discover_process_manager(self) -> BackgroundProcessManager | None:
        if self._process_manager is not None:
            return self._process_manager
        registry = getattr(self.agent, "tool_registry", None)
        if registry is None or not hasattr(registry, "get"):
            return None
        try:
            proc_tool = registry.get("process")
            mgr = getattr(proc_tool, "manager", None)
            if isinstance(mgr, BackgroundProcessManager):
                return mgr
            term = registry.get("terminal")
            mgr2 = getattr(term, "process_manager", None)
            return mgr2 if isinstance(mgr2, BackgroundProcessManager) else None
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

    def _persist_env(self, updates: dict[str, str]) -> None:
        if self.env_path is not None:
            update_env_file(updates, env_path=self.env_path)

    async def _interactive_approval(self, command: str) -> Any:
        """Pause active spinner and present rich multi-option safety approval prompt."""
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
        cli_process_manager = BackgroundProcessManager()
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
            process_manager=cli_process_manager,
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
            process_manager=cli_process_manager,
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
            await cli_process_manager.shutdown_all()
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
