"""CLI parser, interactive TUI loop, and gateway entrypoint (`ivy`)."""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path
from typing import Any, Awaitable, Callable, Sequence

from prompt_toolkit.formatted_text import FormattedText, HTML
from rich.console import Console

from ivy.config import IvyConfig, load_config, update_env_file
from ivy.core.agent import IvyAgent
from ivy.core.llm import LLMClient
from ivy.cron.scheduler import CronScheduler
from ivy.gateway.server import GatewayServer
from ivy.memory.store import MemoryStore
from ivy.skills.engine import SkillEngine
from ivy.skills.generator import SkillGenerator
from ivy.skills.tool import ReadSkillTool
from ivy.tools.approval import SafetyApprovalPolicy
from ivy.tools.files import FileTools
from ivy.tools.registry import ToolRegistry
from ivy.tools.runner import CodeRunnerTool
from ivy.tools.terminal import TerminalTool
from ivy.tools.web import WebFetchTool, WebSearchTool
from ivy.tui.prompt import build_bottom_toolbar, build_prompt_style, create_prompt_session
from ivy.tui.theme import ThemeConfig, get_theme, list_themes
from ivy.tui.views import (
    create_console,
    render_assistant_reply,
    render_banner,
    render_error,
    render_help_table,
    render_memory_results,
    render_skills_table,
    render_tool_event,
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

    return parser


def build_default_tool_registry(
    config: IvyConfig,
    interactive: bool = True,
    unattended: bool = False,
    skill_engine: SkillEngine | None = None,
    approval_callback: Callable[[str], bool | Awaitable[bool]] | None = None,
    approval_policy: SafetyApprovalPolicy | None = None,
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
    FileTools().register_all(registry)
    registry.register(CodeRunnerTool(timeout=config.code_runner_timeout))
    registry.register(WebFetchTool())
    registry.register(WebSearchTool())
    if skill_engine is not None:
        registry.register(ReadSkillTool(skill_engine=skill_engine))
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
        self._active_status: Any = None
        self._prompt_session: Any = None

    def _tool_names(self) -> list[str]:
        registry = getattr(self.agent, "tool_registry", None)
        if registry is None or not hasattr(registry, "list_tools"):
            return []
        try:
            return [t.name for t in registry.list_tools()]
        except Exception:
            return []

    def _render_toolbar(self) -> FormattedText:
        return build_bottom_toolbar(
            model=self.llm_client.model,
            session_id=self.session_id,
            tool_count=len(self._tool_names()),
            skill_count=len(self.skill_engine.list_skills()),
            theme=self.theme,
        )

    def _persist_env(self, updates: dict[str, str]) -> None:
        if self.env_path is not None:
            update_env_file(updates, env_path=self.env_path)

    async def _interactive_approval(self, command: str) -> bool:
        """Pause any active Rich spinner before asking the terminal user for approval."""
        status = self._active_status
        if status is not None:
            status.stop()
        try:
            self.console.print(
                f"\n[bold {self.theme.ui_warn}][SAFETY GUARD][/] Potentially destructive command:\n  {command}"
            )
            answer = await asyncio.to_thread(
                lambda: input("Approve execution? [y/N]: ").strip().lower()
            )
            return answer in ("y", "yes")
        except (EOFError, KeyboardInterrupt):
            return False
        finally:
            if status is not None:
                status.start()

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

    async def handle_slash_command(self, raw_line: str) -> bool:
        """Handle a slash command. Returns False to exit interactive loop."""
        parts = raw_line.strip().split()
        cmd = parts[0].lower()
        args = parts[1:]

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
                self.console.print(f"Active model: [{self.theme.ui_ok}]{self.llm_client.model}[/]")
            else:
                self.llm_client.set_model(args[0])
                self._persist_env({"DEFAULT_MODEL": self.llm_client.model})
                self.console.print(f"Switched model to: [{self.theme.ui_ok}]{self.llm_client.model}[/]")
            return True

        if cmd == "/clear":
            await self.memory_store.clear_session(self.session_id)
            self.console.print(
                f"Cleared history for session [{self.theme.ui_label}]{self.session_id}[/]."
            )
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
            tasks = self.cron_scheduler.list_tasks()
            if not tasks:
                self.console.print("[dim]No scheduled cron tasks registered.[/dim]")
            else:
                for t in tasks:
                    self.console.print(f"- {t.name}: `{t.cron_expr}` (next: {t.next_run})")
            return True

        if cmd == "/gateway":
            tg_state = "configured" if self.config.telegram_token else "not set"
            dc_state = "configured" if self.config.discord_token else "not set"
            self.console.print(f"Gateway tokens — Telegram: {tg_state}, Discord: {dc_state}")
            return True

        self.console.print(f"[{self.theme.ui_warn}]Unknown command:[/] {cmd} (try /help)")
        return True

    async def _run_turn_with_live_stream(self, prompt_text: str) -> None:
        """Execute one turn with spinner, live token streaming, and Ctrl+C cancellation."""
        status = self.console.status(
            f"[{self.theme.ui_accent}]{self.theme.glyph} Thinking...[/]",
            spinner="dots",
        )
        self._active_status = status
        stream_started = False
        rule = f"[{self.theme.ui_accent}]{'─' * 44}[/]"

        def _on_token(chunk: str) -> None:
            nonlocal stream_started
            if not stream_started:
                stream_started = True
                status.stop()
                self.console.print()
                self.console.print(rule)
                self.console.print(f"[bold {self.theme.ui_accent}]{self.theme.glyph} Ivy[/]")
            self.console.file.write(chunk)
            self.console.file.flush()

        def _on_tool(ev: dict[str, Any]) -> None:
            nonlocal stream_started
            if stream_started:
                self.console.file.write("\n")
                self.console.file.flush()
                stream_started = False
                status.start()
            render_tool_event(self.console, ev, theme=self.theme)

        try:
            with status:
                reply = await self.agent.run_turn(
                    prompt=prompt_text,
                    session_id=self.session_id,
                    stream=True,
                    stream_callback=_on_token,
                    tool_callback=_on_tool,
                )
            if stream_started:
                self.console.file.write("\n")
                self.console.file.flush()
                self.console.print(rule)
            else:
                render_assistant_reply(self.console, reply.content or "", theme=self.theme)
        except (asyncio.CancelledError, KeyboardInterrupt):
            if stream_started:
                self.console.file.write("\n")
                self.console.file.flush()
            self.console.print(f"[{self.theme.ui_warn}]Turn cancelled.[/]")
        except Exception as exc:
            render_error(self.console, exc, theme=self.theme)
        finally:
            self._active_status = None

    async def run_interactive(self) -> None:
        """Start interactive prompt_toolkit REPL with Hermes-inspired chrome."""
        render_banner(
            self.console,
            model=self.llm_client.model,
            session_id=self.session_id,
            tool_names=self._tool_names(),
            skills=self.skill_engine.list_skills(),
            theme=self.theme,
        )
        prompt_session = create_prompt_session(
            theme=self.theme,
            bottom_toolbar=self._render_toolbar,
            skill_engine=self.skill_engine,
        )
        self._prompt_session = prompt_session

        while True:
            try:
                user_input = await prompt_session.prompt_async(
                    HTML(f"\n<prompt>{self.theme.prompt_symbol} </prompt>")
                )
            except (EOFError, KeyboardInterrupt):
                self.console.print(f"\n[dim]Exiting Ivy. {self.theme.glyph}[/dim]")
                break

            text = user_input.strip()
            if not text:
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
        registry = build_default_tool_registry(
            config,
            interactive=True,
            unattended=False,
            skill_engine=skills,
            approval_policy=cli_policy,
        )
        agent = IvyAgent(
            llm_client=llm,
            tool_registry=registry,
            memory_store=store,
            skill_engine=skills,
            skill_generator=skill_gen,
            config=config,
        )
        app = IvyCLIApp(
            config=config,
            llm_client=llm,
            memory_store=store,
            skill_engine=skills,
            agent=agent,
            skill_generator=skill_gen,
            session_id=args.session,
            console=console,
            theme_name=args.theme or config.default_theme,
            env_path=".env",
        )
        cli_policy.approval_callback = app._interactive_approval
        if args.command == "cron":
            await app.handle_slash_command("/cron list")
            return 0

        if args.prompt:
            await app._run_turn_with_live_stream(args.prompt)
            return 0

        await app.run_interactive()
        return 0
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
