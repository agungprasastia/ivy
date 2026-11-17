"""CLI parser, interactive TUI loop, and gateway entrypoint (`aether`)."""

from __future__ import annotations

import argparse
import asyncio
import logging
from typing import Sequence

from rich.console import Console

from aether_agent.config import AetherConfig, load_config
from aether_agent.core.agent import AetherAgent
from aether_agent.core.llm import LLMClient
from aether_agent.cron.scheduler import CronScheduler
from aether_agent.gateway.server import GatewayServer
from aether_agent.memory.store import MemoryStore
from aether_agent.skills.engine import SkillEngine
from aether_agent.tools.approval import SafetyApprovalPolicy
from aether_agent.tools.files import FileTools
from aether_agent.tools.registry import ToolRegistry
from aether_agent.tools.runner import CodeRunnerTool
from aether_agent.tools.terminal import TerminalTool
from aether_agent.tools.web import WebFetchTool, WebSearchTool
from aether_agent.tui.prompt import create_prompt_session
from aether_agent.tui.views import (
    create_console,
    render_assistant_reply,
    render_banner,
    render_memory_results,
    render_skills_table,
    render_tool_event,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for `aether`."""
    parser = argparse.ArgumentParser(
        prog="aether",
        description="Aether Agent — Autonomous self-improving Python AI agent",
    )
    parser.add_argument("--model", type=str, default=None, help="Override default LLM model")
    parser.add_argument("--session", type=str, default="cli:main", help="Session ID to resume")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    subparsers = parser.add_subparsers(dest="command")

    # `aether gateway start|status`
    gw_parser = subparsers.add_parser("gateway", help="Manage multi-platform messaging gateway")
    gw_sub = gw_parser.add_subparsers(dest="gateway_action")
    gw_sub.add_parser("start", help="Start the async messaging gateway & cron daemon")
    gw_sub.add_parser("status", help="Show configured gateway adapters")

    # `aether model [name]`
    model_parser = subparsers.add_parser("model", help="Show or test configured model")
    model_parser.add_argument("model_name", nargs="?", default=None, help="Model name")

    # `aether skills list|inspect|remove`
    skills_parser = subparsers.add_parser("skills", help="Inspect and manage learned skills")
    skills_sub = skills_parser.add_subparsers(dest="skills_action")
    skills_sub.add_parser("list", help="List available skills")
    inspect_p = skills_sub.add_parser("inspect", help="Inspect a skill's instructions")
    inspect_p.add_argument("name", help="Skill name")
    remove_p = skills_sub.add_parser("remove", help="Remove a skill")
    remove_p.add_argument("name", help="Skill name")

    # `aether cron list`
    cron_parser = subparsers.add_parser("cron", help="Inspect scheduled cron tasks")
    cron_sub = cron_parser.add_subparsers(dest="cron_action")
    cron_sub.add_parser("list", help="List scheduled tasks")

    return parser


def build_default_tool_registry(
    config: AetherConfig, interactive: bool = True, unattended: bool = False
) -> ToolRegistry:
    """Create and populate the default ToolRegistry."""
    registry = ToolRegistry()
    policy = SafetyApprovalPolicy(
        auto_approve=config.auto_approve_safety,
        interactive=interactive,
        unattended=unattended,
    )
    registry.register(TerminalTool(approval_policy=policy))
    FileTools().register_all(registry)
    registry.register(CodeRunnerTool(timeout=config.code_runner_timeout))
    registry.register(WebFetchTool())
    registry.register(WebSearchTool())
    return registry


class AetherCLIApp:
    """Coordinates interactive TUI session and slash command dispatch."""

    def __init__(
        self,
        config: AetherConfig,
        llm_client: LLMClient,
        memory_store: MemoryStore,
        skill_engine: SkillEngine,
        agent: AetherAgent,
        cron_scheduler: CronScheduler | None = None,
        session_id: str = "cli:main",
        console: Console | None = None,
    ) -> None:
        self.config = config
        self.llm_client = llm_client
        self.memory_store = memory_store
        self.skill_engine = skill_engine
        self.agent = agent
        self.cron_scheduler = cron_scheduler or CronScheduler()
        self.session_id = session_id
        self.console = console or create_console()

    async def handle_slash_command(self, raw_line: str) -> bool:
        """Handle a slash command. Returns False to exit interactive loop."""
        parts = raw_line.strip().split()
        cmd = parts[0].lower()
        args = parts[1:]

        if cmd == "/exit":
            self.console.print("[dim]Goodbye![/dim]")
            return False

        if cmd == "/help":
            self.console.print(
                "/model <name> | /skills list|inspect <name>|remove <name> | "
                "/memory <query> | /cron list | /gateway status | /clear | /exit"
            )
            return True

        if cmd == "/model":
            if not args:
                self.console.print(f"Active model: [green]{self.llm_client.model}[/green]")
            else:
                self.llm_client.set_model(args[0])
                self.console.print(f"Switched model to: [green]{self.llm_client.model}[/green]")
            return True

        if cmd == "/clear":
            await self.memory_store.clear_session(self.session_id)
            self.console.print(f"Cleared history for session [yellow]{self.session_id}[/yellow].")
            return True

        if cmd == "/memory":
            query = " ".join(args).strip()
            if not query:
                self.console.print("[yellow]Usage: /memory <search query>[/yellow]")
            else:
                hits = await self.memory_store.search_fts(query)
                render_memory_results(self.console, hits)
            return True

        if cmd == "/skills":
            action = args[0].lower() if args else "list"
            if action == "list":
                render_skills_table(self.console, self.skill_engine.list_skills())
            elif action == "inspect" and len(args) >= 2:
                skill = self.skill_engine.inspect_skill(args[1])
                if skill is None:
                    self.console.print(f"[red]Skill not found:[/red] {args[1]}")
                else:
                    self.console.print(f"[bold cyan]{skill.name}[/bold cyan] ({skill.status})")
                    self.console.print(skill.instructions)
            elif action == "remove" and len(args) >= 2:
                removed = self.skill_engine.remove_skill(args[1])
                msg = f"Removed skill '{args[1]}'" if removed else f"Skill '{args[1]}' not found"
                self.console.print(msg)
            else:
                self.console.print("[yellow]Usage: /skills [list | inspect <name> | remove <name>][/yellow]")
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

        self.console.print(f"[yellow]Unknown command:[/yellow] {cmd} (try /help)")
        return True

    async def run_interactive(self) -> None:
        """Start interactive prompt_toolkit REPL."""
        render_banner(self.console, self.llm_client.model, self.session_id)
        prompt_session = create_prompt_session()

        while True:
            try:
                user_input = await prompt_session.prompt_async("\naether> ")
            except (EOFError, KeyboardInterrupt):
                self.console.print("\n[dim]Exiting Aether Agent.[/dim]")
                break

            text = user_input.strip()
            if not text:
                continue

            if text.startswith("/"):
                keep_running = await self.handle_slash_command(text)
                if not keep_running:
                    break
                continue

            with self.console.status("[cyan]Thinking...[/cyan]"):
                reply = await self.agent.run_turn(
                    prompt=text,
                    session_id=self.session_id,
                    tool_callback=lambda ev: render_tool_event(self.console, ev),
                )
            render_assistant_reply(self.console, reply.content or "")


async def _run_async(args: argparse.Namespace) -> int:
    config = load_config()
    if args.model:
        config.default_model = args.model
    if args.verbose:
        config.verbose = True
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=getattr(logging, config.log_level.upper(), logging.INFO))

    store = MemoryStore(db_path=config.db_path)
    await store.initialize()
    skills = SkillEngine(skills_dir=config.skills_dir)
    await skills.load_skills()
    llm = LLMClient(config=config)
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
            registry = build_default_tool_registry(
                config, interactive=False, unattended=True
            )
            agent = AetherAgent(
                llm_client=llm,
                tool_registry=registry,
                memory_store=store,
                skill_engine=skills,
                config=config,
            )
            cron = CronScheduler(agent=agent, memory_store=store)
            gw = GatewayServer(agent=agent, config=config, cron_scheduler=cron)
            cron.dispatcher = gw.dispatch
            gw.configure_default_adapters()

            if action == "status":
                console.print(f"Allowed users: {len(config.allowed_users)}")
                return 0

            await gw.start()
            console.print("[green]Aether Gateway daemon started. Press Ctrl+C to stop.[/green]")
            try:
                while True:
                    await asyncio.sleep(1.0)
            except (KeyboardInterrupt, asyncio.CancelledError):
                await gw.stop()
            return 0

        registry = build_default_tool_registry(config, interactive=True, unattended=False)
        agent = AetherAgent(
            llm_client=llm,
            tool_registry=registry,
            memory_store=store,
            skill_engine=skills,
            config=config,
        )
        app = AetherCLIApp(
            config=config,
            llm_client=llm,
            memory_store=store,
            skill_engine=skills,
            agent=agent,
            session_id=args.session,
            console=console,
        )
        if args.command == "cron":
            await app.handle_slash_command("/cron list")
            return 0

        await app.run_interactive()
        return 0
    finally:
        await store.close()


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for `aether`."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    return asyncio.run(_run_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
