from unittest.mock import AsyncMock, MagicMock

import pytest
from aether_agent.config import AetherConfig
from aether_agent.core.llm import LLMClient
from aether_agent.core.types import ChatMessage, Role
from aether_agent.memory.store import MemoryStore
from aether_agent.skills.engine import SkillEngine
from aether_agent.skills.format import SkillDefinition
from aether_agent.tui.app import AetherCLIApp, build_parser
from aether_agent.tui.prompt import SLASH_COMMANDS, build_slash_completer


def test_cli_parser_subcommands():
    parser = build_parser()
    args = parser.parse_args(["gateway", "start"])
    assert args.command == "gateway"
    assert args.gateway_action == "start"

    args_default = parser.parse_args(["--model", "gpt-4o", "--verbose"])
    assert args_default.command is None
    assert args_default.model == "gpt-4o"
    assert args_default.verbose is True

    args_skills = parser.parse_args(["skills", "inspect", "coding-assistant"])
    assert args_skills.command == "skills"
    assert args_skills.skills_action == "inspect"
    assert args_skills.name == "coding-assistant"


def test_slash_completer_contains_all_commands():
    completer = build_slash_completer()
    for cmd in ["/model", "/skills", "/gateway", "/cron", "/memory", "/clear", "/exit"]:
        assert cmd in SLASH_COMMANDS
        assert cmd in completer.words


@pytest.mark.asyncio
async def test_slash_command_dispatch(tmp_path):
    cfg = AetherConfig(db_path=str(tmp_path / "tui.db"), skills_dir=str(tmp_path / "skills"))
    store = MemoryStore(db_path=cfg.db_path)
    await store.initialize()

    skills = SkillEngine(skills_dir=cfg.skills_dir)
    skills.save_skill(
        SkillDefinition(
            name="demo-skill",
            description="Demo skill",
            instructions="Step 1",
        )
    )
    llm = LLMClient(cfg)
    app = AetherCLIApp(
        config=cfg,
        llm_client=llm,
        memory_store=store,
        skill_engine=skills,
        agent=MagicMock(),
    )

    assert await app.handle_slash_command("/model claude-sonnet-4-5") is True
    assert llm.model == "claude-sonnet-4-5"

    assert await app.handle_slash_command("/skills list") is True
    assert await app.handle_slash_command("/skills inspect demo-skill") is True
    assert await app.handle_slash_command("/skills remove demo-skill") is True
    assert skills.inspect_skill("demo-skill") is None

    assert await app.handle_slash_command("/clear") is True
    assert await app.handle_slash_command("/exit") is False

    await store.close()


@pytest.mark.asyncio
async def test_tui_tool_registry_has_read_skill_and_reflect_slash_command(tmp_path):
    from aether_agent.skills.generator import SkillGenerator
    from aether_agent.tui.app import build_default_tool_registry

    cfg = AetherConfig(db_path=str(tmp_path / "tui2.db"), skills_dir=str(tmp_path / "skills"))
    skills = SkillEngine(skills_dir=cfg.skills_dir)
    registry = build_default_tool_registry(cfg, skill_engine=skills)
    assert registry.get("read_skill") is not None

    store = MemoryStore(db_path=cfg.db_path)
    await store.initialize()
    mock_llm = MagicMock()
    mock_llm.model = "gpt-4o-mini"
    mock_llm.chat_completion = AsyncMock(
        return_value=ChatMessage(role=Role.ASSISTANT, content="NO_SKILL")
    )
    gen = SkillGenerator(llm_client=mock_llm, skill_engine=skills)
    app = AetherCLIApp(
        config=cfg,
        llm_client=mock_llm,
        memory_store=store,
        skill_engine=skills,
        agent=MagicMock(),
        skill_generator=gen,
    )
    assert await app.handle_slash_command("/skills reflect") is True
    await store.close()


@pytest.mark.asyncio
async def test_tui_interactive_handles_turn_error_and_pauses_spinner(tmp_path):
    from unittest.mock import AsyncMock, patch
    from io import StringIO
    from rich.console import Console

    cfg = AetherConfig(db_path=str(tmp_path / "tui3.db"), skills_dir=str(tmp_path / "skills"))
    store = MemoryStore(db_path=cfg.db_path)
    await store.initialize()
    skills = SkillEngine(skills_dir=cfg.skills_dir)
    llm = LLMClient(cfg)

    mock_agent = MagicMock()
    mock_agent.run_turn = AsyncMock(side_effect=RuntimeError("LLM API offline"))

    buf = StringIO()
    console = Console(file=buf, force_terminal=False)
    app = AetherCLIApp(
        config=cfg,
        llm_client=llm,
        memory_store=store,
        skill_engine=skills,
        agent=mock_agent,
        console=console,
    )

    prompts = iter(["Halo", "/exit"])
    fake_session = MagicMock()
    fake_session.prompt_async = AsyncMock(side_effect=lambda *a, **k: next(prompts))

    with patch("aether_agent.tui.app.create_prompt_session", return_value=fake_session):
        await app.run_interactive()

    assert "LLM API offline" in buf.getvalue()

    # Interactive approval callback pauses active status before prompting
    mock_status = MagicMock()
    app._active_status = mock_status
    with patch("builtins.input", return_value="y"):
        approved = await app._interactive_approval("rm -rf /tmp/x")
    assert approved is True
    assert mock_status.stop.called
    assert mock_status.start.called

    await store.close()
