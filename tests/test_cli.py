from unittest.mock import MagicMock

import pytest
from aether_agent.config import AetherConfig
from aether_agent.core.llm import LLMClient
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
