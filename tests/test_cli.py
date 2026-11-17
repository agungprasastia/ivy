from unittest.mock import AsyncMock, MagicMock

import pytest
from ivy.config import IvyConfig
from ivy.core.llm import LLMClient
from ivy.core.types import ChatMessage, Role
from ivy.memory.store import MemoryStore
from ivy.skills.engine import SkillEngine
from ivy.skills.format import SkillDefinition
from ivy.tui.app import IvyCLIApp, build_parser
from ivy.tui.prompt import SLASH_COMMANDS, build_slash_completer


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
    cfg = IvyConfig(db_path=str(tmp_path / "tui.db"), skills_dir=str(tmp_path / "skills"))
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
    app = IvyCLIApp(
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
    from ivy.skills.generator import SkillGenerator
    from ivy.tui.app import build_default_tool_registry

    cfg = IvyConfig(db_path=str(tmp_path / "tui2.db"), skills_dir=str(tmp_path / "skills"))
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
    app = IvyCLIApp(
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

    cfg = IvyConfig(db_path=str(tmp_path / "tui3.db"), skills_dir=str(tmp_path / "skills"))
    store = MemoryStore(db_path=cfg.db_path)
    await store.initialize()
    skills = SkillEngine(skills_dir=cfg.skills_dir)
    llm = LLMClient(cfg)

    mock_agent = MagicMock()
    mock_agent.run_turn = AsyncMock(side_effect=RuntimeError("LLM API offline"))

    buf = StringIO()
    console = Console(file=buf, force_terminal=False)
    app = IvyCLIApp(
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

    with patch("ivy.tui.app.create_prompt_session", return_value=fake_session):
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


def test_tui_banner_and_toolbar_render_tools_skills_and_theme(tmp_path):
    from io import StringIO
    from rich.console import Console
    from ivy.tui.views import render_banner

    buf = StringIO()
    console = Console(file=buf, force_terminal=False, width=110)
    render_banner(
        console,
        model="gpt-4o-mini",
        session_id="cli:main",
        cwd=str(tmp_path),
        tool_names=["terminal", "read_file", "write_file", "code_runner", "web_search"],
        skills=[{"name": "coding-assistant", "status": "active", "description": "Code helper"}],
    )
    out = buf.getvalue()
    assert "gpt-4o-mini" in out
    assert "cli:main" in out
    assert "Available Tools" in out
    assert "Available Skills" in out
    assert "coding-assistant" in out
    assert "5 tools" in out


@pytest.mark.asyncio
async def test_tui_theme_switch_and_friendly_auth_error(tmp_path):
    from io import StringIO
    from rich.console import Console
    from ivy.tui.app import configure_logging

    configure_logging(verbose=False, log_level="INFO")
    import logging
    assert logging.getLogger("httpx2").level >= logging.WARNING
    assert logging.getLogger("openai").level >= logging.WARNING

    cfg = IvyConfig(db_path=str(tmp_path / "tui4.db"), skills_dir=str(tmp_path / "skills"))
    store = MemoryStore(db_path=cfg.db_path)
    await store.initialize()
    skills = SkillEngine(skills_dir=cfg.skills_dir)
    llm = LLMClient(cfg)
    buf = StringIO()
    console = Console(file=buf, force_terminal=False, width=100)

    app = IvyCLIApp(
        config=cfg,
        llm_client=llm,
        memory_store=store,
        skill_engine=skills,
        agent=MagicMock(),
        console=console,
    )
    assert await app.handle_slash_command("/theme hermes") is True
    assert app.theme.name == "hermes"
    assert "hermes" in buf.getvalue().lower()

    await store.close()


def test_footer_style_disables_reverse_and_renders_clean_split():
    from ivy.tui.prompt import build_bottom_toolbar, build_prompt_style
    from ivy.tui.theme import get_theme

    theme = get_theme("ivy")
    style = build_prompt_style(theme)
    rules = dict(style.style_rules)
    assert "noreverse" in rules.get("bottom-toolbar", "")
    assert "noreverse" in rules.get("bottom-toolbar.text", "")

    toolbar = build_bottom_toolbar(
        model="gpt-4o-mini",
        session_id="cli:main",
        tool_count=10,
        skill_count=2,
        theme=theme,
        width=100,
    )
    rendered = "".join(text for _, text in toolbar)
    assert "─" in rendered
    assert "gpt-4o-mini" in rendered
    assert "cli:main" in rendered
    assert "/help" in rendered


def test_cli_parser_oneshot_prompt():
    parser = build_parser()
    args = parser.parse_args(["-p", "cek status git"])
    assert args.prompt == "cek status git"


def test_slash_completer_nested_subcommands(tmp_path):
    from prompt_toolkit.document import Document

    skills = SkillEngine(skills_dir=str(tmp_path / "skills"))
    skills.save_skill(
        SkillDefinition(name="uv-helper", description="UV skill", instructions="Step")
    )
    completer = build_slash_completer(skill_engine=skills)

    theme_completions = [
        c.text for c in completer.get_completions(Document("/theme he"), None)
    ]
    assert "hermes" in theme_completions

    skill_completions = [
        c.text for c in completer.get_completions(Document("/skills inspect uv"), None)
    ]
    assert "uv-helper" in skill_completions


@pytest.mark.asyncio
async def test_setup_and_persisted_model_theme_to_env(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("IVY_API_BASE", raising=False)
    monkeypatch.delenv("DEFAULT_MODEL", raising=False)
    monkeypatch.delenv("IVY_THEME", raising=False)

    env_file = tmp_path / ".env"
    cfg = IvyConfig(db_path=str(tmp_path / "tui5.db"), skills_dir=str(tmp_path / "skills"))
    store = MemoryStore(db_path=cfg.db_path)
    await store.initialize()
    skills = SkillEngine(skills_dir=cfg.skills_dir)
    llm = LLMClient(cfg)

    app = IvyCLIApp(
        config=cfg,
        llm_client=llm,
        memory_store=store,
        skill_engine=skills,
        agent=MagicMock(),
        env_path=env_file,
    )
    assert await app.handle_slash_command(
        "/setup http://localhost:20128/v1 sk-9router-xyz openai/gpt-4o"
    ) is True
    assert llm.model == "openai/gpt-4o"
    assert llm.base_url == "http://localhost:20128/v1"
    assert "sk-9router-xyz" in env_file.read_text(encoding="utf-8")

    assert await app.handle_slash_command("/theme ocean") is True
    assert "IVY_THEME=ocean" in env_file.read_text(encoding="utf-8")

    await store.close()


@pytest.mark.asyncio
async def test_ctrl_c_cancels_turn_without_exiting_repl(tmp_path):
    from io import StringIO
    from unittest.mock import patch
    from rich.console import Console

    cfg = IvyConfig(db_path=str(tmp_path / "tui6.db"), skills_dir=str(tmp_path / "skills"))
    store = MemoryStore(db_path=cfg.db_path)
    await store.initialize()
    skills = SkillEngine(skills_dir=cfg.skills_dir)
    llm = LLMClient(cfg)

    mock_agent = MagicMock()
    mock_agent.run_turn = AsyncMock(side_effect=KeyboardInterrupt())

    buf = StringIO()
    console = Console(file=buf, force_terminal=False, width=100)
    app = IvyCLIApp(
        config=cfg,
        llm_client=llm,
        memory_store=store,
        skill_engine=skills,
        agent=mock_agent,
        console=console,
    )

    prompts = iter(["pertanyaan lambat", "/exit"])
    fake_session = MagicMock()
    fake_session.prompt_async = AsyncMock(side_effect=lambda *a, **k: next(prompts))

    with patch("ivy.tui.app.create_prompt_session", return_value=fake_session):
        await app.run_interactive()

    out = buf.getvalue().lower()
    assert "cancel" in out
    assert "goodbye" in out
    await store.close()
