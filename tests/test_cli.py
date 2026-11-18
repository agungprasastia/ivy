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


@pytest.mark.asyncio
async def test_live_stream_renders_rich_markdown_not_raw_asterisks(tmp_path):
    from io import StringIO
    from rich.console import Console

    cfg = IvyConfig(db_path=str(tmp_path / "md.db"), skills_dir=str(tmp_path / "skills"))
    store = MemoryStore(db_path=cfg.db_path)
    await store.initialize()
    skills = SkillEngine(skills_dir=cfg.skills_dir)
    llm = LLMClient(cfg)

    async def fake_stream_turn(prompt, session_id="default", stream=False, stream_callback=None, tool_callback=None, **kwargs):
        for piece in ["Halo! ", "**Inspeksi kode** ", "`ivy`"]:
            if stream_callback:
                stream_callback(piece)
        return ChatMessage(role=Role.ASSISTANT, content="Halo! **Inspeksi kode** `ivy`")

    mock_agent = MagicMock()
    mock_agent.run_turn = AsyncMock(side_effect=fake_stream_turn)

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
    await app._run_turn_with_live_stream("halo")
    rendered = buf.getvalue()
    assert "Inspeksi kode" in rendered
    assert "**Inspeksi kode**" not in rendered
    await store.close()


@pytest.mark.asyncio
async def test_session_and_new_slash_commands(tmp_path):
    from io import StringIO
    from rich.console import Console

    cfg = IvyConfig(db_path=str(tmp_path / "sess_tui.db"), skills_dir=str(tmp_path / "skills"))
    store = MemoryStore(db_path=cfg.db_path)
    await store.initialize()
    await store.save_message("cli:main", ChatMessage(role=Role.USER, content="old"))

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
    assert app.session_id == "cli:main"

    assert await app.handle_slash_command("/session work") is True
    assert app.session_id == "work"

    assert await app.handle_slash_command("/session list") is True
    assert "cli:main" in buf.getvalue()

    assert await app.handle_slash_command("/new") is True
    assert app.session_id != "work"
    assert app.session_id.startswith("cli:")

    await store.close()


def test_hermes_ux_helpers_completion_toolbar_and_parser(tmp_path):
    from prompt_toolkit.completion import CompleteEvent
    from prompt_toolkit.document import Document
    from ivy.tui.helpers import expand_at_mentions, resolve_slash_command
    from ivy.tui.prompt import SLASH_COMMANDS, build_bottom_toolbar

    # Exact, alias, unique prefix, ambiguous prefix, typo
    assert resolve_slash_command("/compress", SLASH_COMMANDS)[0] == "/compact"
    assert resolve_slash_command("/branch", SLASH_COMMANDS)[0] == "/fork"
    assert resolve_slash_command("/compa", SLASH_COMMANDS)[:3:2] == ("/compact", "prefix")
    cmd, cands, mode = resolve_slash_command("/s", SLASH_COMMANDS)
    assert cmd is None and mode == "ambiguous" and "/session" in cands
    cmd2, sugg, mode2 = resolve_slash_command("/modle", SLASH_COMMANDS)
    assert cmd2 is None and mode2 == "unknown" and "/model" in sugg

    # @file mention completion & expansion
    sample = tmp_path / "hello.py"
    sample.write_text("print('ivy')\n", encoding="utf-8")
    completer = build_slash_completer(cwd=str(tmp_path))
    items = [
        c.text for c in completer.get_completions(Document("check @hel", cursor_position=10), CompleteEvent())
    ]
    assert "@hello.py" in items

    augmented, attached = expand_at_mentions("inspect @hello.py", cwd=str(tmp_path))
    assert "@hello.py (2L)" in attached
    assert "print('ivy')" in augmented

    # Rich toolbar context bar, YOLO & FOCUS badges, width shedding
    toolbar = build_bottom_toolbar(
        model="gpt-4o-mini",
        session_id="cli:main",
        ctx_percent=45.0,
        total_tokens=2500,
        last_duration_s=1.4,
        yolo=True,
        focus_mode=True,
        session_title="Refactor UX",
        git_branch="master",
        width=120,
    )
    rendered = "".join(txt for _, txt in toolbar)
    assert "45%" in rendered
    assert "2.5k tok" in rendered
    assert "YOLO" in rendered
    assert "FOCUS" in rendered
    assert "Refactor UX" in rendered

    # -c / --continue parser flag
    args = build_parser().parse_args(["-c"])
    assert args.continue_latest is True


@pytest.mark.asyncio
async def test_hermes_ux_slash_commands_bang_and_approval_flow(tmp_path):
    from io import StringIO
    from unittest.mock import patch
    from rich.console import Console
    from ivy.tools.approval import SafetyApprovalPolicy

    cfg = IvyConfig(db_path=str(tmp_path / "ux_cli.db"), skills_dir=str(tmp_path / "skills"))
    store = MemoryStore(db_path=cfg.db_path)
    await store.initialize()
    await store.save_message("cli:main", ChatMessage(role=Role.USER, content="hello ivy"))
    await store.save_message("cli:main", ChatMessage(role=Role.ASSISTANT, content="hi from ivy!"))

    skills = SkillEngine(skills_dir=cfg.skills_dir)
    llm = LLMClient(cfg)
    mock_agent = MagicMock()
    mock_agent.run_turn = AsyncMock(
        return_value=ChatMessage(
            role=Role.ASSISTANT,
            content="retried ok",
            metadata={"turn_usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "api_calls": 1}},
        )
    )
    policy = SafetyApprovalPolicy(auto_approve=False)

    buf = StringIO()
    console = Console(file=buf, force_terminal=False, width=110)
    app = IvyCLIApp(
        config=cfg,
        llm_client=llm,
        memory_store=store,
        skill_engine=skills,
        agent=mock_agent,
        console=console,
        approval_policy=policy,
    )

    # /title, /status, /context, /usage, /history, /verbose, /focus, /yolo
    assert await app.handle_slash_command("/title UX Upgrade") is True
    assert await app.handle_slash_command("/status") is True
    assert await app.handle_slash_command("/context") is True
    assert await app.handle_slash_command("/usage") is True
    assert await app.handle_slash_command("/history 10") is True
    assert await app.handle_slash_command("/verbose verbose") is True
    assert app.display_mode == "verbose"
    assert await app.handle_slash_command("/focus on") is True
    assert app.display_mode == "focus"
    assert await app.handle_slash_command("/yolo") is True
    assert policy.auto_approve is True
    assert await app.handle_slash_command("/yolo") is True
    assert policy.auto_approve is False

    # /save md & /save json
    md_out = tmp_path / "out.md"
    json_out = tmp_path / "out.json"
    assert await app.handle_slash_command(f"/save md {md_out}") is True
    assert await app.handle_slash_command(f"/save json {json_out}") is True
    assert "hi from ivy!" in md_out.read_text(encoding="utf-8")
    assert '"session_id": "cli:main"' in json_out.read_text(encoding="utf-8")

    # /copy
    with patch("ivy.tui.app.copy_to_clipboard", return_value=(True, "mock")) as mock_clip:
        assert await app.handle_slash_command("/copy") is True
        mock_clip.assert_called_once_with("hi from ivy!")

    # /fork + /undo + /retry + /compact
    assert await app.handle_slash_command("/fork exp") is True
    assert app.session_id == "cli:exp"
    assert await app.handle_slash_command("/retry") is True
    mock_agent.run_turn.assert_awaited()
    assert await app.handle_slash_command("/undo") is True
    assert await app.handle_slash_command("/compact") is True

    # Typo suggestion
    assert await app.handle_slash_command("/modle") is True
    assert "did you mean /model" in buf.getvalue()

    # Bang shell escape (0 tokens, skips agent)
    calls_before = mock_agent.run_turn.await_count
    await app._run_bang_command("echo bang_ok")
    assert "bang_ok" in buf.getvalue()
    assert mock_agent.run_turn.await_count == calls_before

    # Rich _interactive_approval [s] and [t]
    with patch("builtins.input", return_value="s"):
        assert await app._interactive_approval("rm -rf x") == "session"
    inputs = iter(["t", "use python pathlib instead"])
    with patch("builtins.input", side_effect=lambda *_: next(inputs)):
        assert await app._interactive_approval("rm -rf y") == (False, "use python pathlib instead")

    await store.close()


@pytest.mark.asyncio
async def test_interactive_picker_state_session_completion_and_picker_flow(tmp_path):
    from io import StringIO
    from unittest.mock import patch
    from prompt_toolkit.completion import CompleteEvent
    from prompt_toolkit.document import Document
    from rich.console import Console
    from ivy.tui.helpers import format_relative_time
    from ivy.tui.picker import PickerOption, PickerResult, PickerState, render_picker_fragments

    # 1. PickerState filtering, navigation, and custom entries
    opts = [
        PickerOption(value="cli:main", label="Bugfix Login", meta="cli:main · 12m · 5m ago"),
        PickerOption(value="cli:work", label="Refactor API", meta="cli:work · 4m · yesterday"),
    ]
    st = PickerState(options=opts, current_value="cli:work", allow_custom=True)
    assert st.selected is not None and st.selected.value == "cli:work"
    st.move(-1)
    assert st.selected is not None and st.selected.value == "cli:main"
    st.apply_filter("refactor")
    assert len(st.items) == 2  # matched item + 'Use custom: refactor'
    assert st.items[0].value == "cli:work"
    rendered = "".join(txt for _, txt in render_picker_fragments("Switch Session", st, allow_new=True, allow_delete=True))
    assert "Refactor API" in rendered
    assert "Ctrl+N new" in rendered
    assert "Del delete" in rendered

    # 2. Relative timestamp formatting
    assert format_relative_time(None) == ""

    # 3. Session autocomplete dropdown under `/session `
    completer = build_slash_completer(
        session_provider=lambda: [{"session_id": "cli:alpha", "title": "Alpha task", "message_count": 3}]
    )
    completions = list(completer.get_completions(Document("/session cli:", cursor_position=13), CompleteEvent()))
    assert any(c.text == "cli:alpha" and "Alpha task" in str(c.display_meta) for c in completions)

    # 4. Interactive /session picker flow (delete -> new -> select)
    cfg = IvyConfig(db_path=str(tmp_path / "picker.db"), skills_dir=str(tmp_path / "skills"))
    store = MemoryStore(db_path=cfg.db_path)
    await store.initialize()
    await store.save_message("cli:main", ChatMessage(role=Role.USER, content="first"))
    await store.save_message("cli:other", ChatMessage(role=Role.USER, content="second"))
    app = IvyCLIApp(
        config=cfg,
        llm_client=LLMClient(cfg),
        memory_store=store,
        skill_engine=SkillEngine(skills_dir=cfg.skills_dir),
        agent=MagicMock(),
        console=Console(file=StringIO(), force_terminal=False, width=100),
    )

    actions = iter([
        PickerResult(action="delete", value="cli:main"),
        PickerResult(action="select", value="cli:other"),
    ])
    with patch.object(app, "_can_open_picker", return_value=True), patch(
        "ivy.tui.app.run_interactive_picker", new=AsyncMock(side_effect=lambda **kw: next(actions))
    ):
        assert await app.handle_slash_command("/session") is True

    assert app.session_id == "cli:other"
    assert {s["session_id"] for s in await store.list_sessions()} == {"cli:other"}
    await store.close()


@pytest.mark.asyncio
async def test_phase2_think_folding_doctor_btw_rewind_stash_and_cron_crud(tmp_path):
    from io import StringIO
    from unittest.mock import patch
    from rich.console import Console
    from ivy.tools.todo import TodoStore
    from ivy.tui.doctor import run_doctor_checks
    from ivy.tui.helpers import extract_thinking_blocks

    # 1. Thinking block extraction (closed + streaming open tag)
    vis1, th1 = extract_thinking_blocks("<think>step 1\nstep 2</think>\nFinal answer!")
    assert vis1 == "Final answer!"
    assert "step 1" in th1
    vis2, th2 = extract_thinking_blocks("<think>still thinking...")
    assert vis2 == ""
    assert "still thinking..." in th2

    # 2. Setup app with MemoryStore + Mock LLM/Agent
    env_file = tmp_path / ".env"
    skills_dir = tmp_path / "missing_skills"
    cfg = IvyConfig(db_path=str(tmp_path / "p2_cli.db"), skills_dir=str(skills_dir))
    store = MemoryStore(db_path=cfg.db_path)
    await store.initialize()
    await store.save_message("cli:main", ChatMessage(role=Role.USER, content="turn 1"))
    await store.save_message("cli:main", ChatMessage(role=Role.ASSISTANT, content="reply 1"))
    await store.save_message("cli:main", ChatMessage(role=Role.USER, content="turn 2"))
    await store.save_message("cli:main", ChatMessage(role=Role.ASSISTANT, content="reply 2"))

    skills = SkillEngine(skills_dir=str(skills_dir))
    llm = LLMClient(cfg)
    mock_agent = MagicMock()

    buf = StringIO()
    console = Console(file=buf, force_terminal=False, width=110)
    app = IvyCLIApp(
        config=cfg,
        llm_client=llm,
        memory_store=store,
        skill_engine=skills,
        agent=mock_agent,
        console=console,
        env_path=env_file,
    )

    # 3. Doctor --fix creates missing .env and skills directory
    findings = await run_doctor_checks(
        cfg, llm, store, skills, live_probe=False, fix=True, env_path=env_file
    )
    assert env_file.is_file() and skills_dir.is_dir()
    assert any(f.name == "SQLite & FTS5" and f.status == "ok" for f in findings)
    assert await app.handle_slash_command("/doctor --no-live") is True
    assert "Ivy Doctor Diagnostics" in buf.getvalue()

    # 4. Reasoning mode & streaming <think> folding
    assert await app.handle_slash_command("/reasoning collapsed") is True

    async def fake_think_turn(**kwargs):
        cb = kwargs["stream_callback"]
        cb("<think>secret thought</think>\nClean visible output")
        return ChatMessage(role=Role.ASSISTANT, content="<think>secret thought</think>\nClean visible output")

    mock_agent.run_turn = AsyncMock(side_effect=fake_think_turn)
    await app._run_turn_with_live_stream("test think")
    assert "Clean visible output" in buf.getvalue()
    assert app._last_reasoning == "secret thought"
    assert await app.handle_slash_command("/reasoning show") is True
    assert "secret thought" in buf.getvalue()

    # 5. /btw runs off-record without persisting to MemoryStore
    before_count = len(await store.get_messages("cli:main"))
    with patch.object(
        llm, "chat_completion", new=AsyncMock(return_value=ChatMessage(role=Role.ASSISTANT, content="Side info!"))
    ):
        assert await app.handle_slash_command("/btw what time?") is True
    assert len(await store.get_messages("cli:main")) == before_count
    assert "Side info!" in buf.getvalue()

    # 6. /profile set/list/clear & /todo
    assert await app.handle_slash_command("/profile set editor vscode") is True
    assert (await store.get_user_profile())["editor"] == "vscode"
    assert await app.handle_slash_command("/profile list") is True
    assert await app.handle_slash_command("/profile clear") is True
    assert await store.get_user_profile() == {}

    app.todo_store = TodoStore()
    app.todo_store.write("cli:main", [{"content": "Ship Phase 2", "status": "in_progress"}])
    assert await app.handle_slash_command("/todo") is True
    assert "Ship Phase 2" in buf.getvalue()

    # 7. /rewind 1 rolls back turns and prefills turn #1 prompt
    assert await app.handle_slash_command("/rewind 1") is True
    assert app._pending_prefill == "turn 1"
    assert await store.get_messages("cli:main") == []

    # 8. Ctrl+S prompt stash push & pop
    app._handle_stash_trigger("unfinished draft prompt")
    app._pending_prefill = ""
    app._handle_stash_trigger("")
    assert app._pending_prefill == "unfinished draft prompt"

    # 9. /cron add, pause, resume, remove persisted to SQLite
    assert await app.handle_slash_command('/cron add nightly "0 2 * * *" "Check builds"') is True
    assert len(app.cron_scheduler.list_tasks()) == 1
    assert (await store.list_cron_jobs())[0]["name"] == "nightly"
    assert await app.handle_slash_command("/cron pause nightly") is True
    assert (await store.list_cron_jobs())[0]["enabled"] is False
    assert await app.handle_slash_command("/cron resume nightly") is True
    assert (await store.list_cron_jobs())[0]["enabled"] is True
    assert await app.handle_slash_command("/cron remove nightly") is True
    assert await store.list_cron_jobs() == []

    await store.close()


@pytest.mark.asyncio
async def test_phase3_rollback_init_review_goal_and_mcp_slash_commands(tmp_path, monkeypatch):
    import sys
    from io import StringIO
    from pathlib import Path
    from unittest.mock import patch
    from rich.console import Console
    from ivy.tools.checkpoint import FileCheckpointStore
    from ivy.tools.files import FileTools

    monkeypatch.chdir(tmp_path)
    cfg = IvyConfig(db_path=str(tmp_path / "p3_cli.db"), skills_dir=str(tmp_path / "skills"))
    store = MemoryStore(db_path=cfg.db_path)
    await store.initialize()
    skills = SkillEngine(skills_dir=cfg.skills_dir)
    llm = LLMClient(cfg)

    cp_store = FileCheckpointStore()
    ft = FileTools(checkpoint_store=cp_store, session_id_getter=lambda: "cli:main")
    target = tmp_path / "sample.py"
    target.write_text("a = 1\n", encoding="utf-8")
    await ft.patch_tool.execute(path=str(target), target="a = 1", replacement="a = 2")

    mock_agent = MagicMock()
    mock_agent.run_turn = AsyncMock(return_value=ChatMessage(role=Role.ASSISTANT, content="Done"))
    buf = StringIO()
    console = Console(file=buf, force_terminal=False, width=110)

    app = IvyCLIApp(
        config=cfg,
        llm_client=llm,
        memory_store=store,
        skill_engine=skills,
        agent=mock_agent,
        console=console,
    )
    app.checkpoint_store = cp_store

    # 1. /rollback list, diff, restore & /undo hint
    assert await app.handle_slash_command("/rollback list") is True
    assert "sample.py" in buf.getvalue()
    assert await app.handle_slash_command("/rollback diff 1") is True
    assert "+a = 2" in buf.getvalue()
    assert await app.handle_slash_command("/rollback restore 1") is True
    assert target.read_text(encoding="utf-8") == "a = 1\n"

    # 2. /init triggers structured AGENTS.md turn
    assert await app.handle_slash_command("/init focus on tests") is True
    sent_prompt = mock_agent.run_turn.call_args.kwargs["prompt"]
    assert "AGENTS.md" in sent_prompt and "focus on tests" in sent_prompt

    # 3. /review audits diff cleanly
    with patch("ivy.tui.app.run_git_diff", return_value="diff --git a/x.py b/x.py\n+print(1)"), patch.object(
        llm, "chat_completion", new=AsyncMock(return_value=ChatMessage(role=Role.ASSISTANT, content="LGTM: no bugs"))
    ):
        assert await app.handle_slash_command("/review security") is True
    assert "LGTM: no bugs" in buf.getvalue()

    # 4. /goal retries when gate fails first and stops once gate passes
    flag = tmp_path / "gate.flag"
    checker = tmp_path / "check_gate.py"
    checker.write_text(
        "import pathlib, sys\nsys.exit(0 if pathlib.Path('gate.flag').exists() else 1)\n",
        encoding="utf-8",
    )
    gate_cmd = f"{Path(sys.executable).as_posix()} check_gate.py"
    turn_idx = 0

    async def fake_goal_turn(**kwargs):
        nonlocal turn_idx
        turn_idx += 1
        if turn_idx >= 2:
            flag.write_text("ok", encoding="utf-8")
        return ChatMessage(role=Role.ASSISTANT, content=f"Attempt {turn_idx}")

    mock_agent.run_turn = AsyncMock(side_effect=fake_goal_turn)
    assert await app.handle_slash_command(f'/goal Fix gate --gate "{gate_cmd}" --max-turns 4') is True
    assert turn_idx == 2
    assert "Verification gate passed" in buf.getvalue()

    # 5. /mcp list
    assert await app.handle_slash_command("/mcp list") is True
    assert "No MCP servers loaded" in buf.getvalue()

    await store.close()


def test_expand_at_mentions_supports_urls():
    from ivy.tui.helpers import expand_at_mentions

    prompt, attached = expand_at_mentions(
        "Check @url:https://example.com/docs and @https://example.org/spec"
    )
    assert "https://example.com/docs" in prompt
    assert "https://example.org/spec" in prompt
    assert any("example.com/docs" in item for item in attached)
    assert any("example.org/spec" in item for item in attached)


@pytest.mark.asyncio
async def test_save_command_and_exporters_redact_secrets(tmp_path, monkeypatch):
    from io import StringIO
    from pathlib import Path
    from rich.console import Console
    from ivy.tui.helpers import export_session_json, export_session_markdown

    secret_sk = "sk-proj-secret1234567890abcdef12"
    secret_bearer = "tok_secret_bearer_value_998877"
    msgs = [
        ChatMessage(role=Role.USER, content=f"OPENAI_API_KEY={secret_sk}"),
        ChatMessage(
            role=Role.ASSISTANT,
            content=f"Authorization: Bearer {secret_bearer}",
        ),
    ]

    md_text = export_session_markdown("cli:sec", "Secret Check", msgs)
    json_text = export_session_json("cli:sec", "Secret Check", msgs)
    for body in (md_text, json_text):
        assert secret_sk not in body
        assert secret_bearer not in body

    monkeypatch.chdir(tmp_path)
    store = MemoryStore(db_path=str(tmp_path / "save_sec.db"))
    await store.initialize()
    await store.ensure_session("cli:sec")
    for m in msgs:
        await store.save_message("cli:sec", m)

    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    skills = SkillEngine(skills_dir=str(skills_dir))
    await skills.load_skills()
    cfg = IvyConfig(
        openai_api_key="test-key",
        db_path=str(tmp_path / "save_sec.db"),
        skills_dir=str(skills_dir),
    )
    llm = LLMClient(config=cfg)
    mock_agent = MagicMock()
    mock_agent.tool_registry = None
    buf = StringIO()
    app = IvyCLIApp(
        config=cfg,
        llm_client=llm,
        memory_store=store,
        skill_engine=skills,
        agent=mock_agent,
        session_id="cli:sec",
        console=Console(file=buf, force_terminal=False, width=120),
    )

    assert await app.handle_slash_command("/save md") is True
    assert await app.handle_slash_command("/save json") is True
    exported = list(Path(tmp_path).glob("ivy_*"))
    assert len(exported) == 2
    for fp in exported:
        content = fp.read_text(encoding="utf-8")
        assert secret_sk not in content
        assert secret_bearer not in content

    await store.close()


@pytest.mark.asyncio
async def test_ps_slash_command_lists_logs_and_stops_processes(tmp_path):
    import asyncio
    import sys
    from io import StringIO
    from rich.console import Console
    from ivy.tools.process_registry import BackgroundProcessManager, ProcessManageTool
    from ivy.tools.registry import ToolRegistry

    mgr = BackgroundProcessManager()
    reg = ToolRegistry()
    reg.register(ProcessManageTool(manager=mgr))
    script = tmp_path / "ps_alive.py"
    script.write_text("import time; print('bg-alive', flush=True); time.sleep(10)\n", encoding="utf-8")
    cmd = f'& "{sys.executable}" "{script}"' if sys.platform == "win32" else f'"{sys.executable}" "{script}"'
    bp = await mgr.spawn(cmd, working_dir=str(tmp_path))

    deadline = asyncio.get_running_loop().time() + 5.0
    _, out = mgr.read_output(bp.process_id)
    while "bg-alive" not in out and asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(0.1)
        _, out = mgr.read_output(bp.process_id)

    store = MemoryStore(db_path=str(tmp_path / "ps.db"))
    await store.initialize()
    skills = SkillEngine(skills_dir=str(tmp_path / "skills"))
    cfg = IvyConfig(db_path=str(tmp_path / "ps.db"), skills_dir=str(tmp_path / "skills"))
    mock_agent = MagicMock()
    mock_agent.tool_registry = reg
    buf = StringIO()
    app = IvyCLIApp(
        config=cfg,
        llm_client=LLMClient(cfg),
        memory_store=store,
        skill_engine=skills,
        agent=mock_agent,
        console=Console(file=buf, force_terminal=False, width=110),
    )

    assert "/ps" in SLASH_COMMANDS
    assert await app.handle_slash_command("/ps list") is True
    assert bp.process_id in buf.getvalue()
    assert await app.handle_slash_command(f"/ps log {bp.process_id}") is True
    assert "bg-alive" in buf.getvalue()
    assert await app.handle_slash_command(f"/ps stop {bp.process_id}") is True

    await mgr.shutdown_all()
    await store.close()
