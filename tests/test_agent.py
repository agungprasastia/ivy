from unittest.mock import AsyncMock, MagicMock

import pytest
from ivy.core.agent import IvyAgent
from ivy.core.subagent import SubagentRunner
from ivy.core.types import ChatMessage, Role, ToolCall, ToolResult
from ivy.memory.store import MemoryStore
from ivy.skills.engine import SkillEngine
from ivy.skills.format import SkillDefinition
from ivy.tools.base import BaseTool
from ivy.tools.registry import ToolRegistry


@pytest.mark.asyncio
async def test_agent_react_turn_execution():
    mock_llm = MagicMock()
    mock_llm.chat_completion = AsyncMock(
        side_effect=[
            ChatMessage(
                role=Role.ASSISTANT,
                content="",
                tool_calls=[
                    ToolCall(
                        id="call_1",
                        function_name="terminal",
                        arguments={"command": "echo 42"},
                    )
                ],
            ),
            ChatMessage(role=Role.ASSISTANT, content="Hasil perhitungan adalah 42."),
        ]
    )

    mock_tools = MagicMock()
    mock_tools.execute = AsyncMock(
        return_value=ToolResult(tool_call_id="call_1", tool_name="terminal", status="success", output="42")
    )
    mock_tools.get_schemas = MagicMock(return_value=[])

    agent = IvyAgent(llm_client=mock_llm, tool_registry=mock_tools)
    response = await agent.run_turn(prompt="Hitung 42", session_id="test-session")

    assert "42" in (response.content or "")
    assert mock_tools.execute.called


@pytest.mark.asyncio
async def test_agent_handles_rejected_tool_without_crashing():
    mock_llm = MagicMock()
    mock_llm.chat_completion = AsyncMock(
        side_effect=[
            ChatMessage(
                role=Role.ASSISTANT,
                content="",
                tool_calls=[
                    ToolCall(
                        id="call_bad",
                        function_name="terminal",
                        arguments={"command": "rm -rf /"},
                    )
                ],
            ),
            ChatMessage(
                role=Role.ASSISTANT,
                content="Perintah ditolak oleh safety guard, saya akan menggunakan cara aman.",
            ),
        ]
    )

    mock_tools = MagicMock()
    mock_tools.execute = AsyncMock(
        return_value=ToolResult(
            tool_call_id="call_bad",
            tool_name="terminal",
            status="rejected",
            error="Command rejected by safety approval policy",
            output="Command execution rejected by user/safety policy",
        )
    )
    mock_tools.get_schemas = MagicMock(return_value=[])

    agent = IvyAgent(llm_client=mock_llm, tool_registry=mock_tools)
    response = await agent.run_turn(prompt="Hapus root", session_id="test-reject")
    assert "ditolak" in (response.content or "").lower()


@pytest.mark.asyncio
async def test_agent_memory_and_skills_integration(tmp_path):
    store = MemoryStore(db_path=str(tmp_path / "agent.db"))
    await store.initialize()
    await store.set_user_preference("language", "Indonesian")

    skills = SkillEngine(skills_dir=str(tmp_path / "skills"))
    skills.save_skill(
        SkillDefinition(
            name="uv-helper",
            description="Manage Python projects with uv",
            instructions="Use uv sync.",
        )
    )

    captured_messages = []

    async def fake_completion(messages, tools=None, stream=False, **kwargs):
        captured_messages.append(messages)
        return ChatMessage(role=Role.ASSISTANT, content="Siap!")

    mock_llm = MagicMock()
    mock_llm.chat_completion = AsyncMock(side_effect=fake_completion)

    agent = IvyAgent(
        llm_client=mock_llm,
        tool_registry=ToolRegistry(),
        memory_store=store,
        skill_engine=skills,
    )
    resp = await agent.run_turn(prompt="Halo Ivy", session_id="sess-int")
    assert resp.content == "Siap!"
    system_prompt = captured_messages[0][0].content or ""
    assert "uv-helper" in system_prompt
    assert "Indonesian" in system_prompt

    saved = await store.get_messages("sess-int")
    assert len(saved) == 2
    await store.close()


@pytest.mark.asyncio
async def test_subagent_runner_isolation(tmp_path):
    class EchoTool(BaseTool):
        name = "echo_tool"
        description = "Echo"

        async def execute(self, text: str = "", **kwargs):
            return ToolResult(tool_name=self.name, status="success", output=text)

    class SecretTool(BaseTool):
        name = "secret_tool"
        description = "Secret"

        async def execute(self, **kwargs):
            return ToolResult(tool_name=self.name, status="success", output="secret")

    registry = ToolRegistry()
    registry.register(EchoTool())
    registry.register(SecretTool())

    store = MemoryStore(db_path=str(tmp_path / "sub.db"))
    await store.initialize()

    mock_llm = MagicMock()
    mock_llm.chat_completion = AsyncMock(
        return_value=ChatMessage(role=Role.ASSISTANT, content="Subagent summary result")
    )

    runner = SubagentRunner(
        llm_client=mock_llm,
        tool_registry=registry,
        memory_store=store,
        max_iterations=4,
    )
    summary = await runner.run(
        task_prompt="Analyze logs",
        parent_session_id="parent-1",
        allowed_tools=["echo_tool"],
    )
    assert "Subagent summary result" in summary

    # Internal child session messages must NOT pollute parent FTS search
    fts_hits = await store.search_fts("Analyze logs")
    assert len(fts_hits) == 0
    await store.close()


@pytest.mark.asyncio
async def test_agent_auto_reflects_after_tool_turn(tmp_path):
    from ivy.skills.generator import SkillGenerator

    skills = SkillEngine(skills_dir=str(tmp_path / "skills"))
    mock_llm = MagicMock()
    mock_llm.chat_completion = AsyncMock(
        side_effect=[
            ChatMessage(
                role=Role.ASSISTANT,
                content="",
                tool_calls=[
                    ToolCall(id="c1", function_name="terminal", arguments={"command": "echo hi"})
                ],
            ),
            ChatMessage(role=Role.ASSISTANT, content="Done."),
            ChatMessage(
                role=Role.ASSISTANT,
                content="---\nname: echo-check\ndescription: Echo check\nstatus: draft\n---\n1. Run echo\n",
            ),
        ]
    )
    mock_tools = MagicMock()
    mock_tools.execute = AsyncMock(
        return_value=ToolResult(tool_call_id="c1", tool_name="terminal", status="success", output="hi")
    )
    mock_tools.get_schemas = MagicMock(return_value=[])

    gen = SkillGenerator(llm_client=mock_llm, skill_engine=skills)
    agent = IvyAgent(llm_client=mock_llm, tool_registry=mock_tools, skill_generator=gen)
    await agent.run_turn(prompt="Echo hi", session_id="s-reflect")
    assert skills.inspect_skill("echo-check") is not None


@pytest.mark.asyncio
async def test_agent_run_turn_streams_tokens_when_stream_enabled():
    from ivy.config import IvyConfig
    from ivy.core.llm import LLMClient

    cfg = IvyConfig(default_model="gpt-4o-mini", openai_api_key="sk-test")
    llm = LLMClient(cfg)
    llm.stream_turn = AsyncMock(
        side_effect=lambda **kw: (
            kw["on_token"]("Hai ") or kw["on_token"]("dunia!")
            or ChatMessage(role=Role.ASSISTANT, content="Hai dunia!")
        )
    )
    agent = IvyAgent(llm_client=llm, tool_registry=ToolRegistry())
    chunks = []
    reply = await agent.run_turn(
        prompt="Halo",
        session_id="stream-1",
        stream=True,
        stream_callback=chunks.append,
    )
    assert reply.content == "Hai dunia!"
    assert "".join(chunks) == "Hai dunia!"


@pytest.mark.asyncio
async def test_agent_does_not_persist_user_message_on_llm_error(tmp_path):
    store = MemoryStore(db_path=str(tmp_path / "err.db"))
    await store.initialize()

    mock_llm = MagicMock()
    mock_llm.chat_completion = AsyncMock(side_effect=RuntimeError("401 Unauthorized"))

    agent = IvyAgent(
        llm_client=mock_llm,
        tool_registry=ToolRegistry(),
        memory_store=store,
    )
    with pytest.raises(RuntimeError):
        await agent.run_turn(prompt="halo", session_id="err-sess")

    assert await store.get_messages("err-sess") == []
    await store.close()


def test_system_prompt_includes_working_directory(tmp_path, monkeypatch):
    from ivy.core.prompt import build_system_prompt

    monkeypatch.chdir(tmp_path)
    prompt = build_system_prompt()
    assert str(tmp_path) in prompt


def test_system_prompt_includes_workspace_snapshot_context_files_and_guidance(tmp_path, monkeypatch):
    from ivy.core.prompt import build_system_prompt

    (tmp_path / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    (tmp_path / "uv.lock").write_text("", encoding="utf-8")
    (tmp_path / "AGENTS.md").write_text("Always run pytest after edits.", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    prompt = build_system_prompt(
        model="oc/deepseek-v4.1-flash-free",
        platform="cli",
        session_id="cli:main",
        tool_names=["read_file", "search_files", "terminal"],
    )
    assert "Be direct" in prompt
    assert "Finishing the job" in prompt
    assert "Parallel tool calls" in prompt
    assert "Execution discipline" in prompt
    assert f"Root: {tmp_path}" in prompt
    assert "uv" in prompt
    assert "Always run pytest after edits." in prompt
    assert "Model: oc/deepseek-v4.1-flash-free" in prompt


def test_system_prompt_marks_workspace_authoritative_over_upstream_proxies():
    from ivy.core.prompt import build_system_prompt

    prompt = build_system_prompt(model="oc/deepseek-v4.1-flash-free")
    assert "Authoritative" in prompt or "authoritative" in prompt
    assert "router" in prompt.lower() or "proxy" in prompt.lower()


@pytest.mark.asyncio
async def test_agent_requests_graceful_summary_when_max_iterations_reached(tmp_path):
    store = MemoryStore(db_path=str(tmp_path / "maxiter.db"))
    await store.initialize()

    class PingTool(BaseTool):
        name = "ping_tool"
        description = "Ping"
        parameters = {"type": "object", "properties": {}}

        async def execute(self, **kwargs):
            return ToolResult(tool_name=self.name, status="success", output="pong")

    reg = ToolRegistry()
    reg.register(PingTool())

    calls = []

    async def fake_completion(messages, tools=None, stream=False):
        calls.append({"tools": tools})
        if tools:
            return ChatMessage(
                role=Role.ASSISTANT,
                content="",
                tool_calls=[ToolCall(id=f"c{len(calls)}", function_name="ping_tool", arguments={})],
            )
        return ChatMessage(role=Role.ASSISTANT, content="Completed pings and summarized.")

    mock_llm = MagicMock()
    mock_llm.model = "gpt-4o-mini"
    mock_llm.chat_completion = AsyncMock(side_effect=fake_completion)

    agent = IvyAgent(
        llm_client=mock_llm,
        tool_registry=reg,
        memory_store=store,
        max_iterations=2,
    )
    reply = await agent.run_turn("Do pings", session_id="cli:budget")
    assert reply.content == "Completed pings and summarized."
    assert len(calls) == 3
    assert calls[2]["tools"] is None
    assert reply.metadata["tool_calls_count"] == 2
    await store.close()


@pytest.mark.asyncio
async def test_agent_executes_concurrent_safe_tools_in_parallel(tmp_path):
    import asyncio

    store = MemoryStore(db_path=str(tmp_path / "parallel.db"))
    await store.initialize()

    active = 0
    peak = 0
    order = []

    class SlowReadTool(BaseTool):
        name = "slow_read"
        concurrent_safe = True
        description = "Read concurrently"
        parameters = {"type": "object", "properties": {"label": {"type": "string"}}}

        async def execute(self, label: str = "x", **kwargs):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.04)
            order.append(label)
            active -= 1
            return ToolResult(tool_name=self.name, status="success", output=f"data:{label}")

    reg = ToolRegistry()
    reg.register(SlowReadTool())

    mock_llm = MagicMock()
    mock_llm.model = "gpt-4o-mini"
    mock_llm.chat_completion = AsyncMock(
        side_effect=[
            ChatMessage(
                role=Role.ASSISTANT,
                content="",
                tool_calls=[
                    ToolCall(id="c1", function_name="slow_read", arguments={"label": "A"}),
                    ToolCall(id="c2", function_name="slow_read", arguments={"label": "B"}),
                ],
            ),
            ChatMessage(role=Role.ASSISTANT, content="Done reading."),
        ]
    )

    events = []
    agent = IvyAgent(llm_client=mock_llm, tool_registry=reg, memory_store=store)
    reply = await agent.run_turn("Read both", tool_callback=events.append)
    assert reply.content == "Done reading."
    assert peak == 2
    end_outputs = [ev["output"] for ev in events if ev["phase"] == "end"]
    assert end_outputs == ["data:A", "data:B"]
    await store.close()


def test_truncate_head_tail_and_format_tool_result():
    from ivy.core.prompt import format_tool_result, truncate_head_tail

    long_output = "HEAD_START_" + ("x" * 24_000) + "_TAIL_END"
    truncated = truncate_head_tail(long_output, max_chars=16_000)
    assert truncated.startswith("HEAD_START_")
    assert truncated.endswith("_TAIL_END")
    assert len(truncated) < len(long_output)

    formatted = format_tool_result(
        ToolResult(tool_name="terminal", status="success", output=long_output)
    )
    assert "HEAD_START_" in formatted
    assert "_TAIL_END" in formatted


def test_redact_secrets_scrubs_keys_and_tokens():
    from ivy.core.prompt import format_tool_result, redact_secrets

    sample = (
        "key1=sk-proj-1234567890abcdefghij123456\n"
        "OPENAI_API_KEY=sk-live-secret-9876543210abcdef\n"
        "Authorization: Bearer tok_super_secret_bearer_12345\n"
    )
    scrubbed = redact_secrets(sample)
    assert "sk-proj-1234567890abcdefghij123456" not in scrubbed
    assert "sk-live-secret-9876543210abcdef" not in scrubbed
    assert "tok_super_secret_bearer_12345" not in scrubbed

    formatted = format_tool_result(
        ToolResult(tool_name="terminal", status="success", output=sample)
    )
    assert "sk-proj-1234567890abcdefghij123456" not in formatted
    assert "tok_super_secret_bearer_12345" not in formatted


@pytest.mark.asyncio
async def test_agent_auto_compacts_when_token_budget_exceeded(tmp_path):
    from ivy.config import IvyConfig

    store = MemoryStore(db_path=str(tmp_path / "compact.db"))
    await store.initialize()
    for i in range(12):
        await store.save_message("s1", ChatMessage(role=Role.USER, content=f"msg {i} " + ("a" * 6000)))
        await store.save_message("s1", ChatMessage(role=Role.ASSISTANT, content=f"ans {i} " + ("b" * 6000)))

    mock_llm = MagicMock()
    mock_llm.model = "gpt-4o-mini"
    mock_llm.chat_completion = AsyncMock(
        side_effect=[
            ChatMessage(
                role=Role.ASSISTANT,
                content="Summary of earlier work.",
                metadata={"usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}},
            ),
            ChatMessage(
                role=Role.ASSISTANT,
                content="Final answer after compact.",
                metadata={"usage": {"prompt_tokens": 50, "completion_tokens": 10, "total_tokens": 60}},
            ),
        ]
    )

    cfg = IvyConfig(db_path=str(tmp_path / "compact.db"), auto_compact_threshold=0.1)
    agent = IvyAgent(llm_client=mock_llm, tool_registry=ToolRegistry(), memory_store=store, config=cfg)
    reply = await agent.run_turn("What next?", session_id="s1")

    assert reply.content == "Final answer after compact."
    assert reply.metadata.get("auto_compacted", {}).get("removed_messages", 0) > 0
    assert reply.metadata["turn_usage"]["api_calls"] == 2
    assert reply.metadata["turn_usage"]["total_tokens"] == 180
    sent_messages = mock_llm.chat_completion.call_args_list[-1].kwargs["messages"]
    assert sent_messages[0].role == Role.SYSTEM
    await store.close()


@pytest.mark.asyncio
async def test_agent_auto_compaction_fallback_when_llm_fails():
    from ivy.config import IvyConfig

    mock_llm = MagicMock()
    mock_llm.model = "gpt-4o-mini"
    mock_llm.chat_completion = AsyncMock(
        side_effect=[
            RuntimeError("compaction llm unavailable"),
            ChatMessage(role=Role.ASSISTANT, content="Still answered."),
        ]
    )
    cfg = IvyConfig(auto_compact_threshold=0.1)
    agent = IvyAgent(llm_client=mock_llm, tool_registry=ToolRegistry(), config=cfg)
    for i in range(12):
        agent._in_memory_sessions["s_err"].append(
            ChatMessage(role=Role.USER, content=f"q{i} " + ("z" * 6000))
        )
    reply = await agent.run_turn("Continue", session_id="s_err")
    assert reply.content == "Still answered."
    sent_messages = mock_llm.chat_completion.call_args_list[-1].kwargs["messages"]
    assert sent_messages[0].role == Role.SYSTEM
