from unittest.mock import AsyncMock, MagicMock

import pytest
from aether_agent.core.agent import AetherAgent
from aether_agent.core.subagent import SubagentRunner
from aether_agent.core.types import ChatMessage, Role, ToolCall, ToolResult
from aether_agent.memory.store import MemoryStore
from aether_agent.skills.engine import SkillEngine
from aether_agent.skills.format import SkillDefinition
from aether_agent.tools.base import BaseTool
from aether_agent.tools.registry import ToolRegistry


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

    agent = AetherAgent(llm_client=mock_llm, tool_registry=mock_tools)
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

    agent = AetherAgent(llm_client=mock_llm, tool_registry=mock_tools)
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

    agent = AetherAgent(
        llm_client=mock_llm,
        tool_registry=ToolRegistry(),
        memory_store=store,
        skill_engine=skills,
    )
    resp = await agent.run_turn(prompt="Halo Aether", session_id="sess-int")
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
