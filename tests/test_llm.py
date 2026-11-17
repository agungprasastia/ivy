from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aether_agent.config import AetherConfig
from aether_agent.core.llm import LLMClient
from aether_agent.core.types import (
    ChatMessage,
    Role,
    SessionState,
    ToolCall,
    ToolResult,
)


def test_chat_message_serialization():
    msg = ChatMessage(role=Role.USER, content="Hello")
    assert msg.model_dump()["role"] == "user"
    assert msg.content == "Hello"
    assert msg.to_llm_dict() == {"role": "user", "content": "Hello"}


def test_tool_call_and_tool_result():
    tc = ToolCall(id="call_1", function_name="terminal", arguments='{"command": "dir"}')
    assert tc.id == "call_1"
    assert tc.function_name == "terminal"
    assert tc.arguments == {"command": "dir"}

    tr_success = ToolResult(tool_call_id="call_1", status="success", output="file1.txt")
    assert tr_success.status == "success"
    assert tr_success.output == "file1.txt"

    tr_rejected = ToolResult(tool_call_id="call_2", status="rejected", output="User denied")
    assert tr_rejected.status == "rejected"


def test_session_state():
    state = SessionState(session_id="sess_123")
    assert state.session_id == "sess_123"
    assert state.messages == []
    state.messages.append(ChatMessage(role=Role.USER, content="Hi"))
    assert len(state.messages) == 1


def test_llm_client_provider_routing():
    cfg = AetherConfig(
        openai_api_key="sk-openai",
        openrouter_api_key="sk-or",
        gemini_api_key="sk-gem",
        anthropic_api_key="sk-ant",
    )

    c_openai = LLMClient(cfg, model="gpt-4o-mini")
    assert c_openai.provider == "openai"
    assert c_openai.resolved_model == "gpt-4o-mini"

    c_or = LLMClient(cfg, model="openrouter/meta-llama/llama-3.3-70b-instruct")
    assert c_or.provider == "openrouter"
    assert c_or.resolved_model == "meta-llama/llama-3.3-70b-instruct"
    assert "openrouter.ai" in (c_or.base_url or "")

    c_gem = LLMClient(cfg, model="gemini/gemini-2.5-flash")
    assert c_gem.provider == "gemini"
    assert c_gem.resolved_model == "gemini-2.5-flash"
    assert "generativelanguage.googleapis.com" in (c_gem.base_url or "")

    c_ollama = LLMClient(cfg, model="ollama/qwen2.5")
    assert c_ollama.provider == "ollama"
    assert c_ollama.resolved_model == "qwen2.5"
    assert "11434" in (c_ollama.base_url or "")

    c_ant = LLMClient(cfg, model="anthropic/claude-sonnet-4-5")
    assert c_ant.provider == "anthropic"
    assert c_ant.resolved_model == "claude-sonnet-4-5"


@pytest.mark.asyncio
async def test_llm_client_openai_chat_completion():
    cfg = AetherConfig(default_model="gpt-4o-mini", openai_api_key="sk-test")
    client = LLMClient(cfg)

    mock_resp = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="Hello! How can I help you today?",
                    tool_calls=None,
                )
            )
        ]
    )

    mock_create = AsyncMock(return_value=mock_resp)
    with patch.object(client, "_get_openai_client") as get_client:
        get_client.return_value = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=mock_create))
        )
        response = await client.chat_completion(
            messages=[ChatMessage(role=Role.USER, content="Hello")]
        )

        assert isinstance(response, ChatMessage)
        assert response.role == Role.ASSISTANT
        assert response.content == "Hello! How can I help you today?"
        assert response.tool_calls is None


@pytest.mark.asyncio
async def test_llm_client_openai_tool_calls():
    cfg = AetherConfig(default_model="gpt-4o-mini", openai_api_key="sk-test")
    client = LLMClient(cfg)

    mock_resp = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content="",
                    tool_calls=[
                        SimpleNamespace(
                            id="call_abc123",
                            function=SimpleNamespace(
                                name="terminal",
                                arguments='{"command": "echo test"}',
                            ),
                        )
                    ],
                )
            )
        ]
    )

    mock_create = AsyncMock(return_value=mock_resp)
    with patch.object(client, "_get_openai_client") as get_client:
        get_client.return_value = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=mock_create))
        )
        response = await client.chat_completion(
            messages=[ChatMessage(role=Role.USER, content="Run echo test")],
            tools=[{"type": "function", "function": {"name": "terminal"}}],
        )

        assert response.tool_calls is not None
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].id == "call_abc123"
        assert response.tool_calls[0].function_name == "terminal"
        assert response.tool_calls[0].arguments == {"command": "echo test"}


@pytest.mark.asyncio
async def test_llm_client_anthropic_chat_and_tools():
    cfg = AetherConfig(default_model="claude-sonnet-4-5", anthropic_api_key="sk-ant")
    client = LLMClient(cfg)

    mock_resp = SimpleNamespace(
        content=[
            SimpleNamespace(type="text", text="Checking directory..."),
            SimpleNamespace(
                type="tool_use",
                id="toolu_01",
                name="list_directory",
                input={"path": "."},
            ),
        ]
    )
    mock_create = AsyncMock(return_value=mock_resp)
    with patch.object(client, "_get_anthropic_client") as get_client:
        get_client.return_value = SimpleNamespace(
            messages=SimpleNamespace(create=mock_create)
        )
        response = await client.chat_completion(
            messages=[
                ChatMessage(role=Role.SYSTEM, content="You are helpful."),
                ChatMessage(role=Role.USER, content="List files"),
            ],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "list_directory",
                        "description": "List dir",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        )

        assert response.role == Role.ASSISTANT
        assert "Checking directory" in (response.content or "")
        assert response.tool_calls is not None
        assert response.tool_calls[0].id == "toolu_01"
        assert response.tool_calls[0].function_name == "list_directory"
        assert response.tool_calls[0].arguments == {"path": "."}


@pytest.mark.asyncio
async def test_llm_client_streaming():
    cfg = AetherConfig(default_model="gpt-4o-mini", openai_api_key="sk-test")
    client = LLMClient(cfg)

    async def fake_stream():
        for text in ["Hello", " ", "Aether"]:
            yield SimpleNamespace(
                choices=[SimpleNamespace(delta=SimpleNamespace(content=text))]
            )

    mock_create = AsyncMock(return_value=fake_stream())
    with patch.object(client, "_get_openai_client") as get_client:
        get_client.return_value = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=mock_create))
        )
        stream = await client.chat_completion(
            messages=[ChatMessage(role=Role.USER, content="Hi")],
            stream=True,
        )
        chunks = [chunk async for chunk in stream]
        assert "".join(chunks) == "Hello Aether"
