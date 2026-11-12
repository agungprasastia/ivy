from unittest.mock import AsyncMock, patch

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


def test_tool_call_and_tool_result():
    tc = ToolCall(id="call_1", function_name="terminal", arguments={"command": "dir"})
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


@pytest.mark.asyncio
async def test_llm_client_initialization():
    cfg = AetherConfig(default_model="gpt-4o-mini")
    client = LLMClient(cfg)
    assert client.model == "gpt-4o-mini"


@pytest.mark.asyncio
async def test_llm_client_chat_completion():
    cfg = AetherConfig(default_model="gpt-4o-mini")
    client = LLMClient(cfg)

    # Mock litellm.acompletion
    mock_choice = {
        "message": {
            "role": "assistant",
            "content": "Hello! How can I help you today?",
            "tool_calls": None,
        }
    }
    mock_response = {
        "choices": [mock_choice]
    }

    with patch("litellm.acompletion", new=AsyncMock(return_value=mock_response)):
        messages = [ChatMessage(role=Role.USER, content="Hello")]
        response = await client.chat_completion(messages=messages)

        assert isinstance(response, ChatMessage)
        assert response.role == Role.ASSISTANT
        assert response.content == "Hello! How can I help you today?"
        assert response.tool_calls is None


@pytest.mark.asyncio
async def test_llm_client_chat_completion_with_tool_calls():
    cfg = AetherConfig(default_model="gpt-4o-mini")
    client = LLMClient(cfg)

    mock_choice = {
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_abc123",
                    "type": "function",
                    "function": {
                        "name": "terminal",
                        "arguments": '{"command": "echo test"}',
                    },
                }
            ],
        }
    }
    mock_response = {
        "choices": [mock_choice]
    }

    with patch("litellm.acompletion", new=AsyncMock(return_value=mock_response)):
        messages = [ChatMessage(role=Role.USER, content="Run echo test")]
        response = await client.chat_completion(messages=messages)

        assert isinstance(response, ChatMessage)
        assert response.tool_calls is not None
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].id == "call_abc123"
        assert response.tool_calls[0].function_name == "terminal"
        assert response.tool_calls[0].arguments == {"command": "echo test"}
