from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from ivy.config import IvyConfig
from ivy.core.llm import LLMClient
from ivy.core.types import (
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
    cfg = IvyConfig(
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
    cfg = IvyConfig(default_model="gpt-4o-mini", openai_api_key="sk-test")
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
    cfg = IvyConfig(default_model="gpt-4o-mini", openai_api_key="sk-test")
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
    cfg = IvyConfig(default_model="claude-sonnet-4-5", anthropic_api_key="sk-ant")
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
    cfg = IvyConfig(default_model="gpt-4o-mini", openai_api_key="sk-test")
    client = LLMClient(cfg)

    async def fake_stream():
        for text in ["Hello", " ", "Ivy"]:
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
        assert "".join(chunks) == "Hello Ivy"


def test_anthropic_coalesces_parallel_tool_results_and_compressed_history():
    cfg = IvyConfig(default_model="claude-sonnet-4-5", anthropic_api_key="sk-ant")
    client = LLMClient(cfg)

    messages = [
        ChatMessage(role=Role.SYSTEM, content="System prompt"),
        ChatMessage(role=Role.SYSTEM, content="[Compressed 4 earlier messages]"),
        ChatMessage(
            role=Role.ASSISTANT,
            content="Running two tools",
            tool_calls=[
                ToolCall(id="tc_1", function_name="read_file", arguments={"path": "a.txt"}),
                ToolCall(id="tc_2", function_name="read_file", arguments={"path": "b.txt"}),
            ],
        ),
        ChatMessage(role=Role.TOOL, tool_call_id="tc_1", name="read_file", content="A"),
        ChatMessage(role=Role.TOOL, tool_call_id="tc_2", name="read_file", content="B"),
    ]
    system_prompt, formatted, _ = client._format_anthropic_payload(messages)
    assert system_prompt is not None
    assert formatted[0]["role"] == "user"
    # Assistant message followed by ONE user message containing both tool_result blocks
    tool_user_msgs = [
        m for m in formatted
        if m["role"] == "user" and isinstance(m["content"], list)
    ]
    assert len(tool_user_msgs) == 1
    assert len(tool_user_msgs[0]["content"]) == 2
    assert [b["tool_use_id"] for b in tool_user_msgs[0]["content"]] == ["tc_1", "tc_2"]


@pytest.mark.asyncio
async def test_llm_client_reuses_and_closes_clients():
    cfg = IvyConfig(default_model="gpt-4o-mini", openai_api_key="sk-test")
    client = LLMClient(cfg)

    c1 = client._get_openai_client()
    c2 = client._get_openai_client()
    assert c1 is c2

    with patch.object(c1, "close", new=AsyncMock()) as mock_close:
        await client.close()
        assert mock_close.called


@pytest.mark.asyncio
async def test_llm_client_stream_turn_assembles_text_and_tool_calls():
    cfg = IvyConfig(default_model="gpt-4o-mini", openai_api_key="sk-test")
    client = LLMClient(cfg)

    async def fake_stream():
        yield SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content="Hello ", tool_calls=None))]
        )
        yield SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content="Ivy!", tool_calls=None))]
        )

    mock_create = AsyncMock(return_value=fake_stream())
    tokens = []
    with patch.object(client, "_get_openai_client") as get_client:
        get_client.return_value = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=mock_create))
        )
        msg = await client.stream_turn(
            messages=[ChatMessage(role=Role.USER, content="Hi")],
            on_token=tokens.append,
        )
        assert msg.content == "Hello Ivy!"
        assert tokens == ["Hello ", "Ivy!"]


@pytest.mark.asyncio
async def test_llm_client_captures_usage_and_handles_stream_options_fallback():
    cfg = IvyConfig(default_model="gpt-4o-mini", openai_api_key="sk-test")
    client = LLMClient(cfg)

    calls = []

    async def fake_stream():
        yield SimpleNamespace(
            choices=[SimpleNamespace(delta=SimpleNamespace(content="Hi", tool_calls=None))],
            usage=None,
        )
        yield SimpleNamespace(
            choices=[],
            usage=SimpleNamespace(prompt_tokens=15, completion_tokens=5, total_tokens=20),
        )

    async def fake_create(**kwargs):
        calls.append(kwargs)
        if "stream_options" in kwargs:
            raise RuntimeError("Unknown parameter: stream_options")
        return fake_stream()

    with patch.object(client, "_get_openai_client") as get_client:
        get_client.return_value = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create))
        )
        msg = await client.stream_turn(messages=[ChatMessage(role=Role.USER, content="Hi")])

    assert len(calls) == 2
    assert "stream_options" not in calls[1]
    assert msg.metadata["usage"] == {
        "prompt_tokens": 15,
        "completion_tokens": 5,
        "total_tokens": 20,
    }


@pytest.mark.asyncio
async def test_llm_client_retries_transient_error_then_succeeds(monkeypatch):
    class RateLimitErr(Exception):
        status_code = 429

    monkeypatch.setattr("asyncio.sleep", AsyncMock())
    cfg = IvyConfig(
        openai_api_key="test-key",
        default_model="gpt-4o-mini",
        llm_max_retries=2,
        llm_retry_base_delay=0.1,
    )
    client = LLMClient(config=cfg)
    mock_openai = MagicMock()
    mock_openai.chat.completions.create = AsyncMock(
        side_effect=[
            RateLimitErr("rate limit"),
            SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="Recovered", tool_calls=None))],
                usage=None,
            ),
        ]
    )
    monkeypatch.setattr(client, "_get_openai_client", lambda: mock_openai)

    reply = await client.chat_completion([ChatMessage(role=Role.USER, content="hi")])
    assert reply.content == "Recovered"
    assert mock_openai.chat.completions.create.await_count == 2
    await client.close()


@pytest.mark.asyncio
async def test_llm_client_falls_back_after_retry_exhaustion_and_restores_model(monkeypatch):
    class ServerErr(Exception):
        status_code = 503

    monkeypatch.setattr("asyncio.sleep", AsyncMock())
    cfg = IvyConfig(
        openai_api_key="test-key",
        anthropic_api_key="test-ant",
        default_model="gpt-4o-mini",
        fallback_model="anthropic/claude-sonnet-4-5",
        llm_max_retries=1,
        llm_retry_base_delay=0.05,
    )
    client = LLMClient(config=cfg)
    mock_openai = MagicMock()
    mock_openai.chat.completions.create = AsyncMock(side_effect=ServerErr("down"))
    monkeypatch.setattr(client, "_get_openai_client", lambda: mock_openai)
    monkeypatch.setattr(
        client,
        "_anthropic_completion",
        AsyncMock(return_value=ChatMessage(role=Role.ASSISTANT, content="From fallback")),
    )

    reply = await client.chat_completion([ChatMessage(role=Role.USER, content="hi")])
    assert reply.content == "From fallback"
    assert reply.metadata.get("fallback_used") is True
    assert client.model == "gpt-4o-mini"
    assert client.provider == "openai"
    await client.close()


@pytest.mark.asyncio
async def test_llm_client_does_not_retry_auth_error_or_mid_stream_failure(monkeypatch):
    class AuthErr(Exception):
        status_code = 401

    sleep_mock = AsyncMock()
    monkeypatch.setattr("asyncio.sleep", sleep_mock)
    cfg = IvyConfig(
        openai_api_key="bad-key",
        anthropic_api_key="ant-key",
        default_model="gpt-4o-mini",
        fallback_model="anthropic/claude-sonnet-4-5",
        llm_max_retries=2,
    )
    client = LLMClient(config=cfg)
    mock_openai = MagicMock()
    mock_openai.chat.completions.create = AsyncMock(side_effect=AuthErr("401 unauthorized"))
    monkeypatch.setattr(client, "_get_openai_client", lambda: mock_openai)

    with pytest.raises(AuthErr):
        await client.chat_completion([ChatMessage(role=Role.USER, content="hi")])
    assert mock_openai.chat.completions.create.await_count == 1
    assert sleep_mock.await_count == 0

    class MidStreamErr(Exception):
        status_code = 502

    async def partial_stream(*args, **kwargs):
        yield SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="half ", tool_calls=None))], usage=None)
        raise MidStreamErr("dropped mid stream")

    mock_openai.chat.completions.create = AsyncMock(return_value=partial_stream())
    seen = []
    with pytest.raises(MidStreamErr):
        await client.stream_turn([ChatMessage(role=Role.USER, content="hi")], on_token=seen.append)
    assert seen == ["half "]
    assert mock_openai.chat.completions.create.await_count == 1
    await client.close()


@pytest.mark.asyncio
async def test_llm_client_retries_overloaded_529(monkeypatch):
    class OverloadedError(Exception):
        status_code = 529

    monkeypatch.setattr("asyncio.sleep", AsyncMock())
    cfg = IvyConfig(
        openai_api_key="test-key",
        default_model="gpt-4o-mini",
        llm_max_retries=1,
        llm_retry_base_delay=0.01,
    )
    client = LLMClient(config=cfg)
    mock_openai = MagicMock()
    mock_openai.chat.completions.create = AsyncMock(
        side_effect=[
            OverloadedError("overloaded"),
            SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="Ok 529", tool_calls=None))],
                usage=None,
            ),
        ]
    )
    monkeypatch.setattr(client, "_get_openai_client", lambda: mock_openai)

    reply = await client.chat_completion([ChatMessage(role=Role.USER, content="hi")])
    assert reply.content == "Ok 529"
    assert mock_openai.chat.completions.create.await_count == 2
    await client.close()


@pytest.mark.asyncio
async def test_llm_client_fallback_does_not_mutate_shared_model_under_concurrency(monkeypatch):
    import asyncio

    class ServerErr(Exception):
        status_code = 503

    monkeypatch.setattr("asyncio.sleep", AsyncMock())
    cfg = IvyConfig(
        openai_api_key="test-key",
        anthropic_api_key="test-ant",
        default_model="gpt-4o-mini",
        fallback_model="anthropic/claude-sonnet-4-5",
        llm_max_retries=0,
    )
    client = LLMClient(config=cfg)

    seen_openai_models = []
    fallback_entered = asyncio.Event()
    release_fallback = asyncio.Event()

    async def fake_openai_create(**params):
        seen_openai_models.append(params["model"])
        if params.get("messages") and params["messages"][-1]["content"] == "trigger-fallback":
            raise ServerErr("primary down for task A")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Primary OK", tool_calls=None))],
            usage=None,
        )

    async def fake_anthropic(**kwargs):
        fallback_entered.set()
        await release_fallback.wait()
        return ChatMessage(role=Role.ASSISTANT, content="Fallback OK")

    mock_openai = MagicMock()
    mock_openai.chat.completions.create = AsyncMock(side_effect=fake_openai_create)
    monkeypatch.setattr(client, "_get_openai_client", lambda: mock_openai)
    monkeypatch.setattr(client, "_anthropic_completion", fake_anthropic)

    task_a = asyncio.create_task(
        client.chat_completion([ChatMessage(role=Role.USER, content="trigger-fallback")])
    )
    await fallback_entered.wait()
    # While task_a is suspended in Anthropic fallback, shared client.model must remain primary
    assert client.model == "gpt-4o-mini"
    assert client.provider == "openai"
    reply_b = await client.chat_completion([ChatMessage(role=Role.USER, content="normal")])
    release_fallback.set()
    reply_a = await task_a

    assert reply_a.content == "Fallback OK"
    assert reply_b.content == "Primary OK"
    assert seen_openai_models == ["gpt-4o-mini", "gpt-4o-mini"]
    await client.close()
