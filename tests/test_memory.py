import pytest
from ivy.core.types import ChatMessage, Role
from ivy.memory.context import ContextWindowManager
from ivy.memory.fts import sanitize_fts_query
from ivy.memory.store import MemoryStore


@pytest.mark.asyncio
async def test_memory_store_fts_search(tmp_path):
    db_file = str(tmp_path / "test.db")
    store = MemoryStore(db_path=db_file)
    await store.initialize()

    session_id = "test-session-1"
    await store.save_message(
        session_id,
        ChatMessage(role=Role.USER, content="Membahas instalasi Python uv"),
    )
    await store.save_message(
        session_id,
        ChatMessage(role=Role.ASSISTANT, content="uv adalah package manager cepat"),
    )

    results = await store.search_fts("instalasi")
    assert len(results) > 0
    assert "Membahas instalasi" in results[0]["content"]

    # Alias search_history_fts works identically
    alias_results = await store.search_history_fts("package")
    assert len(alias_results) > 0

    await store.close()


@pytest.mark.asyncio
async def test_fts_sanitizes_malformed_queries_and_excludes_subagents(tmp_path):
    db_file = str(tmp_path / "fts_edge.db")
    store = MemoryStore(db_path=db_file)
    await store.initialize()

    await store.save_message(
        "main-sess",
        ChatMessage(role=Role.USER, content='Testing "quoted" AND OR syntax*'),
    )
    await store.save_message(
        "subagent-child",
        ChatMessage(role=Role.ASSISTANT, content="secret internal subagent thought"),
        exclude_from_fts=True,
    )

    # Should never raise sqlite3.OperationalError for unbalanced quotes/operators
    res_weird = await store.search_fts('"unbalanced AND OR NOT * ( ) :')
    assert isinstance(res_weird, list)

    res_sub = await store.search_fts("subagent")
    assert len(res_sub) == 0

    assert sanitize_fts_query('hello "world AND') != ""
    await store.close()


@pytest.mark.asyncio
async def test_user_profile_and_context_compression(tmp_path):
    db_file = str(tmp_path / "profile.db")
    store = MemoryStore(db_path=db_file)
    await store.initialize()

    await store.set_user_preference("preferred_language", "Indonesian")
    await store.set_user_preference("editor", "vscode")
    profile = await store.get_user_profile()
    assert profile["preferred_language"] == "Indonesian"
    assert profile["editor"] == "vscode"

    msgs = [ChatMessage(role=Role.SYSTEM, content="System prompt")]
    for i in range(12):
        msgs.append(ChatMessage(role=Role.USER, content=f"User msg {i}"))
    compressed = ContextWindowManager(max_messages=6, keep_recent=4).compress(msgs)
    assert len(compressed) <= 6
    assert compressed[0].role == Role.SYSTEM
    assert "User msg 11" in (compressed[-1].content or "")

    await store.close()


@pytest.mark.asyncio
async def test_memory_store_list_sessions(tmp_path):
    store = MemoryStore(db_path=str(tmp_path / "sess.db"))
    await store.initialize()

    await store.save_message("cli:main", ChatMessage(role=Role.USER, content="hi"))
    await store.save_message("cli:work", ChatMessage(role=Role.USER, content="code"))
    await store.save_message("cli:work", ChatMessage(role=Role.ASSISTANT, content="ok"))

    sessions = await store.list_sessions()
    by_id = {s["session_id"]: s for s in sessions}
    assert "cli:main" in by_id
    assert "cli:work" in by_id
    assert by_id["cli:work"]["message_count"] == 2
    await store.close()


@pytest.mark.asyncio
async def test_memory_store_title_undo_fork_and_context_compaction(tmp_path):
    from unittest.mock import AsyncMock, MagicMock
    from ivy.memory.context import ContextWindowManager, compute_context_breakdown

    store = MemoryStore(db_path=str(tmp_path / "ux_mem.db"))
    await store.initialize()

    await store.save_message("cli:main", ChatMessage(role=Role.USER, content="first q"))
    await store.save_message("cli:main", ChatMessage(role=Role.ASSISTANT, content="first a"))
    await store.save_message("cli:main", ChatMessage(role=Role.USER, content="second q"))
    await store.save_message("cli:main", ChatMessage(role=Role.ASSISTANT, content="second a"))
    await store.set_session_title("cli:main", "Bugfix Auth")

    assert await store.get_session_title("cli:main") == "Bugfix Auth"
    assert await store.get_latest_session_id() == "cli:main"

    sessions = await store.list_sessions()
    assert sessions[0]["title"] == "Bugfix Auth"

    # Fork before undo
    copied = await store.fork_session("cli:main", "cli:branch1")
    assert copied == 4
    assert len(await store.get_messages("cli:branch1")) == 4

    # Undo removes last turn (second q + second a)
    undone_prompt, deleted = await store.delete_last_turn("cli:main")
    assert undone_prompt == "second q"
    assert deleted == 2
    remaining = await store.get_messages("cli:main")
    assert len(remaining) == 2

    # Context breakdown & LLM compaction
    breakdown = compute_context_breakdown(remaining, model="gpt-4o-mini")
    assert breakdown["used_tokens"] > 0
    assert breakdown["max_tokens"] == 128_000
    assert 0.0 <= breakdown["percent"] <= 100.0

    mock_llm = MagicMock()
    mock_llm.chat_completion = AsyncMock(
        return_value=ChatMessage(role=Role.ASSISTANT, content="- Summary of earlier turns")
    )
    long_msgs = [ChatMessage(role=Role.USER, content=f"Turn {i} details " * 8) for i in range(10)]
    compacted, stats = await ContextWindowManager().compact_with_llm(
        long_msgs, llm_client=mock_llm, keep_recent=4
    )
    assert len(compacted) == 5
    assert stats["removed_messages"] == 6
    assert "Summary of earlier turns" in (compacted[0].content or "")

    # Fallback first-prompt title, ensure_session & delete_session
    await store.save_message("cli:unlabeled", ChatMessage(role=Role.USER, content="tolong fix bug api"))
    await store.ensure_session("cli:empty_new")
    listed = {s["session_id"]: s for s in await store.list_sessions()}
    assert listed["cli:unlabeled"]["title"] == "tolong fix bug api"
    assert "cli:empty_new" in listed
    assert await store.delete_session("cli:empty_new") is True
    assert "cli:empty_new" not in {s["session_id"] for s in await store.list_sessions()}

    await store.close()


def test_should_auto_compact_threshold_and_tail_safety():
    from ivy.memory.context import should_auto_compact

    short_msgs = [ChatMessage(role=Role.USER, content="hello")] * 4
    assert should_auto_compact(short_msgs, model="gpt-4o", threshold=0.85) is False

    # 128_000 window for gpt-4o -> ~115_000 tokens (~90%)
    heavy = [ChatMessage(role=Role.USER, content="x" * 46_000) for _ in range(10)]
    assert should_auto_compact(heavy, model="gpt-4o", threshold=0.85) is True


@pytest.mark.asyncio
async def test_compact_with_llm_includes_tool_calls_and_usage():
    from unittest.mock import AsyncMock, MagicMock
    from ivy.core.types import ToolCall

    mock_llm = MagicMock()
    mock_llm.chat_completion = AsyncMock(
        return_value=ChatMessage(
            role=Role.ASSISTANT,
            content="- Wrote src/app.py",
            metadata={"usage": {"prompt_tokens": 90, "completion_tokens": 15, "total_tokens": 105}},
        )
    )
    msgs = [
        ChatMessage(role=Role.USER, content="Edit src/app.py"),
        ChatMessage(
            role=Role.ASSISTANT,
            content=None,
            tool_calls=[ToolCall(id="c1", function_name="write_file", arguments={"path": "src/app.py"})],
        ),
        ChatMessage(role=Role.TOOL, name="write_file", tool_call_id="c1", content="written"),
        ChatMessage(role=Role.USER, content="Now test it"),
        ChatMessage(role=Role.ASSISTANT, content="Done"),
    ]
    _, stats = await ContextWindowManager().compact_with_llm(msgs, llm_client=mock_llm, keep_recent=2)
    prompt_sent = mock_llm.chat_completion.call_args.kwargs["messages"][1].content
    assert "write_file" in prompt_sent
    assert "src/app.py" in prompt_sent
    assert stats.get("compaction_usage", {}).get("total_tokens") == 105
