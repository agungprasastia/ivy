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
