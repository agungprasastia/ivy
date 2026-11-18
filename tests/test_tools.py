import asyncio
from pathlib import Path

import pytest
from ivy.tools.approval import SafetyApprovalPolicy
from ivy.tools.files import (
    FileTools,
    ListDirectoryTool,
    PatchFileTool,
    ReadFileTool,
    SearchFilesTool,
    WriteFileTool,
)
from ivy.tools.registry import ToolRegistry
from ivy.tools.runner import CodeRunnerTool
from ivy.tools.terminal import TerminalTool
from ivy.tools.web import WebFetchTool, WebSearchTool


@pytest.mark.asyncio
async def test_terminal_tool_safe_execution():
    policy = SafetyApprovalPolicy(auto_approve=True)
    term = TerminalTool(approval_policy=policy)
    res = await term.execute(command='echo "Hello Ivy"')
    assert res.status == "success"
    assert "Hello Ivy" in res.output


@pytest.mark.asyncio
async def test_safety_guard_blocks_destructive_command():
    policy = SafetyApprovalPolicy(auto_approve=False)
    term = TerminalTool(approval_policy=policy)
    res = await term.execute(command="rm -rf /")
    assert res.status == "rejected"
    assert "rejected" in (res.error or "").lower() or "rejected" in res.output.lower()


@pytest.mark.asyncio
async def test_safety_guard_blocks_windows_del_command():
    policy = SafetyApprovalPolicy(auto_approve=False)
    assert policy.is_dangerous("del /f /s /q C:\\Windows")
    assert policy.is_dangerous("rmdir /s /q C:\\test")


@pytest.mark.asyncio
async def test_file_tools(tmp_path: Path):
    read_tool = ReadFileTool()
    write_tool = WriteFileTool()
    list_tool = ListDirectoryTool()
    patch_tool = PatchFileTool()

    target_file = tmp_path / "hello.txt"

    # Write file
    write_res = await write_tool.execute(path=str(target_file), content="Hello World\nLine 2")
    assert write_res.status == "success"
    assert target_file.exists()

    # Read file
    read_res = await read_tool.execute(path=str(target_file))
    assert read_res.status == "success"
    assert "Hello World" in read_res.output

    # Patch file
    patch_res = await patch_tool.execute(
        path=str(target_file),
        target="Hello World",
        replacement="Hello Ivy",
    )
    assert patch_res.status == "success"

    # Verify patch
    read_patched = await read_tool.execute(path=str(target_file))
    assert "Hello Ivy" in read_patched.output

    # List directory
    list_res = await list_tool.execute(path=str(tmp_path))
    assert list_res.status == "success"
    assert "hello.txt" in list_res.output


@pytest.mark.asyncio
async def test_code_runner_execution():
    runner = CodeRunnerTool()
    code = "x = 21 * 2\nprint(f'Answer: {x}')"
    res = await runner.execute(code=code)
    assert res.status == "success"
    assert "Answer: 42" in res.output


@pytest.mark.asyncio
async def test_code_runner_timeout():
    # Use short timeout for test
    runner = CodeRunnerTool(timeout=1.0)
    code = "import time\nwhile True:\n    time.sleep(0.1)"
    res = await runner.execute(code=code)
    assert res.status == "error"
    assert "timed out" in (res.error or "").lower()


@pytest.mark.asyncio
async def test_tool_registry():
    registry = ToolRegistry()
    term = TerminalTool(approval_policy=SafetyApprovalPolicy(auto_approve=True))
    registry.register(term)

    schemas = registry.get_schemas()
    assert len(schemas) == 1
    assert schemas[0]["function"]["name"] == "terminal"

    exec_res = await registry.execute(
        tool_name="terminal",
        arguments={"command": "echo RegistryWorks"},
        tool_call_id="call_reg_1",
    )
    assert exec_res.status == "success"
    assert "RegistryWorks" in exec_res.output
    assert exec_res.tool_call_id == "call_reg_1"


@pytest.mark.asyncio
async def test_file_tools_helper_registration(tmp_path: Path):
    registry = ToolRegistry()
    file_tools = FileTools()
    file_tools.register_all(registry)

    schema_names = [s["function"]["name"] for s in registry.get_schemas()]
    assert "read_file" in schema_names
    assert "write_file" in schema_names
    assert "list_directory" in schema_names
    assert "patch_file" in schema_names


@pytest.mark.asyncio
async def test_search_files_tool(tmp_path: Path):
    (tmp_path / "alpha.py").write_text("def hello():\n    return 'ivy'\n", encoding="utf-8")
    sub = tmp_path / "nested"
    sub.mkdir()
    (sub / "beta.txt").write_text("nothing here\n", encoding="utf-8")

    search = SearchFilesTool()
    res = await search.execute(pattern="ivy", path=str(tmp_path))
    assert res.status == "success"
    assert "alpha.py" in res.output
    assert "beta.txt" not in res.output


@pytest.mark.asyncio
async def test_safety_policy_unattended_and_async_callback():
    # Unattended fail-safe always rejects dangerous commands unless explicitly permitted
    cron_policy = SafetyApprovalPolicy(unattended=True)
    assert await cron_policy.check_approval("rm -rf /tmp/foo") is False

    # Async callback approval (e.g., Telegram/Discord inline button)
    async def approve_cb(cmd: str) -> bool:
        return "safe-target" in cmd

    async_policy = SafetyApprovalPolicy(approval_callback=approve_cb)
    assert await async_policy.check_approval("rm -rf /tmp/safe-target") is True
    assert await async_policy.check_approval("rm -rf /") is False


@pytest.mark.asyncio
async def test_web_tools_extract_and_search(monkeypatch):
    class FakeResponse:
        status_code = 200
        text = "<html><head><title>Ivy</title></head><body><h1>Hello</h1><p>World</p><script>bad()</script></body></html>"

        def raise_for_status(self):
            return None

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def get(self, url, **kwargs):
            return FakeResponse()

    import ivy.tools.web as web_mod

    monkeypatch.setattr(web_mod.httpx, "AsyncClient", FakeClient)

    fetch = WebFetchTool()
    res = await fetch.execute(url="https://example.com")
    assert res.status == "success"
    assert "Hello" in res.output
    assert "bad()" not in res.output

    search = WebSearchTool()
    assert search.get_schema()["function"]["name"] == "web_search"


def test_safety_guard_catches_powershell_aliases_and_rm_r():
    policy = SafetyApprovalPolicy(auto_approve=False)
    for cmd in [
        r"rm -Recurse C:\temp\target",
        r"del -Force C:\temp\target",
        r"ri -r -fo C:\temp\target",
        "rm -r /tmp/target",
        r"del /q /f C:\temp\*",
    ]:
        assert policy.is_dangerous(cmd) is True, f"Expected dangerous: {cmd}"


@pytest.mark.asyncio
async def test_code_runner_cleanup_on_cancel():
    import sys

    runner = CodeRunnerTool(timeout=10.0)
    started = []
    orig_exec = asyncio.create_subprocess_exec
    procs = []

    async def spy_exec(*args, **kwargs):
        proc = await orig_exec(*args, **kwargs)
        if args and args[0] == sys.executable:
            procs.append(proc)
            started.append(True)
        return proc

    code = "import time\nwhile True:\n    time.sleep(0.05)"
    old_exec = asyncio.create_subprocess_exec
    asyncio.create_subprocess_exec = spy_exec
    try:
        task = asyncio.create_task(runner.execute(code=code))
        while not started:
            await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        asyncio.create_subprocess_exec = old_exec

    assert len(procs) == 1
    assert procs[0].returncode is not None


@pytest.mark.asyncio
async def test_file_tools_emit_diff_metadata_and_approval_session_feedback(tmp_path):
    from ivy.tools.files import PatchFileTool, WriteFileTool

    target_file = tmp_path / "app.py"
    writer = WriteFileTool()
    r1 = await writer.execute(path=str(target_file), content="x = 1\ny = 2\n")
    assert r1.status == "success"
    assert r1.metadata["added"] == 2
    assert "+x = 1" in r1.metadata["diff"]

    patcher = PatchFileTool()
    r2 = await patcher.execute(path=str(target_file), target="y = 2", replacement="y = 42\nz = 99")
    assert r2.status == "success"
    assert r2.metadata["added"] == 2
    assert r2.metadata["removed"] == 1
    assert "-y = 2" in r2.metadata["diff"]
    assert "+y = 42" in r2.metadata["diff"]

    # Session allow & deny-with-feedback
    calls = []

    def cb(cmd: str):
        calls.append(cmd)
        if "once-session" in cmd:
            return "session"
        return (False, "Use pytest instead of rm")

    policy = SafetyApprovalPolicy(auto_approve=False, approval_callback=cb)
    ok1, fb1 = await policy.evaluate_approval("rm -rf once-session")
    assert ok1 is True and fb1 is None
    # Second identical command uses session allowlist without invoking callback again
    ok2, fb2 = await policy.evaluate_approval("rm -rf once-session")
    assert ok2 is True and fb2 is None
    assert len(calls) == 1

    term = TerminalTool(approval_policy=policy)
    res_rej = await term.execute(command="rm -rf /tmp/danger")
    assert res_rej.status == "rejected"
    assert "Use pytest instead of rm" in (res_rej.error or "")


@pytest.mark.asyncio
async def test_phase2_tools_todo_clarify_delegate_and_profile(tmp_path):
    from unittest.mock import AsyncMock, MagicMock
    from ivy.core.subagent import SubagentRunner
    from ivy.core.types import ChatMessage, Role
    from ivy.memory.store import MemoryStore
    from ivy.tools.clarify import ClarifyTool, mark_recommended, strip_recommended
    from ivy.tools.delegate import DelegateTaskTool
    from ivy.tools.profile import RememberPreferenceTool
    from ivy.tools.todo import TodoListTool, TodoStore

    # 1. TodoListTool write, merge, and read
    t_store = TodoStore()
    todo = TodoListTool(store=t_store, session_id_getter=lambda: "cli:t1")
    r_write = await todo.execute(
        todos=[
            {"id": "1", "content": "Inspect code", "status": "in_progress"},
            {"id": "2", "content": "Write tests", "status": "pending"},
        ]
    )
    assert r_write.status == "success"
    assert "[>] #1: Inspect code" in r_write.output

    r_merge = await todo.execute(
        todos=[{"id": "1", "content": "", "status": "completed"}],
        merge=True,
    )
    assert "[x] #1: Inspect code" in r_merge.output
    r_read = await todo.execute()
    assert len(r_read.metadata["todos"]) == 2

    # 2. ClarifyTool recommended markers & scoped callback
    marked = mark_recommended(["SQLite", "Postgres"])
    assert marked[0] == "SQLite (Recommended)"
    assert strip_recommended("SQLite (Recommended)") == "SQLite"

    clarify = ClarifyTool()
    err_res = await clarify.execute(questions=[{"question": "Pick DB?"}])
    assert err_res.status == "error"

    async def fake_clarify(question: str, choices: list[str], multi: bool):
        assert choices[0].endswith("(Recommended)")
        return choices[0]

    async with clarify.scoped_context(fake_clarify):
        ok_res = await clarify.execute(
            questions=[{"question": "Which DB?", "choices": ["SQLite", "Postgres"]}]
        )
    assert ok_res.status == "success"
    assert '"answer": "SQLite"' in ok_res.output

    # 3. DelegateTaskTool single & parallel batch + recursion block
    store = MemoryStore(db_path=str(tmp_path / "tools_p2.db"))
    await store.initialize()

    mock_llm = MagicMock()
    mock_llm.chat_completion = AsyncMock(
        return_value=ChatMessage(role=Role.ASSISTANT, content="Subtask done")
    )
    reg = ToolRegistry()
    reg.register(clarify)
    runner = SubagentRunner(llm_client=mock_llm, tool_registry=reg, memory_store=store, max_iterations=2)
    delegate = DelegateTaskTool(runner=runner, session_id_getter=lambda: "cli:parent")
    reg.register(delegate)

    assert delegate._safe_child_tools(None) == []  # clarify & delegate_task blocked from child
    single = await delegate.execute(task="Check module A")
    assert single.status == "success"
    assert "Subtask done" in single.output

    batch = await delegate.execute(tasks=["Task 1", "Task 2"])
    assert batch.status == "success"
    assert batch.metadata["task_count"] == 2
    assert "Subtask 1:" in batch.output and "Subtask 2:" in batch.output

    # 4. RememberPreferenceTool
    pref = RememberPreferenceTool(memory_store=store)
    p_res = await pref.execute(key="lang", value="id")
    assert p_res.status == "success"
    assert (await store.get_user_profile())["lang"] == "id"
    await store.close()
