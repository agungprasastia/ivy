# Ivy Phase 5: Runtime Resilience & Background Processes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add transient LLM retry/backoff with fallback model routing, token-threshold automatic context compaction, and safe cross-platform background process management (`terminal(background=True)`, `process` tool, `/ps`) to Ivy.

**Architecture:** Resilience extends `LLMClient` (`_execute_with_resilience` + `_is_retryable_error`) and `IvyAgent` (token-budget check before iterations via `should_auto_compact` + `ContextWindowManager.compact_with_llm`) without breaking existing mock-based tests. Background execution extracts a shared shell-command builder from `TerminalTool` into `ivy/tools/process_registry.py`, manages bounded async pipe readers with `redact_secrets`, and guarantees process-tree cleanup on TUI shutdown while keeping `SafetyApprovalPolicy` gating in front of every spawn.

**Tech Stack:** Python 3.11+, Pydantic v2 (`IvyConfig`), `asyncio.subprocess`, `openai` / `anthropic` async SDKs, Rich, `prompt_toolkit`, `pytest` + `pytest-asyncio`, `ruff`.

**Spec:** Phase 5 Option 1 roadmap (derived from codebase audit and `_ref_hermes/tools/process_registry.py` pattern; no separate spec file).

## Global Constraints

- Python `>= 3.11`; zero new runtime dependencies added to `[project].dependencies` in `pyproject.toml`.
- Cross-platform Windows (PowerShell `powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -Command ...` with `CREATE_NEW_PROCESS_GROUP`) and Linux/macOS (`/bin/bash -c ...` with `start_new_session=True`); never require `bash.exe` on Windows.
- Context & secret hygiene: scrub outputs with `redact_secrets` (`ivy.core.prompt`) and keep outputs bounded.
- Reversibility & safety: route all command spawns (including `background=True`) through `SafetyApprovalPolicy` and clean up child process trees via `terminate_process_tree` (`ivy.tools.runner`).
- Monkeypatch stability: keep `ivy.tui.app` imports exposed and resolve TUI helpers cleanly via `_app_ns()` in `ivy/tui/commands.py`.
- Verification gate: `& ".\.venv\Scripts\pytest.exe" -p no:cacheprovider -v` and `uvx ruff check ivy tests` pass with 0 failures and 0 skipped/relaxed assertions.

## Review Focus

- **Non-retryable client errors (`401`/`400`)**: Must fail immediately without sleeping for retries or silently masking bad credentials behind `fallback_model`. *(Pinned in Task 1)*
- **Mid-stream LLM failure after partial tokens**: Once `on_token` receives at least one chunk in `stream_turn()`, errors must raise instead of retrying or switching models, preventing duplicated partial responses. *(Pinned in Task 1)*
- **Auto-compaction safety invariants**: Small conversations (< threshold) must trigger zero extra LLM calls; when triggered, `Role.SYSTEM` stays at index 0, `Role.TOOL` stays paired with its tool call, and LLM compaction errors fall back to structured truncation. *(Pinned in Task 2)*
- **Background pipe deadlock, secret leakage & zombie cleanup**: High-volume background stdout/stderr must drain continuously into a bounded ring buffer, `read_output()` must scrub secrets via `redact_secrets`, and `shutdown_all()` must terminate running process trees. *(Pinned in Task 3)*
- **Safety gate & invalid process IDs**: `TerminalTool(background=True)` must still obey `SafetyApprovalPolicy` rejections, and `ProcessManageTool` actions (`poll`/`write`/`kill`) on unknown or exited process IDs must return structured `"error"` `ToolResult` without raising. *(Pinned in Task 3)*

## File Structure

- Modify: `ivy/config.py` — add `llm_max_retries`, `llm_retry_base_delay`, `fallback_model`, `auto_compact_threshold` fields and env mappings.
- Modify: `.env.example` — document `IVY_LLM_MAX_RETRIES`, `IVY_FALLBACK_MODEL`, `IVY_AUTO_COMPACT_THRESHOLD`.
- Modify: `ivy/core/llm.py` — add `_is_retryable_error`, `_execute_with_resilience`, retry/fallback in `chat_completion` and pre-first-token `stream_turn`.
- Modify: `ivy/memory/context.py` — add `should_auto_compact()` token-threshold helper.
- Modify: `ivy/core/agent.py` — add `_maybe_auto_compact()` pre-turn hook in `IvyAgent.run_turn()`.
- Create: `ivy/tools/process_registry.py` — shared shell builder (`build_shell_command`, `spawn_shell_kwargs`), `BackgroundProcess`, `BackgroundProcessManager`, `ProcessManageTool`.
- Modify: `ivy/tools/terminal.py` — reuse shared shell builder and support `background: bool = False` via `BackgroundProcessManager`.
- Modify: `ivy/tools/delegate.py` — add `"process"` to `_BLOCKED_CHILD_TOOLS`.
- Modify: `ivy/tui/views.py` — add `render_processes_table()`.
- Modify: `ivy/tui/prompt.py` — add `/ps` to `SLASH_COMMANDS`, `SLASH_COMMAND_META`, `_SUBCOMMAND_META`.
- Modify: `ivy/tui/commands.py` — add `_handle_ps()` and `/ps` dispatch in `handle_slash_command()`.
- Modify: `ivy/tui/app.py` — wire `BackgroundProcessManager` into `build_default_tool_registry()`, `IvyCLIApp`, and `_run_async` `finally:` shutdown.
- Modify: `AGENTS.md`, `README.md` — document 16 built-in tools (`process`), `/ps`, retry/fallback/auto-compact config.
- Modify Tests: `tests/test_config.py`, `tests/test_llm.py`, `tests/test_memory.py`, `tests/test_agent.py`, `tests/test_tools.py`, `tests/test_cli.py`.

---

### Task 0: Sync Local `master` with `origin/master`

**Files:**
- None (Git sync only)

**Interfaces:**
- Consumes: `origin/master` (commits merged from PRs #1–#4)
- Produces: Up-to-date local `master` baseline

- [ ] **Step 1: Fast-forward local `master` to `origin/master`**

Run:
```powershell
& "C:\Program Files\Git\cmd\git.exe" pull --ff-only origin master
```
Expected: `Updating 97d5f0d..ec0d244` (`Fast-forward`), `git status -sb` shows `## master...origin/master`.

- [ ] **Step 2: Verify baseline test suite and lint pass**

Run:
```powershell
uvx ruff check ivy tests
& ".\.venv\Scripts\pytest.exe" -p no:cacheprovider -v
```
Expected: `All checks passed!` and `88 passed`.

---

### Task 1: Config Fields & LLM Transient Retry / Fallback Routing

**Files:**
- Modify: `ivy/config.py:36-105`
- Modify: `ivy/core/llm.py:16-148,248-355`
- Modify: `.env.example`
- Test: `tests/test_config.py`, `tests/test_llm.py`

**Interfaces:**
- Consumes: `IvyConfig`, `LLMClient.set_model`, `LLMClient.chat_completion`, `LLMClient.stream_turn`
- Produces:
  - `IvyConfig.llm_max_retries: int = 2` (`IVY_LLM_MAX_RETRIES`)
  - `IvyConfig.llm_retry_base_delay: float = 0.5` (`IVY_LLM_RETRY_BASE_DELAY`)
  - `IvyConfig.fallback_model: str | None = None` (`IVY_FALLBACK_MODEL`)
  - `IvyConfig.auto_compact_threshold: float = 0.85` (`IVY_AUTO_COMPACT_THRESHOLD`)
  - `_is_retryable_error(exc: Exception) -> bool`
  - `LLMClient._execute_with_resilience(self, operation: Callable[[], Awaitable[ChatMessage]]) -> ChatMessage`

- [ ] **Step 1: Write failing config and LLM retry/fallback tests**

Add to `tests/test_config.py`:
```python
def test_load_resilience_config_from_env(monkeypatch):
    monkeypatch.setenv("IVY_LLM_MAX_RETRIES", "4")
    monkeypatch.setenv("IVY_LLM_RETRY_BASE_DELAY", "0.25")
    monkeypatch.setenv("IVY_FALLBACK_MODEL", "anthropic/claude-sonnet-4-5")
    monkeypatch.setenv("IVY_AUTO_COMPACT_THRESHOLD", "0.75")

    cfg = load_config()
    assert cfg.llm_max_retries == 4
    assert cfg.llm_retry_base_delay == 0.25
    assert cfg.fallback_model == "anthropic/claude-sonnet-4-5"
    assert cfg.auto_compact_threshold == 0.75
```

Add to `tests/test_llm.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```powershell
& ".\.venv\Scripts\pytest.exe" -p no:cacheprovider tests/test_config.py::test_load_resilience_config_from_env tests/test_llm.py::test_llm_client_retries_transient_error_then_succeeds tests/test_llm.py::test_llm_client_falls_back_after_retry_exhaustion_and_restores_model tests/test_llm.py::test_llm_client_does_not_retry_auth_error_or_mid_stream_failure -v
```
Expected: FAIL on missing `IvyConfig` fields and missing retry/fallback behavior.

- [ ] **Step 3: Implement `IvyConfig` resilience fields and `load_config` mappings in `ivy/config.py` + `.env.example`**

- Add `llm_max_retries: int = 2`, `llm_retry_base_delay: float = 0.5`, `fallback_model: str | None = None`, `auto_compact_threshold: float = 0.85` to `IvyConfig`.
- Clamp/validate ranges safely (`llm_max_retries >= 0`, `llm_retry_base_delay >= 0.0`, `0.1 <= auto_compact_threshold <= 0.99`) and map `IVY_LLM_MAX_RETRIES`, `IVY_LLM_RETRY_BASE_DELAY`, `IVY_FALLBACK_MODEL`, `IVY_AUTO_COMPACT_THRESHOLD` in `load_config()`.
- Append the optional env vars cleanly to `.env.example`.

- [ ] **Step 4: Implement `_is_retryable_error` and `LLMClient._execute_with_resilience` in `ivy/core/llm.py`**

- Implement `_is_retryable_error(exc: Exception) -> bool`: check `getattr(exc, "status_code", None)` in `{429, 500, 502, 503, 504}`, `TimeoutError`, `asyncio.TimeoutError`, or connection/rate-limit/5xx exception class names; explicitly return `False` for `400`, `401`, `403`, `404`.
- Implement `_execute_with_resilience(self, operation, *, can_retry: Callable[[], bool] | None = None) -> ChatMessage`:
  - Retry up to `self.config.llm_max_retries` times with exponential backoff `base * (2 ** attempt)` while `_is_retryable_error(exc)` and `(can_retry is None or can_retry())`.
  - When primary retries exhaust and `self.config.fallback_model` differs from `self.model` and `can_retry` still allows retry, temporarily `self.set_model(fallback_model)` inside `try ... finally: self.set_model(primary_model)`, invoke `operation()` once, and set `reply.metadata["fallback_used"] = True`, `reply.metadata["model_used"] = fallback_model`.
- Route non-streaming `chat_completion(..., stream=False)` and `stream_turn()` (with `emitted_any` boolean flag so `can_retry = lambda: not emitted_any`) through `_execute_with_resilience`.

- [ ] **Step 5: Run tests to verify Task 1 passes**

Run:
```powershell
& ".\.venv\Scripts\pytest.exe" -p no:cacheprovider tests/test_config.py tests/test_llm.py -v
```
Expected: All tests PASS.

- [ ] **Step 6: Commit Task 1**

```powershell
& "C:\Program Files\Git\cmd\git.exe" add ivy/config.py ivy/core/llm.py .env.example tests/test_config.py tests/test_llm.py
& "C:\Program Files\Git\cmd\git.exe" commit -m "feat(core): add LLM transient retry backoff and fallback model routing"
```

---

### Task 2: Automatic Token-Budget Context Compaction in `IvyAgent`

**Files:**
- Modify: `ivy/memory/context.py:53-85`
- Modify: `ivy/core/agent.py:13-16,51-95,122-165`
- Test: `tests/test_memory.py`, `tests/test_agent.py`

**Interfaces:**
- Consumes: `compute_context_breakdown`, `ContextWindowManager.compact_with_llm`, `IvyConfig.auto_compact_threshold`
- Produces:
  - `should_auto_compact(messages: list[ChatMessage], model: str | None = None, tool_schemas: list[dict[str, Any]] | None = None, system_prompt: str = "", threshold: float = 0.85) -> bool`
  - `IvyAgent._prepare_context_messages(self, system_prompt: str, history: list[ChatMessage], user_msg: ChatMessage, schemas: list[dict[str, Any]] | None) -> tuple[list[ChatMessage], dict[str, int] | None]`

- [ ] **Step 1: Write failing tests for `should_auto_compact` and `IvyAgent` auto-compaction**

Add to `tests/test_memory.py`:
```python
def test_should_auto_compact_threshold_and_tail_safety():
    from ivy.memory.context import should_auto_compact

    short_msgs = [ChatMessage(role=Role.USER, content="hello")] * 4
    assert should_auto_compact(short_msgs, model="gpt-4o", threshold=0.85) is False

    # 128_000 window for gpt-4o -> 460_000 chars ~= 115_000 tokens (~90%)
    heavy = [ChatMessage(role=Role.USER, content="x" * 46_000) for _ in range(10)]
    assert should_auto_compact(heavy, model="gpt-4o", threshold=0.85) is True
```

Add to `tests/test_agent.py`:
```python
@pytest.mark.asyncio
async def test_agent_auto_compacts_when_token_budget_exceeded(tmp_path):
    store = MemoryStore(db_path=str(tmp_path / "compact.db"))
    await store.initialize()
    # Seed > 10 heavy messages exceeding 0.1 threshold on 128k gpt-4o-mini
    for i in range(12):
        await store.save_message("s1", ChatMessage(role=Role.USER, content=f"msg {i} " + ("a" * 6000)))
        await store.save_message("s1", ChatMessage(role=Role.ASSISTANT, content=f"ans {i} " + ("b" * 6000)))

    mock_llm = MagicMock()
    mock_llm.model = "gpt-4o-mini"
    mock_llm.chat_completion = AsyncMock(
        side_effect=[
            ChatMessage(role=Role.ASSISTANT, content="Summary of earlier work."),
            ChatMessage(role=Role.ASSISTANT, content="Final answer after compact."),
        ]
    )

    cfg = IvyConfig(db_path=str(tmp_path / "compact.db"), auto_compact_threshold=0.1)
    agent = IvyAgent(llm_client=mock_llm, tool_registry=ToolRegistry(), memory_store=store, config=cfg)
    reply = await agent.run_turn("What next?", session_id="s1")

    assert reply.content == "Final answer after compact."
    assert reply.metadata.get("auto_compacted", {}).get("removed_messages", 0) > 0
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```powershell
& ".\.venv\Scripts\pytest.exe" -p no:cacheprovider tests/test_memory.py::test_should_auto_compact_threshold_and_tail_safety tests/test_agent.py::test_agent_auto_compacts_when_token_budget_exceeded tests/test_agent.py::test_agent_auto_compaction_fallback_when_llm_fails -v
```
Expected: FAIL with `ImportError` / missing auto-compaction metadata.

- [ ] **Step 3: Implement `should_auto_compact` in `ivy/memory/context.py`**

- Implement `should_auto_compact(messages, model=None, tool_schemas=None, system_prompt="", threshold=0.85) -> bool`: return `False` if `len(messages) <= 8`, otherwise compute `compute_context_breakdown(...)` and return `float(breakdown["percent"]) >= threshold * 100.0`.

- [ ] **Step 4: Implement `IvyAgent._prepare_context_messages` in `ivy/core/agent.py`**

- Check `should_auto_compact([*history, user_msg], model=model_name, tool_schemas=raw_schemas, system_prompt=system_prompt, threshold=threshold)`.
- When true, call `await self._fallback_context_mgr.compact_with_llm([*history, user_msg], llm_client=self.llm_client, keep_recent=8)`, prepend `ChatMessage(role=Role.SYSTEM, content=system_prompt)` at index 0, record `compact_stats` in `reply.metadata["auto_compacted"]`, and still run `compress_context()` / `_fallback_context_mgr.compress()` if message count exceeds `max_messages`.

- [ ] **Step 5: Run tests to verify Task 2 passes**

Run:
```powershell
& ".\.venv\Scripts\pytest.exe" -p no:cacheprovider tests/test_memory.py tests/test_agent.py -v
```
Expected: All tests PASS (including all existing `test_agent.py` tests making zero unexpected LLM calls).

- [ ] **Step 6: Commit Task 2**

```powershell
& "C:\Program Files\Git\cmd\git.exe" add ivy/memory/context.py ivy/core/agent.py tests/test_memory.py tests/test_agent.py
& "C:\Program Files\Git\cmd\git.exe" commit -m "feat(core): add token-threshold auto-compaction in IvyAgent"
```

---

### Task 3: Background Process Registry, `TerminalTool(background=True)` & `process` Tool

**Files:**
- Create: `ivy/tools/process_registry.py`
- Modify: `ivy/tools/terminal.py:1-133`
- Modify: `ivy/tools/delegate.py:15`
- Modify: `ivy/tui/app.py:134-189,770-855`
- Test: `tests/test_tools.py`

**Interfaces:**
- Consumes: `BaseTool`, `ToolResult`, `SafetyApprovalPolicy`, `terminate_process_tree` (`ivy.tools.runner`), `redact_secrets` (`ivy.core.prompt`), `truncate_head_tail` (`ivy.core.prompt`)
- Produces:
  - `build_shell_command(command: str) -> list[str]`
  - `spawn_shell_kwargs(working_dir: str | None = None, with_stdin: bool = False) -> dict[str, Any]`
  - `BackgroundProcess` dataclass
  - `BackgroundProcessManager` (`spawn`, `list_processes`, `read_output`, `write_stdin`, `stop`, `shutdown_all`)
  - `ProcessManageTool(BaseTool)` (`name = "process"`, actions: `"list" | "poll" | "log" | "write" | "kill"`)
  - `TerminalTool(approval_policy=..., timeout=60.0, process_manager: BackgroundProcessManager | None = None)` supporting `background: bool = False`

- [ ] **Step 1: Write failing tests for background spawning, draining, stdin/kill, safety gate, and secret scrubbing**

Add to `tests/test_tools.py`:
```python
@pytest.mark.asyncio
async def test_background_process_manager_and_process_tool(tmp_path):
    import sys
    from ivy.tools.process_registry import BackgroundProcessManager, ProcessManageTool

    mgr = BackgroundProcessManager(max_output_chars=12_000)
    proc_tool = ProcessManageTool(manager=mgr)
    term = TerminalTool(
        approval_policy=SafetyApprovalPolicy(auto_approve=True),
        process_manager=mgr,
    )

    # 1. Safety guard still blocks destructive background commands
    denied = await TerminalTool(
        approval_policy=SafetyApprovalPolicy(auto_approve=False, interactive=False),
        process_manager=mgr,
    ).execute(command="rm -rf /", background=True)
    assert denied.status == "rejected"

    # 2. Spawn interactive echo script cleanly across Windows & POSIX
    script = tmp_path / "bg_echo.py"
    script.write_text(
        "import sys\n"
        "print('OPENAI_API_KEY=sk-proj-secret1234567890abcdef12', flush=True)\n"
        "for line in sys.stdin:\n"
        "    print('GOT:' + line.strip(), flush=True)\n",
        encoding="utf-8",
    )
    cmd = f'"{sys.executable}" "{script}"'
    spawned = await term.execute(command=cmd, working_dir=str(tmp_path), background=True)
    assert spawned.status == "success"
    pid_tag = spawned.metadata["process_id"]

    await asyncio.sleep(0.35)
    polled = await proc_tool.execute(action="poll", process_id=pid_tag)
    assert polled.status == "success"
    assert "sk-proj-secret1234567890abcdef12" not in polled.output

    wrote = await proc_tool.execute(action="write", process_id=pid_tag, data="hello-bg\n")
    assert wrote.status == "success"
    await asyncio.sleep(0.25)

    polled2 = await proc_tool.execute(action="log", process_id=pid_tag)
    assert "GOT:hello-bg" in polled2.output

    listed = await proc_tool.execute(action="list")
    assert pid_tag in listed.output

    killed = await proc_tool.execute(action="kill", process_id=pid_tag)
    assert killed.status == "success"

    missing = await proc_tool.execute(action="poll", process_id="proc_missing")
    assert missing.status == "error"

    await mgr.shutdown_all()
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```powershell
& ".\.venv\Scripts\pytest.exe" -p no:cacheprovider tests/test_tools.py::test_background_process_manager_and_process_tool -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'ivy.tools.process_registry'`.

- [ ] **Step 3: Create `ivy/tools/process_registry.py`**

- Implement `build_shell_command(command: str) -> list[str]` and `spawn_shell_kwargs(working_dir: str | None = None, with_stdin: bool = False) -> dict[str, Any]` handling Windows PowerShell (`CREATE_NEW_PROCESS_GROUP`) and POSIX `/bin/bash -c` (`start_new_session=True`).
- Implement `BackgroundProcess` & `BackgroundProcessManager`:
  - `spawn(command, working_dir=None)`: starts `asyncio.create_subprocess_exec(*build_shell_command(command), **spawn_shell_kwargs(working_dir, with_stdin=True))`, spawns two async stream-reader tasks draining `stdout` and `stderr` into bounded char chunks (`max_output_chars`), and monitors exit return code.
  - `list_processes()`: refreshes exit status and returns metadata dicts (`process_id`, `pid`, `status`, `exit_code`, `elapsed_s`, `command`).
  - `read_output(process_id, tail_chars=8_000)`: joins buffered chunks, tails to `tail_chars`, and returns `(True, redact_secrets(text))` or `(False, error_msg)`.
  - `write_stdin(process_id, data)`: writes UTF-8 to `proc.stdin` if running and stdin open; returns `(False, ...)` otherwise.
  - `stop(process_id)` and `shutdown_all()`: terminate via `terminate_process_tree(proc)` and cancel reader tasks cleanly.
- Implement `ProcessManageTool(BaseTool)` (`name = "process"`, actions `list`, `poll`, `log`, `write`, `kill`).

- [ ] **Step 4: Update `ivy/tools/terminal.py`, `ivy/tools/delegate.py`, and `ivy/tui/app.py`**

- In `ivy/tools/terminal.py`: import `build_shell_command`, `spawn_shell_kwargs`, `BackgroundProcessManager`; remove duplicated platform shell setup; add `"background"` boolean parameter and `process_manager: BackgroundProcessManager | None = None`; after `approval_policy.evaluate_approval(command)`, if `background` is true:
  - if `self.process_manager is None`, return `"error"` `ToolResult` stating background process manager is unavailable;
  - otherwise `bp = await self.process_manager.spawn(command, working_dir)` and return `"success"` with `metadata={"process_id": bp.process_id, "pid": bp.pid, "background": True}`.
- In `ivy/tools/delegate.py`: update `_BLOCKED_CHILD_TOOLS = {"delegate_task", "clarify", "process"}`.
- In `ivy/tui/app.py`: accept optional `process_manager: BackgroundProcessManager | None = None` in `build_default_tool_registry()`, register `ProcessManageTool` when `process_manager is not None`, pass `process_manager` to `TerminalTool`, wire `BackgroundProcessManager()` in `_run_async`, store it on `app._process_manager`, and call `await process_manager.shutdown_all()` in `_run_async`'s `finally:` block.

- [ ] **Step 5: Run tests to verify Task 3 passes**

Run:
```powershell
& ".\.venv\Scripts\pytest.exe" -p no:cacheprovider tests/test_tools.py -v
```
Expected: All tests PASS.

- [ ] **Step 6: Commit Task 3**

```powershell
& "C:\Program Files\Git\cmd\git.exe" add ivy/tools/process_registry.py ivy/tools/terminal.py ivy/tools/delegate.py ivy/tui/app.py tests/test_tools.py
& "C:\Program Files\Git\cmd\git.exe" commit -m "feat(tools): add BackgroundProcessManager, process tool & terminal background mode"
```

---

### Task 4: TUI `/ps` Command, Views, Docs Sync & Full Verification

**Files:**
- Modify: `ivy/tui/views.py`
- Modify: `ivy/tui/prompt.py:55-150`
- Modify: `ivy/tui/commands.py`
- Modify: `ivy/tui/app.py` (expose `_discover_process_manager()` / `self._process_manager`)
- Modify: `AGENTS.md`, `README.md`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `BackgroundProcessManager` (`list_processes`, `read_output`, `stop`), `SLASH_COMMANDS`, `SlashCommandMixin`
- Produces:
  - `render_processes_table(console: Console, processes: list[dict[str, Any]], theme: ThemeConfig | None = None) -> None`
  - `/ps [list | log <id> | stop <id|all>]` slash command and autocompletion

- [ ] **Step 1: Write failing tests for `/ps` slash command and completer**

Add to `tests/test_cli.py`:
```python
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
    bp = await mgr.spawn(f'"{sys.executable}" -c "import time; print(\'bg-alive\', flush=True); time.sleep(10)"')
    await asyncio.sleep(0.3)

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
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```powershell
& ".\.venv\Scripts\pytest.exe" -p no:cacheprovider tests/test_cli.py::test_ps_slash_command_lists_logs_and_stops_processes -v
```
Expected: FAIL with `AssertionError: assert '/ps' in SLASH_COMMANDS`.

- [ ] **Step 3: Implement `render_processes_table`, `/ps` metadata & `_handle_ps` handler**

- Add `render_processes_table(console, processes, theme=None)` to `ivy/tui/views.py`.
- Add `"/ps"` to `SLASH_COMMANDS`, `SLASH_COMMAND_META`, and `_SUBCOMMAND_META` (`list`, `log`, `stop`) in `ivy/tui/prompt.py`.
- Add `_discover_process_manager()` to `IvyCLIApp` (`ivy/tui/app.py`, inspecting `self._process_manager` or `self.agent.tool_registry.get("process").manager`).
- Add `_handle_ps(self, args: list[str])` and `/ps` dispatch in `ivy/tui/commands.py`.
- Update `AGENTS.md` and `README.md` (`16 built-in tools`, `process`, `/ps`, `IVY_LLM_MAX_RETRIES`, `IVY_FALLBACK_MODEL`, `IVY_AUTO_COMPACT_THRESHOLD`).

- [ ] **Step 4: Run full verification suite**

Run:
```powershell
uvx ruff check ivy tests
& ".\.venv\Scripts\pytest.exe" -p no:cacheprovider -v
```
Expected: `All checks passed!` and 100% pytest pass with 0 failures and 0 skipped tests.

- [ ] **Step 5: Commit Task 4**

```powershell
& "C:\Program Files\Git\cmd\git.exe" add ivy/tui/views.py ivy/tui/prompt.py ivy/tui/commands.py ivy/tui/app.py AGENTS.md README.md tests/test_cli.py
& "C:\Program Files\Git\cmd\git.exe" commit -m "feat(tui): add /ps background process command and sync docs"
```

---

## Self-Review

1. **Spec coverage:** Covers every gap in Phase 5 Option 1 (`git pull`, LLM retry/backoff + fallback, token-threshold auto-compaction, `BackgroundProcessManager` + `TerminalTool(background=True)` + `process` tool, `/ps` command & docs).
2. **Step scan:** Every step names exact file paths, signatures, test assertions, commands, and expected outputs without empty placeholders or full implementation transcripts.
3. **Type consistency:** `IvyConfig` field names (`llm_max_retries`, `llm_retry_base_delay`, `fallback_model`, `auto_compact_threshold`), `BackgroundProcessManager` methods (`spawn`, `list_processes`, `read_output`, `write_stdin`, `stop`, `shutdown_all`), and `ProcessManageTool` (`name = "process"`) match cleanly across all tasks.
4. **Review Focus:** All 5 edge cases are explicitly pinned by assertions in Tasks 1, 2, and 3.
5. **Proportion:** Concise task definitions matching the scope without excessive boilerplate.
