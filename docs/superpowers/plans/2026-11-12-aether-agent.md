# Aether Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Membangun `aether-agent`, sebuah agen AI otonom modular berbasis Python modern terinspirasi Hermes Agent, lengkap dengan TUI terminal, tool multi-backend, closed learning loop `agentskills.io`, memori SQLite FTS5, gateway Telegram/Discord, cron scheduler, dan subagent delegation.

**Architecture:** Arsitektur modular berorientasi asynchronous event-loop: Core ReAct Loop mengeksekusi tools dengan safety approval, terhubung ke SQLite state store dengan FTS5 dan turn lease. Messaging gateway daemon mengelola multi-platform adapters (CLI, Telegram, Discord) dan background cron scheduler secara non-blocking.

**Tech Stack:** Python 3.11+, Astral `uv`, `litellm` / `openai`, `pydantic`, `prompt_toolkit`, `rich`, `sqlite3` (FTS5), `croniter`, `pytest`, `pytest-asyncio`.

**Spec:** `docs/superpowers/specs/2026-11-12-aether-agent-design.md`

## Global Constraints
- Target platform: Windows (native PowerShell) dan POSIX (Linux/macOS).
- Package & Environment Manager: Wajib menggunakan `uv` (`pyproject.toml`, `uv.lock`).
- Python version: `>=3.11`.
- Skill format: Standar terbuka `agentskills.io` (`SKILL.md` dengan YAML frontmatter).
- Concurrency & State: SQLite terpusat dengan mekanisme async *Turn Lease* untuk mencegah konflik antar-platform.

## Review Focus
1. **Perintah Destruktif Terminal:** Pengguna menolak perintah berbahaya via safety guard -> Sistem tidak boleh crash, harus mengembalikan `ToolResult(status="rejected")` agar agen merefleksikan dan merencanakan alternatif.
2. **Koneksi Telegram/Discord Ilegal:** Pesan dari user ID di luar whitelist konfigurasi -> Harus diabaikan secara diam-diam (*silent drop*) atau ditolak dengan log audit demi keamanan host.
3. **Pencarian Riwayat Sesi (FTS5):** Query pencarian dengan karakter khusus SQLite FTS5 (seperti tanda kutip atau boolean tanpa pasangan) -> Harus disanitasi agar tidak memicu SQLite syntax error.
4. **Eksekusi Code Runner Menggantung:** Skrip Python dengan *infinite loop* -> Subprocess runner harus memicu timeout tepat waktu (30 detik) dan mematikan child process dengan bersih.
5. **Skill Malformed / Rusak:** File `SKILL.md` tanpa YAML frontmatter yang valid -> Engine harus melewatinya dengan log peringatan (*warning*) tanpa menghentikan pemuatan skill lainnya.

---

### Task 1: Project Scaffolding & Configuration with `uv`

**Files:**
- Create: `pyproject.toml`
- Create: `.env.example`
- Create: `aether_agent/__init__.py`
- Create: `aether_agent/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: None (Root setup)
- Produces: `AetherConfig` class dengan konfigurasi model, database paths, dan allowed user lists.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
from aether_agent.config import load_config

def test_load_default_config():
    config = load_config()
    assert config.app_name == "aether-agent"
    assert config.default_model is not None
    assert config.db_path.endswith("aether.db")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL (ModuleNotFoundError: No module named 'aether_agent')

- [ ] **Step 3: Implement project configuration & `pyproject.toml`**

Buat `pyproject.toml` dengan dependensi inti (`litellm`, `openai`, `pydantic>=2.0`, `rich`, `prompt_toolkit`, `croniter`, `python-dotenv`) dan definisikan class `AetherConfig` di `aether_agent/config.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml .env.example aether_agent/ tests/test_config.py
git commit -m "feat: scaffold aether-agent project with uv and config"
```

---

### Task 2: Core Data Types & Universal LLM Adapter

**Files:**
- Create: `aether_agent/core/types.py`
- Create: `aether_agent/core/llm.py`
- Test: `tests/test_llm.py`

**Interfaces:**
- Consumes: `AetherConfig` from `aether_agent/config.py`
- Produces: `ChatMessage`, `ToolCall`, `ToolResult`, dan `LLMClient.chat_completion(messages, tools, stream=...)`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_llm.py
import pytest
from aether_agent.core.types import ChatMessage, Role
from aether_agent.core.llm import LLMClient
from aether_agent.config import AetherConfig

def test_chat_message_serialization():
    msg = ChatMessage(role=Role.USER, content="Hello")
    assert msg.model_dump()["role"] == "user"

@pytest.mark.asyncio
async def test_llm_client_initialization():
    cfg = AetherConfig(default_model="gpt-4o-mini")
    client = LLMClient(cfg)
    assert client.model == "gpt-4o-mini"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_llm.py -v`
Expected: FAIL

- [ ] **Step 3: Implement types and LLMClient**

Implementasikan Pydantic models di `aether_agent/core/types.py` dan adaptor multi-provider asynchronous berbasis `litellm` di `aether_agent/core/llm.py` yang mendukung streaming dan tool calling.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_llm.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add aether_agent/core/ tests/test_llm.py
git commit -m "feat: implement core types and universal LLM adapter"
```

---

### Task 3: Tool Engine, Multi-Backend Tools & Safety Guard

**Files:**
- Create: `aether_agent/tools/base.py`
- Create: `aether_agent/tools/registry.py`
- Create: `aether_agent/tools/terminal.py`
- Create: `aether_agent/tools/files.py`
- Create: `aether_agent/tools/runner.py`
- Create: `aether_agent/tools/approval.py`
- Test: `tests/test_tools.py`

**Interfaces:**
- Consumes: `ToolResult` from `aether_agent/core/types.py`
- Produces: `ToolRegistry`, `BaseTool`, `TerminalTool`, `FileTools`, `CodeRunnerTool`, dan `SafetyApprovalPolicy`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tools.py
import pytest
from aether_agent.tools.registry import ToolRegistry
from aether_agent.tools.terminal import TerminalTool
from aether_agent.tools.approval import SafetyApprovalPolicy

@pytest.mark.asyncio
async def test_terminal_tool_safe_execution():
    policy = SafetyApprovalPolicy(auto_approve=True)
    term = TerminalTool(approval_policy=policy)
    res = await term.execute(command="echo Hello Aether")
    assert res.status == "success"
    assert "Hello Aether" in res.output

@pytest.mark.asyncio
async def test_safety_guard_blocks_destructive_command():
    policy = SafetyApprovalPolicy(auto_approve=False) # user rejects
    term = TerminalTool(approval_policy=policy)
    res = await term.execute(command="rm -rf /")
    assert res.status == "rejected"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_tools.py -v`
Expected: FAIL

- [ ] **Step 3: Implement tools & safety approval engine**

Implementasikan `BaseTool` dengan auto JSON-schema generator, `TerminalTool` (PowerShell di Windows, Bash di Linux/macOS), `FileTools` (read, write, list, patch), `CodeRunnerTool` (isolated Python subprocess with 30s timeout), dan `SafetyApprovalPolicy`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_tools.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add aether_agent/tools/ tests/test_tools.py
git commit -m "feat: implement tool registry, terminal, file tools, and safety guard"
```

---

### Task 4: Memory System & SQLite with FTS5 Search

**Files:**
- Create: `aether_agent/memory/store.py`
- Create: `aether_agent/memory/fts.py`
- Create: `aether_agent/memory/profile.py`
- Create: `aether_agent/memory/context.py`
- Test: `tests/test_memory.py`

**Interfaces:**
- Consumes: `ChatMessage` from `aether_agent/core/types.py`
- Produces: `MemoryStore` dengan metode `save_turn`, `search_history_fts(query)`, `get_user_profile()`, dan `compress_context()`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_memory.py
import pytest
from aether_agent.memory.store import MemoryStore
from aether_agent.core.types import ChatMessage, Role

@pytest.mark.asyncio
async def test_memory_store_fts_search(tmp_path):
    db_file = str(tmp_path / "test.db")
    store = MemoryStore(db_path=db_file)
    await store.initialize()
    
    session_id = "test-session-1"
    await store.save_message(session_id, ChatMessage(role=Role.USER, content="Membahas instalasi Python uv"))
    await store.save_message(session_id, ChatMessage(role=Role.ASSISTANT, content="uv adalah package manager cepat"))
    
    results = await store.search_fts("instalasi")
    assert len(results) > 0
    assert "Membahas instalasi" in results[0]["content"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_memory.py -v`
Expected: FAIL

- [ ] **Step 3: Implement SQLite FTS5 store & memory profile**

Implementasikan inisialisasi tabel SQLite dengan virtual table `FTS5` (`messages_fts`), sanitasi query FTS5, penyimpanan profil pengguna jangka panjang (`user_profile`), dan sliding window context manager.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_memory.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add aether_agent/memory/ tests/test_memory.py
git commit -m "feat: implement memory system with SQLite FTS5 and user profiling"
```

---

### Task 5: Closed Learning Loop & Skills Hub (`agentskills.io`)

**Files:**
- Create: `aether_agent/skills/format.py`
- Create: `aether_agent/skills/engine.py`
- Create: `aether_agent/skills/generator.py`
- Test: `tests/test_skills.py`

**Interfaces:**
- Consumes: `ChatMessage` dan `ToolResult`
- Produces: `SkillEngine` (discovery & `read_skill`), `SkillGenerator` (mengekstrak alur sukses menjadi `SKILL.md`), dan parser `agentskills.io`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_skills.py
import pytest
from aether_agent.skills.engine import SkillEngine
from aether_agent.skills.format import parse_skill_markdown

def test_parse_agentskills_markdown(tmp_path):
    skill_content = """---
name: sample-analyzer
description: Menganalisa file teks lokal
---
# Prosedur
1. Baca file
2. Berikan ringkasan
"""
    skill = parse_skill_markdown(skill_content)
    assert skill.name == "sample-analyzer"
    assert skill.description == "Menganalisa file teks lokal"
    assert "Prosedur" in skill.instructions

@pytest.mark.asyncio
async def test_skill_engine_discovery(tmp_path):
    skills_dir = tmp_path / "skills"
    skill_folder = skills_dir / "sample"
    skill_folder.mkdir(parents=True)
    (skill_folder / "SKILL.md").write_text("---\nname: sample\ndescription: Test skill\n---\nSteps")
    
    engine = SkillEngine(skills_dir=str(skills_dir))
    await engine.load_skills()
    skills = engine.list_skills()
    assert len(skills) == 1
    assert skills[0]["name"] == "sample"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_skills.py -v`
Expected: FAIL

- [ ] **Step 3: Implement skills parser, engine, and generator**

Implementasikan parser format `agentskills.io` di `format.py`, dynamic loader di `engine.py`, dan prompt refleksi otomatis di `generator.py` untuk mensintesis skill baru ke dalam folder `skills/<name>/SKILL.md`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_skills.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add aether_agent/skills/ tests/test_skills.py
git commit -m "feat: implement closed learning loop and skills engine"
```

---

### Task 6: Core ReAct Agent Loop & Subagent Delegation

**Files:**
- Create: `aether_agent/core/prompt.py`
- Create: `aether_agent/core/agent.py`
- Create: `aether_agent/core/subagent.py`
- Test: `tests/test_agent.py`

**Interfaces:**
- Consumes: `LLMClient`, `ToolRegistry`, `MemoryStore`, `SkillEngine`
- Produces: `AetherAgent.run_turn(prompt, session_id, stream_callback=...)` dan `SubagentRunner`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_agent.py
import pytest
from unittest.mock import AsyncMock, MagicMock
from aether_agent.core.agent import AetherAgent
from aether_agent.core.types import ChatMessage, Role, ToolCall

@pytest.mark.asyncio
async def test_agent_react_turn_execution():
    mock_llm = MagicMock()
    # Mock LLM calling a tool then returning final answer
    mock_llm.chat_completion = AsyncMock(side_effect=[
        ChatMessage(role=Role.ASSISTANT, content="", tool_calls=[
            ToolCall(id="call_1", function_name="terminal", arguments={"command": "echo 42"})
        ]),
        ChatMessage(role=Role.ASSISTANT, content="Hasil perhitungan adalah 42.")
    ])
    
    mock_tools = MagicMock()
    mock_tools.execute = AsyncMock(return_value=MagicMock(status="success", output="42"))
    mock_tools.get_schemas = MagicMock(return_value=[])
    
    agent = AetherAgent(llm_client=mock_llm, tool_registry=mock_tools)
    response = await agent.run_turn(prompt="Hitung 42", session_id="test-session")
    
    assert "42" in response.content
    assert mock_tools.execute.called
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_agent.py -v`
Expected: FAIL

- [ ] **Step 3: Implement ReAct agent loop and subagent delegation**

Susun context builder di `prompt.py`, ReAct orchestrator dengan penanganan tool-call & replanning di `agent.py`, serta `SubagentRunner` di `subagent.py` dengan session ID anak terisolasi.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_agent.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add aether_agent/core/ tests/test_agent.py
git commit -m "feat: implement ReAct agent loop and subagent runner"
```

---

### Task 7: Background Cron Scheduler

**Files:**
- Create: `aether_agent/cron/task.py`
- Create: `aether_agent/cron/scheduler.py`
- Test: `tests/test_cron.py`

**Interfaces:**
- Consumes: `AetherAgent`
- Produces: `CronScheduler` (`add_job`, `remove_job`, `start`, `stop`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cron.py
import pytest
import asyncio
from aether_agent.cron.scheduler import CronScheduler
from aether_agent.cron.task import ScheduledTask

@pytest.mark.asyncio
async def test_cron_scheduler_lifecycle():
    scheduler = CronScheduler()
    executed = False
    
    async def sample_handler():
        nonlocal executed
        executed = True
        
    task = ScheduledTask(name="test-job", cron_expr="* * * * *", handler=sample_handler)
    scheduler.add_task(task)
    assert len(scheduler.list_tasks()) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cron.py -v`
Expected: FAIL

- [ ] **Step 3: Implement cron engine with `croniter`**

Implementasikan penjadwal asinkron di `aether_agent/cron/scheduler.py` yang mengevaluasi jadwal berikutnya menggunakan `croniter`, memicu dispatching task tanpa memblokir thread utama, dan mencatat eksekusi ke database.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_cron.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add aether_agent/cron/ tests/test_cron.py
git commit -m "feat: implement background cron scheduler"
```

---

### Task 8: Messaging Gateway & Multi-Platform Adapters

**Files:**
- Create: `aether_agent/gateway/turn_lease.py`
- Create: `aether_agent/gateway/session_hub.py`
- Create: `aether_agent/gateway/server.py`
- Create: `aether_agent/gateway/adapters/base.py`
- Create: `aether_agent/gateway/adapters/telegram_adapter.py`
- Create: `aether_agent/gateway/adapters/discord_adapter.py`
- Test: `tests/test_gateway.py`

**Interfaces:**
- Consumes: `AetherAgent`, `MemoryStore`, `AetherConfig`
- Produces: `GatewayServer`, `TurnLeaseManager`, dan channel adapters.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_gateway.py
import pytest
import asyncio
from aether_agent.gateway.turn_lease import TurnLeaseManager

@pytest.mark.asyncio
async def test_turn_lease_prevents_concurrent_turns():
    lease_mgr = TurnLeaseManager()
    session_id = "session_x"
    
    # Acquire first lease
    lease1 = await lease_mgr.acquire(session_id)
    assert lease1 is True
    
    # Second lease on same session should fail or queue
    lease2 = await lease_mgr.acquire(session_id, timeout=0.1)
    assert lease2 is False
    
    await lease_mgr.release(session_id)
    lease3 = await lease_mgr.acquire(session_id)
    assert lease3 is True
    await lease_mgr.release(session_id)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_gateway.py -v`
Expected: FAIL

- [ ] **Step 3: Implement turn lease, gateway server, and bot adapters**

Implementasikan `TurnLeaseManager` untuk kontrol konkurensi, `GatewayServer` asynchronous, serta adapter Telegram dan Discord dengan whitelist otorisasi ID pengguna.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_gateway.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add aether_agent/gateway/ tests/test_gateway.py
git commit -m "feat: implement gateway daemon with turn leasing and messaging adapters"
```

---

### Task 9: Terminal TUI, Slash Commands & CLI Entrypoint

**Files:**
- Create: `aether_agent/tui/prompt.py`
- Create: `aether_agent/tui/views.py`
- Create: `aether_agent/tui/app.py`
- Modify: `pyproject.toml` (daftarkan script entrypoint `aether = "aether_agent.tui.app:main"`)
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `AetherAgent`, `GatewayServer`, `SkillEngine`, `AetherConfig`
- Produces: CLI binary `aether` dengan subcommands (`aether`, `aether gateway start`, `aether model`, `aether skills`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli.py
from aether_agent.tui.app import build_parser

def test_cli_parser_subcommands():
    parser = build_parser()
    args = parser.parse_args(["gateway", "start"])
    assert args.command == "gateway"
    assert args.gateway_action == "start"
    
    args_default = parser.parse_args([])
    assert args_default.command is None # defaults to interactive TUI
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli.py -v`
Expected: FAIL

- [ ] **Step 3: Implement TUI & CLI application**

Implementasikan `views.py` menggunakan `rich` (formatting Markdown, live spinners, box borders), `prompt.py` dengan `prompt_toolkit` untuk multiline input dan autocomplete slash commands (`/model`, `/skills`, `/clear`, `/exit`), serta `app.py` sebagai entrypoint utama.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_cli.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add aether_agent/tui/ tests/test_cli.py pyproject.toml
git commit -m "feat: implement rich terminal TUI, slash commands, and CLI entrypoint"
```

---

### Task 10: End-to-End System Integration & Verification

**Files:**
- Create: `tests/test_e2e_integration.py`
- Create: `README.md`
- Create: `skills/sample-coder/SKILL.md`

**Interfaces:**
- Consumes: Seluruh modul yang telah diimplementasikan
- Produces: Full workflow verification (CLI -> Agent ReAct -> Tool -> Auto-Skill Creation -> FTS5 Recall).

- [ ] **Step 1: Write end-to-end integration test**

```python
# tests/test_e2e_integration.py
import pytest
from aether_agent.config import AetherConfig
from aether_agent.core.agent import AetherAgent
from aether_agent.tools.registry import ToolRegistry
from aether_agent.tools.terminal import TerminalTool
from aether_agent.tools.approval import SafetyApprovalPolicy
from aether_agent.memory.store import MemoryStore
from aether_agent.skills.engine import SkillEngine

@pytest.mark.asyncio
async def test_full_agent_workflow(tmp_path):
    db_file = str(tmp_path / "aether.db")
    skills_dir = str(tmp_path / "skills")
    
    mem_store = MemoryStore(db_path=db_file)
    await mem_store.initialize()
    
    tools = ToolRegistry()
    tools.register(TerminalTool(approval_policy=SafetyApprovalPolicy(auto_approve=True)))
    
    skills = SkillEngine(skills_dir=skills_dir)
    await skills.load_skills()
    
    # Verify all components integrate smoothly without schema conflicts
    assert len(tools.get_schemas()) >= 1
    assert mem_store.db_path == db_file
```

- [ ] **Step 2: Run all tests in test suite**

Run: `pytest -v`
Expected: All unit & integration tests PASS.

- [ ] **Step 3: Document complete usage in `README.md`**

Tulis dokumentasi instalasi (`uv sync`), konfigurasi `.env`, menjalankan TUI interaktif (`uv run aether`), dan menjalankan gateway daemon (`uv run aether gateway start`).

- [ ] **Step 4: Final verification and commit**

```bash
git add tests/ README.md skills/
git commit -m "feat: complete end-to-end integration tests and documentation"
```
