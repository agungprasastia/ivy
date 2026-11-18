# Phase 6 Security Guards, Skill Scanner & Curated Memory Implementation Plan

> **For agentic workers:** Execute via `superpowers:executing-plans`, following `superpowers:test-driven-development` strictly task-by-task (RED -> GREEN -> commit).

**Goal:** Add path security + read-before-write enforcement, SSRF-safe HTTP fetch redirect validation, static `SKILL.md` security scanning, and SQLite curated memory notes injected into system prompts.

**Architecture:** Zero new third-party dependencies. New focused security helpers (`ivy/tools/path_security.py`, `ivy/tools/url_safety.py`, `ivy/skills/guard.py`) and `memory_notes` store/tool (`ivy/memory/store.py`, `ivy/tools/memory_note.py`) wired cleanly into `build_default_tool_registry()`, `IvyAgent`, and `IvyConfig`.

**Tech Stack:** Python 3.11+, `ipaddress`, `socket`, `urllib.parse`, `re`, `sqlite3`, `httpx`, `pydantic`, `pytest`, `ruff`.

---

### Task 1: Path Security & Read-Before-Write Guard

**Files:**
- Create: `ivy/tools/path_security.py`
- Modify: `ivy/config.py`, `ivy/tools/files.py`, `ivy/tui/app.py`
- Test: `tests/test_tools.py`, `tests/test_config.py`

- [ ] **Step 1 (RED):** Add unit tests in `tests/test_config.py` for `IVY_RESTRICT_WORKSPACE` / `IVY_ALLOW_PRIVATE_URLS` parsing and in `tests/test_tools.py` for:
  - blocking control chars and sensitive credential paths (`.ssh/id_rsa`, `.aws/credentials`, `.env`, `.git-credentials`) while allowing `.env.example`,
  - strict `workspace_root` escape rejection when `workspace_root` is set,
  - `require_read_before_overwrite=True` rejecting overwriting/patching an unread existing file while allowing creating new files and editing after `read_file`.
- [ ] **Step 2:** Run pytest on those new tests and confirm they fail as expected (`ImportError` / missing behavior).
- [ ] **Step 3 (GREEN):** Implement `ivy/tools/path_security.py`, update `IvyConfig` + `load_config()` in `ivy/config.py`, integrate `validate_file_path` and `read_tracker` / `require_read_before_overwrite` in `ivy/tools/files.py`, and wire `workspace_root` + `require_read_before_overwrite=True` in `build_default_tool_registry()` (`ivy/tui/app.py`).
- [ ] **Step 4:** Run `& ".\.venv\Scripts\pytest.exe" -p no:cacheprovider tests/test_tools.py tests/test_config.py -v` and confirm all pass.
- [ ] **Step 5:** Commit Task 1.

---

### Task 2: SSRF & Redirect Validation in `WebFetchTool`

**Files:**
- Create: `ivy/tools/url_safety.py`
- Modify: `ivy/tools/web.py`, `ivy/tui/app.py`
- Test: `tests/test_tools.py`

- [ ] **Step 1 (RED):** Add tests in `tests/test_tools.py` covering:
  - rejecting `127.0.0.1`, `10.0.0.5`, `::ffff:127.0.0.1`, DNS resolving to `192.168.1.10`, and `169.254.169.254` / `metadata.google.internal`,
  - rejecting HTTP 302 redirects from a public URL to `http://127.0.0.1/secret` (or `169.254.169.254`) in `WebFetchTool.execute()`,
  - `allow_private_urls=True` permitting RFC1918/localhost while still blocking cloud metadata (`169.254.169.254`).
- [ ] **Step 2:** Run pytest on those new tests and confirm FAIL.
- [ ] **Step 3 (GREEN):** Implement `ivy/tools/url_safety.py`, update `WebFetchTool` in `ivy/tools/web.py` to validate initial URL + manual redirect loop (`follow_redirects=False`, max 5 hops), and pass `allow_private_urls=config.allow_private_urls` in `ivy/tui/app.py`.
- [ ] **Step 4:** Run `& ".\.venv\Scripts\pytest.exe" -p no:cacheprovider tests/test_tools.py -v` and confirm all pass.
- [ ] **Step 5:** Commit Task 2.

---

### Task 3: Static Skill Security Guard

**Files:**
- Create: `ivy/skills/guard.py`
- Modify: `ivy/tools/skill_manager.py`, `ivy/skills/generator.py`
- Test: `tests/test_skills.py`, `tests/test_tools.py`

- [ ] **Step 1 (RED):** Add tests in `tests/test_skills.py` and `tests/test_tools.py` covering:
  - benign skill passes `scan_skill_content`,
  - blocking skills containing `curl ... | bash`, credential exfiltration (`~/.ssh` + `curl`), leaked `sk-...` key, or hidden zero-width unicode / prompt-injection override,
  - `SkillManagerTool` rejecting unsafe `create`/`update` without writing `SKILL.md` to disk,
  - `SkillGenerator.reflect_and_create()` returning `None` and not saving unsafe synthesized `SKILL.md`.
- [ ] **Step 2:** Run pytest on those tests and confirm FAIL.
- [ ] **Step 3 (GREEN):** Implement `ivy/skills/guard.py` and wire `scan_skill_content` into `SkillManagerTool` (`ivy/tools/skill_manager.py`) and `SkillGenerator` (`ivy/skills/generator.py`).
- [ ] **Step 4:** Run `& ".\.venv\Scripts\pytest.exe" -p no:cacheprovider tests/test_skills.py tests/test_tools.py -v` and confirm all pass.
- [ ] **Step 5:** Commit Task 3.

---

### Task 4: Curated Long-Term Memory & Prompt Injection

**Files:**
- Modify: `ivy/memory/store.py`, `ivy/core/prompt.py`, `ivy/core/agent.py`, `ivy/tui/app.py`, `AGENTS.md`, `README.md`
- Create: `ivy/tools/memory_note.py`
- Test: `tests/test_memory.py`, `tests/test_tools.py`, `tests/test_agent.py`

- [ ] **Step 1 (RED):** Add tests in:
  - `tests/test_memory.py`: `memory_notes` add/list/search/remove, secret redaction, 50-note limit enforcement,
  - `tests/test_tools.py`: `MemoryNoteTool` actions (`add`, `list`, `search`, `remove`, invalid/missing args),
  - `tests/test_agent.py`: `build_system_prompt()` and `IvyAgent.run_turn()` injecting `# Curated Memory` notes from `MemoryStore`.
- [ ] **Step 2:** Run pytest on those tests and confirm FAIL.
- [ ] **Step 3 (GREEN):** Implement `memory_notes` table + methods in `ivy/memory/store.py`, `MemoryNoteTool` in `ivy/tools/memory_note.py`, `# Curated Memory` rendering in `ivy/core/prompt.py`, loading in `IvyAgent.run_turn()`, registration in `build_default_tool_registry()`, and concise updates to `AGENTS.md` / `README.md`.
- [ ] **Step 4:** Run full verification:
  - `uvx ruff check ivy tests`
  - `& ".\.venv\Scripts\pytest.exe" -p no:cacheprovider -v`
- [ ] **Step 5:** Commit Task 4.

---

### Task 5: Fresh Reviewer & Branch Finish

- [ ] Build review package `.superpowers/sdd/2026-11-18-phase6-security-and-memory/review-package.md` (`git diff master..HEAD` + stats + verified test output).
- [ ] Dispatch ONE fresh `research` subagent (`Model="pro"`) to inspect the diff against architecture, edge cases, and regressions.
- [ ] Fix any Critical/Important issues via TDD, re-run full `ruff` + `pytest`, clean up `.superpowers/sdd/...`, and finish via `superpowers:finishing-a-development-branch`.
