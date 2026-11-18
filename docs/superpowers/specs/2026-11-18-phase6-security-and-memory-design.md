# Phase 6 Design: Security Guards, Skill Scanner & Curated Memory

**Date:** 2026-11-18  
**Branch:** `feat/phase6-security-memory`  
**Status:** Approved for planning & execution

## Overview

Phase 6 hardens Ivy's runtime boundaries (filesystem paths, SSRF-safe web fetching, static scanning for synthesized/saved skills) and adds bounded curated long-term memory notes (`memory_note`) injected into the system prompt. All additions use Python's standard library + existing 7 runtime dependencies (`pyproject.toml` untouched).

## Architecture

### 1. Path Security & Read-Before-Write (`ivy/tools/path_security.py`, `ivy/tools/files.py`, `ivy/config.py`)
- **`ivy/tools/path_security.py`**:
  - Reject empty paths and ASCII control characters (`ord < 32` or `127`).
  - Block sensitive credential/system targets on read, write, patch, and directory search/listing: `.ssh`, `.aws`, `.gnupg`, `.kube`, `.git-credentials`, `.netrc`, `.npmrc`, `.pypirc`, `/etc/shadow`, `/etc/passwd`, and `.env` / `.env.*` (except `.env.example`, `.env.template`, `.env.sample`).
  - Support optional strict workspace containment (`workspace_root: Path | None`) via `IVY_RESTRICT_WORKSPACE=true`. Standalone `FileTools()` leaves `workspace_root=None` so existing `tmp_path` tests remain backward-compatible.
- **`ivy/tools/files.py`**:
  - Share a `read_tracker: set[Path]` across `ReadFileTool`, `WriteFileTool`, and `PatchFileTool`.
  - When `require_read_before_overwrite=True` (enabled in `build_default_tool_registry()`), overwriting an **existing** file with `write_file` or patching with `patch_file` requires that resolved path to have been read (or newly written) earlier. Standalone `FileTools()` defaults to `require_read_before_overwrite=False`.

### 2. SSRF & Redirect Guard (`ivy/tools/url_safety.py`, `ivy/tools/web.py`, `ivy/config.py`)
- **`ivy/tools/url_safety.py`**:
  - `validate_url_target(url, *, allow_private=False, resolver=None)` parses `http`/`https` URLs, resolves DNS, unwraps IPv4-mapped IPv6 (`::ffff:x.x.x.x`), and blocks cloud metadata endpoints (`169.254.169.254`, `169.254.170.2`, `169.254.169.253`, `100.100.100.200`, `metadata.google.internal`, `metadata.goog`), link-local, multicast, and (unless `IVY_ALLOW_PRIVATE_URLS=true`) loopback/private/reserved IPs and `.local`/`.internal`/`localhost` hosts. Cloud metadata stays blocked even when `allow_private=True`.
- **`ivy/tools/web.py`**:
  - `WebFetchTool` disables `httpx` auto-redirects (`follow_redirects=False`), validates the initial URL, and re-validates each `Location` hop up to `max_redirects=5`.

### 3. Static Skill Security Guard (`ivy/skills/guard.py`, `ivy/tools/skill_manager.py`, `ivy/skills/generator.py`)
- **`ivy/skills/guard.py`**:
  - Zero-dep static pattern scanner checking skill name, description, and instructions for hidden/bidi unicode control characters, prompt-injection overrides, credential/secret exfiltration combined with network/shell commands, leaked API keys/private keys, remote `curl|bash` pipes, reverse shells, and destructive root commands.
- **Enforcement**:
  - `SkillManagerTool` blocks `create`/`update` when `danger` findings are detected.
  - `SkillGenerator.reflect_and_create` skips saving auto-synthesized skills that fail the scan.

### 4. Curated Long-Term Memory (`ivy/memory/store.py`, `ivy/tools/memory_note.py`, `ivy/core/prompt.py`, `ivy/core/agent.py`, `ivy/tui/app.py`)
- **`ivy/memory/store.py`**:
  - SQLite `memory_notes` table (`id`, `note` UNIQUE, `category`, `updated_at`) with CRUD helpers (`add_memory_note`, `list_memory_notes`, `search_memory_notes`, `remove_memory_note`), capped at 50 notes and 300 chars per note, scrubbed via `redact_secrets`.
- **`ivy/tools/memory_note.py`**:
  - `memory_note` tool (`add`, `list`, `search`, `remove`) registered alongside `remember_preference` and `session_search` in `build_default_tool_registry()`.
- **System prompt injection**:
  - `IvyAgent.run_turn()` fetches curated notes (`limit=50`) and `build_system_prompt()` renders a bounded `# Curated Memory` section ordered deterministically by `id ASC`.
