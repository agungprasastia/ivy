"""Environment, endpoint connectivity, database, and workspace health diagnostics (`ivy doctor`)."""

from __future__ import annotations

import platform
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

try:
    import httpx
except ImportError:  # openai>=3 / anthropic>=2 bundle httpx2 on Python 3.15
    import httpx2 as httpx  # type: ignore[no-redef]

from rich import box
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from ivy.tui.theme import ThemeConfig, get_theme

if TYPE_CHECKING:
    from ivy.config import IvyConfig
    from ivy.core.llm import LLMClient
    from ivy.memory.store import MemoryStore
    from ivy.skills.engine import SkillEngine


@dataclass(frozen=True)
class DoctorFinding:
    """Single diagnostic check result."""

    category: str
    name: str
    status: Literal["ok", "warn", "error"]
    detail: str


async def run_doctor_checks(
    config: IvyConfig,
    llm_client: LLMClient,
    memory_store: MemoryStore,
    skill_engine: SkillEngine,
    live_probe: bool = True,
    fix: bool = False,
    env_path: str | Path = ".env",
) -> list[DoctorFinding]:
    """Run ordered environment, credential, database, skill, and endpoint checks."""
    findings: list[DoctorFinding] = []

    # 1. Python runtime
    findings.append(
        DoctorFinding(
            category="Runtime",
            name="Python",
            status="ok",
            detail=f"Python {platform.python_version()} ({platform.system()})",
        )
    )

    # 2. .env configuration
    env_file = Path(env_path)
    if env_file.is_file():
        findings.append(
            DoctorFinding(
                category="Config",
                name=".env file",
                status="ok",
                detail=f"Found {env_file}",
            )
        )
    elif fix:
        env_file.parent.mkdir(parents=True, exist_ok=True)
        env_file.write_text(
            f"# Ivy configuration\nDEFAULT_MODEL={config.default_model}\n",
            encoding="utf-8",
        )
        findings.append(
            DoctorFinding(
                category="Config",
                name=".env file",
                status="ok",
                detail=f"Created starter {env_file} (--fix)",
            )
        )
    else:
        findings.append(
            DoctorFinding(
                category="Config",
                name=".env file",
                status="warn",
                detail=f"{env_file} not found (run `ivy doctor --fix` or `/setup`)",
            )
        )

    # 3. Provider & API credentials
    provider = getattr(llm_client, "provider", "openai")
    api_key = llm_client._resolve_api_key(provider)
    base_url = llm_client.base_url or (
        "https://api.openai.com/v1" if provider == "openai" else None
    )
    if provider == "ollama":
        findings.append(
            DoctorFinding(
                category="LLM",
                name="Credentials",
                status="ok",
                detail=f"model={llm_client.model}, local provider={provider}",
            )
        )
    elif api_key and api_key != "missing-api-key":
        findings.append(
            DoctorFinding(
                category="LLM",
                name="Credentials",
                status="ok",
                detail=f"model={llm_client.model}, provider={provider}, key=configured",
            )
        )
    else:
        findings.append(
            DoctorFinding(
                category="LLM",
                name="Credentials",
                status="warn",
                detail=f"No API key resolved for provider '{provider}' (run `/setup`)",
            )
        )

    # 4. Optional live endpoint probe (/models)
    if live_probe and base_url and provider != "anthropic":
        probe_url = f"{base_url.rstrip('/')}/models"
        headers: dict[str, str] = {}
        if api_key and api_key != "missing-api-key":
            headers["Authorization"] = f"Bearer {api_key}"
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(probe_url, headers=headers)
            if resp.status_code < 400:
                model_count = ""
                try:
                    data = resp.json().get("data")
                    if isinstance(data, list):
                        model_count = f" ({len(data)} models)"
                except Exception:
                    pass
                findings.append(
                    DoctorFinding(
                        category="LLM",
                        name="Live Endpoint",
                        status="ok",
                        detail=f"GET {probe_url} -> HTTP {resp.status_code}{model_count}",
                    )
                )
            elif resp.status_code in (401, 403):
                findings.append(
                    DoctorFinding(
                        category="LLM",
                        name="Live Endpoint",
                        status="error",
                        detail=f"GET {probe_url} -> HTTP {resp.status_code} (auth rejected)",
                    )
                )
            else:
                findings.append(
                    DoctorFinding(
                        category="LLM",
                        name="Live Endpoint",
                        status="warn",
                        detail=f"GET {probe_url} -> HTTP {resp.status_code}",
                    )
                )
        except Exception as exc:
            findings.append(
                DoctorFinding(
                    category="LLM",
                    name="Live Endpoint",
                    status="warn",
                    detail=f"Unreachable ({probe_url}): {exc}",
                )
            )

    # 5. SQLite state & FTS5
    try:
        conn = memory_store._ensure_conn()
        quick = conn.execute("PRAGMA quick_check").fetchone()
        sessions = await memory_store.list_sessions(limit=100)
        ok_integrity = quick and str(quick[0]).lower() == "ok"
        findings.append(
            DoctorFinding(
                category="Storage",
                name="SQLite & FTS5",
                status="ok" if ok_integrity else "error",
                detail=f"db={memory_store.db_path} ({len(sessions)} sessions, integrity={quick[0] if quick else 'unknown'})",
            )
        )
    except Exception as exc:
        findings.append(
            DoctorFinding(
                category="Storage",
                name="SQLite & FTS5",
                status="error",
                detail=f"SQLite error: {exc}",
            )
        )

    # 6. Skills directory
    skills_path = Path(skill_engine.skills_dir)
    if skills_path.is_dir():
        loaded = len(skill_engine.list_skills())
        findings.append(
            DoctorFinding(
                category="Skills",
                name="Skills Directory",
                status="ok",
                detail=f"{skills_path} ({loaded} skills loaded)",
            )
        )
    elif fix:
        skills_path.mkdir(parents=True, exist_ok=True)
        findings.append(
            DoctorFinding(
                category="Skills",
                name="Skills Directory",
                status="ok",
                detail=f"Created missing directory {skills_path} (--fix)",
            )
        )
    else:
        findings.append(
            DoctorFinding(
                category="Skills",
                name="Skills Directory",
                status="warn",
                detail=f"Missing {skills_path} (run `ivy doctor --fix` to create)",
            )
        )

    # 7. External tools
    git_path = shutil.which("git")
    findings.append(
        DoctorFinding(
            category="Tools",
            name="git CLI",
            status="ok" if git_path else "warn",
            detail=git_path or "git not found on PATH (@diff and /diff unavailable)",
        )
    )

    return findings


def render_doctor_report(
    console: Console,
    findings: list[DoctorFinding],
    theme: ThemeConfig | None = None,
) -> int:
    """Render findings table cleanly and return exit code (0 when error-free, else 1)."""
    t = theme or get_theme()
    table = Table(
        title=f"[{t.banner_title}]{t.glyph} Ivy Doctor Diagnostics[/]",
        box=box.ROUNDED,
        border_style=t.banner_border,
    )
    table.add_column("Status", width=6, justify="center")
    table.add_column("Area", style="dim")
    table.add_column("Check", style=f"bold {t.ui_label}")
    table.add_column("Details")

    counts = {"ok": 0, "warn": 0, "error": 0}
    icons = {
        "ok": f"[{t.ui_ok}]✓ OK[/]",
        "warn": f"[{t.ui_warn}]⚠ WARN[/]",
        "error": f"[{t.ui_error}]✗ ERR[/]",
    }
    for f in findings:
        counts[f.status] = counts.get(f.status, 0) + 1
        table.add_row(icons.get(f.status, f.status), f.category, f.name, escape(f.detail))

    console.print(table)
    console.print(
        f"[dim]Summary: {counts['ok']} ok · {counts['warn']} warnings · {counts['error']} errors[/dim]"
    )
    return 1 if counts["error"] > 0 else 0
