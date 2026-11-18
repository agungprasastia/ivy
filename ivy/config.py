"""Configuration management for Ivy."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, field_validator


class IvyConfig(BaseModel):
    """Central configuration for Ivy."""

    model_config = ConfigDict(extra="ignore")

    app_name: str = "ivy"
    default_model: str = "gpt-4o-mini"
    db_path: str = "./ivy.db"
    skills_dir: str = "skills"
    allowed_users: list[str] = Field(default_factory=list)

    # Provider API keys & endpoint overrides
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    openrouter_api_key: str | None = None
    gemini_api_key: str | None = None
    api_base: str | None = None
    litellm_api_base: str | None = None

    # Gateway tokens
    telegram_token: str | None = None
    discord_token: str | None = None

    # Safety & execution
    auto_approve_safety: bool = False
    verbose: bool = False
    log_level: str = "INFO"
    default_theme: str = "ivy"
    max_iterations: int = 50
    subagent_max_iterations: int = 8
    code_runner_timeout: float = 30.0
    llm_max_retries: int = 2
    llm_retry_base_delay: float = 0.5
    fallback_model: str | None = None
    auto_compact_threshold: float = 0.85

    @field_validator("llm_max_retries", mode="before")
    @classmethod
    def clamp_retries(cls, v: Any) -> int:
        return max(0, int(v))

    @field_validator("llm_retry_base_delay", mode="before")
    @classmethod
    def clamp_retry_delay(cls, v: Any) -> float:
        return max(0.0, float(v))

    @field_validator("fallback_model", mode="before")
    @classmethod
    def normalize_fallback_model(cls, v: Any) -> str | None:
        if v is None:
            return None
        s = str(v).strip()
        return s or None

    @field_validator("auto_compact_threshold", mode="before")
    @classmethod
    def clamp_compact_threshold(cls, v: Any) -> float:
        return min(0.99, max(0.1, float(v)))

    @field_validator("allowed_users", mode="before")
    @classmethod
    def parse_allowed_users(cls, v: Any) -> list[str]:
        if isinstance(v, str):
            return [u.strip() for u in v.split(",") if u.strip()]
        if isinstance(v, (list, tuple, set)):
            return [str(u).strip() for u in v if str(u).strip()]
        return []

    @field_validator("auto_approve_safety", "verbose", mode="before")
    @classmethod
    def parse_bool(cls, v: Any) -> bool:
        if isinstance(v, str):
            return v.strip().lower() in ("true", "1", "yes", "on")
        return bool(v)

    @property
    def effective_api_base(self) -> str | None:
        """Return configured custom API base URL if any."""
        return self.api_base or self.litellm_api_base


def load_config(env_file: str | Path | None = None) -> IvyConfig:
    """Load configuration from environment variables and optional .env file."""
    if env_file is not None:
        load_dotenv(dotenv_path=env_file, override=False)
    else:
        load_dotenv(override=False)

    data: dict[str, Any] = {}

    env_map = {
        "app_name": ["IVY_APP_NAME", "APP_NAME"],
        "default_model": ["DEFAULT_MODEL", "IVY_DEFAULT_MODEL"],
        "db_path": ["IVY_DB_PATH", "DB_PATH"],
        "skills_dir": ["SKILLS_DIR", "IVY_SKILLS_DIR"],
        "allowed_users": ["ALLOWED_USERS", "IVY_ALLOWED_USERS"],
        "openai_api_key": ["OPENAI_API_KEY"],
        "anthropic_api_key": ["ANTHROPIC_API_KEY"],
        "openrouter_api_key": ["OPENROUTER_API_KEY"],
        "gemini_api_key": ["GEMINI_API_KEY"],
        "api_base": ["IVY_API_BASE", "OPENAI_API_BASE", "LITELLM_API_BASE"],
        "litellm_api_base": ["LITELLM_API_BASE"],
        "telegram_token": ["TELEGRAM_TOKEN"],
        "discord_token": ["DISCORD_TOKEN"],
        "auto_approve_safety": ["AUTO_APPROVE_SAFETY"],
        "verbose": ["VERBOSE"],
        "log_level": ["LOG_LEVEL"],
        "default_theme": ["IVY_THEME"],
        "max_iterations": ["MAX_ITERATIONS", "IVY_MAX_ITERATIONS"],
        "llm_max_retries": ["IVY_LLM_MAX_RETRIES"],
        "llm_retry_base_delay": ["IVY_LLM_RETRY_BASE_DELAY"],
        "fallback_model": ["IVY_FALLBACK_MODEL"],
        "auto_compact_threshold": ["IVY_AUTO_COMPACT_THRESHOLD"],
    }

    for field, env_vars in env_map.items():
        for var in env_vars:
            val = os.getenv(var)
            if val is not None:
                data[field] = val
                break

    return IvyConfig(**data)


def update_env_file(
    updates: dict[str, str], env_path: str | Path = ".env"
) -> Path:
    """Persist key-value updates to an env file and refresh os.environ."""
    target = Path(env_path)
    lines: list[str] = []
    if target.exists():
        lines = target.read_text(encoding="utf-8").splitlines()

    remaining = dict(updates)
    new_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in line:
            new_lines.append(line)
            continue
        key, _ = line.split("=", 1)
        clean_key = key.strip()
        if clean_key in remaining:
            new_lines.append(f"{clean_key}={remaining.pop(clean_key)}")
        else:
            new_lines.append(line)

    for key, val in remaining.items():
        new_lines.append(f"{key}={val}")

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

    for key, val in updates.items():
        os.environ[key] = val

    return target
