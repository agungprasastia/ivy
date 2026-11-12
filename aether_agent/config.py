"""Configuration management for Aether Agent."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, field_validator


class AetherConfig(BaseModel):
    """Central configuration for Aether Agent."""

    model_config = ConfigDict(extra="ignore")

    app_name: str = "aether-agent"
    default_model: str = "gpt-4o-mini"
    db_path: str = "./aether.db"
    skills_dir: str = "skills"
    allowed_users: list[str] = Field(default_factory=list)

    # Provider API keys
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    openrouter_api_key: str | None = None
    gemini_api_key: str | None = None
    litellm_api_base: str | None = None

    # Gateway tokens
    telegram_token: str | None = None
    discord_token: str | None = None

    # Safety & execution
    auto_approve_safety: bool = False
    verbose: bool = False
    log_level: str = "INFO"

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


def load_config(env_file: str | Path | None = None) -> AetherConfig:
    """Load configuration from environment variables and optional .env file."""
    if env_file is not None:
        load_dotenv(dotenv_path=env_file, override=False)
    else:
        load_dotenv(override=False)

    data: dict[str, Any] = {}

    env_map = {
        "app_name": ["AETHER_APP_NAME", "APP_NAME"],
        "default_model": ["DEFAULT_MODEL", "AETHER_DEFAULT_MODEL"],
        "db_path": ["AETHER_DB_PATH", "DB_PATH"],
        "skills_dir": ["SKILLS_DIR", "AETHER_SKILLS_DIR"],
        "allowed_users": ["ALLOWED_USERS", "AETHER_ALLOWED_USERS"],
        "openai_api_key": ["OPENAI_API_KEY"],
        "anthropic_api_key": ["ANTHROPIC_API_KEY"],
        "openrouter_api_key": ["OPENROUTER_API_KEY"],
        "gemini_api_key": ["GEMINI_API_KEY"],
        "litellm_api_base": ["LITELLM_API_BASE"],
        "telegram_token": ["TELEGRAM_TOKEN"],
        "discord_token": ["DISCORD_TOKEN"],
        "auto_approve_safety": ["AUTO_APPROVE_SAFETY"],
        "verbose": ["VERBOSE"],
        "log_level": ["LOG_LEVEL"],
    }

    for field, env_vars in env_map.items():
        for var in env_vars:
            val = os.getenv(var)
            if val is not None:
                data[field] = val
                break

    return AetherConfig(**data)
