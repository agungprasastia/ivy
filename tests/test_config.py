from ivy.config import IvyConfig, load_config


def test_load_default_config():
    config = load_config()
    assert config.app_name == "ivy"
    assert config.default_model is not None
    assert config.db_path.endswith("ivy.db")


def test_ivy_config_direct_instantiation():
    cfg = IvyConfig(default_model="gpt-4o-mini")
    assert cfg.default_model == "gpt-4o-mini"
    assert cfg.app_name == "ivy"
    assert cfg.db_path.endswith("ivy.db")


def test_load_config_from_env(monkeypatch):
    monkeypatch.setenv("IVY_APP_NAME", "custom-ivy")
    monkeypatch.setenv("DEFAULT_MODEL", "gpt-4o")
    monkeypatch.setenv("IVY_DB_PATH", "custom_path/ivy.db")
    monkeypatch.setenv("ALLOWED_USERS", "user1,user2,user3")
    monkeypatch.setenv("TELEGRAM_TOKEN", "fake_tg_token")
    monkeypatch.setenv("DISCORD_TOKEN", "fake_dc_token")

    config = load_config()
    assert config.app_name == "custom-ivy"
    assert config.default_model == "gpt-4o"
    assert config.db_path == "custom_path/ivy.db"
    assert config.allowed_users == ["user1", "user2", "user3"]
    assert config.telegram_token == "fake_tg_token"
    assert config.discord_token == "fake_dc_token"


def test_isolated_subpackage_imports():
    import subprocess
    import sys

    for pkg in [
        "ivy.tools",
        "ivy.memory",
        "ivy.cron",
        "ivy.gateway",
        "ivy.skills",
        "ivy.core",
        "ivy.tui",
    ]:
        proc = subprocess.run(
            [sys.executable, "-c", f"import {pkg}"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, f"Failed importing {pkg}: {proc.stderr}"


def test_update_env_file_persists_keys_and_theme(tmp_path, monkeypatch):
    from ivy.config import update_env_file

    env_file = tmp_path / ".env"
    env_file.write_text("# Ivy config\nOPENAI_API_KEY=\nDEFAULT_MODEL=gpt-4o-mini\n", encoding="utf-8")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("IVY_API_BASE", raising=False)
    monkeypatch.delenv("IVY_THEME", raising=False)

    update_env_file(
        {
            "OPENAI_API_KEY": "sk-9router-123",
            "IVY_API_BASE": "http://localhost:20128/v1",
            "DEFAULT_MODEL": "openai/claude-sonnet-4-5",
            "IVY_THEME": "hermes",
        },
        env_path=env_file,
    )

    loaded = load_config(env_file=env_file)
    assert loaded.openai_api_key == "sk-9router-123"
    assert loaded.effective_api_base == "http://localhost:20128/v1"
    assert loaded.default_model == "openai/claude-sonnet-4-5"
    assert loaded.default_theme == "hermes"


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
