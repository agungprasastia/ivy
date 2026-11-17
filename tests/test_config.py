import os
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
