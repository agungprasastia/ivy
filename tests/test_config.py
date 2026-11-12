import os
from aether_agent.config import AetherConfig, load_config


def test_load_default_config():
    config = load_config()
    assert config.app_name == "aether-agent"
    assert config.default_model is not None
    assert config.db_path.endswith("aether.db")


def test_aether_config_direct_instantiation():
    cfg = AetherConfig(default_model="gpt-4o-mini")
    assert cfg.default_model == "gpt-4o-mini"
    assert cfg.app_name == "aether-agent"
    assert cfg.db_path.endswith("aether.db")


def test_load_config_from_env(monkeypatch):
    monkeypatch.setenv("AETHER_APP_NAME", "custom-aether")
    monkeypatch.setenv("DEFAULT_MODEL", "gpt-4o")
    monkeypatch.setenv("AETHER_DB_PATH", "custom_path/aether.db")
    monkeypatch.setenv("ALLOWED_USERS", "user1,user2,user3")
    monkeypatch.setenv("TELEGRAM_TOKEN", "fake_tg_token")
    monkeypatch.setenv("DISCORD_TOKEN", "fake_dc_token")

    config = load_config()
    assert config.app_name == "custom-aether"
    assert config.default_model == "gpt-4o"
    assert config.db_path == "custom_path/aether.db"
    assert config.allowed_users == ["user1", "user2", "user3"]
    assert config.telegram_token == "fake_tg_token"
    assert config.discord_token == "fake_dc_token"
