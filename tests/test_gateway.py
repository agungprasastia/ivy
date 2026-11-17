from unittest.mock import AsyncMock, MagicMock

import pytest
from aether_agent.config import AetherConfig
from aether_agent.core.types import ChatMessage, Role
from aether_agent.gateway.adapters.discord_adapter import DiscordAdapter
from aether_agent.gateway.adapters.telegram_adapter import TelegramAdapter
from aether_agent.gateway.server import GatewayServer
from aether_agent.gateway.session_hub import SessionHub
from aether_agent.gateway.turn_lease import TurnLeaseManager


@pytest.mark.asyncio
async def test_turn_lease_prevents_concurrent_turns():
    lease_mgr = TurnLeaseManager()
    session_id = "session_x"

    lease1 = await lease_mgr.acquire(session_id)
    assert lease1 is True

    lease2 = await lease_mgr.acquire(session_id, timeout=0.1)
    assert lease2 is False

    await lease_mgr.release(session_id)
    lease3 = await lease_mgr.acquire(session_id)
    assert lease3 is True
    await lease_mgr.release(session_id)


def test_session_hub_cross_platform_routing():
    hub = SessionHub()
    s1 = hub.resolve_session(platform="cli", user_id="owner", explicit_session="shared-main")
    s2 = hub.resolve_session(platform="telegram", user_id="12345")
    assert s1 == "shared-main"
    assert s2 == "telegram:12345"

    hub.bind_user_session("telegram", "12345", "shared-main")
    assert hub.resolve_session(platform="telegram", user_id="12345") == "shared-main"


@pytest.mark.asyncio
async def test_gateway_server_whitelist_enforcement_and_routing():
    cfg = AetherConfig(allowed_users=["1001"])
    mock_agent = MagicMock()
    mock_agent.run_turn = AsyncMock(
        return_value=ChatMessage(role=Role.ASSISTANT, content="Approved reply")
    )

    server = GatewayServer(agent=mock_agent, config=cfg)

    # Unauthorized user must be dropped/rejected without invoking the agent
    denied = await server.handle_incoming(
        platform="telegram", user_id="9999", text="rm -rf /"
    )
    assert denied is None
    assert not mock_agent.run_turn.called

    # Authorized user proceeds through turn lease and agent
    reply = await server.handle_incoming(
        platform="telegram", user_id="1001", text="Halo"
    )
    assert reply == "Approved reply"
    assert mock_agent.run_turn.called


def test_bot_adapters_whitelist_check():
    tg = TelegramAdapter(token="fake", allowed_users=["123"])
    assert tg.is_authorized("123") is True
    assert tg.is_authorized("456") is False

    dc = DiscordAdapter(token="fake", allowed_users=["789"])
    assert dc.is_authorized("789") is True
    assert dc.is_authorized("000") is False

    # Empty whitelist must fail-closed for remote bot adapters
    tg_empty = TelegramAdapter(token="fake", allowed_users=[])
    assert tg_empty.is_authorized("123") is False
