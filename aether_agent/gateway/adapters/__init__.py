"""Messaging platform adapters for Aether Gateway."""

from aether_agent.gateway.adapters.base import BaseAdapter, IncomingHandler
from aether_agent.gateway.adapters.cli_adapter import CLIAdapter
from aether_agent.gateway.adapters.discord_adapter import DiscordAdapter
from aether_agent.gateway.adapters.telegram_adapter import TelegramAdapter

__all__ = [
    "BaseAdapter",
    "IncomingHandler",
    "CLIAdapter",
    "TelegramAdapter",
    "DiscordAdapter",
]
