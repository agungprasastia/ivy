"""Messaging platform adapters for Ivy Gateway."""

from ivy.gateway.adapters.base import BaseAdapter, IncomingHandler
from ivy.gateway.adapters.cli_adapter import CLIAdapter
from ivy.gateway.adapters.discord_adapter import DiscordAdapter
from ivy.gateway.adapters.telegram_adapter import TelegramAdapter

__all__ = [
    "BaseAdapter",
    "IncomingHandler",
    "CLIAdapter",
    "TelegramAdapter",
    "DiscordAdapter",
]
