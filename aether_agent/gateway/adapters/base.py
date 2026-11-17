"""Abstract base interface for gateway platform adapters."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Awaitable, Callable

logger = logging.getLogger(__name__)

IncomingHandler = Callable[[str, str, str], Awaitable[str | None]]


class BaseAdapter(ABC):
    """Base contract for CLI, Telegram, and Discord gateway adapters."""

    platform_name: str = "base"

    def __init__(self, allowed_users: list[str] | None = None) -> None:
        self.allowed_users = {str(u).strip() for u in (allowed_users or []) if str(u).strip()}

    def is_authorized(self, user_id: str) -> bool:
        """Fail-closed whitelist check for remote messaging adapters."""
        uid = str(user_id).strip()
        if not self.allowed_users:
            logger.warning(
                "Blocked %s message from user %s: ALLOWED_USERS whitelist is empty.",
                self.platform_name,
                uid,
            )
            return False
        authorized = uid in self.allowed_users
        if not authorized:
            logger.warning(
                "Unauthorized %s access attempt by user_id=%s",
                self.platform_name,
                uid,
            )
        return authorized

    @abstractmethod
    async def start(self, handler: IncomingHandler) -> None:
        """Start receiving messages and forwarding authorized text to handler."""

    @abstractmethod
    async def stop(self) -> None:
        """Stop the adapter cleanly."""

    @abstractmethod
    async def send_message(self, target_id: str, text: str) -> None:
        """Send an outbound text message to target_id."""

    async def request_approval(
        self, target_id: str, command: str, timeout: float = 60.0
    ) -> bool:
        """Request interactive approval for a dangerous command (default deny)."""
        return False
