"""Central asynchronous messaging gateway daemon."""

from __future__ import annotations

import logging
from typing import Any

from ivy.config import IvyConfig
from ivy.cron.scheduler import CronScheduler
from ivy.gateway.adapters.base import BaseAdapter
from ivy.gateway.adapters.discord_adapter import DiscordAdapter
from ivy.gateway.adapters.telegram_adapter import TelegramAdapter
from ivy.gateway.session_hub import SessionHub
from ivy.gateway.turn_lease import TurnLeaseManager
from ivy.tools.approval import SafetyApprovalPolicy

logger = logging.getLogger(__name__)


class GatewayServer:
    """Routes multi-platform messages through whitelist guards and turn leases."""

    def __init__(
        self,
        agent: Any,
        config: IvyConfig,
        cron_scheduler: CronScheduler | None = None,
        lease_manager: TurnLeaseManager | None = None,
        session_hub: SessionHub | None = None,
        approval_policy: SafetyApprovalPolicy | None = None,
    ) -> None:
        self.agent = agent
        self.config = config
        self.cron_scheduler = cron_scheduler
        self.lease_manager = lease_manager or TurnLeaseManager()
        self.session_hub = session_hub or SessionHub()
        self.approval_policy = approval_policy
        self._adapters: dict[str, BaseAdapter] = {}
        self._allowed_users = {str(u).strip() for u in config.allowed_users if str(u).strip()}

    def register_adapter(self, adapter: BaseAdapter) -> None:
        """Register a messaging platform adapter."""
        self._adapters[adapter.platform_name] = adapter

    def configure_default_adapters(self) -> None:
        """Instantiate Telegram and Discord adapters if tokens are configured."""
        if self.config.telegram_token:
            self.register_adapter(
                TelegramAdapter(
                    token=self.config.telegram_token,
                    allowed_users=self.config.allowed_users,
                )
            )
        if self.config.discord_token:
            self.register_adapter(
                DiscordAdapter(
                    token=self.config.discord_token,
                    allowed_users=self.config.allowed_users,
                )
            )

    def is_user_authorized(self, platform: str, user_id: str) -> bool:
        """Enforce fail-closed whitelist authorization for remote platforms."""
        if platform.lower() == "cli":
            return True
        if not self._allowed_users:
            logger.warning(
                "Dropped %s message from %s: ALLOWED_USERS whitelist is empty.",
                platform,
                user_id,
            )
            return False
        authorized = str(user_id).strip() in self._allowed_users
        if not authorized:
            logger.warning(
                "Rejected unauthorized %s message from user_id=%s",
                platform,
                user_id,
            )
        return authorized

    async def handle_incoming(
        self,
        platform: str,
        user_id: str,
        text: str,
        session_id: str | None = None,
        lease_timeout: float = 5.0,
    ) -> str | None:
        """Authorize, acquire turn lease, invoke agent, and return reply text."""
        if not self.is_user_authorized(platform, user_id):
            return None

        resolved_session = self.session_hub.resolve_session(
            platform=platform,
            user_id=user_id,
            explicit_session=session_id,
        )

        acquired = await self.lease_manager.acquire(
            resolved_session, timeout=lease_timeout
        )
        if not acquired:
            return "Session is busy processing another turn. Please try again shortly."

        adapter = self._adapters.get(platform.lower())
        try:
            if self.approval_policy is not None and adapter is not None:
                async with self.approval_policy.scoped_context(
                    approval_callback=lambda cmd: adapter.request_approval(user_id, cmd),
                    unattended=False,
                ):
                    reply = await self.agent.run_turn(
                        prompt=text,
                        session_id=resolved_session,
                        platform=platform.lower(),
                    )
            else:
                reply = await self.agent.run_turn(
                    prompt=text,
                    session_id=resolved_session,
                    platform=platform.lower(),
                )
            return getattr(reply, "content", "") or ""
        finally:
            await self.lease_manager.release(resolved_session)

    async def dispatch(self, channel: str, target_id: str, text: str) -> None:
        """Deliver an outbound message (e.g., from Cron) to a registered adapter."""
        adapter = self._adapters.get(channel.lower())
        if adapter is None:
            logger.warning("No adapter registered for channel %r", channel)
            return
        await adapter.send_message(target_id, text)

    async def start(self) -> None:
        """Start all registered adapters and optional cron scheduler."""
        for adapter in self._adapters.values():
            await adapter.start(self.handle_incoming)
        if self.cron_scheduler is not None:
            self.cron_scheduler.start()

    async def stop(self) -> None:
        """Stop cron scheduler and all registered adapters cleanly."""
        if self.cron_scheduler is not None:
            await self.cron_scheduler.stop()
        for adapter in self._adapters.values():
            await adapter.stop()
