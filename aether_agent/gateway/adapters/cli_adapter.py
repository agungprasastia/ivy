"""CLI adapter for local interactive gateway testing."""

from __future__ import annotations

from aether_agent.gateway.adapters.base import BaseAdapter, IncomingHandler


class CLIAdapter(BaseAdapter):
    """Local terminal adapter (always authorized for local user)."""

    platform_name = "cli"

    def __init__(self) -> None:
        super().__init__(allowed_users=["local"])

    def is_authorized(self, user_id: str) -> bool:
        return True

    async def start(self, handler: IncomingHandler) -> None:
        self._handler = handler

    async def stop(self) -> None:
        return None

    async def send_message(self, target_id: str, text: str) -> None:
        print(f"[{target_id}] {text}")
