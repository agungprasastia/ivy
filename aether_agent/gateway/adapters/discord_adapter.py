"""Discord bot adapter powered by discord.py with whitelist enforcement."""

from __future__ import annotations

import asyncio
import logging

from aether_agent.gateway.adapters.base import BaseAdapter, IncomingHandler

logger = logging.getLogger(__name__)


class DiscordAdapter(BaseAdapter):
    """Discord messaging adapter with ALLOWED_USERS guard and button approvals."""

    platform_name = "discord"

    def __init__(
        self,
        token: str,
        allowed_users: list[str] | None = None,
    ) -> None:
        super().__init__(allowed_users=allowed_users)
        self.token = token
        self._client = None
        self._runner_task: asyncio.Task[None] | None = None

    async def start(self, handler: IncomingHandler) -> None:
        try:
            import discord
        except ImportError as exc:
            raise RuntimeError(
                "discord.py is required for DiscordAdapter. Install with `uv sync --extra gateway`."
            ) from exc

        intents = discord.Intents.default()
        intents.message_content = True
        client = discord.Client(intents=intents)

        @client.event
        async def on_message(message: discord.Message) -> None:
            if client.user and message.author.id == client.user.id:
                return
            uid = str(message.author.id)
            if not self.is_authorized(uid):
                return  # Silent drop for unauthorized users
            content = (message.content or "").strip()
            if not content:
                return
            reply = await handler("discord", uid, content)
            if reply:
                await message.channel.send(reply[:1900])

        self._client = client
        self._runner_task = asyncio.create_task(client.start(self.token))

    async def stop(self) -> None:
        if self._client is not None:
            await self._client.close()
        if self._runner_task is not None:
            self._runner_task.cancel()
            try:
                await self._runner_task
            except asyncio.CancelledError:
                pass

    async def send_message(self, target_id: str, text: str) -> None:
        if self._client is None:
            logger.warning("Discord client is not started; cannot send message.")
            return
        user = await self._client.fetch_user(int(target_id))
        await user.send(text[:1900])

    async def request_approval(
        self, target_id: str, command: str, timeout: float = 60.0
    ) -> bool:
        if self._client is None:
            return False
        import discord

        loop = asyncio.get_running_loop()
        fut: asyncio.Future[bool] = loop.create_future()
        adapter = self

        class ApprovalView(discord.ui.View):
            def __init__(self) -> None:
                super().__init__(timeout=timeout)

            @discord.ui.button(label="Approve", style=discord.ButtonStyle.danger)
            async def approve_btn(
                self, interaction: discord.Interaction, button: discord.ui.Button
            ) -> None:
                if not adapter.is_authorized(str(interaction.user.id)):
                    await interaction.response.send_message("Unauthorized", ephemeral=True)
                    return
                if not fut.done():
                    fut.set_result(True)
                await interaction.response.send_message("Approved", ephemeral=True)
                self.stop()

            @discord.ui.button(label="Reject", style=discord.ButtonStyle.secondary)
            async def reject_btn(
                self, interaction: discord.Interaction, button: discord.ui.Button
            ) -> None:
                if not adapter.is_authorized(str(interaction.user.id)):
                    await interaction.response.send_message("Unauthorized", ephemeral=True)
                    return
                if not fut.done():
                    fut.set_result(False)
                await interaction.response.send_message("Rejected", ephemeral=True)
                self.stop()

        user = await self._client.fetch_user(int(target_id))
        view = ApprovalView()
        await user.send(
            f"[SAFETY GUARD] Approve dangerous command?\n`{command}`", view=view
        )
        try:
            return await asyncio.wait_for(fut, timeout=timeout)
        except asyncio.TimeoutError:
            return False
