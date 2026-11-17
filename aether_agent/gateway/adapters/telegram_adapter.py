"""Telegram bot adapter powered by aiogram with whitelist and inline approval."""

from __future__ import annotations

import asyncio
import logging
from uuid import uuid4

from aether_agent.gateway.adapters.base import BaseAdapter, IncomingHandler

logger = logging.getLogger(__name__)


class TelegramAdapter(BaseAdapter):
    """Telegram messaging adapter with ALLOWED_USERS guard and inline approval buttons."""

    platform_name = "telegram"

    def __init__(
        self,
        token: str,
        allowed_users: list[str] | None = None,
    ) -> None:
        super().__init__(allowed_users=allowed_users)
        self.token = token
        self._bot = None
        self._dp = None
        self._polling_task: asyncio.Task[None] | None = None
        self._pending_approvals: dict[str, asyncio.Future[bool]] = {}

    async def start(self, handler: IncomingHandler) -> None:
        try:
            from aiogram import Bot, Dispatcher, F
            from aiogram.types import CallbackQuery, Message
        except ImportError as exc:
            raise RuntimeError(
                "aiogram is required for TelegramAdapter. Install with `uv sync --extra gateway`."
            ) from exc

        self._bot = Bot(token=self.token)
        self._dp = Dispatcher()

        @self._dp.message()
        async def on_message(message: Message) -> None:
            user = message.from_user
            if user is None:
                return
            uid = str(user.id)
            if not self.is_authorized(uid):
                return  # Silent drop for unauthorized users
            text = (message.text or "").strip()
            if not text:
                return
            reply = await handler("telegram", uid, text)
            if reply:
                await message.answer(reply[:4000])

        @self._dp.callback_query(F.data.startswith("approve:") | F.data.startswith("reject:"))
        async def on_approval_callback(callback: CallbackQuery) -> None:
            uid = str(callback.from_user.id)
            if not self.is_authorized(uid):
                await callback.answer("Unauthorized", show_alert=True)
                return
            action, req_id = (callback.data or "").split(":", 1)
            fut = self._pending_approvals.pop(req_id, None)
            if fut and not fut.done():
                fut.set_result(action == "approve")
            await callback.answer(f"Decision recorded: {action}")

        self._polling_task = asyncio.create_task(self._dp.start_polling(self._bot))

    async def stop(self) -> None:
        if self._dp is not None:
            await self._dp.stop_polling()
        if self._polling_task is not None:
            self._polling_task.cancel()
            try:
                await self._polling_task
            except asyncio.CancelledError:
                pass
        if self._bot is not None:
            await self._bot.session.close()

    async def send_message(self, target_id: str, text: str) -> None:
        if self._bot is None:
            logger.warning("Telegram bot is not started; cannot send message.")
            return
        await self._bot.send_message(chat_id=int(target_id), text=text[:4000])

    async def request_approval(
        self, target_id: str, command: str, timeout: float = 60.0
    ) -> bool:
        if self._bot is None:
            return False
        from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

        req_id = uuid4().hex[:10]
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[bool] = loop.create_future()
        self._pending_approvals[req_id] = fut

        keyboard = InlineKeyboardMarkup(
            inline_keyboard=[
                [
                    InlineKeyboardButton(text="Approve", callback_data=f"approve:{req_id}"),
                    InlineKeyboardButton(text="Reject", callback_data=f"reject:{req_id}"),
                ]
            ]
        )
        await self._bot.send_message(
            chat_id=int(target_id),
            text=f"[SAFETY GUARD] Approve dangerous command?\n`{command}`",
            reply_markup=keyboard,
        )
        try:
            return await asyncio.wait_for(fut, timeout=timeout)
        except asyncio.TimeoutError:
            self._pending_approvals.pop(req_id, None)
            return False
