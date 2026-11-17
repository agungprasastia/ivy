"""Asynchronous background cron scheduler."""

from __future__ import annotations

import asyncio
import inspect
import logging
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

from ivy.cron.task import ScheduledTask
from ivy.memory.store import MemoryStore
from ivy.tools.approval import SafetyApprovalPolicy

logger = logging.getLogger(__name__)


class CronScheduler:
    """Evaluates cron schedules and runs agent/handler tasks asynchronously."""

    def __init__(
        self,
        agent: Any | None = None,
        memory_store: MemoryStore | None = None,
        dispatcher: Callable[[str, str, str], Any | Awaitable[Any]] | None = None,
        approval_policy: SafetyApprovalPolicy | None = None,
        poll_interval: float = 15.0,
    ) -> None:
        self.agent = agent
        self.memory_store = memory_store
        self.dispatcher = dispatcher
        self.approval_policy = approval_policy
        self.poll_interval = poll_interval
        self._tasks: dict[str, ScheduledTask] = {}
        self._runner_task: asyncio.Task[None] | None = None
        self._stop_event = asyncio.Event()

    def add_task(self, task: ScheduledTask) -> None:
        """Register or replace a scheduled task."""
        self._tasks[task.name] = task

    def add_job(self, task: ScheduledTask) -> None:
        """Alias for add_task."""
        self.add_task(task)

    def remove_task(self, name: str) -> bool:
        """Remove a scheduled task by name."""
        return self._tasks.pop(name, None) is not None

    def remove_job(self, name: str) -> bool:
        """Alias for remove_task."""
        return self.remove_task(name)

    def list_tasks(self) -> list[ScheduledTask]:
        """Return all registered scheduled tasks."""
        return list(self._tasks.values())

    async def _execute_task(self, task: ScheduledTask) -> None:
        status = "success"
        output_text = ""
        try:
            if task.handler is not None:
                res = task.handler()
                if inspect.isawaitable(res):
                    res = await res
                output_text = "" if res is None else str(res)
            elif task.prompt and self.agent is not None:
                if self.approval_policy is not None:
                    async with self.approval_policy.scoped_context(unattended=True):
                        reply = await self.agent.run_turn(
                            prompt=task.prompt,
                            session_id=f"cron:{task.name}",
                            platform="cron",
                        )
                else:
                    reply = await self.agent.run_turn(
                        prompt=task.prompt,
                        session_id=f"cron:{task.name}",
                        platform="cron",
                    )
                output_text = getattr(reply, "content", "") or ""
            else:
                status = "skipped"
                output_text = "No handler or agent prompt configured."

            if (
                self.dispatcher is not None
                and task.target_channel
                and task.target_id
                and output_text
            ):
                disp_res = self.dispatcher(
                    task.target_channel, task.target_id, output_text
                )
                if inspect.isawaitable(disp_res):
                    await disp_res
        except Exception as exc:
            status = "error"
            output_text = str(exc)
            logger.exception("Cron task %s failed: %s", task.name, exc)
        finally:
            if self.memory_store is not None:
                try:
                    await self.memory_store.log_cron_run(
                        task.name, status, output_text
                    )
                except Exception:
                    logger.exception("Failed to log cron run for %s", task.name)

    async def run_pending(self, now: datetime | None = None) -> list[str]:
        """Execute all tasks whose next_run <= now and advance their schedule."""
        current = now or datetime.now(timezone.utc)
        if current.tzinfo is None:
            current = current.replace(tzinfo=timezone.utc)

        due_names: list[str] = []
        coros = []
        for task in list(self._tasks.values()):
            if not task.enabled or task.next_run is None:
                continue
            if task.next_run <= current:
                due_names.append(task.name)
                task.next_run = task.compute_next_run(current)
                coros.append(self._execute_task(task))

        if coros:
            await asyncio.gather(*coros)
        return due_names

    async def _loop(self) -> None:
        while not self._stop_event.is_set():
            await self.run_pending()
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(), timeout=self.poll_interval
                )
            except asyncio.TimeoutError:
                continue

    def start(self) -> None:
        """Start the background scheduler loop on the active event loop."""
        if self._runner_task is not None and not self._runner_task.done():
            return
        self._stop_event.clear()
        self._runner_task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        """Gracefully stop the background scheduler loop."""
        self._stop_event.set()
        if self._runner_task is not None:
            try:
                await self._runner_task
            except asyncio.CancelledError:
                pass
            self._runner_task = None
