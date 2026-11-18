"""Agent-callable cron management tool synced with CronScheduler and MemoryStore."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ivy.core.types import ToolResult
from ivy.cron.task import ScheduledTask
from ivy.tools.base import BaseTool

if TYPE_CHECKING:
    from ivy.cron.scheduler import CronScheduler
    from ivy.memory.store import MemoryStore


class CronManageTool(BaseTool):
    """List, add, pause, resume, or remove scheduled recurring cron jobs."""

    name = "cron_manage"
    description = "List, add, pause, resume, or remove scheduled background cron tasks."
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["list", "add", "pause", "resume", "remove"],
                "description": "Cron management action.",
            },
            "name": {"type": "string", "description": "Unique job identifier."},
            "cron_expr": {"type": "string", "description": "5-field cron schedule (e.g. '0 9 * * *')."},
            "prompt": {"type": "string", "description": "Prompt text run when job fires."},
        },
        "required": ["action"],
    }

    def __init__(
        self,
        cron_scheduler: CronScheduler,
        memory_store: MemoryStore | None = None,
    ) -> None:
        self.cron_scheduler = cron_scheduler
        self.memory_store = memory_store

    async def execute(
        self,
        action: str = "list",
        name: str | None = None,
        cron_expr: str | None = None,
        prompt: str | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        act = (action or "list").strip().lower()

        if act == "list":
            tasks = self.cron_scheduler.list_tasks()
            if not tasks:
                return ToolResult(tool_name=self.name, status="success", output="No scheduled cron jobs.")
            lines = [
                f"- {t.name} [{'active' if t.enabled else 'paused'}] `{t.cron_expr}` -> {t.prompt or '(handler)'}"
                for t in tasks
            ]
            return ToolResult(tool_name=self.name, status="success", output="\n".join(lines))

        if not name or not name.strip():
            return ToolResult(tool_name=self.name, status="error", error=f"'name' is required for '{act}'.")
        job_name = name.strip()

        if act == "add":
            if not cron_expr or not prompt:
                return ToolResult(
                    tool_name=self.name,
                    status="error",
                    error="Both 'cron_expr' and 'prompt' are required for 'add'.",
                )
            try:
                task = ScheduledTask(name=job_name, cron_expr=cron_expr.strip(), prompt=prompt.strip(), enabled=True)
            except ValueError as exc:
                return ToolResult(tool_name=self.name, status="error", error=str(exc))
            self.cron_scheduler.add_task(task)
            if self.memory_store is not None:
                await self.memory_store.save_cron_job(task.name, task.cron_expr, task.prompt or "", enabled=True)
            return ToolResult(
                tool_name=self.name,
                status="success",
                output=f"Added cron job '{task.name}' (`{task.cron_expr}`).",
            )

        if act == "pause":
            ok = self.cron_scheduler.pause_task(job_name)
            if ok and self.memory_store is not None:
                await self.memory_store.set_cron_job_enabled(job_name, False)
            return ToolResult(
                tool_name=self.name,
                status="success" if ok else "error",
                output=f"Paused '{job_name}'." if ok else "",
                error=None if ok else f"Cron job '{job_name}' not found.",
            )

        if act == "resume":
            ok = self.cron_scheduler.resume_task(job_name)
            if ok and self.memory_store is not None:
                await self.memory_store.set_cron_job_enabled(job_name, True)
            return ToolResult(
                tool_name=self.name,
                status="success" if ok else "error",
                output=f"Resumed '{job_name}'." if ok else "",
                error=None if ok else f"Cron job '{job_name}' not found.",
            )

        if act == "remove":
            ok = self.cron_scheduler.remove_task(job_name)
            if self.memory_store is not None:
                deleted = await self.memory_store.delete_cron_job(job_name)
                ok = ok or deleted
            return ToolResult(
                tool_name=self.name,
                status="success" if ok else "error",
                output=f"Removed '{job_name}'." if ok else "",
                error=None if ok else f"Cron job '{job_name}' not found.",
            )

        return ToolResult(tool_name=self.name, status="error", error=f"Unknown action '{action}'.")
