"""Background cron scheduler and scheduled job definitions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aether_agent.cron.scheduler import CronScheduler
    from aether_agent.cron.task import ScheduledTask

__all__ = ["CronScheduler", "ScheduledTask"]


def __getattr__(name: str) -> Any:
    if name == "CronScheduler":
        from aether_agent.cron.scheduler import CronScheduler
        return CronScheduler
    if name == "ScheduledTask":
        from aether_agent.cron.task import ScheduledTask
        return ScheduledTask
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
