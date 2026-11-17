"""Scheduled task definition and cron expression helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

from croniter import croniter


@dataclass
class ScheduledTask:
    """Represents a recurring scheduled task evaluated via croniter."""

    name: str
    cron_expr: str
    prompt: str | None = None
    handler: Callable[[], Any | Awaitable[Any]] | None = None
    target_channel: str | None = None
    target_id: str | None = None
    enabled: bool = True
    next_run: datetime | None = field(default=None)

    def __post_init__(self) -> None:
        if not croniter.is_valid(self.cron_expr):
            raise ValueError(f"Invalid cron expression: {self.cron_expr!r}")
        if self.next_run is None:
            self.next_run = self.compute_next_run()

    def compute_next_run(self, base_time: datetime | None = None) -> datetime:
        """Calculate the next UTC run time from base_time."""
        ref = base_time or datetime.now(timezone.utc)
        if ref.tzinfo is None:
            ref = ref.replace(tzinfo=timezone.utc)
        next_dt = croniter(self.cron_expr, ref).get_next(datetime)
        if next_dt.tzinfo is None:
            next_dt = next_dt.replace(tzinfo=timezone.utc)
        return next_dt
