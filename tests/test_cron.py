from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from aether_agent.core.types import ChatMessage, Role
from aether_agent.cron.scheduler import CronScheduler
from aether_agent.cron.task import ScheduledTask
from aether_agent.memory.store import MemoryStore


@pytest.mark.asyncio
async def test_cron_scheduler_lifecycle():
    scheduler = CronScheduler()

    async def sample_handler():
        return "ok"

    task = ScheduledTask(name="test-job", cron_expr="* * * * *", handler=sample_handler)
    scheduler.add_task(task)
    assert len(scheduler.list_tasks()) == 1

    # Also verify add_job / remove_job aliases
    assert scheduler.remove_job("test-job") is True
    assert len(scheduler.list_tasks()) == 0


@pytest.mark.asyncio
async def test_cron_scheduler_executes_due_task_and_dispatches(tmp_path):
    store = MemoryStore(db_path=str(tmp_path / "cron.db"))
    await store.initialize()

    mock_agent = MagicMock()
    mock_agent.run_turn = AsyncMock(
        return_value=ChatMessage(role=Role.ASSISTANT, content="Daily summary ready")
    )

    delivered = []

    async def dispatcher(channel: str, target_id: str, text: str):
        delivered.append((channel, target_id, text))

    scheduler = CronScheduler(agent=mock_agent, memory_store=store, dispatcher=dispatcher)
    past = datetime.now(timezone.utc) - timedelta(minutes=2)
    task = ScheduledTask(
        name="daily-report",
        cron_expr="* * * * *",
        prompt="Summarize system status",
        target_channel="telegram",
        target_id="12345",
        next_run=past,
    )
    scheduler.add_job(task)
    ran = await scheduler.run_pending()
    assert ran == ["daily-report"]
    assert delivered == [("telegram", "12345", "Daily summary ready")]

    await store.close()
