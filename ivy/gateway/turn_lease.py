"""Async per-session turn lease manager to prevent concurrent turn races."""

from __future__ import annotations

import asyncio


class TurnLeaseManager:
    """Manages exclusive per-session turn leases across CLI, bots, and cron."""

    def __init__(self) -> None:
        self._locks: dict[str, asyncio.Lock] = {}

    def _get_lock(self, session_id: str) -> asyncio.Lock:
        if session_id not in self._locks:
            self._locks[session_id] = asyncio.Lock()
        return self._locks[session_id]

    def is_locked(self, session_id: str) -> bool:
        """Return True if the given session currently holds an active lease."""
        lock = self._locks.get(session_id)
        return lock.locked() if lock is not None else False

    async def acquire(self, session_id: str, timeout: float | None = None) -> bool:
        """Acquire exclusive lease for session_id, returning False on timeout."""
        lock = self._get_lock(session_id)
        if timeout is None:
            await lock.acquire()
            return True
        try:
            await asyncio.wait_for(lock.acquire(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    async def release(self, session_id: str) -> None:
        """Release the lease for session_id if currently locked."""
        lock = self._locks.get(session_id)
        if lock is not None and lock.locked():
            lock.release()
