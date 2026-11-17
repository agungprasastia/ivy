"""SQLite memory store, FTS5 search, user profiling, and context window management."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aether_agent.memory.context import ContextWindowManager
    from aether_agent.memory.fts import sanitize_fts_query
    from aether_agent.memory.profile import UserProfileStore
    from aether_agent.memory.store import MemoryStore

__all__ = [
    "MemoryStore",
    "UserProfileStore",
    "ContextWindowManager",
    "sanitize_fts_query",
]

_EXPORT_MAP = {
    "MemoryStore": ("aether_agent.memory.store", "MemoryStore"),
    "UserProfileStore": ("aether_agent.memory.profile", "UserProfileStore"),
    "ContextWindowManager": ("aether_agent.memory.context", "ContextWindowManager"),
    "sanitize_fts_query": ("aether_agent.memory.fts", "sanitize_fts_query"),
}


def __getattr__(name: str) -> Any:
    if name in _EXPORT_MAP:
        mod_path, attr = _EXPORT_MAP[name]
        module = __import__(mod_path, fromlist=[attr])
        return getattr(module, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
