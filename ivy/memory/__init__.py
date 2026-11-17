"""SQLite memory store, FTS5 search, user profiling, and context window management."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ivy.memory.context import ContextWindowManager
    from ivy.memory.fts import sanitize_fts_query
    from ivy.memory.profile import UserProfileStore
    from ivy.memory.store import MemoryStore

__all__ = [
    "MemoryStore",
    "UserProfileStore",
    "ContextWindowManager",
    "sanitize_fts_query",
]

_EXPORT_MAP = {
    "MemoryStore": ("ivy.memory.store", "MemoryStore"),
    "UserProfileStore": ("ivy.memory.profile", "UserProfileStore"),
    "ContextWindowManager": ("ivy.memory.context", "ContextWindowManager"),
    "sanitize_fts_query": ("ivy.memory.fts", "sanitize_fts_query"),
}


def __getattr__(name: str) -> Any:
    if name in _EXPORT_MAP:
        mod_path, attr = _EXPORT_MAP[name]
        module = __import__(mod_path, fromlist=[attr])
        return getattr(module, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
