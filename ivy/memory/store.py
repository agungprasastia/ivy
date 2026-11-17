"""Central SQLite memory store with FTS5 search, sessions, and user profiling."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ivy.core.types import ChatMessage, Role, ToolCall
from ivy.memory.context import ContextWindowManager
from ivy.memory.fts import search_messages_fts
from ivy.memory.profile import UserProfileStore


class MemoryStore:
    """Async-safe SQLite state store with FTS5 full-text search and user profiling."""

    def __init__(self, db_path: str = "./ivy.db") -> None:
        self.db_path = db_path
        self._conn: sqlite3.Connection | None = None
        self._lock = asyncio.Lock()
        self._context_mgr = ContextWindowManager()

    def _ensure_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("MemoryStore is not initialized. Call initialize() first.")
        return self._conn

    async def initialize(self) -> None:
        """Open SQLite connection and ensure schema and FTS5 tables exist."""
        async with self._lock:
            if self._conn is not None:
                return
            if self.db_path != ":memory:":
                Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA foreign_keys=ON;")
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    platform TEXT DEFAULT 'cli',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    tool_calls_json TEXT,
                    tool_call_id TEXT,
                    name TEXT,
                    exclude_from_fts INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL
                );

                CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
                    session_id UNINDEXED,
                    role UNINDEXED,
                    content,
                    created_at UNINDEXED
                );

                CREATE TABLE IF NOT EXISTS user_profile (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS cron_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    output TEXT,
                    ran_at TEXT NOT NULL
                );
                """
            )
            conn.commit()
            self._conn = conn

    async def close(self) -> None:
        """Close SQLite connection."""
        async with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    async def save_message(
        self,
        session_id: str,
        message: ChatMessage,
        platform: str = "cli",
        exclude_from_fts: bool = False,
    ) -> None:
        """Persist a single message and optionally index it in FTS5."""
        async with self._lock:
            conn = self._ensure_conn()
            now = datetime.now(timezone.utc).isoformat()
            conn.execute(
                """
                INSERT INTO sessions (session_id, platform, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET updated_at = excluded.updated_at
                """,
                (session_id, platform, now, now),
            )
            tool_calls_json = (
                json.dumps([tc.model_dump() for tc in message.tool_calls])
                if message.tool_calls
                else None
            )
            content = message.content or ""
            conn.execute(
                """
                INSERT INTO messages (
                    session_id, role, content, tool_calls_json,
                    tool_call_id, name, exclude_from_fts, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    message.role.value,
                    content,
                    tool_calls_json,
                    message.tool_call_id,
                    message.name,
                    1 if exclude_from_fts else 0,
                    now,
                ),
            )
            if not exclude_from_fts and content.strip():
                conn.execute(
                    """
                    INSERT INTO messages_fts (session_id, role, content, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (session_id, message.role.value, content, now),
                )
            conn.commit()

    async def save_turn(
        self,
        session_id: str,
        messages: list[ChatMessage],
        platform: str = "cli",
        exclude_from_fts: bool = False,
    ) -> None:
        """Persist a sequence of messages belonging to one agent turn."""
        for msg in messages:
            await self.save_message(
                session_id,
                msg,
                platform=platform,
                exclude_from_fts=exclude_from_fts,
            )

    async def get_messages(
        self, session_id: str, limit: int = 50
    ) -> list[ChatMessage]:
        """Load recent messages for a session in chronological order."""
        async with self._lock:
            conn = self._ensure_conn()
            rows = conn.execute(
                """
                SELECT role, content, tool_calls_json, tool_call_id, name
                FROM (
                    SELECT id, role, content, tool_calls_json, tool_call_id, name
                    FROM messages
                    WHERE session_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                )
                ORDER BY id ASC
                """,
                (session_id, limit),
            ).fetchall()

        result: list[ChatMessage] = []
        for role_val, content, tc_json, tc_id, name in rows:
            tool_calls = None
            if tc_json:
                tool_calls = [ToolCall(**item) for item in json.loads(tc_json)]
            result.append(
                ChatMessage(
                    role=Role(role_val),
                    content=content,
                    tool_calls=tool_calls,
                    tool_call_id=tc_id,
                    name=name,
                )
            )
        return result

    async def clear_session(self, session_id: str) -> None:
        """Delete messages for a session (without removing cross-session profile)."""
        async with self._lock:
            conn = self._ensure_conn()
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM messages_fts WHERE session_id = ?", (session_id,))
            conn.commit()

    async def search_fts(
        self, query: str, limit: int = 10, session_id: str | None = None
    ) -> list[dict[str, Any]]:
        """Full-text search across stored non-subagent conversation messages."""
        async with self._lock:
            conn = self._ensure_conn()
            return search_messages_fts(conn, query, limit=limit, session_id=session_id)

    async def search_history_fts(
        self, query: str, limit: int = 10, session_id: str | None = None
    ) -> list[dict[str, Any]]:
        """Alias for search_fts."""
        return await self.search_fts(query, limit=limit, session_id=session_id)

    async def set_user_preference(self, key: str, value: str) -> None:
        """Persist a user preference or profile fact."""
        async with self._lock:
            conn = self._ensure_conn()
            UserProfileStore(conn).set_preference(key, value)

    async def get_user_profile(self) -> dict[str, str]:
        """Return all stored user profile preferences."""
        async with self._lock:
            conn = self._ensure_conn()
            return UserProfileStore(conn).get_all()

    async def log_cron_run(self, task_name: str, status: str, output: str = "") -> None:
        """Record a background cron execution in SQLite."""
        async with self._lock:
            conn = self._ensure_conn()
            now = datetime.now(timezone.utc).isoformat()
            conn.execute(
                "INSERT INTO cron_runs (task_name, status, output, ran_at) VALUES (?, ?, ?, ?)",
                (task_name, status, output, now),
            )
            conn.commit()

    def compress_context(
        self, messages: list[ChatMessage], max_messages: int | None = None
    ) -> list[ChatMessage]:
        """Compress message list to fit context window budget."""
        if max_messages is not None:
            return ContextWindowManager(max_messages=max_messages).compress(messages)
        return self._context_mgr.compress(messages)
