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
                    updated_at TEXT NOT NULL,
                    title TEXT
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

                CREATE TABLE IF NOT EXISTS cron_jobs (
                    name TEXT PRIMARY KEY,
                    cron_expr TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    enabled INTEGER DEFAULT 1,
                    created_at TEXT NOT NULL
                );
                """
            )
            # Lightweight schema migration for existing DBs missing sessions.title
            cols = {
                row[1] for row in conn.execute("PRAGMA table_info(sessions)").fetchall()
            }
            if "title" not in cols:
                conn.execute("ALTER TABLE sessions ADD COLUMN title TEXT")
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
                SELECT role, content, tool_calls_json, tool_call_id, name, created_at
                FROM (
                    SELECT id, role, content, tool_calls_json, tool_call_id, name, created_at
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
        for role_val, content, tc_json, tc_id, name, created_at in rows:
            tool_calls = None
            if tc_json:
                tool_calls = [ToolCall(**item) for item in json.loads(tc_json)]
            ts = datetime.now(timezone.utc)
            if created_at:
                try:
                    ts = datetime.fromisoformat(created_at)
                except ValueError:
                    pass
            result.append(
                ChatMessage(
                    role=Role(role_val),
                    content=content,
                    tool_calls=tool_calls,
                    tool_call_id=tc_id,
                    name=name,
                    timestamp=ts,
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

    async def list_sessions(self, limit: int = 20) -> list[dict[str, Any]]:
        """List saved sessions ordered by most recently updated."""
        async with self._lock:
            conn = self._ensure_conn()
            rows = conn.execute(
                """
                SELECT
                    s.session_id,
                    s.platform,
                    s.updated_at,
                    COUNT(m.id) AS message_count,
                    s.title,
                    (
                        SELECT m2.content
                        FROM messages m2
                        WHERE m2.session_id = s.session_id AND m2.role = 'user' AND m2.content != ''
                        ORDER BY m2.id ASC
                        LIMIT 1
                    ) AS first_prompt
                FROM sessions s
                LEFT JOIN messages m ON m.session_id = s.session_id
                GROUP BY s.session_id, s.platform, s.updated_at, s.title
                ORDER BY s.updated_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [
                {
                    "session_id": row[0],
                    "platform": row[1],
                    "updated_at": row[2],
                    "message_count": int(row[3]),
                    "title": (row[4] or "").strip()
                    or (" ".join((row[5] or "").split())[:48] if row[5] else ""),
                }
                for row in rows
            ]

    async def ensure_session(self, session_id: str, platform: str = "cli") -> None:
        """Create a session row if missing so newly started sessions appear in pickers."""
        async with self._lock:
            conn = self._ensure_conn()
            now = datetime.now(timezone.utc).isoformat()
            conn.execute(
                """
                INSERT INTO sessions (session_id, platform, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(session_id) DO NOTHING
                """,
                (session_id, platform, now, now),
            )
            conn.commit()

    async def delete_session(self, session_id: str) -> bool:
        """Permanently delete a session and all of its messages."""
        async with self._lock:
            conn = self._ensure_conn()
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM messages_fts WHERE session_id = ?", (session_id,))
            cur = conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            conn.commit()
            return (cur.rowcount or 0) > 0

    async def set_session_title(
        self, session_id: str, title: str, platform: str = "cli"
    ) -> None:
        """Set or update the human-friendly title for a session."""
        clean = " ".join(title.strip().split())[:80]
        async with self._lock:
            conn = self._ensure_conn()
            now = datetime.now(timezone.utc).isoformat()
            conn.execute(
                """
                INSERT INTO sessions (session_id, platform, created_at, updated_at, title)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    title = excluded.title,
                    updated_at = excluded.updated_at
                """,
                (session_id, platform, now, now, clean),
            )
            conn.commit()

    async def get_session_title(self, session_id: str) -> str | None:
        """Retrieve session title if set."""
        async with self._lock:
            conn = self._ensure_conn()
            row = conn.execute(
                "SELECT title FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            return row[0] if row and row[0] else None

    async def get_latest_session_id(self) -> str | None:
        """Return the most recently updated session ID, or None if empty."""
        async with self._lock:
            conn = self._ensure_conn()
            row = conn.execute(
                "SELECT session_id FROM sessions ORDER BY updated_at DESC LIMIT 1"
            ).fetchone()
            return row[0] if row else None

    async def delete_last_turn(self, session_id: str) -> tuple[str | None, int]:
        """Remove the last user prompt and its subsequent messages from a session."""
        async with self._lock:
            conn = self._ensure_conn()
            row = conn.execute(
                """
                SELECT id, content
                FROM messages
                WHERE session_id = ? AND role = 'user'
                ORDER BY id DESC
                LIMIT 1
                """,
                (session_id,),
            ).fetchone()
            if row is None:
                return None, 0

            cut_id, user_prompt = row[0], row[1]
            cur = conn.execute(
                "DELETE FROM messages WHERE session_id = ? AND id >= ?",
                (session_id, cut_id),
            )
            deleted = int(cur.rowcount or 0)
            # Rebuild FTS entries for this session so FTS stays exact
            conn.execute("DELETE FROM messages_fts WHERE session_id = ?", (session_id,))
            remaining = conn.execute(
                """
                SELECT role, content, created_at
                FROM messages
                WHERE session_id = ? AND exclude_from_fts = 0 AND content != ''
                ORDER BY id ASC
                """,
                (session_id,),
            ).fetchall()
            for r_role, r_content, r_ts in remaining:
                conn.execute(
                    "INSERT INTO messages_fts (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                    (session_id, r_role, r_content, r_ts),
                )
            conn.commit()
            return user_prompt, deleted

    async def replace_session_messages(
        self, session_id: str, messages: list[ChatMessage], platform: str = "cli"
    ) -> None:
        """Atomically replace all messages of a session (used by /compact)."""
        await self.clear_session(session_id)
        await self.save_turn(session_id, messages, platform=platform)

    async def fork_session(
        self,
        src_session_id: str,
        dst_session_id: str,
        title: str | None = None,
        platform: str = "cli",
    ) -> int:
        """Clone messages from src_session_id into a new dst_session_id."""
        src_messages = await self.get_messages(src_session_id, limit=500)
        src_title = await self.get_session_title(src_session_id)
        await self.save_turn(dst_session_id, src_messages, platform=platform)
        fork_title = title or (f"{src_title} (fork)" if src_title else f"Fork of {src_session_id}")
        await self.set_session_title(dst_session_id, fork_title, platform=platform)
        return len(src_messages)

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

    async def delete_user_preference(self, key: str) -> bool:
        """Remove a single preference from user_profile."""
        async with self._lock:
            conn = self._ensure_conn()
            return UserProfileStore(conn).delete_preference(key)

    async def clear_user_profile(self) -> int:
        """Clear all user_profile preferences."""
        async with self._lock:
            conn = self._ensure_conn()
            return UserProfileStore(conn).clear()

    async def get_user_turns(self, session_id: str) -> list[dict[str, Any]]:
        """List chronological user turn anchor rows {id, content, created_at} for /rewind."""
        async with self._lock:
            conn = self._ensure_conn()
            rows = conn.execute(
                """
                SELECT id, content, created_at
                FROM messages
                WHERE session_id = ? AND role = 'user'
                ORDER BY id ASC
                """,
                (session_id,),
            ).fetchall()
            return [
                {"id": int(r[0]), "content": r[1], "created_at": r[2]}
                for r in rows
            ]

    async def rewind_to_message_id(
        self, session_id: str, message_id: int
    ) -> tuple[str | None, int]:
        """Delete messages from `message_id` onward in a session, keeping FTS synced."""
        async with self._lock:
            conn = self._ensure_conn()
            row = conn.execute(
                "SELECT content FROM messages WHERE session_id = ? AND id = ?",
                (session_id, message_id),
            ).fetchone()
            if row is None:
                return None, 0
            target_prompt = row[0]
            cur = conn.execute(
                "DELETE FROM messages WHERE session_id = ? AND id >= ?",
                (session_id, message_id),
            )
            deleted = int(cur.rowcount or 0)
            conn.execute("DELETE FROM messages_fts WHERE session_id = ?", (session_id,))
            remaining = conn.execute(
                """
                SELECT role, content, created_at
                FROM messages
                WHERE session_id = ? AND exclude_from_fts = 0 AND content != ''
                ORDER BY id ASC
                """,
                (session_id,),
            ).fetchall()
            for r_role, r_content, r_ts in remaining:
                conn.execute(
                    "INSERT INTO messages_fts (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                    (session_id, r_role, r_content, r_ts),
                )
            conn.commit()
            return target_prompt, deleted

    async def save_cron_job(
        self, name: str, cron_expr: str, prompt: str, enabled: bool = True
    ) -> None:
        """Persist a scheduled cron job in SQLite."""
        async with self._lock:
            conn = self._ensure_conn()
            now = datetime.now(timezone.utc).isoformat()
            conn.execute(
                """
                INSERT INTO cron_jobs (name, cron_expr, prompt, enabled, created_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    cron_expr = excluded.cron_expr,
                    prompt = excluded.prompt,
                    enabled = excluded.enabled
                """,
                (name, cron_expr, prompt, 1 if enabled else 0, now),
            )
            conn.commit()

    async def delete_cron_job(self, name: str) -> bool:
        """Delete a persisted cron job by name."""
        async with self._lock:
            conn = self._ensure_conn()
            cur = conn.execute("DELETE FROM cron_jobs WHERE name = ?", (name,))
            conn.commit()
            return (cur.rowcount or 0) > 0

    async def set_cron_job_enabled(self, name: str, enabled: bool) -> bool:
        """Enable or disable a persisted cron job."""
        async with self._lock:
            conn = self._ensure_conn()
            cur = conn.execute(
                "UPDATE cron_jobs SET enabled = ? WHERE name = ?",
                (1 if enabled else 0, name),
            )
            conn.commit()
            return (cur.rowcount or 0) > 0

    async def list_cron_jobs(self) -> list[dict[str, Any]]:
        """List all persisted cron jobs ordered by name."""
        async with self._lock:
            conn = self._ensure_conn()
            rows = conn.execute(
                "SELECT name, cron_expr, prompt, enabled, created_at FROM cron_jobs ORDER BY name ASC"
            ).fetchall()
            return [
                {
                    "name": row[0],
                    "cron_expr": row[1],
                    "prompt": row[2],
                    "enabled": bool(row[3]),
                    "created_at": row[4],
                }
                for row in rows
            ]

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
