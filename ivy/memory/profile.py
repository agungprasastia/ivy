"""Long-term user preference and profile storage."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone


class UserProfileStore:
    """Manages persistent key-value user preferences in SQLite."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def set_preference(self, key: str, value: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute(
            """
            INSERT INTO user_profile (key, value, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
            """,
            (key, value, now),
        )
        self.conn.commit()

    def get_preference(self, key: str, default: str | None = None) -> str | None:
        row = self.conn.execute(
            "SELECT value FROM user_profile WHERE key = ?",
            (key,),
        ).fetchone()
        return row[0] if row else default

    def get_all(self) -> dict[str, str]:
        rows = self.conn.execute(
            "SELECT key, value FROM user_profile ORDER BY key"
        ).fetchall()
        return {row[0]: row[1] for row in rows}
