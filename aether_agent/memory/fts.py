"""SQLite FTS5 full-text search helpers and query sanitization."""

from __future__ import annotations

import re
import sqlite3
from typing import Any

_FTS_KEYWORDS = {"AND", "OR", "NOT", "NEAR"}


def sanitize_fts_query(query: str) -> str:
    """Sanitize arbitrary user text into a safe SQLite FTS5 MATCH expression."""
    tokens = re.findall(r"[A-Za-z0-9_\u00C0-\u024F]+", query)
    safe_tokens: list[str] = []
    for tok in tokens:
        if tok.upper() in _FTS_KEYWORDS:
            safe_tokens.append(f'"{tok}"')
        else:
            safe_tokens.append(f'"{tok}"')
    return " OR ".join(safe_tokens)


def search_messages_fts(
    conn: sqlite3.Connection,
    query: str,
    limit: int = 10,
    session_id: str | None = None,
) -> list[dict[str, Any]]:
    """Execute sanitized FTS5 query against messages_fts."""
    safe_query = sanitize_fts_query(query)
    if not safe_query:
        return []

    sql = """
        SELECT session_id, role, content, created_at, bm25(messages_fts) AS rank
        FROM messages_fts
        WHERE messages_fts MATCH ?
    """
    params: list[Any] = [safe_query]
    if session_id:
        sql += " AND session_id = ?"
        params.append(session_id)

    sql += " ORDER BY rank LIMIT ?"
    params.append(limit)

    cursor = conn.execute(sql, params)
    rows = cursor.fetchall()
    return [
        {
            "session_id": row[0],
            "role": row[1],
            "content": row[2],
            "created_at": row[3],
            "rank": row[4],
        }
        for row in rows
    ]
