"""Session recall tool exposing SQLite FTS5 search and history reading to the agent."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ivy.core.types import ToolResult
from ivy.tools.base import BaseTool

if TYPE_CHECKING:
    from ivy.memory.store import MemoryStore

_MAX_MESSAGE_CHARS = 1500
_MAX_SNIPPET_CHARS = 500


class SessionSearchTool(BaseTool):
    """Allow the agent to search past sessions via FTS5 or browse/read session transcripts."""

    name = "session_search"
    concurrent_safe = True
    description = (
        "Search past conversation sessions using FTS5 full-text search ('query'), "
        "inspect messages from a specific 'session_id', or list recent sessions when called with no arguments."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Full-text search query across past conversations.",
            },
            "session_id": {
                "type": "string",
                "description": "Optional specific session ID to inspect or filter search.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results/messages to return (default 10).",
                "default": 10,
            },
        },
        "required": [],
    }

    def __init__(self, memory_store: MemoryStore) -> None:
        self.memory_store = memory_store

    async def execute(
        self,
        query: str | None = None,
        session_id: str | None = None,
        limit: int = 10,
        **kwargs: Any,
    ) -> ToolResult:
        cap = max(1, min(30, int(limit or 10)))

        if query and query.strip():
            hits = await self.memory_store.search_fts(
                query.strip(), limit=cap, session_id=session_id
            )
            if not hits:
                return ToolResult(
                    tool_name=self.name,
                    status="success",
                    output="No matching past messages found.",
                )
            lines = []
            for h in hits:
                snippet = (h.get("content") or "").replace("\n", " ")[:_MAX_SNIPPET_CHARS]
                lines.append(
                    f"[{h.get('session_id', '')}] ({h.get('role', '')} · {str(h.get('created_at', ''))[:16]}): {snippet}"
                )
            return ToolResult(
                tool_name=self.name,
                status="success",
                output="\n".join(lines),
                metadata={"count": len(hits)},
            )

        if session_id and session_id.strip():
            msgs = await self.memory_store.get_messages(session_id.strip(), limit=cap)
            if not msgs:
                return ToolResult(
                    tool_name=self.name,
                    status="success",
                    output=f"No messages found in session '{session_id}'.",
                )
            lines = []
            for m in msgs:
                body = (m.content or "")[:_MAX_MESSAGE_CHARS]
                lines.append(f"[{m.role.value}]: {body}")
            return ToolResult(
                tool_name=self.name,
                status="success",
                output="\n".join(lines),
                metadata={"count": len(msgs)},
            )

        sessions = await self.memory_store.list_sessions(limit=cap)
        if not sessions:
            return ToolResult(
                tool_name=self.name,
                status="success",
                output="No saved sessions found.",
            )
        lines = [
            f"- {s['session_id']} | title={s.get('title') or '(untitled)'} | msgs={s['message_count']} | updated={str(s['updated_at'])[:19]}"
            for s in sessions
        ]
        return ToolResult(
            tool_name=self.name,
            status="success",
            output="\n".join(lines),
            metadata={"count": len(sessions)},
        )
