"""Curated long-term memory note tool backed by SQLite MemoryStore."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ivy.core.types import ToolResult
from ivy.tools.base import BaseTool

if TYPE_CHECKING:
    from ivy.memory.store import MemoryStore


class MemoryNoteTool(BaseTool):
    """Add, list, search, and remove curated cross-session memory notes."""

    name = "memory_note"
    description = (
        "Manage durable curated project/workflow memory notes injected into future sessions "
        "(actions: add, list, search, remove)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["add", "list", "search", "remove"],
                "description": "Operation to perform.",
            },
            "note": {
                "type": "string",
                "description": "Concise durable fact/workflow note (required for 'add', max 300 chars).",
            },
            "category": {
                "type": "string",
                "description": "Optional category tag (e.g. 'workflow', 'arch', 'ops').",
                "default": "general",
            },
            "query": {
                "type": "string",
                "description": "Search substring (required for 'search').",
            },
            "note_id": {
                "type": "integer",
                "description": "Note ID to delete (required for 'remove').",
            },
        },
        "required": ["action"],
    }

    def __init__(self, memory_store: MemoryStore) -> None:
        self.memory_store = memory_store

    async def execute(
        self,
        action: str,
        note: str | None = None,
        category: str = "general",
        query: str | None = None,
        note_id: int | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        act = (action or "").strip().lower()
        if act == "add":
            if not note or not note.strip():
                return ToolResult(
                    tool_name=self.name,
                    status="error",
                    error="'note' is required for action 'add'.",
                )
            try:
                saved = await self.memory_store.add_memory_note(note, category=category)
            except ValueError as exc:
                return ToolResult(tool_name=self.name, status="error", error=str(exc))
            return ToolResult(
                tool_name=self.name,
                status="success",
                output=f"Saved memory note #{saved['id']} ({saved['category']}): {saved['note']}",
            )

        if act == "list":
            rows = await self.memory_store.list_memory_notes(
                category=category if category and category != "general" else None
            )
            if not rows:
                return ToolResult(
                    tool_name=self.name,
                    status="success",
                    output="No curated memory notes saved.",
                )
            lines = [f"[#{r['id']}] ({r['category']}) {r['note']}" for r in rows]
            return ToolResult(tool_name=self.name, status="success", output="\n".join(lines))

        if act == "search":
            if not query or not query.strip():
                return ToolResult(
                    tool_name=self.name,
                    status="error",
                    error="'query' is required for action 'search'.",
                )
            rows = await self.memory_store.search_memory_notes(query)
            if not rows:
                return ToolResult(
                    tool_name=self.name,
                    status="success",
                    output=f"No curated memory notes matching '{query}'.",
                )
            lines = [f"[#{r['id']}] ({r['category']}) {r['note']}" for r in rows]
            return ToolResult(tool_name=self.name, status="success", output="\n".join(lines))

        if act == "remove":
            if note_id is None:
                return ToolResult(
                    tool_name=self.name,
                    status="error",
                    error="'note_id' is required for action 'remove'.",
                )
            removed = await self.memory_store.remove_memory_note(note_id)
            if not removed:
                return ToolResult(
                    tool_name=self.name,
                    status="error",
                    error=f"Memory note #{note_id} not found.",
                )
            return ToolResult(
                tool_name=self.name,
                status="success",
                output=f"Removed memory note #{note_id}.",
            )

        return ToolResult(
            tool_name=self.name,
            status="error",
            error=f"Unknown memory_note action '{action}'. Use add, list, search, or remove.",
        )
