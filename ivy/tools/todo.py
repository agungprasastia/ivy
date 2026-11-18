"""Session-scoped structured task checklist tool for multi-step agent execution."""

from __future__ import annotations

from typing import Any, Callable

from ivy.core.types import ToolResult
from ivy.tools.base import BaseTool

VALID_STATUSES = ("pending", "in_progress", "completed", "cancelled")

_STATUS_MARKERS = {
    "completed": "[x]",
    "in_progress": "[>]",
    "pending": "[ ]",
    "cancelled": "[~]",
}


class TodoStore:
    """In-memory per-session task checklist store."""

    def __init__(self) -> None:
        self._by_session: dict[str, list[dict[str, str]]] = {}

    def get(self, session_id: str) -> list[dict[str, str]]:
        """Return shallow copies of all todo items for a session."""
        return [dict(item) for item in self._by_session.get(session_id, [])]

    def clear(self, session_id: str) -> None:
        """Clear all todo items for a session."""
        self._by_session.pop(session_id, None)

    @staticmethod
    def _normalize_item(raw: dict[str, Any], fallback_idx: int) -> dict[str, str]:
        item_id = str(raw.get("id") or fallback_idx).strip()
        content = " ".join(str(raw.get("content") or "").split())[:500]
        raw_status = str(raw.get("status") or "pending").strip().lower()
        status = raw_status if raw_status in VALID_STATUSES else "pending"
        return {"id": item_id, "content": content, "status": status}

    def write(
        self,
        session_id: str,
        todos: list[dict[str, Any]],
        merge: bool = False,
    ) -> list[dict[str, str]]:
        """Replace or merge todos for a session, returning the normalized list."""
        if not merge:
            normalized = [
                self._normalize_item(item, i)
                for i, item in enumerate(todos, start=1)
                if str(item.get("content") or "").strip()
            ]
            self._by_session[session_id] = normalized
            return self.get(session_id)

        existing = self.get(session_id)
        by_id = {item["id"]: idx for idx, item in enumerate(existing)}
        next_idx = len(existing) + 1
        for raw in todos:
            norm = self._normalize_item(raw, next_idx)
            if norm["id"] in by_id:
                idx = by_id[norm["id"]]
                if not norm["content"]:
                    norm["content"] = existing[idx]["content"]
                existing[idx] = norm
            elif norm["content"]:
                by_id[norm["id"]] = len(existing)
                existing.append(norm)
                next_idx += 1
        self._by_session[session_id] = existing
        return self.get(session_id)

    def format_markdown(self, session_id: str) -> str:
        """Render the session's checklist as plain/markdown bullets."""
        items = self._by_session.get(session_id, [])
        if not items:
            return "(todo list is empty)"
        lines = []
        for item in items:
            marker = _STATUS_MARKERS.get(item["status"], "[ ]")
            lines.append(f"{marker} #{item['id']}: {item['content']} ({item['status']})")
        return "\n".join(lines)


class TodoListTool(BaseTool):
    """Tool allowing the agent to track and update multi-step progress."""

    name = "todo_list"
    description = (
        "Read or update the structured task checklist for multi-step work. "
        "Pass 'todos' to write/update; omit 'todos' to read the current list."
    )
    parameters = {
        "type": "object",
        "properties": {
            "todos": {
                "type": "array",
                "description": "List of task items with optional id, content, and status.",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "content": {"type": "string"},
                        "status": {
                            "type": "string",
                            "enum": list(VALID_STATUSES),
                        },
                    },
                    "required": ["content"],
                },
            },
            "merge": {
                "type": "boolean",
                "description": "When true, update matching IDs and append new items instead of replacing.",
                "default": False,
            },
        },
        "required": [],
    }

    def __init__(
        self,
        store: TodoStore | None = None,
        session_id_getter: Callable[[], str] | None = None,
    ) -> None:
        self.store = store or TodoStore()
        self.session_id_getter = session_id_getter

    async def execute(
        self,
        todos: list[dict[str, Any]] | None = None,
        merge: bool = False,
        **kwargs: Any,
    ) -> ToolResult:
        session_id = str(
            kwargs.get("session_id")
            or (self.session_id_getter() if self.session_id_getter else None)
            or "cli:main"
        )
        if todos is None:
            items = self.store.get(session_id)
        else:
            items = self.store.write(session_id, todos, merge=bool(merge))

        return ToolResult(
            tool_name=self.name,
            status="success",
            output=self.store.format_markdown(session_id),
            metadata={"todos": items, "session_id": session_id},
        )
