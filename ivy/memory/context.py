"""Sliding context window and conversation compressor."""

from __future__ import annotations

from ivy.core.types import ChatMessage, Role


class ContextWindowManager:
    """Compresses long message histories while keeping system prompts and recent turns."""

    def __init__(self, max_messages: int = 24, keep_recent: int = 12) -> None:
        self.max_messages = max(2, max_messages)
        self.keep_recent = max(1, min(keep_recent, self.max_messages - 1))

    def compress(self, messages: list[ChatMessage]) -> list[ChatMessage]:
        """Compress message list if it exceeds max_messages."""
        if len(messages) <= self.max_messages:
            return list(messages)

        system_msgs = [m for m in messages if m.role == Role.SYSTEM]
        non_system = [m for m in messages if m.role != Role.SYSTEM]

        budget = max(1, self.max_messages - len(system_msgs) - 1)
        keep_count = min(self.keep_recent, budget)

        if len(non_system) <= keep_count:
            return system_msgs + non_system

        older = non_system[:-keep_count]
        recent = non_system[-keep_count:]

        # Avoid starting recent slice with an orphaned tool result
        while recent and recent[0].role == Role.TOOL and older:
            recent.insert(0, older.pop())

        summary_lines: list[str] = []
        for msg in older[-10:]:
            snippet = (msg.content or "").strip().replace("\n", " ")
            if len(snippet) > 120:
                snippet = snippet[:117] + "..."
            if snippet:
                summary_lines.append(f"- {msg.role.value}: {snippet}")

        summary_text = (
            f"[Compressed {len(older)} earlier messages]\n" + "\n".join(summary_lines)
        )
        summary_msg = ChatMessage(role=Role.SYSTEM, content=summary_text)
        return system_msgs + [summary_msg] + recent
