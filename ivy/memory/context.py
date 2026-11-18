"""Sliding context window, token estimation, and conversation compaction."""

from __future__ import annotations

import json
from typing import Any

from ivy.core.types import ChatMessage, Role

_MODEL_CONTEXT_WINDOWS: list[tuple[str, int]] = [
    ("gemini", 1_000_000),
    ("claude", 200_000),
    ("gpt-5", 400_000),
    ("gpt-4.1", 1_000_000),
    ("gpt-4o", 128_000),
    ("o1", 200_000),
    ("o3", 200_000),
    ("o4", 200_000),
    ("deepseek", 128_000),
    ("qwen", 128_000),
    ("llama", 128_000),
]


def get_context_window(model: str | None) -> int:
    """Return estimated context window size in tokens for a model identifier."""
    low = (model or "").lower()
    for prefix, window in _MODEL_CONTEXT_WINDOWS:
        if prefix in low:
            return window
    return 128_000


def estimate_tokens(text: str | None) -> int:
    """Fast character-based token estimate (~4 chars/token) with zero dependencies."""
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def estimate_message_tokens(msg: ChatMessage) -> int:
    """Estimate tokens consumed by a single ChatMessage including tool calls."""
    total = 4 + estimate_tokens(msg.content)
    if msg.name:
        total += estimate_tokens(msg.name)
    if msg.tool_calls:
        for tc in msg.tool_calls:
            total += estimate_tokens(tc.function_name)
            total += estimate_tokens(json.dumps(tc.arguments, ensure_ascii=False))
    return total


def compute_context_breakdown(
    messages: list[ChatMessage],
    model: str | None = None,
    tool_schemas: list[dict[str, Any]] | None = None,
    system_prompt: str = "",
) -> dict[str, Any]:
    """Compute local zero-cost token usage breakdown across prompt, tools, and history."""
    system_tokens = estimate_tokens(system_prompt)
    tool_tokens = estimate_tokens(json.dumps(tool_schemas, default=str)) if tool_schemas else 0
    history_tokens = sum(
        estimate_message_tokens(m) for m in messages if m.role != Role.SYSTEM
    )
    if not system_prompt:
        system_tokens = sum(
            estimate_message_tokens(m) for m in messages if m.role == Role.SYSTEM
        )
    max_tokens = get_context_window(model)
    used_tokens = system_tokens + tool_tokens + history_tokens
    percent = min(100.0, round((used_tokens / max_tokens) * 100.0, 1)) if max_tokens else 0.0
    free_tokens = max(0, max_tokens - used_tokens)
    return {
        "used_tokens": used_tokens,
        "max_tokens": max_tokens,
        "percent": percent,
        "estimated": True,
        "system_tokens": system_tokens,
        "tool_tokens": tool_tokens,
        "history_tokens": history_tokens,
        "free_tokens": free_tokens,
        "message_count": len(messages),
    }


def should_auto_compact(
    messages: list[ChatMessage],
    model: str | None = None,
    tool_schemas: list[dict[str, Any]] | None = None,
    system_prompt: str = "",
    threshold: float = 0.85,
) -> bool:
    """True when message history reaches the configured fraction of the model context window."""
    if len(messages) <= 8:
        return False
    breakdown = compute_context_breakdown(
        messages=messages,
        model=model,
        tool_schemas=tool_schemas,
        system_prompt=system_prompt,
    )
    return float(breakdown["percent"]) >= float(threshold) * 100.0


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

    async def compact_with_llm(
        self,
        messages: list[ChatMessage],
        llm_client: Any = None,
        keep_recent: int = 6,
        focus: str | None = None,
    ) -> tuple[list[ChatMessage], dict[str, int]]:
        """Summarize older messages via LLM (or structured fallback) while keeping recent tail."""
        before_tokens = sum(estimate_message_tokens(m) for m in messages)
        if len(messages) <= keep_recent:
            return list(messages), {
                "before_tokens": before_tokens,
                "after_tokens": before_tokens,
                "removed_messages": 0,
                "saved_tokens": 0,
            }

        older = messages[:-keep_recent]
        recent = list(messages[-keep_recent:])
        while recent and recent[0].role == Role.TOOL and older:
            recent.insert(0, older.pop())

        if not older:
            return list(messages), {
                "before_tokens": before_tokens,
                "after_tokens": before_tokens,
                "removed_messages": 0,
                "saved_tokens": 0,
            }

        transcript_lines = []
        for m in older:
            body = (m.content or "").strip().replace("\n", " ")
            if body:
                transcript_lines.append(f"[{m.role.value}]: {body[:400]}")
        joined = "\n".join(transcript_lines[-40:])

        summary_text = ""
        if llm_client is not None:
            focus_clause = f" Pay special attention to: {focus}." if focus else ""
            prompt = (
                "Summarize the earlier conversation concisely into bullet points preserving key facts, "
                "file paths, decisions, errors, and open tasks."
                f"{focus_clause}\n\nTranscript:\n{joined}"
            )
            try:
                reply = await llm_client.chat_completion(
                    messages=[
                        ChatMessage(role=Role.SYSTEM, content="You create concise technical session summaries."),
                        ChatMessage(role=Role.USER, content=prompt),
                    ],
                    stream=False,
                )
                summary_text = (getattr(reply, "content", "") or "").strip()
            except Exception:
                summary_text = ""

        if not summary_text:
            summary_text = "\n".join(transcript_lines[-8:])

        summary_msg = ChatMessage(
            role=Role.ASSISTANT,
            content=f"[Compacted {len(older)} earlier messages]\n{summary_text}",
        )
        compacted = [summary_msg, *recent]
        after_tokens = sum(estimate_message_tokens(m) for m in compacted)
        return compacted, {
            "before_tokens": before_tokens,
            "after_tokens": after_tokens,
            "removed_messages": len(older),
            "saved_tokens": max(0, before_tokens - after_tokens),
        }
