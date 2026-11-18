"""Structured user clarification tool supporting multiple-choice & open-ended prompts."""

from __future__ import annotations

import contextlib
import contextvars
import inspect
import json
from typing import Any, AsyncIterator, Awaitable, Callable

from ivy.core.types import ToolResult
from ivy.tools.base import BaseTool

MAX_QUESTIONS = 4
MAX_CHOICES = 4
RECOMMENDED_LABEL = "(Recommended)"

_ClarifyCallback = Callable[[str, list[str], bool], Any | Awaitable[Any]]
_clarify_callback_var: contextvars.ContextVar[_ClarifyCallback | None] = (
    contextvars.ContextVar("ivy_scoped_clarify_callback", default=None)
)


def strip_recommended(text: str) -> str:
    """Remove trailing '(Recommended)' marker case-insensitively."""
    stripped = str(text).strip()
    if stripped.casefold().endswith(RECOMMENDED_LABEL.casefold()):
        return stripped[: -len(RECOMMENDED_LABEL)].strip()
    return stripped


def mark_recommended(choices: list[str]) -> list[str]:
    """Append '(Recommended)' to the first choice when >=2 choices are offered."""
    cleaned = [str(c).strip() for c in choices if str(c).strip()]
    if len(cleaned) < 2:
        return cleaned
    first = cleaned[0]
    if first != strip_recommended(first):
        return cleaned
    return [f"{first} {RECOMMENDED_LABEL}", *cleaned[1:]]


class ClarifyTool(BaseTool):
    """Tool to ask the user 1–4 structured multiple-choice or free-text questions."""

    name = "clarify"
    description = (
        "Ask the user 1 to 4 structured multiple-choice or open-ended clarifying "
        "questions when requirements are genuinely ambiguous."
    )
    parameters = {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "description": "List of 1-4 questions to ask the user.",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "choices": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Up to 4 choices (put recommended option first).",
                        },
                        "multi_select": {
                            "type": "boolean",
                            "default": False,
                        },
                    },
                    "required": ["question"],
                },
            }
        },
        "required": ["questions"],
    }

    def __init__(self, clarify_callback: _ClarifyCallback | None = None) -> None:
        self.clarify_callback = clarify_callback

    @contextlib.asynccontextmanager
    async def scoped_context(
        self, clarify_callback: _ClarifyCallback | None
    ) -> AsyncIterator[None]:
        """Temporarily bind a UI-specific clarify callback for the current async task."""
        tok = _clarify_callback_var.set(clarify_callback)
        try:
            yield
        finally:
            _clarify_callback_var.reset(tok)

    @staticmethod
    def _clean_response(raw: Any, multi_select: bool) -> Any:
        if multi_select:
            if isinstance(raw, (list, tuple, set)):
                return [strip_recommended(x) for x in raw if str(x).strip()]
            text = strip_recommended(str(raw or ""))
            return [text] if text else []
        if isinstance(raw, (list, tuple)):
            return strip_recommended(str(raw[0])) if raw else ""
        return strip_recommended(str(raw or ""))

    async def execute(
        self, questions: list[dict[str, Any]] | None = None, **kwargs: Any
    ) -> ToolResult:
        if not questions or not isinstance(questions, list):
            return ToolResult(
                tool_name=self.name,
                status="error",
                error="Invalid arguments: provide 1 to 4 question objects in 'questions'.",
            )
        if len(questions) > MAX_QUESTIONS:
            return ToolResult(
                tool_name=self.name,
                status="error",
                error=f"Too many questions ({len(questions)}); maximum is {MAX_QUESTIONS}.",
            )

        callback = _clarify_callback_var.get() or self.clarify_callback
        if callback is None:
            return ToolResult(
                tool_name=self.name,
                status="error",
                error="Clarify tool is not available in unattended mode.",
            )

        answers: list[dict[str, Any]] = []
        for idx, q in enumerate(questions, start=1):
            if not isinstance(q, dict) or not str(q.get("question") or "").strip():
                return ToolResult(
                    tool_name=self.name,
                    status="error",
                    error=f"Question #{idx} is missing a 'question' string.",
                )
            q_text = str(q["question"]).strip()
            raw_choices = q.get("choices") or []
            display_choices = mark_recommended(list(raw_choices)[:MAX_CHOICES])
            multi_select = bool(q.get("multi_select", False))
            try:
                res = callback(q_text, display_choices, multi_select)
                if inspect.isawaitable(res):
                    res = await res
            except Exception as exc:
                return ToolResult(
                    tool_name=self.name,
                    status="error",
                    error=f"User clarification failed: {exc}",
                )
            answers.append(
                {
                    "question": q_text,
                    "answer": self._clean_response(res, multi_select),
                }
            )

        return ToolResult(
            tool_name=self.name,
            status="success",
            output=json.dumps(answers, ensure_ascii=False, indent=2),
            metadata={"answers": answers},
        )
