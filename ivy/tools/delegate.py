"""Subagent delegation tool wrapping SubagentRunner for single & parallel subtasks."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Callable

from ivy.core.types import ToolResult
from ivy.tools.base import BaseTool

if TYPE_CHECKING:
    from ivy.core.subagent import SubagentRunner

MAX_BATCH_TASKS = 4
_BLOCKED_CHILD_TOOLS = {"delegate_task", "clarify", "process"}


class DelegateTaskTool(BaseTool):
    """Delegate independent subtasks to isolated SubagentRunner instances."""

    name = "delegate_task"
    description = (
        "Delegate one independent subtask ('task') or up to 4 parallel subtasks ('tasks') "
        "to isolated child subagents with their own context window, returning concise findings."
    )
    parameters = {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "Single subtask prompt for one isolated child subagent.",
            },
            "tasks": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Batch of up to 4 independent subtask prompts to run in parallel.",
            },
            "allowed_tools": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional whitelist of tool names for child subagents.",
            },
        },
        "required": [],
    }

    def __init__(
        self,
        runner: SubagentRunner,
        session_id_getter: Callable[[], str] | None = None,
    ) -> None:
        self.runner = runner
        self.session_id_getter = session_id_getter

    def _safe_child_tools(self, allowed_tools: list[str] | None) -> list[str]:
        if allowed_tools is not None:
            return [name for name in allowed_tools if name not in _BLOCKED_CHILD_TOOLS]
        return [
            tool.name
            for tool in self.runner.tool_registry.list_tools()
            if tool.name not in _BLOCKED_CHILD_TOOLS
        ]

    async def execute(
        self,
        task: str | None = None,
        tasks: list[str] | None = None,
        allowed_tools: list[str] | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        batch: list[str] = []
        if tasks:
            batch = [str(t).strip() for t in tasks if str(t).strip()]
        elif task and str(task).strip():
            batch = [str(task).strip()]

        if not batch:
            return ToolResult(
                tool_name=self.name,
                status="error",
                error="Provide 'task' (string) or 'tasks' (non-empty list of strings).",
            )
        if len(batch) > MAX_BATCH_TASKS:
            return ToolResult(
                tool_name=self.name,
                status="error",
                error=f"Too many parallel subtasks ({len(batch)}); maximum is {MAX_BATCH_TASKS}.",
            )

        parent_session = str(
            kwargs.get("session_id")
            or (self.session_id_getter() if self.session_id_getter else None)
            or "cli:main"
        )
        child_tools = self._safe_child_tools(allowed_tools)

        try:
            if len(batch) == 1:
                summary = await self.runner.run(
                    task_prompt=batch[0],
                    parent_session_id=parent_session,
                    allowed_tools=child_tools,
                )
                return ToolResult(
                    tool_name=self.name,
                    status="success",
                    output=summary or "(subagent returned empty result)",
                    metadata={"task_count": 1},
                )

            results = await asyncio.gather(
                *(
                    self.runner.run(
                        task_prompt=subtask,
                        parent_session_id=parent_session,
                        allowed_tools=child_tools,
                    )
                    for subtask in batch
                )
            )
            sections = [
                f"### Subtask {idx}: {subtask[:80]}\n{res or '(empty)'}"
                for idx, (subtask, res) in enumerate(zip(batch, results), start=1)
            ]
            return ToolResult(
                tool_name=self.name,
                status="success",
                output="\n\n".join(sections),
                metadata={"task_count": len(batch)},
            )
        except Exception as exc:
            return ToolResult(
                tool_name=self.name,
                status="error",
                error=f"Delegated task failed: {exc}",
            )
