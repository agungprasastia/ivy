"""System prompt builder and tool observation formatters."""

from __future__ import annotations

import os
import platform
from typing import TYPE_CHECKING

from ivy.core.types import ToolResult

if TYPE_CHECKING:
    from ivy.skills.engine import SkillEngine

BASE_SYSTEM_PROMPT = """You are Ivy, an autonomous, helpful, and safety-conscious AI software engineering and productivity assistant.
You operate in a ReAct loop: analyze the request, invoke tools when needed, inspect results carefully, and synthesize a clear final answer.
Always ground file paths and project facts in the actual Working Directory and observed tool outputs; never guess external directories.
If a tool execution is rejected by the safety guard or returns an error, reflect on the reason and propose a safe alternative instead of retrying the exact blocked command."""


def build_system_prompt(
    skill_engine: SkillEngine | None = None,
    user_profile: dict[str, str] | None = None,
    extra_instructions: str | None = None,
) -> str:
    """Build dynamic system prompt including environment, user profile, and active skills."""
    sections = [
        BASE_SYSTEM_PROMPT,
        f"Runtime Environment: OS={platform.system()} ({platform.release()})\nWorking Directory: {os.getcwd()}",
    ]

    if user_profile:
        profile_lines = ["Known User Preferences:"]
        for k, v in sorted(user_profile.items()):
            profile_lines.append(f"- {k}: {v}")
        sections.append("\n".join(profile_lines))

    if skill_engine is not None:
        skills_index = skill_engine.render_prompt_index()
        if skills_index:
            sections.append(skills_index)

    if extra_instructions:
        sections.append(extra_instructions.strip())

    return "\n\n".join(sections)


def format_tool_result(result: ToolResult) -> str:
    """Format a ToolResult into an observation string for the LLM."""
    parts = [f"status={result.status}"]
    if result.output:
        parts.append(f"output:\n{result.output}")
    if result.error:
        parts.append(f"error:\n{result.error}")
    return "\n".join(parts)
