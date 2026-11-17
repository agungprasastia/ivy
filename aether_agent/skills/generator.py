"""Automatic reflection and skill synthesis for the closed learning loop."""

from __future__ import annotations

import logging
import re

from aether_agent.core.llm import LLMClient
from aether_agent.core.types import ChatMessage, Role
from aether_agent.skills.engine import SkillEngine
from aether_agent.skills.format import SkillDefinition, parse_skill_markdown

logger = logging.getLogger(__name__)

_REFLECTION_PROMPT = """Review the completed conversation below.
If it solved a non-trivial reusable workflow, produce ONE concise agentskills.io SKILL.md document with YAML frontmatter (`name`, `description`, `status: draft`) and step-by-step instructions.
If the task was trivial and not reusable, reply with exact text: NO_SKILL.
Output ONLY the SKILL.md content (no extra commentary)."""


class SkillGenerator:
    """Synthesizes reusable SKILL.md files from completed agent workflows."""

    def __init__(self, llm_client: LLMClient, skill_engine: SkillEngine) -> None:
        self.llm_client = llm_client
        self.skill_engine = skill_engine

    async def reflect_and_create(
        self, messages: list[ChatMessage]
    ) -> SkillDefinition | None:
        """Reflect on a completed conversation and persist a new skill if worthwhile."""
        transcript_lines: list[str] = []
        for msg in messages[-20:]:
            if msg.role == Role.SYSTEM:
                continue
            content = (msg.content or "").strip()
            if content:
                transcript_lines.append(f"{msg.role.value.upper()}: {content[:600]}")

        if not transcript_lines:
            return None

        prompt_msgs = [
            ChatMessage(role=Role.SYSTEM, content=_REFLECTION_PROMPT),
            ChatMessage(role=Role.USER, content="\n".join(transcript_lines)),
        ]
        response = await self.llm_client.chat_completion(messages=prompt_msgs)
        if not isinstance(response, ChatMessage):
            return None

        raw = (response.content or "").strip()
        if not raw or raw == "NO_SKILL":
            return None

        # Strip optional markdown code fences around generated SKILL.md
        fenced = re.match(r"^```(?:markdown|md|yaml)?\n(.*?)\n```$", raw, re.DOTALL)
        if fenced:
            raw = fenced.group(1).strip()

        try:
            skill = parse_skill_markdown(raw)
        except Exception as exc:
            logger.warning("Skill reflection produced malformed SKILL.md: %s", exc)
            return None

        self.skill_engine.save_skill(skill)
        return skill
