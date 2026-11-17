"""Dynamic discovery loader and management engine for skills."""

from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path
from typing import Any

from aether_agent.skills.format import (
    SkillDefinition,
    parse_skill_markdown,
    serialize_skill_markdown,
)

logger = logging.getLogger(__name__)


class SkillEngine:
    """Discovers, validates, reads, and manages skills under skills_dir."""

    def __init__(self, skills_dir: str = "skills") -> None:
        self.skills_dir = Path(skills_dir)
        self._skills: dict[str, SkillDefinition] = {}

    async def load_skills(self) -> list[SkillDefinition]:
        """Scan skills_dir/*/SKILL.md, skipping malformed files with a warning."""
        self._skills.clear()
        if not self.skills_dir.exists():
            return []

        for entry in sorted(self.skills_dir.iterdir()):
            if not entry.is_dir():
                continue
            skill_file = entry / "SKILL.md"
            if not skill_file.is_file():
                continue
            try:
                raw = skill_file.read_text(encoding="utf-8")
                skill = parse_skill_markdown(raw)
                skill.path = str(skill_file)
                self._skills[skill.name] = skill
            except Exception as exc:
                logger.warning(
                    "Skipping malformed SKILL.md at %s: %s", skill_file, exc
                )

        return list(self._skills.values())

    def list_skills(self) -> list[dict[str, Any]]:
        """Return metadata summaries of loaded skills."""
        return [
            {
                "name": skill.name,
                "description": skill.description,
                "status": skill.status,
                "path": skill.path,
            }
            for skill in self._skills.values()
        ]

    def inspect_skill(self, name: str) -> SkillDefinition | None:
        """Return full SkillDefinition by name."""
        return self._skills.get(name)

    def read_skill(self, name: str) -> str | None:
        """Return instructions body for a loaded skill."""
        skill = self._skills.get(name)
        return skill.instructions if skill else None

    def save_skill(self, skill: SkillDefinition) -> Path:
        """Write a skill to skills_dir/<slug>/SKILL.md and register it."""
        slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", skill.name.strip()).strip("-.")
        if not slug or slug in (".", ".."):
            raise ValueError("Skill name must contain valid alphanumeric characters")

        base_resolved = self.skills_dir.resolve()
        target_dir = (self.skills_dir / slug).resolve()
        if target_dir.parent != base_resolved:
            raise ValueError(f"Invalid skill path traversal detected for {skill.name!r}")

        target_dir.mkdir(parents=True, exist_ok=True)
        skill_file = target_dir / "SKILL.md"
        skill_file.write_text(serialize_skill_markdown(skill), encoding="utf-8")
        skill.path = str(skill_file)
        self._skills[skill.name] = skill
        return skill_file

    def remove_skill(self, name: str) -> bool:
        """Remove a skill from disk and memory."""
        skill = self._skills.pop(name, None)
        if skill is None:
            return False
        if skill.path:
            skill_dir = Path(skill.path).parent
            if skill_dir.exists() and skill_dir.parent == self.skills_dir:
                shutil.rmtree(skill_dir, ignore_errors=True)
        return True

    def render_prompt_index(self) -> str:
        """Render concise index of active skills for system prompt injection."""
        active = [s for s in self._skills.values() if s.status != "disabled"]
        if not active:
            return ""
        lines = ["Available Skills (use read_skill or follow instructions when relevant):"]
        for skill in active:
            lines.append(f"- {skill.name} [{skill.status}]: {skill.description}")
        return "\n".join(lines)
