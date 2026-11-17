import logging
from unittest.mock import AsyncMock, MagicMock

import pytest
from aether_agent.core.types import ChatMessage, Role
from aether_agent.skills.engine import SkillEngine
from aether_agent.skills.format import (
    SkillDefinition,
    parse_skill_markdown,
    serialize_skill_markdown,
)
from aether_agent.skills.generator import SkillGenerator


def test_parse_agentskills_markdown():
    skill_content = """---
name: sample-analyzer
description: Menganalisa file teks lokal
status: active
---
# Prosedur
1. Baca file
2. Berikan ringkasan
"""
    skill = parse_skill_markdown(skill_content)
    assert skill.name == "sample-analyzer"
    assert skill.description == "Menganalisa file teks lokal"
    assert skill.status == "active"
    assert "Prosedur" in skill.instructions

    serialized = serialize_skill_markdown(skill)
    reparsed = parse_skill_markdown(serialized)
    assert reparsed.name == skill.name
    assert reparsed.description == skill.description


@pytest.mark.asyncio
async def test_skill_engine_discovery_and_crud(tmp_path):
    skills_dir = tmp_path / "skills"
    skill_folder = skills_dir / "sample"
    skill_folder.mkdir(parents=True)
    (skill_folder / "SKILL.md").write_text(
        "---\nname: sample\ndescription: Test skill\n---\nSteps to follow",
        encoding="utf-8",
    )

    engine = SkillEngine(skills_dir=str(skills_dir))
    await engine.load_skills()
    skills = engine.list_skills()
    assert len(skills) == 1
    assert skills[0]["name"] == "sample"
    assert "Steps to follow" in (engine.read_skill("sample") or "")
    assert engine.inspect_skill("sample") is not None
    assert engine.remove_skill("sample") is True
    assert engine.list_skills() == []


@pytest.mark.asyncio
async def test_skill_engine_skips_malformed_skill_with_warning(tmp_path, caplog):
    skills_dir = tmp_path / "skills"
    bad_folder = skills_dir / "broken"
    bad_folder.mkdir(parents=True)
    (bad_folder / "SKILL.md").write_text(
        "This has no frontmatter at all", encoding="utf-8"
    )

    good_folder = skills_dir / "valid"
    good_folder.mkdir(parents=True)
    (good_folder / "SKILL.md").write_text(
        "---\nname: valid\ndescription: Valid skill\n---\nDo work",
        encoding="utf-8",
    )

    engine = SkillEngine(skills_dir=str(skills_dir))
    with caplog.at_level(logging.WARNING):
        await engine.load_skills()

    skills = engine.list_skills()
    assert len(skills) == 1
    assert skills[0]["name"] == "valid"
    assert any("broken" in rec.message or "SKILL.md" in rec.message for rec in caplog.records)


@pytest.mark.asyncio
async def test_skill_generator_creates_skill_file(tmp_path):
    skills_dir = tmp_path / "skills"
    engine = SkillEngine(skills_dir=str(skills_dir))

    mock_llm = MagicMock()
    mock_llm.chat_completion = AsyncMock(
        return_value=ChatMessage(
            role=Role.ASSISTANT,
            content=(
                "---\n"
                "name: auto-uv-check\n"
                "description: Verify uv environment health\n"
                "status: draft\n"
                "---\n"
                "# Steps\n1. Run uv --version\n2. Run pytest\n"
            ),
        )
    )

    generator = SkillGenerator(llm_client=mock_llm, skill_engine=engine)
    messages = [
        ChatMessage(role=Role.USER, content="Check uv and run tests"),
        ChatMessage(role=Role.ASSISTANT, content="Done, all tests pass."),
    ]
    created = await generator.reflect_and_create(messages=messages)
    assert isinstance(created, SkillDefinition)
    assert created.name == "auto-uv-check"
    assert (skills_dir / "auto-uv-check" / "SKILL.md").exists()
    assert engine.inspect_skill("auto-uv-check") is not None
