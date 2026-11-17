from unittest.mock import AsyncMock, MagicMock

import pytest
from ivy.config import IvyConfig
from ivy.core.agent import IvyAgent
from ivy.core.types import ChatMessage, Role, ToolCall
from ivy.gateway.server import GatewayServer
from ivy.memory.store import MemoryStore
from ivy.skills.engine import SkillEngine
from ivy.skills.generator import SkillGenerator
from ivy.tools.approval import SafetyApprovalPolicy
from ivy.tools.files import FileTools
from ivy.tools.registry import ToolRegistry
from ivy.tools.runner import CodeRunnerTool
from ivy.tools.terminal import TerminalTool


@pytest.mark.asyncio
async def test_full_agent_workflow(tmp_path):
    db_file = str(tmp_path / "ivy.db")
    skills_dir = tmp_path / "skills"
    sample_dir = skills_dir / "coding-assistant"
    sample_dir.mkdir(parents=True)
    (sample_dir / "SKILL.md").write_text(
        "---\nname: coding-assistant\ndescription: Code workflow\nstatus: active\n---\n1. Read\n2. Verify\n",
        encoding="utf-8",
    )

    cfg = IvyConfig(
        db_path=db_file,
        skills_dir=str(skills_dir),
        allowed_users=["owner_1"],
    )

    mem_store = MemoryStore(db_path=db_file)
    await mem_store.initialize()

    tools = ToolRegistry()
    tools.register(TerminalTool(approval_policy=SafetyApprovalPolicy(auto_approve=True)))
    FileTools().register_all(tools)
    tools.register(CodeRunnerTool(timeout=5.0))

    skills = SkillEngine(skills_dir=str(skills_dir))
    await skills.load_skills()
    assert len(skills.list_skills()) == 1

    mock_llm = MagicMock()
    mock_llm.chat_completion = AsyncMock(
        side_effect=[
            ChatMessage(
                role=Role.ASSISTANT,
                content="",
                tool_calls=[
                    ToolCall(
                        id="call_py_1",
                        function_name="code_runner",
                        arguments={"code": "print(6 * 7)"},
                    )
                ],
            ),
            ChatMessage(role=Role.ASSISTANT, content="Hasil eksekusi Python adalah 42."),
            ChatMessage(
                role=Role.ASSISTANT,
                content=(
                    "---\n"
                    "name: python-calc\n"
                    "description: Compute arithmetic in Python sandbox\n"
                    "status: draft\n"
                    "---\n"
                    "Use code_runner to evaluate Python expressions.\n"
                ),
            ),
        ]
    )

    agent = IvyAgent(
        llm_client=mock_llm,
        tool_registry=tools,
        memory_store=mem_store,
        skill_engine=skills,
        config=cfg,
    )

    # Execute through GatewayServer with authorized user & shared session continuity
    gateway = GatewayServer(agent=agent, config=cfg)
    reply_text = await gateway.handle_incoming(
        platform="telegram",
        user_id="owner_1",
        text="Hitung 6 * 7 pakai Python",
        session_id="shared-session",
    )
    assert "42" in (reply_text or "")

    # FTS5 cross-session recall finds persisted conversation
    hits = await mem_store.search_fts("Python")
    assert len(hits) >= 1

    # Closed learning loop reflects and saves a new draft skill
    history = await mem_store.get_messages("shared-session")
    generator = SkillGenerator(llm_client=mock_llm, skill_engine=skills)
    new_skill = await generator.reflect_and_create(history)
    assert new_skill is not None
    assert new_skill.name == "python-calc"
    assert (skills_dir / "python-calc" / "SKILL.md").exists()

    await mem_store.close()
