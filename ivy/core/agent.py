"""ReAct autonomous agent loop and orchestrator."""

from __future__ import annotations

import asyncio
import inspect
import time
from collections import defaultdict
from typing import Any, Awaitable, Callable

from ivy.config import IvyConfig
from ivy.core.llm import LLMClient
from ivy.core.prompt import build_system_prompt, format_tool_result
from ivy.core.types import ChatMessage, Role
from ivy.memory.context import ContextWindowManager, should_auto_compact
from ivy.memory.store import MemoryStore
from ivy.skills.engine import SkillEngine
from ivy.skills.generator import SkillGenerator
from ivy.tools.registry import ToolRegistry


class IvyAgent:
    """Autonomous ReAct agent coordinating LLM reasoning, tools, memory, and skills."""

    def __init__(
        self,
        llm_client: LLMClient,
        tool_registry: ToolRegistry,
        memory_store: MemoryStore | None = None,
        skill_engine: SkillEngine | None = None,
        skill_generator: SkillGenerator | None = None,
        config: IvyConfig | None = None,
        max_iterations: int | None = None,
        exclude_from_fts: bool = False,
        extra_system_instructions: str | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.tool_registry = tool_registry
        self.memory_store = memory_store
        self.skill_engine = skill_engine
        self.skill_generator = skill_generator
        self.config = config
        self.max_iterations = max_iterations if max_iterations is not None else (
            config.max_iterations if config else 50
        )
        self.exclude_from_fts = exclude_from_fts
        self.extra_system_instructions = extra_system_instructions
        self._in_memory_sessions: dict[str, list[ChatMessage]] = defaultdict(list)
        self._fallback_context_mgr = ContextWindowManager()

    async def _load_history(self, session_id: str) -> list[ChatMessage]:
        if self.memory_store is not None:
            return await self.memory_store.get_messages(session_id)
        return list(self._in_memory_sessions[session_id])

    async def _persist_message(
        self, session_id: str, message: ChatMessage, platform: str = "cli"
    ) -> None:
        if self.memory_store is not None:
            await self.memory_store.save_message(
                session_id,
                message,
                platform=platform,
                exclude_from_fts=self.exclude_from_fts,
            )
        else:
            self._in_memory_sessions[session_id].append(message)

    async def _call_model(
        self,
        messages: list[ChatMessage],
        tools: list[dict[str, Any]] | None,
        stream: bool,
        stream_callback: Callable[[str], Any | Awaitable[Any]] | None,
    ) -> tuple[ChatMessage, bool]:
        """Invoke LLM with streaming when enabled, returning (reply, did_stream)."""
        if stream and isinstance(self.llm_client, LLMClient):
            reply = await self.llm_client.stream_turn(
                messages=messages,
                tools=tools,
                on_token=stream_callback,
            )
            return reply, True
        reply = await self.llm_client.chat_completion(
            messages=messages,
            tools=tools,
            stream=False,
        )
        if not isinstance(reply, ChatMessage):
            text = "".join([chunk async for chunk in reply])
            reply = ChatMessage(role=Role.ASSISTANT, content=text)
        return reply, False

    async def _prepare_context_messages(
        self,
        system_prompt: str,
        history: list[ChatMessage],
        user_msg: ChatMessage,
        schemas: list[dict[str, Any]] | None,
        model_name: str | None,
    ) -> tuple[list[ChatMessage], dict[str, int] | None]:
        """Build model messages, auto-compacting via LLM if estimated usage exceeds threshold."""
        non_system = [*history, user_msg]
        threshold = (
            float(getattr(self.config, "auto_compact_threshold", 0.85))
            if self.config is not None
            else 0.85
        )
        auto_compacted: dict[str, int] | None = None
        if should_auto_compact(
            non_system,
            model=model_name,
            tool_schemas=schemas,
            system_prompt=system_prompt,
            threshold=threshold,
        ):
            compacted_non_sys, stats = await self._fallback_context_mgr.compact_with_llm(
                non_system,
                llm_client=self.llm_client,
                keep_recent=8,
            )
            messages = [ChatMessage(role=Role.SYSTEM, content=system_prompt), *compacted_non_sys]
            auto_compacted = stats
        else:
            messages = [ChatMessage(role=Role.SYSTEM, content=system_prompt), *non_system]

        if self.memory_store is not None:
            messages = self.memory_store.compress_context(messages)
        else:
            messages = self._fallback_context_mgr.compress(messages)
        return messages, auto_compacted

    @staticmethod
    def _accumulate_usage(turn_usage: dict[str, int], reply: ChatMessage) -> None:
        turn_usage["api_calls"] += 1
        msg_usage = reply.metadata.get("usage") if isinstance(reply.metadata, dict) else None
        if isinstance(msg_usage, dict):
            turn_usage["prompt_tokens"] += int(msg_usage.get("prompt_tokens", 0) or 0)
            turn_usage["completion_tokens"] += int(msg_usage.get("completion_tokens", 0) or 0)
            turn_usage["total_tokens"] += int(msg_usage.get("total_tokens", 0) or 0)

    def _is_batch_concurrent_safe(self, tool_calls: list[Any]) -> bool:
        """True when 2+ tool calls in one batch are all marked concurrent_safe."""
        if len(tool_calls) < 2 or not hasattr(self.tool_registry, "get"):
            return False
        for tc in tool_calls:
            tool_obj = self.tool_registry.get(tc.function_name)
            if tool_obj is None or not getattr(tool_obj, "concurrent_safe", False):
                return False
        return True

    async def _emit(
        self, callback: Callable[[Any], Any] | None, payload: Any
    ) -> None:
        if callback is None:
            return
        res = callback(payload)
        if inspect.isawaitable(res):
            await res

    async def run_turn(
        self,
        prompt: str,
        session_id: str = "default",
        platform: str = "cli",
        stream: bool = False,
        stream_callback: Callable[[str], Any | Awaitable[Any]] | None = None,
        tool_callback: Callable[[dict[str, Any]], Any | Awaitable[Any]] | None = None,
    ) -> ChatMessage:
        """Execute one full user turn through the ReAct loop."""
        user_profile: dict[str, str] | None = None
        memory_notes: list[dict[str, Any]] | None = None
        if self.memory_store is not None:
            user_profile = await self.memory_store.get_user_profile()
            memory_notes = await self.memory_store.list_memory_notes(limit=50)

        raw_model = getattr(self.llm_client, "model", None)
        model_name = raw_model if isinstance(raw_model, str) else None
        tool_names = [s.get("function", {}).get("name", "") for s in (self.tool_registry.get_schemas() or [])]
        system_prompt = build_system_prompt(
            skill_engine=self.skill_engine,
            user_profile=user_profile,
            memory_notes=memory_notes,
            extra_instructions=self.extra_system_instructions,
            model=model_name,
            platform=platform,
            session_id=session_id,
            tool_names=tool_names,
        )

        history = await self._load_history(session_id)
        user_msg = ChatMessage(role=Role.USER, content=prompt)
        raw_schemas = self.tool_registry.get_schemas()
        messages, auto_compacted = await self._prepare_context_messages(
            system_prompt=system_prompt,
            history=history,
            user_msg=user_msg,
            schemas=raw_schemas,
            model_name=model_name,
        )

        schemas = raw_schemas or None
        used_tools = False
        user_persisted = False
        turn_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "api_calls": 0}
        comp_usage = auto_compacted.get("compaction_usage") if isinstance(auto_compacted, dict) else None
        if isinstance(comp_usage, dict):
            turn_usage["api_calls"] += int(comp_usage.get("api_calls", 0) or 0)
            turn_usage["prompt_tokens"] += int(comp_usage.get("prompt_tokens", 0) or 0)
            turn_usage["completion_tokens"] += int(comp_usage.get("completion_tokens", 0) or 0)
            turn_usage["total_tokens"] += int(comp_usage.get("total_tokens", 0) or 0)
        tool_calls_count = 0

        for _ in range(self.max_iterations):
            reply, did_stream = await self._call_model(
                messages=messages,
                tools=schemas,
                stream=stream,
                stream_callback=stream_callback,
            )
            self._accumulate_usage(turn_usage, reply)

            if not user_persisted:
                await self._persist_message(session_id, user_msg, platform=platform)
                user_persisted = True
            await self._persist_message(session_id, reply, platform=platform)
            messages.append(reply)

            if not reply.tool_calls:
                if used_tools and self.skill_generator is not None and not self.exclude_from_fts:
                    try:
                        await self.skill_generator.reflect_and_create(messages)
                    except Exception:
                        pass
                reply.metadata = dict(reply.metadata or {})
                reply.metadata["turn_usage"] = turn_usage
                reply.metadata["tool_calls_count"] = tool_calls_count
                reply.metadata["context_messages"] = len(messages)
                if auto_compacted is not None:
                    reply.metadata["auto_compacted"] = auto_compacted
                if reply.content and not did_stream:
                    await self._emit(stream_callback, reply.content)
                return reply

            used_tools = True

            async def _run_one(tc_item: Any) -> tuple[Any, float]:
                t0 = time.monotonic()
                res = await self.tool_registry.execute(
                    tool_name=tc_item.function_name,
                    arguments=tc_item.arguments,
                    tool_call_id=tc_item.id,
                )
                return res, round(time.monotonic() - t0, 3)

            outcomes: list[tuple[Any, Any, float]] = []
            if self._is_batch_concurrent_safe(reply.tool_calls):
                for tc in reply.tool_calls:
                    tool_calls_count += 1
                    await self._emit(
                        tool_callback,
                        {"phase": "start", "name": tc.function_name, "arguments": tc.arguments},
                    )
                executed = await asyncio.gather(*(_run_one(tc) for tc in reply.tool_calls))
                outcomes = [
                    (tc, res, elapsed_s)
                    for tc, (res, elapsed_s) in zip(reply.tool_calls, executed)
                ]
            else:
                for tc in reply.tool_calls:
                    tool_calls_count += 1
                    await self._emit(
                        tool_callback,
                        {"phase": "start", "name": tc.function_name, "arguments": tc.arguments},
                    )
                    res, elapsed_s = await _run_one(tc)
                    outcomes.append((tc, res, elapsed_s))

            for tc, result, elapsed_s in outcomes:
                await self._emit(
                    tool_callback,
                    {
                        "phase": "end",
                        "name": tc.function_name,
                        "arguments": tc.arguments,
                        "status": result.status,
                        "output": result.output,
                        "error": result.error,
                        "duration_s": elapsed_s,
                        "metadata": result.metadata,
                    },
                )
                tool_msg = ChatMessage(
                    role=Role.TOOL,
                    name=tc.function_name,
                    tool_call_id=tc.id,
                    content=format_tool_result(result),
                )
                await self._persist_message(session_id, tool_msg, platform=platform)
                messages.append(tool_msg)

        # Graceful tool-free summary turn (Hermes pattern) when iteration budget is hit
        if used_tools:
            summary_nudge = ChatMessage(
                role=Role.USER,
                content=(
                    "[Iteration budget reached] Without calling any more tools, provide a concise "
                    "final summary of what was completed, files changed, checks run, and any remaining work."
                ),
            )
            try:
                summary_reply, did_stream = await self._call_model(
                    messages=[*messages, summary_nudge],
                    tools=None,
                    stream=stream,
                    stream_callback=stream_callback,
                )
                self._accumulate_usage(turn_usage, summary_reply)
                if (summary_reply.content or "").strip():
                    summary_reply.metadata = dict(summary_reply.metadata or {})
                    summary_reply.metadata["turn_usage"] = turn_usage
                    summary_reply.metadata["tool_calls_count"] = tool_calls_count
                    summary_reply.metadata["context_messages"] = len(messages)
                    if auto_compacted is not None:
                        summary_reply.metadata["auto_compacted"] = auto_compacted
                    await self._persist_message(session_id, summary_reply, platform=platform)
                    if not did_stream:
                        await self._emit(stream_callback, summary_reply.content)
                    return summary_reply
            except Exception:
                pass

        exhausted_meta: dict[str, Any] = {
            "turn_usage": turn_usage,
            "tool_calls_count": tool_calls_count,
            "context_messages": len(messages),
        }
        if auto_compacted is not None:
            exhausted_meta["auto_compacted"] = auto_compacted
        exhausted = ChatMessage(
            role=Role.ASSISTANT,
            content=f"Reached maximum tool iterations ({self.max_iterations}) without a final summary.",
            metadata=exhausted_meta,
        )
        await self._persist_message(session_id, exhausted, platform=platform)
        await self._emit(stream_callback, exhausted.content)
        return exhausted
