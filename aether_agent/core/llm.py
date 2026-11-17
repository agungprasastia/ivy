"""Universal LLM client using native OpenAI and Anthropic SDKs."""

from __future__ import annotations

import json
import os
from typing import Any, AsyncIterator

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from aether_agent.config import AetherConfig, load_config
from aether_agent.core.types import ChatMessage, Role, ToolCall

_PROVIDER_BASE_URLS: dict[str, str] = {
    "openrouter": "https://openrouter.ai/api/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
    "ollama": "http://localhost:11434/v1",
    "vllm": "http://localhost:8000/v1",
}


class LLMClient:
    """Multi-provider asynchronous LLM client wrapping OpenAI and Anthropic SDKs."""

    def __init__(
        self,
        config: AetherConfig | None = None,
        model: str | None = None,
    ) -> None:
        self.config = config or load_config()
        self.model = model or self.config.default_model
        self.provider, self.resolved_model, self.base_url = self._resolve_provider(
            self.model
        )

    def set_model(self, model: str) -> None:
        """Switch active model and recompute provider routing."""
        self.model = model
        self.provider, self.resolved_model, self.base_url = self._resolve_provider(
            model
        )

    def _resolve_provider(self, model: str) -> tuple[str, str, str | None]:
        """Determine provider, normalized model name, and base URL from model string."""
        custom_base = self.config.effective_api_base
        if "/" in model:
            prefix, rest = model.split("/", 1)
            prefix_lower = prefix.lower()
            if prefix_lower in ("openai", "anthropic", "openrouter", "gemini", "ollama", "vllm"):
                base = custom_base or _PROVIDER_BASE_URLS.get(prefix_lower)
                return prefix_lower, rest, base

        if model.lower().startswith("claude"):
            return "anthropic", model, custom_base
        if model.lower().startswith("gemini"):
            return "gemini", model, custom_base or _PROVIDER_BASE_URLS["gemini"]
        return "openai", model, custom_base

    def _resolve_api_key(self, provider: str) -> str:
        """Resolve API key for the given provider from config or environment."""
        key_map = {
            "openai": self.config.openai_api_key or os.getenv("OPENAI_API_KEY"),
            "anthropic": self.config.anthropic_api_key or os.getenv("ANTHROPIC_API_KEY"),
            "openrouter": self.config.openrouter_api_key or os.getenv("OPENROUTER_API_KEY"),
            "gemini": self.config.gemini_api_key or os.getenv("GEMINI_API_KEY"),
            "ollama": "ollama",
            "vllm": os.getenv("OPENAI_API_KEY") or "vllm",
        }
        return key_map.get(provider) or os.getenv("OPENAI_API_KEY") or "missing-api-key"

    def _get_openai_client(self) -> AsyncOpenAI:
        api_key = self._resolve_api_key(self.provider)
        kwargs: dict[str, Any] = {"api_key": api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        return AsyncOpenAI(**kwargs)

    def _get_anthropic_client(self) -> AsyncAnthropic:
        api_key = self._resolve_api_key("anthropic")
        kwargs: dict[str, Any] = {"api_key": api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        return AsyncAnthropic(**kwargs)

    def _format_openai_messages(
        self, messages: list[ChatMessage | dict[str, Any]]
    ) -> list[dict[str, Any]]:
        formatted: list[dict[str, Any]] = []
        for msg in messages:
            if isinstance(msg, ChatMessage):
                formatted.append(msg.to_llm_dict())
            elif isinstance(msg, dict):
                formatted.append(msg)
            else:
                formatted.append({"role": "user", "content": str(msg)})
        return formatted

    def _format_anthropic_payload(
        self,
        messages: list[ChatMessage | dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> tuple[str | None, list[dict[str, Any]], list[dict[str, Any]] | None]:
        system_parts: list[str] = []
        formatted: list[dict[str, Any]] = []

        for raw_msg in messages:
            msg = (
                raw_msg
                if isinstance(raw_msg, ChatMessage)
                else ChatMessage(**raw_msg)
                if isinstance(raw_msg, dict)
                else ChatMessage(role=Role.USER, content=str(raw_msg))
            )

            if msg.role == Role.SYSTEM:
                if msg.content:
                    system_parts.append(msg.content)
            elif msg.role == Role.TOOL:
                formatted.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": msg.tool_call_id or "",
                                "content": msg.content or "",
                            }
                        ],
                    }
                )
            elif msg.role == Role.ASSISTANT and msg.tool_calls:
                blocks: list[dict[str, Any]] = []
                if msg.content:
                    blocks.append({"type": "text", "text": msg.content})
                for tc in msg.tool_calls:
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.function_name,
                            "input": tc.arguments,
                        }
                    )
                formatted.append({"role": "assistant", "content": blocks})
            else:
                formatted.append(
                    {"role": msg.role.value, "content": msg.content or ""}
                )

        anthropic_tools: list[dict[str, Any]] | None = None
        if tools:
            anthropic_tools = []
            for tool in tools:
                fn = tool.get("function", tool)
                anthropic_tools.append(
                    {
                        "name": fn["name"],
                        "description": fn.get("description", ""),
                        "input_schema": fn.get(
                            "parameters", {"type": "object", "properties": {}}
                        ),
                    }
                )

        system_prompt = "\n\n".join(system_parts) if system_parts else None
        return system_prompt, formatted, anthropic_tools

    async def chat_completion(
        self,
        messages: list[ChatMessage | dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> ChatMessage | AsyncIterator[str]:
        """Execute a chat completion request across OpenAI-compatible or Anthropic providers."""
        if self.provider == "anthropic":
            return await self._anthropic_completion(
                messages=messages,
                tools=tools,
                stream=stream,
                temperature=temperature,
                **kwargs,
            )

        formatted_messages = self._format_openai_messages(messages)
        params: dict[str, Any] = {
            "model": self.resolved_model,
            "messages": formatted_messages,
            "stream": stream,
            **kwargs,
        }
        if temperature is not None:
            params["temperature"] = temperature
        if tools:
            params["tools"] = tools
            params["tool_choice"] = kwargs.get("tool_choice", "auto")

        client = self._get_openai_client()
        if stream:
            return self._stream_openai(client, params)

        response = await client.chat.completions.create(**params)
        return self._parse_openai_response(response)

    async def _stream_openai(
        self, client: AsyncOpenAI, params: dict[str, Any]
    ) -> AsyncIterator[str]:
        response = await client.chat.completions.create(**params)
        async for chunk in response:
            choices = getattr(chunk, "choices", None)
            if choices is None and isinstance(chunk, dict):
                choices = chunk.get("choices", [])
            if not choices:
                continue
            choice = choices[0]
            delta_obj = getattr(choice, "delta", None)
            if delta_obj is None and isinstance(choice, dict):
                delta_obj = choice.get("delta", {})
            content = (
                getattr(delta_obj, "content", None)
                if not isinstance(delta_obj, dict)
                else delta_obj.get("content")
            )
            if content:
                yield content

    async def _anthropic_completion(
        self,
        messages: list[ChatMessage | dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        stream: bool,
        temperature: float | None,
        **kwargs: Any,
    ) -> ChatMessage | AsyncIterator[str]:
        system_prompt, formatted, anthropic_tools = self._format_anthropic_payload(
            messages, tools
        )
        params: dict[str, Any] = {
            "model": self.resolved_model,
            "messages": formatted,
            "max_tokens": kwargs.pop("max_tokens", 4096),
            **kwargs,
        }
        if system_prompt:
            params["system"] = system_prompt
        if temperature is not None:
            params["temperature"] = temperature
        if anthropic_tools:
            params["tools"] = anthropic_tools

        client = self._get_anthropic_client()
        if stream:
            return self._stream_anthropic(client, params)

        response = await client.messages.create(**params)
        return self._parse_anthropic_response(response)

    async def _stream_anthropic(
        self, client: AsyncAnthropic, params: dict[str, Any]
    ) -> AsyncIterator[str]:
        params["stream"] = True
        response = await client.messages.create(**params)
        async for event in response:
            event_type = getattr(event, "type", None) or (
                event.get("type") if isinstance(event, dict) else None
            )
            if event_type == "content_block_delta":
                delta = getattr(event, "delta", None) or (
                    event.get("delta", {}) if isinstance(event, dict) else {}
                )
                text = getattr(delta, "text", None) or (
                    delta.get("text") if isinstance(delta, dict) else None
                )
                if text:
                    yield text

    def _parse_openai_response(self, response: Any) -> ChatMessage:
        """Parse OpenAI SDK completion response into ChatMessage."""
        choices = getattr(response, "choices", None)
        if choices is None and isinstance(response, dict):
            choices = response.get("choices", [])

        if not choices:
            return ChatMessage(role=Role.ASSISTANT, content="")

        first_choice = choices[0]
        msg_obj = getattr(first_choice, "message", None)
        if msg_obj is None and isinstance(first_choice, dict):
            msg_obj = first_choice.get("message", {})

        content = getattr(msg_obj, "content", None)
        if content is None and isinstance(msg_obj, dict):
            content = msg_obj.get("content", "")

        raw_tool_calls = getattr(msg_obj, "tool_calls", None)
        if raw_tool_calls is None and isinstance(msg_obj, dict):
            raw_tool_calls = msg_obj.get("tool_calls", None)

        parsed_tool_calls: list[ToolCall] | None = None
        if raw_tool_calls:
            parsed_tool_calls = []
            for tc in raw_tool_calls:
                tc_id = getattr(tc, "id", None) or (
                    tc.get("id") if isinstance(tc, dict) else ""
                )
                fn = getattr(tc, "function", None) or (
                    tc.get("function") if isinstance(tc, dict) else {}
                )
                fn_name = getattr(fn, "name", None) or (
                    fn.get("name") if isinstance(fn, dict) else ""
                )
                fn_args = getattr(fn, "arguments", None) or (
                    fn.get("arguments") if isinstance(fn, dict) else {}
                )

                if isinstance(fn_args, str):
                    try:
                        fn_args = json.loads(fn_args)
                    except Exception:
                        fn_args = {"raw_input": fn_args}

                parsed_tool_calls.append(
                    ToolCall(
                        id=str(tc_id),
                        function_name=str(fn_name),
                        arguments=fn_args if isinstance(fn_args, dict) else {},
                    )
                )

        return ChatMessage(
            role=Role.ASSISTANT,
            content=content or "",
            tool_calls=parsed_tool_calls if parsed_tool_calls else None,
        )

    def _parse_anthropic_response(self, response: Any) -> ChatMessage:
        """Parse Anthropic Messages API response into ChatMessage."""
        blocks = getattr(response, "content", None)
        if blocks is None and isinstance(response, dict):
            blocks = response.get("content", [])

        text_parts: list[str] = []
        tool_calls: list[ToolCall] = []

        for block in blocks or []:
            btype = getattr(block, "type", None) or (
                block.get("type") if isinstance(block, dict) else None
            )
            if btype == "text":
                text = getattr(block, "text", None) or (
                    block.get("text") if isinstance(block, dict) else ""
                )
                if text:
                    text_parts.append(text)
            elif btype == "tool_use":
                tc_id = getattr(block, "id", None) or (
                    block.get("id") if isinstance(block, dict) else ""
                )
                tc_name = getattr(block, "name", None) or (
                    block.get("name") if isinstance(block, dict) else ""
                )
                tc_input = getattr(block, "input", None) or (
                    block.get("input") if isinstance(block, dict) else {}
                )
                tool_calls.append(
                    ToolCall(
                        id=str(tc_id),
                        function_name=str(tc_name),
                        arguments=tc_input if isinstance(tc_input, dict) else {},
                    )
                )

        return ChatMessage(
            role=Role.ASSISTANT,
            content="\n".join(text_parts),
            tool_calls=tool_calls if tool_calls else None,
        )
