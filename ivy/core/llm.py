"""Universal LLM client using native OpenAI and Anthropic SDKs."""

from __future__ import annotations

import asyncio
import contextvars
import inspect
import json
import logging
import os
from typing import Any, AsyncIterator, Awaitable, Callable

from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

from ivy.config import IvyConfig, load_config
from ivy.core.types import ChatMessage, Role, ToolCall

logger = logging.getLogger(__name__)

_RETRYABLE_STATUS_CODES = {408, 429, 500, 502, 503, 504, 529}
_NON_RETRYABLE_STATUS_CODES = {400, 401, 403, 404, 422, 501, 505}
_RETRYABLE_CLASS_NAMES = {
    "RateLimitError",
    "APITimeoutError",
    "APIConnectionError",
    "InternalServerError",
    "ServiceUnavailableError",
    "OverloadedError",
}


def _is_retryable_error(exc: Exception) -> bool:
    """True when an LLM SDK exception represents a transient rate-limit, timeout, or 5xx error."""
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(exc, "status", None)
    if isinstance(status, int):
        if status in _NON_RETRYABLE_STATUS_CODES:
            return False
        if status in _RETRYABLE_STATUS_CODES or 500 <= status < 600:
            return True
    if isinstance(exc, (TimeoutError, asyncio.TimeoutError, ConnectionError)):
        return True
    return type(exc).__name__ in _RETRYABLE_CLASS_NAMES

def _extract_usage_dict(usage_obj: Any) -> dict[str, int] | None:
    """Normalize OpenAI/Anthropic usage objects or dicts into {prompt_tokens, completion_tokens, total_tokens}."""
    if usage_obj is None:
        return None
    if isinstance(usage_obj, dict):
        prompt = usage_obj.get("prompt_tokens") or usage_obj.get("input_tokens") or 0
        completion = usage_obj.get("completion_tokens") or usage_obj.get("output_tokens") or 0
        total = usage_obj.get("total_tokens") or (int(prompt) + int(completion))
    else:
        prompt = getattr(usage_obj, "prompt_tokens", None)
        if prompt is None:
            prompt = getattr(usage_obj, "input_tokens", 0) or 0
        completion = getattr(usage_obj, "completion_tokens", None)
        if completion is None:
            completion = getattr(usage_obj, "output_tokens", 0) or 0
        total = getattr(usage_obj, "total_tokens", None) or (int(prompt) + int(completion))
    prompt_i, completion_i, total_i = int(prompt or 0), int(completion or 0), int(total or 0)
    if prompt_i == 0 and completion_i == 0 and total_i == 0:
        return None
    return {
        "prompt_tokens": prompt_i,
        "completion_tokens": completion_i,
        "total_tokens": total_i,
    }


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
        config: IvyConfig | None = None,
        model: str | None = None,
    ) -> None:
        self.config = config or load_config()
        self.model = model or self.config.default_model
        self.provider, self.resolved_model, self.base_url = self._resolve_provider(
            self.model
        )
        self._openai_clients: dict[tuple[str, str | None], AsyncOpenAI] = {}
        self._anthropic_clients: dict[tuple[str, str | None], AsyncAnthropic] = {}
        self._route_override: contextvars.ContextVar[
            tuple[str, str, str, str | None] | None
        ] = contextvars.ContextVar(f"ivy_llm_route_{id(self)}", default=None)

    def _active_route(self) -> tuple[str, str, str, str | None]:
        """Return task-local (model, provider, resolved_model, base_url) without mutating shared state."""
        return self._route_override.get() or (
            self.model,
            self.provider,
            self.resolved_model,
            self.base_url,
        )

    def set_model(self, model: str) -> None:
        """Switch active model and recompute provider routing."""
        self.model = model
        self.provider, self.resolved_model, self.base_url = self._resolve_provider(
            model
        )

    def reconfigure(
        self,
        model: str | None = None,
        api_base: str | None = None,
        openai_api_key: str | None = None,
    ) -> None:
        """Update endpoint/key/model configuration at runtime."""
        if api_base is not None:
            self.config.api_base = api_base or None
        if openai_api_key is not None:
            self.config.openai_api_key = openai_api_key or None
        if model:
            self.model = model
        self.provider, self.resolved_model, self.base_url = self._resolve_provider(
            self.model
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
        _, active_provider, _, active_base = self._active_route()
        api_key = self._resolve_api_key(active_provider)
        cache_key = (api_key, active_base)
        client = self._openai_clients.get(cache_key)
        if client is None:
            kwargs: dict[str, Any] = {"api_key": api_key}
            if active_base:
                kwargs["base_url"] = active_base
            client = AsyncOpenAI(**kwargs)
            self._openai_clients[cache_key] = client
        return client

    def _get_anthropic_client(self) -> AsyncAnthropic:
        _, _, _, active_base = self._active_route()
        api_key = self._resolve_api_key("anthropic")
        cache_key = (api_key, active_base)
        client = self._anthropic_clients.get(cache_key)
        if client is None:
            kwargs: dict[str, Any] = {"api_key": api_key}
            if active_base:
                kwargs["base_url"] = active_base
            client = AsyncAnthropic(**kwargs)
            self._anthropic_clients[cache_key] = client
        return client

    async def close(self) -> None:
        """Close all cached HTTP SDK clients."""
        for client in list(self._openai_clients.values()) + list(self._anthropic_clients.values()):
            await client.close()
        self._openai_clients.clear()
        self._anthropic_clients.clear()

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
                tool_block = {
                    "type": "tool_result",
                    "tool_use_id": msg.tool_call_id or "",
                    "content": msg.content or "",
                }
                if (
                    formatted
                    and formatted[-1]["role"] == "user"
                    and isinstance(formatted[-1]["content"], list)
                ):
                    formatted[-1]["content"].append(tool_block)
                else:
                    formatted.append({"role": "user", "content": [tool_block]})
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
                if (
                    msg.role == Role.USER
                    and formatted
                    and formatted[-1]["role"] == "user"
                    and isinstance(formatted[-1]["content"], str)
                ):
                    formatted[-1]["content"] = (
                        f"{formatted[-1]['content']}\n\n{msg.content or ''}".strip()
                    )
                else:
                    formatted.append(
                        {"role": msg.role.value, "content": msg.content or ""}
                    )

        if formatted and formatted[0]["role"] != "user":
            formatted.insert(
                0, {"role": "user", "content": "[Continued conversation context]"}
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

    async def _execute_with_resilience(
        self,
        operation: Callable[[], Awaitable[ChatMessage]],
        *,
        can_retry: Callable[[], bool] | None = None,
    ) -> ChatMessage:
        """Run an async LLM operation with exponential backoff retries and optional fallback model."""
        max_retries = max(0, int(getattr(self.config, "llm_max_retries", 2)))
        base_delay = max(0.0, float(getattr(self.config, "llm_retry_base_delay", 0.5)))
        primary_model, _, _, _ = self._active_route()

        last_exc: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                return await operation()
            except Exception as exc:
                last_exc = exc
                allowed = can_retry() if can_retry is not None else True
                if not allowed or not _is_retryable_error(exc):
                    raise
                if attempt < max_retries:
                    delay = base_delay * (2**attempt)
                    logger.warning(
                        "Transient LLM error on %s (attempt %d/%d): %s; retrying in %.2fs",
                        primary_model,
                        attempt + 1,
                        max_retries + 1,
                        exc,
                        delay,
                    )
                    if delay > 0:
                        await asyncio.sleep(delay)

        fallback = (getattr(self.config, "fallback_model", None) or "").strip()
        if fallback and fallback != primary_model and (can_retry is None or can_retry()):
            logger.warning("Primary model %s exhausted retries; falling back to %s", primary_model, fallback)
            token = self._route_override.set((fallback, *self._resolve_provider(fallback)))
            try:
                reply = await operation()
            finally:
                self._route_override.reset(token)
            reply.metadata = dict(reply.metadata or {})
            reply.metadata["fallback_used"] = True
            reply.metadata["model_used"] = fallback
            return reply

        assert last_exc is not None
        raise last_exc

    async def chat_completion(
        self,
        messages: list[ChatMessage | dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        stream: bool = False,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> ChatMessage | AsyncIterator[str]:
        """Execute a chat completion request across OpenAI-compatible or Anthropic providers."""
        if stream:
            _, active_provider, active_resolved_model, _ = self._active_route()
            if active_provider == "anthropic":
                return await self._anthropic_completion(
                    messages=messages,
                    tools=tools,
                    stream=True,
                    temperature=temperature,
                    **kwargs,
                )
            formatted_messages = self._format_openai_messages(messages)
            params: dict[str, Any] = {
                "model": active_resolved_model,
                "messages": formatted_messages,
                "stream": True,
                **kwargs,
            }
            if temperature is not None:
                params["temperature"] = temperature
            if tools:
                params["tools"] = tools
                params["tool_choice"] = kwargs.get("tool_choice", "auto")
            return self._stream_openai(self._get_openai_client(), params)

        async def _once() -> ChatMessage:
            _, active_provider, active_resolved_model, _ = self._active_route()
            if active_provider == "anthropic":
                res = await self._anthropic_completion(
                    messages=messages,
                    tools=tools,
                    stream=False,
                    temperature=temperature,
                    **kwargs,
                )
                assert isinstance(res, ChatMessage)
                return res

            formatted_messages = self._format_openai_messages(messages)
            params: dict[str, Any] = {
                "model": active_resolved_model,
                "messages": formatted_messages,
                "stream": False,
                **kwargs,
            }
            if temperature is not None:
                params["temperature"] = temperature
            if tools:
                params["tools"] = tools
                params["tool_choice"] = kwargs.get("tool_choice", "auto")

            client = self._get_openai_client()
            response = await client.chat.completions.create(**params)
            return self._parse_openai_response(response)

        return await self._execute_with_resilience(_once)

    async def stream_turn(
        self,
        messages: list[ChatMessage | dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        on_token: Callable[[str], Any | Awaitable[Any]] | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> ChatMessage:
        """Stream response tokens via on_token while assembling text and tool calls."""
        emitted_any = False

        async def _emit(chunk_text: str) -> None:
            nonlocal emitted_any
            if not chunk_text:
                return
            emitted_any = True
            if on_token is None:
                return
            res = on_token(chunk_text)
            if inspect.isawaitable(res):
                await res

        async def _stream_once() -> ChatMessage:
            call_kwargs = dict(kwargs)
            _, active_provider, active_resolved_model, _ = self._active_route()
            if active_provider == "anthropic":
                system_prompt, formatted, anthropic_tools = self._format_anthropic_payload(
                    messages, tools
                )
                params: dict[str, Any] = {
                    "model": active_resolved_model,
                    "messages": formatted,
                    "max_tokens": call_kwargs.pop("max_tokens", 4096),
                    "stream": True,
                    **call_kwargs,
                }
                if system_prompt:
                    params["system"] = system_prompt
                if temperature is not None:
                    params["temperature"] = temperature
                if anthropic_tools:
                    params["tools"] = anthropic_tools

                client = self._get_anthropic_client()
                resp = await client.messages.create(**params)
                text_parts: list[str] = []
                tool_blocks: dict[int, dict[str, Any]] = {}
                usage_stats: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                async for event in resp:
                    etype = getattr(event, "type", None)
                    if etype == "message_start":
                        msg_obj = getattr(event, "message", None)
                        u = _extract_usage_dict(getattr(msg_obj, "usage", None))
                        if u:
                            usage_stats["prompt_tokens"] += u["prompt_tokens"]
                    elif etype == "message_delta":
                        u = _extract_usage_dict(getattr(event, "usage", None))
                        if u:
                            usage_stats["completion_tokens"] += u["completion_tokens"]
                    elif etype == "content_block_start":
                        idx = getattr(event, "index", 0)
                        blk = getattr(event, "content_block", None)
                        if getattr(blk, "type", None) == "tool_use":
                            tool_blocks[idx] = {
                                "id": getattr(blk, "id", ""),
                                "name": getattr(blk, "name", ""),
                                "raw_args": "",
                            }
                    elif etype == "content_block_delta":
                        idx = getattr(event, "index", 0)
                        delta = getattr(event, "delta", None)
                        dtype = getattr(delta, "type", None)
                        if dtype == "text_delta" or getattr(delta, "text", None):
                            t = getattr(delta, "text", "") or ""
                            if t:
                                text_parts.append(t)
                                await _emit(t)
                        elif dtype == "input_json_delta" and idx in tool_blocks:
                            tool_blocks[idx]["raw_args"] += getattr(delta, "partial_json", "") or ""

                parsed_calls: list[ToolCall] = []
                for idx in sorted(tool_blocks):
                    tb = tool_blocks[idx]
                    raw_json = tb["raw_args"].strip()
                    try:
                        args_dict = json.loads(raw_json) if raw_json else {}
                    except Exception:
                        args_dict = {"raw_input": raw_json}
                    parsed_calls.append(
                        ToolCall(
                            id=str(tb["id"]),
                            function_name=str(tb["name"]),
                            arguments=args_dict if isinstance(args_dict, dict) else {},
                        )
                    )
                usage_stats["total_tokens"] = usage_stats["prompt_tokens"] + usage_stats["completion_tokens"]
                meta = {"usage": usage_stats} if usage_stats["total_tokens"] > 0 else {}
                return ChatMessage(
                    role=Role.ASSISTANT,
                    content="".join(text_parts),
                    tool_calls=parsed_calls or None,
                    metadata=meta,
                )

            formatted_messages = self._format_openai_messages(messages)
            params = {
                "model": active_resolved_model,
                "messages": formatted_messages,
                "stream": True,
                "stream_options": {"include_usage": True},
                **call_kwargs,
            }
            if temperature is not None:
                params["temperature"] = temperature
            if tools:
                params["tools"] = tools
                params["tool_choice"] = call_kwargs.get("tool_choice", "auto")

            client = self._get_openai_client()
            try:
                resp = await client.chat.completions.create(**params)
            except Exception as exc:
                # Some OpenAI-compatible proxies reject stream_options; fall back cleanly.
                if "stream_options" in params and ("stream_options" in str(exc).lower() or isinstance(exc, TypeError)):
                    params.pop("stream_options", None)
                    resp = await client.chat.completions.create(**params)
                else:
                    raise
            text_parts = []
            tc_acc: dict[int, dict[str, str]] = {}
            stream_usage: dict[str, int] | None = None

            async for chunk in resp:
                chunk_usage = _extract_usage_dict(getattr(chunk, "usage", None))
                if chunk_usage:
                    stream_usage = chunk_usage
                choices = getattr(chunk, "choices", None)
                if not choices:
                    continue
                delta = getattr(choices[0], "delta", None)
                if delta is None:
                    continue
                content_piece = getattr(delta, "content", None)
                if content_piece:
                    text_parts.append(content_piece)
                    await _emit(content_piece)
                raw_tcs = getattr(delta, "tool_calls", None)
                for tc in raw_tcs or []:
                    idx = getattr(tc, "index", 0)
                    entry = tc_acc.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                    if getattr(tc, "id", None):
                        entry["id"] = str(tc.id)
                    fn = getattr(tc, "function", None)
                    if fn is not None:
                        if getattr(fn, "name", None):
                            entry["name"] += str(fn.name)
                        if getattr(fn, "arguments", None):
                            entry["arguments"] += str(fn.arguments)

            parsed_calls = []
            for idx in sorted(tc_acc):
                item = tc_acc[idx]
                try:
                    args_dict = json.loads(item["arguments"]) if item["arguments"].strip() else {}
                except Exception:
                    args_dict = {"raw_input": item["arguments"]}
                parsed_calls.append(
                    ToolCall(
                        id=item["id"] or f"call_{idx}",
                        function_name=item["name"],
                        arguments=args_dict if isinstance(args_dict, dict) else {},
                    )
                )
            meta = {"usage": stream_usage} if stream_usage else {}
            return ChatMessage(
                role=Role.ASSISTANT,
                content="".join(text_parts),
                tool_calls=parsed_calls or None,
                metadata=meta,
            )

        return await self._execute_with_resilience(_stream_once, can_retry=lambda: not emitted_any)

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
        _, _, active_resolved_model, _ = self._active_route()
        system_prompt, formatted, anthropic_tools = self._format_anthropic_payload(
            messages, tools
        )
        params: dict[str, Any] = {
            "model": active_resolved_model,
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

        usage = _extract_usage_dict(
            getattr(response, "usage", None)
            or (response.get("usage") if isinstance(response, dict) else None)
        )
        meta = {"usage": usage} if usage else {}
        return ChatMessage(
            role=Role.ASSISTANT,
            content=content or "",
            tool_calls=parsed_tool_calls if parsed_tool_calls else None,
            metadata=meta,
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

        usage = _extract_usage_dict(
            getattr(response, "usage", None)
            or (response.get("usage") if isinstance(response, dict) else None)
        )
        meta = {"usage": usage} if usage else {}
        return ChatMessage(
            role=Role.ASSISTANT,
            content="\n".join(text_parts),
            tool_calls=tool_calls if tool_calls else None,
            metadata=meta,
        )
