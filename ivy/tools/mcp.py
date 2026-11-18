"""Zero-dependency JSON-RPC 2.0 stdio MCP client and BaseTool adapter."""

from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ivy.core.types import ToolResult
from ivy.tools.base import BaseTool

if TYPE_CHECKING:
    from ivy.tools.registry import ToolRegistry


def _sanitize_tool_segment(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", value.strip()).strip("_") or "tool"


class MCPStdioServer:
    """Lightweight async JSON-RPC 2.0 client over stdio subprocess pipes."""

    def __init__(
        self,
        name: str,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        timeout: float = 10.0,
    ) -> None:
        self.name = name
        self.command = command
        self.args = list(args or [])
        self.env = env
        self.timeout = timeout
        self._proc: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()
        self._next_id = 1
        self.tools: list[dict[str, Any]] = []
        self.connected = False

    async def _send_json(self, payload: dict[str, Any]) -> None:
        if self._proc is None or self._proc.stdin is None:
            raise RuntimeError(f"MCP server '{self.name}' is not started.")
        raw = (json.dumps(payload) + "\n").encode("utf-8")
        self._proc.stdin.write(raw)
        await self._proc.stdin.drain()

    async def _request(self, method: str, params: dict[str, Any] | None = None) -> Any:
        async with self._lock:
            req_id = self._next_id
            self._next_id += 1
            payload: dict[str, Any] = {"jsonrpc": "2.0", "id": req_id, "method": method}
            if params is not None:
                payload["params"] = params
            await self._send_json(payload)

            if self._proc is None or self._proc.stdout is None:
                raise RuntimeError(f"MCP server '{self.name}' has no stdout.")

            while True:
                line_b = await asyncio.wait_for(
                    self._proc.stdout.readline(), timeout=self.timeout
                )
                if not line_b:
                    raise EOFError(f"MCP server '{self.name}' closed stdout unexpectedly.")
                line = line_b.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if msg.get("id") != req_id:
                    continue
                if "error" in msg and msg["error"] is not None:
                    raise RuntimeError(f"MCP error ({method}): {msg['error']}")
                return msg.get("result")

    async def connect(self) -> list[dict[str, Any]]:
        """Spawn the stdio server, complete MCP handshake, and discover available tools."""
        merged_env = os.environ.copy()
        if self.env:
            merged_env.update({str(k): str(v) for k, v in self.env.items()})

        self._proc = await asyncio.create_subprocess_exec(
            self.command,
            *self.args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=merged_env,
        )
        await self._request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "ivy", "version": "0.1.0"},
            },
        )
        await self._send_json({"jsonrpc": "2.0", "method": "notifications/initialized"})
        res = await self._request("tools/list", {})
        self.tools = list(res.get("tools", [])) if isinstance(res, dict) else []
        self.connected = True
        return self.tools

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> ToolResult:
        """Invoke a remote MCP tool via `tools/call`."""
        try:
            res = await self._request(
                "tools/call",
                {"name": tool_name, "arguments": arguments},
            )
            if not isinstance(res, dict):
                return ToolResult(tool_name=tool_name, status="success", output=str(res))

            blocks = res.get("content") or []
            parts: list[str] = []
            for blk in blocks:
                if isinstance(blk, dict):
                    if blk.get("type") == "text" and blk.get("text") is not None:
                        parts.append(str(blk["text"]))
                    else:
                        parts.append(json.dumps(blk, ensure_ascii=False))
                else:
                    parts.append(str(blk))
            text = "\n".join(parts)
            is_err = bool(res.get("isError", False))
            return ToolResult(
                tool_name=tool_name,
                status="error" if is_err else "success",
                output="" if is_err else text,
                error=text if is_err else None,
            )
        except Exception as exc:
            return ToolResult(
                tool_name=tool_name,
                status="error",
                error=f"MCP call failed ({self.name}.{tool_name}): {exc}",
            )

    async def close(self) -> None:
        """Terminate subprocess cleanly without hanging."""
        self.connected = False
        if self._proc is None:
            return
        try:
            if self._proc.returncode is None:
                self._proc.terminate()
                try:
                    await asyncio.wait_for(self._proc.wait(), timeout=2.0)
                except asyncio.TimeoutError:
                    self._proc.kill()
                    await self._proc.wait()
        finally:
            self._proc = None


class MCPToolAdapter(BaseTool):
    """Adapts an external MCP tool definition onto Ivy's BaseTool interface."""

    def __init__(self, server: MCPStdioServer, tool_spec: dict[str, Any]) -> None:
        self.server = server
        self.remote_name = str(tool_spec.get("name") or "tool")
        self.name = f"mcp_{_sanitize_tool_segment(server.name)}_{_sanitize_tool_segment(self.remote_name)}"
        self.description = str(
            tool_spec.get("description") or f"MCP tool {self.remote_name} from {server.name}"
        )
        schema = tool_spec.get("inputSchema")
        self.parameters = (
            schema
            if isinstance(schema, dict)
            else {"type": "object", "properties": {}, "required": []}
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        res = await self.server.call_tool(self.remote_name, kwargs)
        res.tool_name = self.name
        return res


class MCPManager:
    """Manages `.mcp.json` server lifecycle and registers MCPToolAdapters in ToolRegistry."""

    def __init__(self, timeout: float = 10.0) -> None:
        self.timeout = timeout
        self.servers: dict[str, MCPStdioServer] = {}
        self.registered_tool_names: list[str] = []
        self.errors: dict[str, str] = {}

    @staticmethod
    def load_config_file(config_path: str | Path = ".mcp.json") -> dict[str, dict[str, Any]]:
        path = Path(config_path)
        if not path.is_file():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        raw = data.get("mcpServers") if isinstance(data, dict) else {}
        return raw if isinstance(raw, dict) else {}

    async def reload(
        self, registry: ToolRegistry, config_path: str | Path = ".mcp.json"
    ) -> list[str]:
        """Close previous MCP connections, load `.mcp.json`, and register adapters."""
        for name in self.registered_tool_names:
            registry.unregister(name)
        self.registered_tool_names.clear()
        await self.close_all()
        self.errors.clear()

        specs = self.load_config_file(config_path)
        for srv_name, cfg in specs.items():
            if not isinstance(cfg, dict) or not cfg.get("command"):
                continue
            server = MCPStdioServer(
                name=srv_name,
                command=str(cfg["command"]),
                args=[str(a) for a in (cfg.get("args") or [])],
                env=cfg.get("env") if isinstance(cfg.get("env"), dict) else None,
                timeout=self.timeout,
            )
            try:
                tools = await server.connect()
                self.servers[srv_name] = server
                for spec in tools:
                    adapter = MCPToolAdapter(server, spec)
                    registry.register(adapter)
                    self.registered_tool_names.append(adapter.name)
            except Exception as exc:
                self.errors[srv_name] = str(exc)
                await server.close()
        return list(self.registered_tool_names)

    def list_servers(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for name, srv in self.servers.items():
            rows.append(
                {
                    "name": name,
                    "status": "connected" if srv.connected else "disconnected",
                    "tools": len(srv.tools),
                    "error": "",
                }
            )
        for name, err in self.errors.items():
            rows.append({"name": name, "status": "error", "tools": 0, "error": err})
        return rows

    async def close_all(self) -> None:
        for srv in list(self.servers.values()):
            await srv.close()
        self.servers.clear()
