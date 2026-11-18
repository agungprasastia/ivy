"""Cross-platform background process registry and `process` management tool."""

from __future__ import annotations

import asyncio
import platform
import subprocess
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from ivy.core.prompt import redact_secrets, truncate_head_tail
from ivy.core.types import ToolResult
from ivy.tools.base import BaseTool
from ivy.tools.runner import terminate_process_tree


def build_shell_command(command: str) -> list[str]:
    """Return platform-appropriate shell invocation arguments for a command string."""
    if platform.system() == "Windows":
        return [
            "powershell.exe",
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            command,
        ]
    return ["/bin/bash", "-c", command]


def spawn_shell_kwargs(working_dir: str | None = None, with_stdin: bool = False) -> dict[str, Any]:
    """Build cross-platform asyncio subprocess kwargs with isolated process-group isolation."""
    kwargs: dict[str, Any] = {
        "stdout": asyncio.subprocess.PIPE,
        "stderr": asyncio.subprocess.PIPE,
        "cwd": working_dir,
    }
    if with_stdin:
        kwargs["stdin"] = asyncio.subprocess.PIPE
    if platform.system() == "Windows":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    return kwargs


@dataclass
class BackgroundProcess:
    """Tracked background subprocess with bounded streaming output buffer."""

    process_id: str
    command: str
    working_dir: str | None
    pid: int | None
    started_at: str
    started_monotonic: float
    proc: asyncio.subprocess.Process
    output_chunks: deque[str] = field(default_factory=deque)
    total_chars: int = 0
    exit_code: int | None = None
    tasks: list[asyncio.Task[Any]] = field(default_factory=list)


class BackgroundProcessManager:
    """Spawns, drains, inspects, writes stdin to, and terminates background shell processes."""

    def __init__(self, max_output_chars: int = 64_000) -> None:
        self.max_output_chars = max(1_000, max_output_chars)
        self._processes: dict[str, BackgroundProcess] = {}

    def _append_output(self, bp: BackgroundProcess, text: str) -> None:
        if not text:
            return
        bp.output_chunks.append(text)
        bp.total_chars += len(text)
        while bp.total_chars > self.max_output_chars and bp.output_chunks:
            dropped = bp.output_chunks.popleft()
            bp.total_chars -= len(dropped)

    async def _drain_stream(
        self, bp: BackgroundProcess, stream: asyncio.StreamReader | None
    ) -> None:
        if stream is None:
            return
        try:
            while True:
                chunk = await stream.read(4096)
                if not chunk:
                    break
                decoded = chunk.decode("utf-8", errors="replace")
                self._append_output(bp, decoded)
        except asyncio.CancelledError:
            raise
        except Exception:
            pass

    async def _watch_exit(self, bp: BackgroundProcess) -> None:
        try:
            rc = await bp.proc.wait()
            bp.exit_code = rc
        except asyncio.CancelledError:
            raise
        except Exception:
            pass

    async def spawn(
        self, command: str, working_dir: str | None = None
    ) -> BackgroundProcess:
        """Launch a background shell command and begin asynchronously draining its pipes."""
        shell_cmd = build_shell_command(command)
        exec_kwargs = spawn_shell_kwargs(working_dir=working_dir, with_stdin=True)
        proc = await asyncio.create_subprocess_exec(*shell_cmd, **exec_kwargs)

        process_id = f"proc_{uuid4().hex[:8]}"
        bp = BackgroundProcess(
            process_id=process_id,
            command=command,
            working_dir=working_dir,
            pid=proc.pid,
            started_at=datetime.now(timezone.utc).isoformat(),
            started_monotonic=time.monotonic(),
            proc=proc,
        )
        bp.tasks = [
            asyncio.create_task(self._drain_stream(bp, proc.stdout)),
            asyncio.create_task(self._drain_stream(bp, proc.stderr)),
            asyncio.create_task(self._watch_exit(bp)),
        ]
        self._processes[process_id] = bp
        return bp

    def _refresh_status(self, bp: BackgroundProcess) -> str:
        if bp.proc.returncode is not None:
            bp.exit_code = bp.proc.returncode
        return "running" if bp.proc.returncode is None else "exited"

    def list_processes(self) -> list[dict[str, Any]]:
        """Return status summaries for all tracked background processes."""
        items: list[dict[str, Any]] = []
        for bp in self._processes.values():
            status = self._refresh_status(bp)
            elapsed_s = round(max(0.0, time.monotonic() - bp.started_monotonic), 1)
            items.append(
                {
                    "process_id": bp.process_id,
                    "pid": bp.pid,
                    "status": status,
                    "exit_code": bp.exit_code,
                    "elapsed_s": elapsed_s,
                    "command": bp.command,
                    "started_at": bp.started_at,
                }
            )
        return items

    def read_output(
        self, process_id: str, tail_chars: int = 8_000
    ) -> tuple[bool, str]:
        """Return secret-scrubbed recent stdout/stderr for a background process."""
        bp = self._processes.get(process_id)
        if bp is None:
            return False, f"Process '{process_id}' not found."
        status = self._refresh_status(bp)
        combined = "".join(bp.output_chunks)
        if tail_chars > 0 and len(combined) > tail_chars:
            combined = combined[-tail_chars:]
        scrubbed = redact_secrets(combined).strip()
        header = f"[{bp.process_id} pid={bp.pid} status={status}"
        if bp.exit_code is not None:
            header += f" exit={bp.exit_code}"
        header += "]"
        body = scrubbed or "(no output yet)"
        return True, f"{header}\n{body}"

    async def write_stdin(self, process_id: str, data: str) -> tuple[bool, str]:
        """Write UTF-8 text to stdin of a running background process."""
        bp = self._processes.get(process_id)
        if bp is None:
            return False, f"Process '{process_id}' not found."
        if self._refresh_status(bp) != "running":
            return False, f"Process '{process_id}' has already exited (code={bp.exit_code})."
        stdin = bp.proc.stdin
        if stdin is None or stdin.is_closing():
            return False, f"Process '{process_id}' stdin is closed."
        try:
            stdin.write(data.encode("utf-8"))
            await stdin.drain()
            return True, f"Sent {len(data)} chars to '{process_id}'."
        except Exception as exc:
            return False, f"Failed writing stdin to '{process_id}': {exc}"

    async def stop(self, process_id: str) -> tuple[bool, str]:
        """Terminate a background process tree and wait for reader tasks to unwind."""
        bp = self._processes.get(process_id)
        if bp is None:
            return False, f"Process '{process_id}' not found."
        was_running = self._refresh_status(bp) == "running"
        if bp.proc.stdin is not None and not bp.proc.stdin.is_closing():
            bp.proc.stdin.close()
        await terminate_process_tree(bp.proc)
        for task in bp.tasks:
            if not task.done():
                task.cancel()
        if bp.tasks:
            await asyncio.gather(*bp.tasks, return_exceptions=True)
        transport = getattr(bp.proc, "_transport", None)
        if transport is not None and hasattr(transport, "close"):
            transport.close()
        self._refresh_status(bp)
        if was_running:
            return True, f"Stopped process '{process_id}' (pid={bp.pid})."
        return True, f"Process '{process_id}' already exited (code={bp.exit_code})."

    async def shutdown_all(self) -> None:
        """Stop all tracked background processes cleanly during TUI shutdown."""
        for pid_key in list(self._processes.keys()):
            await self.stop(pid_key)


class ProcessManageTool(BaseTool):
    """Tool allowing the agent to list, poll logs, write stdin, or stop background processes."""

    name = "process"
    description = (
        "Inspect, read output logs from, send stdin to, or terminate background shell processes "
        "started with terminal(..., background=true)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["list", "poll", "log", "write", "kill"],
                "description": "Operation to perform on managed background processes.",
            },
            "process_id": {
                "type": "string",
                "description": "Target process_id (required for poll/log/write/kill).",
            },
            "data": {
                "type": "string",
                "description": "Text payload to write to stdin when action='write'.",
            },
            "tail_chars": {
                "type": "integer",
                "description": "Max trailing characters to return for poll/log (default 8000).",
                "default": 8000,
            },
        },
        "required": ["action"],
    }

    def __init__(self, manager: BackgroundProcessManager) -> None:
        self.manager = manager

    async def execute(
        self,
        action: str,
        process_id: str | None = None,
        data: str = "",
        tail_chars: int = 8_000,
        **kwargs: Any,
    ) -> ToolResult:
        act = action.strip().lower()
        if act == "list":
            items = self.manager.list_processes()
            if not items:
                return ToolResult(tool_name=self.name, status="success", output="No background processes.")
            lines = [
                f"- {it['process_id']} | pid={it['pid']} | {it['status']} | "
                f"{it['elapsed_s']}s | {it['command'][:80]}"
                for it in items
            ]
            return ToolResult(tool_name=self.name, status="success", output="\n".join(lines))

        if not process_id:
            return ToolResult(
                tool_name=self.name,
                status="error",
                error=f"'process_id' is required for action='{act}'.",
                output="",
            )

        if act in ("poll", "log"):
            ok, msg = self.manager.read_output(process_id, tail_chars=int(tail_chars or 8_000))
            if not ok:
                return ToolResult(tool_name=self.name, status="error", error=msg, output="")
            return ToolResult(
                tool_name=self.name,
                status="success",
                output=truncate_head_tail(msg, max_chars=16_000),
            )

        if act == "write":
            ok, msg = await self.manager.write_stdin(process_id, data)
            if not ok:
                return ToolResult(tool_name=self.name, status="error", error=msg, output="")
            return ToolResult(tool_name=self.name, status="success", output=msg)

        if act == "kill":
            ok, msg = await self.manager.stop(process_id)
            if not ok:
                return ToolResult(tool_name=self.name, status="error", error=msg, output="")
            return ToolResult(tool_name=self.name, status="success", output=msg)

        return ToolResult(
            tool_name=self.name,
            status="error",
            error=f"Unknown process action '{action}'. Use list, poll, log, write, or kill.",
            output="",
        )
