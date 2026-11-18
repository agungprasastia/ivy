"""Terminal execution tool for PowerShell on Windows and Bash on Unix/Linux."""

from __future__ import annotations

import asyncio
from typing import Any

from ivy.core.types import ToolResult
from ivy.tools.approval import SafetyApprovalPolicy
from ivy.tools.base import BaseTool
from ivy.tools.process_registry import (
    BackgroundProcessManager,
    build_shell_command,
    spawn_shell_kwargs,
)
from ivy.tools.runner import terminate_process_tree


class TerminalTool(BaseTool):
    """Executes shell commands with safety intercept, timeout, and optional background mode."""

    name = "terminal"
    description = (
        "Execute shell commands in the operating system terminal "
        "(PowerShell on Windows, Bash on Linux/macOS). Set background=true for long-running servers."
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The command string to execute in the shell.",
            },
            "working_dir": {
                "type": "string",
                "description": "Optional working directory path where the command executes.",
            },
            "background": {
                "type": "boolean",
                "description": "Start command in the background and return a process_id immediately.",
                "default": False,
            },
        },
        "required": ["command"],
    }

    def __init__(
        self,
        approval_policy: SafetyApprovalPolicy | None = None,
        timeout: float = 60.0,
        process_manager: BackgroundProcessManager | None = None,
    ) -> None:
        self.approval_policy = approval_policy or SafetyApprovalPolicy()
        self.timeout = timeout
        self.process_manager = process_manager

    async def execute(
        self,
        command: str,
        working_dir: str | None = None,
        background: bool = False,
        **kwargs: Any,
    ) -> ToolResult:
        """Run the command after safety approval."""
        is_approved, feedback = await self.approval_policy.evaluate_approval(command)
        if not is_approved:
            reason = (
                f"Command rejected by user. User instruction: {feedback}"
                if feedback
                else "Command rejected by safety approval policy"
            )
            return ToolResult(
                tool_name=self.name,
                status="rejected",
                error=reason,
                output=reason if feedback else "Command execution rejected by user/safety policy",
            )

        if background:
            if self.process_manager is None:
                return ToolResult(
                    tool_name=self.name,
                    status="error",
                    error="Background process manager is not configured.",
                    output="",
                )
            try:
                bp = await self.process_manager.spawn(command=command, working_dir=working_dir)
                return ToolResult(
                    tool_name=self.name,
                    status="success",
                    output=(
                        f"Started background process {bp.process_id} (pid={bp.pid}). "
                        "Use the `process` tool with action='poll'|'write'|'kill' to manage it."
                    ),
                    metadata={"process_id": bp.process_id, "pid": bp.pid, "background": True},
                )
            except Exception as exc:
                return ToolResult(
                    tool_name=self.name,
                    status="error",
                    error=f"Failed to start background process: {exc}",
                    output="",
                )

        shell_cmd = build_shell_command(command)

        proc: asyncio.subprocess.Process | None = None
        try:
            exec_kwargs = spawn_shell_kwargs(working_dir=working_dir, with_stdin=False)
            proc = await asyncio.create_subprocess_exec(*shell_cmd, **exec_kwargs)

            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(),
                timeout=self.timeout,
            )
        except asyncio.TimeoutError:
            await terminate_process_tree(proc)
            return ToolResult(
                tool_name=self.name,
                status="error",
                error=f"Command timed out after {self.timeout} seconds",
                output="",
            )
        except Exception as exc:
            await terminate_process_tree(proc)
            return ToolResult(
                tool_name=self.name,
                status="error",
                error=f"Failed to execute process: {exc}",
                output="",
            )
        finally:
            await terminate_process_tree(proc)

        stdout_str = stdout_bytes.decode("utf-8", errors="replace").strip()
        stderr_str = stderr_bytes.decode("utf-8", errors="replace").strip()

        if proc.returncode == 0:
            return ToolResult(
                tool_name=self.name,
                status="success",
                output=stdout_str,
                error=None,
            )
        return ToolResult(
            tool_name=self.name,
            status="error",
            output=stdout_str,
            error=stderr_str or f"Process exited with code {proc.returncode}",
        )
