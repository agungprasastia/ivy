"""Terminal execution tool for PowerShell on Windows and Bash on Unix/Linux."""

from __future__ import annotations

import asyncio
import platform
import subprocess
from typing import Any

from aether_agent.core.types import ToolResult
from aether_agent.tools.approval import SafetyApprovalPolicy
from aether_agent.tools.base import BaseTool
from aether_agent.tools.runner import terminate_process_tree


class TerminalTool(BaseTool):
    """Executes shell commands with safety intercept and timeout protection."""

    name = "terminal"
    description = (
        "Execute shell commands in the operating system terminal "
        "(PowerShell on Windows, Bash on Linux/macOS)."
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
        },
        "required": ["command"],
    }

    def __init__(
        self,
        approval_policy: SafetyApprovalPolicy | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.approval_policy = approval_policy or SafetyApprovalPolicy()
        self.timeout = timeout

    async def execute(
        self,
        command: str,
        working_dir: str | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        """Run the command after safety approval."""
        is_approved = await self.approval_policy.check_approval(command)
        if not is_approved:
            return ToolResult(
                tool_name=self.name,
                status="rejected",
                error="Command rejected by safety approval policy",
                output="Command execution rejected by user/safety policy",
            )

        if platform.system() == "Windows":
            shell_cmd = [
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                command,
            ]
        else:
            shell_cmd = ["/bin/bash", "-c", command]

        proc: asyncio.subprocess.Process | None = None
        try:
            exec_kwargs: dict[str, Any] = {
                "stdout": asyncio.subprocess.PIPE,
                "stderr": asyncio.subprocess.PIPE,
                "cwd": working_dir,
            }
            if platform.system() == "Windows":
                exec_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                exec_kwargs["start_new_session"] = True

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
