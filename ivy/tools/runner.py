"""Isolated Python code execution sandbox runner."""

from __future__ import annotations

import asyncio
import os
import platform
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from ivy.core.types import ToolResult
from ivy.tools.base import BaseTool


async def terminate_process_tree(proc: asyncio.subprocess.Process | None) -> None:
    """Terminate a subprocess and its descendants cleanly across Windows and POSIX."""
    if proc is None or proc.returncode is not None:
        return
    if platform.system() == "Windows":
        try:
            killer = await asyncio.create_subprocess_exec(
                "taskkill",
                "/F",
                "/T",
                "/PID",
                str(proc.pid),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await killer.wait()
        except Exception:
            pass
    else:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            pass

    if proc.returncode is None:
        try:
            proc.kill()
        except Exception:
            pass
    try:
        await proc.wait()
    except Exception:
        pass


class CodeRunnerTool(BaseTool):
    """Executes Python snippets in an isolated child process with timeout enforcement."""

    name = "code_runner"
    description = (
        "Execute Python code in an isolated subprocess sandbox. "
        "Terminates automatically if execution exceeds timeout (default 30 seconds)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python source code string to execute.",
            },
        },
        "required": ["code"],
    }

    def __init__(self, timeout: float = 30.0) -> None:
        self.timeout = timeout

    async def execute(self, code: str, **kwargs: Any) -> ToolResult:
        """Run the Python code in a separate process."""
        temp_fd, temp_path_str = tempfile.mkstemp(suffix=".py", text=True)
        temp_path = Path(temp_path_str)
        proc: asyncio.subprocess.Process | None = None

        try:
            with open(temp_fd, "w", encoding="utf-8") as f:
                f.write(code)

            exec_kwargs: dict[str, Any] = {
                "stdout": asyncio.subprocess.PIPE,
                "stderr": asyncio.subprocess.PIPE,
            }
            if platform.system() == "Windows":
                exec_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
            else:
                exec_kwargs["start_new_session"] = True

            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                str(temp_path),
                **exec_kwargs,
            )

            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(),
                timeout=self.timeout,
            )
        except asyncio.TimeoutError:
            await terminate_process_tree(proc)
            return ToolResult(
                tool_name=self.name,
                status="error",
                error=f"Execution timed out after {self.timeout} seconds",
                output="",
            )
        except Exception as exc:
            await terminate_process_tree(proc)
            return ToolResult(
                tool_name=self.name,
                status="error",
                error=f"Failed to start runner: {exc}",
                output="",
            )
        finally:
            await terminate_process_tree(proc)
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass

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
