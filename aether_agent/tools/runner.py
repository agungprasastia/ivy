"""Isolated Python code execution sandbox runner."""

from __future__ import annotations

import asyncio
import sys
import tempfile
from pathlib import Path
from typing import Any

from aether_agent.core.types import ToolResult
from aether_agent.tools.base import BaseTool


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

            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                str(temp_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(),
                timeout=self.timeout,
            )
        except asyncio.TimeoutError:
            if proc is not None:
                try:
                    proc.kill()
                    await proc.wait()
                except Exception:
                    pass
            return ToolResult(
                tool_name=self.name,
                status="error",
                error=f"Execution timed out after {self.timeout} seconds",
                output="",
            )
        except Exception as exc:
            return ToolResult(
                tool_name=self.name,
                status="error",
                error=f"Failed to start runner: {exc}",
                output="",
            )
        finally:
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
