"""In-memory per-session file modification checkpoints for /rollback."""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class FileCheckpoint:
    """Snapshot of a file's state before WriteFileTool or PatchFileTool modified it."""

    id: int
    session_id: str
    path: str
    display_path: str
    tool_name: str
    existed_before: bool
    old_content: str
    new_content: str
    timestamp: str


class FileCheckpointStore:
    """Tracks pre-edit snapshots of files modified during interactive sessions."""

    def __init__(self, max_entries: int = 200) -> None:
        self.max_entries = max(10, max_entries)
        self._next_id = 1
        self._entries: list[FileCheckpoint] = []

    def record(
        self,
        session_id: str,
        path: str,
        display_path: str,
        tool_name: str,
        existed_before: bool,
        old_content: str,
        new_content: str,
    ) -> FileCheckpoint:
        cp = FileCheckpoint(
            id=self._next_id,
            session_id=session_id,
            path=str(path),
            display_path=display_path,
            tool_name=tool_name,
            existed_before=existed_before,
            old_content=old_content,
            new_content=new_content,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self._next_id += 1
        self._entries.append(cp)
        if len(self._entries) > self.max_entries:
            self._entries = self._entries[-self.max_entries :]
        return cp

    def list_checkpoints(self, session_id: str | None = None) -> list[FileCheckpoint]:
        """List checkpoints (newest first), optionally filtered by session_id."""
        items = [
            cp
            for cp in self._entries
            if session_id is None or cp.session_id == session_id
        ]
        return list(reversed(items))

    def get(self, checkpoint_id: int) -> FileCheckpoint | None:
        for cp in self._entries:
            if cp.id == checkpoint_id:
                return cp
        return None

    def latest(self, session_id: str | None = None) -> FileCheckpoint | None:
        items = self.list_checkpoints(session_id=session_id)
        return items[0] if items else None

    def diff_checkpoint(self, checkpoint_id: int) -> str:
        """Compute unified diff between pre-edit snapshot and current disk state."""
        cp = self.get(checkpoint_id)
        if cp is None:
            return f"Checkpoint #{checkpoint_id} not found."
        target = Path(cp.path)
        current_content = (
            target.read_text(encoding="utf-8", errors="replace")
            if target.is_file()
            else cp.new_content
        )
        diff_lines = list(
            difflib.unified_diff(
                cp.old_content.splitlines(keepends=True),
                current_content.splitlines(keepends=True),
                fromfile=f"before/{cp.display_path}",
                tofile=f"after/{cp.display_path}",
                n=3,
            )
        )
        return "".join(diff_lines) if diff_lines else "(no differences)"

    def restore_checkpoint(self, checkpoint_id: int) -> tuple[bool, str]:
        """Restore file on disk to its pre-checkpoint state (or delete newly created file)."""
        cp = self.get(checkpoint_id)
        if cp is None:
            return False, f"Checkpoint #{checkpoint_id} not found."
        target = Path(cp.path)
        try:
            if not cp.existed_before:
                if target.exists():
                    target.unlink()
                return True, f"Removed newly created file {cp.display_path} (checkpoint #{cp.id})."
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(cp.old_content, encoding="utf-8")
            return True, f"Restored {cp.display_path} to state before checkpoint #{cp.id}."
        except OSError as exc:
            return False, f"Failed to restore checkpoint #{cp.id}: {exc}"
