"""Path safety validation helpers for file and search tools."""

from __future__ import annotations

import os
from pathlib import Path

SENSITIVE_DIR_NAMES = frozenset({".ssh", ".aws", ".gnupg", ".kube"})
SENSITIVE_FILE_NAMES = frozenset({".git-credentials", ".netrc", ".npmrc", ".pypirc"})
SAFE_DOTENV_EXAMPLES = frozenset({".env.example", ".env.template", ".env.sample"})


def _strip_windows_ads(segment: str) -> str:
    """Strip NTFS Alternate Data Stream suffix (e.g. `.env:stream` -> `.env`)."""
    if os.name == "nt" and ":" in segment:
        base = segment.split(":", 1)[0]
        if base:
            return base
    return segment


def has_control_chars(raw: str) -> bool:
    """Return True if path string contains ASCII control characters or NUL."""
    return any(ord(ch) < 32 or ord(ch) == 127 for ch in raw)


def is_sensitive_path(path: Path, *, allow_dotenv: bool = False) -> str | None:
    """Return a denial reason when path points to a credential or sensitive system file."""
    lower_parts = [_strip_windows_ads(part).lower() for part in path.parts]
    for part in lower_parts:
        if part in SENSITIVE_DIR_NAMES:
            return f"Access denied: sensitive directory segment '{part}' is protected."

    name = _strip_windows_ads(path.name).lower()
    if name in SENSITIVE_FILE_NAMES:
        return f"Access denied: sensitive credential file '{path.name}' is protected."

    if len(lower_parts) >= 2 and lower_parts[-2:] in (["etc", "shadow"], ["etc", "passwd"], ["etc", "sudoers"]):
        return f"Access denied: sensitive system file '{path}' is protected."

    if not allow_dotenv and (name == ".env" or name.startswith(".env.")) and name not in SAFE_DOTENV_EXAMPLES:
        return f"Access denied: environment secret file '{path.name}' is protected."

    return None


def should_skip_sensitive_entry(path: Path) -> bool:
    """Return True when directory walking should skip a sensitive dir or file."""
    return is_sensitive_path(path) is not None


def validate_file_path(
    raw_path: str,
    *,
    mode: str = "read",
    workspace_root: Path | None = None,
    allow_dotenv: bool = False,
) -> tuple[Path | None, str | None]:
    """Validate and resolve a file/directory path against control-char, root, and secret guards."""
    if not raw_path or not raw_path.strip():
        return None, "Invalid path: path must not be empty."

    if has_control_chars(raw_path):
        return None, "Invalid path: control characters are not allowed in file paths."

    try:
        resolved = Path(raw_path).expanduser().resolve()
    except OSError as exc:
        return None, f"Invalid path '{raw_path}': {exc}"

    if workspace_root is not None:
        root = workspace_root.resolve()
        if not resolved.is_relative_to(root):
            return (
                None,
                f"Access denied ({mode}): path '{raw_path}' escapes workspace root '{root}'.",
            )

    sensitive_err = is_sensitive_path(resolved, allow_dotenv=allow_dotenv)
    if sensitive_err is not None:
        return None, sensitive_err

    return resolved, None
