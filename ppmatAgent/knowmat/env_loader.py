"""Helpers for loading and validating project `.env` files."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple

try:
    from dotenv import find_dotenv, load_dotenv
except ImportError:  # pragma: no cover - optional agent dependency
    find_dotenv = None  # type: ignore[assignment]
    load_dotenv = None  # type: ignore[assignment]


def find_project_dotenv() -> str:
    """Locate the active `.env` file using KnowMat's search order."""
    cwd_dotenv = os.path.join(os.getcwd(), ".env")
    if os.path.isfile(cwd_dotenv):
        return cwd_dotenv

    env_path = os.getenv("KNOWMAT2_ENV_FILE", "")
    if env_path:
        return env_path

    if find_dotenv is None:
        return ""
    return find_dotenv(usecwd=True) or ""


def _find_dotenv_syntax_errors(path: Path) -> List[Tuple[int, str]]:
    """Return simple, actionable syntax errors for a dotenv file.

    `python-dotenv` may continue after parse errors and leave later variables
    unset. We proactively reject malformed quoted values so OCR settings do not
    silently fall back to defaults.
    """
    errors: List[Tuple[int, str]] = []
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError as exc:
        return [(0, f"failed to read file: {exc}")]

    for lineno, raw_line in enumerate(text.splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("export "):
            stripped = stripped[len("export ") :].lstrip()
        if "=" not in stripped:
            continue

        _, value = stripped.split("=", 1)
        value = value.strip()
        if not value:
            continue

        quote = value[0]
        if quote not in ("'", '"'):
            continue

        escaped = False
        closed = False
        for ch in value[1:]:
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                continue
            if ch == quote:
                closed = True
                break
        if not closed:
            errors.append((lineno, f"missing closing {quote} quote"))

    return errors


def validate_dotenv_file(path: str) -> None:
    """Raise a clear error when `.env` contains malformed quoted values."""
    if not path:
        return
    errors = _find_dotenv_syntax_errors(Path(path))
    if not errors:
        return

    details = "; ".join(
        f"line {lineno}: {message}" if lineno > 0 else message
        for lineno, message in errors
    )
    raise RuntimeError(
        f"Invalid .env file at {path}: {details}. "
        "Fix the malformed entry before running KnowMat."
    )


def load_project_dotenv(*, override: bool = False) -> Optional[str]:
    """Load the active `.env` file after validating its syntax."""
    env_path = find_project_dotenv()
    if not env_path:
        return None
    validate_dotenv_file(env_path)
    if load_dotenv is None:
        return env_path
    load_dotenv(env_path, override=override)
    return env_path
