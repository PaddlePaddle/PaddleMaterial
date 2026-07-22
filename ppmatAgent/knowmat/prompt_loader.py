"""Shared prompt template loaders with in-process caching."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable

import yaml


def _prompts_dir() -> Path:
    return Path(__file__).resolve().parent / "prompts"


@lru_cache(maxsize=64)
def load_text_template(filename: str) -> str:
    path = _prompts_dir() / filename
    return path.read_text(encoding="utf-8")


@lru_cache(maxsize=64)
def load_yaml_templates(filename: str) -> Dict[str, str]:
    path = _prompts_dir() / filename
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return {k: v for k, v in data.items() if isinstance(v, str)}


def load_yaml_templates_required(filename: str, required_keys: Iterable[str]) -> Dict[str, str]:
    """Load a YAML prompt file and validate required string keys exist."""
    templates = load_yaml_templates(filename)
    missing = [k for k in required_keys if not templates.get(k)]
    if missing:
        raise ValueError(
            f"Prompt template file {filename!r} is missing required keys: {', '.join(missing)}"
        )
    return templates


@lru_cache(maxsize=64)
def load_routing_supplement(relative_path: str) -> str:
    """Load a routing supplement file from prompts/directions/.

    Parameters
    ----------
    relative_path : str
        Path relative to prompts/directions/, e.g. "base/Metals.txt".

    Returns
    -------
    str
        The file content, or empty string if the file does not exist.
    """
    path = _prompts_dir() / "directions" / relative_path
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8")
