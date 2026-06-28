"""
Global configuration and environment handling for KnowMat 2.0.

This module attempts to locate and load environment variables from a
``.env`` file if present.  It also ensures that critical secrets such as
``LLM_API_KEY`` are set.  When required variables are missing the code will
prompt interactively for them at runtime.

In addition, this module can configure LangSmith tracing when a
``LANGCHAIN_API_KEY`` is provided.
"""

import os
import sys

from ppmatAgent.knowmat.env_loader import load_project_dotenv


# Try to locate and validate a .env file before loading it.
_env_path = load_project_dotenv(override=False) or ""


def _set_env(var: str, required: bool = True) -> None:
    """Ensure an environment variable is set without blocking in non-interactive environments.

    Parameters
    ----------
    var: str
        Name of the environment variable to ensure.
    required: bool
        Whether this variable is required for runtime execution.
    """
    if var in os.environ:
        return
    if not required:
        return

    # In non-interactive environments (no TTY), fail fast with a clear error
    if not sys.stdin or not sys.stdin.isatty():
        raise RuntimeError(
            f"Required environment variable {var!r} is not set and interactive prompts "
            "are not available. Please set it in your environment or .env file."
        )

    import getpass

    os.environ[var] = getpass.getpass(f"{var}: ")


# Ensure that the primary LLM key exists.
_set_env("LLM_API_KEY", required=True)

# Keep LangSmith key optional.
_set_env("LANGCHAIN_API_KEY", required=False)

# Provide OpenAI-compatible env aliases for libraries that still read OPENAI_*.
if os.getenv("LLM_API_KEY") and not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = os.environ["LLM_API_KEY"]
if os.getenv("LLM_BASE_URL") and not os.getenv("OPENAI_BASE_URL"):
    os.environ["OPENAI_BASE_URL"] = os.environ["LLM_BASE_URL"]

# LangSmith tracing is optional. Enable by default only if API key is provided.
if os.getenv("LANGCHAIN_API_KEY"):
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_PROJECT", "KnowMat2")
else:
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
