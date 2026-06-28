"""LLM connection helpers for the HEA CrewAI workflow.

The HEA agent only distinguishes two API styles:

* ``llmone``: the default LLMONE/OpenAI-compatible gateway used by the project.
* ``openai``: any OpenAI-compatible endpoint, selected by changing ``base_url``.

DeepSeek, Qianfan, AI Studio, and other gateways should be configured as
``openai`` with their own ``OPENAI_BASE_URL`` / ``LLM_BASE_URL`` and model name.
Legacy ``provider`` argument names are still accepted as compatibility aliases.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional
from urllib import error, request
from urllib.parse import urlparse

from ppmatAgent.hea_crewai_agent.runtime_config import load_env_file


_LLMONE_DEFAULT_BASE_URL = "https://oneapi-comate.baidu-int.com/v1"
_OPENAI_DEFAULT_BASE_URL = "https://api.openai.com/v1"

_HOST_MODEL_FALLBACKS: Dict[str, List[str]] = {
    "oneapi-comate.baidu-int.com": [
        "gpt-5.5",
        "gpt-5.4",
        "gpt-5.4-mini",
        "gpt-5.3-codex",
        "gpt-5.2",
    ],
    "api.deepseek.com": ["deepseek-chat", "deepseek-reasoner"],
    "qianfan.baidubce.com": ["ernie-4.5-turbo-128k", "ernie-4.5-turbo-32k"],
    "aistudio.baidu.com": ["ernie-5.0-thinking-preview", "ernie-x1.1-preview"],
}

_MODEL_CATALOG_CACHE: Dict[str, Any] = {
    "base_url": None,
    "default_model": None,
    "payload": None,
    "cached_at": 0.0,
}

_RATE_LIMIT_LOCK = threading.RLock()
_LAST_RATE_LIMITED_CALL_AT = 0.0


@dataclass(frozen=True)
class LLMConnectionConfig:
    """Resolved settings for an OpenAI-compatible CrewAI LLM."""

    llm_api: str
    model: str
    api_key: str
    base_url: str
    needs_rate_limit_backoff: bool = False

    @property
    def provider(self) -> str:
        """Compatibility alias for older HEA code."""
        return self.llm_api


# Backward-compatible name used by the previous implementation.
LLMProviderConfig = LLMConnectionConfig


def _first_env(*names: str) -> str:
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return ""


def _normalize_openai_base_url(raw_url: str) -> str:
    """Normalize common OpenAI-compatible hosts without breaking custom paths."""
    cleaned = str(raw_url or "").strip().rstrip("/")
    if not cleaned:
        return cleaned

    parsed = urlparse(cleaned)
    if parsed.scheme and parsed.netloc and parsed.path in {"", "/"}:
        return f"{cleaned}/v1"
    return cleaned


def normalize_llm_api(llm_api: Optional[str] = None) -> str:
    """Return ``llmone`` or ``openai``.

    Old names such as ``deepseek``, ``qianfan``, and ``wenxin-aistudio`` are
    treated as OpenAI-compatible endpoints. Configure them through ``base_url``.
    """
    load_env_file()
    raw = (
        llm_api
        or os.getenv("HEA_LLM_API")
        or os.getenv("HEA_LLM_PROVIDER")  # legacy
        or os.getenv("LLM_API_TYPE")
        or os.getenv("LLM_PROVIDER")  # legacy
        or "llmone"
    )
    key = str(raw).strip().lower().replace("_", "-")
    aliases = {
        "": "llmone",
        "llmone": "llmone",
        "llm-one": "llmone",
        "oneapi": "llmone",
        "baidu-oneapi": "llmone",
        "openai": "openai",
        "openai-compatible": "openai",
        "compatible": "openai",
        "deepseek": "openai",
        "qianfan": "openai",
        "baidu-qianfan": "openai",
        "wenxin": "openai",
        "ernie": "openai",
        "aistudio": "openai",
        "ai-studio": "openai",
        "wenxin-aistudio": "openai",
    }
    resolved = aliases.get(key)
    if resolved is None:
        raise ValueError(
            f"Unsupported HEA LLM API '{raw}'. Use 'llmone' or 'openai'. "
            "For DeepSeek/Qianfan/AI Studio, use 'openai' plus the proper base_url."
        )
    return resolved


# Backward-compatible function name.
def normalize_llm_provider(provider: Optional[str] = None) -> str:
    return normalize_llm_api(provider)


def resolve_llm_connection_config(
    model: str = "gpt-5.2",
    llm_api: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> LLMConnectionConfig:
    """Resolve ``api_key + base_url + model`` for CrewAI."""
    load_env_file()
    resolved_api = normalize_llm_api(llm_api)

    if resolved_api == "llmone":
        resolved_key = api_key or _first_env(
            "LLMONE_API_KEY",
            "HEA_LLM_API_KEY",
            "LLM_API_KEY",
            "OPENAI_API_KEY",
        )
        if not resolved_key:
            raise RuntimeError(
                "HEA LLMONE API requires LLMONE_API_KEY. "
                "You may also set HEA_LLM_API_KEY or LLM_API_KEY."
            )
        resolved_base_url = _normalize_openai_base_url(
            base_url
            or os.getenv("LLMONE_BASE_URL")
            or os.getenv("HEA_LLM_BASE_URL")
            or _LLMONE_DEFAULT_BASE_URL
        )
    else:
        resolved_key = api_key or _first_env(
            "OPENAI_API_KEY",
            "LLM_API_KEY",
            "HEA_LLM_API_KEY",
            "AI_STUDIO_API_KEY",
            "AISTUDIO_ACCESS_TOKEN",
        )
        if not resolved_key:
            raise RuntimeError(
                "OpenAI-compatible HEA LLM requires OPENAI_API_KEY or LLM_API_KEY. "
                "For AI Studio, AI_STUDIO_API_KEY is also accepted."
            )
        resolved_base_url = _normalize_openai_base_url(
            base_url
            or os.getenv("OPENAI_BASE_URL")
            or os.getenv("LLM_BASE_URL")
            or os.getenv("HEA_LLM_BASE_URL")
            or os.getenv("AI_STUDIO_BASE_URL")
            or os.getenv("AISTUDIO_BASE_URL")
            or _OPENAI_DEFAULT_BASE_URL
        )

    return LLMConnectionConfig(
        llm_api=resolved_api,
        model=model,
        api_key=resolved_key,
        base_url=resolved_base_url,
        needs_rate_limit_backoff="aistudio.baidu.com" in resolved_base_url.lower(),
    )


# Backward-compatible function name.
def resolve_llm_provider_config(
    model: str = "gpt-5.2",
    provider: Optional[str] = None,
    base_url: Optional[str] = None,
) -> LLMConnectionConfig:
    return resolve_llm_connection_config(
        model=model,
        llm_api=provider,
        base_url=base_url,
    )


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(float(os.getenv(name, str(default))))
    except Exception:
        return default


def _is_rate_limit_error(exc: Exception) -> bool:
    text = str(exc)
    lowered = text.lower()
    return (
        "访问过于频繁" in text
        or "rate limit" in lowered
        or "too many requests" in lowered
        or "error code: 429" in lowered
        or ("error code: 403" in lowered and "errorcode" in lowered)
    )


def _wait_for_rate_limited_gateway_turn() -> None:
    global _LAST_RATE_LIMITED_CALL_AT

    interval = max(0.0, _env_float("HEA_LLM_MIN_INTERVAL_SECONDS", 20.0))
    if interval <= 0:
        return

    with _RATE_LIMIT_LOCK:
        now = time.monotonic()
        wait_seconds = interval - (now - _LAST_RATE_LIMITED_CALL_AT)
        if wait_seconds > 0:
            print(
                f"[HEA LLM] waiting {wait_seconds:.1f}s before next call "
                "to avoid gateway rate limiting.",
                flush=True,
            )
            time.sleep(wait_seconds)
        _LAST_RATE_LIMITED_CALL_AT = time.monotonic()


def _patch_rate_limit_backoff(llm: Any) -> Any:
    if getattr(llm, "_hea_rate_limit_backoff", False):
        return llm

    original_call: Callable[..., Any] = llm.call

    def call_with_backoff(*args: Any, **kwargs: Any) -> Any:
        max_retries = max(0, _env_int("HEA_LLM_MAX_RETRIES", 4))
        base_delay = max(1.0, _env_float("HEA_LLM_RETRY_BASE_SECONDS", 25.0))
        max_delay = max(base_delay, _env_float("HEA_LLM_RETRY_MAX_SECONDS", 120.0))

        for attempt in range(max_retries + 1):
            _wait_for_rate_limited_gateway_turn()
            try:
                return original_call(*args, **kwargs)
            except Exception as exc:
                if not _is_rate_limit_error(exc) or attempt >= max_retries:
                    raise

                delay = min(max_delay, base_delay * (2 ** attempt))
                print(
                    f"[HEA LLM] gateway was rate-limited; "
                    f"retry {attempt + 1}/{max_retries} in {delay:.1f}s.",
                    flush=True,
                )
                time.sleep(delay)

        raise RuntimeError("HEA LLM backoff exhausted unexpectedly.")

    llm.call = call_with_backoff
    llm._hea_rate_limit_backoff = True
    return llm


def _fallback_models_for_host(host: str, default_model: str) -> List[str]:
    host = host.lower()
    options: List[str] = []

    for key, candidates in _HOST_MODEL_FALLBACKS.items():
        if key in host:
            options.extend(candidates)

    if not options:
        options.extend(["gpt-5.5", "gpt-5.4", "gpt-5.4-mini", default_model])

    if default_model and default_model not in options:
        options.append(default_model)

    deduped: List[str] = []
    for item in options:
        if item not in deduped:
            deduped.append(item)
    return deduped


def discover_gateway_models(
    default_model: str = "gpt-5.2",
    timeout: float = 4.0,
    llm_api: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Best-effort model discovery for the active OpenAI-compatible gateway."""
    load_env_file()
    resolved_api = normalize_llm_api(llm_api)
    if resolved_api == "llmone":
        resolved_base_url = _normalize_openai_base_url(
            base_url
            or os.getenv("LLMONE_BASE_URL")
            or os.getenv("HEA_LLM_BASE_URL")
            or _LLMONE_DEFAULT_BASE_URL
        )
        api_key = _first_env("LLMONE_API_KEY", "HEA_LLM_API_KEY", "LLM_API_KEY", "OPENAI_API_KEY")
    else:
        resolved_base_url = _normalize_openai_base_url(
            base_url
            or os.getenv("OPENAI_BASE_URL")
            or os.getenv("LLM_BASE_URL")
            or os.getenv("HEA_LLM_BASE_URL")
            or os.getenv("AI_STUDIO_BASE_URL")
            or os.getenv("AISTUDIO_BASE_URL")
            or _OPENAI_DEFAULT_BASE_URL
        )
        api_key = _first_env("OPENAI_API_KEY", "LLM_API_KEY", "HEA_LLM_API_KEY", "AI_STUDIO_API_KEY")

    parsed = urlparse(resolved_base_url)
    host = parsed.netloc.lower()
    cache_age = time.time() - float(_MODEL_CATALOG_CACHE.get("cached_at") or 0.0)
    if (
        _MODEL_CATALOG_CACHE.get("payload") is not None
        and _MODEL_CATALOG_CACHE.get("base_url") == resolved_base_url
        and _MODEL_CATALOG_CACHE.get("default_model") == default_model
        and cache_age < 60.0
    ):
        return dict(_MODEL_CATALOG_CACHE["payload"])

    result: Dict[str, Any] = {
        "llm_api": resolved_api,
        "base_url": resolved_base_url,
        "host": host,
        "source": "host_fallback",
        "options": _fallback_models_for_host(host, default_model),
        "configured": bool(api_key),
        "error": None,
    }

    if not resolved_base_url or not api_key:
        if not api_key:
            result["error"] = "missing_api_key"
        _MODEL_CATALOG_CACHE.update(
            {
                "base_url": resolved_base_url,
                "default_model": default_model,
                "payload": dict(result),
                "cached_at": time.time(),
            }
        )
        return result

    models_url = f"{resolved_base_url.rstrip('/')}/models"
    headers = {"Accept": "application/json", "Authorization": f"Bearer {api_key}"}

    try:
        req = request.Request(models_url, headers=headers, method="GET")
        with request.urlopen(req, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8") or "{}")

        data = payload.get("data", []) if isinstance(payload, dict) else []
        live_options = sorted(
            {
                str(item.get("id", "")).strip()
                for item in data
                if isinstance(item, dict) and str(item.get("id", "")).strip()
            }
        )
        if live_options:
            if default_model and default_model not in live_options:
                live_options.insert(0, default_model)
            result["source"] = "gateway_models_api"
            result["options"] = live_options
            result["error"] = None
    except error.HTTPError as exc:
        result["error"] = f"http_{exc.code}"
    except Exception as exc:
        result["error"] = str(exc)

    _MODEL_CATALOG_CACHE.update(
        {
            "base_url": resolved_base_url,
            "default_model": default_model,
            "payload": dict(result),
            "cached_at": time.time(),
        }
    )
    return result


def get_crewai_llm(
    model: str = "gpt-5.2",
    temperature: float = 0.7,
    llm_api: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    provider: Optional[str] = None,
):
    """Create the CrewAI ``LLM`` object.

    ``provider`` is accepted only for older callers; new code should pass
    ``llm_api`` with ``llmone`` or ``openai``.
    """
    try:
        load_env_file()
        from crewai import LLM

        resolved = resolve_llm_connection_config(
            model=model,
            llm_api=llm_api or provider,
            base_url=base_url,
            api_key=api_key,
        )

        llm = LLM(
            model=resolved.model,
            temperature=temperature,
            api_key=resolved.api_key,
            base_url=resolved.base_url,
            api_base=resolved.base_url,
        )
        llm._hea_llm_api = resolved.llm_api
        llm._hea_llm_provider = resolved.llm_api  # compatibility for old diagnostics
        llm._hea_llm_base_url = resolved.base_url
        if resolved.needs_rate_limit_backoff:
            return _patch_rate_limit_backoff(llm)
        return llm
    except ImportError:
        raise ImportError("crewai not installed. Run: pip install crewai")


# Backward-compatible function name.
def get_provider_llm(
    model: str = "gpt-5.2",
    temperature: float = 0.7,
    provider: Optional[str] = None,
    base_url: Optional[str] = None,
):
    return get_crewai_llm(
        model=model,
        temperature=temperature,
        llm_api=provider,
        base_url=base_url,
    )
