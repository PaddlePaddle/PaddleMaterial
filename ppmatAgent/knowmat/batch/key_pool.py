"""Multi-key adaptive rotation pool for OCR API tokens."""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import List, Optional

from ppmatAgent.knowmat.batch.models import KeyInfo

logger = logging.getLogger(__name__)

_DEFAULT_COOLDOWN_SEC = 60.0
_PADDLEOCR_DEFAULT_URL = "https://paddleocr.aistudio-app.com/api/v2/ocr/jobs"


class NoKeysAvailableError(Exception):
    """Raised when no API keys are configured for the requested vendor."""


class KeyPool:
    """Manages multiple API keys with rate-limit-aware adaptive rotation.

    Selection strategy:
    1. Filter out keys in cooldown (rate-limited recently)
    2. Among available keys, pick the least-recently-used one
    3. If all keys are in cooldown, wait for the earliest recovery
    """

    def __init__(self, keys: List[KeyInfo], default_cooldown_sec: float = _DEFAULT_COOLDOWN_SEC):
        if not keys:
            raise NoKeysAvailableError("No API keys provided to KeyPool")
        self._keys = {k.key_id: k for k in keys}
        self._default_cooldown = default_cooldown_sec
        self._last_used: dict[str, float] = {k.key_id: 0.0 for k in keys}
        self._lock = asyncio.Lock()

    @property
    def size(self) -> int:
        return len(self._keys)

    def get_healthy_count(self) -> int:
        now = time.time()
        return sum(1 for k in self._keys.values() if k.cooldown_until <= now)

    async def acquire(self) -> KeyInfo:
        """Acquire a healthy key (LRU among non-cooldown). Waits if all in cooldown."""
        while True:
            async with self._lock:
                now = time.time()
                candidates = [
                    k for k in self._keys.values()
                    if k.cooldown_until <= now
                ]
                if candidates:
                    best = min(candidates, key=lambda k: self._last_used[k.key_id])
                    self._last_used[best.key_id] = now
                    best.total_requests += 1
                    return best

            # All keys in cooldown — find the earliest recovery time
            now = time.time()
            earliest = min(k.cooldown_until for k in self._keys.values())
            wait_time = max(0.1, earliest - now)
            logger.warning(
                "All %d keys in cooldown. Waiting %.1fs for recovery...",
                len(self._keys), wait_time,
            )
            await asyncio.sleep(wait_time)

    def release(self, key_id: str, success: bool = True) -> None:
        """Mark key usage complete. On failure, increment error counter."""
        key = self._keys.get(key_id)
        if not key:
            return
        if not success:
            key.total_errors += 1

    def report_rate_limit(self, key_id: str, retry_after: float = 0) -> None:
        """Put a key into cooldown after receiving a rate-limit response."""
        key = self._keys.get(key_id)
        if not key:
            return
        cooldown = retry_after if retry_after > 0 else self._default_cooldown
        key.cooldown_until = time.time() + cooldown
        logger.warning(
            "Key %s rate-limited. Cooldown for %.0fs (until %.0f)",
            key_id, cooldown, key.cooldown_until,
        )

    def get_status_summary(self) -> str:
        """Human-readable status for progress reporting."""
        now = time.time()
        healthy = sum(1 for k in self._keys.values() if k.cooldown_until <= now)
        total = len(self._keys)
        return f"{healthy}/{total} healthy"

    @classmethod
    def from_env(cls, vendor: str) -> "KeyPool":
        """Construct KeyPool from environment variables.

        Supports:
          PADDLEOCR_API_TOKENS=token1,token2,token3  (comma-separated, new)
          PADDLEOCR_API_TOKEN=single_token            (legacy single key)
          MINERU_API_KEYS=key1,key2                   (comma-separated)
          MINERU_API_KEY=single_key                   (legacy single key)
        """
        keys: List[KeyInfo] = []

        if vendor == "paddleocr":
            tokens_str = os.getenv("PADDLEOCR_API_TOKENS", "").strip()
            if tokens_str:
                tokens = [t.strip() for t in tokens_str.split(",") if t.strip()]
            else:
                single = os.getenv("PADDLEOCR_API_TOKEN", "").strip()
                tokens = [single] if single else []

            base_url = os.getenv(
                "PADDLEOCR_API_BASE_URL", _PADDLEOCR_DEFAULT_URL
            )
            for i, token in enumerate(tokens):
                keys.append(KeyInfo(
                    key_id=f"paddle_{i}",
                    token=token,
                    base_url=base_url,
                ))

        elif vendor == "mineru":
            keys_str = os.getenv("MINERU_API_KEYS", "").strip()
            if keys_str:
                tokens = [t.strip() for t in keys_str.split(",") if t.strip()]
            else:
                single = os.getenv("MINERU_API_KEY", "").strip()
                tokens = [single] if single else []

            base_url = "https://mineru.net"
            for i, token in enumerate(tokens):
                keys.append(KeyInfo(
                    key_id=f"mineru_{i}",
                    token=token,
                    base_url=base_url,
                ))

        if not keys:
            raise NoKeysAvailableError(
                f"No API keys found for vendor '{vendor}'. "
                f"Set PADDLEOCR_API_TOKENS or MINERU_API_KEYS in .env"
            )

        logger.info("KeyPool initialized: %d keys for vendor '%s'", len(keys), vendor)
        return cls(keys)
