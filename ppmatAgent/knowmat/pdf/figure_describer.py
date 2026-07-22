"""Multimodal LLM figure description for KnowMat.

Uses the same OpenAI-compatible endpoint configured in .env (LLM_API_KEY,
LLM_BASE_URL, LLM_MODEL) to generate a concise textual description of a
figure image cropped from a scientific paper.

This module is intentionally dependency-light: it only requires ``openai``
(already a transitive dependency via langchain-openai) and the standard
library.  It never raises — failures are logged and an empty string is
returned so the main pipeline is never blocked.
"""

from __future__ import annotations

import base64
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from ppmatAgent.knowmat.pdf.figure_items import iter_resolved_figure_items, normalize_figure_ocr_items

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a materials science figure analyst. "
    "Describe the key information shown in this figure from a scientific paper. "
    "Focus on: microstructure features, scale bars, phase labels, measurement values, trends, "
    "axis labels, data points, and any quantitative information visible. "
    "Be thorough — extract ALL numerical values, labels, and data visible in the figure. "
    "For multi-panel figures, describe each panel. "
    "Target length: 200-500 words. Do not repeat the caption verbatim. "
    "Return only the final description text. Do not include reasoning, analysis, or <think> tags."
)

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_THINK_TAG_RE = re.compile(r"</?think>", re.IGNORECASE)

_VLM_MAX_RETRIES = 4
_VLM_RETRY_BACKOFF = 3.0


def _is_rate_limit_error(exc: Exception) -> bool:
    """Check if the exception is a rate limit / too-frequent error."""
    msg = str(exc).lower()
    return "429" in msg or "频繁" in msg or "rate" in msg or "too many" in msg


def _is_non_retryable_error(exc: Exception) -> bool:
    """Check if the error will never succeed on retry (so skip immediately).

    Notably the provider's content-moderation rejection: some scientific
    micrographs are flagged "image is sensitive" (HTTP 422 / unprocessable),
    which returns the same 422 on every retry — retrying just wastes time.
    """
    msg = str(exc).lower()
    return (
        "422" in msg
        or "unprocessable" in msg
        or "sensitive" in msg
        or "new_sensitive" in msg
    )


class _VlmKeyPool:
    """Thread-safe round-robin pool for VLM API keys.

    Reads keys from (in priority order):
      VLM_API_KEYS=key1,key2,key3       comma-separated list
      VLM_API_KEY_1, VLM_API_KEY_2, …  numbered env vars
      VLM_API_KEY / LLM_API_KEY         single-key fallback

    On a rate-limit hit the caller should call next_key() to get the
    next key immediately instead of sleeping on the same exhausted key.
    Only when all keys have been tried in one round does the pool back off.
    """

    def __init__(self) -> None:
        self._keys: list[str] = []
        self._idx: int = 0
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        keys: list[str] = []
        multi = os.getenv("VLM_API_KEYS", "")
        if multi:
            keys.extend(k.strip() for k in multi.split(",") if k.strip())
        for i in range(1, 50):
            k = os.getenv(f"VLM_API_KEY_{i}", "")
            if k:
                keys.append(k)
            else:
                break
        single = (
            os.getenv("VLM_API_KEY")
            or os.getenv("LLM_API_KEY")
            or os.getenv("OPENAI_API_KEY", "")
        )
        if single and single not in keys:
            keys.append(single)
        self._keys = keys
        if keys:
            logger.info("VlmKeyPool loaded %d key(s)", len(keys))

    @property
    def size(self) -> int:
        return len(self._keys)

    def all_keys(self) -> list[str]:
        return list(self._keys)

    def next_key(self) -> str:
        with self._lock:
            if not self._keys:
                return ""
            key = self._keys[self._idx % len(self._keys)]
            self._idx += 1
            return key


# Module-level singleton — shared across all threads in the process.
_VLM_POOL: _VlmKeyPool | None = None
_VLM_POOL_LOCK = threading.Lock()


def _get_vlm_pool() -> _VlmKeyPool:
    global _VLM_POOL
    if _VLM_POOL is None:
        with _VLM_POOL_LOCK:
            if _VLM_POOL is None:
                _VLM_POOL = _VlmKeyPool()
    return _VLM_POOL


def _vlm_call_with_pool(
    base_url: str | None,
    create_kwargs: dict,
    image_path,
) -> str:
    """Call VLM API with multi-key rotation and exponential backoff.

    Strategy:
      - Cycle through all configured keys on rate-limit errors (immediate,
        no sleep) — only sleep once every key in the pool has been tried.
      - For non-rate-limit errors, apply per-attempt backoff as before.
    """
    try:
        from openai import OpenAI  # type: ignore
    except ImportError:
        logger.warning("openai package not available; skipping figure description.")
        return ""

    pool = _get_vlm_pool()
    keys = pool.all_keys()
    if not keys:
        logger.warning("No VLM API keys configured.")
        return ""

    n = len(keys)
    max_attempts = n * _VLM_MAX_RETRIES
    last_exc: Exception | None = None

    for attempt in range(max_attempts):
        key = keys[attempt % n]
        client_kwargs: dict = {"api_key": key}
        if base_url:
            client_kwargs["base_url"] = base_url

        try:
            client = OpenAI(**client_kwargs)
            response = client.chat.completions.create(**create_kwargs)
            content = response.choices[0].message.content or ""
            if content.strip():
                return content
            logger.warning(
                "VLM empty response for %s (key %d/%d, attempt %d/%d)",
                image_path, attempt % n + 1, n, attempt + 1, max_attempts,
            )
        except Exception as exc:
            last_exc = exc
            if _is_non_retryable_error(exc):
                # e.g. content-moderation "image is sensitive" (422): same result
                # every retry → give up now so the pipeline moves on.
                logger.warning(
                    "VLM permanently rejected %s (non-retryable): %s",
                    image_path, exc,
                )
                raise
            is_rate = _is_rate_limit_error(exc)
            logger.warning(
                "VLM %s for %s (key %d/%d, attempt %d/%d): %s",
                "rate-limited" if is_rate else "failed",
                image_path, attempt % n + 1, n, attempt + 1, max_attempts, exc,
            )
            if is_rate:
                # Rotate to the next key immediately (no sleep within a round).
                # If this was the last key in the round, all keys are exhausted —
                # back off before starting the next round.
                if (attempt + 1) % n == 0:
                    round_num = (attempt + 1) // n
                    wait = _VLM_RETRY_BACKOFF * (2 ** (round_num - 1))  # 3, 6, 12 …
                    logger.info(
                        "All %d VLM key(s) rate-limited (round %d), "
                        "waiting %.1fs before next round…",
                        n, round_num, wait,
                    )
                    time.sleep(wait)
                continue  # try next key / next round immediately

            # Non-rate-limit error: standard per-attempt backoff
            if attempt < max_attempts - 1:
                wait = _VLM_RETRY_BACKOFF * (2 ** (attempt % _VLM_MAX_RETRIES))
                time.sleep(wait)

    if last_exc:
        raise last_exc
    return ""
    """Check if the exception is a rate limit / too-frequent error."""
    msg = str(exc).lower()
    return "429" in msg or "频繁" in msg or "rate" in msg or "too many" in msg


def _vlm_call_with_retry(client, create_kwargs: dict, image_path, max_retries: int = _VLM_MAX_RETRIES) -> str:
    """Call VLM API with exponential backoff retry on transient failures."""
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(**create_kwargs)
            content = response.choices[0].message.content or ""
            if content.strip():
                return content
            logger.warning("VLM returned empty content for %s (attempt %d/%d)", image_path, attempt + 1, max_retries)
        except Exception as exc:
            last_exc = exc
            logger.warning(
                "VLM call failed for %s (attempt %d/%d): %s",
                image_path, attempt + 1, max_retries, exc,
            )
        if attempt < max_retries - 1:
            base_wait = _VLM_RETRY_BACKOFF * (2 ** attempt)
            if last_exc and _is_rate_limit_error(last_exc):
                base_wait = max(base_wait, 8.0 * (attempt + 1))
            time.sleep(base_wait)
    if last_exc:
        raise last_exc
    return ""


def _sanitize_figure_description(text: str) -> str:
    """Strip reasoning tags and keep only the user-facing description."""
    cleaned = str(text or "").strip()
    if not cleaned:
        return ""
    cleaned = _THINK_BLOCK_RE.sub("", cleaned)
    cleaned = _THINK_TAG_RE.sub("", cleaned)
    cleaned = re.sub(r"^\s*final answer:\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


def _encode_image_base64(image_path: Path) -> Optional[str]:
    """Read an image file and return its base64-encoded content."""
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except OSError as exc:
        logger.warning("Cannot read figure image %s: %s", image_path, exc)
        return None


def _image_media_type(image_path: Path) -> str:
    suffix = image_path.suffix.lower()
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }.get(suffix, "image/jpeg")


def describe_figure_image(
    image_path: Path,
    caption: str = "",
    *,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> str:
    """Generate a textual description of a figure image using a multimodal LLM.

    Parameters
    ----------
    image_path:
        Absolute or relative path to the figure image (JPEG/PNG).
    caption:
        Optional figure caption text for context (e.g. "Fig. 1. SEM image of...").
    model:
        Model ID to use.  Defaults to ``LLM_MODEL`` env var.
    api_key:
        API key.  Defaults to ``LLM_API_KEY`` env var.
    base_url:
        Base URL for OpenAI-compatible endpoint.  Defaults to ``LLM_BASE_URL`` env var.

    Returns
    -------
    str
        A concise description of the figure, or empty string on failure.
    """
    resolved_path = Path(image_path)
    if not resolved_path.is_file():
        logger.debug("Figure image not found, skipping description: %s", image_path)
        return ""

    b64 = _encode_image_base64(resolved_path)
    if b64 is None:
        return ""

    _api_key = api_key or _get_vlm_pool().next_key()
    _base_url = base_url or os.getenv("VLM_BASE_URL") or os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    _model = model or os.getenv("VLM_MODEL") or os.getenv("LLM_MODEL", "")

    if not _api_key:
        logger.warning("No VLM_API_KEY/LLM_API_KEY configured; skipping figure description.")
        return ""

    try:
        from openai import OpenAI  # type: ignore
    except ImportError:
        logger.warning("openai package not available; skipping figure description.")
        return ""

    user_text = "Please describe this scientific figure."
    if caption:
        user_text = f"Caption context: {caption}\n\n{user_text}"
    user_text += "\nReturn only the final description. Do not include <think> tags or hidden reasoning."

    media_type = _image_media_type(resolved_path)

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{media_type};base64,{b64}",
                    },
                },
                {"type": "text", "text": user_text},
            ],
        },
    ]

    create_kwargs: dict = {
        "model": _model,
        "messages": messages,
        "max_tokens": 2048,
        "temperature": 0.2,
    }

    try:
        content = _vlm_call_with_pool(_base_url, create_kwargs, image_path)
        content = _sanitize_figure_description(content)
        logger.debug("Figure description generated (%d chars) for %s", len(content), image_path)
        return content
    except Exception as exc:
        logger.warning("Figure description LLM call failed for %s: %s", image_path, exc)
        return ""


def describe_figure_image_with_context(
    image_path: Path,
    caption: str = "",
    related_sentences: Optional[List[str]] = None,
    *,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> str:
    """Generate a figure description using VLM with additional paper context.

    Like :func:`describe_figure_image` but includes related sentences from the
    paper body as extra context so the VLM can produce a more precise,
    paper-specific description.
    """
    resolved_path = Path(image_path)
    if not resolved_path.is_file():
        logger.debug("Figure image not found, skipping description: %s", image_path)
        return ""

    b64 = _encode_image_base64(resolved_path)
    if b64 is None:
        return ""

    _api_key = api_key or _get_vlm_pool().next_key()
    _base_url = base_url or os.getenv("VLM_BASE_URL") or os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    _model = model or os.getenv("VLM_MODEL") or os.getenv("LLM_MODEL", "")

    if not _api_key:
        logger.warning("No VLM_API_KEY/LLM_API_KEY configured; skipping figure description.")
        return ""

    try:
        from openai import OpenAI  # type: ignore
    except ImportError:
        logger.warning("openai package not available; skipping figure description.")
        return ""

    user_text = ""
    if caption:
        user_text += f"Caption context: {caption}\n\n"
    if related_sentences:
        user_text += "Related text from the paper body (ranked by relevance to this figure):\n"
        for i, sent in enumerate(related_sentences[:5], 1):
            user_text += f"{i}. {sent}\n"
        user_text += "\n"
    user_text += (
        "Using the image, caption, and related text above as context, "
        "describe the key scientific information shown in this figure. "
        "Be concise (2-4 sentences).\n"
        "Return only the final description. Do not include <think> tags or hidden reasoning."
    )

    media_type = _image_media_type(resolved_path)
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{media_type};base64,{b64}",
                    },
                },
                {"type": "text", "text": user_text},
            ],
        },
    ]

    create_kwargs: dict = {
        "model": _model,
        "messages": messages,
        "max_tokens": 2048,
        "temperature": 0.2,
    }

    try:
        content = _vlm_call_with_pool(_base_url, create_kwargs, image_path)
        content = _sanitize_figure_description(content)
        logger.debug("Figure description (with context) generated (%d chars) for %s", len(content), image_path)
        return content
    except Exception as exc:
        logger.warning("Figure description (with context) LLM call failed for %s: %s", image_path, exc)
        return ""


def inject_figure_descriptions(
    text: str,
    ocr_items: List[Dict[str, Any]],
    *,
    max_workers: int = 2,
    paper_id: str = "",
    output_dir: Optional[str] = None,
) -> str:
    """Insert multimodal LLM descriptions above each figure caption in *text*.

    Uses Approach C (CLIP alignment + VLM with context) when possible: CLIP
    finds the most relevant body sentences for each figure, then the VLM
    generates a description using both the image and those sentences as context.
    Falls back to Approach A (VLM-only) if CLIP alignment is unavailable.

    Called from the extraction stage (not OCR), so every LLM call here is
    intentional and gated by ``settings.figure_description_enabled``.

    LLM calls for individual figures run in parallel (up to *max_workers*
    threads) to reduce wall-clock time.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    normalize_figure_ocr_items(ocr_items)
    image_items = iter_resolved_figure_items(ocr_items)
    if not image_items:
        return text

    # --- CLIP alignment (Approach C) ---
    alignment_by_fig: Dict[str, List[str]] = {}
    try:
        from ppmatAgent.knowmat.image_text_alignment.aligner import ImageTextAligner

        _output_dir = output_dir
        if not _output_dir and image_items:
            first_img = image_items[0].get("data", {}).get("image_path", "")
            if first_img:
                _output_dir = str(Path(first_img).parent)
        _paper_id = paper_id or "unknown"

        aligner = ImageTextAligner(
            model="clip", device="cpu", top_k=5,
            min_score=0.0, batch_size=32, caption_blend=0.0,
        )
        alignments = aligner.align(ocr_items, paper_id=_paper_id, output_dir=_output_dir)
        for alignment in alignments:
            fig_num = alignment.figure_num
            if fig_num:
                sentences = [s.text for s in alignment.related_sentences[:5]]
                alignment_by_fig[fig_num] = sentences
        if alignment_by_fig:
            logger.info("CLIP alignment completed: %d figures aligned.", len(alignment_by_fig))
    except Exception as exc:
        logger.warning("CLIP alignment unavailable, using VLM-only fallback: %s", exc)

    # Collect valid items to describe, routing each to a processing mode:
    #   "chart" → VLM digitization (line/bar) → '> [Figure N VLM-digitized]:'
    #   "prose" → existing VLM prose description → '> [Figure N AI Description]:'
    # Routing key: PaddleOCR-VL crop filename (chart_box vs image_box).
    from ppmatAgent.knowmat.pdf.chart_digitizer import is_chart_crop
    from ppmatAgent.knowmat.app_config import settings as _settings

    _chart_enabled = getattr(_settings, "chart_digitization_enabled", True)

    to_describe: List[Dict[str, Any]] = []
    for item in image_items:
        data = item.get("data", {})
        raw_path = data.get("image_path", "")
        img_path = Path(raw_path)
        if not img_path.is_file():
            logger.debug("Figure image not found, skipping description injection: %s", img_path)
            continue
        figure_num = data.get("figure_num", "")
        mode = "chart" if (_chart_enabled and is_chart_crop(raw_path)) else "prose"
        # Skip if this figure already has the corresponding injected block.
        if figure_num:
            if mode == "chart" and f"[Figure {figure_num} VLM-digitized" in text:
                continue
            if mode == "prose" and f"> [Figure {figure_num} AI Description]:" in text:
                continue
        item["_describe_mode"] = mode
        to_describe.append(item)

    if not to_describe:
        return text

    # Parallel LLM calls.  chart items → digitize_chart_image + format_digitized_block;
    # prose items → existing VLM prose (Approach C: VLM + CLIP context when available).
    descriptions: Dict[int, str] = {}

    def _describe(idx: int, item: Dict[str, Any]) -> tuple:
        data = item.get("data", {})
        img_path = Path(data.get("image_path", ""))
        caption = data.get("caption", "")
        figure_num = data.get("figure_num", "")
        mode = item.get("_describe_mode", "prose")

        if mode == "chart":
            from ppmatAgent.knowmat.pdf.chart_digitizer import (
                digitize_chart_image,
                format_digitized_block,
            )
            result = digitize_chart_image(img_path, caption=caption)
            if result:
                block = format_digitized_block(result, figure_num=figure_num)
                if block:
                    # Strip the leading marker; the injection loop re-wraps.
                    return idx, ("chart", block)
            # Non-digitizable chart (xrd/micrograph/other) → fall back to prose
            # so the figure still gets *some* context.

        related = alignment_by_fig.get(figure_num, []) if figure_num else []
        if related:
            desc = describe_figure_image_with_context(
                img_path, caption=caption, related_sentences=related
            )
        else:
            desc = describe_figure_image(img_path, caption=caption)
        desc = _sanitize_figure_description(desc)
        return idx, ("prose", desc)

    workers = min(max_workers, len(to_describe))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_describe, i, item): i for i, item in enumerate(to_describe)}
        for future in as_completed(futures):
            try:
                idx, payload = future.result()
                if payload and payload[1]:
                    descriptions[idx] = payload
            except Exception as exc:
                logger.warning("Figure description failed: %s", exc)

    # Insert descriptions into text (sequential to keep positions correct).
    for idx in sorted(descriptions.keys()):
        item = to_describe[idx]
        data = item.get("data", {})
        figure_num = data.get("figure_num", "")
        kind, content = descriptions[idx]

        if kind == "chart":
            # content is already a fully-formed '> [Figure N VLM-digitized ...]:' block
            description_block = content.rstrip() + "\n\n"
        else:
            label = f"Figure {figure_num}" if figure_num else "Figure"
            description_block = f"> [{label} AI Description]: {content}\n\n"

        if description_block.strip() in text:
            continue

        if figure_num:
            pattern = re.compile(
                r"((?:Fig\.?\s*|Figure\s*)" + re.escape(str(figure_num)) + r"[\s\.\:])",
                re.IGNORECASE,
            )
            match = pattern.search(text)
            if match:
                line_start = text.rfind("\n", 0, match.start()) + 1
                insert_pos = line_start if line_start < match.start() else match.start()
                text = text[:insert_pos] + description_block + text[insert_pos:]
                logger.debug(
                    "Injected %s for Figure %s at position %d", kind, figure_num, insert_pos
                )
                continue

        text = text + "\n\n" + description_block

    return text

