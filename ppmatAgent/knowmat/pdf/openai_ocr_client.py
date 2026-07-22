"""OpenAI-compatible vision OCR client for KnowMat.

This backend renders PDF pages to images and sends each page to an
OpenAI-compatible chat completions endpoint. It is intentionally separate from
the PaddleOCR job API: PaddleOCR remains the preferred structured OCR path,
while this backend provides a simple ``api_key + base_url + model`` fallback for
gateways such as AI Studio, DeepSeek-compatible vision endpoints, or OpenAI.
"""

from __future__ import annotations

import base64
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_THINK_TAG_RE = re.compile(r"</?think>", re.IGNORECASE)

_SYSTEM_PROMPT = (
    "You are a precise OCR engine for materials science papers. "
    "Read the page image and return clean Markdown in natural reading order. "
    "Preserve section headings, equations as LaTeX where visible, tables as Markdown tables, "
    "figure/table captions, units, alloy compositions, temperatures, and all numerical values. "
    "Do not summarize. Do not add commentary. Return only Markdown."
)

_USER_PROMPT = (
    "OCR this scientific paper page into Markdown. "
    "Keep the content faithful to the page, including captions, formulas, tables, and references. "
    "If a region is unreadable, mark it as [unreadable] instead of guessing."
)


class OpenAIOCRAPIError(Exception):
    """Raised when an OpenAI-compatible OCR request fails."""


@dataclass(frozen=True)
class OpenAIOCRConfig:
    api_key: str
    base_url: str
    model: str
    max_tokens: int
    temperature: float


def _first_env(*names: str) -> str:
    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return ""


def _normalize_openai_base_url(raw_url: str) -> str:
    cleaned = str(raw_url or "").strip().rstrip("/")
    if not cleaned:
        return cleaned
    parsed = urlparse(cleaned)
    if parsed.scheme and parsed.netloc and parsed.path in {"", "/"}:
        return f"{cleaned}/v1"
    return cleaned


def _default_model_for_base_url(base_url: str) -> str:
    lowered = base_url.lower()
    if "aistudio.baidu.com" in lowered:
        return "ernie-5.0-thinking-preview"
    return "gpt-4o-mini"


def resolve_openai_ocr_config(
    *,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
) -> OpenAIOCRConfig:
    resolved_base_url = _normalize_openai_base_url(
        base_url
        or os.getenv("KNOWMAT_OCR_BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
        or os.getenv("LLM_BASE_URL")
        or os.getenv("AI_STUDIO_BASE_URL")
        or os.getenv("AISTUDIO_BASE_URL")
        or "https://api.openai.com/v1"
    )
    resolved_api_key = api_key or _first_env(
        "KNOWMAT_OCR_API_KEY",
        "OPENAI_API_KEY",
        "LLM_API_KEY",
        "AI_STUDIO_API_KEY",
        "AISTUDIO_ACCESS_TOKEN",
    )
    if not resolved_api_key:
        raise OpenAIOCRAPIError(
            "OpenAI-compatible OCR requires KNOWMAT_OCR_API_KEY. "
            "OPENAI_API_KEY, LLM_API_KEY, and AI_STUDIO_API_KEY are also accepted."
        )

    resolved_model = (
        model
        or os.getenv("KNOWMAT_OCR_MODEL")
        or os.getenv("VLM_MODEL")
        or os.getenv("OPENAI_MODEL")
        or os.getenv("LLM_MODEL")
        or _default_model_for_base_url(resolved_base_url)
    ).strip()
    if not resolved_model:
        raise OpenAIOCRAPIError("OpenAI-compatible OCR requires KNOWMAT_OCR_MODEL or LLM_MODEL.")

    try:
        max_tokens = int(float(os.getenv("KNOWMAT_OCR_MAX_TOKENS", "4096")))
    except Exception:
        max_tokens = 4096
    try:
        temperature = float(os.getenv("KNOWMAT_OCR_TEMPERATURE", "0"))
    except Exception:
        temperature = 0.0

    return OpenAIOCRConfig(
        api_key=resolved_api_key,
        base_url=resolved_base_url,
        model=resolved_model,
        max_tokens=max_tokens,
        temperature=temperature,
    )


def _image_media_type(image_path: Path) -> str:
    suffix = image_path.suffix.lower()
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }.get(suffix, "image/png")


def _encode_image_base64(image_path: Path) -> str:
    try:
        return base64.b64encode(image_path.read_bytes()).decode("utf-8")
    except OSError as exc:
        raise OpenAIOCRAPIError(f"Cannot read rendered page image {image_path}: {exc}") from exc


def _sanitize_markdown(text: str) -> str:
    cleaned = str(text or "").strip()
    if not cleaned:
        return ""
    cleaned = _THINK_BLOCK_RE.sub("", cleaned)
    cleaned = _THINK_TAG_RE.sub("", cleaned)
    cleaned = re.sub(r"^```(?:markdown|md)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = re.sub(r"\n{4,}", "\n\n\n", cleaned)
    return cleaned.strip()


def _is_retryable_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return (
        "429" in msg
        or "rate limit" in msg
        or "too many" in msg
        or "timeout" in msg
        or "temporarily" in msg
        or "connection" in msg
    )


class OpenAIOCRClient:
    """OCR page images with an OpenAI-compatible vision chat endpoint."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        self.config = resolve_openai_ocr_config(
            api_key=api_key,
            base_url=base_url,
            model=model,
        )

    def extract_page_markdown(self, image_path: Path, *, page_number: int) -> str:
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:
            raise OpenAIOCRAPIError(
                "The openai package is required for OpenAI-compatible OCR. "
                "Install ppmatAgent optional dependencies first."
            ) from exc

        b64 = _encode_image_base64(image_path)
        media_type = _image_media_type(image_path)
        client = OpenAI(api_key=self.config.api_key, base_url=self.config.base_url)
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{media_type};base64,{b64}"},
                    },
                    {"type": "text", "text": f"{_USER_PROMPT}\n\nPage number: {page_number}"},
                ],
            },
        ]
        create_kwargs: Dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
        }

        max_retries = max(0, int(float(os.getenv("KNOWMAT_OCR_MAX_RETRIES", "3"))))
        base_wait = max(1.0, float(os.getenv("KNOWMAT_OCR_RETRY_BASE_SECONDS", "3")))
        last_exc: Optional[Exception] = None

        for attempt in range(max_retries + 1):
            try:
                response = client.chat.completions.create(**create_kwargs)
                content = response.choices[0].message.content or ""
                markdown = _sanitize_markdown(content)
                if markdown:
                    return markdown
                last_exc = OpenAIOCRAPIError("empty OCR response")
            except Exception as exc:
                last_exc = exc
                if not _is_retryable_error(exc) or attempt >= max_retries:
                    break
            if attempt < max_retries:
                wait = base_wait * (2 ** attempt)
                logger.warning(
                    "[OpenAI OCR] page %s failed, retry %d/%d in %.1fs: %s",
                    page_number,
                    attempt + 1,
                    max_retries,
                    wait,
                    last_exc,
                )
                time.sleep(wait)

        raise OpenAIOCRAPIError(f"OpenAI-compatible OCR failed for page {page_number}: {last_exc}")


def page_markdown_to_ocr_items(markdown: str, page_number: int) -> List[Dict[str, Any]]:
    """Convert page Markdown to simple KnowMat OCR items."""
    items: List[Dict[str, Any]] = []
    blocks = [block.strip() for block in re.split(r"\n\s*\n", markdown or "") if block.strip()]
    for block in blocks:
        if "|" in block and "\n" in block:
            items.append(
                {
                    "typer": "table",
                    "page": page_number,
                    "data": {"text": block, "source": "openai_compatible_ocr"},
                }
            )
        else:
            items.append(
                {
                    "typer": "paragraph",
                    "page": page_number,
                    "text": block,
                    "source": "openai_compatible_ocr",
                }
            )
    return items


def build_openai_ocr_result(
    page_markdowns: List[Dict[str, Any]],
    *,
    pdf_path: str,
    config: OpenAIOCRConfig,
) -> tuple[str, Dict[str, Any], List[Dict[str, Any]]]:
    page_markdowns = sorted(page_markdowns, key=lambda item: int(item["page"]))
    page_blocks: List[str] = []
    page_level_metadata: List[Dict[str, Any]] = []
    ocr_items: List[Dict[str, Any]] = []

    for item in page_markdowns:
        page = int(item["page"])
        markdown = _sanitize_markdown(str(item.get("markdown") or ""))
        page_blocks.append(f"## Page {page}\n\n{markdown}")
        lines = [line for line in markdown.splitlines() if line.strip()]
        page_level_metadata.append(
            {
                "page": page,
                "line_count": len(lines),
                "header_text": "\n".join(lines[:5]),
                "footer_text": "\n".join(lines[-3:]) if len(lines) >= 3 else "",
            }
        )
        ocr_items.extend(page_markdown_to_ocr_items(markdown, page))

    text = "\n\n".join(page_blocks).strip()
    metadata: Dict[str, Any] = {
        "backend": "openai_compatible_ocr",
        "source_file": str(pdf_path),
        "pages": len(page_markdowns),
        "model": config.model,
        "base_url": config.base_url,
        "page_level_metadata": page_level_metadata,
        "ocr_quality": {
            "ocr_avg_confidence": None,
            "ocr_low_confidence_pages": [],
            "table_count": sum(1 for item in ocr_items if item.get("typer") == "table"),
            "formula_count": 0,
            "ppstructure_status": "not_applicable",
            "ppstructure_detail": "OpenAI-compatible OCR returns Markdown text, not PaddleOCR layout objects.",
            "ppstructure_replacements": 0,
        },
    }
    return text, metadata, ocr_items
