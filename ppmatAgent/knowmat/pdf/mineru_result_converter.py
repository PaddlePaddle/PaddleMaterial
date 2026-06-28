"""Convert MinerU API output (content_list.json + full.md) to KnowMat internal format."""

from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .html_cleaner import convert_html_to_markdown

logger = logging.getLogger(__name__)

_HEADING_PREFIX = {1: "# ", 2: "## ", 3: "### ", 4: "#### ", 5: "##### ", 6: "###### "}


def _pages_range_from_indices(page_indices: Optional[List[int]]) -> Optional[str]:
    """Convert 1-based page index list to MinerU page_ranges string (e.g. '1-3,5')."""
    if not page_indices:
        return None
    ranges: List[str] = []
    start = page_indices[0]
    end = start
    for p in page_indices[1:]:
        if p == end + 1:
            end = p
        else:
            ranges.append(f"{start}-{end}" if start != end else str(start))
            start = end = p
    ranges.append(f"{start}-{end}" if start != end else str(start))
    return ",".join(ranges)


def _convert_item(item: Dict[str, Any], extracted_dir: Path, figures_dest: Optional[Path]) -> Optional[Dict[str, Any]]:
    """Convert a single MinerU content_list item to KnowMat ocr_item format."""
    item_type = item.get("type", "")
    page_idx = item.get("page_idx", 0)
    bbox = item.get("bbox")

    if item_type == "text":
        text = item.get("text", "").strip()
        if not text:
            return None
        text_level = item.get("text_level", 0)
        result: Dict[str, Any] = {"typer": "paragraph", "text": text, "page": page_idx + 1}
        if bbox:
            result["bbox"] = bbox
        if text_level and text_level > 0:
            result["block_label"] = "title"
        return result

    if item_type == "table":
        table_body = item.get("table_body", "")
        caption_parts = item.get("table_caption", [])
        caption = " ".join(caption_parts) if isinstance(caption_parts, list) else str(caption_parts or "")
        md_text = convert_html_to_markdown(table_body) if table_body else ""
        data: Dict[str, Any] = {"text": md_text, "raw_html": table_body}
        if caption:
            data["caption"] = caption
        result = {"typer": "table", "data": data, "page": page_idx + 1}
        if bbox:
            result["bbox"] = bbox
        result["block_label"] = "table"
        return result

    if item_type == "image" or item_type == "chart":
        img_path_str = item.get("img_path", "")
        caption_parts = item.get("image_caption") or item.get("chart_caption") or []
        caption = " ".join(caption_parts) if isinstance(caption_parts, list) else str(caption_parts or "")
        resolved_path = ""
        if img_path_str:
            src = extracted_dir / img_path_str
            if src.is_file() and figures_dest:
                figures_dest.mkdir(parents=True, exist_ok=True)
                dest_file = figures_dest / src.name
                if not dest_file.exists():
                    shutil.copy2(src, dest_file)
                resolved_path = str(dest_file)
            elif src.is_file():
                resolved_path = str(src)
        data = {"image_path": resolved_path, "caption": caption}
        result = {"typer": "image", "data": data, "page": page_idx + 1}
        if bbox:
            result["bbox"] = bbox
        result["block_label"] = "figure" if item_type == "image" else "chart"
        return result

    if item_type == "equation":
        text = item.get("text", "").strip()
        data = {"text": text, "latex": text}
        result = {"typer": "formula", "data": data, "page": page_idx + 1}
        if bbox:
            result["bbox"] = bbox
        result["block_label"] = "formula"
        return result

    if item_type == "code":
        code_body = item.get("code_body", "").strip()
        sub_type = item.get("sub_type", "code")
        text = f"```{sub_type}\n{code_body}\n```" if code_body else ""
        if not text:
            return None
        result = {"typer": "paragraph", "text": text, "page": page_idx + 1}
        if bbox:
            result["bbox"] = bbox
        return result

    if item_type == "list":
        list_items = item.get("list_items", [])
        if isinstance(list_items, list):
            text = "\n".join(f"- {li}" for li in list_items)
        else:
            text = str(list_items)
        if not text.strip():
            return None
        result = {"typer": "paragraph", "text": text.strip(), "page": page_idx + 1}
        if bbox:
            result["bbox"] = bbox
        return result

    if item_type in ("header", "footer", "page_number", "aside_text", "page_footnote"):
        text = item.get("text", "").strip()
        if not text:
            return None
        result = {"typer": "paragraph", "text": text, "page": page_idx + 1, "is_layout_noise": True}
        if bbox:
            result["bbox"] = bbox
        result["block_label"] = item_type
        return result

    text = item.get("text", "").strip()
    if text:
        result = {"typer": "paragraph", "text": text, "page": page_idx + 1}
        if bbox:
            result["bbox"] = bbox
        return result
    return None


def _build_extracted_text_from_content_list(content_list: List[Dict[str, Any]]) -> str:
    """Build page-separated markdown text from content_list items."""
    pages: Dict[int, List[str]] = {}
    for item in content_list:
        page_idx = item.get("page_idx", 0)
        item_type = item.get("type", "")

        if item_type in ("header", "footer", "page_number", "aside_text", "page_footnote"):
            continue

        if item_type == "text":
            text = item.get("text", "").strip()
            text_level = item.get("text_level", 0)
            if text_level and text_level > 0:
                prefix = _HEADING_PREFIX.get(text_level, "## ")
                text = f"{prefix}{text}"
            if text:
                pages.setdefault(page_idx, []).append(text)
        elif item_type == "table":
            table_body = item.get("table_body", "")
            md = convert_html_to_markdown(table_body) if table_body else ""
            if md:
                pages.setdefault(page_idx, []).append(md)
        elif item_type == "equation":
            text = item.get("text", "").strip()
            if text:
                pages.setdefault(page_idx, []).append(f"$$ {text} $$")
        elif item_type == "code":
            code_body = item.get("code_body", "").strip()
            if code_body:
                pages.setdefault(page_idx, []).append(f"```\n{code_body}\n```")
        elif item_type == "list":
            list_items = item.get("list_items", [])
            if isinstance(list_items, list) and list_items:
                pages.setdefault(page_idx, []).append("\n".join(f"- {li}" for li in list_items))
        elif item_type in ("image", "chart"):
            caption_parts = item.get("image_caption") or item.get("chart_caption") or []
            caption = " ".join(caption_parts) if isinstance(caption_parts, list) else str(caption_parts or "")
            if caption:
                pages.setdefault(page_idx, []).append(caption)

    if not pages:
        return ""
    page_blocks: List[str] = []
    for page_idx in sorted(pages.keys()):
        page_text = "\n\n".join(pages[page_idx])
        page_blocks.append(page_text)
    return "\n\n".join(page_blocks)


def convert_mineru_to_knowmat(
    content_list: List[Dict[str, Any]],
    full_md: str,
    pdf_path: str,
    extracted_dir: Path,
    page_indices: Optional[List[int]] = None,
    figures_dest: Optional[Path] = None,
) -> Tuple[str, Dict[str, Any], List[Dict[str, Any]]]:
    """Convert MinerU output to KnowMat's (extracted_text, metadata, ocr_items) triple.

    Parameters
    ----------
    content_list : list
        Parsed content_list.json from MinerU ZIP output.
    full_md : str
        Content of full.md from MinerU output.
    pdf_path : str
        Path to the source PDF.
    extracted_dir : Path
        Directory where MinerU ZIP was extracted (for resolving image paths).
    page_indices : list, optional
        1-based page indices to filter results by.
    figures_dest : Path, optional
        Directory to copy figure images to.

    Returns
    -------
    tuple of (extracted_text, metadata, ocr_items)
    """
    if page_indices:
        page_idx_set = {p - 1 for p in page_indices}
        content_list = [item for item in content_list if item.get("page_idx", 0) in page_idx_set]

    ocr_items: List[Dict[str, Any]] = []
    for item in content_list:
        converted = _convert_item(item, extracted_dir, figures_dest)
        if converted:
            ocr_items.append(converted)

    extracted_text = full_md.strip() if full_md else _build_extracted_text_from_content_list(content_list)

    total_pages = 0
    if content_list:
        total_pages = max(item.get("page_idx", 0) for item in content_list) + 1

    table_count = sum(1 for it in ocr_items if it.get("typer") == "table")
    formula_count = sum(1 for it in ocr_items if it.get("typer") == "formula")
    image_count = sum(1 for it in ocr_items if it.get("typer") == "image")

    metadata: Dict[str, Any] = {
        "backend": "mineru_api",
        "pages": total_pages,
        "ocr_items": len(ocr_items),
        "ocr_quality": {
            "ocr_avg_confidence": None,
            "ocr_low_confidence_pages": [],
            "table_count": table_count,
            "formula_count": formula_count,
            "image_count": image_count,
            "ppstructure_status": "not_applicable",
            "ppstructure_detail": "parsed by MinerU API",
            "ppstructure_replacements": 0,
        },
    }
    return extracted_text, metadata, ocr_items


def convert_lightweight_md_to_knowmat(
    markdown_text: str,
    pdf_path: str,
) -> Tuple[str, Dict[str, Any], List[Dict[str, Any]]]:
    """Convert Lightweight API markdown-only output to KnowMat format.

    Since Lightweight API only returns markdown (no content_list.json),
    we split into paragraph items.
    """
    from .blocks import text_to_paragraph_items

    ocr_items = text_to_paragraph_items(markdown_text)
    metadata: Dict[str, Any] = {
        "backend": "mineru_lightweight",
        "pages": 0,
        "ocr_items": len(ocr_items),
        "ocr_quality": {
            "ocr_avg_confidence": None,
            "ocr_low_confidence_pages": [],
            "table_count": 0,
            "formula_count": 0,
            "ppstructure_status": "not_applicable",
            "ppstructure_detail": "parsed by MinerU Lightweight API (markdown only)",
            "ppstructure_replacements": 0,
        },
    }
    return markdown_text, metadata, ocr_items
