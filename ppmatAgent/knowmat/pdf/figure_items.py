"""Helpers for normalizing OCR figure items.

PaddleOCR-VL often emits one ``image`` item carrying only an ``image_path``
and a separate nearby ``image`` item carrying the figure caption.  Downstream
figure description needs these fields on the same item, so this module links
them by page-local ordering and normalizes caption metadata.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .blocks import extract_figure_caption


def _image_data(item: Dict[str, Any]) -> Dict[str, Any]:
    data = item.get("data")
    return data if isinstance(data, dict) else {}


def _normalize_caption_fields(data: Dict[str, Any]) -> None:
    caption = str(data.get("caption") or "").strip()
    if data.get("figure_num") or not caption:
        return
    parsed = extract_figure_caption(caption)
    if not parsed:
        return
    data.setdefault("figure_num", parsed.get("figure_num"))
    data.setdefault("caption_type", parsed.get("figure_type"))
    data["caption"] = parsed.get("caption") or caption


def is_figure_caption_item(item: Dict[str, Any]) -> bool:
    """Return True when *item* looks like a real figure caption block."""
    if not isinstance(item, dict) or item.get("typer") != "image":
        return False
    data = _image_data(item)
    _normalize_caption_fields(data)
    if data.get("figure_num"):
        return True
    caption_type = str(data.get("caption_type") or "").strip().lower()
    if caption_type in {"fig", "fig.", "figure"}:
        return True
    caption = str(data.get("caption") or "").strip()
    return extract_figure_caption(caption) is not None


def _figure_identity(item: Dict[str, Any]) -> Optional[Tuple[int, str]]:
    if not is_figure_caption_item(item):
        return None
    data = _image_data(item)
    figure_num = str(data.get("figure_num") or "").strip()
    if not figure_num:
        return None
    return (int(item.get("page") or 0), figure_num)


def _figure_item_score(item: Dict[str, Any]) -> Tuple[int, int]:
    data = _image_data(item)
    caption = str(data.get("caption") or "").strip()
    image_path = str(data.get("image_path") or "").strip()
    score = 0
    if image_path:
        score += 4
        if Path(image_path).is_file():
            score += 6
    if isinstance(item.get("bbox"), list) and len(item.get("bbox")) == 4:
        score += 4
    if caption:
        score += 2
    if data.get("figure_num"):
        score += 1
    return (score, len(caption))


def _should_replace_image_path(current: Any, preferred: str) -> bool:
    preferred_path = str(preferred or "").strip()
    if not preferred_path:
        return False
    current_path = str(current or "").strip()
    if not current_path:
        return True
    preferred_exists = Path(preferred_path).is_file()
    current_exists = Path(current_path).is_file()
    if preferred_exists and not current_exists:
        return True
    return False


def _merge_duplicate_figure_items(ocr_items: List[Dict[str, Any]]) -> None:
    groups: Dict[Tuple[int, str], List[Dict[str, Any]]] = {}
    for item in ocr_items:
        ident = _figure_identity(item)
        if ident is not None:
            groups.setdefault(ident, []).append(item)

    for group in groups.values():
        ranked = sorted(group, key=_figure_item_score, reverse=True)
        bbox = next(
            (
                [float(v) for v in candidate["bbox"]]
                for candidate in ranked
                if isinstance(candidate.get("bbox"), list) and len(candidate.get("bbox")) == 4
            ),
            None,
        )
        image_path = next(
            (
                str((_image_data(candidate).get("image_path") or "")).strip()
                for candidate in ranked
                if str((_image_data(candidate).get("image_path") or "")).strip()
            ),
            "",
        )
        figure_num = next(
            (str((_image_data(candidate).get("figure_num") or "")).strip() for candidate in ranked),
            "",
        )
        caption_type = next(
            (str((_image_data(candidate).get("caption_type") or "")).strip() for candidate in ranked),
            "",
        )
        caption = next(
            (
                str((_image_data(candidate).get("caption") or "")).strip()
                for candidate in sorted(
                    group,
                    key=lambda candidate: len(str((_image_data(candidate).get("caption") or "")).strip()),
                    reverse=True,
                )
                if str((_image_data(candidate).get("caption") or "")).strip()
            ),
            "",
        )

        for item in group:
            data = _image_data(item)
            if figure_num and not data.get("figure_num"):
                data["figure_num"] = figure_num
            if caption_type and not data.get("caption_type"):
                data["caption_type"] = caption_type
            if caption and not data.get("caption"):
                data["caption"] = caption
            if _should_replace_image_path(data.get("image_path"), image_path):
                data["image_path"] = image_path
            if bbox is not None and "bbox" not in item:
                item["bbox"] = list(bbox)


def iter_figure_caption_items(ocr_items: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return one canonical OCR item per resolved figure caption identity."""
    ordered: List[Tuple[Tuple[int, str], Dict[str, Any], Tuple[int, int], int]] = []
    seen: Dict[Tuple[int, str], int] = {}
    for order, item in enumerate(ocr_items):
        ident = _figure_identity(item)
        if ident is None:
            continue
        score = _figure_item_score(item)
        idx = seen.get(ident)
        if idx is None:
            seen[ident] = len(ordered)
            ordered.append((ident, item, score, order))
        else:
            _, current_item, current_score, current_order = ordered[idx]
            if score > current_score or (score == current_score and order >= current_order):
                ordered[idx] = (ident, item, score, order)
            else:
                ordered[idx] = (ident, current_item, current_score, current_order)
    return [item for _, item, _, _ in ordered]


def iter_resolved_figure_items(ocr_items: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return figure items that now have both image path and figure identity."""
    resolved: List[Dict[str, Any]] = []
    for item in iter_figure_caption_items(ocr_items):
        data = _image_data(item)
        if data.get("image_path"):
            resolved.append(item)
    return resolved


def normalize_figure_ocr_items(
    ocr_items: List[Dict[str, Any]],
    *,
    search_window: int = 4,
) -> List[Dict[str, Any]]:
    """Merge split OCR figure path/caption items in place and return the list."""
    if not ocr_items:
        return ocr_items

    page_positions: Dict[int, List[int]] = {}
    for idx, item in enumerate(ocr_items):
        if not isinstance(item, dict) or item.get("typer") != "image":
            continue
        data = _image_data(item)
        _normalize_caption_fields(data)
        page = int(item.get("page") or 0)
        if page > 0:
            page_positions.setdefault(page, []).append(idx)

    for page_indices in page_positions.values():
        used_caption_indices: set[int] = set()
        for pos, item_idx in enumerate(page_indices):
            item = ocr_items[item_idx]
            data = _image_data(item)
            if not data.get("image_path") or is_figure_caption_item(item):
                continue

            match_idx = None
            for distance in range(1, search_window + 1):
                for neighbor_pos in (pos + distance, pos - distance):
                    if neighbor_pos < 0 or neighbor_pos >= len(page_indices):
                        continue
                    candidate_idx = page_indices[neighbor_pos]
                    if candidate_idx in used_caption_indices:
                        continue
                    candidate = ocr_items[candidate_idx]
                    candidate_data = _image_data(candidate)
                    if candidate_data.get("image_path"):
                        continue
                    if not is_figure_caption_item(candidate):
                        continue
                    match_idx = candidate_idx
                    break
                if match_idx is not None:
                    break

            if match_idx is None:
                continue

            used_caption_indices.add(match_idx)
            match_item = ocr_items[match_idx]
            match_data = _image_data(ocr_items[match_idx])
            for key in ("figure_num", "caption", "caption_type"):
                value = match_data.get(key)
                if value and not data.get(key):
                    data[key] = value
            image_path = data.get("image_path")
            if image_path and not match_data.get("image_path"):
                match_data["image_path"] = image_path
            if image_path and "bbox" in item and "bbox" not in match_item:
                match_item["bbox"] = item["bbox"]

    _merge_duplicate_figure_items(ocr_items)
    return ocr_items


def promote_caption_paragraphs(ocr_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert paragraph items that contain figure captions to image items.

    API-mode OCR (PaddleOCR/MinerU) stores figure captions as separate paragraph
    items.  This function promotes them to ``typer: "image"`` items so that
    ``normalize_figure_ocr_items`` can merge them with path-bearing image items.
    """
    import re
    _div_tag_re = re.compile(r"</?div[^>]*>", re.IGNORECASE)

    for item in ocr_items:
        if not isinstance(item, dict) or item.get("typer") != "paragraph":
            continue
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        # Strip <div> wrappers that PaddleOCR API adds around captions
        clean_text = _div_tag_re.sub("", text).strip()
        parsed = extract_figure_caption(clean_text)
        if parsed and parsed.get("figure_num"):
            item["typer"] = "image"
            item["data"] = {
                "image_path": "",
                "caption": parsed.get("caption") or clean_text,
                "figure_num": parsed["figure_num"],
                "caption_type": parsed.get("figure_type", ""),
            }
            item["block_label"] = "figure"
    return ocr_items
