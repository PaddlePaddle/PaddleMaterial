"""Figure-internal fragment filter for OCR block lists.

Axis labels, tick marks, and other text fragments that lie *inside* a figure
or image bounding box are noise for the main text flow.  This module provides
``filter_figure_internal_fragments`` which removes such blocks from
``ocr_items`` while keeping:

- The figure/image item itself.
- Any ``figure_title`` / caption block that is *adjacent to* (not inside) the
  figure bbox.

The filter operates purely on bbox geometry and ``block_label`` metadata; it
does not call any model or LLM.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Labels that represent figure / image regions.
_FIGURE_LABELS = frozenset({"image", "figure", "seal", "chart"})

# Caption-like labels that should be kept even when near a figure.
_CAPTION_LABELS = frozenset({"figure_title", "caption"})

# How much to expand a figure bbox (in pixels) when deciding whether a
# paragraph block is "inside" it.  A small positive margin tolerates slight
# coordinate misalignments from the OCR engine.
_INSIDE_MARGIN = 4.0

# Maximum fraction of a paragraph block's area that may overlap a figure bbox
# before it is considered an internal fragment.
_OVERLAP_FRAC_THRESHOLD = 0.6


def _bbox_area(bbox: List[float]) -> float:
    if len(bbox) < 4:
        return 0.0
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _bbox_intersection_area(a: List[float], b: List[float]) -> float:
    ix0 = max(a[0], b[0])
    iy0 = max(a[1], b[1])
    ix1 = min(a[2], b[2])
    iy1 = min(a[3], b[3])
    return max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)


def _is_inside_figure(
    block_bbox: List[float],
    figure_bbox: List[float],
    margin: float = _INSIDE_MARGIN,
) -> bool:
    """Return True when *block_bbox* is fully contained within *figure_bbox*."""
    fx0, fy0, fx1, fy1 = (
        figure_bbox[0] - margin,
        figure_bbox[1] - margin,
        figure_bbox[2] + margin,
        figure_bbox[3] + margin,
    )
    return (
        block_bbox[0] >= fx0
        and block_bbox[1] >= fy0
        and block_bbox[2] <= fx1
        and block_bbox[3] <= fy1
    )


def _overlap_fraction(block_bbox: List[float], figure_bbox: List[float]) -> float:
    """Return what fraction of *block_bbox*'s area overlaps *figure_bbox*."""
    block_area = _bbox_area(block_bbox)
    if block_area <= 0:
        return 0.0
    inter = _bbox_intersection_area(block_bbox, figure_bbox)
    return inter / block_area


def filter_figure_internal_fragments(
    ocr_items: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], int]:
    """Remove paragraph/text fragments that lie inside figure bounding boxes.

    Parameters
    ----------
    ocr_items:
        The full list of OCR items (as produced by the main OCR pass).

    Returns
    -------
    filtered_items:
        The filtered list with internal figure fragments removed.
    removed_count:
        Number of items that were removed.
    """
    if not ocr_items:
        return ocr_items, 0

    # Collect figure bboxes grouped by page.
    figure_bboxes_by_page: Dict[int, List[List[float]]] = {}
    for item in ocr_items:
        label = item.get("block_label", "") or item.get("typer", "")
        if label in _FIGURE_LABELS:
            bbox = item.get("bbox")
            if bbox and len(bbox) >= 4:
                page = item.get("page", 0)
                figure_bboxes_by_page.setdefault(page, []).append(list(bbox))

    if not figure_bboxes_by_page:
        return ocr_items, 0

    filtered: List[Dict[str, Any]] = []
    removed = 0

    for item in ocr_items:
        label = item.get("block_label", "") or item.get("typer", "")
        typer = item.get("typer", "")

        # Always keep figure / image / caption items.
        if label in _FIGURE_LABELS or label in _CAPTION_LABELS:
            filtered.append(item)
            continue

        # Only consider paragraph/text items for removal.
        if typer not in ("paragraph", "text"):
            filtered.append(item)
            continue

        bbox = item.get("bbox")
        if not bbox or len(bbox) < 4:
            filtered.append(item)
            continue

        page = item.get("page", 0)
        figure_bboxes = figure_bboxes_by_page.get(page, [])

        is_fragment = False
        for fig_bbox in figure_bboxes:
            if _is_inside_figure(bbox, fig_bbox):
                is_fragment = True
                break
            if _overlap_fraction(bbox, fig_bbox) >= _OVERLAP_FRAC_THRESHOLD:
                is_fragment = True
                break

        if is_fragment:
            logger.debug(
                "Removed figure-internal fragment (page=%d bbox=%s text=%.60r)",
                page,
                bbox,
                item.get("text", ""),
            )
            removed += 1
        else:
            filtered.append(item)

    return filtered, removed
