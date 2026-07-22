"""PP-StructureV3 / TSR layout routing and accurate table/formula replacement.

This module sits between the page-level PaddleOCR-VL pass and the final
``ocr_items`` list.

Accurate strategy:
1. Run :class:`paddleocr.PPStructureV3` on a page image to obtain
   ``parsing_res_list`` (region label + bbox + block content).
2. Directly replace the corresponding ``ocr_items`` entries for
   ``table``/``chart`` and ``display_formula``/``formula``/``inline_formula``.
3. Detect plain text tables based on alignment patterns.

The module degrades gracefully: if PP-StructureV3 is unavailable, it returns
the original items unchanged.
"""

from __future__ import annotations

import gc
import logging
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ppmatAgent.knowmat.pdf.ocr_engine import ensure_paddle_device_from_env

logger = logging.getLogger(__name__)

# Labels that PP-Structure / PaddleOCR-VL may return for layout regions.
_TABLE_LABELS = frozenset({"table", "chart"})
_FORMULA_LABELS = frozenset({"display_formula", "formula", "inline_formula"})
_FIGURE_LABELS = frozenset({"image", "figure", "figure_title", "seal"})

# A small cache to avoid re-loading PP-StructureV3 models per page.
_PPSTRUCTUREV3_PIPELINE: Optional[Any] = None
_PIPELINE_LOCK = threading.Lock()

# StarRiver-like markdown ignore labels (best-effort).
_PPV3_MARKDOWN_IGNORE_LABELS = [
    "number",
    "footnote",
    "header",
    "header_image",
    "footer",
    "footer_image",
    "aside_text",
]

# Minimum area (pixels²) for a region to be worth re-OCR-ing.
_MIN_REGION_AREA = 4000

# DPI used for the high-resolution crop re-render.
_CROP_DPI = 400


def release_ppstructurev3_pipeline() -> None:
    """Drop the cached PP-StructureV3 instance so GPU weights can be freed between PDFs.
    
    This function is thread-safe and will:
    1. Delete the cached pipeline instance
    2. Trigger garbage collection
    3. Release Paddle GPU memory
    """
    global _PPSTRUCTUREV3_PIPELINE
    with _PIPELINE_LOCK:
        if _PPSTRUCTUREV3_PIPELINE is not None:
            # Delete the pipeline instance
            del _PPSTRUCTUREV3_PIPELINE
            _PPSTRUCTUREV3_PIPELINE = None
            # Force garbage collection
            gc.collect()
            # Release Paddle GPU memory
            try:
                import paddle  # type: ignore
                is_cuda = False
                if hasattr(paddle.device, "is_compiled_with_cuda"):
                    is_cuda = bool(paddle.device.is_compiled_with_cuda())
                elif hasattr(paddle, "is_compiled_with_cuda"):
                    is_cuda = bool(paddle.is_compiled_with_cuda())
                if is_cuda:
                    try:
                        paddle.device.cuda.empty_cache()
                        logger.debug("Released GPU memory after PP-StructureV3 pipeline cleanup")
                    except Exception:
                        pass
                ensure_paddle_device_from_env()
            except Exception:
                pass


def _bbox_area(bbox: List[float]) -> float:
    """Return area of an [x0, y0, x1, y1] bounding box."""
    if len(bbox) < 4:
        return 0.0
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _crop_page_image(
    pdf_path: str,
    page_idx: int,
    bbox: List[float],
    dpi: int,
    out_path: Path,
) -> Optional[Path]:
    """Render a sub-region of a PDF page at *dpi* and save to *out_path*.

    ``bbox`` is in the coordinate space of a 72-DPI render (PDF points).
    Returns *out_path* on success, None on failure.
    """
    try:
        import fitz  # type: ignore
    except ImportError:
        logger.debug("PyMuPDF not available; skipping crop re-render.")
        return None

    try:
        doc = fitz.open(pdf_path)
        page = doc[page_idx - 1]
        scale = dpi / 72.0
        clip = fitz.Rect(bbox[0], bbox[1], bbox[2], bbox[3])
        mat = fitz.Matrix(scale, scale)
        pix = page.get_pixmap(matrix=mat, clip=clip, alpha=False)
        pix.save(str(out_path))
        doc.close()
        return out_path
    except Exception as exc:
        logger.warning("Crop render failed (page=%d bbox=%s): %s", page_idx, bbox, exc)
        return None


def _run_ocr_on_image(engine: Any, image_path: Path) -> Optional[str]:
    """Run OCR engine on a single image and return extracted text.
    
    Parameters
    ----------
    engine : Any
        The OCR engine instance (PaddleOCR or PaddleOCRVL).
    image_path : Path
        Path to the image file to process.
        
    Returns
    -------
    Optional[str]
        Extracted text, or None if no text could be extracted.
        
    Note
    ----
    This function logs warnings for OCR failures but returns None instead of
    raising exceptions, to allow batch processing to continue with other images.
    Check logs for details when debugging OCR issues.
    """
    img = str(image_path)
    raw = None
    last_error: Optional[Exception] = None
    
    try:
        if hasattr(engine, "predict"):
            try:
                raw = engine.predict(img)
            except TypeError:
                raw = engine.predict(input=img)
        elif hasattr(engine, "ocr"):
            try:
                raw = engine.ocr(img, cls=True)
            except TypeError:
                raw = engine.ocr(img)
        elif callable(engine):
            raw = engine(img)
    except Exception as exc:
        last_error = exc
        # Log detailed error for debugging - this is important for troubleshooting
        logger.warning(
            "Re-OCR failed for %s: %s (type: %s)", 
            image_path, exc, type(exc).__name__
        )
        return None

    if raw is None:
        if last_error is not None:
            logger.debug("OCR returned None for %s (previous error: %s)", image_path, last_error)
        else:
            logger.debug("OCR returned None for %s (no text detected)", image_path)
        return None

    # Extract text from raw result (handles both PaddleOCR and PaddleOCRVL).
    texts: List[str] = []
    _collect_text(raw, texts)
    
    result = "\n".join(texts) if texts else None
    if result:
        logger.debug("Re-OCR extracted %d characters from %s", len(result), image_path)
    return result


def _collect_text(obj: Any, out: List[str]) -> None:
    if obj is None:
        return
    if isinstance(obj, str):
        t = obj.strip()
        if t:
            out.append(t)
        return
    if isinstance(obj, dict):
        for key in ("text", "rec_text", "ocr_text", "content", "transcription"):
            val = obj.get(key)
            if isinstance(val, str) and val.strip():
                out.append(val.strip())
        for val in obj.values():
            _collect_text(val, out)
        return
    if isinstance(obj, (list, tuple)):
        for item in obj:
            _collect_text(item, out)


def _get_ppstructurev3_pipeline_with_status() -> Tuple[Optional[Any], str]:
    """Get or create a cached PPStructureV3 pipeline; return (pipeline, error_message).

    Thread-safe: fast path without lock when the pipeline is already constructed;
    double-checked locking around initialization only.
    """
    global _PPSTRUCTUREV3_PIPELINE
    if _PPSTRUCTUREV3_PIPELINE is not None:
        return _PPSTRUCTUREV3_PIPELINE, ""

    with _PIPELINE_LOCK:
        if _PPSTRUCTUREV3_PIPELINE is not None:
            return _PPSTRUCTUREV3_PIPELINE, ""

        try:
            from paddleocr import PPStructureV3  # type: ignore
        except ImportError as exc:
            msg = "paddleocr.PPStructureV3 is not installed."
            logger.warning("%s (%s)", msg, exc)
            return None, msg

        try:
            _PPSTRUCTUREV3_PIPELINE = PPStructureV3(
                use_region_detection=True,
                use_table_recognition=True,
                use_formula_recognition=True,
                use_chart_recognition=False,
                use_seal_recognition=True,
                format_block_content=True,
                markdown_ignore_labels=_PPV3_MARKDOWN_IGNORE_LABELS,
            )
        except TypeError:
            try:
                _PPSTRUCTUREV3_PIPELINE = PPStructureV3()
            except Exception as exc:
                detail = str(exc)
                logger.warning("PP-StructureV3 initialization failed (fallback ctor): %s", detail)
                return None, detail
        except Exception as exc:
            detail = str(exc)
            logger.warning("PP-StructureV3 initialization failed: %s", detail)
            return None, detail

        return _PPSTRUCTUREV3_PIPELINE, ""


def _run_ppstructurev3_page_regions(
    pipeline: Any,
    image_path: Path,
) -> List[Dict[str, Any]]:
    """Run PP-StructureV3 on one page image and return region list (empty on error)."""
    try:
        output = pipeline.predict(input=str(image_path))
    except Exception as exc:
        logger.warning("PP-StructureV3 predict failed for %s: %s", image_path, exc)
        return []

    res0 = None
    if isinstance(output, list) and output:
        res0 = output[0]
    else:
        res0 = output

    parsing_res_list = getattr(res0, "parsing_res_list", None)
    if parsing_res_list is None and isinstance(res0, dict):
        parsing_res_list = res0.get("parsing_res_list")
    if not parsing_res_list:
        return []

    regions: List[Dict[str, Any]] = []
    for blk in parsing_res_list:
        if not isinstance(blk, dict):
            continue
        blabel = blk.get("block_label") or blk.get("label") or ""
        bbox = blk.get("block_bbox") or blk.get("bbox") or []
        if not blabel or not bbox or len(bbox) < 4:
            continue
        regions.append({"label": blabel, "bbox": bbox, "block": blk})
    return regions


def seed_legacy_complex_items_from_ppstructurev3(
    page_indices: Sequence[int],
    image_paths: Sequence[Path],
) -> List[Dict[str, Any]]:
    """Build table/formula ``ocr_items`` from PP-StructureV3 for legacy PaddleOCR runs.

    Classic line OCR does not emit layout blocks; without these seeds,
    :func:`route_and_reocr` would immediately return ``no_complex_blocks``.
    """
    from .blocks import block_to_item

    pipeline, err = _get_ppstructurev3_pipeline_with_status()
    if pipeline is None:
        if err:
            logger.warning("Legacy PaddleOCR path: PP-StructureV3 seed skipped (%s).", err)
        return []

    items: List[Dict[str, Any]] = []
    for pdf_page, img_path in zip(page_indices, image_paths):
        p = Path(img_path)
        if not p.is_file():
            continue
        for reg in _run_ppstructurev3_page_regions(pipeline, p):
            blk = reg.get("block")
            if not isinstance(blk, dict):
                continue
            item = block_to_item(blk)
            if not item or item.get("typer") not in ("table", "formula"):
                continue
            item["page"] = int(pdf_page)
            bbox = reg.get("bbox")
            if bbox is not None:
                item["bbox"] = bbox
            lbl = reg.get("label")
            if lbl is not None:
                item["block_label"] = lbl
            items.append(item)

    if items:
        logger.info(
            "Legacy PaddleOCR path: seeded %d table/formula item(s) from PP-StructureV3 layout.",
            len(items),
        )
    return items


def route_and_reocr(
    ocr_items: List[Dict[str, Any]],
    _pdf_path: str,
    page_images: Dict[int, Path],
    _engine: Any,
    _work_dir: Path,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Run PP-StructureV3 and directly replace table/formula blocks.

    Returns
    -------
    items, status
        *status* includes ``ppstructure_status``, ``ppstructure_detail``,
        and ``ppstructure_replacements`` for downstream metadata.
    """
    status: Dict[str, Any] = {
        "ppstructure_status": "skipped_empty",
        "ppstructure_detail": "",
        "ppstructure_replacements": 0,
    }
    if not ocr_items:
        return ocr_items, status

    pipeline, err = _get_ppstructurev3_pipeline_with_status()
    if pipeline is None:
        status["ppstructure_status"] = "unavailable"
        status["ppstructure_detail"] = err
        return ocr_items, status

    pages_with_complex: Dict[int, List[int]] = {}
    for i, item in enumerate(ocr_items):
        typer = item.get("typer", "")
        if typer in ("table", "formula"):
            page = item.get("page", 0)
            pages_with_complex.setdefault(page, []).append(i)

    if not pages_with_complex:
        status["ppstructure_status"] = "no_complex_blocks"
        return ocr_items, status

    from .blocks import block_to_item

    replacements = 0
    for page_idx, item_indices in pages_with_complex.items():
        page_img = page_images.get(page_idx)
        if page_img is None or not page_img.exists():
            continue

        regions = _run_ppstructurev3_page_regions(pipeline, page_img)
        if not regions:
            continue

        for ii in item_indices:
            item = ocr_items[ii]
            item_bbox = item.get("bbox")
            item_typer = item.get("typer", "")

            if item_bbox is None or _bbox_area(item_bbox) < _MIN_REGION_AREA:
                continue

            best_region = _find_best_matching_region(item_bbox, regions, item_typer)
            if not best_region:
                continue

            pp_block = best_region.get("block")
            if not isinstance(pp_block, dict):
                continue

            new_item = block_to_item(pp_block)
            if not new_item:
                continue

            new_item["page"] = page_idx
            new_item["bbox"] = best_region.get("bbox")
            new_item["block_label"] = best_region.get("label")

            if "confidence" not in new_item and item.get("confidence") is not None:
                new_item["confidence"] = item["confidence"]

            new_item["reocr_source"] = "ppstructurev3_replace"
            ocr_items[ii] = new_item
            replacements += 1

    status["ppstructure_replacements"] = replacements
    status["ppstructure_status"] = "applied" if replacements > 0 else "no_replacements"
    return ocr_items, status


def reocr_chem_formula_blocks(
    ocr_items: List[Dict[str, Any]],
    pdf_path: str,
    page_images: Dict[int, Path],
    engine: Any,
    work_dir: Path,
) -> List[Dict[str, Any]]:
    """Re-OCR paragraph blocks that appear to contain chemical formulas.

    Detects paragraph items whose text matches an element-number pattern
    (e.g. ``Ti42Hf21``) and re-runs OCR on the corresponding bbox crop at
    higher DPI to improve subscript / decimal accuracy.

    Parameters are the same as :func:`route_and_reocr`.
    """
    import re

    # Lightweight element pattern for detection (using full element list from section_normalizer).
    from .section_normalizer import _ELEMENT_PAT, is_noise_line
    _ELEM_PAT = re.compile(r"\b(?:" + _ELEMENT_PAT + r")\d+(?:[.,]\d+)?")

    crops_dir = work_dir / "chem_crops"
    modified = 0

    for i, item in enumerate(ocr_items):
        if item.get("typer") != "paragraph":
            continue
        text = item.get("text", "")
        if not _ELEM_PAT.search(text):
            continue
        bbox = item.get("bbox")
        if not bbox or _bbox_area(bbox) < _MIN_REGION_AREA:
            continue

        page = item.get("page", 0)
        page_img = page_images.get(page)
        if page_img is None or not page_img.exists():
            continue

        crops_dir.mkdir(parents=True, exist_ok=True)
        crop_path = crops_dir / f"chem-page{page:04d}-item{i:04d}.png"
        rendered = _crop_page_image(pdf_path, page, bbox, _CROP_DPI, crop_path)
        if rendered is None:
            continue

        reocr_text = _run_ocr_on_image(engine, rendered)
        if reocr_text and reocr_text.strip():
            item["text"] = reocr_text.strip()
            item["reocr_source"] = "chem_crop_reocr"
            modified += 1

    if modified:
        logger.info("Re-OCR'd %d chemical formula paragraph block(s).", modified)
    return ocr_items


def _find_best_matching_region(
    item_bbox: List[float],
    regions: List[Dict[str, Any]],
    item_typer: str,
) -> Optional[Dict[str, Any]]:
    """Find the PP-Structure region that best overlaps with *item_bbox*."""
    target_labels = _TABLE_LABELS if item_typer == "table" else _FORMULA_LABELS
    best: Optional[Dict[str, Any]] = None
    best_iou = 0.0

    for region in regions:
        if region.get("label", "").lower() not in target_labels:
            continue
        rbbox = region.get("bbox", [])
        if len(rbbox) < 4:
            continue
        iou = _bbox_iou(item_bbox, rbbox)
        if iou > best_iou:
            best_iou = iou
            best = region

    return best if best_iou > 0.1 else None


def _bbox_iou(a: List[float], b: List[float]) -> float:
    """Compute Intersection-over-Union for two [x0,y0,x1,y1] boxes."""
    ix0 = max(a[0], b[0])
    iy0 = max(a[1], b[1])
    ix1 = min(a[2], b[2])
    iy1 = min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    if inter <= 0.0:
        return 0.0
    area_a = _bbox_area(a)
    area_b = _bbox_area(b)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Plain text table detection utilities
# ---------------------------------------------------------------------------

# Minimum number of rows to consider a block as a table
_MIN_TABLE_ROWS = 2

# Minimum number of columns to consider a block as a table
_MIN_TABLE_COLS = 2

# Characters that indicate column separators
_COLUMN_SEPARATORS = re.compile(r"[|\t]")

# Pattern for detecting column alignment based on whitespace
_WHITESPACE_SEP_RE = re.compile(r"\s{2,}")


def detect_text_table(lines: List[str]) -> Optional[Dict[str, Any]]:
    """Detect if a list of lines forms a plain text table.
    
    This function analyzes text lines to determine if they represent
    a table structure based on:
    1. Consistent column separators (|, tab, or multi-space alignment)
    2. Regular column counts across rows
    3. Presence of header/separator lines
    
    Parameters
    ----------
    lines : List[str]
        Lines of text to analyze.
        
    Returns
    -------
    Optional[Dict[str, Any]]
        Dictionary with keys:
        - columns: List of column names (from first row if header detected)
        - rows: List of row data lists
        - separator: The detected separator type
        Returns None if no table structure is detected.
    """
    if len(lines) < _MIN_TABLE_ROWS:
        return None
    
    # Check for pipe-separated tables
    pipe_result = _detect_pipe_table(lines)
    if pipe_result:
        return pipe_result
    
    # Check for tab-separated tables
    tab_result = _detect_tab_table(lines)
    if tab_result:
        return tab_result
    
    # Check for whitespace-aligned tables
    space_result = _detect_whitespace_table(lines)
    if space_result:
        return space_result
    
    return None


def _detect_pipe_table(lines: List[str]) -> Optional[Dict[str, Any]]:
    """Detect pipe-separated tables (| col1 | col2 |)."""
    pipe_lines = [l for l in lines if "|" in l]
    if len(pipe_lines) < _MIN_TABLE_ROWS:
        return None
    
    # Parse each line
    rows: List[List[str]] = []
    for line in pipe_lines:
        # Skip separator lines (e.g., |---|---|)
        if re.match(r"^\s*\|[-:\s|]+\|\s*$", line):
            continue
        # Split by | and strip whitespace
        cells = [c.strip() for c in line.split("|")]
        # Remove empty cells at start/end (from leading/trailing |)
        if cells and not cells[0]:
            cells = cells[1:]
        if cells and not cells[-1]:
            cells = cells[:-1]
        if cells:
            rows.append(cells)
    
    if len(rows) < _MIN_TABLE_ROWS:
        return None
    
    # Check column consistency
    col_counts = [len(r) for r in rows]
    if min(col_counts) < _MIN_TABLE_COLS:
        return None
    
    # Find the most common column count
    from collections import Counter
    count_freq = Counter(col_counts)
    most_common_count = count_freq.most_common(1)[0][0]
    
    # Filter rows to consistent column count
    consistent_rows = [r for r in rows if len(r) == most_common_count]
    
    if len(consistent_rows) < _MIN_TABLE_ROWS:
        return None
    
    return {
        "columns": consistent_rows[0] if consistent_rows else [],
        "rows": consistent_rows[1:] if len(consistent_rows) > 1 else [],
        "separator": "pipe",
    }


def _detect_tab_table(lines: List[str]) -> Optional[Dict[str, Any]]:
    """Detect tab-separated tables."""
    tab_lines = [l for l in lines if "\t" in l]
    if len(tab_lines) < _MIN_TABLE_ROWS:
        return None
    
    rows: List[List[str]] = []
    for line in tab_lines:
        cells = [c.strip() for c in line.split("\t")]
        if len(cells) >= _MIN_TABLE_COLS:
            rows.append(cells)
    
    if len(rows) < _MIN_TABLE_ROWS:
        return None
    
    # Check column consistency
    col_counts = [len(r) for r in rows]
    from collections import Counter
    count_freq = Counter(col_counts)
    most_common_count = count_freq.most_common(1)[0][0]
    
    consistent_rows = [r for r in rows if len(r) == most_common_count]
    
    if len(consistent_rows) < _MIN_TABLE_ROWS:
        return None
    
    return {
        "columns": consistent_rows[0] if consistent_rows else [],
        "rows": consistent_rows[1:] if len(consistent_rows) > 1 else [],
        "separator": "tab",
    }


def _detect_whitespace_table(lines: List[str]) -> Optional[Dict[str, Any]]:
    """Detect tables based on whitespace column alignment.
    
    This is the most sophisticated detection method, analyzing
    the alignment of text across multiple lines.
    """
    if len(lines) < _MIN_TABLE_ROWS:
        return None
    
    # Find lines with multiple whitespace separators
    candidate_lines: List[Tuple[str, List[int]]] = []
    for line in lines:
        # Find positions of multi-space gaps
        gaps = [m.start() for m in _WHITESPACE_SEP_RE.finditer(line)]
        if len(gaps) >= _MIN_TABLE_COLS - 1:
            candidate_lines.append((line, gaps))
    
    if len(candidate_lines) < _MIN_TABLE_ROWS:
        return None
    
    # Analyze gap positions across lines for alignment
    all_gaps = [pos for _, gaps in candidate_lines for pos in gaps]
    if not all_gaps:
        return None
    
    # Cluster gap positions (within 3 characters tolerance)
    tolerance = 3
    gap_clusters: List[List[int]] = []
    for gap in sorted(all_gaps):
        placed = False
        for cluster in gap_clusters:
            if abs(cluster[0] - gap) <= tolerance:
                cluster.append(gap)
                placed = True
                break
        if not placed:
            gap_clusters.append([gap])
    
    # Need at least MIN_TABLE_COLS-1 consistent column boundaries
    if len(gap_clusters) < _MIN_TABLE_COLS - 1:
        return None
    
    # Use cluster centers as column boundaries
    col_positions = [int(sum(c) / len(c)) for c in gap_clusters]
    col_positions = sorted(col_positions)
    
    # Split lines by column positions
    rows: List[List[str]] = []
    for line, _ in candidate_lines:
        cells: List[str] = []
        prev_pos = 0
        for pos in col_positions:
            cell = line[prev_pos:pos].strip()
            cells.append(cell)
            prev_pos = pos
        # Add the last cell
        cells.append(line[prev_pos:].strip())
        if len(cells) >= _MIN_TABLE_COLS:
            rows.append(cells)
    
    if len(rows) < _MIN_TABLE_ROWS:
        return None
    
    # Normalize column counts
    max_cols = max(len(r) for r in rows)
    normalized_rows = []
    for r in rows:
        if len(r) < max_cols:
            r = r + [""] * (max_cols - len(r))
        normalized_rows.append(r)
    
    return {
        "columns": normalized_rows[0] if normalized_rows else [],
        "rows": normalized_rows[1:] if len(normalized_rows) > 1 else [],
        "separator": "whitespace",
    }


def text_table_to_markdown(table_data: Dict[str, Any]) -> str:
    """Convert detected table data to Markdown format.
    
    Parameters
    ----------
    table_data : Dict[str, Any]
        Table data from detect_text_table().
        
    Returns
    -------
    str
        Markdown-formatted table string.
    """
    columns = table_data.get("columns", [])
    rows = table_data.get("rows", [])
    
    if not columns and not rows:
        return ""
    
    # If no columns detected, create generic column names
    if not columns and rows:
        num_cols = len(rows[0]) if rows else 0
        columns = [f"Col{i+1}" for i in range(num_cols)]
    
    # Build header
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    
    # Build rows
    md_rows = []
    for row in rows:
        # Pad row if necessary
        if len(row) < len(columns):
            row = list(row) + [""] * (len(columns) - len(row))
        md_rows.append("| " + " | ".join(str(c) for c in row[:len(columns)]) + " |")
    
    return "\n".join([header, separator] + md_rows)


def detect_and_convert_text_tables(text: str) -> str:
    """Process text and convert detected plain text tables to Markdown.
    
    This is a convenience function that processes an entire document,
    detecting and converting any plain text tables found.
    
    Parameters
    ----------
    text : str
        The input document text.
        
    Returns
    -------
    str
        Text with detected tables converted to Markdown format.
    """
    lines = text.splitlines()
    result_lines: List[str] = []
    buffer: List[str] = []
    
    for line in lines:
        # Accumulate lines in buffer
        buffer.append(line)
        
        # Try to detect a table in the buffer
        table = detect_text_table(buffer)
        
        if table:
            # Convert table to Markdown
            md_table = text_table_to_markdown(table)
            if md_table:
                result_lines.append(md_table)
                buffer = []
        else:
            # If buffer is large enough and no table detected, flush
            if len(buffer) >= 10:  # Max expected table size
                result_lines.extend(buffer)
                buffer = []
    
    # Flush remaining buffer
    result_lines.extend(buffer)
    
    return "\n".join(result_lines)
