"""PDF parsing node using PaddleOCR-first OCR backends.

The default backend is PaddleOCR: API mode when ``PADDLEOCR_API_TOKEN`` is
configured, otherwise local PaddleOCR-VL. MinerU remains available only as a
compatibility backend via ``KNOWMAT_OCR_BACKEND=mineru`` or the CLI flag.
OpenAI-compatible vision OCR is available via ``KNOWMAT_OCR_BACKEND=openai``.
Legacy PaddleOCR mode still runs PP-StructureV3 layout seeding and refinement.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import tempfile
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ppmatAgent.knowmat.pdf.blocks import block_to_item, sanitize_ocr_items_vl_artifacts, text_to_paragraph_items
from ppmatAgent.knowmat.pdf.block_filter import filter_figure_internal_fragments
from ppmatAgent.knowmat.pdf.doi_extractor import (
    extract_doi_from_pdf_metadata,
    extract_first_doi,
    extract_first_doi_from_ocr_items,
)
from ppmatAgent.knowmat.pdf.figure_items import (
    iter_figure_caption_items,
    iter_resolved_figure_items,
    normalize_figure_ocr_items,
)
from ppmatAgent.knowmat.pdf.html_cleaner import convert_html_to_markdown
from ppmatAgent.knowmat.pdf.ocr_cache import (
    cache_signature_key,
    md5_file_digest,
    ocr_cache_bucket,
    pages_key_for_cache,
    parse_pages_argument,
    save_ocr_cache,
    try_load_ocr_cache,
)
from ppmatAgent.knowmat.pdf.ocr_engine import (
    create_ocr_engine,
    default_model_dir,
    ensure_paddle_device_from_env,
    normalize_lines,
    paddleocr_raw_to_lines,
    run_ocr_batch,
    run_ocr_sequential,
    supports_batch_predict,
    try_release_paddle_gpu_memory,
)
from ppmatAgent.knowmat.pdf.table_structure import (
    reocr_chem_formula_blocks,
    release_ppstructurev3_pipeline,
    route_and_reocr,
    seed_legacy_complex_items_from_ppstructurev3,
    _crop_page_image,
)
from ppmatAgent.knowmat.pdf.section_normalizer import (
    normalize_alloy_strings,
    normalize_leading_masthead_and_title,
    normalize_plain_author_superscripts,
    repair_keywords_abstract_two_column_ocr,
    strip_references_section,
    structure_sections,
)
from ppmatAgent.knowmat.pdf.formula_formatter import collapse_orphan_display_formulas, format_formula_text
from ppmatAgent.knowmat.pdf.mineru_api_client import MineruAPIError, MineruLightweightClient, MineruPrecisionClient
from ppmatAgent.knowmat.pdf.mineru_result_converter import (
    _pages_range_from_indices,
    convert_lightweight_md_to_knowmat,
    convert_mineru_to_knowmat,
)
from ppmatAgent.knowmat.pdf.paddleocr_api_client import PaddleOCRAPIClient, PaddleOCRAPIError
from ppmatAgent.knowmat.pdf.paddleocr_api_result_converter import (
    convert_paddleocr_api_to_knowmat,
    extract_formulas_per_page,
    extract_tables_per_page,
)
from ppmatAgent.knowmat.pdf.openai_ocr_client import (
    OpenAIOCRAPIError,
    OpenAIOCRClient,
    build_openai_ocr_result,
)
from ppmatAgent.knowmat.app_config import settings
from ppmatAgent.knowmat.states import KnowMatState

logger = logging.getLogger(__name__)

# Defaults (override with OCR_MAX_RENDER_WORKERS, OCR_HEADER_LINES, OCR_FOOTER_LINES, etc.).
_DEFAULT_MAX_RENDER_WORKERS = 4
_DEFAULT_OCR_MAX_WORKERS = 1
_DEFAULT_OCR_HEADER_LINES = 5
_DEFAULT_OCR_FOOTER_LINES = 3
_DEFAULT_OCR_LOW_CONFIDENCE = 0.5

_LAYOUT_NOISE_LABELS = frozenset(
    {
        "header",
        "footer",
        "header_image",
        "footer_image",
        "aside_text",
        "number",
        "footnote",
    }
)

_SIDEBAR_X_FRAC = 0.15
_HEADER_Y_FRAC = 0.08
_FOOTER_Y_FRAC = 0.92


def _env_truthy(name: str) -> bool:
    v = os.getenv(name, "").strip().lower()
    return v in ("1", "true", "yes", "on")


def _append_missing_ocr_paragraphs_enabled() -> bool:
    """Default on: append OCR paragraph lines missing from merged body (set to 0/false to disable)."""
    v = os.getenv("KNOWMAT_APPEND_MISSING_OCR_PARAGRAPHS", "1").strip().lower()
    return v not in ("0", "false", "no", "off")


def _append_missing_paragraph_hints(
    body: str,
    items: Optional[List[Dict[str, Any]]],
    *,
    max_chars: int = 4000,
    max_line_len: int = 500,
) -> str:
    """Append short paragraph texts from OCR items that are absent from *body* (substring check).

    Covers masthead/DOI lines dropped by markdown export without duplicating tables/formulas.
    """
    if not items or not body:
        return body
    extras: List[str] = []
    total = 0
    seen: set[str] = set()
    for it in items:
        if not isinstance(it, dict) or it.get("typer") != "paragraph":
            continue
        t = (it.get("text") or "").strip()
        if len(t) < 6 or len(t) > max_line_len:
            continue
        if t in seen:
            continue
        if t in body:
            continue
        seen.add(t)
        extras.append(t)
        total += len(t) + 2
        if total >= max_chars:
            break
    if not extras:
        return body
    block = (
        "\n\n=== SUPPLEMENTARY OCR LINES (paragraph blocks missing from main text) ===\n\n"
        + "\n\n".join(extras)
    )
    return body + block


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw.strip())
    except ValueError:
        logger.warning("%s=%r is not a valid integer; using %d.", name, raw, default)
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw.strip())
    except ValueError:
        logger.warning("%s=%r is not a valid float; using %s.", name, raw, default)
        return default


def _chunked(seq: List[Any], size: int) -> List[List[Any]]:
    if size <= 0:
        return [seq]
    return [seq[i : i + size] for i in range(0, len(seq), size)]


def _build_ocr_quality_report(
    ocr_items: List[Dict[str, Any]],
    pp_status: Dict[str, Any],
    low_threshold: float,
) -> Dict[str, Any]:
    confidences: List[float] = []
    low_pages: set[int] = set()
    for it in ocr_items:
        c = it.get("confidence")
        if c is not None and isinstance(c, (int, float)):
            confidences.append(float(c))
            p = int(it.get("page", 0) or 0)
            if float(c) < low_threshold and p > 0:
                low_pages.add(p)
    tables = sum(1 for it in ocr_items if it.get("typer") == "table")
    formulas = sum(1 for it in ocr_items if it.get("typer") == "formula")
    avg_conf: Optional[float] = None
    if confidences:
        avg_conf = sum(confidences) / len(confidences)
    return {
        "ocr_avg_confidence": avg_conf,
        "ocr_low_confidence_pages": sorted(low_pages),
        "table_count": tables,
        "formula_count": formulas,
        "ppstructure_status": pp_status.get("ppstructure_status"),
        "ppstructure_detail": pp_status.get("ppstructure_detail"),
        "ppstructure_replacements": pp_status.get("ppstructure_replacements", 0),
    }


def _env_low_confidence_action() -> str:
    v = os.getenv("KNOWMAT_OCR_LOW_CONF_ACTION", "none").strip().lower()
    if v in ("none", "tag", "drop"):
        return v
    return "none"


def _apply_ocr_low_confidence_policy(
    items: List[Dict[str, Any]],
    threshold: float,
    action: str,
) -> List[Dict[str, Any]]:
    """Apply ``OCR_LOW_CONFIDENCE_THRESHOLD`` to ``paragraph`` items only (tables/formulas unchanged)."""
    if action == "none" or not items:
        return items
    out: List[Dict[str, Any]] = []
    for it in items:
        if it.get("typer") != "paragraph":
            out.append(it)
            continue
        conf = it.get("confidence")
        if conf is None or not isinstance(conf, (int, float)):
            out.append(it)
            continue
        if float(conf) >= threshold:
            out.append(it)
            continue
        if action == "drop":
            logger.debug(
                "Dropped paragraph below OCR_LOW_CONFIDENCE_THRESHOLD on page %s",
                it.get("page"),
            )
            continue
        tagged = dict(it)
        tagged["low_confidence"] = True
        txt = tagged.get("text") or ""
        prefix = "[Low confidence text] "
        if txt and not txt.startswith(prefix):
            tagged["text"] = prefix + txt
        out.append(tagged)
    return out


def _render_page(args: Tuple[str, int, int, str]) -> Path:
    import fitz  # type: ignore

    pdf_path_str, page_idx, dpi, image_dir_str = args
    image_dir = Path(image_dir_str)
    pdf_stem = Path(pdf_path_str).stem
    image_path = image_dir / f"{pdf_stem}-page-{page_idx:04d}.png"

    doc = fitz.open(pdf_path_str)
    try:
        page = doc[page_idx - 1]
        page.get_pixmap(dpi=dpi, alpha=False).save(str(image_path))
    finally:
        doc.close()
    return image_path


def _save_ocr_json(raw: Any, dest: Path) -> None:
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(raw, f, ensure_ascii=False, indent=2, default=str)


def _export_tables_to_csv(ocr_items: List[Dict[str, Any]], tables_dir: Path) -> None:
    """Export structured table items to CSV files for traceability."""
    import csv

    table_items = [it for it in ocr_items if it.get("typer") == "table"]
    if not table_items:
        return

    tables_dir.mkdir(parents=True, exist_ok=True)
    counters: Dict[int, int] = {}
    for item in table_items:
        page = item.get("page", 0)
        counters[page] = counters.get(page, 0) + 1
        n = counters[page]
        data = item.get("data", {})
        structured = data if isinstance(data, dict) and "rows" in data else None
        if structured is None:
            continue
        rows = structured.get("rows", [])
        columns = [c["name"] for c in structured.get("columns", [])]
        if not rows or not columns:
            continue
        csv_path = tables_dir / f"page{page:04d}-table{n:02d}.csv"
        try:
            with open(csv_path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows)
        except OSError as exc:
            logger.warning("Failed to write table CSV %s: %s", csv_path, exc)


_FIGURE_CROP_DPI = 200
_LEGACY_IMAGE_BOX_RE = re.compile(
    r"img_in_image_box_(\d+)_(\d+)_(\d+)_(\d+)\.(?:png|jpe?g|webp)$",
    re.IGNORECASE,
)


def _legacy_image_box_to_pdf_bbox(image_path: str, render_dpi: int) -> Optional[List[float]]:
    match = _LEGACY_IMAGE_BOX_RE.search(Path(str(image_path or "")).name)
    if not match:
        return None
    scale = 72.0 / max(1, render_dpi)
    x0, y0, x1, y1 = (float(group) for group in match.groups())
    return [x0 * scale, y0 * scale, x1 * scale, y1 * scale]


def _resolve_item_bbox_in_pdf_points(item: Dict[str, Any], render_dpi: int) -> Optional[List[float]]:
    bbox = item.get("bbox")
    if isinstance(bbox, list) and len(bbox) == 4:
        return [float(v) for v in bbox]
    data = item.get("data")
    if not isinstance(data, dict):
        return None
    legacy = _legacy_image_box_to_pdf_bbox(str(data.get("image_path") or ""), render_dpi)
    if legacy is not None:
        item["bbox"] = legacy
    return legacy


def _caption_block_for_figure(
    pdf_path: str,
    page_idx: int,
    figure_num: str,
    caption: str,
) -> Optional[Tuple[float, float, float, float]]:
    try:
        import fitz  # type: ignore
    except ImportError:
        return None

    block_prefixes = [f"figure {figure_num}", f"fig. {figure_num}", f"fig {figure_num}"]
    caption_prefix = " ".join(str(caption or "").strip().split())[:48].lower()
    try:
        doc = fitz.open(pdf_path)
        page = doc[page_idx - 1]
        for block in page.get_text("blocks"):
            x0, y0, x1, y1, text, *_ = block
            normalized = " ".join(str(text or "").strip().split()).lower()
            if not normalized:
                continue
            if any(normalized.startswith(prefix) for prefix in block_prefixes):
                doc.close()
                return (float(x0), float(y0), float(x1), float(y1))
            if caption_prefix and caption_prefix[:24] in normalized:
                doc.close()
                return (float(x0), float(y0), float(x1), float(y1))
        doc.close()
    except Exception as exc:
        logger.debug("Could not resolve caption block for Figure %s on page %d: %s", figure_num, page_idx, exc)
    return None


def _estimate_figure_bbox_from_caption(
    pdf_path: str,
    page_idx: int,
    figure_num: str,
    caption: str,
) -> Optional[List[float]]:
    try:
        import fitz  # type: ignore
    except ImportError:
        return None

    caption_block = _caption_block_for_figure(pdf_path, page_idx, figure_num, caption)
    if caption_block is None:
        return None

    try:
        doc = fitz.open(pdf_path)
        page = doc[page_idx - 1]
        page_rect = page.rect
        cx0, cy0, cx1, _ = caption_block
        caption_width = max(1.0, cx1 - cx0)

        x_pad = min(96.0, max(36.0, caption_width * 0.22))
        x0 = max(float(page_rect.x0) + 18.0, cx0 - x_pad)
        x1 = min(float(page_rect.x1) - 18.0, cx1 + x_pad)

        nearest_above = float(page_rect.y0) + 36.0
        for block in page.get_text("blocks"):
            bx0, by0, bx1, by1, text, *_ = block
            normalized = " ".join(str(text or "").strip().split())
            if not normalized:
                continue
            if by1 >= cy0 - 4.0:
                continue
            overlap = min(x1, float(bx1)) - max(x0, float(bx0))
            if overlap <= 0:
                continue
            nearest_above = max(nearest_above, float(by1) + 8.0)

        y0 = nearest_above
        y1 = max(y0 + 80.0, cy0 - 8.0)
        if y1 - y0 < 120.0:
            y0 = max(float(page_rect.y0) + 36.0, y1 - 220.0)

        doc.close()
        return [x0, y0, x1, y1]
    except Exception as exc:
        logger.debug("Could not estimate figure bbox for Figure %s on page %d: %s", figure_num, page_idx, exc)
        return None


def _persist_figure_images(
    ocr_items: List[Dict[str, Any]],
    pdf_path: str,
    figures_dir: Path,
    *,
    render_dpi: int,
) -> List[Dict[str, Any]]:
    """Persist durable figure images for resolved figure OCR items.

    Supports both current ``bbox``-bearing items and legacy PaddleOCR-VL image
    paths like ``imgs/img_in_image_box_<x0>_<y0>_<x1>_<y1>.jpg`` by decoding the
    embedded crop coordinates back into PDF-point space.
    """
    normalize_figure_ocr_items(ocr_items)
    image_items = iter_figure_caption_items(ocr_items)
    if not image_items:
        return ocr_items

    try:
        figures_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.warning("Cannot create figures directory %s: %s", figures_dir, exc)
        return ocr_items

    counters: Dict[int, int] = {}
    updated_any = False
    for item in image_items:
        data = item.setdefault("data", {})
        page = int(item.get("page") or 0)
        if page <= 0:
            continue
        figure_num = str(data.get("figure_num") or "").strip()
        figure_key = re.sub(r"[^0-9A-Za-z._-]+", "_", figure_num).strip("_")
        if figure_key:
            out_path = figures_dir / f"page{page:04d}-figure{figure_key}.jpg"
        else:
            counters[page] = counters.get(page, 0) + 1
            n = counters[page]
            out_path = figures_dir / f"page{page:04d}-fig{n:02d}.jpg"

        if out_path.is_file():
            data["image_path"] = str(out_path.resolve())
            updated_any = True
            continue

        # If MinerU API already extracted the image, copy it to figures_dir
        existing_img = data.get("image_path", "")
        if existing_img and Path(existing_img).is_file():
            import shutil as _shutil
            _shutil.copy2(existing_img, out_path)
            data["image_path"] = str(out_path.resolve())
            updated_any = True
            continue

        bbox = _resolve_item_bbox_in_pdf_points(item, render_dpi)
        saved = None
        if bbox is None and figure_num:
            bbox = _estimate_figure_bbox_from_caption(
                pdf_path,
                page,
                figure_num,
                str(data.get("caption") or ""),
            )
            if bbox is not None:
                item["bbox"] = bbox
                updated_any = True
        if bbox is not None:
            saved = _crop_page_image(pdf_path, page, bbox, _FIGURE_CROP_DPI, out_path)
        if saved is not None:
            data["image_path"] = str(saved.resolve())
            updated_any = True
            logger.debug("Saved figure crop: %s", saved)

    if updated_any:
        normalize_figure_ocr_items(ocr_items)
        image_items = iter_figure_caption_items(ocr_items)
    figure_count = sum(
        1 for item in image_items if Path(str(item.get("data", {}).get("image_path") or "")).is_file()
    )
    if figure_count:
        logger.info("Cropped and saved %d figure image(s) to %s", figure_count, figures_dir)
    return ocr_items


def _candidate_pdf_for_text_source(source_path: Path) -> Optional[Path]:
    stem = source_path.stem
    candidates = [
        source_path.with_suffix(".pdf"),
        source_path.parent / f"{stem}.pdf",
        source_path.parent.parent / f"{stem}.pdf",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _pdf_page_count_if_available(pdf_path: Path) -> Optional[int]:
    """Return PDF page count when PyMuPDF is installed; otherwise None."""
    try:
        import fitz  # type: ignore
    except ImportError:
        return None

    doc = fitz.open(str(pdf_path))
    try:
        return len(doc)
    finally:
        doc.close()


def _extract_pdf_with_paddleocrvl(
    pdf_path: str,
    output_dir: str,
    model_dir: Path,
    save_intermediate: bool = True,
    page_indices: Optional[List[int]] = None,
) -> Tuple[str, Dict[str, Any], List[Dict[str, Any]]]:
    try:
        import fitz  # type: ignore
    except ImportError as exc:
        raise ImportError("PyMuPDF is required for PaddleOCR parsing. Install with: pip install pymupdf") from exc

    pdf = Path(pdf_path)
    out_dir = Path(output_dir)
    temp_pdf_ctx = None
    work_pdf = pdf
    if not all(ord(c) < 128 for c in pdf_path):
        temp_pdf_ctx = tempfile.TemporaryDirectory()
        work_pdf = Path(temp_pdf_ctx.name) / "source.pdf"
        shutil.copy2(pdf_path, work_pdf)

    temp_dir = None
    if save_intermediate:
        image_dir = out_dir / "page_images"
        raw_dir = out_dir / "ocr_raw"
        image_dir.mkdir(parents=True, exist_ok=True)
        raw_dir.mkdir(parents=True, exist_ok=True)
    else:
        temp_dir = tempfile.TemporaryDirectory()
        image_dir = Path(temp_dir.name)
        raw_dir = None

    max_render_workers = _env_int("OCR_MAX_RENDER_WORKERS", _DEFAULT_MAX_RENDER_WORKERS)
    header_lines = _env_int("OCR_HEADER_LINES", _DEFAULT_OCR_HEADER_LINES)
    footer_lines = _env_int("OCR_FOOTER_LINES", _DEFAULT_OCR_FOOTER_LINES)
    low_conf_thr = _env_float("OCR_LOW_CONFIDENCE_THRESHOLD", _DEFAULT_OCR_LOW_CONFIDENCE)
    pages_per_release = _env_int("OCR_PAGES_PER_RELEASE", 0)

    engine: Any = None
    backend = ""
    try:
        engine, backend = create_ocr_engine(model_dir=model_dir)

        doc = fitz.open(str(work_pdf))
        total_pages = len(doc)
        if total_pages <= 0:
            doc.close()
            return "", {"backend": backend, "pages": 0, "ocr_items": 0}, []

        if page_indices is None:
            selected_pages = list(range(1, total_pages + 1))
        else:
            selected_pages = sorted({p for p in page_indices if 1 <= p <= total_pages})
            if not selected_pages:
                doc.close()
                raise ValueError("No valid pages in ocr_pages range for this PDF.")

        if 1 in selected_pages:
            first_page_pymupdf_text = doc[0].get_text("text")
        else:
            first_page_pymupdf_text = doc[selected_pages[0] - 1].get_text("text")
        doc.close()

        render_dpi = _env_int("OCR_RENDER_DPI", 300)
        if render_dpi < 72 or render_dpi > 600:
            logger.warning("OCR_RENDER_DPI=%s out of range [72,600]; using 300.", render_dpi)
            render_dpi = 300

        render_args = [(str(work_pdf), idx, render_dpi, str(image_dir)) for idx in selected_pages]
        num_workers = min(max_render_workers, max(1, os.cpu_count() or 1), len(selected_pages))
        with ThreadPoolExecutor(max_workers=num_workers) as render_pool:
            image_paths: List[Path] = list(render_pool.map(_render_page, render_args))

        page_blocks: List[str] = []
        page_level_meta: List[Dict[str, Any]] = []
        ocr_items: List[Dict[str, Any]] = []
        first_page_full_text = first_page_pymupdf_text
        bg_futures: List[Future[Any]] = []
        vl_results: List[Any] = []

        batch_size = max(1, _env_int("OCR_BATCH_SIZE", 2))
        raw_results: List[Any] = []

        if pages_per_release > 0:
            logger.info(
                "OCR_PAGES_PER_RELEASE=%d: running inference in segments with GPU memory release between segments.",
                pages_per_release,
            )
            for chunk_paths in _chunked(image_paths, pages_per_release):
                if supports_batch_predict(engine):
                    bs = min(batch_size, len(chunk_paths))
                    raw_results.extend(run_ocr_batch(engine, chunk_paths, bs))
                else:
                    raw_results.extend(run_ocr_sequential(engine, chunk_paths))
                try_release_paddle_gpu_memory()
        else:
            if supports_batch_predict(engine):
                logger.info(
                    "Using batch OCR inference (batch_size=%d) for %d pages...",
                    batch_size,
                    len(image_paths),
                )
                raw_results = run_ocr_batch(engine, image_paths, batch_size)
                logger.info("✓ OCR batch inference completed.")
            else:
                max_ocr_workers = _env_int("OCR_MAX_WORKERS", _DEFAULT_OCR_MAX_WORKERS)
                if max_ocr_workers > 1:
                    logger.warning(
                        "OCR_MAX_WORKERS=%d ignored: shared OCR engine is not safe for concurrent inference; using sequential mode.",
                        max_ocr_workers,
                    )
                logger.info("Using sequential OCR inference for %d pages...", len(image_paths))
                raw_results = run_ocr_sequential(engine, image_paths)
                logger.info("✓ OCR sequential inference completed.")

        with ThreadPoolExecutor(max_workers=2) as bg_io_pool:
            try:
                for page_idx, raw in zip(selected_pages, raw_results):
                    if raw is not None and raw_dir is not None:
                        dest = raw_dir / f"page-{page_idx:04d}.json"
                        bg_futures.append(bg_io_pool.submit(_save_ocr_json, raw, dest))

                    if backend == "paddleocrvl":
                        if raw is not None:
                            if isinstance(raw, list):
                                vl_results.extend(raw)
                            else:
                                vl_results.append(raw)
                    else:
                        lines = normalize_lines(paddleocr_raw_to_lines(raw))
                        if not lines:
                            doc = fitz.open(str(work_pdf))
                            try:
                                fallback = doc[page_idx - 1].get_text("text")
                            finally:
                                doc.close()
                            lines = [x.strip() for x in fallback.splitlines() if x.strip()]

                        page_text = convert_html_to_markdown("\n".join(lines).strip())
                        page_blocks.append(f"## Page {page_idx}\n\n{page_text}")
                        page_level_meta.append(
                            {
                                "page": page_idx,
                                "header_text": "\n".join(lines[:header_lines]),
                                "footer_text": "\n".join(lines[-footer_lines:])
                                if len(lines) >= footer_lines
                                else "",
                                "line_count": len(lines),
                            }
                        )
                    if page_idx == selected_pages[0]:
                        first_page_full_text = (
                            first_page_pymupdf_text
                            if backend == "paddleocrvl"
                            else page_text + "\n" + first_page_pymupdf_text
                        )
            finally:
                for fut in bg_futures:
                    fut.result()

            if backend == "paddleocrvl" and vl_results:
                logger.info("Starting restructure_pages for layout analysis...")
                try:
                    _ignore_labels = list(_LAYOUT_NOISE_LABELS)
                    restructure_kwargs: Dict[str, Any] = {
                        "merge_tables": True,
                        "relevel_titles": True,
                        # False keeps one restructure result per OCR page; True returns a single
                        # merged page and breaks zip(selected_pages, restructured) (wrong meta/blocks).
                        "concatenate_pages": False,
                    }
                    try:
                        restructured = engine.restructure_pages(
                            vl_results, markdown_ignore_labels=_ignore_labels, **restructure_kwargs
                        )
                    except TypeError:
                        restructured = engine.restructure_pages(vl_results, **restructure_kwargs)
                    restructured = list(restructured)
                    logger.info("✓ restructure_pages completed: %d pages", len(restructured))
                except (RuntimeError, ValueError, TypeError) as exc:
                    logger.warning("restructure_pages failed for %s: %s", pdf, exc, exc_info=True)
                    restructured = []

                if len(restructured) != len(selected_pages):
                    logger.warning(
                        "restructure_pages returned %d page(s) but %d page(s) were OCR'd; aligning by position.",
                        len(restructured),
                        len(selected_pages),
                    )

                for pdf_page, res in zip(selected_pages, restructured):
                    try:
                        md_info = res._to_markdown(pretty=True, show_formula_number=False)
                        page_text = md_info.get("markdown_texts", "")
                    except (AttributeError, TypeError, KeyError) as exc:
                        logger.warning("_to_markdown failed for page %d of %s: %s", pdf_page, pdf, exc)
                        page_text = ""
                    page_text = convert_html_to_markdown(page_text)
                    if page_text:
                        page_blocks.append(page_text)

                    lines = [ln for ln in page_text.splitlines() if ln.strip()]
                    page_level_meta.append(
                        {
                            "page": pdf_page,
                            "header_text": "\n".join(lines[:header_lines]),
                            "footer_text": "\n".join(lines[-footer_lines:])
                            if len(lines) >= footer_lines
                            else "",
                            "line_count": len(lines),
                        }
                    )

                    try:
                        blocks = res["parsing_res_list"]
                    except (KeyError, TypeError, IndexError):
                        blocks = getattr(res, "parsing_res_list", [])
                    for block in blocks or []:
                        blabel = (
                            block.get("block_label")
                            if isinstance(block, dict)
                            else getattr(block, "block_label", None)
                        )
                        bbox = (
                            block.get("block_bbox")
                            if isinstance(block, dict)
                            else getattr(block, "block_bbox", None)
                        )
                        is_noise = blabel in _LAYOUT_NOISE_LABELS
                        
                        # Filter out general layout noise based on bbox position for header/footer
                        if bbox is not None and len(bbox) == 4:
                            _, y0, _, y1 = bbox
                            # Use typical PDF dimensions (e.g. 842 points height) as heuristic if needed
                            # but simple fractional check works if we assume 1.0 is max
                            # Since we don't have page height here, we rely primarily on blabel
                            pass
                            
                        item = block_to_item(block)
                        if item:
                            item["page"] = pdf_page
                            if bbox is not None:
                                item["bbox"] = bbox
                            if blabel is not None:
                                item["block_label"] = blabel
                            if is_noise:
                                item["is_layout_noise"] = True
                            ocr_items.append(item)

        merged = "\n\n".join(page_blocks).strip()
        if not ocr_items and merged:
            ocr_items = text_to_paragraph_items(merged)
        if backend == "paddleocr":
            seeded = seed_legacy_complex_items_from_ppstructurev3(selected_pages, image_paths)
            if seeded:
                ocr_items.extend(seeded)

        if ocr_items:
            try:
                ocr_items, n_removed = filter_figure_internal_fragments(ocr_items)
                if n_removed:
                    logger.info("Removed %d figure-internal fragment(s) from %s", n_removed, pdf)
            except Exception as exc:
                logger.warning("filter_figure_internal_fragments failed: %s", exc, exc_info=True)

        lc_action = _env_low_confidence_action()
        if ocr_items and lc_action != "none":
            ocr_items = _apply_ocr_low_confidence_policy(ocr_items, low_conf_thr, lc_action)

        pp_report: Dict[str, Any] = {
            "ppstructure_status": "not_applicable",
            "ppstructure_detail": "",
            "ppstructure_replacements": 0,
        }
        if ocr_items:
            logger.info("Starting PP-StructureV3 refinement for tables and formulas...")
            page_images_map: Dict[int, Path] = dict(zip(selected_pages, image_paths))
            reocr_work_dir = out_dir if save_intermediate else Path(image_dir)
            try:
                ocr_items, pp_report = route_and_reocr(
                    ocr_items,
                    str(work_pdf),
                    page_images_map,
                    engine,
                    reocr_work_dir,
                )
                logger.info("✓ PP-StructureV3 table/formula refinement completed.")
            except Exception as exc:
                logger.warning("route_and_reocr failed: %s", exc, exc_info=True)
                pp_report = {
                    "ppstructure_status": "failed",
                    "ppstructure_detail": str(exc),
                    "ppstructure_replacements": 0,
                }
            # Release GPU memory after PP-StructureV3 refinement to reduce peak usage
            # This is especially important for 8GB VRAM cards when processing PDFs with many tables/formulas
            try_release_paddle_gpu_memory()
            
            if not _env_truthy("KNOWMAT_SKIP_CHEM_REOCR"):
                try:
                    ocr_items = reocr_chem_formula_blocks(
                        ocr_items,
                        str(work_pdf),
                        page_images_map,
                        engine,
                        reocr_work_dir,
                    )
                    logger.info("✓ Chemical formula re-OCR completed.")
                except Exception as exc:
                    logger.warning("reocr_chem_formula_blocks failed: %s", exc, exc_info=True)
                # Release GPU memory after chemical formula re-OCR
                try_release_paddle_gpu_memory()
            else:
                logger.info("KNOWMAT_SKIP_CHEM_REOCR set: skipping chemical formula crop re-OCR.")

        doi = extract_doi_from_pdf_metadata(str(work_pdf)) or extract_first_doi(first_page_full_text)

        if save_intermediate and raw_dir is not None:
            tables_dir = out_dir / "tables"
            _export_tables_to_csv(ocr_items, tables_dir)

        quality = _build_ocr_quality_report(ocr_items, pp_report, low_conf_thr)

        metadata = {
            "backend": backend,
            "model_dir": str(model_dir),
            "pages": len(selected_pages),
            "render_dpi": render_dpi,
            "image_dir": str(image_dir),
            "ocr_raw_dir": str(raw_dir) if raw_dir is not None else None,
            "doi": doi,
            "page_level_metadata": page_level_meta,
            "ocr_items": len(ocr_items),
            "ocr_quality": quality,
        }
        return merged, metadata, ocr_items
    finally:
        # Release OCR engine resources
        if engine is not None:
            # PaddleOCR-VL's close() can tear down the CUDA context such that the next PDF
            # still hits Place(undefined:0) inside Paddle despite set_device("gpu:0"). Skip
            # close by default; opt in with KNOWMAT_PADDLEOCR_VL_EXPLICIT_CLOSE=1 if you need
            # aggressive native cleanup between PDFs.
            skip_vl_close = backend == "paddleocrvl" and not _env_truthy(
                "KNOWMAT_PADDLEOCR_VL_EXPLICIT_CLOSE"
            )
            if skip_vl_close:
                logger.debug(
                    "Skipping PaddleOCR-VL engine.close() between PDFs (set "
                    "KNOWMAT_PADDLEOCR_VL_EXPLICIT_CLOSE=1 to force close)."
                )
            else:
                if hasattr(engine, "close"):
                    try:
                        engine.close()
                        logger.debug("OCR engine closed via close() method")
                    except Exception as e:
                        logger.debug("OCR engine close() failed (non-critical): %s", e)
                elif hasattr(engine, "destroy"):
                    try:
                        engine.destroy()
                        logger.debug("OCR engine destroyed via destroy() method")
                    except Exception as e:
                        logger.debug("OCR engine destroy() failed (non-critical): %s", e)
            del engine
            engine = None
        # Release PP-StructureV3 pipeline
        release_ppstructurev3_pipeline()
        # Release Paddle GPU memory
        try_release_paddle_gpu_memory()
        # After close()/pipeline drop/empty_cache, Paddle may leave Place(undefined:0); fix for next PDF.
        ensure_paddle_device_from_env()
        # Cleanup temporary directories
        if temp_dir is not None:
            temp_dir.cleanup()
        if temp_pdf_ctx is not None:
            temp_pdf_ctx.cleanup()


def _read_txt_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="ignore")


def _use_mineru_api_mode() -> str:
    """Determine MinerU compatibility mode: 'precision', 'lightweight', or disabled."""
    backend = os.getenv("KNOWMAT_OCR_BACKEND", "").strip().lower()
    if backend not in {"mineru", "mineru-api"} and not _env_truthy("KNOWMAT_USE_MINERU_API"):
        return ""
    if os.getenv("MINERU_API_KEY", "").strip():
        return "precision"
    if _env_truthy("MINERU_USE_LIGHTWEIGHT"):
        return "lightweight"
    return ""


def _extract_pdf_with_mineru_api(
    pdf_path: str,
    output_dir: str,
    save_intermediate: bool = True,
    page_indices: Optional[List[int]] = None,
) -> Tuple[str, Dict[str, Any], List[Dict[str, Any]]]:
    """Extract PDF content via MinerU cloud API.

    Produces the same (extracted_text, metadata, ocr_items) triple as
    _extract_pdf_with_paddleocrvl() so downstream code is unaffected.
    """
    api_key = os.getenv("MINERU_API_KEY", "").strip()
    base_url = os.getenv("MINERU_API_BASE_URL", "https://mineru.net").strip()
    model_version = os.getenv("MINERU_MODEL_VERSION", "vlm").strip()
    timeout = _env_float("MINERU_API_TIMEOUT_SEC", 600.0)
    language = os.getenv("MINERU_LANGUAGE", "en").strip()

    pdf = Path(pdf_path)
    out_dir = Path(output_dir)
    page_ranges = _pages_range_from_indices(page_indices)

    mode = _use_mineru_api_mode()

    if mode == "lightweight":
        client = MineruLightweightClient(base_url)
        markdown_text = client.upload_and_parse(pdf, page_range=page_ranges)
        extracted_text, metadata, ocr_items = convert_lightweight_md_to_knowmat(markdown_text, pdf_path)
        return extracted_text, metadata, ocr_items

    client = MineruPrecisionClient(api_key, base_url)
    task_result = client.upload_and_parse(
        pdf,
        model_version=model_version,
        is_ocr=True,
        enable_formula=True,
        enable_table=True,
        language=language,
        page_ranges=page_ranges,
    )

    zip_url = task_result.get("full_zip_url", "")
    if not zip_url:
        raise MineruAPIError("MinerU task completed but no full_zip_url in response")

    mineru_raw_dir = out_dir / "mineru_raw" if save_intermediate else Path(tempfile.mkdtemp())
    extracted_dir = client.download_and_extract_zip(zip_url, mineru_raw_dir)

    content_list: List[Dict[str, Any]] = []
    full_md = ""
    # MinerU ZIP files are named <uuid>_content_list.json, not just content_list.json
    cl_candidates = sorted(extracted_dir.glob("*_content_list.json"))
    # Exclude v2 format
    cl_candidates = [f for f in cl_candidates if "_content_list_v2" not in f.name]
    if not cl_candidates:
        cl_candidates = list(extracted_dir.rglob("*content_list.json"))
        cl_candidates = [f for f in cl_candidates if "_content_list_v2" not in f.name]
    if cl_candidates:
        cl_path = cl_candidates[0]
        content_list = json.loads(cl_path.read_text("utf-8"))
        logger.info("[MinerU API] Loaded content_list from %s (%d items)", cl_path.name, len(content_list))

    md_path = extracted_dir / "full.md"
    if md_path.is_file():
        full_md = md_path.read_text("utf-8")
    elif not full_md:
        for md in extracted_dir.rglob("*.md"):
            full_md = md.read_text("utf-8")
            break

    figures_dest = out_dir / "images"
    figures_dest.mkdir(parents=True, exist_ok=True)
    extracted_text, metadata, ocr_items = convert_mineru_to_knowmat(
        content_list, full_md, pdf_path, extracted_dir, page_indices, figures_dest
    )

    metadata["mineru_task_id"] = task_result.get("task_id", "")
    metadata["mineru_model_version"] = model_version

    if not save_intermediate and mineru_raw_dir != out_dir / "mineru_raw":
        shutil.rmtree(mineru_raw_dir, ignore_errors=True)

    return extracted_text, metadata, ocr_items


def _use_paddleocr_api_mode() -> bool:
    """Return whether PaddleOCR API should be used as the primary OCR backend."""
    backend = os.getenv("KNOWMAT_OCR_BACKEND", "auto").strip().lower()
    if backend in {"paddleocr-api", "paddle-api"}:
        return True
    if backend in {"paddleocr-local", "local", "mineru", "mineru-api", "openai", "openai-compatible"}:
        return False
    return _env_truthy("KNOWMAT_USE_PADDLEOCR_API") or bool(
        os.getenv("PADDLEOCR_API_TOKEN", "").strip()
    )


def _use_openai_ocr_mode() -> bool:
    """Return whether OpenAI-compatible vision OCR should be used."""
    backend = os.getenv("KNOWMAT_OCR_BACKEND", "").strip().lower()
    return backend in {"openai", "openai-compatible"} or _env_truthy("KNOWMAT_USE_OPENAI_OCR")


def _extract_pdf_with_openai_ocr(
    pdf_path: str,
    output_dir: str,
    save_intermediate: bool = True,
    page_indices: Optional[List[int]] = None,
) -> Tuple[str, Dict[str, Any], List[Dict[str, Any]]]:
    """Extract PDF content via an OpenAI-compatible vision chat endpoint."""
    try:
        import fitz  # type: ignore
    except ImportError as exc:
        raise RuntimeError("PyMuPDF is required for OpenAI-compatible OCR.") from exc

    pdf = Path(pdf_path)
    out_dir = Path(output_dir)
    render_dpi = _env_int("KNOWMAT_OPENAI_OCR_RENDER_DPI", _env_int("OCR_RENDER_DPI", 200))
    if render_dpi < 72 or render_dpi > 400:
        logger.warning("KNOWMAT_OPENAI_OCR_RENDER_DPI=%s out of range [72,400]; using 200.", render_dpi)
        render_dpi = 200

    doc = fitz.open(str(pdf))
    try:
        total_pages = len(doc)
    finally:
        doc.close()

    if page_indices is None:
        selected_pages = list(range(1, total_pages + 1))
    else:
        selected_pages = sorted({p for p in page_indices if 1 <= p <= total_pages})
        if not selected_pages:
            raise ValueError("No valid pages in ocr_pages range for this PDF.")

    if save_intermediate:
        image_dir = out_dir / "openai_ocr_pages"
        raw_dir = out_dir / "openai_ocr_raw"
        image_dir.mkdir(parents=True, exist_ok=True)
        raw_dir.mkdir(parents=True, exist_ok=True)
        temp_dir = None
    else:
        temp_dir = tempfile.TemporaryDirectory()
        image_dir = Path(temp_dir.name) / "pages"
        raw_dir = None
        image_dir.mkdir(parents=True, exist_ok=True)

    client = OpenAIOCRClient()
    page_results: List[Dict[str, Any]] = []
    try:
        for page in selected_pages:
            image_path = _render_page((str(pdf), page, render_dpi, str(image_dir)))
            logger.info("[OpenAI OCR] Running %s on %s page %d", client.config.model, pdf.name, page)
            markdown = client.extract_page_markdown(image_path, page_number=page)
            page_payload = {
                "page": page,
                "markdown": markdown,
                "image_path": str(image_path),
            }
            page_results.append(page_payload)
            if raw_dir is not None:
                _save_ocr_json(page_payload, raw_dir / f"page-{page:04d}.json")

        extracted_text, metadata, ocr_items = build_openai_ocr_result(
            page_results,
            pdf_path=pdf_path,
            config=client.config,
        )
        metadata["render_dpi"] = render_dpi
        metadata["selected_pages"] = selected_pages
        return extracted_text, metadata, ocr_items
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


def _extract_pdf_with_paddleocr_api(
    pdf_path: str,
    output_dir: str,
    save_intermediate: bool = True,
    page_indices: Optional[List[int]] = None,
) -> Tuple[str, Dict[str, Any], List[Dict[str, Any]]]:
    """Extract PDF content via PaddleOCR cloud API (VL-1.5 + PP-StructureV3).

    Produces the same (extracted_text, metadata, ocr_items) triple as other backends.
    """
    token = os.getenv("PADDLEOCR_API_TOKEN", "").strip()
    base_url = os.getenv("PADDLEOCR_API_URL", "https://paddleocr.aistudio-app.com/api/v2/ocr/jobs").strip()
    timeout = _env_float("PADDLEOCR_API_TIMEOUT_SEC", 600.0)

    pdf = Path(pdf_path)
    out_dir = Path(output_dir)

    client = PaddleOCRAPIClient(token, base_url)

    # Step 1: Run PaddleOCR-VL for primary OCR (version from env, default 1.5)
    from ppmatAgent.knowmat.pdf.paddleocr_api_client import get_paddleocr_vl_model
    vl_model = get_paddleocr_vl_model()
    logger.info("[PaddleOCR API] Running %s on %s...", vl_model, pdf.name)
    job_result = client.upload_and_parse(pdf, model=vl_model, timeout_sec=timeout)

    jsonl_url = job_result.get("resultUrl", {}).get("jsonUrl", "")
    if not jsonl_url:
        raise PaddleOCRAPIError("PaddleOCR job done but no jsonUrl in resultUrl")

    pages_data = client.download_jsonl(jsonl_url)

    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    extracted_text, metadata, ocr_items = convert_paddleocr_api_to_knowmat(
        pages_data, pdf_path, images_dir
    )

    # Step 2: Run PP-StructureV3 for formula/table refinement
    ocr_items, pp_report = _refine_with_ppstructurev3_api(pdf_path, ocr_items, output_dir)
    metadata["ocr_quality"].update(pp_report)

    metadata["paddleocr_api_model"] = vl_model
    return extracted_text, metadata, ocr_items


def _refine_with_ppstructurev3_api(
    pdf_path: str,
    ocr_items: List[Dict[str, Any]],
    output_dir: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Run PP-StructureV3 API on the PDF and refine formula/table items in ocr_items.

    Matches formulas and tables by page number and reading order index.
    Returns (updated_ocr_items, pp_report_dict).
    """
    token = os.getenv("PADDLEOCR_API_TOKEN", "").strip()
    if not token:
        return ocr_items, {
            "ppstructure_status": "skipped",
            "ppstructure_detail": "PADDLEOCR_API_TOKEN not set",
            "ppstructure_replacements": 0,
        }

    base_url = os.getenv("PADDLEOCR_API_URL", "https://paddleocr.aistudio-app.com/api/v2/ocr/jobs").strip()
    timeout = _env_float("PADDLEOCR_API_TIMEOUT_SEC", 600.0)
    pdf = Path(pdf_path)

    client = PaddleOCRAPIClient(token, base_url)

    try:
        logger.info("[PP-StructureV3 API] Running formula/table refinement on %s...", pdf.name)
        job_result = client.upload_and_parse(pdf, model="PP-StructureV3", timeout_sec=timeout)

        jsonl_url = job_result.get("resultUrl", {}).get("jsonUrl", "")
        if not jsonl_url:
            logger.warning("[PP-StructureV3 API] No jsonUrl in result, skipping refinement")
            return ocr_items, {
                "ppstructure_status": "failed",
                "ppstructure_detail": "no jsonUrl returned",
                "ppstructure_replacements": 0,
            }

        pp_pages_data = client.download_jsonl(jsonl_url)
    except (PaddleOCRAPIError, Exception) as exc:
        logger.warning("[PP-StructureV3 API] Refinement failed: %s", exc)
        return ocr_items, {
            "ppstructure_status": "failed",
            "ppstructure_detail": str(exc)[:200],
            "ppstructure_replacements": 0,
        }

    # Extract formulas and tables from PP-StructureV3 result
    pp_formulas = extract_formulas_per_page(pp_pages_data)
    pp_tables = extract_tables_per_page(pp_pages_data)

    replacements = 0

    # Group formula ocr_items by page
    formula_items_by_page: Dict[int, List[int]] = {}
    table_items_by_page: Dict[int, List[int]] = {}
    for idx, item in enumerate(ocr_items):
        page = item.get("page", 0)
        if item.get("typer") == "formula":
            formula_items_by_page.setdefault(page, []).append(idx)
        elif item.get("typer") == "table":
            table_items_by_page.setdefault(page, []).append(idx)

    # Replace formulas by page + order matching
    for page, pp_formula_list in pp_formulas.items():
        item_indices = formula_items_by_page.get(page, [])
        for i, latex in enumerate(pp_formula_list):
            if i < len(item_indices):
                idx = item_indices[i]
                old_item = ocr_items[idx]
                old_data = old_item.get("data", {})
                old_data["text"] = latex
                old_data["latex"] = latex
                old_item["data"] = old_data
                old_item["reocr_source"] = "ppstructurev3_api_replace"
                ocr_items[idx] = old_item
                replacements += 1

    # Replace tables by page + order matching
    for page, pp_table_list in pp_tables.items():
        item_indices = table_items_by_page.get(page, [])
        for i, table_md in enumerate(pp_table_list):
            if i < len(item_indices):
                idx = item_indices[i]
                old_item = ocr_items[idx]
                old_data = old_item.get("data", {})
                old_data["text"] = table_md
                old_item["data"] = old_data
                old_item["reocr_source"] = "ppstructurev3_api_replace"
                ocr_items[idx] = old_item
                replacements += 1

    logger.info("[PP-StructureV3 API] Refinement done: %d replacements", replacements)
    return ocr_items, {
        "ppstructure_status": "applied_api",
        "ppstructure_detail": f"PP-StructureV3 API: {replacements} replacements",
        "ppstructure_replacements": replacements,
    }


def _finalize_pdf_parse(
    source_path: Path,
    parse_output_dir: Path,
    save_intermediate: bool,
    extracted_text: str,
    metadata: Dict[str, Any],
    _ocr_items: List[Dict[str, Any]],
    is_api_mode: bool = False,
    clean_as_paddle_api: bool = False,
) -> Dict[str, Any]:
    if clean_as_paddle_api:
        from ppmatAgent.knowmat.pdf.paddleocr_api_result_converter import clean_api_markdown
        extracted_text = clean_api_markdown(extracted_text)

    if _ocr_items:
        sanitize_ocr_items_vl_artifacts(_ocr_items)
        normalize_figure_ocr_items(_ocr_items)

    structured_text = normalize_leading_masthead_and_title(extracted_text)
    structured_text = structure_sections(structured_text)
    structured_text = repair_keywords_abstract_two_column_ocr(structured_text)
    structured_text = normalize_plain_author_superscripts(structured_text)
    structured_text = normalize_alloy_strings(structured_text)
    structured_text = format_formula_text(structured_text)
    structured_text = collapse_orphan_display_formulas(structured_text)
    cleaned_text = (
        strip_references_section(structured_text)
        if settings.trim_references_section
        else structured_text
    )
    # Prefer DOI from full OCR block list: merged markdown can omit lines still on parsing_res_list.
    resolved_doi = extract_first_doi_from_ocr_items(_ocr_items) or metadata.get("doi")
    if resolved_doi and resolved_doi not in cleaned_text:
        cleaned_text = f"DOI: {resolved_doi}\n\n{cleaned_text}"

    if _append_missing_ocr_paragraphs_enabled() and not is_api_mode:
        cleaned_text = _append_missing_paragraph_hints(cleaned_text, _ocr_items)

    pdf_name = source_path.stem
    final_md_path: Optional[Path] = None
    if save_intermediate:
        final_md_path = parse_output_dir / f"{pdf_name}_final_output.md"
        with open(final_md_path, "w", encoding="utf-8") as f:
            f.write(cleaned_text)
        print(f"Saved final markdown output to: {final_md_path}")

    doc_meta = {
        "backend": metadata.get("backend", "paddleocrvl"),
        "model_dir": metadata.get("model_dir"),
        "pages": metadata.get("pages"),
        "doi": resolved_doi,
        "page_level_metadata": metadata.get("page_level_metadata"),
        "ocr_quality": metadata.get("ocr_quality"),
        "paper_text_path": str(final_md_path) if final_md_path is not None else None,
        "figure_dir": metadata.get("figure_dir"),
    }
    if save_intermediate:
        meta_path = parse_output_dir / f"{pdf_name}_parse_metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        print(f"Saved parser metadata to: {meta_path}")
    return {
        "paper_text": cleaned_text,
        "document_metadata": doc_meta,
        "metadata": doc_meta,
        "ocr_items": _ocr_items or text_to_paragraph_items(cleaned_text),
        "paper_text_path": str(final_md_path) if final_md_path is not None else None,
    }


def parse_pdf_with_paddleocrvl(state: KnowMatState) -> dict:
    input_path = state.get("pdf_path")
    if not input_path:
        raise ValueError("No input file path provided in state for parse_pdf_with_paddleocrvl node.")
    save_intermediate = bool(state.get("save_intermediate", True))
    output_dir = state.get("output_dir", ".")
    source_path = Path(input_path)
    suffix = source_path.suffix.lower()

    if suffix in (".txt", ".md"):
        parse_output_dir = Path(output_dir) / "txt_parse" if save_intermediate else Path(output_dir)
        if save_intermediate:
            parse_output_dir.mkdir(parents=True, exist_ok=True)

        raw_text = _read_txt_file(source_path)
        md_text = convert_html_to_markdown(raw_text)
        md_text = normalize_leading_masthead_and_title(md_text)
        md_text = structure_sections(md_text)
        md_text = repair_keywords_abstract_two_column_ocr(md_text)
        md_text = normalize_plain_author_superscripts(md_text)
        md_text = normalize_alloy_strings(md_text)
        md_text = format_formula_text(md_text)
        cleaned_text = (
            strip_references_section(md_text) if settings.trim_references_section else md_text
        )
        stem = source_path.stem

        sidecar_items: Optional[List[Dict[str, Any]]] = None
        sidecar_path = source_path.with_suffix(".json")
        if sidecar_path.is_file():
            try:
                raw_side = json.loads(sidecar_path.read_text(encoding="utf-8"))
                if isinstance(raw_side, list):
                    sidecar_items = raw_side
            except (json.JSONDecodeError, OSError) as exc:
                logger.debug("Could not read OCR sidecar %s: %s", sidecar_path, exc)

        render_dpi = _env_int("OCR_RENDER_DPI", 300)
        # Detect API mode: images/ directory present means API-produced OCR
        api_images_dir = source_path.parent / "images"
        is_api_produced = api_images_dir.is_dir() and any(api_images_dir.iterdir())
        if sidecar_items:
            if is_api_produced:
                from ppmatAgent.knowmat.pdf.figure_items import promote_caption_paragraphs
                promote_caption_paragraphs(sidecar_items)
            normalize_figure_ocr_items(sidecar_items)
            if not is_api_produced:
                source_pdf = _candidate_pdf_for_text_source(source_path)
                if source_pdf is not None:
                    figures_dir = source_path.parent / "_ocr_cache" / "resolved_figures"
                    sidecar_items = _persist_figure_images(
                        sidecar_items,
                        str(source_pdf),
                        figures_dir,
                        render_dpi=render_dpi,
                    )

        doi_from_sidecar = extract_first_doi_from_ocr_items(sidecar_items)
        doi_from_text = extract_first_doi(cleaned_text[:5000])
        doi = doi_from_sidecar or doi_from_text
        if doi and doi not in cleaned_text:
            cleaned_text = f"DOI: {doi}\n\n{cleaned_text}"

        if is_api_produced:
            from ppmatAgent.knowmat.pdf.paddleocr_api_result_converter import clean_api_markdown
            cleaned_text = clean_api_markdown(cleaned_text)
        elif sidecar_items and _append_missing_ocr_paragraphs_enabled():
            cleaned_text = _append_missing_paragraph_hints(cleaned_text, sidecar_items)

        final_md_path: Optional[Path] = None
        if save_intermediate:
            final_md_path = parse_output_dir / f"{stem}_final_output.md"
            with open(final_md_path, "w", encoding="utf-8") as f:
                f.write(cleaned_text)
            print(f"Saved txt parsed output to: {final_md_path}")

        doc_meta: Dict[str, Any] = {
            "backend": "txt-direct",
            "source_file": str(source_path),
            "doi": doi,
            "ocr_sidecar": str(sidecar_path) if sidecar_path.is_file() else None,
            "paper_text_path": str(final_md_path) if final_md_path is not None else None,
        }
        if save_intermediate:
            meta_path = parse_output_dir / f"{stem}_parse_metadata.json"
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(doc_meta, f, ensure_ascii=False, indent=2)
            print(f"Saved parser metadata to: {meta_path}")
        ocr_items_out: List[Dict[str, Any]]
        if sidecar_items:
            ocr_items_out = list(sidecar_items)
        else:
            ocr_items_out = text_to_paragraph_items(cleaned_text)
        return {
            "paper_text": cleaned_text,
            "document_metadata": doc_meta,
            "metadata": doc_meta,
            "ocr_items": ocr_items_out,
            "paper_text_path": str(final_md_path) if final_md_path is not None else None,
        }

    if suffix != ".pdf":
        raise ValueError(
            f"Unsupported file type: {source_path.suffix}. Only .pdf, .txt, and .md are supported."
        )
    parse_output_dir = Path(output_dir) / "paddleocrvl_parse" if save_intermediate else Path(output_dir)
    if save_intermediate:
        parse_output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = default_model_dir()

    paddleocr_api_mode = _use_paddleocr_api_mode()
    mineru_mode = _use_mineru_api_mode()
    openai_ocr_mode = _use_openai_ocr_mode()
    use_api_mode = paddleocr_api_mode or mineru_mode or openai_ocr_mode

    total_pages = _pdf_page_count_if_available(source_path)
    if total_pages is None and not use_api_mode:
        raise RuntimeError("PyMuPDF is required for local PDF parsing. Install with: pip install pymupdf")

    raw_pages = state.get("ocr_pages")
    if raw_pages is None:
        raw_pages = os.getenv("KNOWMAT_OCR_PAGES", "")
    pages_spec = (raw_pages or "").strip()
    if pages_spec:
        if total_pages is None:
            raise RuntimeError("PyMuPDF is required to resolve --ocr-pages for API OCR backends.")
        selected_pages = parse_pages_argument(pages_spec, total_pages)
        if not selected_pages:
            raise ValueError("No valid pages in --ocr-pages / ocr_pages for this PDF.")
    else:
        selected_pages = list(range(1, total_pages + 1)) if total_pages is not None else None

    render_dpi = _env_int("OCR_RENDER_DPI", 300)
    if render_dpi < 72 or render_dpi > 600:
        render_dpi = 300
    vl_version = os.getenv("PADDLEOCRVL_VERSION", "1.5").strip() or "1.5"
    # PP-StructureV3 精修始终参与；缓存键固定为完整 VL+Structure 管线
    skip_pp = False
    skip_chem = _env_truthy("KNOWMAT_SKIP_CHEM_REOCR")
    pages_key = pages_key_for_cache(selected_pages or [], total_pages or 0)
    digest = md5_file_digest(source_path)

    # Cache signature only needed for local OCR mode
    sig = ""
    if not (paddleocr_api_mode or mineru_mode or openai_ocr_mode):
        sig = cache_signature_key(
            digest,
            render_dpi=render_dpi,
            vl_version=vl_version,
            pages_key=pages_key,
            skip_ppstructure=skip_pp,
            skip_chem_reocr=skip_chem,
        )
    # For API modes, skip _ocr_cache — figures are saved to images/ during extraction,
    # and the .md/.json files themselves serve as the cache.
    if not use_api_mode:
        cache_bucket = ocr_cache_bucket(Path(output_dir), sig)
        skip_cache_read = bool(state.get("ocr_skip_cached", False)) or _env_truthy(
            "KNOWMAT_OCR_SKIP_CACHED"
        )
        no_cache_write = _env_truthy("KNOWMAT_OCR_NO_CACHE_WRITE")

        if not skip_cache_read:
            cached = try_load_ocr_cache(cache_bucket)
            if cached is not None:
                cached_items = list(cached.get("ocr_items") or [])
                cached_metadata = dict(cached.get("metadata") or {})
                cached_render_dpi = int(cached_metadata.get("render_dpi") or render_dpi)
                figures_dir = cache_bucket / "figures"
                cached_items = _persist_figure_images(
                    cached_items,
                    str(source_path),
                    figures_dir,
                    render_dpi=cached_render_dpi,
                )
                if figures_dir.exists():
                    cached_metadata["figure_dir"] = str(figures_dir.resolve())
                if not no_cache_write:
                    try:
                        save_ocr_cache(
                            cache_bucket,
                            {
                                "extracted_text": cached["extracted_text"],
                                "metadata": cached_metadata,
                                "ocr_items": cached_items,
                            },
                        )
                    except OSError as exc:
                        logger.warning("Could not refresh OCR cache at %s: %s", cache_bucket, exc)
                logger.info("Loaded OCR result from cache: %s", cache_bucket)
                return _finalize_pdf_parse(
                    source_path,
                    parse_output_dir,
                    save_intermediate,
                    cached["extracted_text"],
                    cached_metadata,
                    cached_items,
                )

    try:
        if openai_ocr_mode:
            extracted_text, metadata, _ocr_items = _extract_pdf_with_openai_ocr(
                str(source_path),
                str(parse_output_dir),
                save_intermediate=save_intermediate,
                page_indices=selected_pages,
            )
        elif paddleocr_api_mode:
            extracted_text, metadata, _ocr_items = _extract_pdf_with_paddleocr_api(
                str(source_path),
                str(parse_output_dir),
                save_intermediate=save_intermediate,
                page_indices=selected_pages,
            )
        elif mineru_mode:
            extracted_text, metadata, _ocr_items = _extract_pdf_with_mineru_api(
                str(source_path),
                str(parse_output_dir),
                save_intermediate=save_intermediate,
                page_indices=selected_pages,
            )
            # Apply PP-StructureV3 API refinement if token available
            if os.getenv("PADDLEOCR_API_TOKEN", "").strip():
                _ocr_items, pp_report = _refine_with_ppstructurev3_api(
                    str(source_path), _ocr_items, str(parse_output_dir)
                )
                metadata.setdefault("ocr_quality", {}).update(pp_report)
        else:
            extracted_text, metadata, _ocr_items = _extract_pdf_with_paddleocrvl(
                str(source_path),
                str(parse_output_dir),
                model_dir,
                save_intermediate=save_intermediate,
                page_indices=selected_pages,
            )

        # For local OCR, persist figures via bbox cropping and use _ocr_cache
        if not use_api_mode:
            figures_dir = cache_bucket / "figures"
            _ocr_items = _persist_figure_images(
                _ocr_items,
                str(source_path),
                figures_dir,
                render_dpi=int(metadata.get("render_dpi") or render_dpi),
            )
            if figures_dir.exists():
                metadata["figure_dir"] = str(figures_dir.resolve())
            if not no_cache_write:
                try:
                    save_ocr_cache(
                        cache_bucket,
                        {
                            "extracted_text": extracted_text,
                            "metadata": metadata,
                            "ocr_items": _ocr_items,
                        },
                    )
                except OSError as exc:
                    logger.warning("Could not write OCR cache to %s: %s", cache_bucket, exc)
        else:
            # For API modes, figure_dir is images/ under the paper directory
            images_dir = Path(parse_output_dir) / "images"
            if images_dir.exists():
                metadata["figure_dir"] = str(images_dir.resolve())

        result = _finalize_pdf_parse(
            source_path,
            parse_output_dir,
            save_intermediate,
            extracted_text,
            metadata,
            _ocr_items,
            is_api_mode=use_api_mode,
            clean_as_paddle_api=paddleocr_api_mode or bool(mineru_mode),
        )
        return result
    except MineruAPIError as exc:
        raise RuntimeError(f"Failed to parse PDF with MinerU API: {str(exc)}") from exc
    except PaddleOCRAPIError as exc:
        raise RuntimeError(f"Failed to parse PDF with PaddleOCR API: {str(exc)}") from exc
    except OpenAIOCRAPIError as exc:
        raise RuntimeError(f"Failed to parse PDF with OpenAI-compatible OCR: {str(exc)}") from exc
    except Exception as exc:
        if openai_ocr_mode:
            backend_name = "OpenAI-compatible OCR"
        elif paddleocr_api_mode:
            backend_name = "PaddleOCR API"
        elif mineru_mode:
            backend_name = "MinerU API"
        else:
            backend_name = "PaddleOCR-VL"
        raise RuntimeError(f"Failed to parse PDF with {backend_name}: {str(exc)}") from exc
