"""Batch OCR → _final.md pipeline orchestrator.

Two-phase workflow:
  Phase 1: Submit PDFs to cloud OCR API, enrich with CLIP+VLM as OCR completes.
  Phase 2: Repair loop — retry any paper whose _final.md is missing AI descriptions
           for figures that have valid image files. Retries indefinitely with
           exponential backoff until every describable figure has a description.

A paper is COMPLETE only when every figure with a valid image file has a
corresponding 'AI Description]:' block in its _final.md.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


# ── Completeness validation ───────────────────────────────────────────────────

def _load_figures_with_valid_images(paper_id: str, raw_dir: Path) -> List[str]:
    """Return figure_nums that have a valid image file on disk."""
    json_path = raw_dir / paper_id / f"{paper_id}.json"
    if not json_path.exists():
        return []
    try:
        ocr_items = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("[%s] Failed to read OCR JSON: %s", paper_id, exc)
        return []

    figures = [
        item for item in ocr_items
        if item.get("typer") == "image" or item.get("block_label") == "figure"
    ]

    seen: set = set()
    unique: List = []
    for fig in figures:
        fnum = fig.get("data", {}).get("figure_num", "")
        if fnum and fnum not in seen:
            seen.add(fnum)
            unique.append(fig)
        elif not fnum:
            unique.append(fig)

    valid = []
    for fig in unique:
        data = fig.get("data", {})
        fnum = data.get("figure_num", "")
        img_path_str = data.get("image_path", "")
        if img_path_str and Path(img_path_str).is_file() and fnum:
            valid.append(fnum)
    return valid


def _count_ai_descriptions(final_md_path: Path) -> int:
    """Count injected figure blocks in a _final.md (-1 if missing).

    Counts BOTH enrichment block kinds, since each figure gets exactly one:
      - prose:        '> [Figure N AI Description]:'
      - chart digitize: '> [Figure N VLM-digitized ...]:'
    """
    if not final_md_path.exists():
        return -1
    try:
        text = final_md_path.read_text(encoding="utf-8")
        return text.count("AI Description]:") + text.count("VLM-digitized")
    except Exception:
        return -1


def check_completeness(
    paper_id: str,
    raw_dir: Path,
    output_dir: Path,
) -> Tuple[bool, int, int]:
    """Return (is_complete, expected_count, actual_count).

    expected_count = figures with valid image files (VLM must describe these)
    actual_count   = AI Description blocks in _final.md (-1 = file missing)
    """
    final_md = output_dir / paper_id / f"{paper_id}_final.md"
    actual = _count_ai_descriptions(final_md)

    if actual == -1:
        return False, -1, -1

    expected_fnums = _load_figures_with_valid_images(paper_id, raw_dir)
    expected = len(expected_fnums)

    if expected == 0:
        return True, 0, actual

    return actual >= expected, expected, actual


# ── Per-paper guaranteed enrichment ──────────────────────────────────────────

def enrich_until_complete(
    paper_id: str,
    raw_dir: Path,
    output_dir: Path,
    vlm_workers: int = 1,
) -> bool:
    """Enrich one paper, retrying until all figures have AI descriptions.

    Retries indefinitely on VLM API failures (exponential backoff, max 5 min).
    Returns True when complete, False only if OCR source files are missing.
    """
    from ppmatAgent.knowmat.pdf.pipeline_c import enrich_paper_text

    md_path = raw_dir / paper_id / f"{paper_id}.md"
    json_path = raw_dir / paper_id / f"{paper_id}.json"
    if not md_path.exists() or not json_path.exists():
        logger.error("[%s] OCR output missing (%s or %s)", paper_id, md_path, json_path)
        return False

    out_dir = output_dir / paper_id
    final_md_path = out_dir / f"{paper_id}_final.md"

    attempt = 0
    backoff = 30  # doubles each round, capped at 300 s

    while True:
        attempt += 1
        if attempt > 1:
            logger.info("[%s] VLM-retry attempt %d (backoff=%ds)...", paper_id, attempt, backoff)
            time.sleep(backoff)
            backoff = min(backoff * 2, 300)

        try:
            enriched = enrich_paper_text(paper_id, raw_dir, vlm_workers=vlm_workers)
            if not enriched:
                logger.warning("[%s] enrich_paper_text returned None/empty", paper_id)
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            final_md_path.write_text(enriched, encoding="utf-8")
        except Exception as exc:
            logger.error("[%s] Enrichment exception: %s", paper_id, exc)
            continue

        complete, expected, actual = check_completeness(paper_id, raw_dir, output_dir)
        if complete:
            logger.info("[%s] COMPLETE: %d/%d AI descriptions written", paper_id, actual, expected)
            return True

        logger.warning(
            "[%s] Incomplete after attempt %d: %d/%d descriptions — retrying",
            paper_id, attempt, actual, expected,
        )


# ── Phase 2: Repair loop ──────────────────────────────────────────────────────

def run_repair_loop(
    raw_dir: Path,
    output_dir: Path,
    vlm_workers: int = 1,
    max_workers: int = 4,
) -> None:
    """Find papers with incomplete _final.md and repair them concurrently.

    Loops until every paper with OCR output has a complete _final.md.
    """
    logger.info("=" * 60)
    logger.info("Phase 2: Repair loop — scanning for incomplete papers...")

    while True:
        ocr_ready = [
            p.stem for p in sorted(raw_dir.glob("*.pdf"))
            if (raw_dir / p.stem / f"{p.stem}.md").exists()
            and (raw_dir / p.stem / f"{p.stem}.json").exists()
        ]

        incomplete: List[str] = []
        for pid in ocr_ready:
            complete, _, _ = check_completeness(pid, raw_dir, output_dir)
            if not complete:
                incomplete.append(pid)

        if not incomplete:
            logger.info(
                "All %d papers with OCR output have complete _final.md. Done!",
                len(ocr_ready),
            )
            break

        logger.info(
            "Repair: %d/%d papers incomplete — dispatching repair workers...",
            len(incomplete), len(ocr_ready),
        )

        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="repair") as pool:
            future_to_pid = {
                pool.submit(enrich_until_complete, pid, raw_dir, output_dir, vlm_workers): pid
                for pid in incomplete
            }
            for future in as_completed(future_to_pid):
                pid = future_to_pid[future]
                try:
                    ok = future.result()
                    if not ok:
                        logger.error("[%s] OCR source missing — cannot repair", pid)
                except Exception as exc:
                    logger.error("[%s] Repair thread exception: %s", pid, exc)

        time.sleep(15)


# ── Phase 1: Batch OCR + initial enrichment ───────────────────────────────────

def run_phase1(
    raw_dir: Path,
    output_dir: Path,
    vendor: str,
    max_ocr_concurrent: int,
    max_enrich_concurrent: int,
    vlm_workers: int,
    skip_existing: bool,
    db_path: Optional[Path],
    max_retries: int,
    poll_interval: float,
    ocr_timeout: float,
) -> None:
    """Submit all PDFs to OCR API and run CLIP+VLM enrichment as they complete."""
    from ppmatAgent.knowmat.batch.enrich_runner import EnrichRunner

    runner = EnrichRunner(
        input_folder=raw_dir,
        output_dir=output_dir,
        vendor=vendor,
        db_path=db_path,
        max_ocr_concurrent=max_ocr_concurrent,
        max_enrich_concurrent=max_enrich_concurrent,
        vlm_workers=vlm_workers,
        max_retries=max_retries,
        poll_interval=poll_interval,
        ocr_timeout=ocr_timeout,
        skip_existing=skip_existing,
        retry_incomplete=True,
    )

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(runner.run())


# ── Summary report ────────────────────────────────────────────────────────────

def print_summary(raw_dir: Path, output_dir: Path) -> None:
    all_pdfs = sorted(raw_dir.glob("*.pdf"))
    total = len(all_pdfs)
    ocr_done = complete = incomplete = missing_ocr = 0

    for pdf in all_pdfs:
        pid = pdf.stem
        has_ocr = (
            (raw_dir / pid / f"{pid}.md").exists()
            and (raw_dir / pid / f"{pid}.json").exists()
        )
        if not has_ocr:
            missing_ocr += 1
            continue
        ocr_done += 1
        ok, _, _ = check_completeness(pid, raw_dir, output_dir)
        if ok:
            complete += 1
        else:
            incomplete += 1

    print()
    print("=" * 60)
    print("FINAL SUMMARY")
    print(f"  Total PDFs      : {total}")
    print(f"  OCR complete    : {ocr_done}")
    print(f"  _final.md OK    : {complete}")
    print(f"  Incomplete      : {incomplete}")
    print(f"  Missing OCR     : {missing_ocr}")
    print("=" * 60)
