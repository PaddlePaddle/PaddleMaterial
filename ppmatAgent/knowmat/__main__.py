"""
Entry point for running the ppmatAgent.knowmat pipeline via the command line.

Usage
-----
::

    python -m ppmatAgent.knowmat --input-folder path/to/files [--output-dir out] [--max-runs 1]

This will parse all supported files (PDF/TXT/MD) in the given folder, run the
agentic extraction workflow and write the results to the specified output
directory. Each file is processed
sequentially. The final JSON, rationale and intermediate run records are saved
separately for each paper.
"""

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path

from ppmatAgent.knowmat.nodes.paddleocrvl_parse_pdf import parse_pdf_with_paddleocrvl
from ppmatAgent.knowmat.app_config import settings
from ppmatAgent.knowmat.config import ensure_runtime_env

# 进度打印间隔（秒）
_PROGRESS_INTERVAL_SEC = 60

def _ensure_utf8_output() -> None:
    """Best-effort: make stdout/stderr emit UTF-8."""
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

def _run_with_elapsed_progress(description: str, current_file: str, fn, *args, **kwargs):
    """Run fn in a thread and print progress periodically."""
    result_holder: list = []
    exc_holder: list = []

    def target():
        try:
            result_holder.append(fn(*args, **kwargs))
        except Exception as e:
            exc_holder.append(e)

    th = threading.Thread(target=target, daemon=False)
    th.start()
    start = time.time()
    while th.is_alive():
        time.sleep(_PROGRESS_INTERVAL_SEC)
        if not th.is_alive():
            break
        mins = int((time.time() - start) / 60)
        print(f"... {description} 仍在处理中 (当前: {current_file}, 已等待约 {mins} 分钟)")
    th.join()
    if exc_holder:
        raise exc_holder[0]
    return result_holder[0] if result_holder else None


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the ppmatAgent.knowmat extraction pipeline for PDF/TXT/MD inputs.")
    parser.add_argument("--input-folder", default=None, help="Path to the folder containing PDF/TXT files to process.")
    parser.add_argument("--pdf-folder", default=None, help="Legacy alias of --input-folder.")
    parser.add_argument("--output-dir", default=None, help="Directory for extraction results (default: data/output). OCR intermediates stay under input-folder.")
    parser.add_argument("--max-runs", type=int, default=1, help="Maximum extraction/evaluation cycles per paper.")
    parser.add_argument("--workers", type=int, default=1, help="Number of files to process concurrently.")
    parser.add_argument("--ocr-workers", type=int, default=1, help="Number of PDFs to OCR concurrently (GPU is usually best with 1).")
    parser.add_argument("--full-pipeline", action="store_true", help="Enable full multi-stage pipeline.")
    parser.add_argument(
        "--enable-property-standardization",
        action="store_true",
        help="Enable optional property standardization (slower, more LLM calls).",
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Re-run OCR and extraction for all files (ignore existing .md and extraction JSON).",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="Only process files whose stem or full name matches any of the given values.",
    )
    parser.add_argument(
        "--ocr-only",
        action="store_true",
        help="Only run OCR on PDFs (no LLM extraction). Writes <input>/<stem>/<stem>.md and <stem>.json. Run without --ocr-only to extract from these .md into data/output.",
    )

    # Per-agent model overrides
    parser.add_argument("--subfield-model", default=None, help="Model for subfield detection agent.")
    parser.add_argument("--extraction-model", default=None, help="Model for extraction agent.")
    parser.add_argument("--evaluation-model", default=None, help="Model for evaluation agent.")
    parser.add_argument("--manager-model", default=None, help="Model for validation agent (Stage 2: hallucination correction).")
    parser.add_argument("--flagging-model", default=None, help="Model for flagging/quality assessment agent.")
    parser.add_argument("--ocr-log-level", default=None, help="OCR/PaddleX log level (e.g., DEBUG, INFO, WARNING). Overrides PADDLE_PDX_LOG_LEVEL if set.")
    parser.add_argument("--paddleocrvl-version", default=None, help="PaddleOCR-VL version to use: '1.5' (default) or '1.0'.")
    parser.add_argument(
        "--ocr-pages",
        default=None,
        help="1-based pages to OCR, e.g. '1-5,8,10-12'. Default: all pages. Same as env KNOWMAT_OCR_PAGES.",
    )
    parser.add_argument(
        "--skip-cached-ocr",
        action="store_true",
        help="Ignore saved OCR cache under _ocr_cache and re-run inference.",
    )
    parser.add_argument(
        "--mineru-api",
        action="store_true",
        help="Use MinerU cloud API as a compatibility OCR backend. Requires MINERU_API_KEY in .env.",
    )
    parser.add_argument(
        "--paddleocr-api",
        action="store_true",
        help="Use PaddleOCR cloud API for OCR (PaddleOCR-VL-1.5 + PP-StructureV3). Requires PADDLEOCR_API_TOKEN in .env.",
    )
    parser.add_argument(
        "--openai-ocr",
        action="store_true",
        help="Use OpenAI-compatible vision OCR. Requires KNOWMAT_OCR_API_KEY/OPENAI_API_KEY and KNOWMAT_OCR_BASE_URL/OPENAI_BASE_URL.",
    )
    parser.add_argument(
        "--ocr-backend",
        choices=("auto", "paddleocr-api", "paddleocr-local", "openai", "mineru"),
        default="auto",
        help=(
            "OCR backend. auto prefers PaddleOCR API when PADDLEOCR_API_TOKEN is set, "
            "otherwise local PaddleOCR-VL. Use openai for OpenAI-compatible vision OCR. "
            "MinerU is compatibility-only."
        ),
    )
    parser.add_argument(
        "--clear-ocr-cache",
        action="store_true",
        help="Remove all _ocr_cache directories under the input folder before processing.",
    )

    # Batch parallel mode arguments
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Enable batch parallel mode: async OCR submission + persistent state + crash recovery. "
             "Requires PaddleOCR API or the compatibility MinerU API.",
    )
    parser.add_argument(
        "--max-ocr-concurrent",
        type=int,
        default=20,
        help="(Batch/Final-MD mode) Max concurrent OCR API submissions in flight.",
    )
    parser.add_argument(
        "--max-llm-concurrent",
        type=int,
        default=4,
        help="(Batch mode) Max concurrent LLM extraction threads.",
    )
    parser.add_argument(
        "--batch-db",
        default=None,
        help="(Batch mode) Path to SQLite state database. Default: <input-folder>/.knowmat_batch.db",
    )
    parser.add_argument(
        "--ocr-poll-interval",
        type=float,
        default=10.0,
        help="(Batch/Final-MD mode) Seconds between OCR job poll cycles.",
    )

    # Final-MD mode arguments (CLIP+VLM enrichment → _final.md, no LLM extraction)
    parser.add_argument(
        "--final-md",
        action="store_true",
        help="Final-MD mode: batch OCR + CLIP+VLM enrichment → _final.md per paper. "
             "No LLM extraction. Requires PaddleOCR API or the compatibility MinerU API "
             "(unless --repair-only is set).",
    )
    parser.add_argument(
        "--max-enrich-concurrent",
        type=int,
        default=2,
        help="(Final-MD mode) Max concurrent CLIP+VLM enrichment workers. "
             "Keep low (2) to avoid OOM — CLIP model is loaded once and shared.",
    )
    parser.add_argument(
        "--vlm-workers",
        type=int,
        default=4,
        help="(Final-MD mode) VLM API concurrency per paper (default: 4). "
             "Reduce if you see 403 rate-limit errors.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="(Final-MD mode) Skip papers that already have a complete _final.md.",
    )
    parser.add_argument(
        "--repair-only",
        action="store_true",
        help="(Final-MD mode) Skip Phase 1 (OCR), only run Phase 2 repair loop "
             "on papers with existing OCR output whose _final.md is incomplete.",
    )

    args = parser.parse_args(argv)
    if args.ocr_log_level:
        os.environ["PADDLE_PDX_LOG_LEVEL"] = args.ocr_log_level
    if args.paddleocrvl_version:
        os.environ["PADDLEOCRVL_VERSION"] = args.paddleocrvl_version

    selected_backend = args.ocr_backend
    if args.paddleocr_api:
        selected_backend = "paddleocr-api"
    elif args.openai_ocr:
        selected_backend = "openai"
    elif args.mineru_api:
        selected_backend = "mineru"
    elif selected_backend == "auto":
        selected_backend = (
            "paddleocr-api"
            if os.getenv("PADDLEOCR_API_TOKEN", "").strip()
            else "paddleocr-local"
        )

    args.paddleocr_api = selected_backend == "paddleocr-api"
    args.openai_ocr = selected_backend == "openai"
    args.mineru_api = selected_backend == "mineru"
    os.environ["KNOWMAT_OCR_BACKEND"] = selected_backend

    if args.mineru_api:
        if not os.getenv("MINERU_API_KEY", "").strip():
            print("Error: MinerU compatibility backend requires MINERU_API_KEY to be set in .env or environment.")
            return
        os.environ["KNOWMAT_USE_MINERU_API"] = "1"
    if args.paddleocr_api:
        if not os.getenv("PADDLEOCR_API_TOKEN", "").strip():
            print("Error: PaddleOCR API backend requires PADDLEOCR_API_TOKEN to be set in .env or environment.")
            return
        os.environ["KNOWMAT_USE_PADDLEOCR_API"] = "1"
    if args.openai_ocr:
        if not (
            os.getenv("KNOWMAT_OCR_API_KEY", "").strip()
            or os.getenv("OPENAI_API_KEY", "").strip()
            or os.getenv("LLM_API_KEY", "").strip()
            or os.getenv("AI_STUDIO_API_KEY", "").strip()
            or os.getenv("AISTUDIO_ACCESS_TOKEN", "").strip()
        ):
            print(
                "Error: OpenAI-compatible OCR requires KNOWMAT_OCR_API_KEY, "
                "OPENAI_API_KEY, LLM_API_KEY, or AI_STUDIO_API_KEY."
            )
            return
        os.environ["KNOWMAT_USE_OPENAI_OCR"] = "1"
    _ensure_utf8_output()
    
    # 优先级：CLI (--input-folder / --pdf-folder) > 环境变量 KNOWMAT2_INPUT_DIR > 默认 "data/raw"
    input_folder_arg = args.input_folder or args.pdf_folder or settings.input_dir
    input_folder = Path(input_folder_arg)
    if not input_folder.exists():
        # 若目录不存在，则创建空目录并提示用户
        print(f"Input folder not found, creating: {input_folder}")
        input_folder.mkdir(parents=True, exist_ok=True)
    
    if not input_folder.is_dir():
        print(f"Error: Path is not a directory: {input_folder}")
        return

    if args.clear_ocr_cache:
        from ppmatAgent.knowmat.pdf.ocr_cache import clear_all_ocr_caches_under

        cleared = clear_all_ocr_caches_under(input_folder)
        print(f"Cleared {cleared} _ocr_cache director(y/ies) under {input_folder}.")

    # OCR 中间产物（.md / .json）始终在 input_folder 下按论文子目录存放（如 data/raw/论文A/论文A.md）
    # 抽取结果（extraction JSON、报告等）写入 output_dir，默认 data/output，与 raw 分离
    extraction_output_dir = args.output_dir if args.output_dir else settings.output_dir
    print(f"Input (raw + OCR intermediates): {input_folder}")
    print(f"Extraction output:               {extraction_output_dir}")

    requires_llm = not args.ocr_only and not args.final_md
    if requires_llm:
        try:
            ensure_runtime_env(require_llm=True)
        except RuntimeError as exc:
            print(f"Error: {exc}")
            return

    # --- Batch parallel mode ---
    if args.batch:
        if not (args.paddleocr_api or args.mineru_api):
            print("Error: --batch requires PaddleOCR API or the compatibility MinerU API.")
            if args.openai_ocr:
                print("       OpenAI-compatible OCR is synchronous and is not supported by --batch yet.")
            return

        import asyncio
        from ppmatAgent.knowmat.batch.batch_runner import BatchRunner

        vendor = "paddleocr" if args.paddleocr_api else "mineru"
        db_path = Path(args.batch_db) if args.batch_db else None

        runner = BatchRunner(
            input_folder=input_folder,
            output_dir=Path(extraction_output_dir),
            vendor=vendor,
            db_path=db_path,
            max_ocr_concurrent=args.max_ocr_concurrent,
            max_llm_concurrent=args.max_llm_concurrent,
            max_retries=3,
            poll_interval=args.ocr_poll_interval,
            ocr_only=args.ocr_only,
            max_runs=args.max_runs,
            full_pipeline=args.full_pipeline,
            enable_property_standardization=args.enable_property_standardization,
            subfield_model=getattr(args, "subfield_model", None),
            extraction_model=getattr(args, "extraction_model", None),
            evaluation_model=getattr(args, "evaluation_model", None),
            manager_model=getattr(args, "manager_model", None),
            flagging_model=getattr(args, "flagging_model", None),
        )

        asyncio.run(runner.run())
        return

    # --- Final-MD mode (CLIP+VLM enrichment → _final.md, no LLM extraction) ---
    if args.final_md:
        if not args.repair_only and not (args.paddleocr_api or args.mineru_api):
            print(
                "Error: --final-md requires PaddleOCR API or the compatibility MinerU API. "
                "Use --repair-only to skip OCR and only repair existing incomplete _final.md files. "
                "OpenAI-compatible OCR is supported in normal --ocr-only / extraction mode, not --final-md."
            )
            return

        import logging as _logging
        _logging.basicConfig(
            level=_logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )

        from ppmatAgent.knowmat.batch.finalmd_pipeline import run_phase1, run_repair_loop, print_summary

        output_dir = Path(extraction_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        vendor = "paddleocr" if args.paddleocr_api else "mineru"
        db_path = Path(args.batch_db) if args.batch_db else None

        if not args.repair_only:
            print(f"[FINAL-MD] Phase 1: Batch OCR + CLIP+VLM enrichment (vendor={vendor})...")
            run_phase1(
                raw_dir=input_folder,
                output_dir=output_dir,
                vendor=vendor,
                max_ocr_concurrent=args.max_ocr_concurrent,
                max_enrich_concurrent=args.max_enrich_concurrent,
                vlm_workers=args.vlm_workers,
                skip_existing=args.skip_existing,
                db_path=db_path,
                max_retries=3,
                poll_interval=args.ocr_poll_interval,
                ocr_timeout=600.0,
            )
            print("[FINAL-MD] Phase 1 complete.")

        print("[FINAL-MD] Phase 2: Repair loop (validate completeness + retry VLM failures)...")
        run_repair_loop(
            raw_dir=input_folder,
            output_dir=output_dir,
            vlm_workers=args.vlm_workers,
            max_workers=args.max_enrich_concurrent,
        )

        print_summary(input_folder, output_dir)
        return

    pdf_files = sorted(
        [p for p in input_folder.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"],
        key=lambda x: x.name.lower(),
    )
    def _is_text_candidate(path: Path) -> bool:
        if not path.is_file() or path.suffix.lower() not in (".txt", ".md"):
            return False
        rel_parts = path.relative_to(input_folder).parts
        skip_dirs = {"_ocr_cache", "_bad_txt", "processed"}
        if any(part in skip_dirs for part in rel_parts):
            return False
        return True

    text_files = sorted(
        [p for p in input_folder.rglob("*") if _is_text_candidate(p)],
        key=lambda x: x.name.lower(),
    )
    text_by_stem = {}
    for p in text_files:
        stem = p.stem
        if stem in text_by_stem and text_by_stem[stem].suffix.lower() == ".md":
            continue
        if p.suffix.lower() == ".md":
            text_by_stem[stem] = p
        elif stem not in text_by_stem:
            text_by_stem[stem] = p

    existing_txt_files = sorted(text_by_stem.values(), key=lambda x: x.name.lower())

    if not pdf_files and not existing_txt_files:
        print(f"Error: No supported files (.pdf/.txt/.md) found in: {input_folder}")
        print("Please place your PDF/TXT/MD files into this folder or specify --input-folder explicitly.")
        return

    if args.only:
        requested = set(args.only)
        before_pdf = len(pdf_files)
        before_txt = len(existing_txt_files)
        pdf_files = [p for p in pdf_files if p.stem in requested or p.name in requested]
        existing_txt_files = [p for p in existing_txt_files if p.stem in requested or p.name in requested]
        if not pdf_files and not existing_txt_files:
            print(f"Error: No files matched --only filter in: {input_folder}")
            print(f"Requested: {', '.join(sorted(requested))}")
            return
        else:
            print(f"\nFiltered files with --only (pdf: {before_pdf} -> {len(pdf_files)}, txt: {before_txt} -> {len(existing_txt_files)})")

    txt_by_stem = {p.stem: p for p in existing_txt_files}
    # 强制重跑时对所有 PDF 重新 OCR；否则仅对尚无 .md/.txt 的 PDF 做 OCR
    pdfs_missing_txt = list(pdf_files) if args.force_rerun else [p for p in pdf_files if p.stem not in txt_by_stem]

    # Guard: local OCR requires NVIDIA GPU — reject CPU-only environments early
    if pdfs_missing_txt and not (args.paddleocr_api or args.mineru_api or args.openai_ocr):
        try:
            import paddle  # type: ignore
            if not paddle.device.is_compiled_with_cuda():
                print(
                    "Error: Local OCR requires an NVIDIA GPU (paddlepaddle-gpu with CUDA). "
                    "CPU-only local OCR is no longer supported.\n"
                    "Solutions:\n"
                    "  1. Use --paddleocr-api for cloud OCR (recommended, no GPU needed)\n"
                    "  2. Use --ocr-backend openai for OpenAI-compatible vision OCR\n"
                    "  3. Use --ocr-backend mineru for the compatibility MinerU cloud OCR\n"
                    "  4. Install paddlepaddle-gpu on a machine with NVIDIA GPU"
                )
                return
        except ImportError:
            print(
                "Error: PaddlePaddle is not installed. Local OCR is unavailable.\n"
                "Solutions:\n"
                "  1. Use --paddleocr-api for cloud OCR (recommended, no GPU needed)\n"
                "  2. Use --ocr-backend openai for OpenAI-compatible vision OCR\n"
                "  3. Use --ocr-backend mineru for the compatibility MinerU cloud OCR\n"
                "  4. Install paddlepaddle-gpu: pip install -r requirements-gpu.txt"
            )
            return

    if not existing_txt_files and not pdfs_missing_txt:
        print(f"Error: No text/markdown files available for processing in: {input_folder}")
        return

    if args.ocr_only:
        if not pdfs_missing_txt:
            print("No PDFs need OCR (all have corresponding .md/.txt). Nothing to do.")
            return
        print(f"\n[OCR-only] Queued {len(pdfs_missing_txt)} PDFs for OCR (no LLM extraction).")
    else:
        if existing_txt_files:
            print(f"\nQueued {len(existing_txt_files)} existing text files for extraction")
        if pdfs_missing_txt:
            print(f"Queued {len(pdfs_missing_txt)} PDFs for OCR")

    # 计算当前运行实际使用的输出根目录（CLI 参数优先于配置默认值）

    def _ocr_pdf_to_md(pdf: Path) -> Path | None:
        stem = pdf.stem
        print(f"\nNo MD found for {pdf.name}. Running OCR to create {stem}.md ...")
        paper_dir = input_folder / stem
        paper_dir.mkdir(parents=True, exist_ok=True)
        parse_output_dir = paper_dir
        parse_output_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "pdf_path": str(pdf),
            "output_dir": str(parse_output_dir),
            "save_intermediate": False,
            "ocr_pages": args.ocr_pages,
            "ocr_skip_cached": bool(args.skip_cached_ocr),
        }
        try:
            result = _run_with_elapsed_progress("OCR", pdf.name, parse_pdf_with_paddleocrvl, state)
        except Exception as exc:
            print(f"[ERROR] Failed to OCR {pdf.name}: {exc}")
            return None
        text = result.get("paper_text") if isinstance(result, dict) else None
        if not text:
            print(f"[ERROR] OCR returned no text for {pdf.name}, skipping.")
            return None
        md_path = paper_dir / f"{stem}.md"
        md_path.write_text(text, encoding="utf-8")
        print(f"Saved OCR markdown to: {md_path}")
        ocr_items = result.get("ocr_items") if isinstance(result, dict) else None
        if ocr_items is None:
            ocr_items = []
        json_path = paper_dir / f"{stem}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(ocr_items, f, ensure_ascii=False, indent=2)
        print(f"Saved OCR structured output to: {json_path}")
        pages = result.get("metadata", {}).get("pages") if isinstance(result, dict) else None
        if pages is not None:
            print(f"[OCR] 完成; {pdf.name}: {pages} 页")
        return md_path

    def _ensure_md(text_path: Path) -> Path | None:
        stem = text_path.stem
        paper_dir = input_folder / stem
        paper_dir.mkdir(parents=True, exist_ok=True)
        md_path = paper_dir / f"{stem}.md"
        if text_path.suffix.lower() == ".md" and text_path.resolve() == md_path.resolve():
            return md_path
        # Convert txt/md into cleaned markdown via parser
        state = {
            "pdf_path": str(text_path),
            "output_dir": str(paper_dir),
            "save_intermediate": False,
            "ocr_pages": None,
            "ocr_skip_cached": False,
        }
        try:
            result = _run_with_elapsed_progress("TXT->MD", text_path.name, parse_pdf_with_paddleocrvl, state)
        except Exception as exc:
            print(f"[ERROR] Failed to normalize {text_path.name}: {exc}")
            return None
        text = result.get("paper_text") if isinstance(result, dict) else None
        if not text:
            print(f"[ERROR] Normalization returned no text for {text_path.name}, skipping.")
            return None
        md_path.write_text(text, encoding="utf-8")
        print(f"Saved normalized markdown to: {md_path}")
        return md_path

    def _log_summary(summary: dict) -> None:
        if summary.get("success"):
            if summary.get("skipped"):
                print(f"\nSkipped (already processed): {summary.get('file')}")
                print(f"   Output: {summary.get('output_dir')}")
                print(f"   Materials: {summary.get('compositions', 0)}")
            else:
                flag_str = "[FLAGGED]" if summary.get("flag") else "[OK]"
                print(f"\nFinished extraction: {summary.get('file')}")
                print(f"   Status: {flag_str}")
                print(f"   Output: {summary.get('output_dir')}")
                print(f"   Materials: {summary.get('compositions', 0)}")
        else:
            err = summary.get("error", "")
            print(f"\nError processing {summary.get('file')}: {err}")
            if "401" in err and ("invalid_model" in err or "does not exist" in err.lower()):
                print("   → 请检查 .env：LLM_MODEL 需为千帆控制台创建的端点 ID（如 ep_xxxxx），且当前账号有该模型/端点访问权限。")

    root_output_dir = Path(extraction_output_dir)

    def _process_one(file_path: Path) -> dict:
        try:
            base_name = file_path.stem
            paper_output_dir = root_output_dir / base_name
            extraction_path = paper_output_dir / f"{base_name}_extraction.json"
            if extraction_path.exists() and not args.force_rerun:
                try:
                    data = json.loads(extraction_path.read_text(encoding="utf-8"))
                    items = data.get("items") or data.get("Materials", [])
                    compositions_count = len(items)
                except Exception:
                    compositions_count = 0
                print(f"Skipping {file_path.name}: existing extraction found at {extraction_path}")
                return {
                    "file": file_path.name,
                    "success": True,
                    "flag": False,
                    "compositions": compositions_count,
                    "output_dir": str(paper_output_dir),
                    "skipped": True,
                }

            from ppmatAgent.knowmat.orchestrator import run as run_extraction

            result = _run_with_elapsed_progress(
                "LLM",
                file_path.name,
                run_extraction,
                pdf_path=str(file_path),
                output_dir=extraction_output_dir,
                model_name=None,  # Use defaults from settings
                max_runs=args.max_runs,
                subfield_model=args.subfield_model,
                extraction_model=args.extraction_model,
                evaluation_model=args.evaluation_model,
                manager_model=args.manager_model,
                flagging_model=args.flagging_model,
                full_pipeline=args.full_pipeline,
                enable_property_standardization=args.enable_property_standardization,
            )

            items = result.get("final_data", {}).get("items") or result.get("final_data", {}).get("Materials", [])
            compositions_count = len(items)
            return {
                "file": file_path.name,
                "success": True,
                "flag": result.get("flag"),
                "compositions": compositions_count,
                "output_dir": result.get("output_dir"),
                "skipped": False,
            }
        except Exception as e:
            return {"file": file_path.name, "success": False, "error": str(e)}

    results_summary = []
    workers = max(1, args.workers)
    ocr_workers = max(1, args.ocr_workers)
    
    # Warn if ocr_workers > 1 on low-VRAM GPUs (< 16 GB), but allow it on high-VRAM cards.
    _GPU_SAFE_VRAM_GB = 16.0
    try:
        from ppmatAgent.knowmat.pdf.ocr_engine import get_gpu_memory_info
        used_gb, total_gb = get_gpu_memory_info()
        if total_gb > 0:
            print(f"GPU Memory: {used_gb:.1f}/{total_gb:.1f} GB used")
            if ocr_workers > 1 and total_gb < _GPU_SAFE_VRAM_GB:
                print(
                    f"\nWarning: GPU detected ({total_gb:.1f} GB < {_GPU_SAFE_VRAM_GB:.0f} GB threshold). "
                    f"Forcing --ocr-workers to 1 to prevent OOM."
                )
                ocr_workers = 1
    except Exception:
        pass
    
    # Also check for LLM workers - multiple concurrent LLM calls can overwhelm API limits
    if workers > 1:
        print(f"\nNote: --workers={workers} will run {workers} LLM extractions concurrently.")
        print("      If you encounter rate limits, consider reducing to --workers 1.")

    with ThreadPoolExecutor(max_workers=workers) as llm_pool, ThreadPoolExecutor(max_workers=ocr_workers) as ocr_pool:
        llm_futures = {}
        ocr_futures = {}

        def _submit_llm(path: Path) -> None:
            fut = llm_pool.submit(_process_one, path)
            llm_futures[fut] = path

        # Existing .md/.txt files are always valid extraction inputs. --force-rerun
        # only disables the extraction-output skip inside _process_one.
        if not args.ocr_only:
            for text_path in sorted(existing_txt_files, key=lambda x: x.name.lower()):
                md_path = _ensure_md(text_path)
                if md_path:
                    _submit_llm(md_path)
                else:
                    results_summary.append({"file": text_path.name, "success": False, "error": "Normalization failed"})

        # Start OCR for PDFs missing TXT
        for pdf in pdfs_missing_txt:
            ocr_futures[ocr_pool.submit(_ocr_pdf_to_md, pdf)] = pdf

        if not llm_futures and not ocr_futures:
            print(f"Error: No work to process in: {input_folder}")
            return

        while ocr_futures or llm_futures:
            done, _ = wait(set(ocr_futures.keys()) | set(llm_futures.keys()), return_when=FIRST_COMPLETED)
            for fut in done:
                if fut in ocr_futures:
                    pdf = ocr_futures.pop(fut)
                    try:
                        txt_path = fut.result()
                    except Exception as exc:
                        print(f"[ERROR] OCR failed for {pdf.name}: {exc}")
                        results_summary.append({"file": pdf.name, "success": False, "error": f"OCR failed: {exc}"})
                        continue
                    if args.ocr_only:
                        results_summary.append({"file": pdf.name, "success": txt_path is not None})
                    elif txt_path:
                        _submit_llm(txt_path)
                    else:
                        results_summary.append({"file": pdf.name, "success": False, "error": "OCR returned no result"})
                else:
                    path = llm_futures.pop(fut)
                    try:
                        summary = fut.result()
                    except Exception as exc:
                        summary = {"file": path.name, "success": False, "error": str(exc)}
                    results_summary.append(summary)
                    _log_summary(summary)
    # Print final summary
    print(f"\n{'='*60}")
    if args.ocr_only:
        print("OCR-ONLY SUMMARY")
    else:
        print("PROCESSING SUMMARY")
    print(f"{'='*60}\n")

    successful = sum(1 for r in results_summary if r["success"])
    failed = len(results_summary) - successful

    print(f"Total files: {len(results_summary)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")

    if not args.ocr_only and successful > 0:
        flagged = sum(1 for r in results_summary if r["success"] and r.get("flag"))
        print(f"Flagged for review: {flagged}")
        total_compositions = sum(r.get("compositions", 0) for r in results_summary if r["success"])
        print(f"Total materials: {total_compositions}")

    if args.ocr_only:
        print("\nOCR output: each PDF -> <input-folder>/<stem>/<stem>.md and <stem>.json")
        print("Run again without --ocr-only to run LLM extraction on these .md files.")
    print(f"\n{'='*60}\n")

    # Print individual results
    for r in results_summary:
        if r["success"]:
            if args.ocr_only:
                print(f"[OK] {r['file']}: OCR done -> <stem>.md + <stem>.json")
            elif r.get("skipped"):
                print(f"[SKIPPED] {r['file']}: {r.get('compositions', 0)} materials (already processed)")
            else:
                flag_icon = "[FLAGGED]" if r.get("flag") else "[OK]"
                print(f"{flag_icon} {r['file']}: {r.get('compositions', 0)} materials")
        else:
            print(f"[ERROR] {r['file']}: {r.get('error', 'Unknown error')}")


if __name__ == "__main__":  # pragma: no cover
    main()
