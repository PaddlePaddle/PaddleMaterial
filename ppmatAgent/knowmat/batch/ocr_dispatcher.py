"""Async OCR job dispatcher: submit, poll, download, convert to .md."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ppmatAgent.knowmat.batch.key_pool import KeyPool
from ppmatAgent.knowmat.batch.models import KeyInfo, TaskRecord, TaskStatus
from ppmatAgent.knowmat.batch.task_store import TaskStore

logger = logging.getLogger(__name__)


class OCRDispatcher:
    """Handles async OCR job lifecycle: submit, poll, download.

    Uses existing sync PaddleOCRAPIClient/MineruPrecisionClient via
    run_in_executor to avoid blocking the event loop.
    """

    def __init__(
        self,
        store: TaskStore,
        key_pool: KeyPool,
        vendor: str,
        input_folder: Path,
        max_concurrent_submit: int = 20,
        poll_interval: float = 10.0,
        ocr_timeout: float = 600.0,
    ):
        self._store = store
        self._key_pool = key_pool
        self._vendor = vendor
        self._input_folder = input_folder
        self._submit_sem = asyncio.Semaphore(max_concurrent_submit)
        self._poll_interval = poll_interval
        self._ocr_timeout = ocr_timeout

    # ------------------------------------------------------------------
    # Submit
    # ------------------------------------------------------------------

    async def submit_one(self, task: TaskRecord, shutdown_event: asyncio.Event) -> bool:
        """Submit a single PDF to OCR API. Returns True on success."""
        if shutdown_event.is_set():
            return False

        async with self._submit_sem:
            key = await self._key_pool.acquire()
            loop = asyncio.get_event_loop()
            try:
                if self._vendor == "paddleocr":
                    job_id = await loop.run_in_executor(
                        None, self._submit_paddleocr, task.pdf_path, key
                    )
                else:
                    job_id = await loop.run_in_executor(
                        None, self._submit_mineru, task.pdf_path, key
                    )

                self._store.update_status(
                    task.task_id,
                    TaskStatus.OCR_SUBMITTED,
                    ocr_job_id=job_id,
                    ocr_vendor=self._vendor,
                    api_key_id=key.key_id,
                )
                self._key_pool.release(key.key_id, success=True)
                logger.info("Submitted OCR job %s for %s", job_id, task.task_id)
                return True

            except Exception as exc:
                error_msg = str(exc)
                if _is_rate_limit_error(exc):
                    retry_after = _parse_retry_after(exc)
                    self._key_pool.report_rate_limit(key.key_id, retry_after)
                    logger.warning(
                        "Rate limited on key %s for %s, will retry",
                        key.key_id, task.task_id,
                    )
                    # Leave as PENDING so it gets picked up again
                    return False
                elif _is_transient_error(exc):
                    self._key_pool.release(key.key_id, success=False)
                    logger.warning(
                        "Transient error for %s on key %s, will retry: %s",
                        task.task_id, key.key_id, error_msg[:200],
                    )
                    # Leave as PENDING for retry
                    return False
                else:
                    self._key_pool.release(key.key_id, success=False)
                    self._store.mark_failed(task.task_id, f"OCR submit error: {error_msg}")
                    logger.error("OCR submit failed for %s: %s", task.task_id, error_msg)
                    return False

    def _submit_paddleocr(self, pdf_path: str, key: KeyInfo) -> str:
        """Sync: submit job to PaddleOCR API. Returns job_id.

        VL model is resolved from env (PADDLEOCR_API_MODEL / PADDLEOCRVL_VERSION,
        default PaddleOCR-VL-1.5) inside the client.
        """
        from ppmatAgent.knowmat.pdf.paddleocr_api_client import PaddleOCRAPIClient

        client = PaddleOCRAPIClient(key.token, key.base_url)
        return client.submit_job(Path(pdf_path))

    def _submit_mineru(self, pdf_path: str, key: KeyInfo) -> str:
        """Sync: submit job to MinerU API. Returns batch_id."""
        from ppmatAgent.knowmat.pdf.mineru_api_client import MineruPrecisionClient

        client = MineruPrecisionClient(key.token)
        batch_id, _ = client.submit_file(Path(pdf_path))
        return batch_id

    # ------------------------------------------------------------------
    # Poll
    # ------------------------------------------------------------------

    async def poll_all_submitted(
        self,
        completion_queue: asyncio.Queue,
        shutdown_event: asyncio.Event,
    ) -> None:
        """Periodically poll all OCR_SUBMITTED tasks. Push completed ones to queue."""
        while not shutdown_event.is_set():
            submitted_tasks = self._store.get_all_submitted()
            if not submitted_tasks:
                await asyncio.sleep(self._poll_interval)
                continue

            # Poll in batches concurrently (up to 50 at a time)
            poll_sem = asyncio.Semaphore(50)
            poll_tasks = []
            for task in submitted_tasks:
                coro = self._poll_one_guarded(task, poll_sem, completion_queue)
                poll_tasks.append(asyncio.create_task(coro))

            if poll_tasks:
                await asyncio.gather(*poll_tasks, return_exceptions=True)

            await asyncio.sleep(self._poll_interval)

    async def _poll_one_guarded(
        self,
        task: TaskRecord,
        sem: asyncio.Semaphore,
        completion_queue: asyncio.Queue,
    ) -> None:
        """Poll a single job with semaphore guard."""
        async with sem:
            try:
                result = await self._poll_one(task)
                if result == "done":
                    # Download and convert
                    md_path = await self._download_and_convert(task)
                    if md_path:
                        self._store.update_status(
                            task.task_id, TaskStatus.OCR_DONE, md_path=str(md_path)
                        )
                        task.md_path = str(md_path)
                        task.status = TaskStatus.OCR_DONE
                        await completion_queue.put(task)
                        logger.info("OCR done for %s → %s", task.task_id, md_path.name)
                    else:
                        self._store.mark_failed(task.task_id, "OCR download/convert failed")
                elif result == "failed":
                    pass  # Already marked failed in _poll_one
                # "running" → do nothing, poll again next cycle
            except Exception as exc:
                logger.error("Poll error for %s: %s", task.task_id, exc)

    async def _poll_one(self, task: TaskRecord) -> str:
        """Poll a single OCR job. Returns 'done', 'running', or 'failed'."""
        loop = asyncio.get_event_loop()
        try:
            if self._vendor == "paddleocr":
                data = await loop.run_in_executor(
                    None, self._poll_paddleocr, task.ocr_job_id, task.api_key_id
                )
                state = data.get("state", "")
                if state == "done":
                    # Store result URL in task for download phase
                    task._poll_result = data
                    return "done"
                elif state == "failed":
                    error_msg = data.get("errorMsg", "unknown")
                    self._store.mark_failed(task.task_id, f"OCR job failed: {error_msg}")
                    return "failed"
                return "running"
            else:
                status, result_url = await loop.run_in_executor(
                    None, self._poll_mineru, task.ocr_job_id, task.api_key_id
                )
                if status == "done":
                    task._poll_result = result_url
                    return "done"
                elif status == "failed":
                    self._store.mark_failed(task.task_id, "MinerU job failed")
                    return "failed"
                return "running"
        except Exception as exc:
            if _is_not_found_error(exc):
                self._store.mark_failed(task.task_id, f"OCR job not found: {task.ocr_job_id}")
                return "failed"
            logger.debug("Poll transient error for %s: %s", task.task_id, exc)
            return "running"

    def _poll_paddleocr(self, job_id: str, key_id: Optional[str]) -> Dict[str, Any]:
        """Sync: single poll request to PaddleOCR API."""
        from ppmatAgent.knowmat.pdf.paddleocr_api_client import PaddleOCRAPIClient
        import requests

        key = self._get_key_by_id(key_id)
        url = f"{key.base_url.rstrip('/')}/{job_id}"
        headers = {"Authorization": f"bearer {key.token}"}
        resp = requests.get(url, headers=headers, timeout=30)
        if resp.status_code != 200:
            raise RuntimeError(f"Poll HTTP {resp.status_code}: {resp.text[:200]}")
        return resp.json().get("data", {})

    def _poll_mineru(self, batch_id: str, key_id: Optional[str]) -> Tuple[str, Optional[str]]:
        """Sync: poll MinerU batch status. Returns (status, result_url_or_None)."""
        from ppmatAgent.knowmat.pdf.mineru_api_client import MineruPrecisionClient

        key = self._get_key_by_id(key_id)
        client = MineruPrecisionClient(key.token)
        status, results = client.poll_batch(batch_id)
        if status == "done" and results:
            return "done", results[0].get("full_zip_url")
        elif status == "failed":
            return "failed", None
        return "running", None

    # ------------------------------------------------------------------
    # Download & Convert
    # ------------------------------------------------------------------

    async def _download_and_convert(self, task: TaskRecord) -> Optional[Path]:
        """Download OCR result and convert to .md + .json."""
        loop = asyncio.get_event_loop()
        try:
            if self._vendor == "paddleocr":
                return await loop.run_in_executor(
                    None, self._download_paddleocr, task
                )
            else:
                return await loop.run_in_executor(
                    None, self._download_mineru, task
                )
        except Exception as exc:
            logger.error("Download/convert failed for %s: %s", task.task_id, exc)
            return None

    def _download_paddleocr(self, task: TaskRecord) -> Optional[Path]:
        """Sync: download JSONL result + PP-StructureV3 refinement → .md + .json"""
        from ppmatAgent.knowmat.pdf.paddleocr_api_client import PaddleOCRAPIClient
        from ppmatAgent.knowmat.pdf.paddleocr_api_result_converter import convert_paddleocr_api_to_knowmat

        poll_data = getattr(task, "_poll_result", None)
        if not poll_data:
            return None

        jsonl_url = poll_data.get("resultUrl", {}).get("jsonUrl", "")
        if not jsonl_url:
            return None

        key = self._get_key_by_id(task.api_key_id)
        client = PaddleOCRAPIClient(key.token, key.base_url)

        # Download primary OCR result
        pages_data = client.download_jsonl(jsonl_url)

        # Set up output paths
        stem = Path(task.pdf_path).stem
        paper_dir = self._input_folder / stem
        paper_dir.mkdir(parents=True, exist_ok=True)
        images_dir = paper_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        # Convert to knowmat format
        extracted_text, metadata, ocr_items = convert_paddleocr_api_to_knowmat(
            pages_data, task.pdf_path, images_dir
        )

        # PP-StructureV3 refinement (reuses same key)
        ocr_items, pp_report = self._refine_ppstructurev3(task.pdf_path, ocr_items, str(paper_dir), key)
        metadata.setdefault("ocr_quality", {}).update(pp_report)

        # Apply post-processing from the existing pipeline
        from ppmatAgent.knowmat.pdf.paddleocr_api_result_converter import clean_api_markdown
        from ppmatAgent.knowmat.pdf.section_normalizer import (
            normalize_alloy_strings,
            normalize_leading_masthead_and_title,
            normalize_plain_author_superscripts,
            repair_keywords_abstract_two_column_ocr,
            structure_sections,
        )
        from ppmatAgent.knowmat.pdf.formula_formatter import format_formula_text
        from ppmatAgent.knowmat.pdf.doi_extractor import extract_first_doi, extract_first_doi_from_ocr_items

        md_text = extracted_text
        md_text = normalize_leading_masthead_and_title(md_text)
        md_text = structure_sections(md_text)
        md_text = repair_keywords_abstract_two_column_ocr(md_text)
        md_text = normalize_plain_author_superscripts(md_text)
        md_text = normalize_alloy_strings(md_text)
        md_text = format_formula_text(md_text)
        md_text = clean_api_markdown(md_text)

        # DOI extraction
        doi = extract_first_doi_from_ocr_items(ocr_items) or extract_first_doi(md_text[:5000])
        if doi and doi not in md_text:
            md_text = f"DOI: {doi}\n\n{md_text}"

        # Save .md and .json
        md_path = paper_dir / f"{stem}.md"
        md_path.write_text(md_text, encoding="utf-8")

        json_path = paper_dir / f"{stem}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(ocr_items, f, ensure_ascii=False, indent=2)

        logger.info("Saved OCR output: %s (%d items)", md_path.name, len(ocr_items))
        return md_path

    def _download_mineru(self, task: TaskRecord) -> Optional[Path]:
        """Sync: download MinerU result ZIP → .md + .json

        Output structure matches PaddleOCR API batch:
          data/raw/<stem>/images/    (figures)
          data/raw/<stem>/<stem>.md  (final markdown)
          data/raw/<stem>/<stem>.json (ocr_items)
        """
        import io
        import shutil
        import tempfile
        import zipfile

        import requests

        from ppmatAgent.knowmat.pdf.mineru_api_client import MineruPrecisionClient
        from ppmatAgent.knowmat.pdf.mineru_result_converter import convert_mineru_to_knowmat

        result_url = getattr(task, "_poll_result", None)
        if not result_url:
            return None

        key = self._get_key_by_id(task.api_key_id)
        client = MineruPrecisionClient(key.token)

        stem = Path(task.pdf_path).stem
        paper_dir = self._input_folder / stem
        paper_dir.mkdir(parents=True, exist_ok=True)
        images_dir = paper_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        # Download and extract ZIP to a temp directory
        resp = requests.get(result_url, timeout=120)
        resp.raise_for_status()

        tmp_dir = Path(tempfile.mkdtemp())
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            zf.extractall(str(tmp_dir))

        # Find content_list.json (MinerU names it <uuid>_content_list.json)
        cl_candidates = sorted(tmp_dir.glob("*_content_list.json"))
        cl_candidates = [f for f in cl_candidates if "_content_list_v2" not in f.name]
        if not cl_candidates:
            cl_candidates = list(tmp_dir.rglob("*content_list.json"))
            cl_candidates = [f for f in cl_candidates if "_content_list_v2" not in f.name]

        content_list = []
        if cl_candidates:
            content_list = json.loads(cl_candidates[0].read_text("utf-8"))

        # Find full.md
        full_md = ""
        md_file = tmp_dir / "full.md"
        if md_file.is_file():
            full_md = md_file.read_text("utf-8")
        else:
            for md in tmp_dir.rglob("*.md"):
                full_md = md.read_text("utf-8")
                break

        if not content_list and not full_md:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            return None

        # Convert using the standard converter (images go to paper_dir/images/)
        extracted_text, metadata, ocr_items = convert_mineru_to_knowmat(
            content_list, full_md, task.pdf_path, tmp_dir, None, images_dir
        )

        # PP-StructureV3 refinement if paddleocr token available
        paddleocr_token = os.getenv("PADDLEOCR_API_TOKEN", "").strip()
        if paddleocr_token:
            from ppmatAgent.knowmat.batch.models import KeyInfo as _KI
            pp_key = _KI(key_id="pp_refine", token=paddleocr_token,
                         base_url="https://paddleocr.aistudio-app.com/api/v2/ocr/jobs")
            ocr_items, pp_report = self._refine_ppstructurev3(
                task.pdf_path, ocr_items, str(paper_dir), pp_key
            )
            metadata.setdefault("ocr_quality", {}).update(pp_report)

        # Post-processing (same as PaddleOCR API batch path)
        from ppmatAgent.knowmat.pdf.section_normalizer import (
            normalize_alloy_strings,
            normalize_leading_masthead_and_title,
            normalize_plain_author_superscripts,
            repair_keywords_abstract_two_column_ocr,
            structure_sections,
        )
        from ppmatAgent.knowmat.pdf.formula_formatter import format_formula_text
        from ppmatAgent.knowmat.pdf.doi_extractor import extract_first_doi, extract_first_doi_from_ocr_items

        md_text = extracted_text
        md_text = normalize_leading_masthead_and_title(md_text)
        md_text = structure_sections(md_text)
        md_text = repair_keywords_abstract_two_column_ocr(md_text)
        md_text = normalize_plain_author_superscripts(md_text)
        md_text = normalize_alloy_strings(md_text)
        md_text = format_formula_text(md_text)

        doi = extract_first_doi_from_ocr_items(ocr_items) or extract_first_doi(md_text[:5000])
        if doi and doi not in md_text:
            md_text = f"DOI: {doi}\n\n{md_text}"

        # Write final output (same structure as PaddleOCR API batch)
        md_path = paper_dir / f"{stem}.md"
        md_path.write_text(md_text, encoding="utf-8")

        json_path = paper_dir / f"{stem}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(ocr_items, f, ensure_ascii=False, indent=2)

        # Cleanup temp dir
        shutil.rmtree(tmp_dir, ignore_errors=True)

        return md_path

    def _refine_ppstructurev3(
        self,
        pdf_path: str,
        ocr_items: List[Dict[str, Any]],
        output_dir: str,
        key: KeyInfo,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Run PP-StructureV3 API refinement on ocr_items."""
        from ppmatAgent.knowmat.pdf.paddleocr_api_client import PaddleOCRAPIClient, PaddleOCRAPIError
        from ppmatAgent.knowmat.pdf.paddleocr_api_result_converter import (
            extract_formulas_per_page,
            extract_tables_per_page,
        )

        try:
            client = PaddleOCRAPIClient(key.token, key.base_url)
            job_result = client.upload_and_parse(
                Path(pdf_path), model="PP-StructureV3", timeout_sec=self._ocr_timeout
            )
            jsonl_url = job_result.get("resultUrl", {}).get("jsonUrl", "")
            if not jsonl_url:
                return ocr_items, {"ppstructure_status": "failed", "ppstructure_replacements": 0}

            pp_pages_data = client.download_jsonl(jsonl_url)
        except (PaddleOCRAPIError, Exception) as exc:
            logger.warning("PP-StructureV3 refinement failed for %s: %s", Path(pdf_path).name, exc)
            return ocr_items, {
                "ppstructure_status": "failed",
                "ppstructure_detail": str(exc)[:200],
                "ppstructure_replacements": 0,
            }

        pp_formulas = extract_formulas_per_page(pp_pages_data)
        pp_tables = extract_tables_per_page(pp_pages_data)

        replacements = 0
        formula_items_by_page: Dict[int, List[int]] = {}
        table_items_by_page: Dict[int, List[int]] = {}

        for idx, item in enumerate(ocr_items):
            page = item.get("page", 0)
            itype = item.get("type", "")
            if itype == "formula":
                formula_items_by_page.setdefault(page, []).append(idx)
            elif itype == "table":
                table_items_by_page.setdefault(page, []).append(idx)

        # Replace formula text with PP-StructureV3 output
        for page, indices in formula_items_by_page.items():
            page_formulas = pp_formulas.get(page, [])
            for i, idx in enumerate(indices):
                if i < len(page_formulas) and page_formulas[i]:
                    ocr_items[idx]["text"] = page_formulas[i]
                    replacements += 1

        # Replace table HTML with PP-StructureV3 output
        for page, indices in table_items_by_page.items():
            page_tables = pp_tables.get(page, [])
            for i, idx in enumerate(indices):
                if i < len(page_tables) and page_tables[i]:
                    ocr_items[idx]["text"] = page_tables[i]
                    replacements += 1

        return ocr_items, {
            "ppstructure_status": "done",
            "ppstructure_replacements": replacements,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_key_by_id(self, key_id: Optional[str]) -> KeyInfo:
        """Look up a key by ID, falling back to first available."""
        if key_id and key_id in self._key_pool._keys:
            return self._key_pool._keys[key_id]
        # Fallback: use any key (for recovery scenarios)
        return next(iter(self._key_pool._keys.values()))


def _is_rate_limit_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "429" in msg or "rate limit" in msg or "too many requests" in msg


def _is_transient_error(exc: Exception) -> bool:
    """Detect transient errors that should be retried (network, server-side)."""
    from urllib.error import URLError
    msg = str(exc).lower()
    # Network / timeout errors
    if isinstance(exc, (TimeoutError, ConnectionError, OSError)):
        return True
    if isinstance(exc, URLError):
        return True
    # HTTP 5xx server errors
    if any(code in msg for code in ("500", "502", "503", "504")):
        return True
    # Common transient keywords
    if any(kw in msg for kw in ("timeout", "timed out", "connection", "reset by peer",
                                 "broken pipe", "temporarily unavailable")):
        return True
    return False


def _is_not_found_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "404" in msg or "not found" in msg


def _parse_retry_after(exc: Exception) -> float:
    """Try to extract Retry-After value from error. Default 60s."""
    msg = str(exc)
    import re
    match = re.search(r"retry.after[:\s]*(\d+)", msg, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return 60.0
