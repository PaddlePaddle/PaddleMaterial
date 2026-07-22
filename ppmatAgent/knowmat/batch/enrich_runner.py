"""Async batch OCR → CLIP+VLM enrichment runner.

Combines async batch OCR (cloud API) with concurrent CLIP+VLM enrichment.
As each paper's OCR completes, enrichment starts immediately.

Usage (via python -m ppmatAgent.knowmat --final-md):
    See ppmatAgent/knowmat/__main__.py for the integrated CLI.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Optional

from ppmatAgent.knowmat.batch.key_pool import KeyPool
from ppmatAgent.knowmat.batch.models import TaskRecord, TaskStatus
from ppmatAgent.knowmat.batch.ocr_dispatcher import OCRDispatcher
from ppmatAgent.knowmat.batch.task_store import TaskStore

logger = logging.getLogger(__name__)

_PROGRESS_INTERVAL = 30.0
_RETRY_INTERVAL = 60.0


class EnrichRunner:
    """Batch OCR → CLIP+VLM enrichment pipeline producing _final.md per paper.

    Architecture:
      [Discovery] → [OCR Submit] → [OCR Poll] → [Enrich Consumer] → [DONE]

    All existing OCR_DONE tasks are pre-queued synchronously at startup to
    avoid the race condition where the consumer exits before the drain fills
    the queue.
    """

    def __init__(
        self,
        input_folder: Path,
        output_dir: Path,
        vendor: str = "paddleocr",
        db_path: Optional[Path] = None,
        max_ocr_concurrent: int = 20,
        max_enrich_concurrent: int = 4,
        vlm_workers: int = 1,
        max_retries: int = 3,
        poll_interval: float = 10.0,
        ocr_timeout: float = 600.0,
        skip_existing: bool = False,
        retry_incomplete: bool = False,
    ):
        self.input_folder = input_folder
        self.output_dir = output_dir
        self.vendor = vendor
        self.max_ocr_concurrent = max_ocr_concurrent
        self.max_enrich_concurrent = max_enrich_concurrent
        self.vlm_workers = vlm_workers
        self.max_retries = max_retries
        self.poll_interval = poll_interval
        self.ocr_timeout = ocr_timeout
        self.skip_existing = skip_existing
        self.retry_incomplete = retry_incomplete

        self._db_path = db_path or (input_folder / ".knowmat_batch_enrich.db")
        self._store: Optional[TaskStore] = None
        self._key_pool: Optional[KeyPool] = None
        self._enrich_pool: Optional[ThreadPoolExecutor] = None
        self._shutdown_event = asyncio.Event()
        self._start_time = 0.0
        self._enrich_done_count = 0

    async def run(self) -> Dict[str, Any]:
        self._start_time = time.time()
        self._store = TaskStore(self._db_path)
        self._key_pool = KeyPool.from_env(self.vendor)
        self._enrich_pool = ThreadPoolExecutor(
            max_workers=self.max_enrich_concurrent,
            thread_name_prefix="enrich-worker",
        )

        loop = asyncio.get_event_loop()
        if sys.platform != "win32":
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, self._request_shutdown)

        try:
            new_count = self._discover_tasks()
            self._mark_already_done()
            self._reset_stuck_tasks()

            stats = self._store.get_statistics()
            total = stats["total"]
            done = stats.get(TaskStatus.DONE.value, 0) + stats.get(TaskStatus.SKIPPED.value, 0)
            pending = stats.get(TaskStatus.PENDING.value, 0)
            submitted = stats.get(TaskStatus.OCR_SUBMITTED.value, 0)
            ocr_done = stats.get(TaskStatus.OCR_DONE.value, 0)

            print(f"\n[ENRICH] Database: {self._db_path}")
            print(f"[ENRICH] Total: {total} | New: {new_count} | Done: {done}")
            print(f"[ENRICH] To process: {pending} pending + {submitted} submitted + {ocr_done} ocr_done")
            print(f"[ENRICH] Keys: {self._key_pool.size} ({self._key_pool.get_status_summary()})")
            print(f"[ENRICH] Concurrency: OCR={self.max_ocr_concurrent} Enrich={self.max_enrich_concurrent} VLM/paper={self.vlm_workers}")
            print()

            if pending == 0 and submitted == 0 and ocr_done == 0:
                print("[ENRICH] Nothing to process.")
                return self._store.get_statistics()

            dispatcher = OCRDispatcher(
                store=self._store,
                key_pool=self._key_pool,
                vendor=self.vendor,
                input_folder=self.input_folder,
                max_concurrent_submit=self.max_ocr_concurrent,
                poll_interval=self.poll_interval,
                ocr_timeout=self.ocr_timeout,
            )

            ocr_completion_queue: asyncio.Queue[TaskRecord] = asyncio.Queue()

            # Pre-populate queue synchronously to avoid consumer/drain race condition.
            self._pre_drain_ocr_done(ocr_completion_queue)

            await asyncio.gather(
                self._ocr_submit_loop(dispatcher),
                dispatcher.poll_all_submitted(ocr_completion_queue, self._shutdown_event),
                self._enrich_consumer(ocr_completion_queue),
                self._retry_loop(),
                self._progress_reporter(),
                return_exceptions=True,
            )
        except asyncio.CancelledError:
            logger.info("Enrich runner cancelled")
        finally:
            if self._enrich_pool:
                self._enrich_pool.shutdown(wait=True)
            if self._store:
                final_stats = self._store.get_statistics()
                self._store.close()
                self._print_final_summary(final_stats)
                return final_stats
        return {}

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def _discover_tasks(self) -> int:
        pdf_files = sorted(self.input_folder.glob("*.pdf"), key=lambda p: p.name.lower())
        if not pdf_files:
            print(f"[ENRICH] No PDF files found in {self.input_folder}")
            return 0
        return self._store.bulk_insert_pending(pdf_files, max_retries=self.max_retries)

    def _mark_already_done(self) -> None:
        if not self.skip_existing:
            return
        pending = self._store.get_tasks_by_status(TaskStatus.PENDING, limit=50000)
        skipped = 0
        for task in pending:
            stem = Path(task.pdf_path).stem
            final_md = self.output_dir / stem / f"{stem}_final.md"
            if final_md.exists():
                self._store.update_status(task.task_id, TaskStatus.SKIPPED)
                skipped += 1
        if skipped:
            print(f"[ENRICH] Skipped {skipped} papers with existing _final.md")

    def _reset_stuck_tasks(self) -> None:
        count = self._store.reset_stuck_tasks()
        if count:
            print(f"[ENRICH] Reset {count} stuck llm_processing tasks → ocr_done")

    def _pre_drain_ocr_done(self, queue: asyncio.Queue) -> None:
        """Synchronously enqueue all existing OCR_DONE tasks before the event loop starts."""
        tasks = self._store.get_tasks_by_status(TaskStatus.OCR_DONE, limit=50000)
        for task in tasks:
            queue.put_nowait(task)
        if tasks:
            print(f"[ENRICH] Pre-queued {len(tasks)} existing OCR_DONE tasks")

    # ------------------------------------------------------------------
    # OCR Submit Loop
    # ------------------------------------------------------------------

    async def _ocr_submit_loop(self, dispatcher: OCRDispatcher) -> None:
        while not self._shutdown_event.is_set():
            tasks = self._store.get_tasks_by_status(TaskStatus.PENDING, limit=50)
            if not tasks:
                await asyncio.sleep(5)
                if self._all_ocr_done():
                    break
                continue
            for task in tasks:
                if self._shutdown_event.is_set():
                    return
                try:
                    await dispatcher.submit_one(task, self._shutdown_event)
                except Exception as exc:
                    logger.error("Submit failed for %s: %s", task.task_id, exc)
                    self._store.mark_failed(task.task_id, str(exc))

    def _all_ocr_done(self) -> bool:
        stats = self._store.get_statistics()
        return (
            stats.get(TaskStatus.PENDING.value, 0) == 0
            and stats.get(TaskStatus.OCR_SUBMITTED.value, 0) == 0
        )

    # ------------------------------------------------------------------
    # Enrich Consumer
    # ------------------------------------------------------------------

    async def _enrich_consumer(self, queue: asyncio.Queue[TaskRecord]) -> None:
        loop = asyncio.get_event_loop()
        sem = asyncio.Semaphore(self.max_enrich_concurrent)

        async def _process(task: TaskRecord):
            async with sem:
                try:
                    self._store.update_status(task.task_id, TaskStatus.LLM_PROCESSING)
                    await loop.run_in_executor(
                        self._enrich_pool, self._enrich_one, task
                    )
                    self._store.update_status(task.task_id, TaskStatus.DONE)
                    self._enrich_done_count += 1
                except Exception as exc:
                    logger.error("Enrich failed for %s: %s", task.task_id, exc)
                    self._store.mark_failed(task.task_id, str(exc))

        tasks_spawned = []
        while not self._shutdown_event.is_set():
            try:
                task = await asyncio.wait_for(queue.get(), timeout=5.0)
                t = asyncio.create_task(_process(task))
                tasks_spawned.append(t)
            except asyncio.TimeoutError:
                if (
                    self._all_ocr_done()
                    and queue.empty()
                    and self._store.count_by_status(TaskStatus.OCR_DONE) == 0
                ):
                    break
                continue

        if tasks_spawned:
            await asyncio.gather(*tasks_spawned, return_exceptions=True)

    def _enrich_one(self, task: TaskRecord) -> None:
        from ppmatAgent.knowmat.pdf.pipeline_c import enrich_paper_text

        stem = Path(task.pdf_path).stem
        out_dir = self.output_dir / stem
        final_md_path = out_dir / f"{stem}_final.md"

        if final_md_path.exists():
            if self.skip_existing and not self.retry_incomplete:
                return
            if self.retry_incomplete:
                _existing = final_md_path.read_text(encoding="utf-8")
                if "AI Description]:" in _existing or "VLM-digitized" in _existing:
                    return

        enriched_text = enrich_paper_text(stem, self.input_folder, vlm_workers=self.vlm_workers)
        if enriched_text:
            out_dir.mkdir(parents=True, exist_ok=True)
            final_md_path.write_text(enriched_text, encoding="utf-8")
        else:
            raise RuntimeError(f"enrich_paper_text returned None for {stem}")

    # ------------------------------------------------------------------
    # Retry & Progress
    # ------------------------------------------------------------------

    async def _retry_loop(self) -> None:
        while not self._shutdown_event.is_set():
            await asyncio.sleep(_RETRY_INTERVAL)
            failed = self._store.get_tasks_by_status(TaskStatus.FAILED, limit=100)
            retried = sum(1 for t in failed if self._store.increment_retry(t.task_id))
            if retried:
                logger.info("Retried %d failed tasks", retried)

    async def _progress_reporter(self) -> None:
        while not self._shutdown_event.is_set():
            await asyncio.sleep(_PROGRESS_INTERVAL)
            stats = self._store.get_statistics()
            elapsed = time.time() - self._start_time
            total = stats["total"]
            done = stats.get(TaskStatus.DONE.value, 0)
            skipped = stats.get(TaskStatus.SKIPPED.value, 0)
            failed = stats.get(TaskStatus.FAILED.value, 0)
            pending = stats.get(TaskStatus.PENDING.value, 0)
            submitted = stats.get(TaskStatus.OCR_SUBMITTED.value, 0)
            ocr_done = stats.get(TaskStatus.OCR_DONE.value, 0)
            enriching = stats.get(TaskStatus.LLM_PROCESSING.value, 0)
            print(
                f"[{elapsed:.0f}s] "
                f"done={done} enriching={enriching} ocr_done={ocr_done} "
                f"submitted={submitted} pending={pending} "
                f"failed={failed} skipped={skipped} total={total}"
            )
            if done + skipped + failed >= total:
                break

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _request_shutdown(self) -> None:
        print("\n[ENRICH] Shutdown requested, finishing in-flight tasks...")
        self._shutdown_event.set()

    def _print_final_summary(self, stats: Dict[str, Any]) -> None:
        elapsed = time.time() - self._start_time
        done = stats.get(TaskStatus.DONE.value, 0)
        failed = stats.get(TaskStatus.FAILED.value, 0)
        skipped = stats.get(TaskStatus.SKIPPED.value, 0)
        total = stats["total"]
        print(f"\n{'='*60}")
        print(f"[ENRICH] Complete in {elapsed:.1f}s")
        print(f"  Done: {done} | Failed: {failed} | Skipped: {skipped} | Total: {total}")
        if failed:
            print("  Re-run to retry failed tasks (state persisted in DB)")
        print(f"{'='*60}")
