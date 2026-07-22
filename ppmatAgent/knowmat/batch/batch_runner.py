"""Main batch processing orchestrator using asyncio."""

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


class BatchRunner:
    """Orchestrates large-scale parallel PDF processing.

    Architecture:
      [Discovery] → [OCR Submit Loop] → [OCR Poll Loop] → [LLM Consumer]
                                                                  ↓
                                                              [DONE]

    OCR and LLM run concurrently. As soon as any OCR job completes,
    the result is pushed to the LLM queue for immediate processing.
    """

    def __init__(
        self,
        input_folder: Path,
        output_dir: Path,
        vendor: str = "paddleocr",
        db_path: Optional[Path] = None,
        max_ocr_concurrent: int = 20,
        max_llm_concurrent: int = 4,
        max_retries: int = 3,
        poll_interval: float = 10.0,
        ocr_timeout: float = 600.0,
        ocr_only: bool = False,
        # Pipeline options passed through to orchestrator.run()
        max_runs: int = 1,
        full_pipeline: bool = False,
        enable_property_standardization: bool = False,
        **pipeline_kwargs: Any,
    ):
        self.input_folder = input_folder
        self.output_dir = output_dir
        self.vendor = vendor
        self.max_ocr_concurrent = max_ocr_concurrent
        self.max_llm_concurrent = max_llm_concurrent
        self.max_retries = max_retries
        self.poll_interval = poll_interval
        self.ocr_timeout = ocr_timeout
        self.ocr_only = ocr_only
        self.max_runs = max_runs
        self.full_pipeline = full_pipeline
        self.enable_property_standardization = enable_property_standardization
        self.pipeline_kwargs = pipeline_kwargs

        self._db_path = db_path or (input_folder / ".knowmat_batch.db")
        self._store: Optional[TaskStore] = None
        self._key_pool: Optional[KeyPool] = None
        self._llm_pool: Optional[ThreadPoolExecutor] = None
        self._shutdown_event = asyncio.Event()
        self._start_time = 0.0
        self._llm_done_count = 0

    async def run(self) -> Dict[str, Any]:
        """Main entry point. Returns summary statistics."""
        self._start_time = time.time()
        self._store = TaskStore(self._db_path)
        self._key_pool = KeyPool.from_env(self.vendor)
        self._llm_pool = ThreadPoolExecutor(
            max_workers=self.max_llm_concurrent,
            thread_name_prefix="llm-worker",
        )

        # Register signal handlers for graceful shutdown
        loop = asyncio.get_event_loop()
        if sys.platform != "win32":
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, self._request_shutdown)

        try:
            # Phase 1: Discover PDFs and populate task store
            new_count = self._discover_tasks()
            self._mark_already_done()

            stats = self._store.get_statistics()
            total = stats["total"]
            done = stats.get(TaskStatus.DONE.value, 0) + stats.get(TaskStatus.SKIPPED.value, 0)
            pending = stats.get(TaskStatus.PENDING.value, 0)
            submitted = stats.get(TaskStatus.OCR_SUBMITTED.value, 0)
            ocr_done = stats.get(TaskStatus.OCR_DONE.value, 0)

            print(f"\n[BATCH] Database: {self._db_path}")
            print(f"[BATCH] Total tasks: {total} | New: {new_count} | Done: {done}")
            print(f"[BATCH] To process: {pending} pending + {submitted} submitted + {ocr_done} ocr_done")
            print(f"[BATCH] Keys: {self._key_pool.size} ({self._key_pool.get_status_summary()})")
            if self.ocr_only:
                print(f"[BATCH] Mode: OCR-only (no LLM extraction)")
                print(f"[BATCH] Concurrency: OCR={self.max_ocr_concurrent}")
            else:
                print(f"[BATCH] Concurrency: OCR={self.max_ocr_concurrent} LLM={self.max_llm_concurrent}")
            print()

            if pending == 0 and submitted == 0 and ocr_done == 0:
                print("[BATCH] Nothing to process. All tasks are done or skipped.")
                return self._store.get_statistics()

            # Phase 2: Create dispatcher
            dispatcher = OCRDispatcher(
                store=self._store,
                key_pool=self._key_pool,
                vendor=self.vendor,
                input_folder=self.input_folder,
                max_concurrent_submit=self.max_ocr_concurrent,
                poll_interval=self.poll_interval,
                ocr_timeout=self.ocr_timeout,
            )

            # Phase 3: Run all loops concurrently
            ocr_completion_queue: asyncio.Queue[TaskRecord] = asyncio.Queue()

            if self.ocr_only:
                await asyncio.gather(
                    self._ocr_submit_loop(dispatcher),
                    dispatcher.poll_all_submitted(ocr_completion_queue, self._shutdown_event),
                    self._ocr_only_consumer(ocr_completion_queue),
                    self._retry_loop(),
                    self._progress_reporter(),
                    return_exceptions=True,
                )
            else:
                await asyncio.gather(
                    self._ocr_submit_loop(dispatcher),
                    dispatcher.poll_all_submitted(ocr_completion_queue, self._shutdown_event),
                    self._llm_consumer(ocr_completion_queue),
                    self._retry_loop(),
                    self._progress_reporter(),
                    return_exceptions=True,
                )

        except asyncio.CancelledError:
            logger.info("Batch runner cancelled")
        finally:
            if self._llm_pool:
                self._llm_pool.shutdown(wait=True)
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
        """Scan input_folder for PDFs, insert as PENDING."""
        pdf_files = sorted(self.input_folder.glob("*.pdf"), key=lambda p: p.name.lower())
        if not pdf_files:
            print(f"[BATCH] No PDF files found in {self.input_folder}")
            return 0

        new_count = self._store.bulk_insert_pending(pdf_files, max_retries=self.max_retries)
        logger.info("Discovered %d PDFs, %d new tasks created", len(pdf_files), new_count)
        return new_count

    def _mark_already_done(self) -> None:
        """Mark tasks that already have _extraction.json as SKIPPED."""
        pending = self._store.get_tasks_by_status(TaskStatus.PENDING, limit=50000)
        skipped = 0
        for task in pending:
            stem = Path(task.pdf_path).stem
            extraction_path = self.output_dir / stem / f"{stem}_extraction.json"
            if extraction_path.exists():
                self._store.update_status(task.task_id, TaskStatus.SKIPPED)
                skipped += 1
        if skipped:
            logger.info("Skipped %d tasks with existing extraction results", skipped)

    # ------------------------------------------------------------------
    # OCR Submit Loop
    # ------------------------------------------------------------------

    async def _ocr_submit_loop(self, dispatcher: OCRDispatcher) -> None:
        """Continuously submit PENDING tasks for OCR."""
        while not self._shutdown_event.is_set():
            pending = self._store.get_tasks_by_status(TaskStatus.PENDING, limit=100)
            if not pending:
                # Check if there's still work in other states
                stats = self._store.get_statistics()
                active = (
                    stats.get(TaskStatus.OCR_SUBMITTED.value, 0)
                    + stats.get(TaskStatus.OCR_DONE.value, 0)
                    + stats.get(TaskStatus.LLM_PROCESSING.value, 0)
                )
                if active == 0:
                    # All work is done
                    self._shutdown_event.set()
                    return
                await asyncio.sleep(5.0)
                continue

            # Submit batch concurrently
            submit_tasks = []
            for task in pending:
                coro = dispatcher.submit_one(task, self._shutdown_event)
                submit_tasks.append(asyncio.create_task(coro))

            await asyncio.gather(*submit_tasks, return_exceptions=True)
            # Brief pause to avoid tight-looping
            await asyncio.sleep(1.0)

    # ------------------------------------------------------------------
    # OCR-only Consumer (mark OCR_DONE as DONE, skip LLM)
    # ------------------------------------------------------------------

    async def _ocr_only_consumer(self, ocr_queue: asyncio.Queue) -> None:
        """In OCR-only mode, mark completed OCR tasks as DONE immediately."""
        ocr_done_tasks = self._store.get_tasks_by_status(TaskStatus.OCR_DONE, limit=10000)
        for task in ocr_done_tasks:
            self._store.update_status(task.task_id, TaskStatus.DONE)
            self._llm_done_count += 1

        while not self._shutdown_event.is_set() or not ocr_queue.empty():
            try:
                task = await asyncio.wait_for(ocr_queue.get(), timeout=3.0)
            except asyncio.TimeoutError:
                if self._shutdown_event.is_set() and ocr_queue.empty():
                    return
                continue
            self._store.update_status(task.task_id, TaskStatus.DONE)
            self._llm_done_count += 1
            logger.info("OCR-only done for %s", task.task_id)

    # ------------------------------------------------------------------
    # LLM Consumer
    # ------------------------------------------------------------------

    async def _llm_consumer(self, ocr_queue: asyncio.Queue) -> None:
        """Pull OCR-complete tasks, dispatch LLM processing in thread pool."""
        sem = asyncio.Semaphore(self.max_llm_concurrent)
        active_tasks: set = set()

        # Also pick up any OCR_DONE tasks from previous run (recovery)
        ocr_done_tasks = self._store.get_tasks_by_status(TaskStatus.OCR_DONE, limit=10000)
        for task in ocr_done_tasks:
            await ocr_queue.put(task)

        while not self._shutdown_event.is_set() or not ocr_queue.empty() or active_tasks:
            try:
                task = await asyncio.wait_for(ocr_queue.get(), timeout=3.0)
            except asyncio.TimeoutError:
                if self._shutdown_event.is_set() and ocr_queue.empty() and not active_tasks:
                    return
                continue

            await sem.acquire()
            coro = self._dispatch_llm(task, sem)
            t = asyncio.create_task(coro)
            active_tasks.add(t)
            t.add_done_callback(active_tasks.discard)

        # Wait for remaining LLM tasks to finish
        if active_tasks:
            await asyncio.gather(*active_tasks, return_exceptions=True)

    async def _dispatch_llm(self, task: TaskRecord, sem: asyncio.Semaphore) -> None:
        """Run sync orchestrator.run() in thread pool."""
        loop = asyncio.get_event_loop()
        try:
            self._store.update_status(task.task_id, TaskStatus.LLM_PROCESSING)
            await loop.run_in_executor(self._llm_pool, self._run_llm_sync, task)
            self._store.update_status(task.task_id, TaskStatus.DONE)
            self._llm_done_count += 1
            logger.info("LLM extraction done for %s", task.task_id)
        except Exception as exc:
            self._store.mark_failed(task.task_id, f"LLM error: {str(exc)[:500]}")
            logger.error("LLM failed for %s: %s", task.task_id, exc)
        finally:
            sem.release()

    def _run_llm_sync(self, task: TaskRecord) -> None:
        """Called in a thread. Reuses existing orchestrator.run()."""
        from ppmatAgent.knowmat.orchestrator import run

        md_path = task.md_path
        if not md_path:
            raise ValueError(f"No md_path for task {task.task_id}")

        run(
            pdf_path=md_path,
            output_dir=str(self.output_dir),
            max_runs=self.max_runs,
            full_pipeline=self.full_pipeline,
            enable_property_standardization=self.enable_property_standardization,
            subfield_model=self.pipeline_kwargs.get("subfield_model"),
            extraction_model=self.pipeline_kwargs.get("extraction_model"),
            evaluation_model=self.pipeline_kwargs.get("evaluation_model"),
            manager_model=self.pipeline_kwargs.get("manager_model"),
            flagging_model=self.pipeline_kwargs.get("flagging_model"),
        )

    # ------------------------------------------------------------------
    # Retry Loop
    # ------------------------------------------------------------------

    async def _retry_loop(self) -> None:
        """Periodically retry failed tasks that haven't exceeded max_retries."""
        while not self._shutdown_event.is_set():
            await asyncio.sleep(_RETRY_INTERVAL)
            if self._shutdown_event.is_set():
                return

            retryable = self._store.get_retryable_failed()
            retried = 0
            for task in retryable:
                if self._store.increment_retry(task.task_id):
                    retried += 1

            if retried:
                logger.info("Retried %d failed tasks", retried)

    # ------------------------------------------------------------------
    # Progress Reporter
    # ------------------------------------------------------------------

    async def _progress_reporter(self) -> None:
        """Periodically print progress statistics."""
        while not self._shutdown_event.is_set():
            await asyncio.sleep(_PROGRESS_INTERVAL)
            if self._shutdown_event.is_set():
                return
            self._print_progress()

    def _print_progress(self) -> None:
        stats = self._store.get_statistics()
        elapsed = time.time() - self._start_time
        elapsed_min = elapsed / 60

        done = stats.get(TaskStatus.DONE.value, 0)
        skipped = stats.get(TaskStatus.SKIPPED.value, 0)
        submitted = stats.get(TaskStatus.OCR_SUBMITTED.value, 0)
        ocr_done = stats.get(TaskStatus.OCR_DONE.value, 0)
        llm_proc = stats.get(TaskStatus.LLM_PROCESSING.value, 0)
        pending = stats.get(TaskStatus.PENDING.value, 0)
        failed = stats.get(TaskStatus.FAILED.value, 0)
        total = stats["total"]

        completed = done + skipped
        rate = done / elapsed_min if elapsed_min > 0 else 0

        timestamp = time.strftime("%H:%M:%S")
        print(
            f"[BATCH] {timestamp} | "
            f"done: {completed}/{total} | "
            f"ocr_submitted: {submitted} | "
            f"ocr_done: {ocr_done} | "
            f"llm: {llm_proc} | "
            f"pending: {pending} | "
            f"failed: {failed} | "
            f"rate: {rate:.1f}/min | "
            f"keys: {self._key_pool.get_status_summary()}"
        )

    def _print_final_summary(self, stats: Dict[str, Any]) -> None:
        elapsed = time.time() - self._start_time
        elapsed_min = elapsed / 60

        done = stats.get(TaskStatus.DONE.value, 0)
        skipped = stats.get(TaskStatus.SKIPPED.value, 0)
        failed = stats.get(TaskStatus.FAILED.value, 0)
        total = stats["total"]

        print(f"\n{'='*60}")
        print(f"[BATCH] COMPLETED in {elapsed_min:.1f} minutes")
        print(f"[BATCH] Total: {total} | Done: {done} | Skipped: {skipped} | Failed: {failed}")
        if elapsed_min > 0 and done > 0:
            print(f"[BATCH] Average throughput: {done / elapsed_min:.1f} papers/min")
        if failed > 0:
            print(f"[BATCH] {failed} failed tasks. Re-run with same command to retry.")
        print(f"{'='*60}")

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def _request_shutdown(self) -> None:
        """Signal handler for graceful shutdown."""
        if self._shutdown_event.is_set():
            print("\n[BATCH] Force exit requested. Terminating...")
            sys.exit(1)
        print("\n[BATCH] Shutdown requested. Finishing in-progress work...")
        self._shutdown_event.set()
