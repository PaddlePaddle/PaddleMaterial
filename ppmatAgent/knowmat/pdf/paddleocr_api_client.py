"""PaddleOCR cloud API client (supports PaddleOCR-VL-* and PP-StructureV3 models)."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://paddleocr.aistudio-app.com/api/v2/ocr/jobs"

# Default VL model when neither caller nor env specifies one.  Kept at 1.5 for
# backward compatibility; override via env to use a newer release (e.g. 1.6).
_DEFAULT_VL_MODEL = "PaddleOCR-VL-1.5"


def get_paddleocr_vl_model() -> str:
    """Return the PaddleOCR-VL model name for the primary OCR pass.

    Resolution order:
      1. ``PADDLEOCR_API_MODEL`` env var (e.g. "PaddleOCR-VL-1.6")
      2. ``PADDLEOCRVL_VERSION`` env var (a bare version like "1.6" → "PaddleOCR-VL-1.6")
      3. built-in default (``PaddleOCR-VL-1.5``)

    Only affects the VL OCR model; PP-StructureV3 refinement is unrelated and
    always uses "PP-StructureV3".
    """
    explicit = os.getenv("PADDLEOCR_API_MODEL", "").strip()
    if explicit:
        return explicit
    version = os.getenv("PADDLEOCRVL_VERSION", "").strip()
    if version:
        # Accept either a bare version ("1.6") or an already-qualified name.
        return version if version.lower().startswith("paddleocr") else f"PaddleOCR-VL-{version}"
    return _DEFAULT_VL_MODEL


_DEFAULT_OPTIONS = {
    "useDocOrientationClassify": False,
    "useDocUnwarping": False,
    "useChartRecognition": False,
}


class PaddleOCRAPIError(Exception):
    """Raised on non-recoverable PaddleOCR API errors."""

    def __init__(self, message: str, status_code: int = 0, response_body: str = ""):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


class PaddleOCRAPIClient:
    """Client for PaddleOCR cloud API v2."""

    def __init__(self, token: str, base_url: str = _DEFAULT_BASE_URL):
        self.token = token
        self.base_url = base_url.rstrip("/")

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"bearer {self.token}"}

    def submit_job(
        self,
        pdf_path: Path,
        model: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Upload PDF and submit OCR job. Returns jobId.

        When *model* is None, the VL model is resolved from env via
        :func:`get_paddleocr_vl_model` (default ``PaddleOCR-VL-1.5``).
        Callers that need PP-StructureV3 pass it explicitly.
        """
        model = model or get_paddleocr_vl_model()
        opts = {**_DEFAULT_OPTIONS, **(options or {})}
        file_size = pdf_path.stat().st_size
        logger.info(
            "[PaddleOCR API] Submitting %s (%.1f MB) with model=%s",
            pdf_path.name, file_size / 1e6, model,
        )

        data = {
            "model": model,
            "optionalPayload": json.dumps(opts),
        }

        with open(pdf_path, "rb") as f:
            files = {"file": (pdf_path.name, f, "application/pdf")}
            resp = requests.post(
                self.base_url,
                headers=self._headers(),
                data=data,
                files=files,
                timeout=120,
            )

        if resp.status_code != 200:
            raise PaddleOCRAPIError(
                f"Job submission failed (HTTP {resp.status_code}): {resp.text[:500]}",
                status_code=resp.status_code,
                response_body=resp.text,
            )

        result = resp.json()
        job_id = result.get("data", {}).get("jobId", "")
        if not job_id:
            raise PaddleOCRAPIError(f"No jobId in response: {result}")

        logger.info("[PaddleOCR API] Job submitted: %s", job_id)
        return job_id

    def poll_job(
        self,
        job_id: str,
        timeout_sec: float = 600,
        poll_interval: float = 5,
    ) -> Dict[str, Any]:
        """Poll job until done or failed. Returns job result data."""
        url = f"{self.base_url}/{job_id}"
        start = time.time()

        while True:
            elapsed = time.time() - start
            if elapsed > timeout_sec:
                raise PaddleOCRAPIError(
                    f"Polling timeout after {timeout_sec}s for job {job_id}"
                )

            resp = requests.get(url, headers=self._headers(), timeout=30)
            if resp.status_code != 200:
                raise PaddleOCRAPIError(
                    f"Poll request failed (HTTP {resp.status_code}): {resp.text[:300]}",
                    status_code=resp.status_code,
                )

            data = resp.json().get("data", {})
            state = data.get("state", "")

            if state == "done":
                progress = data.get("extractProgress", {})
                logger.info(
                    "[PaddleOCR API] Job complete (%.0fs), pages=%s",
                    elapsed, progress.get("extractedPages", "?"),
                )
                return data

            if state == "failed":
                error_msg = data.get("errorMsg", "unknown error")
                raise PaddleOCRAPIError(f"PaddleOCR job failed: {error_msg}")

            if state == "running":
                progress = data.get("extractProgress", {})
                extracted = progress.get("extractedPages", "?")
                total = progress.get("totalPages", "?")
                logger.debug(
                    "[PaddleOCR API] Running: %s/%s pages (%.0fs)",
                    extracted, total, elapsed,
                )

            time.sleep(poll_interval)

    def download_jsonl(self, jsonl_url: str) -> List[Dict[str, Any]]:
        """Download JSONL result and parse into list of page data dicts."""
        logger.info("[PaddleOCR API] Downloading JSONL from %s...", jsonl_url[:80])
        resp = requests.get(jsonl_url, timeout=120)
        resp.raise_for_status()

        pages: List[Dict[str, Any]] = []
        for line in resp.text.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            pages.append(json.loads(line))

        logger.info("[PaddleOCR API] Downloaded %d page(s) of results", len(pages))
        return pages

    def download_image(self, url: str, dest_path: Path) -> Path:
        """Download a single image from URL to local path."""
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        dest_path.write_bytes(resp.content)
        return dest_path

    def upload_and_parse(
        self,
        pdf_path: Path,
        model: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        timeout_sec: float = 600,
    ) -> Dict[str, Any]:
        """Upload PDF, poll until done, return full job result data.

        The returned dict contains:
        - resultUrl.jsonUrl: URL to download JSONL results
        - extractProgress: {totalPages, extractedPages, startTime, endTime}
        """
        job_id = self.submit_job(pdf_path, model=model, options=options)
        return self.poll_job(job_id, timeout_sec=timeout_sec)
