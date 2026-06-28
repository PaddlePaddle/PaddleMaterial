"""MinerU cloud API client for PDF parsing (Precision API v4 + Lightweight API v1)."""

from __future__ import annotations

import io
import json
import logging
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

_PRECISION_BASE = "https://mineru.net"
_LIGHTWEIGHT_BASE = "https://mineru.net"


class MineruAPIError(Exception):
    """Raised on non-recoverable MinerU API errors."""

    def __init__(self, message: str, status_code: int = 0, response_body: str = ""):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


def _json_request(url: str, *, method: str = "GET", data: Optional[dict] = None,
                  headers: Optional[Dict[str, str]] = None, timeout: float = 60) -> dict:
    """Make an HTTP request and return parsed JSON response."""
    hdrs = headers or {}
    body = None
    if data is not None:
        body = json.dumps(data).encode("utf-8")
        hdrs.setdefault("Content-Type", "application/json")
    req = Request(url, data=body, headers=hdrs, method=method)
    try:
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw)
    except HTTPError as exc:
        body_text = ""
        try:
            body_text = exc.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        raise MineruAPIError(
            f"HTTP {exc.code} from {url}: {body_text[:500]}",
            status_code=exc.code,
            response_body=body_text,
        ) from exc
    except URLError as exc:
        raise MineruAPIError(f"Network error calling {url}: {exc.reason}") from exc


def _put_binary(url: str, file_path: Path, timeout: float = 300) -> None:
    """PUT file binary to a pre-signed URL."""
    data = file_path.read_bytes()
    req = Request(url, data=data, method="PUT")
    # OSS signed URLs require matching Content-Type; empty string prevents urllib's default
    req.add_header("Content-Type", "")
    try:
        with urlopen(req, timeout=timeout) as resp:
            _ = resp.read()
    except HTTPError as exc:
        body_text = ""
        try:
            body_text = exc.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        raise MineruAPIError(
            f"File upload failed (HTTP {exc.code}): {body_text[:300]}",
            status_code=exc.code,
        ) from exc


class MineruPrecisionClient:
    """Client for MinerU Precision API v4 (requires API key)."""

    def __init__(self, api_key: str, base_url: str = _PRECISION_BASE):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def upload_and_parse(
        self,
        pdf_path: Path,
        *,
        model_version: str = "vlm",
        is_ocr: bool = True,
        enable_formula: bool = True,
        enable_table: bool = True,
        language: str = "en",
        page_ranges: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Upload a local PDF and parse it. Returns task result dict with full_zip_url."""
        file_name = pdf_path.name
        logger.info("[MinerU API] Uploading %s (%.1f MB)...", file_name, pdf_path.stat().st_size / 1e6)

        file_meta: Dict[str, Any] = {"name": file_name, "is_ocr": is_ocr}
        batch_body: Dict[str, Any] = {
            "files": [file_meta],
            "model_version": model_version,
            "enable_formula": enable_formula,
            "enable_table": enable_table,
            "language": language,
        }
        if page_ranges:
            file_meta["page_ranges"] = page_ranges

        resp = _json_request(
            f"{self.base_url}/api/v4/file-urls/batch",
            method="POST",
            data=batch_body,
            headers=self._headers(),
        )
        if resp.get("code") != 0:
            raise MineruAPIError(f"file-urls/batch failed: {resp.get('msg', resp)}")

        batch_data = resp["data"]
        batch_id = batch_data["batch_id"]
        file_urls = batch_data.get("file_urls", [])
        if not file_urls:
            raise MineruAPIError("No file_urls returned from batch endpoint")

        upload_url = file_urls[0]
        logger.info("[MinerU API] Uploading file to signed URL...")
        _put_binary(upload_url, pdf_path)
        logger.info("[MinerU API] Upload complete. batch_id=%s", batch_id)

        return self._poll_batch(batch_id)

    def submit_file(
        self,
        pdf_path: Path,
        *,
        model_version: str = "vlm",
        is_ocr: bool = True,
        enable_formula: bool = True,
        enable_table: bool = True,
        language: str = "en",
    ) -> tuple:
        """Upload a PDF and return (batch_id, batch_data) without polling.

        Used by the batch dispatcher which handles polling separately.
        """
        file_name = pdf_path.name
        logger.info("[MinerU API] Submitting %s (%.1f MB)...", file_name, pdf_path.stat().st_size / 1e6)

        file_meta: Dict[str, Any] = {"name": file_name, "is_ocr": is_ocr}
        batch_body: Dict[str, Any] = {
            "files": [file_meta],
            "model_version": model_version,
            "enable_formula": enable_formula,
            "enable_table": enable_table,
            "language": language,
        }

        resp = _json_request(
            f"{self.base_url}/api/v4/file-urls/batch",
            method="POST",
            data=batch_body,
            headers=self._headers(),
        )
        if resp.get("code") != 0:
            raise MineruAPIError(f"file-urls/batch failed: {resp.get('msg', resp)}")

        batch_data = resp["data"]
        batch_id = batch_data["batch_id"]
        file_urls = batch_data.get("file_urls", [])
        if not file_urls:
            raise MineruAPIError("No file_urls returned from batch endpoint")

        upload_url = file_urls[0]
        _put_binary(upload_url, pdf_path)
        logger.info("[MinerU API] Submitted. batch_id=%s", batch_id)

        return batch_id, batch_data

    def poll_batch(self, batch_id: str) -> tuple:
        """Single non-blocking poll. Returns (status, results_list).

        status is one of: 'done', 'failed', 'running'.
        results_list contains task dicts when status is 'done'.
        Used by the batch dispatcher for periodic polling.
        """
        url = f"{self.base_url}/api/v4/extract-results/batch/{batch_id}"
        resp = _json_request(url, headers=self._headers(), timeout=30)
        if resp.get("code") != 0:
            raise MineruAPIError(f"Batch poll error: {resp.get('msg', resp)}")

        results = resp.get("data", {}).get("extract_result", [])
        if results:
            task = results[0]
            state = task.get("state", "")
            if state == "done":
                return "done", results
            if state == "failed":
                return "failed", results
        return "running", []

    def _poll_batch(self, batch_id: str, timeout_sec: float = 600, poll_interval: float = 5) -> Dict[str, Any]:
        """Poll batch results until done or timeout."""
        url = f"{self.base_url}/api/v4/extract-results/batch/{batch_id}"
        start = time.time()
        while True:
            elapsed = time.time() - start
            if elapsed > timeout_sec:
                raise MineruAPIError(f"Polling timeout after {timeout_sec}s for batch {batch_id}")

            resp = _json_request(url, headers=self._headers(), timeout=30)
            if resp.get("code") != 0:
                raise MineruAPIError(f"Batch poll error: {resp.get('msg', resp)}")

            results = resp.get("data", {}).get("extract_result", [])
            if results:
                task = results[0]
                state = task.get("state", "")
                if state == "done":
                    logger.info("[MinerU API] Parsing complete (%.0fs elapsed)", elapsed)
                    return task
                if state == "failed":
                    raise MineruAPIError(f"MinerU task failed: {task.get('err_msg', 'unknown error')}")
                logger.debug("[MinerU API] State=%s, elapsed=%.0fs", state, elapsed)

            time.sleep(poll_interval)

    def download_and_extract_zip(self, zip_url: str, dest_dir: Path) -> Path:
        """Download result ZIP and extract to dest_dir. Returns extracted directory."""
        dest_dir.mkdir(parents=True, exist_ok=True)
        logger.info("[MinerU API] Downloading results from %s...", zip_url[:80])

        req = Request(zip_url)
        with urlopen(req, timeout=120) as resp:
            zip_data = resp.read()

        with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
            zf.extractall(dest_dir)

        logger.info("[MinerU API] Extracted %d files to %s", len(list(dest_dir.rglob("*"))), dest_dir)
        return dest_dir


class MineruLightweightClient:
    """Client for MinerU Agent Lightweight API v1 (no auth required)."""

    def __init__(self, base_url: str = _LIGHTWEIGHT_BASE):
        self.base_url = base_url.rstrip("/")

    def upload_and_parse(self, pdf_path: Path, *, page_range: Optional[str] = None) -> str:
        """Upload a local PDF and parse it. Returns markdown text."""
        file_name = pdf_path.name
        file_size = pdf_path.stat().st_size
        if file_size > 10 * 1024 * 1024:
            raise MineruAPIError(f"File too large for Lightweight API ({file_size / 1e6:.1f} MB > 10 MB limit)")

        logger.info("[MinerU Lightweight] Uploading %s (%.1f MB)...", file_name, file_size / 1e6)

        body: Dict[str, Any] = {"file_name": file_name}
        if page_range:
            body["page_range"] = page_range

        resp = _json_request(
            f"{self.base_url}/api/v1/agent/parse/file",
            method="POST",
            data=body,
        )
        if resp.get("code") != 0:
            raise MineruAPIError(f"Lightweight file endpoint failed: {resp.get('msg', resp)}")

        task_data = resp["data"]
        task_id = task_data["task_id"]
        file_url = task_data.get("file_url", "")

        if file_url:
            logger.info("[MinerU Lightweight] Uploading file to signed URL...")
            _put_binary(file_url, pdf_path)

        return self._poll_task(task_id)

    def _poll_task(self, task_id: str, timeout_sec: float = 300, poll_interval: float = 5) -> str:
        """Poll task until done, then fetch and return markdown content."""
        url = f"{self.base_url}/api/v1/agent/parse/{task_id}"
        start = time.time()
        while True:
            elapsed = time.time() - start
            if elapsed > timeout_sec:
                raise MineruAPIError(f"Lightweight polling timeout after {timeout_sec}s for task {task_id}")

            resp = _json_request(url, timeout=30)
            if resp.get("code") != 0:
                raise MineruAPIError(f"Lightweight poll error: {resp.get('msg', resp)}")

            task = resp.get("data", {})
            state = task.get("state", "")
            if state == "done":
                markdown_url = task.get("markdown_url", "")
                if not markdown_url:
                    raise MineruAPIError("Task done but no markdown_url returned")
                logger.info("[MinerU Lightweight] Parsing complete (%.0fs), fetching markdown...", elapsed)
                return self._fetch_markdown(markdown_url)
            if state == "failed":
                raise MineruAPIError(
                    f"Lightweight task failed: {task.get('err_msg', 'unknown')} (code={task.get('err_code')})"
                )
            logger.debug("[MinerU Lightweight] State=%s, elapsed=%.0fs", state, elapsed)
            time.sleep(poll_interval)

    def _fetch_markdown(self, url: str) -> str:
        """Download markdown content from CDN URL."""
        req = Request(url)
        with urlopen(req, timeout=60) as resp:
            return resp.read().decode("utf-8")
