"""Data models for the batch processing pipeline."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class TaskStatus(str, Enum):
    PENDING = "pending"
    OCR_SUBMITTED = "ocr_submitted"
    OCR_DONE = "ocr_done"
    LLM_PROCESSING = "llm_processing"
    DONE = "done"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class TaskRecord:
    task_id: str
    pdf_path: str
    status: TaskStatus
    ocr_vendor: Optional[str] = None
    ocr_job_id: Optional[str] = None
    md_path: Optional[str] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    api_key_id: Optional[str] = None


@dataclass
class KeyInfo:
    key_id: str
    token: str
    base_url: str
    cooldown_until: float = 0.0
    total_requests: int = 0
    total_errors: int = 0
