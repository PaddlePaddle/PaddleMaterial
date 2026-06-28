"""Disk cache for full OCR pipeline results (per PDF and OCR settings)."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

CACHE_VERSION = 1
RESULT_FILENAME = "cached_ocr_result.json"


def md5_file_digest(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return hex MD5 digest of file contents."""
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_pages_argument(spec: str, total_pages: int) -> List[int]:
    """Parse ``1-3,5,7-9`` style spec into sorted unique 1-based page indices."""
    spec = spec.strip()
    if not spec:
        return list(range(1, total_pages + 1))
    pages: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, _, b = part.partition("-")
            start, end = int(a.strip()), int(b.strip())
            if start > end:
                start, end = end, start
            for p in range(start, end + 1):
                if 1 <= p <= total_pages:
                    pages.add(p)
        else:
            p = int(part)
            if 1 <= p <= total_pages:
                pages.add(p)
    return sorted(pages)


def pages_key_for_cache(selected: List[int], total_pages: int) -> str:
    """Short string describing page selection for cache signatures."""
    full = list(range(1, total_pages + 1))
    if selected == full:
        return f"all_{total_pages}"
    body = ",".join(str(p) for p in selected)
    return hashlib.md5(body.encode("utf-8")).hexdigest()[:16]


def cache_signature_key(
    pdf_digest: str,
    *,
    render_dpi: int,
    vl_version: str,
    pages_key: str,
    skip_ppstructure: bool,
    skip_chem_reocr: bool,
) -> str:
    raw = "|".join(
        [
            pdf_digest,
            str(render_dpi),
            vl_version,
            pages_key,
            "ppS1" if skip_ppstructure else "ppS0",
            "ch1" if skip_chem_reocr else "ch0",
            str(CACHE_VERSION),
        ]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def ocr_cache_bucket(output_dir: Path, sig: str) -> Path:
    return Path(output_dir) / "_ocr_cache" / sig


def try_load_ocr_cache(bucket: Path) -> Optional[Dict[str, Any]]:
    path = bucket / RESULT_FILENAME
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    if data.get("cache_version") != CACHE_VERSION:
        return None
    return data


def save_ocr_cache(bucket: Path, payload: Dict[str, Any]) -> None:
    bucket.mkdir(parents=True, exist_ok=True)
    out = {**payload, "cache_version": CACHE_VERSION}
    (bucket / RESULT_FILENAME).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


def clear_all_ocr_caches_under(root: Path) -> int:
    """Remove every ``_ocr_cache`` directory under *root*. Returns count removed."""
    removed = 0
    if not root.exists():
        return 0
    for p in sorted(root.rglob("_ocr_cache"), reverse=True):
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
            removed += 1
    return removed
