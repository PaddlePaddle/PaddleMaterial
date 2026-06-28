"""DOI extraction helpers used by parser and schema conversion."""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DOI_RE = re.compile(r"\b(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)\b")
DOI_URL_RE = re.compile(r"https?://doi\.org/(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)")


def extract_first_doi(text: str) -> Optional[str]:
    """Return the first DOI found in *text*, or None."""
    if not text:
        return None
    m = DOI_URL_RE.search(text)
    if m:
        return m.group(1).rstrip(".")
    m = DOI_RE.search(text)
    if m:
        return m.group(1).rstrip(".")
    return None


def extract_first_doi_from_ocr_items(items: Optional[List[Dict[str, Any]]]) -> Optional[str]:
    """Scan paragraph texts in OCR block list (same structure as stem ``.json``).

    ``_to_markdown`` / merged page text can omit lines that still exist on
    ``parsing_res_list`` blocks; this recovers DOIs and similar identifiers.
    """
    if not items:
        return None
    parts: List[str] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        if it.get("typer") != "paragraph":
            continue
        text = (it.get("text") or "").strip()
        if text:
            parts.append(text)
    if not parts:
        return None
    return extract_first_doi("\n".join(parts))


def extract_doi_from_pdf_metadata(pdf_path: str) -> Optional[str]:
    """Try to extract DOI from the PDF metadata fields."""
    try:
        import fitz  # type: ignore
    except ImportError:
        return None
    try:
        doc = fitz.open(pdf_path)
        meta = doc.metadata or {}
        doc.close()
        for key in ("doi", "subject", "keywords", "title"):
            val = meta.get(key, "")
            if val:
                doi = extract_first_doi(val)
                if doi:
                    return doi
    except (RuntimeError, ValueError, KeyError) as exc:
        logger.debug("Could not read PDF metadata from %s: %s", pdf_path, exc)
    return None

