"""Pipeline C: CLIP alignment + VLM figure enrichment → enriched _final.md.

This module is now a thin wrapper around the main extraction-path enrichment
in :func:`ppmatAgent.knowmat.pdf.figure_describer.inject_figure_descriptions`.  Both the
``--batch`` full pipeline (via the extraction node) and the ``--final-md`` mode
(via :mod:`ppmatAgent.knowmat.batch.finalmd_pipeline` / :mod:`ppmatAgent.knowmat.batch.enrich_runner`)
therefore share ONE enrichment implementation, including the chart-digitization
routing (chart_box crops → ``VLM-digitized`` blocks; everything else → prose
``AI Description`` blocks).

Public API
----------
enrich_paper_text(paper_id, raw_dir, vlm_workers=1) -> str | None
    Load raw OCR (.md + .json) → inject figure descriptions / chart
    digitizations → return enriched text. None if source files are missing.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional


def _clean_html(text: str) -> str:
    """Strip decorative HTML tags; preserve table and formula markup."""
    text = re.sub(r'</?div[^>]*>', '', text)
    text = re.sub(r'</?span[^>]*>', '', text)
    text = re.sub(r'</?p[^>]*>', '', text)
    text = re.sub(r'<br\s*/?>', '\n', text)
    text = re.sub(r'<img[^>]*/?>', '', text)
    text = re.sub(r'<a[^>]*>(.*?)</a>', r'\1', text)
    text = re.sub(r'</?sup>', '', text)
    text = re.sub(r'</?sub>', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text


def enrich_paper_text(
    paper_id: str,
    raw_dir: Path,
    vlm_workers: int = 1,
) -> Optional[str]:
    """Full enrichment via the shared main-path injector.

    Steps:
      1. Load ``raw_dir/paper_id/{paper_id}.md`` and ``.json``.
      2. Delegate to ``inject_figure_descriptions`` — this performs figure
         caption/image pairing, CLIP alignment (when available), and routes
         each figure to chart digitization or prose description.
      3. Clean residual decorative HTML.

    Returns the enriched text, or None if the OCR source files are missing.
    """
    paper_dir = raw_dir / paper_id
    md_path = paper_dir / f"{paper_id}.md"
    json_path = paper_dir / f"{paper_id}.json"

    if not md_path.exists() or not json_path.exists():
        print(f"  [{paper_id}] Missing .md or .json in {paper_dir}")
        return None

    paper_text = md_path.read_text(encoding="utf-8")
    ocr_items = json.loads(json_path.read_text(encoding="utf-8"))

    from ppmatAgent.knowmat.pdf.figure_describer import inject_figure_descriptions

    enriched = inject_figure_descriptions(
        paper_text,
        ocr_items,
        max_workers=max(1, vlm_workers),
        paper_id=paper_id,
        output_dir=str(paper_dir),
    )
    return _clean_html(enriched)
