"""Chart digitization for KnowMat (line/bar charts → structured VLM-digitized block).

Companion to :mod:`ppmatAgent.knowmat.pdf.figure_describer`.  Where ``figure_describer``
produces a *prose* description (200-500 words) used as soft context only, this
module produces a *structured* digitization that the extraction stage is
allowed to consume as data (under a dedicated whitelist rule keyed on the
``VLM-digitized`` marker).

Two-stage triage:
  1. Cheap routing on the OCR crop filename: PaddleOCR-VL names figure crops
     ``img_in_chart_box_*`` (plottable data) vs ``img_in_image_box_*``
     (micrograph/photo).  Only ``chart_box`` crops are candidates here.
  2. A VLM classification gate confirms the chart type and chooses an output
     contract:
       - bar/histogram  → CSV of discrete bar heights (directly readable)
       - line/scatter   → key-points + trend summary ONLY (axis vars, series
         labels, monotonicity, start/end endpoints, extrema).  We deliberately
         do NOT emit a per-point CSV for continuous curves: VLMs fabricate
         evenly-spaced points and smooth away non-monotonic features.
       - xrd/micrograph/other → skipped.

Like ``figure_describer`` it never raises — failures log and return empty so
the main pipeline is never blocked.  It reuses the VLM key pool / retry from
``figure_describer``.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from ppmatAgent.knowmat.pdf.figure_describer import (
    _get_vlm_pool,
    _image_media_type,
    _vlm_call_with_pool,
)

logger = logging.getLogger(__name__)

_CHART_BOX_TOKEN = "chart_box"
_IMAGE_BOX_TOKEN = "image_box"

# Enum of chart types the VLM may return; anything else is coerced to "other".
_VALID_TYPES = {"line", "bar", "xrd", "micrograph", "other"}

_CHART_PROMPT = """You are a scientific chart reader. Look at this cropped figure image.

STEP 1 — Classify. Output exactly one of these enum values for "type":
  line      : line chart / scatter plot with continuous numeric X and Y axes
  bar       : bar / column chart or histogram (categorical or binned X axis)
  xrd       : diffraction pattern / spectrum (intensity vs 2-theta, sharp peaks)
  micrograph: SEM/TEM/optical image, photo, schematic, or map (NOT a data plot)
  other     : phase diagram, flowchart, multi-panel mix, unreadable, or none of the above
If a crop contains MULTIPLE panels of different kinds, pick the single
dominant data-plot type if one clearly dominates, else "other".

STEP 2 — Read the chart according to its type. Two DIFFERENT contracts:

  IF type == "bar":
    Digitize to CSV (discrete, directly-readable bar heights).
    - One CSV table, first row = header.
    - Encode variable + unit in each header (e.g. "Lamellar_Thickness_nm").
    - Each series = its own column; preserve series labels.
    - One row per bar / bin. Read the bar heights off the Y axis.
    Put the CSV string in "csv". Set "line_summary" to null.

  IF type == "line":
    DO NOT output a per-point CSV. Reading exact point coordinates off a
    continuous curve produces fabricated, evenly-spaced, over-smoothed data.
    Extract ONLY what is reliably visible, into "line_summary":
    - x_axis / y_axis : variable + unit (e.g. "Temperature (C)")
    - series          : list of curve labels (e.g. ["theta=0", "theta=45"])
    - per series: monotonic (true/false), start point [x,y], end point [x,y],
      and any local extrema (peaks/valleys) as {point:[x,y], kind, note}.
    - ONLY include numbers you can actually read from axis ticks. Use null for
      anything you cannot read. NEVER invent intermediate points.
    Set "csv" to "".

  IF type in (xrd, micrograph, other): digitizable=false, "csv"="", "line_summary"=null.

confidence = how readable/reliable your extraction is (0.0-1.0).

Return STRICT JSON only (no markdown fences, no prose, no <think>):
{
  "type": "line|bar|xrd|micrograph|other",
  "digitizable": true,
  "confidence": 0.0,
  "reason": "one short sentence",
  "csv": "",
  "line_summary": {
    "x_axis": "", "y_axis": "",
    "series": [
      {"label": "", "monotonic": true,
       "start": [null, null], "end": [null, null],
       "extrema": [{"point": [null, null], "kind": "peak", "note": ""}]}
    ]
  }
}"""


def is_chart_crop(image_path: str) -> bool:
    """True if the OCR crop filename marks it as a plottable chart."""
    return _CHART_BOX_TOKEN in Path(str(image_path or "")).name


def is_micrograph_crop(image_path: str) -> bool:
    """True if the OCR crop filename marks it as a micrograph/photo."""
    return _IMAGE_BOX_TOKEN in Path(str(image_path or "")).name


def _encode_image_base64(image_path: Path) -> Optional[str]:
    try:
        return base64.b64encode(image_path.read_bytes()).decode("utf-8")
    except OSError as exc:
        logger.warning("Cannot read chart image %s: %s", image_path, exc)
        return None


def _parse_vlm_json(raw: str) -> Optional[Dict[str, Any]]:
    """Best-effort parse of the VLM JSON response."""
    text = str(raw or "").strip()
    if not text:
        return None
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I).strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, re.S)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return None
    return None


def digitize_chart_image(
    image_path: Path,
    caption: str = "",
    *,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Classify + digitize a single chart crop. Returns parsed dict or None.

    The returned dict has keys: type, digitizable, confidence, reason, csv,
    line_summary.  ``type`` is coerced into the valid enum (else "other").
    Returns None on hard failure (unreadable image, no keys, VLM error).
    """
    resolved = Path(image_path)
    if not resolved.is_file():
        logger.debug("Chart image not found, skipping digitization: %s", image_path)
        return None

    b64 = _encode_image_base64(resolved)
    if b64 is None:
        return None

    _api_key = api_key or _get_vlm_pool().next_key()
    _base_url = (
        base_url
        or os.getenv("VLM_BASE_URL")
        or os.getenv("LLM_BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
    )
    _model = model or os.getenv("VLM_MODEL") or os.getenv("LLM_MODEL", "")
    if not _api_key:
        logger.warning("No VLM_API_KEY/LLM_API_KEY configured; skipping chart digitization.")
        return None

    user_text = _CHART_PROMPT
    if caption:
        user_text = f"Caption: {caption}\n\n{user_text}"

    media_type = _image_media_type(resolved)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{media_type};base64,{b64}"},
                },
                {"type": "text", "text": user_text},
            ],
        },
    ]
    create_kwargs: dict = {
        "model": _model,
        "messages": messages,
        "max_tokens": 2048,
        "temperature": 0.1,
    }

    try:
        raw = _vlm_call_with_pool(_base_url, create_kwargs, image_path)
    except Exception as exc:
        logger.warning("Chart digitization VLM call failed for %s: %s", image_path, exc)
        return None

    parsed = _parse_vlm_json(raw)
    if not parsed:
        logger.warning("Chart digitization JSON parse failed for %s", image_path)
        return None

    ctype = str(parsed.get("type") or "").strip().lower()
    if ctype not in _VALID_TYPES:
        # Coerce non-enum (e.g. "bar|line", "multi-panel mix") to "other" so
        # neither contract fires on an ambiguous crop.
        logger.debug("Chart type '%s' not a valid enum for %s; coercing to 'other'.", ctype, image_path)
        ctype = "other"
    parsed["type"] = ctype
    return parsed


def _fmt_pt(pt: Any) -> str:
    if isinstance(pt, (list, tuple)) and len(pt) == 2:
        return f"[{pt[0]}, {pt[1]}]"
    return str(pt)


def format_digitized_block(result: Dict[str, Any], figure_num: str = "") -> str:
    """Render a VLM-digitized injection block, or "" if nothing usable.

    Output is plain text under the dedicated ``VLM-digitized`` marker so the
    extraction prompt whitelist can target it.  bar → CSV; line → key-points
    + trend summary.  Everything else returns "".
    """
    if not isinstance(result, dict):
        return ""
    ctype = result.get("type")
    label = f"Figure {figure_num}" if figure_num else "Figure"

    if ctype == "bar":
        csv = str(result.get("csv") or "").strip()
        if not csv:
            return ""
        return f"> [{label} VLM-digitized | bar chart, estimated from pixels]:\n{csv}"

    if ctype == "line":
        ls = result.get("line_summary")
        if not isinstance(ls, dict):
            return ""
        lines: List[str] = [
            "chart_type: line (estimated from pixels — key points & trend only, NOT exact data)",
            f"x_axis: {ls.get('x_axis')}",
            f"y_axis: {ls.get('y_axis')}",
        ]
        series = ls.get("series") or []
        if not series:
            return ""
        for s in series:
            if not isinstance(s, dict):
                continue
            extrema = s.get("extrema") or []
            ex_str = "; ".join(
                f"{e.get('kind')}@{_fmt_pt(e.get('point'))}"
                + (f" ({e.get('note')})" if e.get("note") else "")
                for e in extrema
                if isinstance(e, dict)
            )
            lines.append(
                f"series {s.get('label')}: monotonic={s.get('monotonic')}, "
                f"start={_fmt_pt(s.get('start'))}, end={_fmt_pt(s.get('end'))}"
                + (f", extrema=[{ex_str}]" if ex_str else "")
            )
        return f"> [{label} VLM-digitized | line chart, estimated from pixels]:\n" + "\n".join(lines)

    return ""
