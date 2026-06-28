"""Tokenizer for extracting visual tokens and sentence tokens from KnowMat ocr_items.

Adapted from sci-align's ContentListTokenizer to work with KnowMat's ocr_items format.
"""

from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── figure ID patterns ─────────────────────────────────────────────────────

# Matches: Fig. 1 / Figure 2 / Figs. 3a,b / Fig 4b
# Subfigure letter must be at a word boundary (not part of a longer word like "XRD").
_FIG_RE = re.compile(
    r"\b(?:Fig(?:ure)?s?\.?\s*)(\d{1,3})\s*([a-z]\b(?:[,\s\-–][a-z]\b)*)?",
    re.IGNORECASE,
)

# Matches subfigure range: Fig. 3a–d  /  Fig 3(a-d)  /  Fig 3 a-d
_FIG_RANGE_RE = re.compile(
    r"\b(?:Fig(?:ure)?s?\.?\s*)(\d{1,3})\s*\(?\s*([a-z])\s*[–\-]\s*([a-z])\s*\)?",
    re.IGNORECASE,
)


def _expand_subfig_range(num: str, start: str, end: str) -> List[str]:
    """Expand 'fig_3a–d' into ['fig_3a', 'fig_3b', 'fig_3c', 'fig_3d']."""
    s, e = ord(start.lower()), ord(end.lower())
    if s > e:
        return [f"fig_{num}{start.lower()}"]
    return [f"fig_{num}{chr(c)}" for c in range(s, e + 1)]


def _parse_figure_ids(text: str) -> List[str]:
    """Return all normalised figure IDs found in text, e.g. ['fig_1', 'fig_2a']."""
    ids = []
    # First handle ranges (fig 3a–d) before generic matches consume them
    for m in _FIG_RANGE_RE.finditer(text):
        ids.extend(_expand_subfig_range(m.group(1), m.group(2), m.group(3)))
    # Then handle regular fig refs (non-range)
    for m in _FIG_RE.finditer(text):
        num = m.group(1)
        sub_raw = (m.group(2) or "").strip()
        sub_chars = re.findall(r"[a-z]", sub_raw.lower())
        if sub_chars:
            for ch in sub_chars:
                ids.append(f"fig_{num}{ch}")
        else:
            ids.append(f"fig_{num}")
    # deduplicate, preserve order
    seen: set = set()
    result = []
    for fid in ids:
        if fid not in seen:
            seen.add(fid)
            result.append(fid)
    return result


def _parse_primary_figure(caption: str) -> Tuple[Optional[str], Optional[int], Optional[str], List[str]]:
    """
    Parse primary figure ID from a caption string.
    Returns (normalized_figure_id, figure_number, subfigure_id, all_figure_ids).
    all_figure_ids expands ranges like Fig. 3a–d → [fig_3a, fig_3b, fig_3c, fig_3d].
    """
    # Check for range first
    rm = _FIG_RANGE_RE.search(caption)
    if rm:
        num = int(rm.group(1))
        start, end = rm.group(2).lower(), rm.group(3).lower()
        all_ids = _expand_subfig_range(str(num), start, end)
        norm = all_ids[0]  # primary = first subfig
        return norm, num, start, all_ids

    m = _FIG_RE.search(caption)
    if not m:
        return None, None, None, []
    num = int(m.group(1))
    sub_raw = (m.group(2) or "").strip()
    sub_chars = re.findall(r"[a-z]", sub_raw.lower())
    sub = sub_chars[0] if sub_chars else None
    norm = f"fig_{num}{sub or ''}"
    return norm, num, sub, [norm]


# ── sentence filters ───────────────────────────────────────────────────────

_REFERENCES_HEADER = re.compile(
    r"^\s*(?:References?|Bibliography|Works\s+Cited|Literature\s+Cited"
    r"|Acknowledgements?|Funding|Conflict\s+of\s+Interest)\s*$",
    re.IGNORECASE,
)

_METADATA_RE = re.compile(
    r"(?i)(?:received\s*:?\s*\d|accepted\s*:?\s*\d|published\s*:?\s*\d"
    r"|copyright\s*[©(]|©\s*20\d\d|cc\s+by|doi\s*:\s*10\."
    r"|correspondence\s*:|author\s+contributions?\s*:"
    r"|competing\s+interests?\s*:|conflict\s+of\s+interest"
    r"|supplementary\s+(?:data|material|information)"
    r"|e-?mail\s*:|orcid\s*:"
    r"|all\s+authors\s+(?:read|reviewed|approved|contributed)"
    r"|this\s+(?:research|study|work)\s+(?:was\s+)?(?:not\s+)?funded"
    r"|no\s+(?:funding|financial\s+support)"
    r"|data\s+availability|code\s+availability)",
)

_CITATION_RE = re.compile(
    r"\b[A-Z][a-z]+(?:\s+et\s+al\.|\s+and\s+[A-Z][a-z]+)?\s*\(\d{4}\b",
)

_BIBLIO_RE = re.compile(
    r"^[A-Z][a-z]{1,20}\s+[A-Z]{1,3}[\s,]",
)

_JOURNAL_CITE_RE = re.compile(
    r"^\s*[A-Z][A-Za-z\s\.]{2,40}\d+[\s(]\d*[):\s]+\d+",
)

_MIN_SENT_LEN = 50


def _is_reference_entry(text: str) -> bool:
    """True if sentence looks like a bibliography/reference list entry."""
    if _CITATION_RE.search(text) and _BIBLIO_RE.match(text.strip()):
        return True
    if len(_CITATION_RE.findall(text)) >= 2:
        return True
    if _JOURNAL_CITE_RE.match(text.strip()):
        return True
    return False


def _normalize_text(text: str) -> str:
    """Lowercase + collapse whitespace + unicode normalize for dedup."""
    return " ".join(unicodedata.normalize("NFKC", text).lower().split())


def _is_garbled(text: str) -> bool:
    """True if text has too many non-printable or non-Latin/CJK characters."""
    bad = sum(1 for c in text if not (c.isprintable() or c in "\n\t"))
    non_ascii = sum(
        1 for c in text if ord(c) > 127 and not ("一" <= c <= "鿿")
    )
    return (bad / max(len(text), 1) > 0.1) or (non_ascii / max(len(text), 1) > 0.35)


def _is_section_title_like(text: str) -> bool:
    """Heuristic: very short, no terminal period, mostly title-case words."""
    stripped = text.strip()
    if len(stripped) > 120 or stripped.endswith("."):
        return False
    words = stripped.split()
    if len(words) <= 3:
        return True
    upper_ratio = sum(1 for w in words if w and w[0].isupper()) / len(words)
    return upper_ratio >= 0.85


def _should_filter(text: str, in_references: bool) -> bool:
    """Return True if the sentence should be discarded."""
    if in_references:
        return True
    if len(text) < _MIN_SENT_LEN:
        return True
    if _is_garbled(text):
        return True
    if _is_section_title_like(text):
        return True
    if _METADATA_RE.search(text):
        return True
    if _is_reference_entry(text):
        return True
    return False


# ── sentence splitter ──────────────────────────────────────────────────────

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z一-鿿])")


def _split_sentences(text: str) -> List[str]:
    parts = _SENT_SPLIT.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def _text_hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


# ── Data classes ───────────────────────────────────────────────────────────

@dataclass
class VisualToken:
    """One figure image extracted from ocr_items."""

    token_id: str  # "{paper_id}::{figure_id}"
    paper_id: str
    figure_id: str
    image_path: str
    caption: str = ""
    figure_type: str = "Figure"
    bbox: Optional[List[float]] = None
    page_number: Optional[int] = None
    # Parsed from caption text
    normalized_figure_id: Optional[str] = None  # e.g. "fig_1", "fig_2a"
    figure_number: Optional[int] = None          # e.g. 1, 2
    subfigure_id: Optional[str] = None           # e.g. "a", "b", None
    all_figure_ids: List[str] = field(default_factory=list)  # e.g. ["fig_3a","fig_3b","fig_3c"]

    @classmethod
    def make_id(cls, paper_id: str, figure_id: str) -> str:
        return f"{paper_id}::{figure_id}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "token_id": self.token_id,
            "paper_id": self.paper_id,
            "figure_id": self.figure_id,
            "image_path": self.image_path,
            "caption": self.caption,
            "figure_type": self.figure_type,
            "bbox": self.bbox,
            "page_number": self.page_number,
            "normalized_figure_id": self.normalized_figure_id,
            "figure_number": self.figure_number,
            "subfigure_id": self.subfigure_id,
            "all_figure_ids": self.all_figure_ids,
        }


@dataclass
class SentenceToken:
    """One sentence extracted from paper text."""

    token_id: str  # "{paper_id}::sent::{idx}"
    paper_id: str
    text: str
    source: str  # "paragraph" | "table_caption" | "image_caption"
    page_number: Optional[int] = None
    section: Optional[str] = None
    # Parsed figure references from sentence text
    mentioned_figures: List[str] = field(default_factory=list)  # e.g. ["fig_1", "fig_2a"]
    has_figure_reference: bool = False

    @classmethod
    def make_id(cls, paper_id: str, idx: int) -> str:
        return f"{paper_id}::sent::{idx}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "token_id": self.token_id,
            "paper_id": self.paper_id,
            "text": self.text,
            "source": self.source,
            "page_number": self.page_number,
            "section": self.section,
            "mentioned_figures": self.mentioned_figures,
            "has_figure_reference": self.has_figure_reference,
        }


# ── Tokenizer ─────────────────────────────────────────────────────────────

class OcrItemTokenizer:
    """
    Tokenize KnowMat ocr_items into VisualToken + SentenceToken lists.
    """

    VISUAL_TYPES = {"image", "chart", "figure"}
    SKIP_TYPES = {"header", "footer", "page_number", "aside_text", "page_footnote"}

    def __init__(self, min_caption_len: int = 5) -> None:
        self.min_caption_len = min_caption_len

    def tokenize(
        self,
        ocr_items: List[Dict[str, Any]],
        paper_id: str,
        output_dir: Optional[str] = None,
    ) -> Tuple[List[VisualToken], List[SentenceToken]]:
        """
        Extract visual and sentence tokens from KnowMat ocr_items.

        Parameters
        ----------
        ocr_items : List[Dict[str, Any]]
            The ocr_items list from KnowMat's parse_pdf node.
        paper_id : str
            The paper identifier.
        output_dir : Optional[str]
            The output directory for resolving relative image paths.

        Returns
        -------
        Tuple[List[VisualToken], List[SentenceToken]]
            visual_tokens and sentence_tokens
        """
        visual_tokens: List[VisualToken] = []
        sentence_tokens: List[SentenceToken] = []
        seen_hashes: set = set()
        seen_norm: set = set()
        vis_idx = 0
        sent_idx = 0
        in_references = False
        current_section: Optional[str] = None

        # Resolve output directory for image paths
        base_dir = Path(output_dir) if output_dir else Path.cwd()

        for item in ocr_items:
            if not isinstance(item, dict):
                continue

            itype = item.get("typer", "")
            page = item.get("page")

            # ── Track section heading (block_label == "title" or text_level) ─────
            if itype == "paragraph" and item.get("block_label") == "title":
                heading_text = item.get("text", "").strip()
                current_section = heading_text
                if _REFERENCES_HEADER.match(heading_text):
                    in_references = True
                    logger.debug("Entering references section at: %r", heading_text)
                else:
                    in_references = False
                continue

            # ── Visual items ─────────────────────────────────────────────
            if itype in self.VISUAL_TYPES:
                data = item.get("data") or {}
                caption = data.get("caption", "")
                image_path = data.get("image_path", "")

                if not caption or len(caption) < self.min_caption_len:
                    continue

                # Resolve image path
                if image_path and not Path(image_path).is_absolute():
                    image_path = str(base_dir / image_path)

                norm_fid, fig_num, sub_id, all_fids = _parse_primary_figure(caption)
                tid = VisualToken.make_id(paper_id, f"{itype}_{vis_idx:04d}")
                figure_type = item.get("block_label", itype.capitalize())
                if figure_type == "figure" or figure_type == "chart":
                    figure_type = "Figure"

                visual_tokens.append(
                    VisualToken(
                        token_id=tid,
                        paper_id=paper_id,
                        figure_id=f"{itype}_{vis_idx:04d}",
                        image_path=image_path,
                        caption=caption,
                        figure_type=figure_type,
                        bbox=item.get("bbox"),
                        page_number=page,
                        normalized_figure_id=norm_fid,
                        figure_number=fig_num,
                        subfigure_id=sub_id,
                        all_figure_ids=all_fids,
                    )
                )
                vis_idx += 1

            # ── Text items ────────────────────────────────────────────
            elif itype == "paragraph" and itype not in self.SKIP_TYPES:
                raw = item.get("text", "").strip()
                if not raw:
                    continue

                # Skip title paragraphs (handled separately for section tracking)
                if item.get("block_label") == "title":
                    continue

                for sent in _split_sentences(raw):
                    if _should_filter(sent, in_references):
                        continue
                    # Exact dedup
                    h = _text_hash(sent)
                    if h in seen_hashes:
                        continue
                    # Normalized text dedup
                    norm = _normalize_text(sent)
                    if norm in seen_norm:
                        continue
                    seen_hashes.add(h)
                    seen_norm.add(norm)

                    # Parse figure mentions
                    fig_mentions = _parse_figure_ids(sent)
                    sentence_tokens.append(
                        SentenceToken(
                            token_id=SentenceToken.make_id(paper_id, sent_idx),
                            paper_id=paper_id,
                            text=sent,
                            source="paragraph",
                            page_number=page,
                            section=current_section,
                            mentioned_figures=fig_mentions,
                            has_figure_reference=bool(fig_mentions),
                        )
                    )
                    sent_idx += 1

            # ── Table items (extract captions as sentences) ───────────────
            elif itype == "table":
                data = item.get("data") or {}
                table_text = data.get("text", "")
                caption = data.get("caption", "")

                # Process table text
                for sent in _split_sentences(table_text):
                    if _should_filter(sent, in_references):
                        continue
                    h = _text_hash(sent)
                    if h in seen_hashes:
                        continue
                    norm = _normalize_text(sent)
                    if norm in seen_norm:
                        continue
                    seen_hashes.add(h)
                    seen_norm.add(norm)

                    fig_mentions = _parse_figure_ids(sent)
                    sentence_tokens.append(
                        SentenceToken(
                            token_id=SentenceToken.make_id(paper_id, sent_idx),
                            paper_id=paper_id,
                            text=sent,
                            source="table_content",
                            page_number=page,
                            section=current_section,
                            mentioned_figures=fig_mentions,
                            has_figure_reference=bool(fig_mentions),
                        )
                    )
                    sent_idx += 1

                # Process table caption
                if caption:
                    h = _text_hash(caption)
                    if h not in seen_hashes:
                        norm = _normalize_text(caption)
                        if norm not in seen_norm:
                            seen_hashes.add(h)
                            seen_norm.add(norm)

                            fig_mentions = _parse_figure_ids(caption)
                            sentence_tokens.append(
                                SentenceToken(
                                    token_id=SentenceToken.make_id(paper_id, sent_idx),
                                    paper_id=paper_id,
                                    text=caption,
                                    source="table_caption",
                                    page_number=page,
                                    section=current_section,
                                    mentioned_figures=fig_mentions,
                                    has_figure_reference=bool(fig_mentions),
                                )
                            )
                            sent_idx += 1

        # Post-process: mark sentences that are substantially caption text
        caption_norms = {_normalize_text(vt.caption) for vt in visual_tokens if vt.caption}
        for st in sentence_tokens:
            if st.source in ("paragraph", "table_content") and self._caption_overlap(
                _normalize_text(st.text), caption_norms
            ):
                st.source = "caption_text"

        n_caption_text = sum(1 for s in sentence_tokens if s.source == "caption_text")
        n_fig_ref = sum(1 for s in sentence_tokens if s.has_figure_reference)
        logger.info(
            "paper=%s  visual=%d  sentences=%d  (with_fig_ref=%d  caption_text=%d)",
            paper_id,
            len(visual_tokens),
            len(sentence_tokens),
            n_fig_ref,
            n_caption_text,
        )
        return visual_tokens, sentence_tokens

    @staticmethod
    def _caption_overlap(sent_norm: str, caption_norms: set, threshold: float = 0.75) -> bool:
        """True if sentence is substantially contained in or overlapping with any caption."""
        if len(sent_norm) < 20:
            return False
        sw = set(sent_norm.split())
        for cap_norm in caption_norms:
            if not cap_norm or len(cap_norm) < 5:
                continue
            if sent_norm in cap_norm:
                return True
            cw = set(cap_norm.split())
            union = sw | cw
            if union and len(sw & cw) / len(union) >= threshold:
                return True
        return False
