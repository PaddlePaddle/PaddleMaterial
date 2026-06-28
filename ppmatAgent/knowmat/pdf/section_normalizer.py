"""Section/title normalization and noise filtering for OCR text."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Tuple

from .blocks import strip_paddle_vl_block_artifacts
from .doi_extractor import DOI_RE, DOI_URL_RE
from .heading_detector import _title_looks_like_figure_chart_axis, detect_heading

# ---------------------------------------------------------------------------
# Figure schematic noise (scale bars, axes) and author/affiliation formatting
# ---------------------------------------------------------------------------

# Micro sign U+00B5, Greek mu U+03BC, common OCR confusions
_UM_UNIT_FRAGMENT = r"(?:µm|μm|\u00b5m|\u03bcm|um)"
_UNIT_ONLY_TITLE_RE = re.compile(
    r"^(?:"
    r"\u00b5m|\u03bcm|µm|μm|um|nm|pm|mm|cm|°C|K"
    r")\s*$",
    re.IGNORECASE,
)

# Stop author/affiliation normalization before journal meta / abstract (not author list).
_AUTHOR_FRONT_MATTER_END_RE = re.compile(
    r"^#+\s*(?:"
    r"ABSTRACT|ARTICLE\s+INFO|Keywords?|INTRODUCTION|\d+\.\s+Introduction"
    r")\b",
    re.IGNORECASE,
)


def _strip_leading_heading_marks(line: str) -> str:
    return re.sub(r"^#+\s*", "", line.strip())


def _normalize_spaces_for_noise(line: str) -> str:
    s = _strip_leading_heading_marks(line)
    s = re.sub(r"[\u00a0\u1680\u2000-\u200b\u202f\u205f\u3000\ufeff]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _is_unit_only_section_title(title: str) -> bool:
    t = _normalize_spaces_for_noise(title)
    return bool(t and _UNIT_ONLY_TITLE_RE.match(t))


def _is_spurious_generic_numeric_section(sec_num: str, title: str) -> bool:
    """Reject ``SECTION_PATTERNS`` matches that are chart ticks (e.g. ``200. 250``, ``3.44853. A``)."""
    if _title_looks_like_figure_chart_axis(title):
        return True
    sn = sec_num.strip().rstrip(".")
    t = title.strip()
    low = t.lower()
    if not t:
        return True
    if t.isdigit():
        if len(t) >= 2:
            return True
        if sn.isdigit() and int(sn) >= 10:
            return True
    if sn.isdigit() and t.isdigit() and int(sn) >= 15 and int(t) >= 10:
        return True
    if re.search(r"\d+\.\d{3,}", sn):
        if len(t) <= 8 and re.match(r"^[A-Za-z%°ÅΩµμm\-\.]+$", t, re.I):
            return True
    if t in ("%", "-", "A", "Å") and re.search(r"\d+\.\d+", sn):
        return True
    # LaTeX / math / table fragments misread as "N. Title" by wide SECTION_PATTERNS
    if "\\" in t or "×" in t or ("−" in t and "m" in low):
        return True
    if re.match(r"^K[, ]", t) and sn.isdigit() and len(sn) >= 3 and 273 <= int(sn) <= 5000:
        return True
    if t.lower().startswith("(si)") and len(t) < 80:
        return True
    if re.match(r"^(?:MPa|GPa|kPa|pim|µm|μm|nm)\s*$", t, re.I):
        return True
    return False


_NAME_WITH_PLAIN_AFFILIATION_RE = re.compile(
    r"(?P<name>(?:[A-Z]\.){1,3}\s+[A-Z][a-z]+|[A-Z][a-z]+(?:\s+[A-Z][a-z\.]+)+)"
    r"\s+(?P<aff>[a-z\*](?:,[a-z\*])*(?:,\*)?\*?)"
    r"(?=\s*[,;]|\s+\*|\s+(?:[A-Z]\.|[A-Z][a-z])|\s*$)"
)


def _merge_hanging_affiliation_suffix_lines(lines: List[str]) -> List[str]:
    """Join 'Weihua Wang' + next line 'b,f' split by OCR."""
    out: List[str] = []
    i = 0
    while i < len(lines):
        cur = lines[i].rstrip()
        nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
        if (
            nxt
            and re.fullmatch(r"[a-z](?:,[a-z])+\*?\s*", nxt, re.I)
            and not re.match(r"^[a-z]\s+[A-Z]", cur.strip(), re.I)
            and re.search(r"[A-Za-z]", cur)
        ):
            out.append(f"{cur.strip()} {nxt}")
            i += 2
            continue
        out.append(lines[i])
        i += 1
    return out


def _apply_plain_author_affiliation_superscripts(line: str) -> str:
    """Turn 'Zhang a,b,' into 'Zhang $^{a,b}$,' for OCR author lines (no HTML <sup>)."""

    def repl(m: re.Match[str]) -> str:
        return f"{m.group('name')} $^{{{m.group('aff')}}}$"

    return _NAME_WITH_PLAIN_AFFILIATION_RE.sub(repl, line)


def _normalize_affiliation_institution_line(line: str) -> str:
    """Format 'a Research Institute...' → '$^{a}$ Research Institute...'; fix 'cDepartment'."""
    s = line.strip()
    m = re.match(r"^([a-z])\s+(\S.*)$", s, re.I)
    if m and m.group(2) and m.group(2)[0].isupper():
        letter = m.group(1).lower()
        return f"$^{{{letter}}}$ {m.group(2)}"
    m2 = re.match(r"^([a-z])([A-Z][a-z].+)$", s)
    if m2:
        return f"$^{{{m2.group(1).lower()}}}$ {m2.group(2)}"
    return line


def _process_author_front_matter_line(line: str) -> str:
    st = line.strip()
    if not st:
        return line
    if re.match(r"^[a-z]\s+[A-Z]", st, re.I):
        return _normalize_affiliation_institution_line(line)
    if re.match(r"^[a-z][A-Z][a-z]", st):
        return _normalize_affiliation_institution_line(line)
    if "*" in st or re.search(r"[a-z\*](?:,[a-z])", st, re.IGNORECASE):
        return _apply_plain_author_affiliation_superscripts(line)
    return line


def normalize_plain_author_superscripts(text: str) -> str:
    """Normalize plain-text author markers and affiliation prefixes before Abstract.

    Converts trailing ' a,b' on names to ``$^{a,b}$`` and ``a Institution`` lines
    to ``$^{a}$ Institution`` (Elsevier-style OCR without HTML <sup>).
    """
    lines = text.split("\n")
    end = len(lines)
    for i, ln in enumerate(lines):
        if _AUTHOR_FRONT_MATTER_END_RE.match(ln.strip()):
            end = i
            break
    if end <= 0:
        return text
    head = _merge_hanging_affiliation_suffix_lines(lines[:end])
    head = [_process_author_front_matter_line(ln) for ln in head]
    return "\n".join(head + lines[end:])

# ---------------------------------------------------------------------------
# Chemical element symbols (ordered longest-first to avoid prefix ambiguity).
# Loaded from ``knowmat/data/elements.json`` when present; else built-in tuple.
# ---------------------------------------------------------------------------
_BUILTIN_ELEMENT_SYMBOLS: Tuple[str, ...] = (
    "Uut", "Uup", "Uus", "Uuo",
    "He", "Li", "Be", "Ne", "Na", "Mg", "Al", "Si", "Cl", "Ar",
    "Ca", "Sc", "Ti", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Zr", "Nb",
    "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb",
    "Te", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm",
    "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf",
    "Ta", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi",
    "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "Np", "Pu",
    "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf",
    "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Fl", "Lv",
    "B", "C", "N", "O", "F", "P", "S", "K", "V", "Y", "I", "W",
    "U", "Y",
)


def _load_element_symbols() -> Tuple[str, ...]:
    path = Path(__file__).resolve().parent.parent / "data" / "elements.json"
    try:
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list) and data:
                return tuple(str(x) for x in data)
    except (OSError, json.JSONDecodeError, TypeError):
        pass
    return _BUILTIN_ELEMENT_SYMBOLS


_ELEMENT_SYMBOLS = _load_element_symbols()
_ELEMENT_PAT = "|".join(re.escape(e) for e in _ELEMENT_SYMBOLS)

# Pattern: one or more (Element)(number with optional decimal) pairs, e.g.
# Ti42Hf21Nb21V16  or  Ni61.3Cr25.3  or  Ni61,3Cr25,3 (comma as decimal)
# The [.,] allows both period and comma as decimal separators; the comma
# variant is normalised to period inside _normalize_alloy_formula.
# NOTE: This requires at least 2 element-number pairs to avoid false positives
# on words like "Ti6Al4V" alloy names that might be misread.
_ALLOY_FORMULA_RE = re.compile(
    r"\b((?:(?:" + _ELEMENT_PAT + r")\d+(?:[.,]\d+)?){2,})\b"
)

# Single element-number pattern for cases like "Nb15" appearing alone.
# Used only when the line has clear materials science context.
_SINGLE_ELEMENT_NUM_RE = re.compile(
    r"\b((" + _ELEMENT_PAT + r")(\d+(?:[.,]\d+)?))\b"
)

# Parenthesized alloy formula with subscript: (Nb15Ta10W75)98.5C1.5
# Pattern: (element-number pairs) followed by optional subscript and more elements
_PAREN_ALLOY_RE = re.compile(
    r"\(((" + _ELEMENT_PAT + r")\d+(?:[.,]\d+)?"
    r"(?:(" + _ELEMENT_PAT + r")\d+(?:[.,]\d+)?)*"
    r")\)(\d+(?:[.,]\d+)?)((" + _ELEMENT_PAT + r")\d+(?:[.,]\d+)?)*"
)

# Keywords that indicate materials science / chemistry context
_MATERIALS_CONTEXT_KEYWORDS = re.compile(
    r"\b(?:alloy|alloys|steel|steels|metal|metals|ceramic|composite|phase|phases|"
    r"precipitate|precipitates|microstructure|grain|grains|matrix|solid|liquid|"
    r"atom|atoms|mole|moles|atomic|molar|composition|element|elements|compound|"
    r"compounds|solution|solutions|BCC|FCC|HCP|crystal|crystalline|amorphous|"
    r"yield|strength|hardness|ductility|tensile|compression|MPa|GPa|temperature|"
    r"thermal|melting|solidification|NbMoTaW|HEA|MPEA|RHEA|RMPEA|high-entropy|"
    r"refractory|superalloy|intermetallic|conductivity|bandgap|anode|cathode|"
    r"battery|semiconductor|polymer|doping|dielectric|piezoelectric|magnetic|"
    r"optical|catalyst|electrolyte|titanium|niobium|tantalum|tungsten|molybdenum|"
    r"hafnium|vanadium|chromium|nickel|cobalt|iron|aluminum|zirconium)\b",
    re.IGNORECASE
)

# Decimal normalisation: digit followed by Chinese comma / full-width period /
# middle dot, followed by digit  →  replace separator with ASCII period.
# (ASCII comma is handled separately inside chemical formula contexts.)
_DECIMAL_NOISE_RE = re.compile(r"(\d)[，、·。](\d)")

ALLOY_OCR_FIXES: List[Tuple[re.Pattern, str]] = []

SPACED_TITLE_RE = re.compile(r"^(#+\s+)?([A-Z](?:\s[A-Z]){3,})$")

SECTION_PATTERNS: List[Tuple[re.Pattern, str | None]] = [
    (re.compile(r"^(?:#+\s*)?A\s*B\s*S\s*T\s*R\s*A\s*C\s*T\s*$", re.IGNORECASE), "## ABSTRACT"),
    (re.compile(r"^(?:#+\s*)?ABSTRACT\s*$", re.IGNORECASE), "## ABSTRACT"),
    (re.compile(r"^(?:#+\s*)?ARTICLE\s+INFO\s*$", re.IGNORECASE), "## ARTICLE INFO"),
    (re.compile(r"^(?:#+\s*)?KEYWORDS?\s*:?\s*$", re.IGNORECASE), "## Keywords"),
    (re.compile(r"^(?:#+\s*)?(ACKNOWLEDGEMENTS?|ACKNOWLEDGMENTS?)\s*$", re.IGNORECASE), "## Acknowledgements"),
    (re.compile(r"^(?:#+\s*)?CRediT\s+authorship\s+contribution\s+statement\s*$", re.IGNORECASE), "## CRediT Authorship"),
    (re.compile(r"^(?:#+\s*)?DECLARATION\s+OF\s+(COMPETING\s+)?INTEREST.*$", re.IGNORECASE), "## Declaration of Interest"),
    (re.compile(r"^(?:#+\s*)?DATA\s+AVAILABILITY.*$", re.IGNORECASE), "## Data Availability"),
    (re.compile(r"^(?:#+\s*)?SUPPLEMENTARY\s+(DATA|MATERIAL|INFORMATION).*$", re.IGNORECASE), "## Supplementary Material"),
    (re.compile(r"^(?:#+\s*)?(\d+(?:\.\d+)*\.?)\s+(.+)$"), None),
]

NOISE_LINE_PATTERNS: List[Tuple[re.Pattern, bool]] = [
    (re.compile(r"^Contents lists available at", re.IGNORECASE), False),
    (re.compile(r"^Available online", re.IGNORECASE), False),
    (re.compile(r"^Received \d+", re.IGNORECASE), False),
    (re.compile(r"^Accepted \d+", re.IGNORECASE), False),
    (re.compile(r"^journal homepage", re.IGNORECASE), False),
    (re.compile(r"^https?://www\.(sciencedirect|elsevier|springer|wiley|nature|science|acs)", re.IGNORECASE), False),
    (re.compile(r"^Full length article\s*$", re.IGNORECASE), False),
    (re.compile(r"^\d{4}-\d{3}[\dX]/\s*�", re.IGNORECASE), False),
    (re.compile(r"^�\s*\d{4}", re.IGNORECASE), False),
    (re.compile(r"^(Elsevier|Springer|Wiley|ACS)", re.IGNORECASE), False),
    (re.compile(r"^\* Corresponding author", re.IGNORECASE), False),
    (re.compile(r"^\*\* Corresponding author", re.IGNORECASE), False),
    (re.compile(r"^E-mail address", re.IGNORECASE), False),
    (re.compile(r"^\d+$"), False),
    (re.compile(r"^[A-Za-z]:\\"), False),
    (re.compile(r"^/tmp/"), False),
    (re.compile(r"^min\s*$", re.IGNORECASE), False),
    (re.compile(r"^general\s*$", re.IGNORECASE), False),
    (re.compile(r"^(STRUCTURE|PROPERTIES|MODELLING|MODELIINC|PROCESSING)\s*$"), False),
    (re.compile(r"^Check\s+for\s*$", re.IGNORECASE), False),
    (re.compile(r"^updates?\s*$", re.IGNORECASE), False),
    (re.compile(r"^[A-Z]\.\s+\w+\s+et\s+al\.?\s*$", re.IGNORECASE), False),
    (re.compile(r"^(obs|diff|cal)\.?\s*$", re.IGNORECASE), False),
    (re.compile(r"^[=\\-]\s*$"), False),
    (re.compile(r"^\\leftarrow\s*$"), False),
    (re.compile(r"^\d+\.\d{1,8}\s*$"), False),
    (re.compile(r"^[A-Za-z]\s*$"), False),
    (re.compile(r"^Volume density", re.I), False),
    (re.compile(r"^Cumulative percent", re.I), False),
    (re.compile(r"^Particle size", re.I), False),
    (re.compile(r"^Scanning strategy", re.I), False),
    (re.compile(r"^Kernel Aurer|Misorient|KeroeI Aver", re.I), False),
    (re.compile(r"^D\d+\s*=", re.I), False),
    (re.compile(r"^G\(r\)\s*$", re.I), False),
    (re.compile(r"^O≤\s*\d+", re.I), False),
    (re.compile(r"^\d+\.\s+\d+\s*$"), False),
    (re.compile(r"^\\(?:tau|rho|xi|alpha|lambda|theta|phi)\s*$"), False),
    (re.compile(r"^\\xi\\alpha\s*$"), False),
    (re.compile(r"^p/q\s*$", re.I), False),
    (re.compile(r"^V0\s*$", re.I), False),
]

# ---------------------------------------------------------------------------
# Journal masthead / page-header OCR (Elsevier MSEA and similar)
# ---------------------------------------------------------------------------
_MASTHEAD_LINE_PATTERNS: List[re.Pattern[str]] = [
    # General publisher / journal UI fragments
    re.compile(r"^Contents lists available at", re.IGNORECASE),
    re.compile(r"^(ELSEVIER|Springer|Wiley|ACS|Nature|Science)\s*$", re.IGNORECASE),
    re.compile(r"^journal homepage:\s*.*$", re.IGNORECASE),
    re.compile(r"^Check for\s*$", re.IGNORECASE),
    re.compile(r"^updates\s*$", re.IGNORECASE),
]


def strip_leading_journal_masthead(text: str) -> str:
    """Remove ScienceDirect-style journal banner lines at the top of OCR text."""
    text = text.lstrip("\ufeff")
    lines = text.split("\n")
    while lines and not lines[0].strip():
        lines.pop(0)
    # Merged OCR often starts with ``## Page 1`` then temp PNG path / sidebar noise before the banner.
    if lines and re.match(r"^##\s+Page\s+1\s*$", lines[0].strip()):
        lines.pop(0)
        while lines and not lines[0].strip():
            lines.pop(0)
    text = "\n".join(lines)
    text = strip_paddle_vl_block_artifacts(text)
    lines = text.split("\n")
    while lines:
        first = lines[0].strip()
        if not first:
            lines.pop(0)
            continue
        if any(p.match(first) for p in _MASTHEAD_LINE_PATTERNS):
            lines.pop(0)
            continue
        break
    return "\n".join(lines).lstrip("\n")


def _looks_like_split_title_continuation(prev_line: str, next_line: str) -> bool:
    prev_line = prev_line.strip()
    next_line = next_line.strip()
    if not prev_line or not next_line:
        return False
    if prev_line.startswith("#"):
        return False
    if "$" in prev_line or "$" in next_line:
        return False
    if len(prev_line) < 18:
        return False
    # Second line continues a sentence (typical OCR title wrap)
    if next_line[0].islower():
        return True
    return False


def promote_split_paper_title_to_h2(text: str) -> str:
    """If the first two non-empty lines form a wrapped paper title, merge into ``## Title``."""
    lines = text.split("\n")
    while lines and not lines[0].strip():
        lines.pop(0)
    if len(lines) < 2:
        return text
    l0, l1 = lines[0].strip(), lines[1].strip()
    if not _looks_like_split_title_continuation(l0, l1):
        return text
    title = f"## {l0} {l1}"
    return "\n".join([title] + lines[2:])


def _remove_elsevier_check_for_updates_lines(text: str) -> str:
    """Drop standalone Elsevier UI lines (often OCR'd into the middle of a wrapped title)."""
    lines = text.split("\n")
    if not lines:
        return text
    head_n = min(48, len(lines))
    head = lines[:head_n]
    tail = lines[head_n:]
    noise = frozenset({"check for", "updates"})
    head = [ln for ln in head if ln.strip().lower() not in noise]
    return "\n".join(head + tail)


def normalize_leading_masthead_and_title(text: str) -> str:
    """Strip journal masthead OCR, then promote a two-line wrapped title to Markdown H2."""
    text = strip_leading_journal_masthead(text)
    text = _remove_elsevier_check_for_updates_lines(text)
    return promote_split_paper_title_to_h2(text)


# ---------------------------------------------------------------------------
# Two-column OCR bleed: Keywords (left) interleaved with Abstract (right)
# ---------------------------------------------------------------------------
_ABSTRACT_LINE_START_RE = re.compile(
    r"^(This was|This study|This alloy|This paper|This work|Due to|Furthermore|Additionally|"
    r"Finally|Moreover|In this study|While |Although |The |These |Here, )",
    re.IGNORECASE,
)

# Phrases that almost never appear on a standalone keyword line
_ABSTRACT_PHRASE_HINTS = (
    " in this study",
    " furthermore",
    " due to ",
    " which can be ",
    " revealed by ",
    " ensuring ",
    " opens up ",
    " accomplished by ",
    " significant ",
    " negligible ",
    " first-principles ",
    " lattice distortion and ",
)


def _classify_keyword_vs_abstract_line(line: str) -> str:
    """Return 'blank', 'kw', 'kw_label', or 'abs' for a non-heading content line."""
    s = line.strip()
    if not s:
        return "blank"
    sl = s.lower()
    if sl in ("keywords:", "keyword:"):
        return "kw_label"
    if s[0].islower():
        return "abs"
    if _ABSTRACT_LINE_START_RE.match(s):
        return "abs"
    for hint in _ABSTRACT_PHRASE_HINTS:
        if hint in sl:
            return "abs"
    # Long closing sentences (not keyword lines)
    if re.search(r"\bRHEAs\b", s) and len(s) > 28:
        return "abs"
    if len(s) > 120:
        return "abs"
    if s.count(".") >= 2:
        return "abs"
    if "," in s and len(s) > 70 and s.count(",") >= 2:
        return "abs"
    return "kw"


def _needs_kw_abs_interleave_repair(types: List[str]) -> bool:
    """True if keyword lines appear after the first abstract fragment (column bleed)."""
    seq = [t for t in types if t not in ("blank", "kw_label")]
    if len(seq) < 4:
        return False
    if not any(t == "kw" for t in seq) or not any(t == "abs" for t in seq):
        return False
    first_abs = next((i for i, t in enumerate(seq) if t == "abs"), None)
    if first_abs is None:
        return False
    last_kw = max((i for i, t in enumerate(seq) if t == "kw"), default=-1)
    return last_kw > first_abs


def _join_abstract_fragments(frags: List[str]) -> str:
    if not frags:
        return ""
    result = frags[0].strip()
    for nxt in frags[1:]:
        n = nxt.strip()
        if not n:
            continue
        if result.endswith("-"):
            result = result[:-1].rstrip() + n
        else:
            result = result + " " + n
    return result


def _section_break_after_keywords_block(line: str) -> bool:
    """First-line Markdown section after Keywords / ABSTRACT front matter."""
    s = line.strip()
    if not s.startswith("##"):
        return False
    if re.match(r"^##\s+Keywords\s*$", s, re.IGNORECASE):
        return False
    if re.match(r"^##\s+ARTICLE\s+INFO\s*$", s, re.IGNORECASE):
        return False
    if re.match(r"^##\s+ABSTRACT\s*$", s, re.IGNORECASE):
        return False
    if re.match(r"^##\s+Abstract\s*$", s, re.IGNORECASE):
        return False
    if re.match(r"^##\s+\d+\.", s):
        return True
    if re.match(
        r"^##\s+(Introduction|Experimental|Discussion|Conclusions?|Results|Methods|References?|Acknowledg)",
        s,
        re.IGNORECASE,
    ):
        return True
    return False


def _collapse_empty_abstract_heading_before_keywords(text: str) -> str:
    """Remove standalone empty ``## Abstract`` between ARTICLE INFO and Keywords."""
    t = re.sub(
        r"\n##\s+ABSTRACT\s*\n+(?=\s*##\s+Keywords\s*\n)",
        "\n",
        text,
        flags=re.IGNORECASE,
    )
    t = re.sub(
        r"\n##\s+Abstract\s*\n+(?=\s*##\s+Keywords\s*\n)",
        "\n",
        t,
        flags=re.IGNORECASE,
    )
    return t


def repair_keywords_abstract_two_column_ocr(text: str) -> str:
    """Un-scramble Elsevier-style two-column OCR where Keywords and Abstract lines alternate."""
    text = _collapse_empty_abstract_heading_before_keywords(text)
    lines = text.splitlines()
    out: List[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if not re.match(r"^##\s+Keywords\s*$", line.strip(), re.IGNORECASE):
            out.append(line)
            i += 1
            continue

        out.append(line)
        i += 1
        content: List[str] = []
        while i < len(lines):
            if _section_break_after_keywords_block(lines[i]):
                break
            content.append(lines[i])
            i += 1

        nonempty = [ln for ln in content if ln.strip()]
        if not nonempty:
            out.append("")
            continue

        types = [_classify_keyword_vs_abstract_line(ln) for ln in content]
        if not _needs_kw_abs_interleave_repair(types):
            out.extend(content)
            continue

        kw_lines: List[str] = []
        abs_frags: List[str] = []
        for ln, ty in zip(content, types):
            if ty == "blank":
                continue
            if ty in ("kw", "kw_label"):
                kw_lines.append(ln)
            elif ty == "abs":
                abs_frags.append(ln)

        abstract_joined = _join_abstract_fragments(abs_frags)
        rebuilt: List[str] = []
        for kl in kw_lines:
            if kl.strip():
                rebuilt.append(kl)
        rebuilt.append("")
        rebuilt.append("## ABSTRACT")
        rebuilt.append("")
        rebuilt.append(abstract_joined)
        rebuilt.append("")
        out.extend(rebuilt)

    result = "\n".join(out)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


# ---------------------------------------------------------------------------
# Greek symbol normalisation for materials science OCR.
# ---------------------------------------------------------------------------

# OCR commonly misreads Greek letters in phase/property contexts.
# Each entry: (wrong_char, correct_unicode, context_keywords).
# The fix is applied only when the line contains at least one context keyword.
_GREEK_OCR_FIXES: List[Tuple[str, str, Tuple[str, ...]]] = [
    # σ (sigma) phase, stress, yield strength context.
    # OCR often outputs 'o' or 'σ' (already correct) or 'ơ'.
    ("ơ", "σ", ("phase", "stress", "strength", "MPa", "GPa", "yield", "σ", "phase")),
    # μ (mu) phase context; OCR may output 'u' adjacent to 'phase' or 'μm'.
    # We only fix standalone 'u' that looks like a Greek mu in context.
    # Pattern: word-boundary 'u' followed by 'm' (μm) or before 'phase'.
    # Handled via regex below.
]

# Regex patterns for context-gated Greek symbol fixes.
# Each tuple: (compiled_pattern, replacement, context_keywords_tuple)
_GREEK_REGEX_FIXES: List[Tuple[re.Pattern, str, Tuple[str, ...]]] = [
    # 'o phase' / 'o-phase' → 'σ phase' / 'σ-phase' (sigma phase)
    (
        re.compile(r"\bo\s*(?=-?phase\b)", re.IGNORECASE),
        "σ",
        ("phase", "precipitation", "alloy", "material", "structure", "crystal"),
    ),
    # 'u phase' / 'u-phase' → 'μ phase' / 'μ-phase'
    (
        re.compile(r"\bu\s*(?=-?phase\b)", re.IGNORECASE),
        "μ",
        ("phase", "precipitation", "alloy", "material", "structure", "crystal"),
    ),
    # standalone 'um' as unit (e.g. '10 um') → '10 μm'
    (
        re.compile(r"(?<=\d\s)um\b"),
        "μm",
        ("grain", "size", "diameter", "thickness", "spacing", "μm", "nm"),
    ),
    # '10um' (no space) → '10μm'
    (
        re.compile(r"(?<=\d)um\b"),
        "μm",
        ("grain", "size", "diameter", "thickness", "spacing", "μm", "nm"),
    ),
]


def _is_phase_context(line: str, keywords: Tuple[str, ...]) -> bool:
    """Return True when the line contains at least one of the context keywords."""
    line_lower = line.lower()
    return any(kw.lower() in line_lower for kw in keywords)


def normalize_greek_symbols(text: str) -> str:
    """Apply context-gated Greek symbol normalisation to *text*.

    Only modifies lines that contain at least one of the required context
    keywords for each fix, avoiding false positives in non-physics text.
    """
    lines: List[str] = []
    for line in text.splitlines():
        # Quick check for general materials context to avoid over-matching in general text
        if bool(_MATERIALS_CONTEXT_KEYWORDS.search(line)):
            for char_wrong, char_correct, keywords in _GREEK_OCR_FIXES:
                if char_wrong in line and _is_phase_context(line, keywords):
                    line = line.replace(char_wrong, char_correct)
            for pat, repl, keywords in _GREEK_REGEX_FIXES:
                if _is_phase_context(line, keywords):
                    line = pat.sub(repl, line)
        lines.append(line)
    return "\n".join(lines)


def _is_chemical_context(line: str) -> bool:
    """Return True when the line is likely to contain chemical formula notation.
    
    A line is considered a chemical context if ANY of the following is true:
    1. Contains at least one element symbol followed immediately by a digit
    2. Contains materials science context keywords
    """
    # Check for element-number pattern
    if re.search(r"(?:" + _ELEMENT_PAT + r")\d", line):
        return True
    # ``_MATERIALS_CONTEXT_KEYWORDS`` is a compiled regex, not a list of strings.
    return bool(_MATERIALS_CONTEXT_KEYWORDS.search(line))


def _normalize_single_element_formula(match: re.Match) -> str:
    """Convert a single element-number pair like Nb15 to Nb_{15}."""
    element = match.group(2)
    number = match.group(3)
    # Normalise decimal separators
    number = _DECIMAL_NOISE_RE.sub(_DECIMAL_REPL, number)
    number = _COMMA_DECIMAL_RE.sub(_DECIMAL_REPL, number)
    return f"{element}_{{{number}}}"


def _normalize_paren_alloy_formula(match: re.Match) -> str:
    """Convert parenthesized alloy like (Nb15Ta10W75)98.5C1.5 to proper LaTeX.
    
    Output: (Nb_{15}Ta_{10}W_{75})_{98.5}C_{1.5}
    """
    inner = match.group(1)  # Nb15Ta10W75
    subscript = match.group(4)  # 98.5
    trailing = match.group(5) or ""  # C1.5
    
    # Normalize inner part
    inner_normalized = re.sub(
        r"(?:" + _ELEMENT_PAT + r")(\d+(?:[.,]\d+)?)",
        lambda m: m.group(0)[0:len(m.group(0))-len(m.group(1))] + "_{" + _DECIMAL_NOISE_RE.sub(_DECIMAL_REPL, _COMMA_DECIMAL_RE.sub(_DECIMAL_REPL, m.group(1))) + "}",
        inner
    )
    
    # Normalize subscript
    subscript_normalized = _DECIMAL_NOISE_RE.sub(_DECIMAL_REPL, subscript)
    subscript_normalized = _COMMA_DECIMAL_RE.sub(_DECIMAL_REPL, subscript_normalized)
    
    # Normalize trailing elements
    trailing_normalized = ""
    if trailing:
        trailing_normalized = re.sub(
            r"(?:" + _ELEMENT_PAT + r")(\d+(?:[.,]\d+)?)",
            lambda m: m.group(0)[0:len(m.group(0))-len(m.group(1))] + "_{" + _DECIMAL_NOISE_RE.sub(_DECIMAL_REPL, _COMMA_DECIMAL_RE.sub(_DECIMAL_REPL, m.group(1))) + "}",
            trailing
        )
    
    return f"({inner_normalized})_{{{subscript_normalized}}}{trailing_normalized}"


_DECIMAL_REPL = r"\1.\2"
_COMMA_DECIMAL_RE = re.compile(r"(\d),(\d)")


def _normalize_alloy_formula(match: re.Match) -> str:
    """Convert a raw alloy string like Ti42Hf21 to Ti_{42}Hf_{21}."""
    raw = match.group(0)
    # Normalise decimal separators: Chinese comma / ASCII comma → ASCII period.
    # ASCII comma is treated as a decimal separator only inside alloy formulas.
    raw = _DECIMAL_NOISE_RE.sub(_DECIMAL_REPL, raw)
    raw = _COMMA_DECIMAL_RE.sub(_DECIMAL_REPL, raw)
    # Insert LaTeX subscript notation around the numeric parts.
    result = re.sub(
        r"(\d+(?:\.\d+)?)",
        lambda m: "_{" + m.group(1) + "}",
        raw,
    )
    return result


def normalize_alloy_strings(text: str) -> str:
    """Apply OCR-specific fixes and chemical formula normalisation.

    Steps:
    1. Apply hard-coded alloy OCR fixes (specific known misreads).
    2. Normalise decimal separators (Chinese comma → ASCII period) in numeric
       contexts.
    3. Convert parenthesized alloy formulas like (Nb15Ta10W75)98.5C1.5.
    4. Convert multi-element formulas (e.g. Ti42Hf21) to LaTeX subscript.
    5. Convert single element-number pairs (e.g. Nb15) when in materials context.
    6. Apply context-gated Greek symbol normalisation (σ/μ phase etc.).
    """
    for pat, repl in ALLOY_OCR_FIXES:
        text = pat.sub(repl, text)

    # Process line-by-line so the chemical context gate is per-line.
    lines: List[str] = []
    for line in text.splitlines():
        # Decimal normalisation: always safe within a numeric context.
        line = _DECIMAL_NOISE_RE.sub(_DECIMAL_REPL, line)
        
        # Process only in chemical/materials contexts
        if _is_chemical_context(line):
            # 1. Handle parenthesized alloy formulas first (most specific)
            # e.g., (Nb15Ta10W75)98.5C1.5 → (Nb_{15}Ta_{10}W_{75})_{98.5}C_{1.5}
            line = _PAREN_ALLOY_RE.sub(_normalize_paren_alloy_formula, line)
            
            # 2. Handle multi-element alloy formulas (2+ element-number pairs)
            line = _ALLOY_FORMULA_RE.sub(_normalize_alloy_formula, line)
            
            # 3. Handle single element-number pairs (most general)
            # Only apply if the line has strong materials science context
            # to avoid false positives on things like "Co2" in "Co2 emissions"
            if any(kw in line.lower() for kw in ("alloy", "phase", "composition", 
                                                   "matrix", "element", "atom", "mole",
                                                   "yield", "strength", "MPa", "GPa",
                                                   "microstructure", "grain", "precipitate")):
                line = _SINGLE_ELEMENT_NUM_RE.sub(_normalize_single_element_formula, line)
        
        lines.append(line)
    text = "\n".join(lines)

    # Greek symbol normalisation (context-gated).
    text = normalize_greek_symbols(text)

    return text


def is_noise_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    numeric_unit = _normalize_spaces_for_noise(stripped)
    if re.match(r"^\d+\.?(?:\d+)?\s*(?:um|nm|pm|°C|C)\b", numeric_unit, re.IGNORECASE):
        return True
    # Diagram scale / axis labels: optional '.' after integer (e.g. '200.' before µm), Unicode µ, mm/cm.
    if re.match(
        r"^\d+\.?(?:\d+)?\s*(?:\u00b5m|\u03bcm|µm|μm|um|nm|pm|mm|cm)\s*$",
        numeric_unit,
        re.IGNORECASE,
    ):
        return True
    # Isolated axis letters / arrows from schematic figures (not real section titles).
    if re.fullmatch(r"[XYZ]", numeric_unit):
        return True
    if re.fullmatch(
        r"\\(?:leftarrow|rightarrow|leftrightarrow|Leftarrow|Rightarrow|longrightarrow|longleftarrow)",
        numeric_unit,
    ):
        return True
    if re.match(r"^\d+\.?(?:\d+)?\s*(?:°C|C)\s*[~\-]\s*\d+\.?(?:\d+)?\s*(?:°C|C)\b", numeric_unit):
        return True

    has_doi = bool(DOI_RE.search(stripped)) or bool(DOI_URL_RE.search(stripped))
    for pat, skip_if_doi in NOISE_LINE_PATTERNS:
        if pat.match(stripped):
            if skip_if_doi and has_doi:
                return False
            return True
    return False


# OCR often reads schematic labels / EDS map keys beside figures as separate lines
# immediately before the real "Fig. N." caption; strip those runs only when a
# figure caption line follows (avoids deleting legitimate short lines elsewhere).
# Also match "ig. N." when the leading "F" is lost in OCR.
_ELEMENT_SYMBOL_LOWER = frozenset(e.lower() for e in _ELEMENT_SYMBOLS)
_FIG_LEGEND_ET_AL = re.compile(
    r"^(?:\.+\s*\w+\s+et\s+al\.?\s*|z[a-z]{2,}\s+et\s+al\.?\s*)$",
    re.IGNORECASE,
)
_FIG_LEGEND_COLON_ELEM = re.compile(
    r"^(?:\*\s*:\s*|:\s*|\^\s*:\s*)([A-Za-z]{1,3})\s*$",
)
_FIG_LEGEND_N_LAYER = re.compile(r"^N(?:\+\d+)?\s+layer\s*$", re.IGNORECASE)
_FIG_LEGEND_PHRASE = re.compile(
    r"^(?:Laser beam|Gas-driven powder|Mixed powder)\s*$",
    re.IGNORECASE,
)
_FIG_SCHEMATIC_WORDS = frozenset({
    "intensity", "relative", "argon", "lens", "motion", "atmosphere",
    "deposition", "head", "delivery", "px", "surface", "cracks", "crack",
    "dimples", "river", "patterns", "extensive", "highly", "deformed", "areas",
    "dislocation", "pinning", "loops", "slip", "bands", "wavy", "slips",
    "screws", "phase", "ipf", "kam", "um", "this work", "rt",
    "stress", "strain", "temperature", "time", "voltage", "current", "energy"
})
_FIG_SOLO_ELEMENT_LINE = re.compile(r"^(" + _ELEMENT_PAT + r"|TI)\s*$", re.IGNORECASE)
# "ig. N." missing leading F; "Figure" / "Fig." normal
_FIG_CAPTION_START = re.compile(
    r"^(?:Figure|Fig\.?|ig\.?)\s*\d+\s*[.:]",
    re.IGNORECASE,
)
_FIG_PANEL_MARK = re.compile(r"^\(([ivx]{1,5}|[a-z])\)\s*$", re.IGNORECASE)
_FIG_MILLER_SIMPLE = re.compile(r"^\([01\s\-]{3,12}\)\s*$")
_FIG_MILLER_BRACKET = re.compile(r"^\[[\d\s\-]+\]\s*$")
_FIG_SINGLE_PUNCT = re.compile(r"^[|│\[\]⟦⟧⟨⟩]$")
_FIG_SCALE_OR_UNIT = re.compile(
    r"^(?:\.\d+|\d+)\s*nm\s*$|^\d+\s*prm\s*$|^[Cc]\s+\d+\.\d\s*$|^(?:22\*|Q2|ei|TTL)$",
    re.IGNORECASE,
)
_FIG_AXIS_LINE = re.compile(
    r"^(?:Intensity|Concentration|Strain|d-spacing|r\([ÅA]\)|Engineering\s+stress|"
    r"Engineering\s+strain|Temperature|Relative\s+position|Mass\s+gain|Time\s*/\s*h|"
    r"Young'?s\s+Modulus|Lattice\s+constant|Shear\s+modulus|d-spacing\s*\(Å\)|"
    r"Intensity\s*\(a\.u\.\)|Voltage|Current|Capacity|Energy\s+Density)\b",
    re.IGNORECASE,
)
_FIG_COLON_MAP = re.compile(r"^:\s*[A-Za-z]{1,4}\d*\s*$")
_FIG_TIME_OX = re.compile(r"^\d{1,3}h\s*$", re.IGNORECASE)
_FIG_GIBBERISH_CAPS = re.compile(r"^(?:[A-Z]{2,3}[A-Z0-9]{2,}|[A-Z]{5,})$")


def _is_figure_legend_fragment_line(line: str) -> bool:
    """Detect if a line looks like an orphan label from a figure (e.g. "FeCoCrNi", "Intensity (a.u.)")."""
    text = line.strip()
    if not text:
        return False

    if _FIG_CAPTION_START.match(text):
        return False

    if _FIG_AXIS_LINE.match(text):
        return True

    if _FIG_SOLO_ELEMENT_LINE.match(text):
        return True
    if _FIG_PANEL_MARK.match(text):
        return True
    if _FIG_MILLER_SIMPLE.match(text) or _FIG_MILLER_BRACKET.match(text):
        return True
    if _FIG_SINGLE_PUNCT.match(text):
        return True
    if _FIG_SCALE_OR_UNIT.match(text):
        return True

    # Generic heuristic for short fragments
    if len(text) <= 25:
        # High density of symbols or mostly caps might indicate a label
        symbol_density = sum(1 for c in text if not c.isalnum() and not c.isspace()) / len(text)
        cap_density = sum(1 for c in text if c.isupper()) / len(text)
        
        # Lacks typical sentence structure
        starts_upper = text[0].isupper()
        ends_period = text.endswith('.')
        
        # If it's a short phrase without sentence structure, it's likely a fragment
        if not (starts_upper and ends_period):
            if symbol_density > 0.2 or cap_density > 0.3:
                return True
                
            words = set(re.findall(r"[A-Za-z]+", text.lower()))
            if words and words.issubset(_FIG_SCHEMATIC_WORDS):
                return True

    return False


def _strip_figure_legend_prefix_lines(lines: List[str]) -> List[str]:
    out: List[str] = []
    for line in lines:
        stripped = line.strip()
        if _FIG_CAPTION_START.match(stripped):
            # Pop trailing legend lines; also pop blank lines between caption and legend so
            # a spacer does not block removal of fragments above the spacer (two-column OCR).
            while out:
                last_s = out[-1].strip()
                if _is_figure_legend_fragment_line(last_s):
                    out.pop()
                elif not last_s:
                    out.pop()
                else:
                    break
        out.append(line)
    return out


def structure_sections(text: str) -> str:
    """Structure document sections with proper heading levels.
    
    This function processes each line of text, detecting headings and
    converting them to appropriate Markdown heading levels. It supports:
    - Numbered sections (1., 1.1, 1.1.1)
    - Common section titles (Introduction, Methods, etc.)
    - Multi-level heading hierarchy
    """
    output_lines: List[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if is_noise_line(stripped):
            continue

        # Markdown "heading" lines that are really body / chart fragments (often from prior passes)
        m_hash = re.match(r"^#{1,6}\s+(\d+(?:\.\d+)*\.?)\s+(.+)$", stripped)
        if m_hash:
            sec_num_h = m_hash.group(1).rstrip(".")
            sec_title_h = m_hash.group(2).strip()
            if _is_spurious_generic_numeric_section(sec_num_h, sec_title_h):
                output_lines.append(re.sub(r"^#+\s*", "", stripped))
                continue
        
        # Try the new multi-level heading detector first
        heading = detect_heading(stripped)
        if heading:
            if heading.number is not None and heading.title and _is_spurious_generic_numeric_section(
                str(heading.number), str(heading.title)
            ):
                output_lines.append(line)
                continue
            output_lines.append("")
            output_lines.append(heading.to_markdown())
            output_lines.append("")
            continue
        
        # Fall back to original pattern matching for special cases
        matched = False
        for pat, replacement in SECTION_PATTERNS:
            m = pat.match(stripped)
            if m:
                if replacement:
                    output_lines.append("")
                    output_lines.append(replacement)
                    output_lines.append("")
                else:
                    # This case is now handled by detect_heading above,
                    # but keep for backwards compatibility
                    sec_title = m.group(2).strip()
                    if _is_unit_only_section_title(sec_title):
                        matched = True
                        break
                    sec_num = m.group(1).rstrip(".")
                    if _is_spurious_generic_numeric_section(sec_num, sec_title):
                        output_lines.append(line)
                        matched = True
                        break
                    output_lines.append("")
                    output_lines.append(f"## {sec_num}. {sec_title}")
                    output_lines.append("")
                matched = True
                break
        if matched:
            continue
        m = SPACED_TITLE_RE.match(stripped)
        if m:
            collapsed = m.group(2).replace(" ", "")
            output_lines.append("")
            output_lines.append(f"## {collapsed}")
            output_lines.append("")
            continue
        heading_match = re.match(r"^(#{3,6})\s+(.*)", stripped)
        if heading_match:
            title = heading_match.group(2).strip()
            if _is_unit_only_section_title(title) or is_noise_line(title):
                continue
            output_lines.append(f"## {title}")
            continue
        if re.match(r"^##\s+Page\s+\d+\s*$", stripped):
            output_lines.append("")
            continue
        output_lines.append(line)

    output_lines = _strip_figure_legend_prefix_lines(output_lines)
    result = "\n".join(output_lines)
    result = re.sub(r"\n{3,}", "\n\n", result)
    return result.strip()


def strip_references_section(text: str) -> str:
    out: List[str] = []
    skipping_refs = False

    for line in text.splitlines():
        stripped = line.strip()

        # Check for References start
        if re.match(r"^#+\s*(references?|bibliography|citations?)\s*$", stripped, re.IGNORECASE) or re.match(
            r"^(references?|bibliography|citations?)\s*$", stripped, re.IGNORECASE
        ):
            skipping_refs = True
            continue

        # Check for Appendix / Supplementary start to stop skipping
        # Matches lines starting with # (header) and containing Appendix or Supplementary
        if skipping_refs and re.match(r"^#+\s*(appendix|supplementary)\b.*$", stripped, re.IGNORECASE):
            skipping_refs = False
            out.append(line)
            continue

        if not skipping_refs:
            out.append(line)

    return "\n".join(out).strip()

