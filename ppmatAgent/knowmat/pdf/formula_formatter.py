"""Mathematical formula formatting utilities for OCR output.

This module provides functions to convert plain text mathematical expressions
to LaTeX format, handling Greek letters, subscripts, superscripts, and special
operators commonly found in scientific documents.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Greek letter mappings (Unicode to LaTeX)
# ---------------------------------------------------------------------------
GREEK_TO_LATEX: Dict[str, str] = {
    # Lowercase Greek letters
    "α": "\\alpha",
    "β": "\\beta",
    "γ": "\\gamma",
    "δ": "\\delta",
    "ε": "\\epsilon",
    "ζ": "\\zeta",
    "η": "\\eta",
    "θ": "\\theta",
    "ι": "\\iota",
    "κ": "\\kappa",
    "λ": "\\lambda",
    "μ": "\\mu",
    "ν": "\\nu",
    "ξ": "\\xi",
    "π": "\\pi",
    "ρ": "\\rho",
    "σ": "\\sigma",
    "τ": "\\tau",
    "υ": "\\upsilon",
    "φ": "\\phi",
    "χ": "\\chi",
    "ψ": "\\psi",
    "ω": "\\omega",
    # Uppercase Greek letters
    "Α": "A",  # Often rendered as Latin A
    "Β": "B",
    "Γ": "\\Gamma",
    "Δ": "\\Delta",
    "Ε": "E",
    "Ζ": "Z",
    "Η": "H",
    "Θ": "\\Theta",
    "Ι": "I",
    "Κ": "K",
    "Λ": "\\Lambda",
    "Μ": "M",
    "Ν": "N",
    "Ξ": "\\Xi",
    "Ο": "O",
    "Π": "\\Pi",
    "Ρ": "P",
    "Σ": "\\Sigma",
    "Τ": "T",
    "Υ": "\\Upsilon",
    "Φ": "\\Phi",
    "Χ": "X",
    "Ψ": "\\Psi",
    "Ω": "\\Omega",
}

# Reverse mapping for LaTeX to Unicode (useful for testing)
LATEX_TO_GREEK: Dict[str, str] = {v: k for k, v in GREEK_TO_LATEX.items()}

# ---------------------------------------------------------------------------
# Mathematical operators and symbols
# ---------------------------------------------------------------------------
MATH_SYMBOLS: Dict[str, str] = {
    # Comparison and relation operators
    "≤": "\\leq",
    "≥": "\\geq",
    "≠": "\\neq",
    "≈": "\\approx",
    "≡": "\\equiv",
    "≪": "\\ll",
    "≫": "\\gg",
    # Arithmetic operators
    "×": "\\times",
    "÷": "\\div",
    "±": "\\pm",
    "∓": "\\mp",
    # Miscellaneous symbols
    "∞": "\\infty",
    "∂": "\\partial",
    "∇": "\\nabla",
    "∫": "\\int",
    "∑": "\\sum",
    "∏": "\\prod",
    "√": "\\sqrt",
    "°": "^\\circ",
    # Arrows
    "→": "\\rightarrow",
    "←": "\\leftarrow",
    "↔": "\\leftrightarrow",
    "⇒": "\\Rightarrow",
    "⇐": "\\Leftarrow",
    "⇔": "\\Leftrightarrow",
}

# ---------------------------------------------------------------------------
# Patterns for mathematical expression detection
# ---------------------------------------------------------------------------

# Pattern for scientific notation: 1.5 × 10^7
_SCI_NOTATION_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*[×x]\s*10\^(\d+)"
)

# After convert_greek_to_latex: \sigma0.2 / \sigma 0.2 → \sigma_{0.2}
# Only matches standard LaTeX Greek command names (not prose words).
_GREEK_CMD_NUMERIC_SUB_RE = re.compile(
    r"\\("
    r"alpha|beta|gamma|delta|epsilon|zeta|eta|theta|iota|kappa|lambda|mu|nu|xi|pi|"
    r"rho|sigma|tau|upsilon|phi|chi|psi|omega|"
    r"Delta|Gamma|Theta|Lambda|Xi|Pi|Sigma|Upsilon|Phi|Psi|Omega"
    r")\s*(\d+(?:\.\d+)?)\b"
)

# Line-level hints for integer-only subscripts (e.g. \sigma1). Decimal subscripts
# like 0.2 are converted without this gate.
_PHYSICS_SUBSCRIPT_LINE_KEYWORDS = frozenset({
    "mpa", "gpa", "stress", "strain", "yield", "tensile", "hardness",
    "strength", "elongation", "modulus", "toughness", "fracture",
    "dislocation", "slip", "twin", "shear", "elastic", "plastic",
    "equation", "eq.", "fig.", "table", "model", "simulation",
    "temperature", "thermal", "diffusion", "lattice",
})


def _normalize_greek_command_subscripts_line(line: str) -> str:
    """Turn \\sigma0.2 into \\sigma_{0.2} outside math mode; conservative on integers."""
    parts = re.split(r"(\$[^$]*\$)", line)
    line_lower = line.lower()
    has_physics_kw = any(kw in line_lower for kw in _PHYSICS_SUBSCRIPT_LINE_KEYWORDS)

    def fix_segment(seg: str) -> str:
        def repl(m: re.Match[str]) -> str:
            cmd = m.group(1)
            num = m.group(2)
            if "." in num or has_physics_kw:
                return f"\\{cmd}_{{{num}}}"
            return m.group(0)

        return _GREEK_CMD_NUMERIC_SUB_RE.sub(repl, seg)

    out: List[str] = []
    for i, part in enumerate(parts):
        out.append(fix_segment(part) if i % 2 == 0 else part)
    return "".join(out)


def normalize_greek_command_subscripts(text: str) -> str:
    """Attach numeric subscripts to LaTeX Greek commands (e.g. \\sigma0.2 → \\sigma_{0.2}).

    Runs line-by-line and skips regions already wrapped in ``$...$`` to avoid
    corrupting display math. Integer subscripts (e.g. \\sigma1) apply only when
    the line contains mechanical/physics cues, so ordinary prose is mostly spared.
    """
    return "\n".join(_normalize_greek_command_subscripts_line(ln) for ln in text.splitlines())


def convert_greek_to_latex(text: str) -> str:
    """Convert Greek Unicode characters to LaTeX commands.
    
    Parameters
    ----------
    text : str
        Input text containing Greek characters.
        
    Returns
    -------
    str
        Text with Greek characters replaced by LaTeX commands.
    """
    for greek, latex in GREEK_TO_LATEX.items():
        text = text.replace(greek, latex)
    return text


def convert_math_symbols_to_latex(text: str) -> str:
    """Convert mathematical Unicode symbols to LaTeX commands.
    
    Parameters
    ----------
    text : str
        Input text containing mathematical symbols.
        
    Returns
    -------
    str
        Text with mathematical symbols replaced by LaTeX commands.
    """
    for symbol, latex in MATH_SYMBOLS.items():
        text = text.replace(symbol, latex)
    return text


def format_scientific_notation(text: str) -> str:
    """Convert plain scientific notation to LaTeX format.
    
    E.g., "1.5 × 10^7" → "$1.5 \\times 10^{7}$"
    
    Parameters
    ----------
    text : str
        Input text possibly containing scientific notation.
        
    Returns
    -------
    str
        Text with scientific notation formatted in LaTeX.
    """
    
    def _replace_sci_notation(match: re.Match) -> str:
        coeff = match.group(1)
        exp = match.group(2)
        return f"${coeff} \\times 10^{{{exp}}}$"
    
    return _SCI_NOTATION_RE.sub(_replace_sci_notation, text)


def wrap_inline_math(text: str) -> str:
    """Wrap inline mathematical expressions with $ delimiters.
    
    This function identifies isolated mathematical expressions (Greek letters,
    LaTeX commands, subscripts) that are not already wrapped in $...$.
    
    Parameters
    ----------
    text : str
        Input text.
        
    Returns
    -------
    str
        Text with inline math expressions wrapped in $...$.
    """
    # Pattern to find standalone LaTeX commands or Greek letters
    # that are not already in math mode
    latex_pattern = r"(?<!\$)\\(alpha|beta|gamma|delta|epsilon|zeta|eta|theta|iota|kappa|lambda|mu|nu|xi|pi|rho|sigma|tau|upsilon|phi|chi|psi|omega|Delta|Gamma|Theta|Lambda|Xi|Pi|Sigma|Upsilon|Phi|Psi|Omega|times|div|pm|leq|geq|approx|infty|partial|nabla)(?!\$)"
    
    # Pattern for subscript/superscript expressions not in math mode
    subscript_pattern = r"(?<!\$)([A-Za-z])_\{[^}]+\}(?!\$)"
    
    lines: List[str] = []
    for line in text.splitlines():
        # Skip lines that are already mostly math or code
        if line.strip().startswith("$") or line.strip().startswith("$$"):
            lines.append(line)
            continue
        
        # Wrap isolated LaTeX commands
        new_line = re.sub(latex_pattern, r"$\\\1$", line)
        
        # Wrap subscript expressions
        new_line = re.sub(subscript_pattern, r"$\1_{...}$", new_line)
        
        lines.append(new_line)
    
    return "\n".join(lines)


def format_formula_text(text: str) -> str:
    """Apply all formula formatting transformations.
    
    This is the main entry point for formula formatting, applying:
    1. Greek letter conversion (Unicode → ``\\alpha``, ``\\sigma``, …)
    2. Numeric subscripts on those commands (``\\sigma0.2`` → ``\\sigma_{0.2}``), only
       outside ``$...$`` and with a stricter rule for integer-only subscripts
    3. Math symbol conversion
    4. Scientific notation formatting
    
    Latin-word subscript guessing (e.g. Materials → M_{ATERIALS}) is not used;
    alloy / multi-element chemistry is handled in ``normalize_alloy_strings``.
    
    Parameters
    ----------
    text : str
        Input text with raw OCR output.
        
    Returns
    -------
    str
        Formatted text with LaTeX math expressions.
    """
    # Step 1: Convert Greek letters
    text = convert_greek_to_latex(text)
    
    # Step 2: Greek command + numeric subscript (does not touch Latin prose)
    text = normalize_greek_command_subscripts(text)
    
    # Step 3: Convert mathematical symbols
    text = convert_math_symbols_to_latex(text)
    
    # Step 4: Format scientific notation
    text = format_scientific_notation(text)
    
    return text


def extract_formula_context(text: str, pattern: str) -> List[Tuple[str, int, int]]:
    """Extract all occurrences of a formula pattern from text.

    Parameters
    ----------
    text : str
        Text to search.
    pattern : str
        Regex pattern to match.

    Returns
    -------
    List[Tuple[str, int, int]]
        List of (matched_text, start_pos, end_pos) tuples.
    """
    results: List[Tuple[str, int, int]] = []
    compiled = re.compile(pattern)
    for match in compiled.finditer(text):
        results.append((match.group(0), match.start(), match.end()))
    return results


# ---------------------------------------------------------------------------
# Orphan inline-formula collapse (post-processing)
# ---------------------------------------------------------------------------

_DISPLAY_MATH_LINE_RE = re.compile(r"^\s*\$\$\s*(.*?)\s*\$\$\s*$")

_SHORT_FORMULA_MAX_LEN = 80

_INLINE_FORMULA_INDICATORS = re.compile(
    r"\\mathrm\{|\\text\{|\\mathcal\{|\\textcircled\{|"
    r"[A-Z][a-z]?_?\{?\d|"  # element+subscript like Fe_0, LCFE_{2}
    r"^[A-Za-z\s\-_,.()|]+$|"  # pure text/punctuation (chemical name fragments)
    r"^[\d^{}]+$"  # pure numbers/superscripts like 10^{6}
)

_TRUE_EQUATION_INDICATORS = re.compile(
    r"\\frac\{|\\int|\\sum|\\prod|\\lim|\\begin\{|"
    r"\\[a-z]+\{[^}]{20,}|"  # long nested expressions
    r"=.*[a-zA-Z]|"  # equals sign followed by variable (real equation)
    r"\\rightarrow|\\to|\\longrightarrow|"  # reaction arrows
    r"\\d[xy]|\\partial"  # calculus
)


def _is_inline_formula_content(latex: str) -> bool:
    """Heuristic: is this display-math content actually an inline chemical/short formula?"""
    if len(latex) > _SHORT_FORMULA_MAX_LEN:
        return False
    if _TRUE_EQUATION_INDICATORS.search(latex):
        return False
    if _INLINE_FORMULA_INDICATORS.search(latex):
        return True
    if len(latex) <= 30:
        return True
    return False


def _latex_to_readable(latex: str) -> str:
    """Convert short LaTeX to a readable inline form for LLM consumption."""
    s = latex.strip().rstrip(",").rstrip(".")
    s = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mathbf\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mathit\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mathcal\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\textcircled\{([^}]*)\}", r"(\1)", s)
    s = re.sub(r"\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\left|\\right", "", s)
    s = re.sub(r"_(\w)", r"_\1", s)
    s = re.sub(r"\^(\w)", r"^\1", s)
    s = s.replace("\\", "").replace("  ", " ").strip()
    trailing = ""
    if latex.strip().endswith(","):
        trailing = ","
    elif latex.strip().endswith("."):
        trailing = "."
    return s + trailing


def collapse_orphan_display_formulas(text: str) -> str:
    """Collapse sequences of short display-math blocks (orphan inline formulas) into inline text.

    MinerU/PaddleOCR sometimes classifies inline chemical formula references as standalone
    'formula' blocks, producing sequences of ``$$ \\mathrm{FE_0} $$`` on separate lines with
    all surrounding paragraph text lost. This function detects such sequences (≥2 consecutive
    display-math-only lines containing short formula fragments) and collapses them into a
    single inline annotation that the LLM can parse.
    """
    lines = text.split("\n")
    result: List[str] = []
    i = 0

    while i < len(lines):
        line = lines[i]
        m = _DISPLAY_MATH_LINE_RE.match(line)

        if m and _is_inline_formula_content(m.group(1)):
            # Start collecting consecutive inline-formula display-math lines
            formula_run: List[str] = [m.group(1)]
            j = i + 1
            while j < len(lines):
                if lines[j].strip() == "":
                    # skip blank lines between consecutive formulas
                    j += 1
                    continue
                mj = _DISPLAY_MATH_LINE_RE.match(lines[j])
                if mj and _is_inline_formula_content(mj.group(1)):
                    formula_run.append(mj.group(1))
                    j += 1
                else:
                    break

            if len(formula_run) >= 2:
                # Collapse: convert each to readable form and join
                readable_parts = [_latex_to_readable(f) for f in formula_run if f.strip()]
                readable_parts = [p for p in readable_parts if p]
                if readable_parts:
                    collapsed = "[Inline formula references: " + ", ".join(readable_parts) + "]"
                    result.append(collapsed)
                i = j
            else:
                # Single inline formula on its own — convert to inline $ ... $
                readable = _latex_to_readable(formula_run[0])
                if readable:
                    result.append(f"$ {formula_run[0]} $")
                i += 1
        else:
            result.append(line)
            i += 1

    return "\n".join(result)