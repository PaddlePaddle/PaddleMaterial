"""Convert PaddleOCR cloud API JSONL output to KnowMat internal format."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

logger = logging.getLogger(__name__)

_FORMULA_PATTERN = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
_INLINE_FORMULA_PATTERN = re.compile(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)")
_IMAGE_PATTERN = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
_HTML_IMG_PATTERN = re.compile(r'<img\s+[^>]*src="([^"]+)"[^>]*/?>',re.IGNORECASE)
_IMG_WIDTH_PATTERN = re.compile(r'width="(\d+)%"', re.IGNORECASE)
_TABLE_LINE_PATTERN = re.compile(r"^\|.+\|$")
_HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$")
_HTML_TABLE_PATTERN = re.compile(
    r"<table\b[^>]*>.*?</table>", re.DOTALL | re.IGNORECASE
)
_YOU_MAY_ALSO_LIKE_PATTERN = re.compile(
    r"##\s+You may also like.*?(?=\n##\s|\n#\s|\n[A-Z][a-z].*\n)",
    re.DOTALL | re.IGNORECASE,
)
_FIGURE_CAPTION_DIV_PATTERN = re.compile(
    r'<div[^>]*>\s*((?:Fig(?:ure)?\.?\s*\d+)[^<]*)</div>',
    re.IGNORECASE,
)
_FIGURE_NUM_PATTERN = re.compile(
    r"(?:Fig(?:ure)?\.?\s*)(\d+[a-zA-Z]?)", re.IGNORECASE
)


def _parse_markdown_blocks(
    markdown_text: str,
    page_num: int,
    images_dir: Optional[Path],
    image_urls: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """Parse a page's markdown into structured ocr_items."""
    items: List[Dict[str, Any]] = []
    lines = markdown_text.split("\n")
    i = 0

    while i < len(lines):
        line = lines[i]

        # Display formula block ($$...$$)
        if line.strip().startswith("$$"):
            formula_lines = [line]
            if not line.strip().endswith("$$") or line.strip() == "$$":
                i += 1
                while i < len(lines):
                    formula_lines.append(lines[i])
                    if lines[i].strip().endswith("$$"):
                        i += 1
                        break
                    i += 1
            else:
                i += 1
            formula_text = "\n".join(formula_lines)
            latex = formula_text.strip().strip("$").strip()
            if latex:
                items.append({
                    "typer": "formula",
                    "data": {"text": latex, "latex": latex},
                    "page": page_num,
                    "block_label": "formula",
                })
            continue

        # Table block (consecutive | lines)
        if _TABLE_LINE_PATTERN.match(line.strip()):
            table_lines = []
            while i < len(lines) and _TABLE_LINE_PATTERN.match(lines[i].strip()):
                table_lines.append(lines[i])
                i += 1
            table_md = "\n".join(table_lines)
            if table_md.strip():
                items.append({
                    "typer": "table",
                    "data": {"text": table_md, "raw_html": ""},
                    "page": page_num,
                    "block_label": "table",
                })
            continue

        # Image reference (markdown syntax)
        img_match = _IMAGE_PATTERN.match(line.strip())
        if not img_match:
            # Also check for HTML <img> tag (PaddleOCR API uses this format)
            html_img_match = _HTML_IMG_PATTERN.search(line)
            if html_img_match:
                # Skip tiny images (icons/logos) based on width percentage
                width_match = _IMG_WIDTH_PATTERN.search(line)
                img_width_pct = int(width_match.group(1)) if width_match else 100
                if img_width_pct <= 10:
                    i += 1
                    continue
                img_ref = html_img_match.group(1)
                caption = ""
                resolved_path = ""
                if image_urls and img_ref in image_urls and images_dir:
                    img_url = image_urls[img_ref]
                    local_name = Path(img_ref).name or f"page{page_num}_img.jpg"
                    local_path = images_dir / local_name
                    if not local_path.exists():
                        try:
                            resp = requests.get(img_url, timeout=60)
                            resp.raise_for_status()
                            local_path.parent.mkdir(parents=True, exist_ok=True)
                            local_path.write_bytes(resp.content)
                            resolved_path = str(local_path)
                        except Exception as exc:
                            logger.warning("Failed to download image %s: %s", img_ref, exc)
                    else:
                        resolved_path = str(local_path)
                if resolved_path:
                    items.append({
                        "typer": "image",
                        "data": {"image_path": resolved_path, "caption": caption},
                        "page": page_num,
                        "block_label": "figure",
                    })
                i += 1
                continue
        if img_match:
            caption = img_match.group(1)
            img_ref = img_match.group(2)
            resolved_path = ""
            if image_urls and img_ref in image_urls and images_dir:
                img_url = image_urls[img_ref]
                local_name = Path(img_ref).name or f"page{page_num}_img.jpg"
                local_path = images_dir / local_name
                if not local_path.exists():
                    try:
                        resp = requests.get(img_url, timeout=60)
                        resp.raise_for_status()
                        local_path.parent.mkdir(parents=True, exist_ok=True)
                        local_path.write_bytes(resp.content)
                        resolved_path = str(local_path)
                    except Exception as exc:
                        logger.warning("Failed to download image %s: %s", img_ref, exc)
                else:
                    resolved_path = str(local_path)
            items.append({
                "typer": "image",
                "data": {"image_path": resolved_path, "caption": caption},
                "page": page_num,
                "block_label": "figure",
            })
            i += 1
            continue

        # Heading
        heading_match = _HEADING_PATTERN.match(line.strip())
        if heading_match:
            text = heading_match.group(2).strip()
            if text:
                items.append({
                    "typer": "paragraph",
                    "text": text,
                    "page": page_num,
                    "block_label": "title",
                })
            i += 1
            continue

        # Figure caption in <div> (PaddleOCR API format: <div>Figure N. ...</div>)
        fig_cap_match = _FIGURE_CAPTION_DIV_PATTERN.match(line.strip())
        if fig_cap_match:
            cap_text = fig_cap_match.group(1).strip()
            fig_num_match = _FIGURE_NUM_PATTERN.match(cap_text)
            fig_num = fig_num_match.group(1) if fig_num_match else ""
            if fig_num:
                items.append({
                    "typer": "image",
                    "data": {
                        "image_path": "",
                        "caption": cap_text,
                        "figure_num": fig_num,
                    },
                    "page": page_num,
                    "block_label": "figure",
                })
            else:
                items.append({
                    "typer": "paragraph",
                    "text": cap_text,
                    "page": page_num,
                })
            i += 1
            continue

        # Regular paragraph (collect consecutive non-empty non-special lines)
        para_lines = []
        while i < len(lines):
            curr = lines[i]
            if not curr.strip():
                i += 1
                break
            if curr.strip().startswith("$$"):
                break
            if _TABLE_LINE_PATTERN.match(curr.strip()):
                break
            if _IMAGE_PATTERN.match(curr.strip()):
                break
            if _HTML_IMG_PATTERN.search(curr):
                # Only break for non-tiny images
                w_match = _IMG_WIDTH_PATTERN.search(curr)
                if not w_match or int(w_match.group(1)) > 10:
                    break
            if _HEADING_PATTERN.match(curr.strip()):
                break
            if _FIGURE_CAPTION_DIV_PATTERN.match(curr.strip()):
                break
            para_lines.append(curr)
            i += 1

        text = " ".join(para_lines).strip()
        if text:
            # Check for inline formulas and mark them
            items.append({
                "typer": "paragraph",
                "text": text,
                "page": page_num,
            })

    return items


def convert_paddleocr_api_to_knowmat(
    pages_data: List[Dict[str, Any]],
    pdf_path: str,
    images_dir: Optional[Path] = None,
) -> Tuple[str, Dict[str, Any], List[Dict[str, Any]]]:
    """Convert PaddleOCR API JSONL page results to KnowMat format.

    Parameters
    ----------
    pages_data : list
        Parsed JSONL lines from PaddleOCR API (one dict per JSONL line).
    pdf_path : str
        Path to the source PDF.
    images_dir : Path, optional
        Directory to save downloaded images.

    Returns
    -------
    tuple of (extracted_text, metadata, ocr_items)
    """
    all_ocr_items: List[Dict[str, Any]] = []
    page_texts: List[str] = []
    total_pages = 0

    for line_idx, page_data in enumerate(pages_data):
        result = page_data.get("result", {})
        layout_results = result.get("layoutParsingResults", [])

        for layout_res in layout_results:
            total_pages += 1
            page_num = total_pages

            markdown_info = layout_res.get("markdown", {})
            md_text = markdown_info.get("text", "")
            image_urls = markdown_info.get("images", {})

            if md_text.strip():
                page_texts.append(md_text)

            page_items = _parse_markdown_blocks(
                md_text, page_num, images_dir, image_urls
            )
            all_ocr_items.extend(page_items)

    extracted_text = "\n\n".join(page_texts)

    table_count = sum(1 for it in all_ocr_items if it.get("typer") == "table")
    formula_count = sum(1 for it in all_ocr_items if it.get("typer") == "formula")
    image_count = sum(1 for it in all_ocr_items if it.get("typer") == "image")

    metadata: Dict[str, Any] = {
        "backend": "paddleocr_api",
        "pages": total_pages,
        "ocr_items": len(all_ocr_items),
        "ocr_quality": {
            "ocr_avg_confidence": None,
            "ocr_low_confidence_pages": [],
            "table_count": table_count,
            "formula_count": formula_count,
            "image_count": image_count,
            "ppstructure_status": "not_applicable",
            "ppstructure_detail": "parsed by PaddleOCR cloud API",
            "ppstructure_replacements": 0,
        },
    }

    return extracted_text, metadata, all_ocr_items


def clean_api_markdown(text: str) -> str:
    """Clean PaddleOCR/MinerU API markdown output.

    - Remove journal boilerplate ("You may also like", etc.)
    - Remove all image references (figures are replaced by AI descriptions)
    - Convert simple HTML tables to markdown (keep complex ones as HTML)
    """
    from ppmatAgent.knowmat.pdf.html_cleaner import (
        _has_complex_cell_attributes,
        _html_table_to_markdown,
    )

    if not text:
        return ""

    # 1. Strip "You may also like" section (heading + bullet list + images)
    result = _YOU_MAY_ALSO_LIKE_PATTERN.sub("", text)

    # 2. Remove all image references (markdown ![](images/...) and HTML <div><img></div>)
    result = re.sub(r"!\[[^\]]*\]\([^)]*\)\s*", "", result)
    result = re.sub(
        r'<div[^>]*>\s*<img\s+[^>]*/?>.*?</div>\s*',
        "",
        result,
        flags=re.DOTALL | re.IGNORECASE,
    )

    # 3. Convert simple HTML tables to markdown
    def _convert_table(m: re.Match) -> str:
        table_html = m.group(0)
        if _has_complex_cell_attributes(table_html):
            return table_html
        md_table = _html_table_to_markdown(table_html)
        return md_table if md_table != table_html else table_html

    result = _HTML_TABLE_PATTERN.sub(_convert_table, result)

    # 4. Collapse excessive blank lines
    result = re.sub(r"\n{4,}", "\n\n\n", result)

    return result.strip()


def extract_formulas_per_page(
    pages_data: List[Dict[str, Any]],
) -> Dict[int, List[str]]:
    """Extract formula LaTeX strings per page from PP-StructureV3 API result.

    Returns dict of {page_number: [latex1, latex2, ...]} in reading order.
    Used for PP-StructureV3 formula refinement on other backends (MinerU, etc.).
    """
    formulas_by_page: Dict[int, List[str]] = {}
    page_num = 0

    for page_data in pages_data:
        result = page_data.get("result", {})
        layout_results = result.get("layoutParsingResults", [])

        for layout_res in layout_results:
            page_num += 1
            md_text = layout_res.get("markdown", {}).get("text", "")

            page_formulas: List[str] = []
            # Find display formulas
            for match in _FORMULA_PATTERN.finditer(md_text):
                latex = match.group(1).strip()
                if latex:
                    page_formulas.append(latex)

            if page_formulas:
                formulas_by_page[page_num] = page_formulas

    return formulas_by_page


def extract_tables_per_page(
    pages_data: List[Dict[str, Any]],
) -> Dict[int, List[str]]:
    """Extract markdown table strings per page from PP-StructureV3 API result.

    Returns dict of {page_number: [table_md_1, table_md_2, ...]} in reading order.
    """
    tables_by_page: Dict[int, List[str]] = {}
    page_num = 0

    for page_data in pages_data:
        result = page_data.get("result", {})
        layout_results = result.get("layoutParsingResults", [])

        for layout_res in layout_results:
            page_num += 1
            md_text = layout_res.get("markdown", {}).get("text", "")

            page_tables: List[str] = []
            lines = md_text.split("\n")
            i = 0
            while i < len(lines):
                if _TABLE_LINE_PATTERN.match(lines[i].strip()):
                    table_lines = []
                    while i < len(lines) and _TABLE_LINE_PATTERN.match(lines[i].strip()):
                        table_lines.append(lines[i])
                        i += 1
                    table_md = "\n".join(table_lines)
                    if table_md.strip():
                        page_tables.append(table_md)
                else:
                    i += 1

            if page_tables:
                tables_by_page[page_num] = page_tables

    return tables_by_page
