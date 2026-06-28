"""PDF parsing helper modules for KnowMat."""

from .doi_extractor import extract_first_doi, extract_doi_from_pdf_metadata
from .html_cleaner import convert_html_to_markdown, html_table_to_structured
from .section_normalizer import (
    normalize_alloy_strings,
    normalize_leading_masthead_and_title,
    normalize_plain_author_superscripts,
    repair_keywords_abstract_two_column_ocr,
    structure_sections,
    strip_references_section,
)
from .ocr_engine import (
    default_model_dir,
    create_ocr_engine,
    supports_batch_predict,
    run_ocr_batch,
    run_ocr_sequential,
    collect_text,
    normalize_lines,
)
from .blocks import block_to_item, text_to_paragraph_items

__all__ = [
    "extract_first_doi",
    "extract_doi_from_pdf_metadata",
    "convert_html_to_markdown",
    "html_table_to_structured",
    "normalize_alloy_strings",
    "normalize_leading_masthead_and_title",
    "normalize_plain_author_superscripts",
    "repair_keywords_abstract_two_column_ocr",
    "structure_sections",
    "strip_references_section",
    "default_model_dir",
    "create_ocr_engine",
    "supports_batch_predict",
    "run_ocr_batch",
    "run_ocr_sequential",
    "collect_text",
    "normalize_lines",
    "block_to_item",
    "text_to_paragraph_items",
]