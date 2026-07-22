"""
Backward-compatible wrapper for the legacy docling parser module.

The pipeline now uses PaddleOCR-VL. Existing imports of
`parse_pdf_with_docling` are mapped to the new parser implementation.
"""

from ppmatAgent.knowmat.nodes.paddleocrvl_parse_pdf import parse_pdf_with_paddleocrvl


def parse_pdf_with_docling(state):
    """Compatibility alias to the PaddleOCR-VL parser."""
    return parse_pdf_with_paddleocrvl(state)
