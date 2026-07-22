"""Image-text alignment module for KnowMat.

This module provides image-text alignment functionality, allowing
automatic association of figure images with relevant sentences
from the paper text.

Usage
-----
    from ppmatAgent.knowmat.image_text_alignment import ImageTextAligner

    aligner = ImageTextAligner(device="cpu")
    alignments = aligner.align(ocr_items, paper_id)
"""

from .aligner import ImageTextAligner, ImageTextAlignment, RelatedSentence
from .config import AlignmentConfig
from .exporter import add_to_extraction_json, export_to_knowmat_format
from .tokenizer import OcrItemTokenizer, SentenceToken, VisualToken

__all__ = [
    "ImageTextAligner",
    "ImageTextAlignment",
    "RelatedSentence",
    "AlignmentConfig",
    "OcrItemTokenizer",
    "VisualToken",
    "SentenceToken",
    "export_to_knowmat_format",
    "add_to_extraction_json",
]
