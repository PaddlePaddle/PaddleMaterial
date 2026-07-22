"""LangGraph node for image-text alignment."""

import logging

from ppmatAgent.knowmat.app_config import settings
from ppmatAgent.knowmat.image_text_alignment import ImageTextAligner
from ppmatAgent.knowmat.image_text_alignment.exporter import export_to_knowmat_format
from ppmatAgent.knowmat.states import KnowMatState

logger = logging.getLogger(__name__)


def align_images_with_text(state: KnowMatState) -> dict:
    """
    Execute image-text alignment node.

    Extracts images and sentences from ocr_items, computes embeddings
    and similarity, and returns alignment results.

    Parameters
    ----------
    state : KnowMatState
        The current workflow state. Must contain 'ocr_items'.

    Returns
    -------
    dict
        Updates containing 'image_text_alignments'.
    """
    # Check if alignment is enabled
    if not getattr(settings, "alignment_enabled", True):
        logger.info("Image-text alignment is disabled")
        return {"image_text_alignments": []}

    ocr_items = state.get("ocr_items", [])
    if not ocr_items:
        logger.warning("No ocr_items found, skipping image-text alignment")
        return {"image_text_alignments": []}

    # Extract paper_id from pdf_path
    pdf_path = state.get("pdf_path", "")
    import os

    paper_id = os.path.splitext(os.path.basename(pdf_path))[0]

    # Get output directory
    output_dir = state.get("output_dir")

    # Create aligner
    aligner = ImageTextAligner(
        model=getattr(settings, "alignment_model", "clip"),
        device=getattr(settings, "alignment_device", "cpu"),
        top_k=getattr(settings, "alignment_top_k", 5),
        min_score=getattr(settings, "alignment_min_score", 0.0),
        batch_size=getattr(settings, "alignment_batch_size", 32),
        caption_blend=getattr(settings, "alignment_caption_blend", 0.0),
        save_dataset=getattr(settings, "alignment_save_dataset", False),
    )

    # Perform alignment
    logger.info("Running image-text alignment for paper: %s", paper_id)
    alignments = aligner.align(ocr_items, paper_id, output_dir)

    # Convert to dict format for JSON serialization
    alignment_dicts = export_to_knowmat_format(alignments)

    logger.info(
        "Image-text alignment complete: %d images aligned",
        len(alignment_dicts),
    )

    return {"image_text_alignments": alignment_dicts}
