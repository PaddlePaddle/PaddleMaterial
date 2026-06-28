"""Exporter for converting alignment results to KnowMat JSON format."""

from __future__ import annotations

from typing import Any, Dict, List

from .aligner import ImageTextAlignment, RelatedSentence


def export_to_knowmat_format(
    alignments: List[ImageTextAlignment],
) -> List[Dict[str, Any]]:
    """
    Convert alignment results to KnowMat JSON format.

    Parameters
    ----------
    alignments : List[ImageTextAlignment]
        Alignment results from ImageTextAligner.

    Returns
    -------
    List[Dict[str, Any]]
        Alignment data in KnowMat format for inclusion in extraction.json.
    """
    result = []

    for alignment in alignments:
        alignment_dict = {
            "image_id": alignment.image_id,
            "image_path": alignment.image_path,
            "figure_num": alignment.figure_num,
            "caption": alignment.caption,
            "normalized_figure_id": alignment.normalized_figure_id,
            "figure_number": alignment.figure_number,
            "subfigure_id": alignment.subfigure_id,
            "all_figure_ids": alignment.all_figure_ids,
            "page_number": alignment.page_number,
            "related_sentences": [
                {
                    "text": rs.text,
                    "page": rs.page,
                    "section": rs.section,
                    "score": rs.score,
                    "cosine_score": rs.cosine_score,
                    "caption_text_cosine": rs.caption_text_cosine,
                    "rank": rs.rank,
                    "mentioned_figures": rs.mentioned_figures,
                    "has_same_figure_anchor": rs.has_same_figure_anchor,
                    "has_wrong_figure_anchor": rs.has_wrong_figure_anchor,
                    "source": rs.source,
                    "token_id": rs.token_id,
                    "confidence": rs.confidence,
                }
                for rs in alignment.related_sentences
            ],
        }
        result.append(alignment_dict)

    return result


def add_to_extraction_json(
    extraction_data: Dict[str, Any],
    alignments: List[ImageTextAlignment],
) -> Dict[str, Any]:
    """
    Add alignment results to existing extraction JSON.

    Parameters
    ----------
    extraction_data : Dict[str, Any]
        Existing extraction data from KnowMat.
    alignments : List[ImageTextAlignment]
        Alignment results to add.

    Returns
    -------
    Dict[str, Any]
        Updated extraction data with alignments added.
    """
    # Convert alignments to KnowMat format
    alignment_data = export_to_knowmat_format(alignments)

    # Add to extraction data
    result = dict(extraction_data)
    result["Image_Text_Alignments"] = alignment_data

    # Also add summary to metadata
    if "Paper_Metadata" not in result:
        result["Paper_Metadata"] = {}

    result["Paper_Metadata"]["image_text_alignment"] = {
        "num_images": len(alignments),
        "num_related_sentences": sum(len(a.related_sentences) for a in alignments),
        "alignment_model": "clip",  # Could be made configurable
    }

    return result
