"""
LangGraph node for converting internal extraction format to the target schema.

Wraps :class:`~ppmatAgent.knowmat.schema_converter.SchemaConverter` so that the
conversion is part of the graph checkpoint and its output is recorded
in ``KnowMatState.final_data``.
"""

import logging
from typing import Any, Dict

from ppmatAgent.knowmat.schema_converter import SchemaConverter
from ppmatAgent.knowmat.states import KnowMatState

logger = logging.getLogger(__name__)


def convert_to_target_schema(state: KnowMatState) -> Dict[str, Any]:
    """Convert internal extraction dict to the target HEA dataset schema.

    Reads ``final_data``, ``pdf_path``, ``paper_text``, and
    ``document_metadata`` from state and returns the converted data
    under ``final_data``.
    """
    final_data = state.get("final_data")
    if not final_data:
        final_data = state.get("latest_extracted_data", {})

    pdf_path = state.get("pdf_path", "")

    converter = SchemaConverter()
    converted = converter.convert(
        final_data,
        pdf_path,
        paper_text=state.get("paper_text"),
        document_metadata=state.get("document_metadata"),
    )
    return {"final_data": converted}
