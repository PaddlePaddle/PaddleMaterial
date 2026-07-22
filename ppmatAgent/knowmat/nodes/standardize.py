"""
LangGraph node for optional property-name standardization.

Wraps :class:`~ppmatAgent.knowmat.post_processing.PostProcessor` so that the
standardization step participates in the graph checkpoint and its
results are recorded in :class:`~ppmatAgent.knowmat.states.KnowMatState`.
"""

import os
import logging
from typing import Any, Dict

from ppmatAgent.knowmat.states import KnowMatState

logger = logging.getLogger(__name__)


def standardize_properties(state: KnowMatState) -> Dict[str, Any]:
    """Standardize extracted property names against ``properties.json``.

    When ``state["enable_property_standardization"]`` is falsy the node
    returns immediately without modification (pass-through).

    Returns
    -------
    dict
        ``final_data`` with standardized property fields, or unchanged.
    """
    if not state.get("enable_property_standardization"):
        return {}

    final_data = state.get("final_data", {})
    if not final_data or not final_data.get("compositions"):
        return {}

    api_key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    if not api_key:
        logger.warning("LLM_API_KEY not found. Skipping property standardization.")
        return {}

    properties_file = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "properties.json"
    )
    if not os.path.exists(properties_file):
        logger.warning("properties.json not found at %s. Skipping.", properties_file)
        return {}

    from ppmatAgent.knowmat.post_processing import PostProcessor
    from ppmatAgent.knowmat.app_config import settings

    try:
        processor = PostProcessor(
            properties_file=properties_file,
            api_key=api_key,
            base_url=base_url,
            gpt_model=settings.flagging_model or settings.model_name,
        )
        mock_result = [{"data": final_data}]
        processor.update_extracted_json(mock_result)
        standardized = mock_result[0]["data"]
        processor._print_match_stats()
        logger.info("Property standardization complete")
        return {"final_data": standardized}
    except Exception as exc:
        logger.warning("Property standardization failed: %s", exc)
        return {}
