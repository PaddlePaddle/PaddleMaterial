"""Node for three-dimensional paper routing and supplement loading.

This node classifies the paper along three axes (base_material, application,
research_paradigm) and loads the corresponding supplement files that define
domain-specific extraction field structures and priorities.
"""

import logging
from typing import Dict, List

from ppmatAgent.knowmat.extractors import routing_extractor, PaperRouting
from ppmatAgent.knowmat.prompt_loader import load_routing_supplement, load_yaml_templates_required
from ppmatAgent.knowmat.states import KnowMatState

logger = logging.getLogger(__name__)

_ROUTING_PROMPT_TEMPLATE = load_yaml_templates_required(
    "subfield_detection.yaml", ("prompt_template",)
).get("prompt_template", "")

_BASE_MAP = {
    "Metals": "base/Metals.txt",
    "Ceramics_Inorganic": "base/Ceramics_Inorganic.txt",
    "Polymers": "base/Polymers.txt",
    "Composites": "base/Composites.txt",
    "Organic_Small_Molecules": "base/Organic_Small_Molecules.txt",
}

_APPLICATION_MAP = {
    "Structural": "application/Structural.txt",
    "Functional": "application/Functional.txt",
}

_PARADIGM_MAP = {
    "Experimental": "paradigm/Experimental.txt",
    "Pure_Simulation": "paradigm/Pure_Simulation.txt",
    "Hybrid": "paradigm/Hybrid.txt",
}

_DOMAIN_MAP = {
    "Machining": "domain/Machining.txt",
    "Coating": "domain/Coating.txt",
    "Battery": "domain/Battery.txt",
}


def _build_routing_supplements(routing: Dict) -> str:
    """Load and concatenate supplement files based on routing classification."""
    parts: List[str] = []
    supplements_loaded: List[str] = []

    base = routing.get("base_material", "")
    if base in _BASE_MAP:
        content = load_routing_supplement(_BASE_MAP[base])
        if content:
            parts.append(f"=== Base Material Supplement: {base} ===\n{content}")
            supplements_loaded.append(f"directions/{_BASE_MAP[base]}")

    app = routing.get("application", "")
    if app in _APPLICATION_MAP:
        content = load_routing_supplement(_APPLICATION_MAP[app])
        if content:
            parts.append(f"=== Application Supplement: {app} ===\n{content}")
            supplements_loaded.append(f"directions/{_APPLICATION_MAP[app]}")

    paradigm = routing.get("research_paradigm", "")
    if paradigm in _PARADIGM_MAP:
        content = load_routing_supplement(_PARADIGM_MAP[paradigm])
        if content:
            parts.append(f"=== Paradigm Supplement: {paradigm} ===\n{content}")
            supplements_loaded.append(f"directions/{_PARADIGM_MAP[paradigm]}")

    for domain in routing.get("domain_overlays", []):
        if domain in _DOMAIN_MAP:
            content = load_routing_supplement(_DOMAIN_MAP[domain])
            if content:
                parts.append(f"=== Domain Overlay: {domain} ===\n{content}")
                supplements_loaded.append(f"directions/{_DOMAIN_MAP[domain]}")

    routing["supplements_loaded"] = supplements_loaded
    return "\n\n".join(parts)


def detect_sub_field(state: KnowMatState) -> dict:
    """Classify the paper and load routing supplements.

    Returns
    -------
    dict
        Updates containing ``paper_routing``, ``routing_supplements``,
        and ``sub_field`` (for backwards compatibility).
    """
    paper_text = state.get("paper_text", "")
    prompt = _ROUTING_PROMPT_TEMPLATE.format(paper_text=paper_text)

    result = routing_extractor.invoke(prompt)
    response = result.get("responses", [None])[0]

    if not response:
        logger.warning("Routing extractor returned no response, using defaults")
        routing = {
            "base_material": "Metals",
            "application": "Structural",
            "research_paradigm": "Experimental",
            "domain_overlays": [],
            "patch_tags": [],
        }
    elif isinstance(response, PaperRouting):
        routing = {
            "base_material": response.base_material,
            "application": response.application,
            "research_paradigm": response.research_paradigm,
            "domain_overlays": response.domain_overlays,
            "patch_tags": response.patch_tags,
        }
    else:
        routing = dict(response)

    supplements_text = _build_routing_supplements(routing)

    logger.info(
        "Paper routing: %s × %s × %s | overlays=%s | supplements=%d chars",
        routing.get("base_material"),
        routing.get("application"),
        routing.get("research_paradigm"),
        routing.get("domain_overlays", []),
        len(supplements_text),
    )

    return {
        "paper_routing": routing,
        "routing_supplements": supplements_text,
        "sub_field": routing.get("base_material", "experimental"),
    }
