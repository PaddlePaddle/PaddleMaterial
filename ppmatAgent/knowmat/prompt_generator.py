"""Prompt generation utilities for ppmatAgent.knowmat.

Constructs system and user prompts for the extraction agent.
The system prompt includes routing supplements loaded based on the
paper's three-dimensional classification.
"""

from ppmatAgent.knowmat.prompt_loader import load_text_template


def generate_system_prompt(routing_supplements: str = "") -> str:
    """Return the system prompt for the extraction agent.

    Parameters
    ----------
    routing_supplements : str
        Concatenated supplement text loaded from routing classification.
        Injected at the ``{routing_supplements}`` placeholder in the template.
    """
    template = load_text_template("extraction_system_template.txt")
    return template.replace("{routing_supplements}", routing_supplements)


def generate_user_prompt(
    paper_text: str,
    target_base_material: str = "",
    target_application: str = "",
    target_research_paradigm: str = "",
    target_patch_tags: str = "",
) -> str:
    """Wrap the user message around the paper text with routing hints.

    Parameters
    ----------
    paper_text : str
        The full OCR-parsed paper text.
    target_base_material, target_application, target_research_paradigm : str
        Routing hints from the classification step.
    target_patch_tags : str
        Comma-separated patch tags hint.
    """
    template = load_text_template("extraction_user_template.txt")
    return (
        template
        .replace("{paper_text}", paper_text)
        .replace("{target_base_material}", target_base_material)
        .replace("{target_application}", target_application)
        .replace("{target_research_paradigm}", target_research_paradigm)
        .replace("{target_patch_tags}", target_patch_tags)
    )
