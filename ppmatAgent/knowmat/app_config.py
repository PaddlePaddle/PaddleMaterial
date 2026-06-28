"""
Application configuration for ppmatAgent.knowmat.

This module defines a ``Settings`` class using pydantic's ``BaseSettings``
mechanism to manage environment‑configurable options such as the default
output directory, the model to use and the generation temperature.

Environment variables are prefixed with ``KNOWMAT2_``.  For example,
``KNOWMAT2_OUTPUT_DIR`` overrides the default output directory and
``KNOWMAT2_MODEL_NAME`` changes the base model.  See the attributes of
``Settings`` for supported options.
"""

import os
from typing import Any

try:
    from pydantic_settings import BaseSettings
    from pydantic import ConfigDict
except ImportError:  # pragma: no cover - optional agent dependency
    BaseSettings = object  # type: ignore[assignment]
    ConfigDict = None  # type: ignore[assignment]

from ppmatAgent.knowmat.env_loader import load_project_dotenv

# Load .env early so model defaults can read LLM_MODEL before settings instantiation.
load_project_dotenv(override=False)

DEFAULT_LLM_MODEL = os.getenv("LLM_MODEL", "gpt-5")


class Settings(BaseSettings):
    """Configuration options for ppmatAgent.knowmat.

    Attributes
    ----------
    input_dir: str
        Default folder where raw ``.pdf``/``.txt`` files are stored. Defaults to
        ``"data/raw"`` relative to the current working directory.

    output_dir: str
        Where extracted results and artifacts will be written (LLM extraction
        JSON, reports, etc.).  Defaults to ``"data/output"`` so that raw/OCR
        (data/raw) and extraction output are kept separate.
    
    model_name: str
        The default model name for all agents.  Defaults to ``LLM_MODEL`` when
        set, otherwise ``"gpt-5"``.
    
    temperature: float
        Sampling temperature when generating with the language model.  A
        temperature of 0 yields deterministic outputs.  The default is 0.0.
        Note: GPT-5 models don't support custom temperature settings.
    
    subfield_model: str
        Model for subfield detection agent. Defaults to ``LLM_MODEL``.
    
    extraction_model: str
        Model for extraction agent. Defaults to ``LLM_MODEL``.
    
    evaluation_model: str
        Model for evaluation agent. Defaults to ``LLM_MODEL``.
    
    manager_model: str
        Model for validation agent (Stage 2: hallucination correction).
        Note: "manager_model" name kept for backward compatibility.
        Defaults to ``LLM_MODEL``.
    
    flagging_model: str
        Model for flagging/quality assessment agent. Defaults to ``LLM_MODEL``.

    trim_references_section: bool
        Whether to trim content after References/Bibliography/Citations during parsing.
        Defaults to ``False`` to preserve full text (including appendix/supplementary).

    figure_description_enabled: bool
        When ``True``, uses a multimodal LLM to generate a textual description of
        each detected figure image and inserts it into ``paper_text`` above the
        corresponding figure caption.  Requires ``LLM_MODEL`` to support vision.
        Defaults to ``True``.  Disable via ``KNOWMAT2_FIGURE_DESCRIPTION_ENABLED=0``.

    chart_digitization_enabled: bool
        When ``True``, figure crops whose OCR filename marks them as charts
        (``img_in_chart_box_*``) are routed to a VLM digitization pass instead
        of prose: bar charts → CSV, line charts → key-points + trend summary,
        injected as ``> [Figure N VLM-digitized]`` blocks that the extraction
        stage may consume as ``image_digitized`` estimates.  Non-chart crops and
        non-digitizable charts (XRD/micrograph) still get prose descriptions.
        Defaults to ``True``.  Disable via ``KNOWMAT2_CHART_DIGITIZATION_ENABLED=0``.
    """

    # IO defaults (can be overridden by env or CLI)
    input_dir: str = "data/raw"
    output_dir: str = "data/output"
    model_name: str = DEFAULT_LLM_MODEL
    temperature: float = 0.0  # Note: ignored for GPT-5 models
    
    # Per-agent model configuration
    subfield_model: str = DEFAULT_LLM_MODEL
    extraction_model: str = DEFAULT_LLM_MODEL
    evaluation_model: str = DEFAULT_LLM_MODEL
    manager_model: str = DEFAULT_LLM_MODEL
    flagging_model: str = DEFAULT_LLM_MODEL
    trim_references_section: bool = False
    figure_description_enabled: bool = True
    chart_digitization_enabled: bool = True

    if ConfigDict is not None:
        model_config = ConfigDict(env_prefix="KNOWMAT2_")

    def __init__(self, **kwargs: Any):
        if ConfigDict is not None:
            super().__init__(**kwargs)
            return

        prefix = "KNOWMAT2_"
        annotations = getattr(self, "__annotations__", {})
        for name, typ in annotations.items():
            default = getattr(type(self), name)
            raw = os.getenv(prefix + name.upper())
            if raw is None:
                value = kwargs.get(name, default)
            elif typ is bool:
                value = raw.strip().lower() in {"1", "true", "yes", "on"}
            elif typ is float:
                value = float(raw)
            else:
                value = raw
            setattr(self, name, value)


# Singleton instance to be imported throughout the package
settings = Settings()
