"""
ppmatAgent KnowMat workflow
===========================

This package contains the PaddleMaterials-integrated KnowMat data extraction
pipeline.

It follows the architecture of the MI‑Agent project and uses LangGraph to wire
together several agents (nodes) that collaborate to perform end‑to‑end
extraction of structured materials science information from unstructured PDF
documents.  The agents include:

* A parser that reads PDF files and extracts their text while stripping out
  references.
* A sub‑field detection agent that determines which niche area of materials
  science a paper belongs to (e.g. experimental, computational, simulation,
  machine learning or hybrid) and generates an updated extraction prompt
  tailored to that sub‑field.
* An extraction agent that uses a GPT model via TrustCall to produce a
  structured JSON representation of compositions, processing conditions,
  characterisation details and properties.
* An evaluation agent that compares the extracted data against the source
  document, assigns a confidence score, provides a rationale, and optionally
  suggests prompt refinements.  The evaluation agent is capable of running up
  to three extraction/evaluation cycles to improve quality.
* A manager agent that aggregates multiple extraction runs into a single
  final result and flags extractions that require human review.

The package also defines the shared state used by the LangGraph workflow, the
Pydantic schemas for TrustCall extractors and a top‑level ``run`` function
invoked by the CLI.

``run`` is exposed here as a lazy wrapper so that importing any subpackage
(e.g. ``ppmatAgent.knowmat.image_text_alignment``) does not eagerly load the full
LangGraph pipeline and its heavy dependencies (torch models, etc.).
There is no circular-import issue; the lazy import is purely a load-time
performance guard.
"""


def run(*args, **kwargs):
    # Deferred import: keeps `import ppmatAgent.knowmat` lightweight for callers that only
    # need subpackages (e.g. image_text_alignment) without the full pipeline.
    from .orchestrator import run as _run
    return _run(*args, **kwargs)
