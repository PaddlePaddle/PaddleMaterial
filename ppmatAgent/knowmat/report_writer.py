"""
Report generation for ppmatAgent.knowmat.

Writes a human-readable analysis report summarising extraction runs,
manager aggregation decisions, and flagging results.
"""

from __future__ import annotations

import textwrap
from typing import IO, Any, Dict, List

from ppmatAgent.knowmat.states import load_run_extraction


def write_comprehensive_report(f: IO[str], final_state: Dict[str, Any]) -> None:
    """Write a comprehensive analysis report to the file handle *f*."""

    f.write(f"{'=' * 80}\n")
    f.write("KNOWMAT 2.0 EXTRACTION ANALYSIS REPORT\n")
    f.write(f"{'=' * 80}\n\n")

    _write_final_assessment(f, final_state)
    _write_aggregation_section(f, final_state)
    _write_human_review_section(f, final_state)
    _write_per_run_analysis(f, final_state)
    _write_statistics(f, final_state)

    f.write(f"\n{'=' * 80}\n")
    f.write("END OF REPORT\n")
    f.write(f"{'=' * 80}\n")


# ------------------------------------------------------------------
# Internal section writers
# ------------------------------------------------------------------

def _write_final_assessment(f: IO[str], state: Dict[str, Any]) -> None:
    f.write(f"{'█' * 80}\n")
    f.write("FINAL ASSESSMENT\n")
    f.write(f"{'█' * 80}\n\n")

    final_confidence = state.get("final_confidence_score", "N/A")
    needs_review = state.get("needs_human_review", True)
    confidence_rationale = state.get("confidence_rationale", "")

    f.write(f"Final Confidence Score: {final_confidence}\n")
    f.write(f"Human Review Required: {'Yes' if needs_review else 'No'}\n")
    f.write(f"Review Flag: {'FLAGGED' if needs_review else 'PASSED'}\n\n")

    if confidence_rationale:
        f.write("Confidence Assessment Rationale:\n")
        f.write(f"{'-' * 40}\n")
        wrapped = textwrap.fill(confidence_rationale, width=80, subsequent_indent="  ")
        f.write(f"{wrapped}\n\n")


def _write_aggregation_section(f: IO[str], state: Dict[str, Any]) -> None:
    f.write(f"{'█' * 80}\n")
    f.write("MANAGER AGGREGATION ANALYSIS\n")
    f.write(f"{'█' * 80}\n\n")

    aggregation_rationale = state.get("aggregation_rationale", "")
    if aggregation_rationale:
        f.write("How Data Was Combined:\n")
        f.write(f"{'-' * 25}\n")
        wrapped = textwrap.fill(aggregation_rationale, width=80, subsequent_indent="  ")
        f.write(f"{wrapped}\n\n")


def _write_human_review_section(f: IO[str], state: Dict[str, Any]) -> None:
    f.write(f"{'█' * 80}\n")
    f.write("HUMAN REVIEW GUIDANCE\n")
    f.write(f"{'█' * 80}\n\n")

    guide = state.get("human_review_guide", "")
    if guide:
        f.write("Items to Double-Check:\n")
        f.write(f"{'-' * 22}\n")
        wrapped = textwrap.fill(guide, width=80, subsequent_indent="  ")
        f.write(f"{wrapped}\n\n")


def _write_per_run_analysis(f: IO[str], state: Dict[str, Any]) -> None:
    f.write(f"{'█' * 80}\n")
    f.write("INDIVIDUAL RUN ANALYSIS\n")
    f.write(f"{'█' * 80}\n\n")

    run_results: List[Dict[str, Any]] = state.get("run_results", [])
    for i, run in enumerate(run_results, 1):
        f.write(f"{'▓' * 60}\n")
        f.write(f"RUN {run.get('run_id', i)} DETAILS\n")
        f.write(f"{'▓' * 60}\n")
        f.write(f"Confidence Score: {run.get('confidence_score', 0.0):.2f}\n\n")

        rationale_text = run.get("rationale", "N/A")
        f.write("Evaluation Rationale:\n")
        wrapped = textwrap.fill(rationale_text, width=80, subsequent_indent="  ")
        f.write(f"  {wrapped}\n\n")

        missing = run.get("missing_fields") or []
        if missing:
            f.write(f"Missing Fields ({len(missing)} items):\n")
            for j, field in enumerate(missing[:15], 1):
                f.write(f"  {j:2d}. {field}\n")
            if len(missing) > 15:
                f.write(f"      ... and {len(missing) - 15} more items\n")
            f.write("\n")

        hallucinated = run.get("hallucinated_fields") or []
        if hallucinated:
            f.write(f"Hallucinated Fields ({len(hallucinated)} items):\n")
            for j, field in enumerate(hallucinated[:15], 1):
                f.write(f"  {j:2d}. {field}\n")
            if len(hallucinated) > 15:
                f.write(f"      ... and {len(hallucinated) - 15} more items\n")
            f.write("\n")

        suggestions = run.get("suggested_prompt")
        if suggestions and suggestions.strip():
            f.write("Improvement Suggestions:\n")
            wrapped = textwrap.fill(suggestions, width=80, subsequent_indent="  ")
            f.write(f"  {wrapped}\n\n")

        extracted_data = load_run_extraction(run)
        compositions = extracted_data.get("compositions", [])
        f.write(f"Compositions Extracted: {len(compositions)}\n")
        if compositions:
            f.write("Sample Compositions: ")
            comp_names = [comp.get("composition", "Unknown") for comp in compositions[:3]]
            f.write(", ".join(comp_names))
            if len(compositions) > 3:
                f.write(f" (and {len(compositions) - 3} more)")
            f.write("\n")
        f.write("\n")


def _write_statistics(f: IO[str], state: Dict[str, Any]) -> None:
    f.write(f"{'█' * 80}\n")
    f.write("EXTRACTION STATISTICS\n")
    f.write(f"{'█' * 80}\n\n")

    run_results: List[Dict[str, Any]] = state.get("run_results", [])
    if run_results:
        scores = [run.get("confidence_score", 0.0) for run in run_results]
        f.write(f"Number of Extraction Runs: {len(run_results)}\n")
        f.write(f"Average Run Confidence: {sum(scores) / len(scores):.2f}\n")
        f.write(f"Best Run Confidence: {max(scores):.2f}\n")
        f.write(f"Worst Run Confidence: {min(scores):.2f}\n\n")

        total_missing = sum(len(run.get("missing_fields") or []) for run in run_results)
        total_hallucinated = sum(len(run.get("hallucinated_fields") or []) for run in run_results)
        f.write(f"Total Missing Fields Across Runs: {total_missing}\n")
        f.write(f"Total Hallucinated Fields Across Runs: {total_hallucinated}\n\n")

    final_data = state.get("final_data", {})
    final_compositions = final_data.get("compositions", [])
    f.write(f"Final Extracted Compositions: {len(final_compositions)}\n")
    if final_compositions:
        total_props = sum(
            len(comp.get("properties_of_composition", [])) for comp in final_compositions
        )
        f.write(f"Total Properties in Final Result: {total_props}\n")
