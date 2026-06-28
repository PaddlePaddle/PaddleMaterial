"""
Validation Agent (Stage 2 of Two-Stage Manager)

This agent's responsibility is to validate and correct the merged data from Stage 1.
It contains ALL the hallucination correction logic, safety checks, and quality validation
from the original Manager agent.

Responsibilities:
- Detect hallucinations using evaluation feedback
- Correct hallucinations when evaluation provides the fix
- Validate ML-ready format compliance
- Generate human review guide
- Apply ALL safety checks (placeholder detection, lazy fallback, etc.)
- Retry mechanism with stronger prompts if needed

Inputs from Stage 1:
- aggregated_data: Merged compositions from all runs
- aggregation_notes: How the data was merged
- run_results: Original evaluation feedback (for hallucination detection)
"""

import json
from typing import Dict, Any

from ppmatAgent.knowmat.extractors import manager_extractor, ManagerFeedback, CompositionList
from ppmatAgent.knowmat.prompt_loader import load_yaml_templates_required
from ppmatAgent.knowmat.states import KnowMatState, load_run_extraction

_VALIDATOR_TEMPLATES = load_yaml_templates_required(
    "validator.yaml",
    (
        "system",
        "stage1_notes_prefix",
        "aggregated_data_prefix",
        "evaluation_feedback_header",
        "validation_tail",
    ),
)


def _first_response(result: Dict[str, Any]) -> Any:
    responses = result.get("responses")
    if not responses:
        return None
    return responses[0]


def _coerce_final_extracted_data(final_extracted_data: Any) -> Dict[str, Any] | None:
    if isinstance(final_extracted_data, str):
        try:
            final_extracted_data = json.loads(final_extracted_data)
        except Exception:
            return None
    if isinstance(final_extracted_data, dict) and "compositions" in final_extracted_data:
        return final_extracted_data
    if hasattr(final_extracted_data, "compositions"):
        return {"compositions": final_extracted_data.compositions}
    return None


def _composition_signature(comp: Dict[str, Any]) -> tuple[str, str, str]:
    name = str(comp.get("alloy_name_raw") or comp.get("composition") or "").strip().lower()
    process = str(comp.get("process_category") or "").strip().lower()
    conditions = str(comp.get("processing_conditions") or "").strip().lower()
    return name, process, conditions


def _validator_collapsed_item_boundaries(
    aggregated_data: Dict[str, Any],
    final_data: Dict[str, Any],
) -> bool:
    aggregated = aggregated_data.get("compositions", []) or []
    validated = final_data.get("compositions", []) or []
    if len(aggregated) <= 1 or len(validated) >= len(aggregated):
        return False
    aggregated_signatures = {_composition_signature(comp) for comp in aggregated}
    validated_signatures = {_composition_signature(comp) for comp in validated}
    if len(validated_signatures) < len(aggregated_signatures):
        return True
    return not aggregated_signatures.issubset(validated_signatures)


def _preserve_aggregated_item_boundaries(
    aggregated_data: Dict[str, Any],
    aggregation_notes: str,
    aggregation_rationale: str,
    human_review_guide: str,
) -> Dict[str, Any]:
    combined_rationale = (
        f"STAGE 1 - AGGREGATION:\n{aggregation_notes}\n\n"
        "STAGE 2 - VALIDATION & CORRECTION:\n"
        f"{aggregation_rationale}\n\n"
        "SAFETY OVERRIDE:\n"
        "Validator output collapsed distinct laboratory item boundaries, so the aggregated data was preserved."
    )
    review_guide = (
        human_review_guide.strip() + "\n\nPreserved aggregated item boundaries because validation collapsed distinct variants."
        if human_review_guide.strip()
        else "Preserved aggregated item boundaries because validation collapsed distinct variants."
    )
    return {
        "final_data": aggregated_data,
        "aggregation_rationale": combined_rationale,
        "human_review_guide": review_guide,
    }


def validate_and_correct(state: KnowMatState) -> Dict[str, Any]:
    """Validate merged data and correct hallucinations.
    
    Takes the aggregated data from Stage 1 and:
    1. Checks for hallucinations using evaluation feedback
    2. Corrects hallucinations when possible (evaluation tells us how)
    3. Validates ML-ready format
    4. Generates human review guide
    5. Applies all safety checks from original manager
    
    Parameters
    ----------
    state : KnowMatState
        Current workflow state containing:
        - aggregated_data: Merged data from Stage 1
        - aggregation_notes: Merge strategy notes
        - run_results: Evaluation feedback
        - paper_text: Original paper content
    
    Returns
    -------
    dict
        Updates containing:
        - final_data: Validated and corrected compositions
        - aggregation_rationale: Full explanation (merge + validation)
        - human_review_guide: Specific items to verify
    """
    aggregated_data = state.get("aggregated_data", {})
    aggregation_notes = state.get("aggregation_notes", "")
    run_results = state.get("run_results", [])
    paper_text = state.get("paper_text", "")
    
    if not aggregated_data or not aggregated_data.get("compositions"):
        # Empty aggregation - fallback
        print("Warning: Stage 1 aggregation returned empty data. Using fallback.")
        return _fallback_to_best_run(run_results)
    
    print(f"\nValidation Stage 2:")
    print(f"  Validating {len(aggregated_data.get('compositions', []))} aggregated compositions...")
    
    # Build validation prompt with ALL the hallucination correction logic
    # This is the COMPLETE prompt from the original manager
    validation_prompt = _build_validation_prompt(
        aggregated_data,
        aggregation_notes,
        run_results,
        paper_text
    )
    
    # Invoke the validation LLM (first attempt)
    result = manager_extractor.invoke(validation_prompt)
    response = _first_response(result)
    
    if response is None:
        print("Warning: Validation LLM returned no response. Using fallback.")
        return _fallback_to_best_run(run_results)
    
    # Convert response to dict
    if isinstance(response, ManagerFeedback):
        validation_dict = response.model_dump()
    else:
        validation_dict = dict(response)
    
    # Extract results
    final_extracted_data = validation_dict.get("final_extracted_data", {})
    aggregation_rationale = validation_dict.get("aggregation_rationale", "")
    human_review_guide = validation_dict.get("human_review_guide", "")
    
    # Ensure proper data structure
    final_data = _coerce_final_extracted_data(final_extracted_data)
    if final_data is None:
        print("Warning: Validation returned invalid data structure. Using fallback.")
        return _fallback_to_best_run(run_results)
    
    # ═══════════════════════════════════════════════════════════════════════
    # SAFETY CHECKS (ALL from original manager - PRESERVED)
    # ═══════════════════════════════════════════════════════════════════════
    compositions = final_data.get("compositions", [])
    avg_run_confidence = sum(r.get("confidence_score", 0.0) for r in run_results) / max(len(run_results), 1)
    
    # Detect failure patterns - RELAXED VERSION (original was too aggressive)
    # Only flag as placeholder if there's CLEAR evidence of failure
    has_no_compositions = not compositions or len(compositions) == 0
    has_trivial_rationale = len(aggregation_rationale.strip()) < 100  # Raised from 50
    has_todo_markers = any(marker in aggregation_rationale for marker in ["TODO", "[INSERT", "PLACEHOLDER_", "XXX"])
    has_trivial_review = human_review_guide.strip() in ["1) Verify.", "Verify.", ""]
    
    is_placeholder_response = (
        has_no_compositions or
        has_trivial_rationale or
        has_todo_markers or
        has_trivial_review
    )
    # NOTE: Removed "placeholder" keyword check - too many false positives
    #       (validator might mention "placeholder values" in rationale legitimately)
    
    # Detect lazy fallback (EXACT logic from original manager)
    is_lazy_fallback = (
        "Fallback: Selected run" in aggregation_rationale and
        len(compositions) > 0 and
        avg_run_confidence > 0.85 and
        len(run_results) > 1
    )
    
    if is_placeholder_response:
        print("  Warning: Validator returned empty/placeholder response.")
        print(f"    Compositions: {len(compositions)}, Rationale length: {len(aggregation_rationale)}")
        print(f"    Triggers: no_comps={has_no_compositions}, trivial_rationale={has_trivial_rationale}, ")
        print(f"              todo_markers={has_todo_markers}, trivial_review={has_trivial_review}")
        if has_todo_markers:
            print(f"    First 200 chars of rationale: {aggregation_rationale[:200]}")
        print("  Using fallback aggregation.")
        return _fallback_to_best_run(run_results)
    
    if is_lazy_fallback:
        print("  Warning: Validator chose lazy fallback despite good data.")
        print(f"    Avg run confidence: {avg_run_confidence:.2f}")
        print("  Retrying with stronger instructions...")
        
        # Retry with stronger prompt (EXACT logic from original manager)
        retry_result = _retry_validation_with_explicit_schema(
            aggregated_data,
            aggregation_notes,
            run_results,
            paper_text,
            validation_prompt
        )
        
        if retry_result:
            print("  Retry successful!")
            return retry_result
        else:
            print("  Retry failed. Using fallback.")
            return _fallback_to_best_run(run_results)

    if _validator_collapsed_item_boundaries(aggregated_data, final_data):
        print("  Warning: Validator collapsed distinct item boundaries. Preserving aggregated data.")
        return _preserve_aggregated_item_boundaries(
            aggregated_data,
            aggregation_notes,
            aggregation_rationale,
            human_review_guide,
        )
    
    # Success - validation completed
    print(f"  Validation complete: {len(compositions)} compositions validated")
    
    # Combine aggregation notes with validation rationale
    combined_rationale = (
        f"STAGE 1 - AGGREGATION:\n{aggregation_notes}\n\n"
        f"STAGE 2 - VALIDATION & CORRECTION:\n{aggregation_rationale}"
    )
    
    return {
        "final_data": final_data,
        "aggregation_rationale": combined_rationale,
        "human_review_guide": human_review_guide,
    }


def _build_validation_prompt(aggregated_data, aggregation_notes, run_results, paper_text) -> str:
    """Build the complete validation prompt with ALL hallucination correction logic.
    
    This contains the ENTIRE prompt from the original manager, focused on
    validation and correction rather than aggregation.
    """
    t = _VALIDATOR_TEMPLATES
    parts = []
    parts.append(t.get("system", "").strip() + "\n\n")
    parts.append(t.get("stage1_notes_prefix", "STAGE 1 AGGREGATION NOTES:\n"))
    parts.append(f"{aggregation_notes}\n\n")

    parts.append(t.get("aggregated_data_prefix", "AGGREGATED DATA TO VALIDATE:\n"))
    parts.append(f"{json.dumps(aggregated_data, ensure_ascii=False, indent=2)}\n")
    parts.append(t.get("aggregated_data_suffix", "") + "\n\n")

    parts.append(t.get("evaluation_feedback_header", "EVALUATION FEEDBACK (for hallucination correction):\n") + "\n")

    run_block_template = t.get("run_block_template", "")
    missing_prefix = t.get("missing_fields_prefix", "Missing Fields (<<MISSING_COUNT>>):\n")
    hallucinated_prefix = t.get(
        "hallucinated_fields_prefix",
        "HALLUCINATED FIELDS (<<HALLUCINATED_COUNT>>) - READ FOR CORRECTION CLUES:\n",
    )
    no_hallucinated = t.get("no_hallucinated_fields", "No hallucinated fields in this run\n")

    for i, run in enumerate(run_results, 1):
        run_text = (
            run_block_template.replace("<<RUN_ID>>", str(run.get("run_id", i)))
            .replace("<<CONFIDENCE>>", f"{run.get('confidence_score', 0.0):.2f}")
            .replace("<<RATIONALE>>", str(run.get("rationale", "No rationale")))
        )
        parts.append(run_text + "\n")

        missing = run.get("missing_fields", [])
        if missing:
            parts.append(missing_prefix.replace("<<MISSING_COUNT>>", str(len(missing))))
            for field in missing[:15]:
                parts.append(f"  - {field}\n")
            if len(missing) > 15:
                parts.append(f"  ... and {len(missing) - 15} more\n")
            parts.append("\n")

        hallucinated = run.get("hallucinated_fields", [])
        if hallucinated:
            parts.append(hallucinated_prefix.replace("<<HALLUCINATED_COUNT>>", str(len(hallucinated))))
            for j, field in enumerate(hallucinated[:15], 1):
                parts.append(f"  {j:2d}. {field}\n")
            if len(hallucinated) > 15:
                parts.append(f"       ... and {len(hallucinated) - 15} more\n")
            parts.append("\n")
        else:
            parts.append(no_hallucinated + "\n")

    if paper_text:
        parts.append("ORIGINAL PAPER TEXT:\n")
        parts.append(f"{paper_text}\n\n")

    parts.append(t.get("validation_tail", "BEGIN VALIDATION:\n"))
    return "".join(parts)


def _retry_validation_with_explicit_schema(aggregated_data, aggregation_notes, run_results, paper_text, original_prompt):
    """Retry validation with stronger instructions if lazy fallback detected.
    
    This is the EXACT retry logic from the original manager.
    """
    retry_prompt = original_prompt + (
        "\n\n"
        f"{'═' * 80}\n"
        "RETRY - VALIDATION REQUIRED\n"
        f"{'═' * 80}\n\n"
        "Your previous response was not satisfactory. You returned a lazy fallback\n"
        "instead of validating the aggregated data.\n\n"
        
        "WHAT YOU NEED TO DO:\n"
        "1. Read the AGGREGATED DATA structure shown above\n"
        "2. Review each property against the EVALUATION FEEDBACK\n"
        "3. Apply corrections for any hallucinations\n"
        "4. Return a COMPLETE validated dataset\n\n"
        
        "The data has already been merged. Your job is VALIDATION only.\n"
        "Provide a thorough validation with specific corrections and rationale.\n"
        f"{'═' * 80}\n"
    )
    
    # Invoke validation LLM with retry prompt
    result = manager_extractor.invoke(retry_prompt)
    response = _first_response(result)
    
    if response is None:
        return None
    
    # Convert response
    if isinstance(response, ManagerFeedback):
        validation_dict = response.model_dump()
    else:
        validation_dict = dict(response)
    
    # Extract results
    final_extracted_data = validation_dict.get("final_extracted_data", {})
    aggregation_rationale = validation_dict.get("aggregation_rationale", "")
    human_review_guide = validation_dict.get("human_review_guide", "")
    
    # Ensure proper structure
    final_data = _coerce_final_extracted_data(final_extracted_data)
    if final_data is None:
        return None
    
    compositions = final_data.get("compositions", [])
    
    # Check if retry still failed
    if "Fallback: Selected run" in aggregation_rationale:
        return None
    
    if not compositions or len(aggregation_rationale.strip()) < 50:
        return None

    if _validator_collapsed_item_boundaries(aggregated_data, final_data):
        return _preserve_aggregated_item_boundaries(
            aggregated_data,
            aggregation_notes,
            aggregation_rationale,
            human_review_guide,
        )
    
    # Success - combine with aggregation notes
    combined_rationale = (
        f"STAGE 1 - AGGREGATION:\n{aggregation_notes}\n\n"
        f"STAGE 2 - VALIDATION & CORRECTION (RETRY):\n{aggregation_rationale}"
    )
    
    return {
        "final_data": final_data,
        "aggregation_rationale": combined_rationale,
        "human_review_guide": human_review_guide,
    }


def _fallback_to_best_run(run_results):
    """Fallback aggregation if validation fails completely.
    
    This is the EXACT fallback logic from the original manager.
    """
    if not run_results:
        return {
            "final_data": {"compositions": []},
            "aggregation_rationale": "Validation failed with no run data available.",
            "human_review_guide": "Manual review required - validation pipeline failed.",
        }
    
    sorted_runs = sorted(run_results, key=lambda r: r.get("confidence_score", 0.0), reverse=True)
    best_run = sorted_runs[0]

    final_data = load_run_extraction(best_run)
    
    return {
        "final_data": final_data,
        "aggregation_rationale": f"Fallback: Validation failed. Selected run {best_run.get('run_id')} with highest confidence {best_run.get('confidence_score', 0.0):.2f}.",
        "human_review_guide": "Review extraction quality due to validation fallback.",
    }
