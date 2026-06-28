"""
LLM-based node for assessing final extraction quality and determining review needs.

The flagging agent evaluates the manager's aggregation result, considers the
complexity of issues, quality of corrections made, and assigns a confidence 
score with review recommendations.
"""

from typing import Dict, Any

from ppmatAgent.knowmat.extractors import flagging_extractor, FlaggingFeedback
from ppmatAgent.knowmat.prompt_loader import load_yaml_templates_required
from ppmatAgent.knowmat.states import KnowMatState

_FLAGGING_TEMPLATES = load_yaml_templates_required(
    "flagging.yaml",
    (
        "intro",
        "run_stats_header",
        "manager_header",
        "review_header",
        "completeness_header",
        "task_header",
        "output_requirements",
    ),
)


def assess_final_quality(state: KnowMatState) -> Dict[str, Any]:
    """Use LLM to assess the final aggregated extraction quality.

    Parameters
    ----------
    state: KnowMatState
        The current workflow state containing aggregation results and run data.

    Returns
    -------
    dict
        Updates containing ``final_confidence_score``, ``confidence_rationale``, 
        ``needs_human_review``, and ``flag``.
    """
    run_results = state.get("run_results", [])
    aggregation_rationale = state.get("aggregation_rationale", "")
    human_review_guide = state.get("human_review_guide", "")
    final_data = state.get("final_data", {})
    
    if not run_results:
        # No runs to assess
        return {
            "final_confidence_score": 0.5,
            "confidence_rationale": "No evaluation runs available for assessment.",
            "needs_human_review": True,
            "flag": True,
        }
    
    t = _FLAGGING_TEMPLATES
    flagging_prompt = t.get("intro", "") + "\n\n"
    
    # Add run summary statistics WITH CORRECTION CONTEXT
    flagging_prompt += t.get("run_stats_header", "")
    
    if run_results:
        scores = [run.get('confidence_score', 0.0) for run in run_results]
        avg_confidence = sum(scores) / len(scores)
        min_confidence = min(scores)
        max_confidence = max(scores)
        
        flagging_prompt += f"Number of Runs: {len(run_results)}\n"
        flagging_prompt += f"Average Run Confidence: {avg_confidence:.2f}\n"
        flagging_prompt += f"Confidence Range: {min_confidence:.2f} - {max_confidence:.2f}\n"
        flagging_prompt += f"Confidence Spread: {max_confidence - min_confidence:.2f} (consistency indicator)\n\n"
        
        # Count issues across runs (handle None values)
        total_missing = sum(len(run.get('missing_fields') or []) for run in run_results)
        total_hallucinated = sum(len(run.get('hallucinated_fields') or []) for run in run_results)
        
        flagging_prompt += "ORIGINAL EXTRACTION ISSUES (Before Manager Corrections):\n"
        flagging_prompt += f"- Total missing fields across runs: {total_missing}\n"
        flagging_prompt += f"- Total hallucinated fields across runs: {total_hallucinated}\n"
        
        # Provide context about hallucination types
        flagging_prompt += "\nHallucination Breakdown by Run:\n"
        for i, run in enumerate(run_results, 1):
            h_fields = run.get('hallucinated_fields') or []  # Handle None
            if h_fields:
                flagging_prompt += f"  Run {i}: {len(h_fields)} hallucinations\n"
                # Show first few for context
                for j, h in enumerate(h_fields[:3], 1):
                    flagging_prompt += f"    {j}. {h[:100]}{'...' if len(h) > 100 else ''}\n"
                if len(h_fields) > 3:
                    flagging_prompt += f"    ... and {len(h_fields) - 3} more\n"
        flagging_prompt += "\n"
    
    # Add manager's assessment WITH CORRECTION ANALYSIS
    flagging_prompt += t.get("manager_header", "")
    
    flagging_prompt += f"Manager's Aggregation Rationale:\n"
    flagging_prompt += f"{'-'*40}\n{aggregation_rationale}\n\n"
    
    # Add human review guide ANALYSIS
    flagging_prompt += t.get("review_header", "")
    
    flagging_prompt += f"Human Review Guide:\n"
    flagging_prompt += f"{'-'*40}\n{human_review_guide}\n\n"
    
    # Add final data completeness METRICS
    flagging_prompt += t.get("completeness_header", "")
    
    compositions = final_data.get('compositions', [])
    flagging_prompt += f"Number of Compositions: {len(compositions)}\n"
    
    if compositions:
        total_properties = sum(len(comp.get('properties_of_composition', [])) for comp in compositions)
        total_ns_properties = sum(len(comp.get('non_standard_properties_of_composition', [])) for comp in compositions)
        avg_props = total_properties / len(compositions) if compositions else 0
        
        flagging_prompt += f"Total Standard Properties: {total_properties}\n"
        flagging_prompt += f"Total Non-Standard Properties: {total_ns_properties}\n"
        flagging_prompt += f"Average Properties per Composition: {avg_props:.1f}\n\n"
        
        flagging_prompt += "COMPLETENESS INTERPRETATION:\n"
        flagging_prompt += "- >10 props/composition: Excellent completeness\n"
        flagging_prompt += "- 5-10 props/composition: Good completeness\n"
        flagging_prompt += "- 2-5 props/composition: Fair completeness (may be conservative)\n"
        flagging_prompt += "- <2 props/composition: Poor completeness (likely too conservative or issues)\n\n"
        
        # Sample composition for richness check
        if compositions:
            sample = compositions[0]
            flagging_prompt += f"Sample Composition '{sample.get('composition', 'Unknown')}':\n"
            flagging_prompt += f"  Properties: {len(sample.get('properties_of_composition', []))}\n"
            flagging_prompt += f"  Processing: {'Present' if sample.get('processing_conditions') else 'Missing'}\n"
            flagging_prompt += f"  Characterisation: {len(sample.get('characterisation', {}))} techniques\n"
    else:
        flagging_prompt += "WARNING: No compositions extracted! This indicates major failure.\n"
    
    flagging_prompt += "\n"
    
    # Final instructions
    flagging_prompt += t.get("task_header", "")
    flagging_prompt += t.get("output_requirements", "")
    
    # Invoke the flagging extractor
    result = flagging_extractor.invoke(flagging_prompt)
    response = result.get("responses", [None])[0]
    
    if response is None:
        # Fallback assessment
        avg_confidence = sum(run.get('confidence_score', 0.0) for run in run_results) / len(run_results) if run_results else 0.0
        return {
            "final_confidence_score": avg_confidence,
            "confidence_rationale": "Fallback assessment: averaged run confidence scores.",
            "needs_human_review": avg_confidence < 0.8,
            "flag": avg_confidence < 0.8,
        }
    
    # Convert response to dict
    if isinstance(response, FlaggingFeedback):
        flagging_dict = response.model_dump()
    else:
        flagging_dict = dict(response)
    
    final_confidence = flagging_dict.get("final_confidence_score", 0.0)
    confidence_rationale = flagging_dict.get("confidence_rationale", "")
    needs_review = flagging_dict.get("needs_human_review", False)
    
    return {
        "final_confidence_score": final_confidence,
        "confidence_rationale": confidence_rationale,
        "needs_human_review": needs_review,
        "flag": needs_review,
    }