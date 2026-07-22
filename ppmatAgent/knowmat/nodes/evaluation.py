"""
Node for evaluating an extracted JSON result against the paper text.

This node calls the ``evaluation_extractor`` defined in
:mod:`knowmat2.extractors` to compare the most recent extraction with the
original document.  The extractor should output a confidence score,
a rationale, lists of missing and hallucinated fields, an optional
prompt update and a boolean indicating whether to perform another
extraction.  The evaluation result is appended to ``state['run_results']``
and ``state['run_count']`` is incremented.  ``state['needs_rerun']`` is
set based on the extractor's output.

If the extractor returns no response, the node assumes the extraction
should be accepted as‑is and sets ``needs_rerun`` to False.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, List

from ppmatAgent.knowmat.extractors import evaluation_extractor, EvaluationFeedback
from ppmatAgent.knowmat.prompt_loader import load_yaml_templates_required
from ppmatAgent.knowmat.states import KnowMatState, EvaluationRun

_EVAL_TEMPLATES = load_yaml_templates_required(
    "evaluation.yaml", ("system", "user_template")
)


def _build_evaluation_prompt(
    extracted_data: Dict[str, Any],
    paper_text: str,
    run_results: List[EvaluationRun],
) -> str:
    templates = _EVAL_TEMPLATES
    system_prompt = templates["system"].strip()
    user_prompt = templates["user_template"].format(
        extracted_data=json.dumps(extracted_data, ensure_ascii=False, indent=2)
    ).strip()
    parts = [system_prompt, "", user_prompt]
    if run_results:
        parts.extend(["", "PREVIOUS RUN HISTORY:", ""])
        for run in run_results:
            parts.append(f"Run {run.get('run_id', '?')} | Confidence: {run.get('confidence_score', 0.0):.2f}")
            parts.append(f"Rationale: {run.get('rationale', '')}")
            missing = run.get("missing_fields") or []
            hallucinated = run.get("hallucinated_fields") or []
            if missing:
                parts.append("Missing Fields:")
                parts.extend(f"- {field}" for field in missing)
            if hallucinated:
                parts.append("Hallucinated Fields:")
                parts.extend(f"- {field}" for field in hallucinated)
            suggested_prompt = run.get("suggested_prompt")
            if suggested_prompt:
                parts.append(f"Suggested Prompt Update: {suggested_prompt}")
            parts.append("")
    if paper_text:
        parts.extend(["ORIGINAL PAPER TEXT:", paper_text.strip()])
    return "\n".join(parts).strip()


def evaluate_data(state: KnowMatState) -> Dict[str, Any]:
    """Evaluate the extracted data against the source and decide on a rerun.

    Parameters
    ----------
    state: KnowMatState
        The current workflow state.  Must include ``paper_text`` and
        ``latest_extracted_data``.  ``updated_prompt`` may contain
        previous prompt updates.

    Returns
    -------
    dict
        Updates containing ``run_results``, ``run_count``, ``needs_rerun``
        and possibly an updated ``updated_prompt`` if the evaluation agent
        suggests additional refinements.
    """
    paper_text = state.get("paper_text", "")
    extracted_data = state.get("latest_extracted_data", {})
    run_count = state.get("run_count", 0)
    run_results: List[EvaluationRun] = list(state.get("run_results", []))

    evaluation_prompt = _build_evaluation_prompt(
        extracted_data=extracted_data,
        paper_text=paper_text,
        run_results=run_results,
    )

    result = evaluation_extractor.invoke(evaluation_prompt)
    response = result.get("responses", [None])[0]
    if response is None:
        # No evaluation returned; accept the extraction as is
        return {
            "run_results": run_results,
            "run_count": run_count + 1,
            "needs_rerun": False,
        }
    # Convert to a dictionary regardless of underlying type
    if isinstance(response, EvaluationFeedback):
        eval_dict = response.model_dump()
    else:
        eval_dict = dict(response)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SAFETY CHECK: Detect incomplete/invalid evaluation responses
    # ═══════════════════════════════════════════════════════════════════════════
    # If the LLM fails to properly invoke the tool or returns an incomplete 
    # response, we detect it here and force a rerun rather than accepting 
    # invalid evaluation data (confidence=0.0 with empty rationale).
    confidence = eval_dict.get("confidence_score", 0.0)
    rationale = eval_dict.get("rationale", "")
    needs_rerun_from_llm = eval_dict.get("needs_rerun", False)
    
    # Check for invalid evaluation: zero confidence AND empty rationale
    # This indicates the LLM failed to properly evaluate (technical error)
    if confidence == 0.0 and not rationale.strip():
        print("⚠️  WARNING: Evaluation agent returned incomplete response (confidence=0.0, empty rationale)")
        print(f"   This appears to be a technical error. Forcing rerun {run_count + 1}/{state.get('max_runs', 3)}")
        
        # Override the evaluation to force a rerun
        eval_dict["rationale"] = (
            "⚠️ WARNING: Evaluation agent returned incomplete response (possible LLM tool invocation failure). "
            "Forcing automatic rerun to obtain valid evaluation."
        )
        eval_dict["needs_rerun"] = True
        eval_dict["confidence_score"] = 0.0  # Keep 0.0 to indicate invalid evaluation
        
        # Set the corrected values for later use
        confidence = 0.0
        rationale = eval_dict["rationale"]
        needs_rerun_from_llm = True
    
    # Persist extraction data to disk to keep state lightweight
    run_id = run_count + 1
    output_dir = state.get("output_dir", ".")
    runs_dir = Path(output_dir) / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    extraction_path = runs_dir / f"run_{run_id}_extraction.json"
    with open(extraction_path, "w", encoding="utf-8") as fp:
        json.dump(extracted_data, fp, ensure_ascii=False, indent=2)

    record: EvaluationRun = {
        "run_id": run_id,
        "confidence_score": confidence,
        "rationale": rationale,
        "missing_fields": eval_dict.get("missing_fields"),
        "hallucinated_fields": eval_dict.get("hallucinated_fields"),
        "suggested_prompt": eval_dict.get("update_prompt"),
        "extracted_data_path": str(extraction_path),
    }
    run_results.append(record)
    # Update the extraction prompt if suggested
    updated_prompt = state.get("updated_prompt", "")
    update_text = eval_dict.get("update_prompt")
    if update_text:
        if updated_prompt:
            updated_prompt = updated_prompt.strip() + "\n\n" + update_text.strip()
        else:
            updated_prompt = update_text.strip()
    
    # Use the needs_rerun value (potentially corrected by safety check above)
    needs_rerun = bool(needs_rerun_from_llm)
    
    return {
        "run_results": run_results,
        "run_count": run_id,
        "needs_rerun": needs_rerun,
        "updated_prompt": updated_prompt,
    }
