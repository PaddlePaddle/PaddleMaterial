"""
HEA Multi-Agent System based on CrewAI
面向高熵合金组分设计的主动学习型多智能体优化系统

架构来源: Ku 知识库文档 - 基于模型和效用的学习型智能体
六大智能体角色:
  1. Planner Agent        - 目标分析 & 搜索策略
  2. Knowledge Agent      - 文献检索 & 历史经验 RAG
  3. Thermodynamic Agent  - CALPHAD + 相稳定性分析
  4. Property Prediction Agent - 强度/延展性预测
  5. Optimization Agent   - 进化算法控制
  6. Memory Agent         - 历史轨迹 & 失败案例管理
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import threading
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# 把项目根目录加入 Python 路径
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from crewai import Agent, Crew, Task, Process
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from ppmatAgent.hea_crewai_agent.agents.llm_interface import get_crewai_llm
from ppmatAgent.hea_crewai_agent.core.evaluator import ThermodynamicEvaluator, MultiObjectiveEvaluator
from ppmatAgent.hea_crewai_agent.core.evolution_engine import AdaptiveEvolutionEngine
from ppmatAgent.hea_crewai_agent.core.material import Alloy
from ppmatAgent.hea_crewai_agent.core.materials_graphrag import query_materials_graph
from ppmatAgent.hea_crewai_agent.core.state import BeliefState
from ppmatAgent.hea_crewai_agent.agents.strategy_agent import StrategyAgent
from ppmatAgent.hea_crewai_agent.runtime_config import (
    get_active_learning_settings,
    get_calphad_settings,
    get_constraint_settings,
    get_evaluation_objectives,
    get_evaluation_search_mode,
    get_evolution_settings,
    load_runtime_config,
    resolve_calphad_database,
)


# ─────────────────────────── 工具定义 ───────────────────────────


_OBJECTIVE_ALIASES = {
    "stability": "thermodynamic",
    "phase_stability": "thermodynamic",
    "thermodynamic": "thermodynamic",
    "strength": "mechanical",
    "ductility": "mechanical",
    "hardness": "mechanical",
    "mechanical": "mechanical",
    "corrosion": "corrosion",
    "corrosion_resistance": "corrosion",
    "cost": "cost",
}

_MEMORY_FILE_LOCK = threading.RLock()


def _json_default(value: Any):
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _round_nested(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _round_nested(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_round_nested(item) for item in value]
    if isinstance(value, float):
        return round(value, 6)
    return value


def _to_plain_data(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _to_plain_data(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_plain_data(item) for item in value]

    model_dump = getattr(value, 'model_dump', None)
    if callable(model_dump):
        try:
            return _to_plain_data(model_dump())
        except Exception:
            pass

    dict_method = getattr(value, 'dict', None)
    if callable(dict_method):
        try:
            return _to_plain_data(dict_method())
        except Exception:
            pass

    tolist = getattr(value, 'tolist', None)
    if callable(tolist):
        try:
            return _to_plain_data(tolist())
        except Exception:
            pass

    item = getattr(value, 'item', None)
    if callable(item):
        try:
            return item()
        except Exception:
            pass

    return str(value)


def _normalize_composition_dict(composition: Dict[str, Any]) -> Dict[str, float]:
    if not isinstance(composition, dict):
        return {}

    for nested_key in (
        "composition",
        "elements",
        "composition_atomic_fraction",
        "composition_at_percent",
        "as_at_percent",
    ):
        nested = composition.get(nested_key)
        if isinstance(nested, dict):
            composition = nested
            break

    cleaned: Dict[str, float] = {}
    total = 0.0
    for element, fraction in composition.items():
        try:
            value = float(fraction)
        except Exception:
            continue
        if value <= 0:
            continue
        cleaned[str(element)] = value
        total += value

    if total <= 0:
        return {}

    return {
        element: round(value / total, 6)
        for element, value in sorted(cleaned.items())
    }


def _composition_signature(composition: Dict[str, Any], decimals: int = 6):
    normalized = _normalize_composition_dict(composition)
    return tuple(
        sorted(
            (element, round(float(fraction), decimals))
            for element, fraction in normalized.items()
        )
    )


def _normalize_memory_record(record: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(record)

    if "best_composition" in normalized and "composition" not in normalized:
        normalized = {
            "composition": normalized.get("best_composition", {}),
            "fitness": normalized.get(
                "best_fitness",
                normalized.get("weighted_fitness", 0.0),
            ),
            "properties": {
                "objective_scores": normalized.get("objective_scores", {}),
                "weighted_fitness": normalized.get(
                    "weighted_fitness",
                    normalized.get("best_fitness", 0.0),
                ),
                "acquisition_score": normalized.get("acquisition_score"),
                "novelty_score": normalized.get("novelty_score"),
                "model_uncertainty": normalized.get("model_uncertainty"),
                "data_density": normalized.get("data_density"),
                "search_mode": normalized.get("search_mode"),
                "strategy_source": normalized.get("strategy_source"),
                "evaluation_backend": normalized.get("evaluation_backend"),
                "calphad_database": normalized.get("calphad_database"),
                "evaluation_method": normalized.get("evaluation_method"),
                "evolution_summary": normalized.get("evolution_summary", {}),
            },
            "belief_state": normalized.get("belief_state", {}),
            "notes": normalized.get("notes", "Saved from evolution_search_runner"),
        }

    composition = normalized.get("composition", {})
    if isinstance(composition, dict):
        normalized["composition"] = _normalize_composition_dict(composition)

    fitness = normalized.get("fitness")
    if fitness is not None:
        normalized["fitness"] = round(float(fitness), 6)

    properties = normalized.get("properties", {})
    if isinstance(properties, dict):
        normalized["properties"] = _round_nested(properties)

    belief_state = normalized.get("belief_state")
    if isinstance(belief_state, dict) and "composition" in belief_state:
        belief_state = dict(belief_state)
        belief_state["composition"] = _normalize_composition_dict(
            belief_state.get("composition", {})
        )
        normalized["belief_state"] = _round_nested(belief_state)

    normalized["updated_at"] = normalized.get(
        "updated_at",
        datetime.now(timezone.utc).isoformat(),
    )
    return normalized


def _merge_memory_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged_records: Dict[Any, Dict[str, Any]] = {}
    for item in records:
        if not isinstance(item, dict):
            continue
        normalized = _normalize_memory_record(item)
        signature = _composition_signature(normalized.get("composition", {}))
        if not signature:
            continue
        current = merged_records.get(signature)
        if current is None or float(normalized.get("fitness", 0.0)) >= float(current.get("fitness", 0.0)):
            merged_records[signature] = normalized

    return sorted(
        merged_records.values(),
        key=lambda r: float(r.get("fitness", 0.0)),
        reverse=True,
    )[:100]


def _recover_memory_payload(raw_text: str) -> tuple[Dict[str, Any], bool]:
    recovered = False
    payload_candidates: List[Dict[str, Any]] = []
    try:
        payload = json.loads(raw_text)
        if isinstance(payload, dict):
            payload_candidates.append(payload)
    except Exception:
        recovered = True
        payload_candidates.extend(_extract_json_objects_from_text(raw_text, limit=64))

    records: List[Dict[str, Any]] = []
    for payload in payload_candidates:
        if not isinstance(payload, dict):
            continue
        candidate_records = payload.get("records", [])
        if isinstance(candidate_records, list):
            records.extend(candidate_records)

    return {"records": _merge_memory_records(records)}, recovered


def _iter_memory_backup_files(memory_file: Path) -> List[Path]:
    candidates = [
        candidate
        for candidate in memory_file.parent.glob(f"{memory_file.stem}*.json")
        if candidate != memory_file and candidate.is_file()
    ]
    candidates.sort(key=lambda candidate: candidate.stat().st_mtime, reverse=True)
    return candidates


def _write_memory_payload_atomic(memory_file: Path, payload: Dict[str, Any]) -> None:
    memory_file.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=memory_file.parent,
        prefix=f"{memory_file.stem}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        temp_path = Path(handle.name)

    os.replace(temp_path, memory_file)


def _backup_corrupted_memory_file(memory_file: Path) -> Optional[Path]:
    if not memory_file.exists():
        return None

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = memory_file.with_name(f"{memory_file.stem}.corrupt-{timestamp}{memory_file.suffix}")
    shutil.copy2(memory_file, backup_path)
    return backup_path


def _repair_memory_file(memory_file: Optional[Path] = None) -> Dict[str, Any]:
    path = memory_file or (_ROOT / ".ark" / "output" / "hea_memory.json")
    with _MEMORY_FILE_LOCK:
        if not path.exists():
            payload = {"records": []}
            _write_memory_payload_atomic(path, payload)
            return {
                "repaired": False,
                "created": True,
                "backup_path": None,
                "record_count": 0,
                "recovered_from_backup_paths": [],
            }

        raw_text = path.read_text(encoding="utf-8")
        payload, recovered = _recover_memory_payload(raw_text)
        merged_records = list(payload.get("records", []))
        primary_count = len(merged_records)
        recovered_from_backup_paths: List[str] = []

        if recovered or primary_count <= 1:
            for backup_file in _iter_memory_backup_files(path):
                backup_payload, backup_recovered = _recover_memory_payload(
                    backup_file.read_text(encoding="utf-8")
                )
                backup_records = backup_payload.get("records", [])
                if not backup_records:
                    continue
                if len(backup_records) > primary_count or primary_count <= 1:
                    merged_records = _merge_memory_records(merged_records + backup_records)
                    recovered_from_backup_paths.append(str(backup_file))
                    recovered = True or backup_recovered

        final_payload = {"records": _merge_memory_records(merged_records)}
        backup_path = None
        if recovered or len(final_payload["records"]) != primary_count:
            backup_path = _backup_corrupted_memory_file(path)
            _write_memory_payload_atomic(path, final_payload)

        return {
            "repaired": recovered,
            "created": False,
            "backup_path": str(backup_path) if backup_path else None,
            "record_count": len(final_payload.get("records", [])),
            "recovered_from_backup_paths": recovered_from_backup_paths,
        }


def _load_memory_records(memory_file: Optional[Path] = None) -> List[Dict[str, Any]]:
    path = memory_file or (_ROOT / ".ark" / "output" / "hea_memory.json")
    if not path.exists():
        return []
    with _MEMORY_FILE_LOCK:
        try:
            payload, _ = _recover_memory_payload(path.read_text(encoding="utf-8"))
        except Exception:
            return []

        return payload.get("records", [])


def _extract_json_object_from_text(text: str) -> Optional[Dict[str, Any]]:
    cleaned = text.strip()
    if not cleaned:
        return None

    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    decoder = json.JSONDecoder()
    for index, char in enumerate(cleaned):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(cleaned[index:])
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _extract_json_objects_from_text(text: str, limit: int = 24) -> List[Dict[str, Any]]:
    cleaned = text.strip()
    if not cleaned:
        return []

    decoder = json.JSONDecoder()
    results: List[Dict[str, Any]] = []
    seen: set[str] = set()

    for index, char in enumerate(cleaned):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(cleaned[index:])
        except Exception:
            continue
        if not isinstance(parsed, dict):
            continue

        try:
            signature = json.dumps(_to_plain_data(parsed), ensure_ascii=False, sort_keys=True)
        except Exception:
            signature = str(parsed)

        if signature in seen:
            continue
        seen.add(signature)
        results.append(parsed)
        if len(results) >= limit:
            break

    return results


def _looks_like_memory_record(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False

    if isinstance(payload.get("composition"), dict):
        return any(key in payload for key in ("fitness", "best_fitness", "weighted_fitness"))

    if isinstance(payload.get("best_composition"), dict):
        return any(key in payload for key in ("fitness", "best_fitness", "weighted_fitness"))

    return False


def _find_nested_record(payload: Any, target_key: str) -> Optional[Dict[str, Any]]:
    if isinstance(payload, dict):
        value = payload.get(target_key)
        if isinstance(value, dict):
            return value
        for nested in payload.values():
            found = _find_nested_record(nested, target_key)
            if found is not None:
                return found
    elif isinstance(payload, list):
        for item in payload:
            found = _find_nested_record(item, target_key)
            if found is not None:
                return found
    return None


def _extract_task_payload(task_output: Any) -> Optional[Dict[str, Any]]:
    json_dict = getattr(task_output, "json_dict", None)
    if isinstance(json_dict, dict):
        return json_dict

    raw = getattr(task_output, "raw", "")
    if isinstance(raw, str):
        return _extract_json_object_from_text(raw)
    return None


def _composition_distance(
    composition_a: Dict[str, Any],
    composition_b: Dict[str, Any],
    basis_elements: Optional[List[str]] = None,
) -> float:
    normalized_a = _normalize_composition_dict(composition_a)
    normalized_b = _normalize_composition_dict(composition_b)
    elements = basis_elements or sorted(
        set(normalized_a.keys()) | set(normalized_b.keys())
    )
    distance_sq = 0.0
    for element in elements:
        distance_sq += (
            normalized_a.get(element, 0.0) - normalized_b.get(element, 0.0)
        ) ** 2
    return distance_sq ** 0.5


def _estimate_novelty_and_density(
    composition: Dict[str, Any],
    reference_compositions: List[Dict[str, Any]],
    basis_elements: List[str],
) -> tuple[float, float]:
    if not reference_compositions:
        return 1.0, 0.0

    distances = sorted(
        _composition_distance(composition, reference, basis_elements)
        for reference in reference_compositions
    )
    if not distances:
        return 1.0, 0.0

    max_distance = 2 ** 0.5
    nearest = distances[0]
    neighborhood = distances[: min(3, len(distances))]
    average_neighborhood = sum(neighborhood) / len(neighborhood)

    novelty = min(max(nearest / max_distance, 0.0), 1.0)
    density = 1.0 - min(max(average_neighborhood / max_distance, 0.0), 1.0)
    return round(novelty, 6), round(density, 6)


def _estimate_uncertainty_scores(
    alloy: Alloy,
    novelty_score: float,
    data_density: float,
) -> tuple[float, float]:
    evaluation_method = str(alloy.properties.get("evaluation_method", "simple_only"))
    calphad_status = str(alloy.properties.get("calphad_status", ""))
    objective_scores = alloy.properties.get("objective_scores", {})

    model_uncertainty = 0.7
    if evaluation_method == "hybrid_calphad":
        model_uncertainty -= 0.25
    elif calphad_status == "success":
        model_uncertainty -= 0.2

    if isinstance(objective_scores, dict) and objective_scores:
        model_uncertainty -= 0.05

    model_uncertainty += 0.20 * novelty_score
    model_uncertainty -= 0.15 * data_density

    if calphad_status.startswith("error"):
        model_uncertainty += 0.10

    model_uncertainty = min(max(model_uncertainty, 0.05), 0.95)

    calphad_confidence = 0.2
    if evaluation_method == "hybrid_calphad":
        calphad_confidence = 0.9
    elif calphad_status == "success":
        calphad_confidence = 0.8
    elif calphad_status and "error" not in calphad_status:
        calphad_confidence = 0.35

    return round(model_uncertainty, 6), round(calphad_confidence, 6)


def _score_active_learning_candidates(
    candidates: List[Alloy],
    reference_compositions: List[Dict[str, Any]],
    basis_elements: List[str],
    active_learning_settings: Dict[str, Any],
) -> List[Dict[str, Any]]:
    fitness_weight = float(active_learning_settings.get("fitness_weight", 0.6))
    uncertainty_weight = float(active_learning_settings.get("uncertainty_weight", 0.2))
    novelty_weight = float(active_learning_settings.get("novelty_weight", 0.2))
    total_weight = fitness_weight + uncertainty_weight + novelty_weight or 1.0

    scored_candidates: List[Dict[str, Any]] = []
    for alloy in candidates:
        novelty_score, data_density = _estimate_novelty_and_density(
            alloy.composition,
            reference_compositions,
            basis_elements,
        )
        model_uncertainty, calphad_confidence = _estimate_uncertainty_scores(
            alloy,
            novelty_score,
            data_density,
        )
        exploit_score = float(
            alloy.properties.get(
                "weighted_fitness",
                alloy.properties.get("fitness", 0.0),
            )
        )
        acquisition_score = (
            fitness_weight * exploit_score
            + uncertainty_weight * model_uncertainty
            + novelty_weight * novelty_score
        ) / total_weight

        alloy.properties["novelty_score"] = novelty_score
        alloy.properties["data_density"] = data_density
        alloy.properties["model_uncertainty"] = model_uncertainty
        alloy.properties["calphad_confidence"] = calphad_confidence
        alloy.properties["acquisition_score"] = round(acquisition_score, 6)

        scored_candidates.append(
            {
                "alloy": alloy,
                "composition": {
                    element: round(fraction, 6)
                    for element, fraction in sorted(alloy.composition.items())
                },
                "fitness": round(float(alloy.properties.get("fitness", 0.0)), 6),
                "weighted_fitness": round(exploit_score, 6),
                "objective_scores": _round_nested(
                    alloy.properties.get("objective_scores", {})
                ),
                "evaluation_method": alloy.properties.get("evaluation_method", "simple_only"),
                "novelty_score": novelty_score,
                "data_density": data_density,
                "model_uncertainty": model_uncertainty,
                "calphad_confidence": calphad_confidence,
                "acquisition_score": round(acquisition_score, 6),
                "mutation": alloy.metadata.get("mutation", {}),
            }
        )

    scored_candidates.sort(
        key=lambda item: item["acquisition_score"],
        reverse=True,
    )
    return scored_candidates


def _normalize_objective_name(name: str, search_mode: str) -> Optional[str]:
    normalized = name.strip().lower().replace("-", "_").replace(" ", "_")
    normalized = _OBJECTIVE_ALIASES.get(normalized, normalized)
    if not normalized:
        return None
    if normalized == "thermodynamic" and search_mode == "eutectic":
        return "thermodynamic_eutectic"
    return normalized


def _merge_objectives(
    objectives: List[str],
    weights: List[float],
) -> tuple[List[str], List[float]]:
    merged_weights: Dict[str, float] = {}
    ordered_names: List[str] = []

    for objective, weight in zip(objectives, weights):
        if objective not in merged_weights:
            ordered_names.append(objective)
            merged_weights[objective] = 0.0
        merged_weights[objective] += float(weight)

    total = sum(merged_weights.values()) or 1.0
    normalized_weights = [merged_weights[name] / total for name in ordered_names]
    return ordered_names, normalized_weights


def _resolve_objective_settings(
    requested_objectives: str,
    runtime_config: Dict[str, Any],
    search_mode: str,
) -> tuple[List[str], List[float]]:
    if requested_objectives.strip():
        requested = [
            item.strip()
            for item in requested_objectives.split(",")
            if item.strip()
        ]
        resolved = [
            _normalize_objective_name(item, search_mode)
            for item in requested
        ]
        resolved = [item for item in resolved if item]
        if resolved:
            return _merge_objectives(
                resolved,
                [1.0 / len(resolved)] * len(resolved),
            )

    defaults = get_evaluation_objectives(runtime_config)
    resolved_defaults = [
        _normalize_objective_name(item, search_mode)
        for item in defaults["objectives"]
    ]
    filtered_pairs = [
        (objective, weight)
        for objective, weight in zip(resolved_defaults, defaults["weights"])
        if objective
    ]
    return _merge_objectives(
        [objective for objective, _ in filtered_pairs],
        [weight for _, weight in filtered_pairs],
    )


class ThermodynamicEvalInput(BaseModel):
    composition_json: str = Field(description="JSON string of composition dict, e.g. '{\"Ni\":0.3,\"Co\":0.25}'")


class ThermodynamicEvalTool(BaseTool):
    """用热力学模型（ΔS_mix, ΔH_mix, δ 参数）评估给定合金成分。"""

    name: str = "thermodynamic_evaluator"
    description: str = (
        "Evaluate a high-entropy alloy composition using thermodynamic criteria "
        "(mixing entropy, mixing enthalpy, delta parameter). "
        "Input: JSON string of {element: fraction} composition."
    )
    args_schema: type[BaseModel] = ThermodynamicEvalInput

    def _run(self, composition_json: str) -> str:
        try:
            composition = json.loads(composition_json)
            alloy = Alloy(composition=composition)
            runtime_config = load_runtime_config()
            evaluator = ThermodynamicEvaluator(
                search_mode=get_evaluation_search_mode(runtime_config)
            )
            fitness = evaluator.evaluate(alloy)
            result = {
                "fitness": round(fitness, 4),
                "mixing_entropy": round(alloy.properties.get("mixing_entropy", 0), 4),
                "mixing_enthalpy": round(alloy.properties.get("mixing_enthalpy", 0), 4),
                "delta": round(alloy.properties.get("delta", 0), 4),
                "search_mode": alloy.properties.get("thermo_search_mode", "single_phase"),
            }
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            return f"Error: {e}"


class CALPHADEvalInput(BaseModel):
    composition_json: str = Field(description="JSON string of composition dict")
    temperature_k: Optional[float] = Field(default=None, description="Temperature in Kelvin")


class CALPHADEvalTool(BaseTool):
    """调用 CALPHAD (pycalphad) 计算相平衡，评估相稳定性。"""

    name: str = "calphad_evaluator"
    description: str = (
        "Calculate phase equilibrium for a high-entropy alloy using CALPHAD (pycalphad). "
        "Returns phase stability, Gibbs free energy analysis. "
        "Input: JSON composition string and optional temperature in Kelvin."
    )
    args_schema: type[BaseModel] = CALPHADEvalInput

    def _run(self, composition_json: str, temperature_k: Optional[float] = None) -> str:
        try:
            from ppmatAgent.hea_crewai_agent.core.calphad_evaluator import CALPHADEvaluator

            runtime_config = load_runtime_config()
            calphad_settings = get_calphad_settings(runtime_config)
            search_mode = get_evaluation_search_mode(runtime_config)
            composition = json.loads(composition_json)
            alloy = Alloy(composition=composition)
            database_entry = resolve_calphad_database(composition.keys(), runtime_config)
            if database_entry is None:
                return json.dumps({
                    "calphad_available": False,
                    "error": (
                        "No compatible CALPHAD database configured or found for elements: "
                        + ",".join(sorted(composition.keys()))
                    ),
                })

            effective_temperature = (
                temperature_k
                if temperature_k is not None
                else calphad_settings["default_temperature_k"]
            )
            evaluator = CALPHADEvaluator(
                database_path=database_entry["path"],
                temperature=effective_temperature,
                target_phases=(
                    None
                    if search_mode == "eutectic"
                    else database_entry.get("recommended_phases")
                ),
                prefer_single_phase=(search_mode != "eutectic"),
                search_mode=search_mode,
            )
            fitness = evaluator.evaluate(alloy)
            result = {
                "calphad_fitness": round(fitness, 4),
                "database": database_entry["filename"],
                "search_mode": search_mode,
                "status": alloy.properties.get("calphad_status", "unknown"),
                "phases": alloy.properties.get("calphad_phases", {}),
                "calphad_available": True,
            }
            return json.dumps(result, ensure_ascii=False)
        except Exception as e:
            return json.dumps({"calphad_available": False, "error": str(e)})


class MELMutationInput(BaseModel):
    composition_json: str = Field(description="Current alloy composition as JSON")
    objectives: str = Field(description="Optimization objective keywords, comma-separated")
    allowed_elements: str = Field(default="", description="Allowed elements, comma-separated")


class MELMutationTool(BaseTool):
    """用 LLM 驱动的 StrategyAgent 生成 MEL 变异操作字符串。"""

    name: str = "mel_mutation_generator"
    description: str = (
        "Generate a MEL (Material Edit Language) mutation operation for alloy composition evolution. "
        "Uses LLM to propose chemically-informed mutations. "
        "Input: JSON composition, optimization objectives, and allowed element pool."
    )
    args_schema: type[BaseModel] = MELMutationInput
    llm_model: str = Field(default="gpt-5.2", exclude=True)
    llm_api: Optional[str] = Field(default=None, exclude=True)
    llm_base_url: Optional[str] = Field(default=None, exclude=True)
    runtime_config: Dict[str, Any] = Field(default_factory=dict, exclude=True)

    def _run(self, composition_json: str, objectives: str, allowed_elements: str = "") -> str:
        try:
            runtime_config = self.runtime_config or load_runtime_config()
            strategy_config = runtime_config.get("strategy_agent", {})
            graphrag_config = (
                strategy_config.get("graphrag", {})
                if isinstance(strategy_config.get("graphrag", {}), dict)
                else {}
            )
            use_graphrag = bool(graphrag_config.get("enabled", True))
            graphrag_max_results = int(graphrag_config.get("max_results", 5))
            memory_records = _load_memory_records() if use_graphrag else []

            composition = json.loads(composition_json)
            alloy = Alloy(composition=composition)
            allowed = [e.strip() for e in allowed_elements.split(",") if e.strip()] or sorted(composition.keys())

            # 默认使用 CrewAI 自带 LLM，GraphRAG 证据作为额外上下文注入。
            agent = StrategyAgent(
                llm_model=self.llm_model,
                crewai_llm=get_crewai_llm(
                    model=self.llm_model,
                    temperature=0.2,
                    llm_api=self.llm_api,
                    base_url=self.llm_base_url,
                ),
                fallback_to_rules=True,
                use_graphrag=use_graphrag,
                memory_records=memory_records,
                graphrag_max_results=graphrag_max_results,
            )
            mel_str = agent.generate_mutation(
                alloy=alloy,
                allowed_elements=allowed,
                evolution_context={"objectives": objectives.split(",")},
            )
            return json.dumps({
                "mel_operation": mel_str,
                "source": agent.last_source,
                "rationale": agent.last_rationale or "N/A",
                "graphrag": agent.last_graphrag_summary,
            })
        except Exception as e:
            return f"Error: {e}"


class EvolutionSearchInput(BaseModel):
    seed_composition_json: str = Field(
        default="",
        description="Optional JSON string of a seed composition dict."
    )
    objectives: str = Field(
        default="",
        description="Optional objective keywords, comma-separated. Example: 'stability,strength,corrosion'."
    )
    allowed_elements: str = Field(
        default="",
        description="Optional allowed elements, comma-separated."
    )
    population_size: Optional[int] = Field(
        default=None,
        description="Optional override for genetic population size."
    )
    max_generations: Optional[int] = Field(
        default=None,
        description="Optional override for evolution generations."
    )


class EvolutionSearchTool(BaseTool):
    """运行真实遗传算法搜索，并返回结构化候选状态。"""

    name: str = "evolution_search_runner"
    description: str = (
        "Run the real adaptive genetic search engine for HEA optimization. "
        "Use this as the primary optimization tool when you already have a promising seed composition "
        "or an allowed element pool. Returns best composition, objective scores, evolution summary, "
        "and a structured belief_state JSON object."
    )
    args_schema: type[BaseModel] = EvolutionSearchInput

    element_pool: List[str] = Field(default_factory=list, exclude=True)
    runtime_config: Dict[str, Any] = Field(default_factory=dict, exclude=True)
    llm_model: str = Field(default="gpt-5.2", exclude=True)
    llm_api: Optional[str] = Field(default=None, exclude=True)
    llm_base_url: Optional[str] = Field(default=None, exclude=True)

    def _run(
        self,
        seed_composition_json: str = "",
        objectives: str = "",
        allowed_elements: str = "",
        population_size: Optional[int] = None,
        max_generations: Optional[int] = None,
    ) -> str:
        try:
            runtime_config = self.runtime_config or load_runtime_config()
            search_mode = get_evaluation_search_mode(runtime_config)
            evolution_settings = get_evolution_settings(runtime_config)
            calphad_settings = get_calphad_settings(runtime_config)
            active_learning_settings = get_active_learning_settings(runtime_config)

            resolved_allowed = [
                item.strip()
                for item in allowed_elements.split(",")
                if item.strip()
            ] or list(self.element_pool)
            constraints = get_constraint_settings(runtime_config, resolved_allowed or None)
            resolved_allowed = constraints["allowed_elements"]
            if not resolved_allowed:
                return json.dumps({
                    "error": "No allowed elements available for evolution search.",
                }, ensure_ascii=False)
            constraints["max_elements"] = min(
                constraints["max_elements"],
                max(1, len(resolved_allowed)),
            )

            objective_names, objective_weights = _resolve_objective_settings(
                objectives,
                runtime_config,
                search_mode,
            )
            simple_evaluator = MultiObjectiveEvaluator(
                objectives=objective_names,
                weights=objective_weights,
            )
            evaluator_fn = simple_evaluator.evaluate
            hybrid_evaluator = None
            evaluation_backend = "simple_multi_objective"
            evaluation_method = "simple_only"
            evaluation_stats: Dict[str, Any] = {}
            calphad_database = None
            calphad_warning = None

            if calphad_settings.get("use_hybrid_evaluator", False):
                try:
                    from ppmatAgent.hea_crewai_agent.core.calphad_evaluator import CALPHADEvaluator, HybridEvaluator

                    database_entry = resolve_calphad_database(resolved_allowed, runtime_config)
                    if database_entry is None:
                        calphad_warning = (
                            "Hybrid evaluator requested but no compatible CALPHAD database was found."
                        )
                    else:
                        calphad_database = database_entry["filename"]
                        calphad_evaluator = CALPHADEvaluator(
                            database_path=database_entry["path"],
                            temperature=calphad_settings["default_temperature_k"],
                            target_phases=(
                                None
                                if search_mode == "eutectic"
                                else database_entry.get("recommended_phases")
                            ),
                            prefer_single_phase=(search_mode != "eutectic"),
                            search_mode=search_mode,
                        )
                        hybrid_evaluator = HybridEvaluator(
                            calphad_evaluator=calphad_evaluator,
                            simple_evaluator=simple_evaluator,
                            calphad_threshold=calphad_settings["calphad_threshold"],
                            use_calphad_probability=calphad_settings["use_calphad_probability"],
                        )
                        evaluator_fn = hybrid_evaluator.evaluate
                        evaluation_backend = "hybrid_calphad"
                        evaluation_stats = hybrid_evaluator.get_statistics()
                except Exception as exc:
                    calphad_warning = str(exc)

            seed_alloys: List[Alloy] = []
            if seed_composition_json.strip():
                seed_composition = json.loads(seed_composition_json)
                seed_alloy = Alloy(composition=seed_composition)
                if not seed_alloy.is_valid(constraints):
                    return json.dumps({
                        "error": "Seed composition violates current constraints.",
                        "constraints": constraints,
                        "seed_composition": seed_alloy.composition,
                    }, ensure_ascii=False)
                seed_alloys.append(seed_alloy)
            elif resolved_allowed:
                initial_elements = resolved_allowed[:constraints["max_elements"]]
                equiatomic = {
                    element: round(1.0 / len(initial_elements), 6)
                    for element in initial_elements
                }
                seed_alloys.append(Alloy(composition=equiatomic))

            strategy_agent = None
            strategy_warning = None
            strategy_config = runtime_config.get("strategy_agent", {})
            graphrag_config = (
                strategy_config.get("graphrag", {})
                if isinstance(strategy_config.get("graphrag", {}), dict)
                else {}
            )
            strategy_use_graphrag = bool(graphrag_config.get("enabled", True))
            strategy_graphrag_max_results = int(graphrag_config.get("max_results", 5))
            memory_records = _load_memory_records()
            reference_compositions = [
                record.get("composition", {})
                for record in memory_records
                if isinstance(record.get("composition"), dict)
            ]
            if bool(strategy_config.get("enabled", False)):
                try:
                    strategy_agent = StrategyAgent(
                        llm_model=str(strategy_config.get("model", self.llm_model)),
                        temperature=float(strategy_config.get("temperature", 0.2)),
                        max_tokens=int(strategy_config.get("max_tokens", 160)),
                        crewai_llm=get_crewai_llm(
                            model=str(strategy_config.get("model", self.llm_model)),
                            temperature=float(strategy_config.get("temperature", 0.2)),
                            llm_api=self.llm_api,
                            base_url=self.llm_base_url,
                        ),
                        fallback_to_rules=True,
                        use_graphrag=strategy_use_graphrag,
                        memory_records=memory_records if strategy_use_graphrag else [],
                        graphrag_max_results=strategy_graphrag_max_results,
                    )
                except Exception as exc:
                    strategy_warning = str(exc)
            elif strategy_use_graphrag:
                strategy_agent = StrategyAgent(
                    llm_model=str(strategy_config.get("model", self.llm_model)),
                    temperature=float(strategy_config.get("temperature", 0.2)),
                    max_tokens=int(strategy_config.get("max_tokens", 160)),
                    max_retries=0,
                    fallback_to_rules=True,
                    use_graphrag=True,
                    memory_records=memory_records,
                    graphrag_max_results=strategy_graphrag_max_results,
                )
                strategy_warning = (
                    "strategy_agent LLM disabled in config; using GraphRAG/rule-guided mutations"
                )
            else:
                strategy_warning = "strategy_agent disabled in config; using random mutations"

            engine = AdaptiveEvolutionEngine(
                population_size=population_size or evolution_settings["population_size"],
                max_generations=max_generations or evolution_settings["max_generations"],
                target_elements=resolved_allowed,
                mutation_rate=evolution_settings["mutation_rate"],
                crossover_rate=evolution_settings["crossover_rate"],
                elite_ratio=evolution_settings["elite_ratio"],
                constraints=constraints,
                seed_alloys=seed_alloys,
            )

            best_alloy = engine.evolve(
                evaluator=evaluator_fn,
                strategy_agent=strategy_agent,
                verbose=False,
            )

            final_population = getattr(engine, "final_population", None)
            candidate_alloys = (
                list(final_population.alloys)
                if final_population is not None and final_population.alloys
                else [best_alloy]
            )
            scored_candidates = _score_active_learning_candidates(
                candidates=candidate_alloys,
                reference_compositions=reference_compositions,
                basis_elements=resolved_allowed,
                active_learning_settings=active_learning_settings,
            )
            candidate_pool_size = max(
                1,
                min(
                    len(scored_candidates),
                    int(active_learning_settings.get("candidate_pool_size", 5)),
                ),
            )
            top_acquisition_candidates = [
                {key: value for key, value in candidate.items() if key != "alloy"}
                for candidate in scored_candidates[:candidate_pool_size]
            ]
            recommended_candidate_raw = scored_candidates[0] if scored_candidates else None
            recommended_candidate = (
                {key: value for key, value in recommended_candidate_raw.items() if key != "alloy"}
                if recommended_candidate_raw
                else None
            )
            recommended_alloy = (
                recommended_candidate_raw["alloy"]
                if recommended_candidate_raw is not None
                else best_alloy
            )
            if recommended_candidate is not None:
                active_learning_mode = (
                    "explore"
                    if (
                        recommended_candidate["novelty_score"] >= 0.45
                        or recommended_candidate["model_uncertainty"] >= 0.55
                    )
                    else "exploit"
                )
                recommendation_reason = (
                    f"Selected for {active_learning_mode}: "
                    f"fitness={recommended_candidate['weighted_fitness']}, "
                    f"novelty={recommended_candidate['novelty_score']}, "
                    f"uncertainty={recommended_candidate['model_uncertainty']}, "
                    f"acquisition={recommended_candidate['acquisition_score']}"
                )
            else:
                active_learning_mode = "exploit"
                recommendation_reason = "No ranked candidate pool was available; defaulted to best alloy."

            best_fitness = float(best_alloy.properties.get("fitness", 0.0))
            evaluation_method = str(
                best_alloy.properties.get("evaluation_method", evaluation_method)
            )
            if evaluation_backend == "hybrid_calphad":
                evaluation_stats = hybrid_evaluator.get_statistics() if hybrid_evaluator else {}
            previous_candidates = [alloy.composition for alloy in seed_alloys]
            previous_scores = [
                float(alloy.properties.get("fitness", 0.0))
                for alloy in seed_alloys
                if alloy.properties.get("fitness") is not None
            ]
            belief_state = BeliefState.from_alloy(
                best_alloy,
                history={
                    "previous_candidates": previous_candidates,
                    "previous_scores": previous_scores,
                    "failed_regions": [],
                },
                summary={
                    "search_mode": search_mode,
                    "objectives": objective_names,
                    "weights": objective_weights,
                    "strategy_source": (
                        strategy_agent.last_source if strategy_agent is not None else "random_only"
                    ),
                    "evaluation_backend": evaluation_backend,
                    "calphad_database": calphad_database,
                    "active_learning_mode": active_learning_mode,
                    "acquisition_score": best_alloy.properties.get("acquisition_score"),
                },
            )
            recommended_belief_state = BeliefState.from_alloy(
                recommended_alloy,
                history={
                    "previous_candidates": previous_candidates,
                    "previous_scores": previous_scores,
                    "failed_regions": [],
                },
                summary={
                    "search_mode": search_mode,
                    "objectives": objective_names,
                    "weights": objective_weights,
                    "strategy_source": (
                        strategy_agent.last_source if strategy_agent is not None else "random_only"
                    ),
                    "evaluation_backend": evaluation_backend,
                    "calphad_database": calphad_database,
                    "active_learning_mode": active_learning_mode,
                    "acquisition_score": recommended_alloy.properties.get("acquisition_score"),
                },
            )
            validation_memory_record = None
            if recommended_candidate is not None:
                recommended_signature = _composition_signature(
                    recommended_candidate.get("composition", {})
                )
                best_signature = _composition_signature(best_alloy.composition)
                if recommended_signature and recommended_signature != best_signature:
                    validation_memory_record = _normalize_memory_record({
                        "composition": recommended_candidate.get("composition", {}),
                        "fitness": recommended_candidate.get("fitness"),
                        "properties": {
                            "objective_scores": recommended_candidate.get("objective_scores", {}),
                            "weighted_fitness": recommended_candidate.get("weighted_fitness"),
                            "acquisition_score": recommended_candidate.get("acquisition_score"),
                            "novelty_score": recommended_candidate.get("novelty_score"),
                            "model_uncertainty": recommended_candidate.get("model_uncertainty"),
                            "data_density": recommended_candidate.get("data_density"),
                            "calphad_confidence": recommended_candidate.get("calphad_confidence"),
                            "evaluation_method": recommended_candidate.get("evaluation_method"),
                            "search_mode": search_mode,
                            "evaluation_backend": evaluation_backend,
                            "strategy_source": (
                                strategy_agent.last_source if strategy_agent is not None else "random_only"
                            ),
                        },
                        "belief_state": recommended_belief_state.to_dict(),
                        "notes": (
                            "Active-learning recommended validation candidate "
                            f"({active_learning_mode}) from adaptive evolution search."
                        ),
                    })

            memory_record = _normalize_memory_record({
                "best_composition": {
                    element: round(fraction, 6)
                    for element, fraction in sorted(best_alloy.composition.items())
                },
                "best_fitness": round(best_fitness, 6),
                "objective_scores": best_alloy.properties.get("objective_scores", {}),
                "weighted_fitness": best_alloy.properties.get("weighted_fitness", best_fitness),
                "acquisition_score": best_alloy.properties.get("acquisition_score"),
                "novelty_score": best_alloy.properties.get("novelty_score"),
                "model_uncertainty": best_alloy.properties.get("model_uncertainty"),
                "data_density": best_alloy.properties.get("data_density"),
                "search_mode": search_mode,
                "strategy_source": (
                    strategy_agent.last_source if strategy_agent is not None else "random_only"
                ),
                "evaluation_backend": evaluation_backend,
                "calphad_database": calphad_database,
                "evaluation_method": evaluation_method,
                "evolution_summary": engine.get_evolution_summary(),
                "belief_state": belief_state.to_dict(),
                "notes": (
                    f"Adaptive evolution search over {','.join(resolved_allowed)} "
                    f"with objectives {', '.join(objective_names)}; "
                    f"active_learning={active_learning_mode}"
                ),
            })

            result = {
                "best_composition": {
                    element: round(fraction, 6)
                    for element, fraction in sorted(best_alloy.composition.items())
                },
                "best_fitness": round(best_fitness, 6),
                "objective_scores": best_alloy.properties.get("objective_scores", {}),
                "weighted_fitness": best_alloy.properties.get("weighted_fitness", best_fitness),
                "search_mode": search_mode,
                "constraints": constraints,
                "objectives": objective_names,
                "weights": objective_weights,
                "evaluation_backend": evaluation_backend,
                "evaluation_method": evaluation_method,
                "evaluation_stats": _round_nested(evaluation_stats),
                "calphad_database": calphad_database,
                "calphad_warning": calphad_warning,
                "strategy_source": (
                    strategy_agent.last_source if strategy_agent is not None else "random_only"
                ),
                "strategy_warning": strategy_warning,
                "strategy_graphrag": (
                    strategy_agent.last_graphrag_summary
                    if strategy_agent is not None
                    else None
                ),
                "mutation": best_alloy.metadata.get("mutation", {}),
                "evolution_summary": engine.get_evolution_summary(),
                "belief_state": belief_state.to_dict(),
                "memory_record": memory_record,
                "validation_memory_record": validation_memory_record,
                "active_learning": {
                    "enabled": bool(active_learning_settings.get("enabled", True)),
                    "mode": active_learning_mode,
                    "reason": recommendation_reason,
                    "memory_reference_count": len(reference_compositions),
                    "scoring_weights": {
                        "fitness_weight": active_learning_settings.get("fitness_weight", 0.6),
                        "uncertainty_weight": active_learning_settings.get("uncertainty_weight", 0.2),
                        "novelty_weight": active_learning_settings.get("novelty_weight", 0.2),
                    },
                    "recommended_candidate": recommended_candidate,
                    "recommended_belief_state": recommended_belief_state.to_dict(),
                    "top_candidates": top_acquisition_candidates,
                },
            }
            return json.dumps(result, ensure_ascii=False, default=_json_default)
        except Exception as e:
            return json.dumps({"error": str(e)}, ensure_ascii=False)


class MemoryReadInput(BaseModel):
    query: str = Field(description="Search query for historical alloy data")


class MemoryReadTool(BaseTool):
    """从内存/历史文件读取之前搜索过的合金信息。"""

    name: str = "memory_reader"
    description: str = (
        "Search historical alloy optimization results from memory. "
        "Returns best previously found compositions and their properties."
    )
    args_schema: type[BaseModel] = MemoryReadInput

    _memory_file: str = str(_ROOT / ".ark" / "output" / "hea_memory.json")

    def _run(self, query: str) -> str:
        try:
            records = _load_memory_records(Path(self._memory_file))
            if not records:
                return json.dumps({"message": "No memory found yet.", "records": []})
            # 简单关键字过滤
            keywords = query.lower().split()
            matched = [r for r in records if any(k in json.dumps(r).lower() for k in keywords)]
            return json.dumps({"matched_count": len(matched), "records": matched[:5]}, ensure_ascii=False)
        except Exception as e:
            return f"Error reading memory: {e}"


class MaterialsGraphRAGInput(BaseModel):
    query: str = Field(description="GraphRAG query over HEA memory, elements, properties, and surrogate models")
    max_results: Optional[int] = Field(default=6, description="Maximum number of graph nodes to retrieve")
    include_models: bool = Field(default=True, description="Whether to include surrogate model candidates")


class MaterialsGraphRAGTool(BaseTool):
    """GraphRAG 工具：把 memory、元素、性能与代理模型候选组织成图检索。"""

    name: str = "materials_graphrag"
    description: str = (
        "Query a lightweight materials GraphRAG index built from long-term HEA memory, "
        "element nodes, observed properties, and external surrogate-model candidates "
        "such as PaddleMaterials, MatGL/M3GNet, CHGNet, ALIGNN, and CrabNet. "
        "Use it before planning, property prediction, surrogate-model selection, "
        "or active-learning policy updates."
    )
    args_schema: type[BaseModel] = MaterialsGraphRAGInput

    _memory_file: str = str(_ROOT / ".ark" / "output" / "hea_memory.json")

    def _run(
        self,
        query: str,
        max_results: Optional[int] = 6,
        include_models: bool = True,
    ) -> str:
        try:
            records = _load_memory_records(Path(self._memory_file))
            result = query_materials_graph(
                query=query,
                memory_records=records,
                max_results=max_results or 6,
                include_models=include_models,
            )
            return json.dumps(result, ensure_ascii=False, default=_json_default)
        except Exception as e:
            return json.dumps({"error": str(e)}, ensure_ascii=False)


class MemoryWriteInput(BaseModel):
    record_json: str = Field(description="JSON string of alloy record to save")


class MemoryWriteTool(BaseTool):
    """将新发现的高价值合金成分保存到记忆文件。"""

    name: str = "memory_writer"
    description: str = (
        "Save an alloy optimization result to long-term memory for future reference. "
        "Input: JSON record with composition, fitness, and properties."
    )
    args_schema: type[BaseModel] = MemoryWriteInput

    _memory_file: str = str(_ROOT / ".ark" / "output" / "hea_memory.json")

    def _run(self, record_json: str) -> str:
        try:
            try:
                parsed_record = json.loads(record_json)
            except Exception:
                parsed_record = _extract_json_object_from_text(record_json)

            if not isinstance(parsed_record, dict):
                raise ValueError("record_json must contain exactly one JSON object")

            record = _normalize_memory_record(parsed_record)
            memory_path = Path(self._memory_file)
            memory_path.parent.mkdir(parents=True, exist_ok=True)

            with _MEMORY_FILE_LOCK:
                repair_info = _repair_memory_file(memory_path)
                existing_records = _load_memory_records(memory_path)
                merged_records = _merge_memory_records(existing_records + [record])
                memory_payload = {"records": merged_records}
                _write_memory_payload_atomic(memory_path, memory_payload)

            response = {
                "saved": True,
                "total_records": len(memory_payload["records"]),
                "top_composition": record.get("composition", {}),
                "fitness": record.get("fitness"),
            }
            if repair_info.get("repaired"):
                response["recovered_memory_file"] = True
                response["backup_path"] = repair_info.get("backup_path")
            if repair_info.get("recovered_from_backup_paths"):
                response["recovered_from_backup_paths"] = repair_info.get("recovered_from_backup_paths")
            elif repair_info.get("created"):
                response["created_memory_file"] = True

            return json.dumps(response, ensure_ascii=False)
        except Exception as e:
            return f"Error writing memory: {e}"


# ─────────────────────────── Agent 定义 ───────────────────────────


def build_agents(
    llm,
    runtime_config: Optional[Dict[str, Any]] = None,
    element_pool: Optional[List[str]] = None,
    llm_model: str = "gpt-5.2",
    llm_api: Optional[str] = None,
    llm_base_url: Optional[str] = None,
    provider: Optional[str] = None,
) -> Dict[str, Agent]:
    """构建六大智能体角色"""
    runtime_config = runtime_config or load_runtime_config()
    element_pool = element_pool or []
    resolved_llm_api = llm_api or provider

    planner = Agent(
        role="HEA Optimization Planner",
        goal=(
            "Analyze the user's alloy design objectives, decompose them into quantitative "
            "performance targets (strength, ductility, phase stability), and produce a "
            "step-by-step optimization strategy with element pool selection."
        ),
        backstory=(
            "You are a senior materials scientist with expertise in high-entropy alloy design. "
            "You understand multi-objective optimization, alloy thermodynamics, and how to "
            "translate vague performance requirements into concrete search parameters. "
            "You always output a structured JSON plan."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=True,
    )

    knowledge_agent = Agent(
        role="Materials Knowledge Agent",
        goal=(
            "Retrieve and summarize relevant knowledge about the target alloy system: "
            "known phase diagrams, prior experimental data, literature alloy compositions, "
            "and thermodynamic constraints. Provide grounded context for other agents."
        ),
        backstory=(
            "You are a materials informatics specialist who has read thousands of papers on "
            "high-entropy alloys. You can recall empirical rules (Hume-Rothery, VEC, "
            "Ω parameter) and connect them to practical alloy design decisions. "
            "You use memory tools to leverage prior search history."
        ),
        llm=llm,
        tools=[MemoryReadTool(), MaterialsGraphRAGTool()],
        verbose=True,
    )

    thermo_agent = Agent(
        role="Thermodynamic Evaluator Agent",
        goal=(
            "Evaluate candidate alloy compositions using thermodynamic models: "
            "compute ΔS_mix, ΔH_mix, δ parameter via fast analytical models and "
            "CALPHAD phase equilibrium when available. Report phase stability scores."
        ),
        backstory=(
            "You are a computational thermodynamics expert. You can invoke pycalphad for "
            "precise Gibbs free energy minimization, or fall back to simplified analytical "
            "models when the database lacks data. You provide quantitative scores and "
            "explain whether a composition is likely to form a single FCC phase or "
            "risky intermetallic compounds."
        ),
        llm=llm,
        tools=[ThermodynamicEvalTool(), CALPHADEvalTool()],
        verbose=True,
    )

    property_agent = Agent(
        role="Property Prediction Agent",
        goal=(
            "Predict mechanical properties (yield strength, ultimate tensile strength, "
            "ductility, hardness) and functional properties (corrosion resistance, "
            "oxidation resistance) of candidate alloy compositions using empirical models "
            "and ML surrogate models."
        ),
        backstory=(
            "You are an alloy properties prediction specialist. You apply rule-of-mixture "
            "models, Hall-Petch estimates, and empirical correlations between composition, "
            "VEC (valence electron concentration), and properties. "
            "When data is sparse you clearly communicate uncertainty."
        ),
        llm=llm,
        tools=[ThermodynamicEvalTool(), MaterialsGraphRAGTool()],
        verbose=True,
    )

    optimization_agent = Agent(
        role="Optimization Strategy Agent",
        goal=(
            "Drive the evolutionary search: select next candidate compositions using "
            "genetic algorithm mutations, Bayesian acquisition (exploit vs explore), "
            "and MEL (Material Edit Language) operations. Integrate feedback from "
            "evaluations to improve search efficiency."
        ),
        backstory=(
            "You are an optimization algorithms expert combining evolutionary computing with "
            "Bayesian active learning. You generate MEL mutation operations informed by "
            "thermodynamic and property scores, maintaining a diverse population while "
            "converging toward high-performance regions of composition space."
        ),
        llm=llm,
        tools=[
            EvolutionSearchTool(
                element_pool=element_pool,
                runtime_config=runtime_config,
                llm_model=llm_model,
                llm_api=resolved_llm_api,
                llm_base_url=llm_base_url,
            ),
            MELMutationTool(
                llm_model=llm_model,
                llm_api=resolved_llm_api,
                llm_base_url=llm_base_url,
                runtime_config=runtime_config,
            ),
            ThermodynamicEvalTool(),
            MaterialsGraphRAGTool(),
        ],
        verbose=True,
    )

    memory_agent = Agent(
        role="Memory and Learning Agent",
        goal=(
            "Maintain the agent's long-term memory: record all evaluated compositions "
            "with their scores, identify high-value regions, flag failed/toxic compositions, "
            "and summarize search progress. Enable knowledge reuse across optimization runs."
        ),
        backstory=(
            "You are a knowledge management expert for materials discovery. You organize "
            "experimental and simulation results, identify patterns in successful alloy "
            "compositions, and provide summaries that help the Planner make better decisions "
            "in future iterations."
        ),
        llm=llm,
        tools=[MemoryReadTool(), MemoryWriteTool(), MaterialsGraphRAGTool()],
        verbose=True,
    )

    return {
        "planner": planner,
        "knowledge": knowledge_agent,
        "thermodynamic": thermo_agent,
        "property": property_agent,
        "optimization": optimization_agent,
        "memory": memory_agent,
    }


# ─────────────────────────── Task 定义 ───────────────────────────


def build_tasks(agents: Dict[str, Agent], user_requirement: str, element_pool: List[str]) -> List[Task]:
    """构建工作流 Tasks（顺序执行）"""

    task_plan = Task(
        description=textwrap.dedent(f"""
            用户的合金设计需求如下：
            {user_requirement}

            可用元素池：{element_pool}

            请完成以下工作：
            1. 将需求拆解为量化的性能目标（如：屈服强度 > 800 MPa，延伸率 > 15%）
            2. 推荐初始搜索的元素子集（3-6个元素）
            3. 设定各目标的权重（合计为1.0）
            4. 给出搜索策略（进化代数、种群大小、变异重点）

            输出格式（JSON）：
            {{
              "performance_targets": {{"strength": "...", "ductility": "..."}},
              "element_subset": ["Co", "Cr", "Fe", "Ni", "V"],
              "weights": {{"stability": 0.3, "strength": 0.3, "ductility": 0.2, "cost": 0.2}},
              "strategy": "..."
            }}
        """),
        expected_output="JSON optimization plan with element subset, weights, and strategy",
        agent=agents["planner"],
    )

    task_knowledge = Task(
        description=textwrap.dedent(f"""
            基于规划智能体的目标（参考上一步输出），从历史记忆和材料知识中获取：
            1. 该元素体系中已知的高性能合金成分（从 memory_reader 搜索）
            2. 调用 materials_graphrag 检索 memory-元素-性能-代理模型图，特别关注：
               - 已有高分/失败成分与元素富集趋势
               - 屈服强度、延伸率、CALPHAD 置信度相关证据路径
               - 可接入代理模型（优先考虑 PaddleMaterials，同时比较 MatGL/CHGNet/ALIGNN/CrabNet）
            3. 关键经验规则（HEA 判据：ΔS_mix > 11.5 J/mol·K，δ < 6%，Ω > 1.1）
            4. 该元素组合中容易出现的有害相（σ相、Laves 相等）及规避方法

            输出：结构化的先验知识摘要，包含推荐初始成分范围、GraphRAG 证据路径和代理模型接入建议
        """),
        expected_output="Prior knowledge summary with recommended starting composition ranges, GraphRAG evidence paths, and surrogate model suggestions",
        agent=agents["knowledge"],
        context=[task_plan],
    )

    task_evaluate_candidates = Task(
        description=textwrap.dedent("""
            根据规划输出的元素子集，生成 3 个候选初始合金成分并进行热力学评估：

            候选成分生成规则：
            - 成分1：等原子比 (equiatomic)
            - 成分2：Ni 或 Co 富集（占比约 35%，其余均分）
            - 成分3：基于先验知识推荐的高熵组合

            对每个候选成分：
            1. 调用 thermodynamic_evaluator 计算 ΔS_mix, ΔH_mix, δ 参数
            2. 尝试调用 calphad_evaluator 进行相平衡验证
            3. 给出相稳定性综合评分（0-1）

            输出：3个候选成分及其评估结果的对比表格
        """),
        expected_output="Table of 3 candidate compositions with thermodynamic scores",
        agent=agents["thermodynamic"],
        context=[task_plan, task_knowledge],
    )

    task_property_predict = Task(
        description=textwrap.dedent("""
            对热力学评估中得分最高的候选成分，进行力学性能预测：

            0. 先调用 materials_graphrag 查询与“yield strength / ductility / PaddleMaterials surrogate”相关的 memory 和模型候选
            1. 基于 VEC 预测相结构（VEC > 8.0 → FCC; VEC < 6.87 → BCC; 中间 → 混合相）
            2. 基于规则混合 (rule of mixture) 估算：
               - 屈服强度（MPa）
               - 极限拉伸强度（MPa）
               - 延伸率（%）
               - 维氏硬度（HV）
            3. 评估耐腐蚀性（Cr 含量相关）
            4. 明确说明当前预测是规则/代理模型，PaddleMaterials/MatGL/CHGNet 等结构模型是否可用
            5. 给出性能置信度（低/中/高）

            输出：详细的力学性能预测报告，包含 GraphRAG 检索摘要和可接入代理模型建议
        """),
        expected_output="Mechanical property prediction report with GraphRAG evidence, surrogate model suggestions, and confidence levels",
        agent=agents["property"],
        context=[task_evaluate_candidates],
    )

    task_optimize = Task(
        description=textwrap.dedent(f"""
            基于热力学和性能评估结果，执行一轮进化搜索优化：

            1. 调用 materials_graphrag 查询“高分区、失败区、uncertainty、surrogate model”，用于确定 exploit/explore 策略
            2. 以得分最高的候选成分为基础，优先调用 evolution_search_runner
            3. 将候选成分作为 seed_composition_json 传入，并携带 allowed_elements
            4. 让真实 AdaptiveEvolutionEngine 跑一轮多目标遗传搜索
            5. 检查输出中的 active_learning 字段，识别下一步最值得验证的候选
            6. 如需补充说明，可再调用 mel_mutation_generator 分析关键变异方向
            7. 输出最优成分、目标得分、evolution_summary、belief_state、active_learning 推荐和 GraphRAG 证据摘要

            优化目标（来自规划输出）：{element_pool}

            输出：结构化 JSON，总结最优成分 + 适应度对比 + belief_state + active_learning + GraphRAG evidence
        """),
        expected_output="Structured JSON with best composition, fitness scores, evolution summary, belief_state, active_learning recommendation, and GraphRAG evidence",
        agent=agents["optimization"],
        context=[task_evaluate_candidates, task_property_predict],
    )

    task_memory = Task(
        description=textwrap.dedent("""
            将本次优化轮次的所有关键结果保存到长期记忆，并生成总结报告：

            1. 优先从 optimization task 输出中提取 memory_record 和 validation_memory_record，并调用 memory_writer 保存
            2. 如需手工构造记录，保存格式为：
               {"composition": {...}, "fitness": 0.xx, "properties": {...}, "belief_state": {...}, "notes": "..."}
            3. memory_writer 会自动做成分归一化、重复候选去重和结构化存储
            4. 结合 active_learning.recommended_candidate，总结下一步优先验证对象
            5. 标记发现的高价值区域（哪些元素比例范围效果好）
            6. 标记失败区域（有害相风险高的组合）
            7. 生成最终推荐报告，包含：
               - 最优成分及其量化预测性能
               - 下一步优先 CALPHAD/实验验证候选及原因
                - 推荐热处理工艺
                - 后续实验验证建议
                - 搜索效率统计

            输出：完整的优化总结 Markdown 报告
        """),
        expected_output="Complete optimization summary report in Markdown with best composition recommendation",
        agent=agents["memory"],
        context=[task_plan, task_evaluate_candidates, task_property_predict, task_optimize],
    )

    return [
        task_plan,
        task_knowledge,
        task_evaluate_candidates,
        task_property_predict,
        task_optimize,
        task_memory,
    ]


# ─────────────────────────── Crew 入口 ───────────────────────────


class HEACrewOptimizer:
    """
    高熵合金 CrewAI 多智能体优化器

    使用方式:
        optimizer = HEACrewOptimizer(
            element_pool=["Co", "Cr", "Fe", "Ni", "V"],
            user_requirement="设计一种800°C下屈服强度>900MPa，延伸率>10%的高熵合金"
        )
        result = optimizer.run()
    """

    def __init__(
        self,
        element_pool: Optional[List[str]] = None,
        user_requirement: str = "设计一种热稳定性好、高强度的五元高熵合金",
        model: str = "gpt-5.2",
        llm_api: Optional[str] = None,
        base_url: Optional[str] = None,
        provider: Optional[str] = None,
        temperature: float = 0.7,
        verbose: bool = True,
        config_path: Optional[str] = None,
    ):
        if config_path:
            os.environ["HEA_CONFIG_PATH"] = str(Path(config_path).expanduser().resolve())
        self.element_pool = element_pool or ["Co", "Cr", "Fe", "Ni", "V"]
        self.user_requirement = user_requirement
        self.verbose = verbose
        self.model = model
        self.runtime_config = load_runtime_config(config_path)
        crewai_config = self.runtime_config.get("crewai", {})
        self.llm_api = (
            llm_api
            or provider
            or os.getenv("HEA_LLM_API")
            or os.getenv("HEA_LLM_PROVIDER")
            or crewai_config.get("default_llm_api")
            or crewai_config.get("default_provider")
            or "llmone"
        )
        self.provider = self.llm_api
        self.llm_base_url = base_url
        self.llm = get_crewai_llm(
            model=model,
            temperature=temperature,
            llm_api=self.llm_api,
            base_url=self.llm_base_url,
        )
        self.last_crew_output: Any = None
        self.last_saved_memory_records: List[Dict[str, Any]] = []
        self.last_memory_sync_messages: List[str] = []
        self.last_optimization_payload: Optional[Dict[str, Any]] = None

    def _collect_memory_records_from_crew_output(self, crew_output: Any) -> List[Dict[str, Any]]:
        payload_candidates: List[Dict[str, Any]] = []
        raw_text_candidates: List[str] = []
        records: List[Dict[str, Any]] = []

        crew_payload = getattr(crew_output, "json_dict", None)
        if isinstance(crew_payload, dict):
            payload_candidates.append(crew_payload)

        crew_raw = getattr(crew_output, "raw", None)
        if isinstance(crew_raw, str) and crew_raw.strip():
            raw_text_candidates.append(crew_raw)

        for task_output in getattr(crew_output, "tasks_output", []) or []:
            payload = _extract_task_payload(task_output)
            if isinstance(payload, dict):
                payload_candidates.append(payload)
            raw_text = getattr(task_output, "raw", None)
            if isinstance(raw_text, str) and raw_text.strip():
                raw_text_candidates.append(raw_text)

        selected_payload = next(
            (
                payload
                for payload in reversed(payload_candidates)
                if (
                    _find_nested_record(payload, "memory_record") is not None
                    or _find_nested_record(payload, "validation_memory_record") is not None
                    or "active_learning" in payload
                )
            ),
            None,
        )
        self.last_optimization_payload = selected_payload

        if selected_payload is None:
            selected_payload = {}

        for key in ("memory_record", "validation_memory_record"):
            record = _find_nested_record(selected_payload, key)
            if isinstance(record, dict):
                records.append(_normalize_memory_record(record))

        for raw_text in raw_text_candidates:
            for payload in _extract_json_objects_from_text(raw_text):
                for key in ("memory_record", "validation_memory_record"):
                    nested_record = _find_nested_record(payload, key)
                    if isinstance(nested_record, dict):
                        records.append(_normalize_memory_record(nested_record))
                if _looks_like_memory_record(payload):
                    records.append(_normalize_memory_record(payload))

        deduped_records: Dict[Any, Dict[str, Any]] = {}
        for record in records:
            signature = _composition_signature(record.get("composition", {}))
            if not signature:
                continue
            current = deduped_records.get(signature)
            if current is None or float(record.get("fitness", 0.0)) >= float(current.get("fitness", 0.0)):
                deduped_records[signature] = record

        return list(deduped_records.values())

    def _persist_memory_records(self, records: List[Dict[str, Any]]) -> None:
        self.last_saved_memory_records = []
        self.last_memory_sync_messages = []
        if not records:
            self.last_memory_sync_messages.append("No structured memory records found in crew output.")
            return

        memory_writer = MemoryWriteTool()
        for record in records:
            response = memory_writer._run(json.dumps(record, ensure_ascii=False))
            try:
                payload = json.loads(response)
            except Exception:
                payload = {"saved": False, "message": response}

            if payload.get("saved"):
                self.last_saved_memory_records.append(record)
            self.last_memory_sync_messages.append(str(payload))

    def _serialize_task_output(self, task_output: Any, index: int) -> Dict[str, Any]:
        payload = _extract_task_payload(task_output)

        agent_name = None
        agent = getattr(task_output, "agent", None)
        if isinstance(agent, str):
            agent_name = agent
        elif agent is not None:
            agent_name = (
                getattr(agent, "role", None)
                or getattr(agent, "name", None)
                or str(agent)
            )

        description = getattr(task_output, "description", None)
        title = getattr(task_output, "name", None) or getattr(task_output, "title", None)
        if not title and isinstance(description, str) and description.strip():
            title = description.strip().splitlines()[0][:96]

        serialized = {
            "index": index + 1,
            "title": title or f"Task {index + 1}",
            "agent": agent_name,
            "summary": getattr(task_output, "summary", None),
            "description": description,
            "expected_output": getattr(task_output, "expected_output", None),
            "payload": payload,
            "raw": getattr(task_output, "raw", None),
        }
        return _round_nested(_to_plain_data(serialized))

    def build_run_details(self, report: Optional[str] = None) -> Dict[str, Any]:
        crew_output = self.last_crew_output
        report_text = report
        if report_text is None and crew_output is not None:
            report_text = getattr(crew_output, "raw", None) or str(crew_output)

        optimization_payload = _to_plain_data(self.last_optimization_payload) if self.last_optimization_payload else None
        active_learning = None
        if isinstance(optimization_payload, dict):
            active_learning = optimization_payload.get("active_learning")

        details = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "element_pool": list(self.element_pool),
            "requirement": self.user_requirement,
            "model": self.model,
            "llm_api": self.llm_api,
            "llm_base_url": self.llm_base_url,
            "report": report_text or "",
            "tasks": [
                self._serialize_task_output(task_output, index)
                for index, task_output in enumerate(getattr(crew_output, "tasks_output", []) or [])
            ],
            "optimization": optimization_payload,
            "active_learning": active_learning,
            "memory_records": _to_plain_data(self.last_saved_memory_records),
            "memory_sync_messages": list(self.last_memory_sync_messages),
            "token_usage": _to_plain_data(getattr(crew_output, "token_usage", None)),
            "crew_output_text": str(crew_output) if crew_output is not None else "",
        }
        return _round_nested(_to_plain_data(details))

    def run(self) -> str:
        """执行多智能体优化，返回最终报告字符串"""
        agents = build_agents(
            self.llm,
            runtime_config=self.runtime_config,
            element_pool=self.element_pool,
            llm_model=self.model,
            llm_api=self.llm_api,
            llm_base_url=self.llm_base_url,
        )
        tasks = build_tasks(agents, self.user_requirement, self.element_pool)

        crew = Crew(
            agents=list(agents.values()),
            tasks=tasks,
            process=Process.sequential,
            verbose=self.verbose,
        )

        result = crew.kickoff()
        self.last_crew_output = result
        memory_records = self._collect_memory_records_from_crew_output(result)
        self._persist_memory_records(memory_records)
        return str(result)


def main():
    """命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(
        description="HEA CrewAI Multi-Agent Optimizer"
    )
    parser.add_argument(
        "--elements",
        default="Co,Cr,Fe,Ni,V",
        help="Comma-separated element pool (default: Co,Cr,Fe,Ni,V)",
    )
    parser.add_argument(
        "--requirement",
        default="设计一种在800°C下屈服强度>900MPa、延伸率>10%的五元高熵合金",
        help="Natural language optimization requirement",
    )
    parser.add_argument(
        "--model",
        default="gpt-5.2",
        help="LLM model name (default: gpt-5.2)",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output file path for the final report (optional)",
    )
    args = parser.parse_args()

    element_pool = [e.strip() for e in args.elements.split(",")]

    print("=" * 60)
    print("  高熵合金 CrewAI 多智能体优化系统 v1.0")
    print("=" * 60)
    print(f"元素池: {element_pool}")
    print(f"优化需求: {args.requirement}")
    print(f"使用模型: {args.model}")
    print("=" * 60)

    optimizer = HEACrewOptimizer(
        element_pool=element_pool,
        user_requirement=args.requirement,
        model=args.model,
    )

    report = optimizer.run()

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(report, encoding="utf-8")
        print(f"\n报告已保存至: {args.output}")
    else:
        print("\n" + "=" * 60)
        print("最终优化报告:")
        print("=" * 60)
        print(report)


if __name__ == "__main__":
    main()
