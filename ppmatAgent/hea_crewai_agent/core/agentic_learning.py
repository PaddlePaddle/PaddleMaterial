"""Agentic learning state for the materials discovery workflow.

The module does not train a neural network directly. It exposes the decision
state that tells the agent when a deep-learning surrogate is ready to train,
when more labels are needed, and what scientific questions should be generated
for the next closed-loop iteration.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ppmatAgent.hea_crewai_agent.core.surrogate_models import recommend_surrogate_stack


DEFAULT_AGENTIC_MATERIALS_SETTINGS: Dict[str, Any] = {
    "enabled": True,
    "surrogate_family": "composition_graph_neural_surrogate",
    "target_fitness": 0.92,
    "min_train_samples": 24,
    "performance_floor": 0.82,
    "uncertainty_collect_threshold": 0.55,
    "calphad_confidence_threshold": 0.45,
    "problem_batch_size": 4,
    "property_targets": {
        "yield_strength_MPa": {"min": 900, "unit": "MPa"},
        "elongation_percent": {"min": 10, "unit": "%"},
    },
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _to_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return default


def _deep_merge(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def resolve_agentic_materials_settings(runtime_config: Dict[str, Any]) -> Dict[str, Any]:
    """Return normalized Agentic Materials Agent settings."""

    configured = _as_dict(runtime_config.get("agentic_materials"))
    settings = _deep_merge(DEFAULT_AGENTIC_MATERIALS_SETTINGS, configured)

    return {
        "enabled": bool(settings.get("enabled", True)),
        "surrogate_family": str(
            settings.get("surrogate_family")
            or DEFAULT_AGENTIC_MATERIALS_SETTINGS["surrogate_family"]
        ),
        "target_fitness": float(settings.get("target_fitness", 0.92)),
        "min_train_samples": max(1, int(settings.get("min_train_samples", 24))),
        "performance_floor": float(settings.get("performance_floor", 0.82)),
        "uncertainty_collect_threshold": float(
            settings.get("uncertainty_collect_threshold", 0.55)
        ),
        "calphad_confidence_threshold": float(
            settings.get("calphad_confidence_threshold", 0.45)
        ),
        "problem_batch_size": max(1, int(settings.get("problem_batch_size", 4))),
        "property_targets": _as_dict(settings.get("property_targets")),
    }


def _normalize_composition(raw: Any) -> Dict[str, float]:
    if not isinstance(raw, dict):
        return {}

    composition = raw
    for key in (
        "composition",
        "composition_at_percent",
        "composition_atomic_fraction",
        "as_at_percent",
        "elements",
    ):
        nested = raw.get(key)
        if isinstance(nested, dict):
            composition = nested
            break

    cleaned: Dict[str, float] = {}
    total = 0.0
    for element, value in composition.items():
        number = _to_float(value)
        if number is None or number <= 0:
            continue
        cleaned[str(element)] = number
        total += number

    if total <= 0:
        return {}

    if total > 1.000001:
        return {
            element: round(number / total, 6)
            for element, number in sorted(cleaned.items())
        }

    return {
        element: round(number, 6)
        for element, number in sorted(cleaned.items())
    }


def _format_composition(composition: Dict[str, float]) -> str:
    if not composition:
        return "—"
    return " · ".join(
        f"{element} {fraction * 100:.1f} at.%"
        for element, fraction in sorted(composition.items())
    )


def _extract_fitness(record: Dict[str, Any]) -> Optional[float]:
    candidates = [
        record.get("fitness"),
        record.get("best_fitness"),
        record.get("weighted_fitness"),
        _as_dict(record.get("properties")).get("weighted_fitness"),
        _as_dict(record.get("properties")).get("fitness"),
    ]
    for value in candidates:
        number = _to_float(value)
        if number is not None:
            return number
    return None


def _extract_uncertainty(record: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    properties = _as_dict(record.get("properties"))
    belief_state = _as_dict(record.get("belief_state"))
    uncertainty = _as_dict(belief_state.get("uncertainty"))

    model_uncertainty = (
        _to_float(record.get("model_uncertainty"))
        or _to_float(properties.get("model_uncertainty"))
        or _to_float(uncertainty.get("model_uncertainty"))
    )
    calphad_confidence = (
        _to_float(record.get("calphad_confidence"))
        or _to_float(properties.get("calphad_confidence"))
        or _to_float(uncertainty.get("calphad_confidence"))
    )
    return model_uncertainty, calphad_confidence


def _extract_property(record: Dict[str, Any], metric: str) -> Optional[float]:
    properties = _as_dict(record.get("properties"))
    aliases = {
        "yield_strength_MPa": [
            "yield_strength_MPa",
            "predicted_yield_strength_MPa",
            "yield_strength_mpa",
            "predicted_yield_strength",
            "yield_strength",
        ],
        "elongation_percent": [
            "elongation_percent",
            "predicted_elongation_percent",
            "elongation",
            "ductility_percent",
        ],
    }

    for key in aliases.get(metric, [metric]):
        value = properties.get(key)
        if isinstance(value, dict):
            value = value.get("recommended") or value.get("mean") or value.get("value")
        number = _to_float(value)
        if number is not None:
            return number
    return None


def _rank_records(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    ranked: List[Dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        fitness = _extract_fitness(record)
        composition = _normalize_composition(record)
        ranked.append(
            {
                "raw": record,
                "fitness": fitness,
                "composition": composition,
                "model_uncertainty": _extract_uncertainty(record)[0],
                "calphad_confidence": _extract_uncertainty(record)[1],
            }
        )

    return sorted(
        ranked,
        key=lambda item: item["fitness"] if item["fitness"] is not None else -1.0,
        reverse=True,
    )


def _build_performance_state(
    ranked_records: List[Dict[str, Any]],
    settings: Dict[str, Any],
) -> Dict[str, Any]:
    best = ranked_records[0] if ranked_records else None
    best_record = _as_dict(best.get("raw")) if best else {}
    best_fitness = best.get("fitness") if best else None
    target_fitness = settings["target_fitness"]
    property_gaps: List[Dict[str, Any]] = []

    for metric, target in settings["property_targets"].items():
        target_map = _as_dict(target)
        minimum = _to_float(target_map.get("min"))
        if minimum is None:
            continue

        observed = _extract_property(best_record, metric)
        satisfied = observed is not None and observed >= minimum
        property_gaps.append(
            {
                "metric": metric,
                "observed": observed,
                "target_min": minimum,
                "unit": target_map.get("unit") or "",
                "satisfied": bool(satisfied),
                "gap": round(minimum - observed, 6) if observed is not None and not satisfied else 0.0,
            }
        )

    fitness_satisfied = best_fitness is not None and best_fitness >= target_fitness
    property_satisfied = all(item["satisfied"] for item in property_gaps) if property_gaps else True
    satisfied = bool(fitness_satisfied and property_satisfied)

    if not ranked_records:
        status = "no_memory"
    elif satisfied:
        status = "target_met"
    elif best_fitness is not None and best_fitness < settings["performance_floor"]:
        status = "below_floor"
    else:
        status = "target_gap"

    return {
        "status": status,
        "satisfied": satisfied,
        "target_fitness": target_fitness,
        "best_fitness": best_fitness,
        "best_composition": best.get("composition") if best else {},
        "best_composition_text": _format_composition(best.get("composition", {}) if best else {}),
        "property_gaps": property_gaps,
    }


def _build_model_state(
    dataset_size: int,
    performance: Dict[str, Any],
    settings: Dict[str, Any],
) -> Dict[str, Any]:
    min_train_samples = settings["min_train_samples"]
    train_ready = dataset_size >= min_train_samples

    if performance["satisfied"]:
        trigger = "monitor"
        recommendation = "当前目标已满足，保持监控并继续用新数据校准代理模型。"
    elif train_ready:
        trigger = "train_surrogate"
        recommendation = "数据量已达到训练阈值，下一步应训练/刷新深度学习代理模型。"
    else:
        trigger = "collect_data_first"
        recommendation = "性能目标未满足，但样本不足；应先补充有标签的计算/实验数据。"

    surrogate_recommendation = recommend_surrogate_stack(
        "high entropy alloy yield strength ductility property prediction",
        prefer_paddle=True,
        top_k=3,
    )

    return {
        "surrogate_family": settings["surrogate_family"],
        "dataset_size": dataset_size,
        "min_train_samples": min_train_samples,
        "train_ready": train_ready,
        "training_trigger": trigger,
        "features": [
            "composition_at_percent",
            "mixing_entropy",
            "mixing_enthalpy",
            "atomic_radius_mismatch",
            "VEC",
            "phase_descriptors",
        ],
        "labels": ["weighted_fitness", *settings["property_targets"].keys()],
        "recommended_action": recommendation,
        "surrogate_recommendation": surrogate_recommendation,
    }


def _build_element_learning(ranked_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not ranked_records:
        return {"status": "empty", "elements": [], "summary": "暂无 memory 样本，无法学习元素贡献。"}

    fitness_values = [
        item["fitness"]
        for item in ranked_records
        if item["fitness"] is not None
    ]
    mean_fitness = sum(fitness_values) / len(fitness_values) if fitness_values else None
    top_count = max(1, min(len(ranked_records), len(ranked_records) // 3 or 1))
    top_records = ranked_records[:top_count]
    all_elements = sorted(
        {
            element
            for item in ranked_records
            for element in item["composition"].keys()
        }
    )

    elements = []
    for element in all_elements:
        values = [
            item["composition"].get(element, 0.0)
            for item in ranked_records
            if item["composition"]
        ]
        top_values = [
            item["composition"].get(element, 0.0)
            for item in top_records
            if item["composition"]
        ]
        if not values:
            continue
        mean_value = sum(values) / len(values)
        top_mean = sum(top_values) / len(top_values) if top_values else 0.0
        enrichment = top_mean - mean_value
        if enrichment > 0.015:
            role = "top_region_enriched"
        elif enrichment < -0.015:
            role = "top_region_depleted"
        else:
            role = "neutral_or_uncertain"
        elements.append(
            {
                "element": element,
                "mean_at_percent": round(mean_value * 100, 3),
                "top_region_mean_at_percent": round(top_mean * 100, 3),
                "enrichment_in_top_at_percent": round(enrichment * 100, 3),
                "role": role,
            }
        )

    return {
        "status": "learned" if len(ranked_records) >= 3 else "low_sample",
        "record_count": len(ranked_records),
        "mean_fitness": round(mean_fitness, 6) if mean_fitness is not None else None,
        "elements": elements,
        "summary": "基于当前长期记忆估计元素在高分区域的富集/规避趋势。",
    }


def _build_data_collection_queue(
    ranked_records: List[Dict[str, Any]],
    performance: Dict[str, Any],
    model_state: Dict[str, Any],
    settings: Dict[str, Any],
) -> List[Dict[str, Any]]:
    queue: List[Dict[str, Any]] = []
    dataset_size = model_state["dataset_size"]
    missing_samples = max(0, model_state["min_train_samples"] - dataset_size)

    if missing_samples:
        queue.append(
            {
                "priority": "P0",
                "type": "labeled_training_data",
                "source": "memory + CALPHAD + targeted experiments",
                "request": f"补充至少 {missing_samples} 条 composition-property 标签样本",
                "reason": "深度学习代理模型训练样本不足，不能可靠刷新 surrogate。",
            }
        )

    for gap in performance.get("property_gaps", []):
        if gap.get("satisfied"):
            continue
        queue.append(
            {
                "priority": "P0",
                "type": "property_label",
                "source": "mechanical surrogate / experiment / literature extraction",
                "request": (
                    f"围绕 {performance.get('best_composition_text', '当前最优候选')} "
                    f"补充 {gap['metric']} 标签；当前 {gap.get('observed')}，目标 ≥ {gap.get('target_min')} {gap.get('unit', '')}"
                ),
                "reason": "当前最优候选未满足关键性能约束。",
            }
        )

    uncertain = [
        item for item in ranked_records[:6]
        if (
            (item.get("model_uncertainty") or 0.0) >= settings["uncertainty_collect_threshold"]
            or (item.get("calphad_confidence") or 1.0) <= settings["calphad_confidence_threshold"]
        )
    ]
    if uncertain:
        queue.append(
            {
                "priority": "P1",
                "type": "uncertainty_reduction",
                "source": "CALPHAD validation / ablation batch",
                "request": "优先验证高不确定性或低 CALPHAD 置信度的高分候选",
                "reason": "降低模型外推风险，提升下一轮 acquisition 的可靠性。",
                "candidate_count": len(uncertain),
            }
        )

    if not queue and not performance["satisfied"]:
        queue.append(
            {
                "priority": "P1",
                "type": "search_space_expansion",
                "source": "problem generator",
                "request": "扩展元素组合或局部微调当前高分区域，构造新一批候选。",
                "reason": "性能仍未达标，需要新的探索问题而非重复当前搜索。",
            }
        )

    return queue


def _build_generated_questions(
    performance: Dict[str, Any],
    element_learning: Dict[str, Any],
    queue: List[Dict[str, Any]],
    batch_size: int,
) -> List[str]:
    questions: List[str] = []
    best_text = performance.get("best_composition_text") or "当前最优候选"

    for gap in performance.get("property_gaps", []):
        if gap.get("satisfied"):
            continue
        questions.append(
            f"如何在保持 {best_text} 相稳定性的同时，把 {gap['metric']} 从 {gap.get('observed', '未知')} 提升到 ≥ {gap.get('target_min')} {gap.get('unit', '')}？"
        )

    enriched = [
        item for item in element_learning.get("elements", [])
        if item.get("role") == "top_region_enriched"
    ][:3]
    if enriched:
        element_text = "、".join(item["element"] for item in enriched)
        questions.append(
            f"高分 memory 中 {element_text} 出现富集趋势，下一轮是否应围绕这些元素做局部成分扫描？"
        )

    if any(item.get("type") == "uncertainty_reduction" for item in queue):
        questions.append(
            "哪些高 acquisition 候选最能降低模型不确定性，并且值得优先做 CALPHAD 或实验验证？"
        )

    if any(item.get("type") == "labeled_training_data" for item in queue):
        questions.append(
            "应该选择哪些覆盖高分区、失败区和边界区的成分点，来组成下一批深度学习训练样本？"
        )

    if not questions:
        questions.append(
            "当前目标基本满足后，是否应转向鲁棒性、多温度窗口或成本约束下的二级优化？"
        )

    return questions[:batch_size]


def _capability(status: str, title: str, detail: str) -> Dict[str, str]:
    return {"status": status, "title": title, "detail": detail}


def _build_capability_status(
    performance: Dict[str, Any],
    model_state: Dict[str, Any],
    queue: List[Dict[str, Any]],
    questions: List[str],
) -> Dict[str, Any]:
    if model_state["training_trigger"] == "train_surrogate":
        surrogate_status = "ready"
    elif model_state["training_trigger"] == "collect_data_first":
        surrogate_status = "waiting_for_data"
    else:
        surrogate_status = "monitoring"

    return {
        "memory_learning": _capability(
            "active",
            "长期记忆学习",
            "从历史 composition-fitness-property 记录中提取元素趋势和失败区域。",
        ),
        "data_collection": _capability(
            "active" if queue else "standby",
            "自动数据收集",
            f"当前数据队列 {len(queue)} 项。",
        ),
        "data_analysis": _capability(
            "active",
            "数据分析",
            f"性能状态：{performance['status']}。",
        ),
        "deep_learning_surrogate": _capability(
            surrogate_status,
            "深度学习代理模型",
            model_state["recommended_action"],
        ),
        "problem_generator": _capability(
            "active" if questions else "standby",
            "问题生成器",
            f"已生成 {len(questions)} 个下一轮科学问题。",
        ),
        "policy_update": _capability(
            "active" if not performance["satisfied"] else "monitoring",
            "策略更新",
            "根据性能缺口、元素趋势和不确定性更新下一轮搜索策略。",
        ),
    }


def _build_self_evolution_loop(
    performance: Dict[str, Any],
    model_state: Dict[str, Any],
    queue: List[Dict[str, Any]],
    questions: List[str],
) -> List[Dict[str, str]]:
    collect_active = model_state["training_trigger"] == "collect_data_first"
    train_active = model_state["training_trigger"] == "train_surrogate"
    monitor_active = model_state["training_trigger"] == "monitor"

    return [
        {
            "id": "observe",
            "label": "Observe memory/history",
            "status": "complete",
            "detail": "读取长期记忆、批次历史和最新性能指标。",
        },
        {
            "id": "diagnose",
            "label": "Diagnose performance gap",
            "status": "complete" if performance["best_fitness"] is not None else "waiting",
            "detail": f"当前状态：{performance['status']}。",
        },
        {
            "id": "collect",
            "label": "Collect missing data",
            "status": "active" if collect_active else ("complete" if queue else "standby"),
            "detail": f"数据收集队列 {len(queue)} 项。",
        },
        {
            "id": "analyze",
            "label": "Analyze element trends",
            "status": "complete",
            "detail": "从高分/低分区域中学习元素富集和规避趋势。",
        },
        {
            "id": "train",
            "label": "Train deep-learning surrogate",
            "status": "active" if train_active else ("waiting" if collect_active else "standby"),
            "detail": model_state["recommended_action"],
        },
        {
            "id": "generate",
            "label": "Generate next problems",
            "status": "active" if questions and not monitor_active else "standby",
            "detail": f"下一轮问题数：{len(questions)}。",
        },
        {
            "id": "update_policy",
            "label": "Update search policy",
            "status": "active" if not performance["satisfied"] else "monitoring",
            "detail": "将数据、模型和问题生成结果反馈给下一轮 CrewAI 搜索。",
        },
    ]


def build_agentic_materials_state(
    *,
    memory_records: List[Dict[str, Any]],
    history_entries: List[Dict[str, Any]],
    runtime_config: Dict[str, Any],
) -> Dict[str, Any]:
    """Build the visible self-evolution state for the Materials Agent."""

    settings = resolve_agentic_materials_settings(runtime_config)
    ranked_records = _rank_records(memory_records)
    performance = _build_performance_state(ranked_records, settings)
    model_state = _build_model_state(len(ranked_records), performance, settings)
    element_learning = _build_element_learning(ranked_records)
    data_collection_queue = _build_data_collection_queue(
        ranked_records,
        performance,
        model_state,
        settings,
    )
    generated_questions = _build_generated_questions(
        performance,
        element_learning,
        data_collection_queue,
        settings["problem_batch_size"],
    )
    capability_status = _build_capability_status(
        performance,
        model_state,
        data_collection_queue,
        generated_questions,
    )
    self_evolution_loop = _build_self_evolution_loop(
        performance,
        model_state,
        data_collection_queue,
        generated_questions,
    )

    return {
        "generated_at": _utc_now(),
        "enabled": settings["enabled"],
        "name": "Agentic Materials Agent",
        "mission": "Use memory, models, data collection, analysis, training, and problem generation to improve the materials-design loop.",
        "history_count": len(_as_list(history_entries)),
        "performance": performance,
        "model_state": model_state,
        "element_learning": element_learning,
        "data_collection_queue": data_collection_queue,
        "problem_generator": {
            "status": "active" if generated_questions else "standby",
            "generated_questions": generated_questions,
        },
        "capability_status": capability_status,
        "self_evolution_loop": self_evolution_loop,
    }
