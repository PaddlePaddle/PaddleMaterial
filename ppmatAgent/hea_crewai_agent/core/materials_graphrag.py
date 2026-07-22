"""Lightweight GraphRAG over HEA memory and surrogate-model knowledge.

This is a local, dependency-free graph retrieval layer. It follows the GraphRAG
idea of retrieving through entities and relationships, while staying small
enough to run inside the current CrewAI tool without a separate indexing job.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Tuple

from ppmatAgent.hea_crewai_agent.core.surrogate_models import list_surrogate_model_candidates


PROPERTY_ALIASES = {
    "yield": "predicted_yield_strength_MPa",
    "strength": "predicted_yield_strength_MPa",
    "ductility": "predicted_elongation_percent",
    "elongation": "predicted_elongation_percent",
    "calphad": "calphad_confidence",
    "phase": "phase_stability",
    "fitness": "fitness",
}


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _tokens(text: str) -> set[str]:
    normalized = "".join(
        char.lower() if char.isalnum() else " "
        for char in str(text or "")
    )
    return {token for token in normalized.split() if token}


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
        if number <= 0:
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


def _composition_label(composition: Dict[str, float]) -> str:
    if not composition:
        return "unknown composition"
    return " ".join(
        f"{element}{fraction * 100:.1f}"
        for element, fraction in sorted(composition.items())
    )


def _composition_signature(composition: Dict[str, float]) -> str:
    return "|".join(
        f"{element}:{fraction:.4f}"
        for element, fraction in sorted(composition.items())
    )


def _node(
    node_id: str,
    node_type: str,
    label: str,
    content: str = "",
    metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    return {
        "id": node_id,
        "type": node_type,
        "label": label,
        "content": content,
        "metadata": metadata or {},
    }


def _edge(
    source: str,
    target: str,
    relation: str,
    weight: float = 1.0,
    metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    return {
        "source": source,
        "target": target,
        "relation": relation,
        "weight": round(float(weight), 6),
        "metadata": metadata or {},
    }


def _record_content(record: Dict[str, Any], composition: Dict[str, float]) -> str:
    properties = _as_dict(record.get("properties"))
    snippets = [
        _composition_label(composition),
        f"fitness={record.get('fitness', record.get('best_fitness', 'unknown'))}",
        str(record.get("notes", "")),
        json.dumps(properties, ensure_ascii=False)[:800],
    ]
    return " | ".join(item for item in snippets if item)


def build_materials_knowledge_graph(
    memory_records: List[Dict[str, Any]],
    include_models: bool = True,
    model_query: str = "",
) -> Dict[str, Any]:
    """Build a small entity graph from long-term memory and model candidates."""

    node_map: Dict[str, Dict[str, Any]] = {}
    edges: List[Dict[str, Any]] = []

    def add_node(item: Dict[str, Any]) -> None:
        if item["id"] not in node_map:
            node_map[item["id"]] = item

    for index, record in enumerate(memory_records):
        if not isinstance(record, dict):
            continue
        composition = _normalize_composition(record)
        if not composition:
            continue

        signature = _composition_signature(composition)
        composition_id = f"composition:{signature}"
        add_node(
            _node(
                composition_id,
                "composition",
                _composition_label(composition),
                _record_content(record, composition),
                {
                    "fitness": record.get("fitness") or record.get("best_fitness"),
                    "record_index": index,
                    "composition": composition,
                },
            )
        )

        for element, fraction in composition.items():
            element_id = f"element:{element}"
            add_node(_node(element_id, "element", element, f"Element {element}"))
            edges.append(
                _edge(
                    composition_id,
                    element_id,
                    "contains_element",
                    fraction,
                    {"at_percent": round(fraction * 100, 4)},
                )
            )

        properties = _as_dict(record.get("properties"))
        for prop_name, prop_value in properties.items():
            if isinstance(prop_value, (dict, list)):
                continue
            property_id = f"property:{prop_name}"
            add_node(
                _node(
                    property_id,
                    "property",
                    prop_name,
                    f"{prop_name} observed in HEA memory.",
                )
            )
            edges.append(
                _edge(
                    composition_id,
                    property_id,
                    "has_property",
                    1.0,
                    {"value": prop_value},
                )
            )

    if include_models:
        for model in list_surrogate_model_candidates(model_query):
            model_id = f"model:{model['id']}"
            add_node(
                _node(
                    model_id,
                    "surrogate_model",
                    model["name"],
                    model["best_for"],
                    model,
                )
            )
            for task in model.get("tasks", []):
                task_id = f"task:{task}"
                add_node(_node(task_id, "model_task", task, f"Surrogate task: {task}"))
                edges.append(_edge(model_id, task_id, "supports_task", 1.0))
            for requirement in model.get("input_requirements", []):
                requirement_id = f"input:{requirement}"
                add_node(
                    _node(
                        requirement_id,
                        "input_requirement",
                        requirement,
                        "Input needed by surrogate model.",
                    )
                )
                edges.append(_edge(model_id, requirement_id, "requires_input", 1.0))

    return {
        "nodes": list(node_map.values()),
        "edges": edges,
        "node_count": len(node_map),
        "edge_count": len(edges),
    }


def _expanded_query_tokens(query: str) -> set[str]:
    tokens = _tokens(query)
    expanded = set(tokens)
    for token in tokens:
        alias = PROPERTY_ALIASES.get(token)
        if alias:
            expanded.update(_tokens(alias))
    return expanded


def _rank_nodes(nodes: Iterable[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
    query_tokens = _expanded_query_tokens(query)
    model_intent = bool(
        query_tokens
        & {
            "surrogate",
            "model",
            "models",
            "paddlematerials",
            "matgl",
            "m3gnet",
            "chgnet",
            "alignn",
            "crabnet",
        }
    )
    ranked: List[Dict[str, Any]] = []
    for node in nodes:
        node_text = " ".join(
            [
                node.get("id", ""),
                node.get("type", ""),
                node.get("label", ""),
                node.get("content", ""),
                json.dumps(node.get("metadata", {}), ensure_ascii=False)[:1000],
            ]
        )
        overlap = len(query_tokens & _tokens(node_text))
        if overlap <= 0:
            continue
        type_bonus = {
            "composition": 0.4,
            "surrogate_model": 0.6,
            "property": 0.3,
            "element": 0.2,
        }.get(node.get("type"), 0.0)
        if model_intent and node.get("type") == "surrogate_model":
            type_bonus += 2.5
        ranked.append({**node, "score": overlap + type_bonus})

    return sorted(ranked, key=lambda item: item["score"], reverse=True)


def _neighbors(node_id: str, graph: Dict[str, Any], limit: int = 4) -> List[Dict[str, Any]]:
    node_by_id = {node["id"]: node for node in graph["nodes"]}
    paths: List[Dict[str, Any]] = []
    for edge in graph["edges"]:
        if edge["source"] == node_id:
            other = node_by_id.get(edge["target"])
            if other:
                paths.append({"edge": edge, "node": other})
        elif edge["target"] == node_id:
            other = node_by_id.get(edge["source"])
            if other:
                paths.append({"edge": edge, "node": other})
        if len(paths) >= limit:
            break
    return paths


def _build_answer(results: List[Dict[str, Any]]) -> str:
    if not results:
        return "GraphRAG 未找到匹配证据。建议先扩大 query 或补充 memory 数据。"

    model_hits = [item for item in results if item.get("type") == "surrogate_model"]
    composition_hits = [item for item in results if item.get("type") == "composition"]
    element_hits = [item for item in results if item.get("type") == "element"]

    parts: List[str] = []
    if model_hits:
        parts.append(
            "可接入代理模型: "
            + "; ".join(
                f"{item['label']} ({item['metadata'].get('backend', 'unknown backend')})"
                for item in model_hits[:3]
            )
        )
    if composition_hits:
        parts.append(
            "相关 memory 候选: "
            + "; ".join(
                f"{item['label']} fitness={item['metadata'].get('fitness', '—')}"
                for item in composition_hits[:3]
            )
        )
    if element_hits:
        parts.append(
            "相关元素节点: "
            + ", ".join(item["label"] for item in element_hits[:5])
        )

    return "；".join(parts) if parts else "GraphRAG 找到若干相关属性/任务节点，请查看 results 和 evidence_paths。"


def query_materials_graph(
    *,
    query: str,
    memory_records: List[Dict[str, Any]],
    max_results: int = 6,
    include_models: bool = True,
) -> Dict[str, Any]:
    """Query the local materials graph and return evidence paths."""

    graph = build_materials_knowledge_graph(
        memory_records,
        include_models=include_models,
        model_query=query,
    )
    ranked = _rank_nodes(graph["nodes"], query)[: max(1, int(max_results))]
    evidence_paths: List[Dict[str, Any]] = []
    for node in ranked:
        evidence_paths.append(
            {
                "center": {
                    "id": node["id"],
                    "type": node["type"],
                    "label": node["label"],
                    "score": node["score"],
                },
                "neighbors": [
                    {
                        "relation": item["edge"]["relation"],
                        "node_id": item["node"]["id"],
                        "node_type": item["node"]["type"],
                        "label": item["node"]["label"],
                        "weight": item["edge"]["weight"],
                        "metadata": item["edge"].get("metadata", {}),
                    }
                    for item in _neighbors(node["id"], graph)
                ],
            }
        )

    return {
        "query": query,
        "answer": _build_answer(ranked),
        "results": [
            {
                "id": node["id"],
                "type": node["type"],
                "label": node["label"],
                "score": round(float(node["score"]), 4),
                "content": node.get("content", ""),
                "metadata": node.get("metadata", {}),
            }
            for node in ranked
        ],
        "evidence_paths": evidence_paths,
        "graph": {
            "node_count": graph["node_count"],
            "edge_count": graph["edge_count"],
        },
    }
