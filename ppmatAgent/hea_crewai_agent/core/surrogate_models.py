"""Surrogate model registry for the Agentic Materials Agent.

The registry is intentionally lightweight: it records which external model
families are worth integrating and detects whether their optional Python
packages are available. Heavy model packages are not imported here.
"""

from __future__ import annotations

import importlib.util
from typing import Any, Dict, Iterable, List


SURROGATE_MODEL_CANDIDATES: List[Dict[str, Any]] = [
    {
        "id": "paddlematerials",
        "name": "PaddleMaterials / PPMat",
        "backend": "PaddlePaddle",
        "tasks": [
            "property_prediction",
            "interatomic_potential",
            "structure_generation",
            "electronic_structure",
        ],
        "models": [
            "MEGNet",
            "iComformer",
            "DimeNet++",
            "CHGNet",
            "MatterSim",
            "MatterGen",
            "DiffCSP",
        ],
        "best_for": (
            "百度/PaddlePaddle 生态、国产硬件、结构属性预测、MLIP 和后续生产化部署。"
        ),
        "input_requirements": [
            "CIF/晶体结构用于 property prediction 与 MLIP",
            "composition-only HEA 需要先生成近似结构或接入结构生成器",
        ],
        "package_candidates": ["paddlematerials", "ppmat", "paddle"],
        "integration_priority": 1,
        "source": "https://github.com/PaddlePaddle/PaddleMaterials",
    },
    {
        "id": "matgl",
        "name": "MatGL / M3GNet / MEGNet",
        "backend": "PyTorch + DGL/PyG",
        "tasks": [
            "property_prediction",
            "interatomic_potential",
            "fine_tuning",
        ],
        "models": ["M3GNet", "MEGNet", "TensorNet", "SO3Net", "CHGNet"],
        "best_for": "成熟的材料图神经网络、预训练势函数、结构属性预测和微调。",
        "input_requirements": ["晶体结构/CIF", "元素类型与周期性结构图"],
        "package_candidates": ["matgl"],
        "integration_priority": 2,
        "source": "https://matgl.ai/",
    },
    {
        "id": "chgnet",
        "name": "CHGNet",
        "backend": "PyTorch",
        "tasks": ["interatomic_potential", "relaxation", "molecular_dynamics"],
        "models": ["CHGNet"],
        "best_for": "通用晶体势函数、能量/力/应力预测、结构弛豫和分子动力学。",
        "input_requirements": ["晶体结构/CIF", "pymatgen Structure"],
        "package_candidates": ["chgnet"],
        "integration_priority": 3,
        "source": "https://github.com/CederGroupHub/chgnet",
    },
    {
        "id": "alignn",
        "name": "ALIGNN",
        "backend": "PyTorch + DGL",
        "tasks": ["property_prediction", "phonon", "dos_prediction"],
        "models": ["ALIGNN", "ALIGNN-FF"],
        "best_for": "含键角信息的晶体属性预测、声子/DOS/热力学相关任务。",
        "input_requirements": ["晶体结构/CIF", "JARVIS/ASE Atoms"],
        "package_candidates": ["alignn", "jarvis"],
        "integration_priority": 4,
        "source": "https://github.com/usnistgov/alignn",
    },
    {
        "id": "crabnet",
        "name": "CrabNet",
        "backend": "PyTorch",
        "tasks": ["composition_property_prediction", "interpretable_surrogate"],
        "models": ["Compositionally-Restricted Attention-Based Network"],
        "best_for": "只有成分、没有结构时的快速 composition-only HEA 代理模型。",
        "input_requirements": ["化学式或 composition at.%"],
        "package_candidates": ["crabnet"],
        "integration_priority": 5,
        "source": "https://github.com/anthony-wang/CrabNet",
    },
]


def _module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except Exception:
        return False


def detect_surrogate_backends() -> Dict[str, Dict[str, Any]]:
    """Detect optional surrogate-model backends without importing heavy modules."""

    status: Dict[str, Dict[str, Any]] = {}
    for candidate in SURROGATE_MODEL_CANDIDATES:
        packages = list(candidate.get("package_candidates", []))
        installed_packages = [
            package for package in packages if _module_available(package)
        ]
        status[candidate["id"]] = {
            "installed": bool(installed_packages),
            "installed_packages": installed_packages,
            "package_candidates": packages,
            "adapter_ready": bool(installed_packages),
        }
    return status


def _tokens(text: str) -> set[str]:
    normalized = "".join(
        char.lower() if char.isalnum() else " "
        for char in str(text or "")
    )
    return {token for token in normalized.split() if token}


def _candidate_text(candidate: Dict[str, Any]) -> str:
    fields: List[str] = [
        candidate.get("id", ""),
        candidate.get("name", ""),
        candidate.get("backend", ""),
        candidate.get("best_for", ""),
        " ".join(candidate.get("tasks", [])),
        " ".join(candidate.get("models", [])),
        " ".join(candidate.get("input_requirements", [])),
    ]
    return " ".join(fields)


def _score_candidate(candidate: Dict[str, Any], query_tokens: set[str]) -> float:
    candidate_tokens = _tokens(_candidate_text(candidate))
    overlap = len(query_tokens & candidate_tokens)
    priority_bonus = 1.0 / float(candidate.get("integration_priority", 99))
    return overlap + priority_bonus


def list_surrogate_model_candidates(query: str = "") -> List[Dict[str, Any]]:
    """List model families with install status and simple query ranking."""

    backend_status = detect_surrogate_backends()
    query_tokens = _tokens(query)
    enriched: List[Dict[str, Any]] = []
    for candidate in SURROGATE_MODEL_CANDIDATES:
        item = dict(candidate)
        item["install_status"] = backend_status.get(candidate["id"], {})
        item["score"] = _score_candidate(candidate, query_tokens) if query_tokens else (
            1.0 / float(candidate.get("integration_priority", 99))
        )
        enriched.append(item)

    return sorted(
        enriched,
        key=lambda item: (
            item["score"],
            -float(item.get("integration_priority", 99)),
        ),
        reverse=True,
    )


def recommend_surrogate_stack(
    requirement: str,
    prefer_paddle: bool = True,
    top_k: int = 3,
) -> Dict[str, Any]:
    """Recommend a pragmatic surrogate-model stack for the current task."""

    candidates = list_surrogate_model_candidates(requirement)
    if prefer_paddle:
        candidates = sorted(
            candidates,
            key=lambda item: (item["id"] == "paddlematerials", item["score"]),
            reverse=True,
        )

    recommended = candidates[: max(1, int(top_k))]
    return {
        "recommended": recommended,
        "adapter_plan": [
            "先接 composition-only 或现有启发式 surrogate，保持当前 HEA 流程可运行。",
            "当有 CIF/结构生成能力后，将 PaddleMaterials/MatGL/CHGNet 接入为结构代理模型。",
            "训练触发由 agentic_materials_agent.model_state.training_trigger 控制。",
        ],
        "detected_backends": detect_surrogate_backends(),
    }


def model_nodes_for_graphrag(query: str = "") -> List[Dict[str, Any]]:
    """Return graph-ready model nodes."""

    nodes: List[Dict[str, Any]] = []
    for candidate in list_surrogate_model_candidates(query):
        nodes.append(
            {
                "id": f"model:{candidate['id']}",
                "type": "surrogate_model",
                "label": candidate["name"],
                "content": candidate["best_for"],
                "metadata": candidate,
            }
        )
    return nodes
