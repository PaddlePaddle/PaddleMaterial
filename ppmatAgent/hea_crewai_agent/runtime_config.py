"""Runtime configuration helpers for the HEA CrewAI project."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    import yaml
except ImportError:  # pragma: no cover - dependency guard
    yaml = None


_PROJECT_ROOT = Path(__file__).resolve().parent
_DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "config.yaml"
_DEFAULT_ENV_PATH = _PROJECT_ROOT / ".env"

_DEFAULT_RUNTIME_CONFIG: Dict[str, Any] = {
    "evolution": {
        "population_size": 50,
        "max_generations": 100,
        "mutation_rate": 0.15,
        "crossover_rate": 0.8,
        "elite_ratio": 0.1,
    },
    "constraints": {
        "max_elements": 7,
        "min_concentration": 0.05,
        "max_concentration": 0.40,
        "allowed_elements": [
            "Co",
            "Cr",
            "Fe",
            "Ni",
            "Mn",
            "Al",
            "Ti",
            "V",
            "Mo",
            "W",
            "Cu",
            "Nb",
            "Ta",
        ],
    },
    "crewai": {
        "default_elements": ["Co", "Cr", "Fe", "Ni", "V"],
        "default_requirement": (
            "设计一种在800°C下屈服强度>900MPa、延伸率>10%的五元高熵合金，要求热力学稳定"
        ),
        "default_llm_api": "llmone",
        "default_model": "gpt-5.2",
        "output_report": ".ark/output/crewai_result/report.md",
    },
    "calphad": {
        "database_path": None,
        "allow_simplified_database": False,
        "default_temperature_k": 1273.15,
        "use_hybrid_evaluator": False,
        "calphad_threshold": 0.72,
        "use_calphad_probability": 0.1,
    },
    "evaluation": {
        "fitness_profile": "single_phase",
        "objectives": [
            {"name": "thermodynamic", "weight": 0.3, "target": "maximize"},
            {"name": "mechanical", "weight": 0.5, "target": "maximize"},
            {"name": "corrosion", "weight": 0.2, "target": "maximize"},
        ],
    },
    "strategy_agent": {
        "enabled": False,
        "model": "gpt-5.2",
        "temperature": 0.2,
        "max_tokens": 160,
        "graphrag": {
            "enabled": True,
            "max_results": 5,
        },
    },
    "active_learning": {
        "enabled": True,
        "fitness_weight": 0.6,
        "uncertainty_weight": 0.2,
        "novelty_weight": 0.2,
        "candidate_pool_size": 5,
    },
    "loop": {
        "default_run_mode": "single",
        "max_rounds": 3,
        "patience": 1,
        "min_improvement": 0.01,
        "enable_validation": True,
        "stop_on_stable_validation": False,
    },
    "agentic_materials": {
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
    },
}


def _deep_merge(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def resolve_config_path(config_path: Optional[str | Path] = None) -> Path:
    candidate = config_path or os.getenv("HEA_CONFIG_PATH") or _DEFAULT_CONFIG_PATH
    return Path(candidate).expanduser().resolve()


def resolve_env_path(env_path: Optional[str | Path] = None) -> Path:
    candidate = env_path or os.getenv("HEA_ENV_PATH") or _DEFAULT_ENV_PATH
    return Path(candidate).expanduser().resolve()


def load_env_file(
    env_path: Optional[str | Path] = None,
    override: bool = False,
) -> Dict[str, str]:
    path = resolve_env_path(env_path)
    loaded: Dict[str, str] = {}

    if not path.exists():
        return loaded

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue

        if (
            len(value) >= 2
            and value[0] == value[-1]
            and value[0] in {"'", '"'}
        ):
            value = value[1:-1]

        if override or key not in os.environ:
            os.environ[key] = value
        loaded[key] = os.environ.get(key, value)

    return loaded


def load_runtime_config(config_path: Optional[str | Path] = None) -> Dict[str, Any]:
    path = resolve_config_path(config_path)
    loaded: Dict[str, Any] = {}

    if path.exists():
        if yaml is None:
            raise ImportError(
                "PyYAML is required to load config.yaml. Run `pip install -r ppmatAgent/requirements-optional.txt`."
            )
        with path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"Config file must contain a top-level mapping: {path}")
        loaded = raw

    return _deep_merge(_DEFAULT_RUNTIME_CONFIG, loaded)


def get_crewai_defaults(runtime_config: Dict[str, Any]) -> Dict[str, Any]:
    crewai_config = runtime_config.get("crewai", {})
    return {
        "default_elements": list(crewai_config.get("default_elements", [])),
        "default_requirement": crewai_config.get(
            "default_requirement",
            _DEFAULT_RUNTIME_CONFIG["crewai"]["default_requirement"],
        ),
        "default_model": crewai_config.get(
            "default_model",
            _DEFAULT_RUNTIME_CONFIG["crewai"]["default_model"],
        ),
        "default_llm_api": crewai_config.get(
            "default_llm_api",
            crewai_config.get(
                "default_provider",
                _DEFAULT_RUNTIME_CONFIG["crewai"]["default_llm_api"],
            ),
        ),
        "output_report": crewai_config.get(
            "output_report",
            _DEFAULT_RUNTIME_CONFIG["crewai"]["output_report"],
        ),
    }


def get_evaluation_search_mode(runtime_config: Dict[str, Any]) -> str:
    evaluation = runtime_config.get("evaluation", {})
    mode = str(evaluation.get("fitness_profile", "single_phase")).strip().lower()
    if mode not in {"single_phase", "eutectic"}:
        return "single_phase"
    return mode


def get_evolution_settings(runtime_config: Dict[str, Any]) -> Dict[str, Any]:
    evolution = runtime_config.get("evolution", {})
    defaults = _DEFAULT_RUNTIME_CONFIG["evolution"]
    return {
        "population_size": int(evolution.get("population_size", defaults["population_size"])),
        "max_generations": int(evolution.get("max_generations", defaults["max_generations"])),
        "mutation_rate": float(evolution.get("mutation_rate", defaults["mutation_rate"])),
        "crossover_rate": float(evolution.get("crossover_rate", defaults["crossover_rate"])),
        "elite_ratio": float(evolution.get("elite_ratio", defaults["elite_ratio"])),
    }


def get_constraint_settings(
    runtime_config: Dict[str, Any],
    allowed_elements: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    constraints = runtime_config.get("constraints", {})
    defaults = _DEFAULT_RUNTIME_CONFIG["constraints"]
    resolved_allowed = list(allowed_elements or constraints.get("allowed_elements") or defaults["allowed_elements"])

    return {
        "max_elements": int(constraints.get("max_elements", min(len(resolved_allowed), defaults["max_elements"]))),
        "min_concentration": float(
            constraints.get("min_concentration", defaults["min_concentration"])
        ),
        "max_concentration": float(
            constraints.get("max_concentration", defaults["max_concentration"])
        ),
        "allowed_elements": resolved_allowed,
    }


def get_evaluation_objectives(runtime_config: Dict[str, Any]) -> Dict[str, List[Any]]:
    evaluation = runtime_config.get("evaluation", {})
    configured = evaluation.get("objectives", [])
    objectives: List[str] = []
    weights: List[float] = []

    for item in configured:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        objectives.append(name)
        weights.append(float(item.get("weight", 0.0)))

    if not objectives:
        defaults = _DEFAULT_RUNTIME_CONFIG["evaluation"]["objectives"]
        objectives = [item["name"] for item in defaults]
        weights = [float(item["weight"]) for item in defaults]

    return {"objectives": objectives, "weights": weights}


def get_active_learning_settings(runtime_config: Dict[str, Any]) -> Dict[str, Any]:
    active_learning = runtime_config.get("active_learning", {})
    defaults = _DEFAULT_RUNTIME_CONFIG["active_learning"]
    return {
        "enabled": bool(active_learning.get("enabled", defaults["enabled"])),
        "fitness_weight": float(
            active_learning.get("fitness_weight", defaults["fitness_weight"])
        ),
        "uncertainty_weight": float(
            active_learning.get("uncertainty_weight", defaults["uncertainty_weight"])
        ),
        "novelty_weight": float(
            active_learning.get("novelty_weight", defaults["novelty_weight"])
        ),
        "candidate_pool_size": int(
            active_learning.get(
                "candidate_pool_size", defaults["candidate_pool_size"]
            )
        ),
    }


def get_calphad_settings(runtime_config: Dict[str, Any]) -> Dict[str, Any]:
    calphad = runtime_config.get("calphad", {})
    return {
        "database_path": calphad.get("database_path"),
        "allow_simplified_database": bool(
            calphad.get("allow_simplified_database", False)
        ),
        "default_temperature_k": float(
            calphad.get("default_temperature_k", 1273.15)
        ),
        "use_hybrid_evaluator": bool(
            calphad.get("use_hybrid_evaluator", False)
        ),
        "calphad_threshold": float(
            calphad.get("calphad_threshold", 0.72)
        ),
        "use_calphad_probability": float(
            calphad.get("use_calphad_probability", 0.1)
        ),
    }


def get_loop_settings(runtime_config: Dict[str, Any]) -> Dict[str, Any]:
    loop = runtime_config.get("loop", {})
    defaults = _DEFAULT_RUNTIME_CONFIG["loop"]
    run_mode = str(loop.get("default_run_mode", defaults["default_run_mode"])).strip().lower()
    if run_mode not in {"single", "loop"}:
        run_mode = defaults["default_run_mode"]

    return {
        "default_run_mode": run_mode,
        "max_rounds": int(loop.get("max_rounds", defaults["max_rounds"])),
        "patience": int(loop.get("patience", defaults["patience"])),
        "min_improvement": float(
            loop.get("min_improvement", defaults["min_improvement"])
        ),
        "enable_validation": bool(
            loop.get("enable_validation", defaults["enable_validation"])
        ),
        "stop_on_stable_validation": bool(
            loop.get(
                "stop_on_stable_validation",
                defaults["stop_on_stable_validation"],
            )
        ),
    }


def get_agentic_materials_settings(runtime_config: Dict[str, Any]) -> Dict[str, Any]:
    agentic = runtime_config.get("agentic_materials", {})
    defaults = _DEFAULT_RUNTIME_CONFIG["agentic_materials"]
    merged = _deep_merge(defaults, agentic if isinstance(agentic, dict) else {})

    return {
        "enabled": bool(merged.get("enabled", defaults["enabled"])),
        "surrogate_family": str(
            merged.get("surrogate_family", defaults["surrogate_family"])
        ),
        "target_fitness": float(
            merged.get("target_fitness", defaults["target_fitness"])
        ),
        "min_train_samples": int(
            merged.get("min_train_samples", defaults["min_train_samples"])
        ),
        "performance_floor": float(
            merged.get("performance_floor", defaults["performance_floor"])
        ),
        "uncertainty_collect_threshold": float(
            merged.get(
                "uncertainty_collect_threshold",
                defaults["uncertainty_collect_threshold"],
            )
        ),
        "calphad_confidence_threshold": float(
            merged.get(
                "calphad_confidence_threshold",
                defaults["calphad_confidence_threshold"],
            )
        ),
        "problem_batch_size": int(
            merged.get("problem_batch_size", defaults["problem_batch_size"])
        ),
        "property_targets": dict(
            merged.get("property_targets", defaults["property_targets"])
        ),
    }


def _load_tdb_registry_module():
    module_path = _PROJECT_ROOT / "core" / "tdb_registry.py"
    spec = importlib.util.spec_from_file_location("_hea_tdb_registry", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load TDB registry module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_explicit_database_entry(
    database_path: Path,
    registry_entries: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not database_path.exists():
        return None

    matched_entry = next(
        (entry for entry in registry_entries if Path(entry["path"]).resolve() == database_path),
        None,
    )

    if matched_entry is not None:
        if not matched_entry.get("pycalphad_loadable", False):
            return None
        return matched_entry

    return {
        "filename": database_path.name,
        "path": str(database_path),
        "exists": True,
        "pycalphad_loadable": True,
        "database_kind": "custom",
        "supported_elements": set(),
        "recommended_phases": None,
        "recommended_use": "User-provided CALPHAD database path.",
        "system_name": database_path.stem,
    }


def resolve_calphad_database(
    required_elements: Iterable[str],
    runtime_config: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    runtime_config = runtime_config or load_runtime_config()
    calphad_settings = get_calphad_settings(runtime_config)

    tdb_registry = _load_tdb_registry_module()
    registry_entries = tdb_registry.get_local_tdb_registry(_PROJECT_ROOT / "tdb_files")

    explicit_path = os.getenv("CALPHAD_DATABASE") or calphad_settings["database_path"]
    if explicit_path:
        explicit_entry = _build_explicit_database_entry(
            Path(explicit_path).expanduser().resolve(),
            registry_entries,
        )
        if explicit_entry is not None:
            return explicit_entry

    return tdb_registry.find_best_local_tdb(
        required_elements,
        tdb_dir=_PROJECT_ROOT / "tdb_files",
        allow_simplified=calphad_settings["allow_simplified_database"],
    )


def describe_dependency_status() -> Dict[str, List[str]]:
    required_modules = ["crewai", "yaml"]
    optional_modules = ["pycalphad", "ase"]

    missing_required = [
        module for module in required_modules if importlib.util.find_spec(module) is None
    ]
    missing_optional = [
        module for module in optional_modules if importlib.util.find_spec(module) is None
    ]

    return {
        "missing_required": missing_required,
        "missing_optional": missing_optional,
    }
