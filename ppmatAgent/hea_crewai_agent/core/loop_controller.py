"""Automatic multi-round loop controller for the HEA CrewAI workflow."""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Optional

from ppmatAgent.hea_crewai_agent.core.material import Alloy
from ppmatAgent.hea_crewai_agent.core.state import BeliefState
from ppmatAgent.hea_crewai_agent.runtime_config import (
    get_calphad_settings,
    get_evaluation_search_mode,
    load_runtime_config,
    resolve_calphad_database,
)

try:
    from ppmatAgent.hea_crewai_agent.core.calphad_evaluator import CALPHADEvaluator, PYCALPHAD_AVAILABLE
except ImportError:  # pragma: no cover - optional dependency
    CALPHADEvaluator = None
    PYCALPHAD_AVAILABLE = False


EventCallback = Callable[[str, str, str, str, Optional[Dict[str, Any]]], None]
LogCallback = Callable[[str], None]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _normalize_composition(raw: Any) -> Dict[str, float]:
    if not isinstance(raw, dict):
        return {}

    composition = raw
    for key in (
        "composition",
        "composition_at_percent",
        "composition_atomic_fraction",
        "as_at_percent",
    ):
        nested = raw.get(key)
        if isinstance(nested, dict):
            composition = nested
            break

    cleaned: Dict[str, float] = {}
    total = 0.0
    for element, raw_value in composition.items():
        try:
            value = float(raw_value)
        except Exception:
            continue
        if value <= 0:
            continue
        cleaned[str(element)] = value
        total += value

    if total <= 0:
        return {}

    scale = 1.0 / total
    if total > 1.000001:
        scale = 1.0 / total

    return {
        element: round(value * scale, 6)
        for element, value in sorted(cleaned.items())
    }


def _format_composition(composition: Dict[str, float]) -> str:
    if not composition:
        return "—"
    return " · ".join(
        f"{element}{value * 100:.1f}"
        for element, value in sorted(composition.items())
    )


def _extract_best_fitness(run_details: Dict[str, Any]) -> Optional[float]:
    optimization = run_details.get("optimization") or {}
    best = optimization.get("best_fitness")
    if best is not None:
        return _to_float(best)

    scores = (
        optimization.get("fitness_scores")
        or optimization.get("objective_scores")
        or {}
    )
    for key in ("weighted_fitness", "fitness"):
        if scores.get(key) is not None:
            return _to_float(scores.get(key))
    return None


def _extract_validation_candidate(run_details: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    active_learning = run_details.get("active_learning") or {}
    optimization = run_details.get("optimization") or {}
    return (
        active_learning.get("recommended_candidate_to_validate_next")
        or active_learning.get("recommended_next_to_validate")
        or active_learning.get("recommended_candidate")
        or optimization.get("best_composition")
    )


def _extract_phase_risk(phases: Dict[str, float]) -> float:
    harmful_markers = ("SIGMA", "MU", "LAVES", "CHI", "B2")
    risk = 0.0
    for phase, fraction in phases.items():
        if any(marker in str(phase).upper() for marker in harmful_markers):
            risk += _to_float(fraction)
    return round(min(max(risk, 0.0), 1.0), 4)


class AutoLoopController:
    """Run the existing CrewAI optimizer for multiple rounds with validation feedback."""

    def __init__(
        self,
        *,
        element_pool: List[str],
        requirement: str,
        model: str,
        config_path: Optional[str] = None,
        max_rounds: int = 3,
        patience: int = 1,
        min_improvement: float = 0.01,
        enable_validation: bool = True,
        stop_on_stable_validation: bool = False,
        event_callback: Optional[EventCallback] = None,
        log_callback: Optional[LogCallback] = None,
    ):
        self.element_pool = list(element_pool)
        self.requirement = requirement
        self.model = model
        self.config_path = config_path
        self.max_rounds = max(1, int(max_rounds))
        self.patience = max(1, int(patience))
        self.min_improvement = max(0.0, float(min_improvement))
        self.enable_validation = bool(enable_validation)
        self.stop_on_stable_validation = bool(stop_on_stable_validation)
        self.event_callback = event_callback
        self.log_callback = log_callback
        self.runtime_config = load_runtime_config(config_path)
        self.calphad_settings = get_calphad_settings(self.runtime_config)
        self.search_mode = get_evaluation_search_mode(self.runtime_config)

    def _emit(
        self,
        stage: str,
        title: str,
        detail: str = "",
        kind: str = "event",
        payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        if self.event_callback is not None:
            self.event_callback(stage, title, detail, kind, payload)

    def _log(self, line: str) -> None:
        if self.log_callback is not None:
            self.log_callback(line)

    def _round_requirement(
        self,
        round_index: int,
        previous_round: Optional[Dict[str, Any]],
    ) -> str:
        if not previous_round:
            return self.requirement

        validation = previous_round.get("validation") or {}
        validation_candidate = validation.get("candidate_composition") or {}
        phase_risk = _to_float(validation.get("intermetallic_risk"), 0.0)
        calphad_status = validation.get("status") or "not_run"
        calphad_fitness = validation.get("calphad_fitness")
        previous_score = previous_round.get("best_fitness")

        loop_note = (
            f"\n\n[自动闭环第 {round_index} 轮备注]\n"
            f"- 上一轮最高 weighted_fitness: {previous_score if previous_score is not None else '—'}\n"
            f"- 上一轮优先验证候选: {_format_composition(validation_candidate)}\n"
            f"- CALPHAD 状态: {calphad_status}\n"
            f"- CALPHAD fitness: {calphad_fitness if calphad_fitness is not None else '未执行'}\n"
            f"- TCP / 金属间化合物风险: {phase_risk:.4f}\n"
            "- 请结合当前长期记忆，延续高价值区域并主动规避已验证风险区域。"
        )
        return f"{self.requirement}{loop_note}"

    def _validate_candidate(
        self,
        round_index: int,
        run_details: Dict[str, Any],
    ) -> Dict[str, Any]:
        candidate = _extract_validation_candidate(run_details)
        composition = _normalize_composition(candidate)
        if not composition:
            return {
                "performed": False,
                "available": False,
                "reason": "missing_candidate_composition",
            }

        if not self.enable_validation:
            return {
                "performed": False,
                "available": False,
                "reason": "validation_disabled",
                "candidate_composition": composition,
            }

        if not PYCALPHAD_AVAILABLE or CALPHADEvaluator is None:
            return {
                "performed": False,
                "available": False,
                "reason": "pycalphad_unavailable",
                "candidate_composition": composition,
            }

        database_entry = resolve_calphad_database(composition.keys(), self.runtime_config)
        if database_entry is None:
            return {
                "performed": False,
                "available": False,
                "reason": "no_compatible_database",
                "candidate_composition": composition,
            }

        alloy = Alloy(composition=composition)
        evaluator = CALPHADEvaluator(
            database_path=database_entry["path"],
            temperature=self.calphad_settings["default_temperature_k"],
            target_phases=(
                None
                if self.search_mode == "eutectic"
                else database_entry.get("recommended_phases")
            ),
            prefer_single_phase=(self.search_mode != "eutectic"),
            search_mode=self.search_mode,
        )
        calphad_fitness = evaluator.evaluate(alloy)
        phases = dict(alloy.properties.get("calphad_phases", {}) or {})
        intermetallic_risk = _extract_phase_risk(phases)
        belief_state = BeliefState.from_alloy(
            alloy,
            summary={
                "loop_round": round_index,
                "validation_backend": "calphad",
                "database": database_entry["filename"],
            },
        ).to_dict()

        return {
            "performed": True,
            "available": True,
            "reason": None,
            "candidate_composition": composition,
            "calphad_fitness": round(calphad_fitness, 6),
            "database": database_entry["filename"],
            "database_path": database_entry["path"],
            "status": alloy.properties.get("calphad_status", "unknown"),
            "phases": phases,
            "intermetallic_risk": intermetallic_risk,
            "belief_state": belief_state,
        }

    def _persist_validation_memory(
        self,
        round_index: int,
        run_details: Dict[str, Any],
        validation: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not validation.get("performed"):
            return {
                "saved": False,
                "message": validation.get("reason") or "validation_skipped",
                "record": None,
            }

        from ppmatAgent.hea_crewai_agent.agents.hea_crew import MemoryWriteTool

        candidate = validation.get("candidate_composition") or {}
        best_fitness = _extract_best_fitness(run_details)
        record = {
            "composition": candidate,
            "fitness": validation.get("calphad_fitness", best_fitness or 0.0),
            "properties": {
                "calphad_fitness": validation.get("calphad_fitness"),
                "calphad_status": validation.get("status"),
                "calphad_phases": validation.get("phases"),
                "intermetallic_risk": validation.get("intermetallic_risk"),
                "weighted_fitness_reference": best_fitness,
            },
            "belief_state": validation.get("belief_state"),
            "notes": (
                f"Auto loop round {round_index} CALPHAD validation on recommended candidate; "
                f"database={validation.get('database')} status={validation.get('status')}"
            ),
        }

        response = MemoryWriteTool()._run(json.dumps(record, ensure_ascii=False))
        try:
            payload = json.loads(response)
        except Exception:
            payload = {"saved": False, "message": response}

        payload["record"] = record
        return payload

    def _is_stable_validation(self, validation: Dict[str, Any]) -> bool:
        if not validation.get("performed"):
            return False

        phases = validation.get("phases") or {}
        dominant_fraction = max((_to_float(value) for value in phases.values()), default=0.0)
        return (
            validation.get("status") == "success"
            and _to_float(validation.get("intermetallic_risk"), 1.0) <= 0.12
            and dominant_fraction >= 0.65
        )

    def _compose_loop_report(
        self,
        loop_summary: Dict[str, Any],
        primary_round: Dict[str, Any],
    ) -> str:
        lines = [
            "# 自动闭环优化摘要",
            "",
            f"- 运行模式：自动闭环",
            f"- 总轮次：{loop_summary['completed_rounds']} / {loop_summary['max_rounds']}",
            f"- 最优轮次：第 {loop_summary['best_round']} 轮",
            f"- 停止原因：{loop_summary['stop_reason']}",
            f"- 最优 weighted_fitness：{loop_summary['best_fitness'] if loop_summary['best_fitness'] is not None else '—'}",
            "",
            "## 各轮概览",
            "",
        ]

        for round_summary in loop_summary["rounds"]:
            validation = round_summary.get("validation") or {}
            lines.extend(
                [
                    f"- 第 {round_summary['round']} 轮：fitness={round_summary['best_fitness'] if round_summary['best_fitness'] is not None else '—'}；"
                    f"验证={validation.get('status') or validation.get('reason') or 'not_run'}；"
                    f"memory_saved={round_summary.get('validation_memory_saved', False)}",
                ]
            )

        lines.extend(
            [
                "",
                "---",
                "",
                primary_round.get("report") or "",
            ]
        )
        return "\n".join(lines).strip()

    def run(self) -> Dict[str, Any]:
        rounds: List[Dict[str, Any]] = []
        best_fitness: Optional[float] = None
        best_round = 1
        no_improvement_rounds = 0
        stop_reason = "max_rounds_reached"
        saved_memory_records: List[Dict[str, Any]] = []
        memory_sync_messages: List[str] = []

        for round_index in range(1, self.max_rounds + 1):
            previous_round = rounds[-1] if rounds else None
            round_requirement = self._round_requirement(round_index, previous_round)
            self._emit(
                "optimization",
                f"Loop round {round_index} started",
                detail=f"automatic closed-loop round {round_index}/{self.max_rounds}",
                kind="loop",
            )
            self._log(f"[AutoLoop] 第 {round_index} 轮开始，model={self.model}")

            from ppmatAgent.hea_crewai_agent.agents.hea_crew import HEACrewOptimizer

            optimizer = HEACrewOptimizer(
                element_pool=self.element_pool,
                user_requirement=round_requirement,
                model=self.model,
                config_path=self.config_path,
            )
            report = optimizer.run()
            run_details = optimizer.build_run_details(report)
            current_best_fitness = _extract_best_fitness(run_details)

            saved_memory_records.extend(optimizer.last_saved_memory_records)
            memory_sync_messages.extend(optimizer.last_memory_sync_messages)

            validation = self._validate_candidate(round_index, run_details)
            if validation.get("performed"):
                self._emit(
                    "thermodynamic",
                    f"Loop round {round_index} validation completed",
                    detail=(
                        f"CALPHAD {validation.get('status')} | "
                        f"fitness={validation.get('calphad_fitness')} | "
                        f"risk={validation.get('intermetallic_risk')}"
                    ),
                    kind="validation",
                    payload=validation,
                )
                self._log(
                    f"[AutoLoop] 第 {round_index} 轮 CALPHAD 验证完成："
                    f"fitness={validation.get('calphad_fitness')} "
                    f"risk={validation.get('intermetallic_risk')}"
                )
            else:
                self._emit(
                    "thermodynamic",
                    f"Loop round {round_index} validation skipped",
                    detail=str(validation.get("reason") or "validation unavailable"),
                    kind="warning",
                )

            validation_memory = self._persist_validation_memory(round_index, run_details, validation)
            if validation_memory.get("saved") and validation_memory.get("record"):
                saved_memory_records.append(validation_memory["record"])
            memory_sync_messages.append(json.dumps(validation_memory, ensure_ascii=False))

            improved = (
                best_fitness is None
                or (
                    current_best_fitness is not None
                    and current_best_fitness > best_fitness + self.min_improvement
                )
            )
            if improved and current_best_fitness is not None:
                best_fitness = current_best_fitness
                best_round = round_index
                no_improvement_rounds = 0
            else:
                no_improvement_rounds += 1

            round_summary = {
                "round": round_index,
                "started_at": run_details.get("generated_at"),
                "completed_at": _utc_now(),
                "best_fitness": current_best_fitness,
                "best_composition": (run_details.get("optimization") or {}).get("best_composition"),
                "validation": validation,
                "validation_memory_saved": bool(validation_memory.get("saved")),
                "validation_memory_message": validation_memory.get("message"),
                "report": report,
                "run_details": deepcopy(run_details),
            }
            rounds.append(round_summary)

            self._emit(
                "optimization",
                f"Loop round {round_index} completed",
                detail=(
                    f"fitness={current_best_fitness if current_best_fitness is not None else '—'} | "
                    f"improved={improved}"
                ),
                kind="loop",
            )

            if self.stop_on_stable_validation and self._is_stable_validation(validation):
                stop_reason = "stable_validation_reached"
                break

            if no_improvement_rounds >= self.patience:
                stop_reason = f"no_improvement_for_{self.patience}_rounds"
                break

        primary_round = rounds[max(best_round - 1, 0)] if rounds else {
            "report": "",
            "run_details": {},
        }
        loop_summary = {
            "enabled": True,
            "run_mode": "loop",
            "max_rounds": self.max_rounds,
            "completed_rounds": len(rounds),
            "patience": self.patience,
            "min_improvement": self.min_improvement,
            "best_round": best_round,
            "best_fitness": best_fitness,
            "stop_reason": stop_reason,
            "rounds": [
                {
                    "round": item["round"],
                    "completed_at": item["completed_at"],
                    "best_fitness": item["best_fitness"],
                    "best_composition": item["best_composition"],
                    "validation": item["validation"],
                    "validation_memory_saved": item["validation_memory_saved"],
                }
                for item in rounds
            ],
        }
        combined_report = self._compose_loop_report(loop_summary, primary_round)
        final_run_details = deepcopy(primary_round.get("run_details") or {})
        final_run_details["report"] = combined_report
        final_run_details["loop"] = loop_summary
        final_run_details["loop_rounds"] = loop_summary["rounds"]
        final_run_details["memory_sync_messages"] = memory_sync_messages

        return {
            "report": combined_report,
            "run_details": final_run_details,
            "loop_summary": loop_summary,
            "saved_memory_records": saved_memory_records,
            "memory_sync_messages": memory_sync_messages,
        }
