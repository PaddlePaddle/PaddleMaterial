"""Structured belief-state representation for HEA optimization."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from ppmatAgent.hea_crewai_agent.core.material import Alloy, get_element_property


def _weighted_average_property(alloy: Alloy, property_name: str) -> Optional[float]:
    if not alloy.composition:
        return None

    total = 0.0
    for element, fraction in alloy.composition.items():
        total += fraction * get_element_property(element, property_name)
    return total


def _calculate_vec(alloy: Alloy) -> Optional[float]:
    return _weighted_average_property(alloy, "valence_electrons")


def _guess_phase_prediction(alloy: Alloy) -> Optional[str]:
    explicit = alloy.properties.get("phase_prediction") or alloy.properties.get("predicted_phase")
    if explicit:
        return str(explicit)

    vec = _calculate_vec(alloy)
    if vec is None:
        return None
    if vec > 8.0:
        return "FCC"
    if vec < 6.87:
        return "BCC"
    return "FCC+BCC"


def _estimate_intermetallic_risk(alloy: Alloy) -> Optional[float]:
    phases = alloy.properties.get("calphad_phases", {})
    if isinstance(phases, dict) and phases:
        harmful_markers = ("SIGMA", "MU", "LAVES", "CHI", "B2")
        risk = 0.0
        for phase, fraction in phases.items():
            if any(marker in str(phase).upper() for marker in harmful_markers):
                risk += float(fraction)
        return min(max(risk, 0.0), 1.0)

    thermo_score = alloy.properties.get("thermo_fitness")
    if thermo_score is not None:
        return round(max(0.0, 1.0 - float(thermo_score)), 4)
    return None


@dataclass
class DescriptorState:
    atomic_radius_mismatch: Optional[float] = None
    mixing_entropy: Optional[float] = None
    mixing_enthalpy: Optional[float] = None
    vec: Optional[float] = None
    melting_point_average: Optional[float] = None
    density: Optional[float] = None
    cost: Optional[float] = None


@dataclass
class PhaseInfoState:
    predicted_phase: Optional[str] = None
    phase_fraction: Dict[str, float] = field(default_factory=dict)
    solid_solution_probability: Optional[float] = None
    intermetallic_risk: Optional[float] = None
    liquidus_temperature: Optional[float] = None
    solidus_temperature: Optional[float] = None
    calphad_status: Optional[str] = None


@dataclass
class PropertyPredictionState:
    strength: Optional[float] = None
    ductility: Optional[float] = None
    hardness: Optional[float] = None
    corrosion_resistance: Optional[float] = None
    oxidation_resistance: Optional[float] = None


@dataclass
class UncertaintyState:
    model_uncertainty: Optional[float] = None
    calphad_confidence: Optional[float] = None
    data_density: Optional[float] = None


@dataclass
class HistoryState:
    previous_candidates: List[Dict[str, float]] = field(default_factory=list)
    previous_scores: List[float] = field(default_factory=list)
    failed_regions: List[str] = field(default_factory=list)
    last_mutation: Optional[str] = None


@dataclass
class BeliefState:
    composition: Dict[str, float]
    descriptors: DescriptorState
    phase_info: PhaseInfoState
    property_prediction: PropertyPredictionState
    uncertainty: UncertaintyState
    history: HistoryState
    summary: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_alloy(
        cls,
        alloy: Alloy,
        history: Optional[Dict[str, Any]] = None,
        summary: Optional[Dict[str, Any]] = None,
    ) -> "BeliefState":
        history = history or {}
        summary = summary or {}

        descriptors = DescriptorState(
            atomic_radius_mismatch=alloy.properties.get("delta"),
            mixing_entropy=alloy.properties.get("mixing_entropy"),
            mixing_enthalpy=alloy.properties.get("mixing_enthalpy"),
            vec=_calculate_vec(alloy),
            melting_point_average=_weighted_average_property(alloy, "melting_point"),
            density=alloy.properties.get("estimated_density"),
            cost=alloy.properties.get("estimated_cost"),
        )

        phase_info = PhaseInfoState(
            predicted_phase=_guess_phase_prediction(alloy),
            phase_fraction=dict(alloy.properties.get("calphad_phases", {}) or {}),
            solid_solution_probability=alloy.properties.get("calphad_fitness")
            or alloy.properties.get("thermo_fitness"),
            intermetallic_risk=_estimate_intermetallic_risk(alloy),
            liquidus_temperature=alloy.properties.get("calphad_liquidus_temp"),
            solidus_temperature=alloy.properties.get("calphad_solidus_temp"),
            calphad_status=alloy.properties.get("calphad_status"),
        )

        property_prediction = PropertyPredictionState(
            strength=alloy.properties.get("estimated_strength"),
            ductility=alloy.properties.get("estimated_ductility"),
            hardness=alloy.properties.get("estimated_hardness"),
            corrosion_resistance=alloy.properties.get("corrosion_resistance"),
            oxidation_resistance=alloy.properties.get("oxidation_resistance"),
        )

        has_calphad = alloy.properties.get("calphad_status") == "success"
        has_properties = any(
            property_prediction.__dict__[key] is not None
            for key in ("strength", "ductility", "hardness", "corrosion_resistance")
        )
        model_uncertainty = 0.75
        if has_properties:
            model_uncertainty = 0.45
        if has_properties and has_calphad:
            model_uncertainty = 0.25

        previous_scores = list(history.get("previous_scores", []) or [])
        previous_candidates = list(history.get("previous_candidates", []) or [])
        data_density = min(len(previous_candidates) / 10.0, 1.0) if previous_candidates else 0.1

        uncertainty = UncertaintyState(
            model_uncertainty=round(
                float(alloy.properties.get("model_uncertainty", model_uncertainty)),
                4,
            ),
            calphad_confidence=round(
                float(alloy.properties.get("calphad_confidence", 0.9 if has_calphad else 0.2)),
                4,
            ),
            data_density=round(
                float(alloy.properties.get("data_density", data_density)),
                4,
            ),
        )

        history_state = HistoryState(
            previous_candidates=previous_candidates,
            previous_scores=previous_scores,
            failed_regions=list(history.get("failed_regions", []) or []),
            last_mutation=(alloy.metadata.get("mutation", {}) or {}).get("mel"),
        )

        return cls(
            composition=dict(alloy.composition),
            descriptors=descriptors,
            phase_info=phase_info,
            property_prediction=property_prediction,
            uncertainty=uncertainty,
            history=history_state,
            summary=dict(summary),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
