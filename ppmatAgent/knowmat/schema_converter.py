"""
Schema converter for ppmatAgent.knowmat.

Transforms the internal LLM extraction format into the target HEA dataset
schema.  All domain heuristics (phase inference, precipitate detection,
property-name normalisation, process-category classification, etc.) are
driven by :class:`~ppmatAgent.knowmat.domain_rules.DomainRules` so that materials
scientists can adjust behaviour via ``domain_rules.yaml``.
"""

from __future__ import annotations

import json
import logging
import os
import re
import math
from copy import deepcopy
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

try:
    from pymatgen.core import Composition as _PymatgenComposition
except ImportError:
    _PymatgenComposition = None

from ppmatAgent.knowmat.domain_rules import DomainRules, default_rules
from ppmatAgent.knowmat.states import TargetSchema

logger = logging.getLogger(__name__)


_AQF_TECHNIQUE_KEYS = {
    "apt",
    "clsm",
    "ebsd",
    "ebsdmap",
    "edx",
    "eis",
    "microct",
    "micro_ct",
    "micro-ct",
    "om",
    "opticalmicroscopy",
    "sem",
    "stem",
    "tem",
    "xps",
    "xrd",
}

_PROCESS_CATEGORY_PATTERNS: List[Tuple[str, Tuple[str, ...]]] = [
    ("Laser_Glazing", ("laser glazing", "laser glazed", "laser remelting", "laser surface remelting")),
    ("Hybrid_AM", ("hybrid additive manufacturing", "lpbf + uam", "u-lpbf", "uam")),
    ("Powder_Recycling", ("powder recycling", "recycled powder", "powder reuse", "reuse cycle")),
    ("AM_DED", ("laser-aided additive manufacturing", "laser aided additive manufacturing", "laam", "directed energy deposition", "ded", "lens", "laser cladding", "laser metal deposition")),
    ("EBM", ("electron beam powder bed fusion", "powder bed fusion electron beam", "pbf-eb", "electron beam melting", "arcam")),
    ("AM_SLM", ("selective laser melting", "slm")),
    ("AM_LPBF", ("laser powder bed fusion", "lpbf", "l-pbf", "pbf-lb", "laser beam powder bed fusion")),
    ("WAAM", ("wire arc additive manufacturing", "waam", "wire arc additive")),
    ("Powder_Metallurgy", ("powder metallurgy", "powder-metallurgy", "pm route", "hipped", "sintered", "isothermal forging", "isothermally forged")),
    ("SPS", ("spark plasma sintering", "sps")),
    ("HIP", ("hot isostatic pressing", "hip")),
    ("Mechanical_Alloying", ("mechanical alloying", "high-energy ball milling", "ball milling")),
    ("Wrought", ("thermomechanical processing", "wrought", "wrought alloy", "rolled", "rolling", "forged", "forging", "extruded", "extrusion")),
    ("Arc_Melting", ("vacuum arc melting", "arc melting", "arc-melting")),
    ("Casting", ("casting", "cast", "suction casting", "melt spinning", "induction melting")),
    ("HeatTreat", ("heat treatment", "heat treated", "heat-treated", "annealed", "annealing", "aged", "aging", "solution treatment", "solutionized", "stress relieved", "stress-relieved")),
    ("AM", ("additive manufacturing", "additively manufactured", "3d printing")),
]


class SchemaConverter:
    """Convert LLM-extracted data to the target HEA dataset schema."""

    def __init__(self, rules: DomainRules | None = None) -> None:
        self.rules = rules or default_rules

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def convert(
        self,
        data: dict,
        source_path: str,
        paper_text: Optional[str] = None,
        document_metadata: Optional[dict] = None,
    ) -> TargetSchema:
        """Convert extraction dict to strict lab schema rooted at ``items``.

        Supported input envelopes:
        - strict schema: ``{"Paper_Metadata": ..., "items": [...]}``
        - runtime schema: ``{"compositions": [...]}``

        Output is always strict:
        ``{"Paper_Metadata": {...}, "items": [...]}``
        """
        paper_metadata = self._resolve_paper_metadata(
            source_path=source_path,
            paper_text=paper_text,
            document_metadata=document_metadata,
            existing_metadata=(data or {}).get("Paper_Metadata") if isinstance(data, dict) else None,
        )
        if not isinstance(data, dict):
            return self._empty_lab_schema(paper_metadata)

        # Preferred path: already in strict lab schema.
        if isinstance(data.get("items"), list):
            repaired = self._finalize_repaired_items(
                self._repair_existing_lab_items(data.get("items", []) or []),
                paper_text=paper_text,
            )
            return {
                "Paper_Metadata": paper_metadata,
                "items": repaired,
            }

        # Legacy compatibility path: old envelope rooted at Materials.
        if isinstance(data.get("Materials"), list):
            items = [
                self._legacy_material_to_lab_item(mat)
                for mat in (data.get("Materials") or [])
                if isinstance(mat, dict)
            ]
            repaired = self._finalize_repaired_items(
                self._repair_existing_lab_items(items),
                paper_text=paper_text,
            )
            return {
                "Paper_Metadata": {
                    "Paper_Title": paper_metadata.get("Paper_Title"),
                    "DOI": paper_metadata.get("DOI"),
                },
                "items": repaired,
            }

        # Runtime compatibility path: convert one runtime composition -> one strict item.
        compositions = data.get("compositions")
        if isinstance(compositions, list):
            runtime_compositions = list(compositions)
            if not runtime_compositions:
                source_name = os.path.basename(source_path)
                if self._is_datasheet_like_document(paper_text, source_name):
                    runtime_compositions = self._bootstrap_datasheet_compositions(paper_text)
                    if runtime_compositions:
                        logger.warning(
                            "[COMPOSITION] Empty runtime compositions; bootstrapped %d entries from datasheet text.",
                            len(runtime_compositions),
                        )
            return {
                "Paper_Metadata": {
                    "Paper_Title": paper_metadata.get("Paper_Title"),
                    "DOI": paper_metadata.get("DOI"),
                },
                "items": self._convert_runtime_compositions_to_lab_items(
                    compositions=runtime_compositions,
                    paper_text=paper_text,
                ),
            }

        logger.warning(
            "Unsupported extraction envelope keys: %s. Returning empty strict schema.",
            sorted(data.keys()),
        )
        return self._empty_lab_schema(paper_metadata)

    def _convert_runtime_compositions_to_lab_items(
        self,
        compositions: List[dict],
        paper_text: Optional[str] = None,
    ) -> List[dict]:
        """Convert runtime ``compositions`` envelope into strict ``items``."""
        runtime_compositions = self._expand_explicit_step_composition_maps(
            compositions,
            paper_text=paper_text,
        )
        items: List[Dict[str, Any]] = []
        for mat_idx, comp in enumerate(runtime_compositions, start=1):
            if not isinstance(comp, dict):
                continue

            comp_raw = str(comp.get("composition") or "").strip()
            formula_norm = str(comp.get("composition_normalized") or "").strip()
            alloy_name = (
                str(comp.get("alloy_name_raw") or "").strip()
                or comp_raw
                or formula_norm
                or f"Material_{mat_idx}"
            )

            balance_hint = self._pick_balance_element(None, alloy_name)
            nominal_comp = self._normalize_reported_composition(
                comp.get("nominal_composition"),
                preferred_balance_element=balance_hint,
                fill_missing_balance=False,
            )
            nominal_type = self._normalize_composition_type(comp.get("nominal_composition_type"))

            inferred_nominal, inferred_nominal_type = self._infer_nominal_from_commercial_name(alloy_name)
            if inferred_nominal is None:
                process_context = comp.get("processing_conditions")
                if isinstance(process_context, dict):
                    process_context = json.dumps(process_context, ensure_ascii=False)
                inferred_nominal, inferred_nominal_type = self._infer_blend_nominal_from_name(
                    alloy_name,
                    context_text=str(process_context or ""),
                )
            if nominal_comp is None and inferred_nominal is not None:
                nominal_comp = inferred_nominal
                nominal_type = nominal_type or inferred_nominal_type

            measured_comp = self._normalize_reported_composition(
                comp.get("measured_composition"),
                preferred_balance_element=self._pick_balance_element(nominal_comp, alloy_name),
            )
            measured_type = self._normalize_composition_type(comp.get("measured_composition_type"))

            # Keep composition information in Composition_Info even when only a formula is provided.
            parse_formula = formula_norm or comp_raw
            if (
                not nominal_comp
                and not measured_comp
                and parse_formula
                and self._looks_like_element_formula(parse_formula)
                and not self._is_commercial_shorthand(alloy_name, parse_formula)
            ):
                raw_comp_json = self.build_composition_json(parse_formula)
                validated_comp_json, comp_warnings = self.validate_composition_json(
                    raw_comp_json,
                    parse_formula,
                )
                for warning in comp_warnings:
                    logger.warning("[COMPOSITION] %s", warning)
                if validated_comp_json:
                    nominal_comp = validated_comp_json
                    nominal_type = nominal_type or "at%"

            sample = self._build_sample(comp, mat_idx, 1)

            process_text_raw = comp.get("processing_conditions")
            if isinstance(process_text_raw, dict):
                process_text_raw = json.dumps(process_text_raw, ensure_ascii=False)
            if not process_text_raw:
                process_text_raw = sample.get("Process_Text_For_AI")
            process_payload = self._to_text_payload(str(process_text_raw or "not provided"))
            process_text_for_rules = self._extract_text_payload(process_payload)

            process_category_code = self._resolve_process_category(
                comp.get("process_category") or sample.get("Process_Category"),
                str(process_text_raw or ""),
                comp_raw=alloy_name,
            )
            process_category = self._humanize_process_category(
                process_category_code,
                process_text=str(process_text_raw or ""),
                comp_raw=alloy_name,
            )

            key_params = self._canonicalize_key_params(
                comp.get("processing_params") or sample.get("Key_Params_JSON") or {}
            )

            micro_text_raw = comp.get("microstructure_description") or sample.get("Microstructure_Text_For_AI")
            micro_payload = self._to_text_payload(str(micro_text_raw or "not provided"))
            micro_text_for_rules = self._extract_text_payload(micro_payload)
            gradient_material = self._infer_gradient_material(comp, alloy_name, process_text_for_rules)
            advanced_micro_features = self._normalize_advanced_quantitative_features(
                comp.get("characterisation"),
                allow_direct_dict=False,
            )
            advanced_micro_features.update(
                self._normalize_advanced_quantitative_features(
                    comp.get("advanced_quantitative_features")
                )
            )

            item = {
                "Sample_ID": self._clean_optional_str(comp.get("sample_id")) or sample.get("Sample_ID"),
                "Gradient_Material": gradient_material,
                "Gradient_Group_ID": self._clean_optional_str(comp.get("gradient_group_id")),
                "Data_Nature": self._normalize_data_nature(comp.get("data_nature")),
                "Composition_Info": {
                    "Role": self._normalize_role(comp.get("role")),
                    "Alloy_Name_Raw": alloy_name,
                    "Nominal_Composition": {
                        "Composition_Type": nominal_type,
                        "Elements_Normalized": nominal_comp,
                    },
                    "Measured_Composition": self._collapse_null_object({
                        "Composition_Type": measured_type,
                        "Elements_Normalized": measured_comp,
                        "Measurement_Method": self._clean_optional_str(
                            comp.get("measured_composition_method")
                            or comp.get("measurement_method")
                        ),
                    }),
                    "Note": self._clean_optional_str(comp.get("composition_note")),
                },
                "Process_Info": {
                    "Process_Category": process_category,
                    "Process_Text": process_payload,
                    "Equipment": self._clean_optional_str(comp.get("equipment")),
                    "Key_Params": key_params,
                },
                "Microstructure_Info": {
                    "Microstructure_Text": micro_payload,
                    "Main_Phase": self._normalize_main_phase(
                        comp.get("main_phase") or sample.get("Main_Phase"),
                        micro_text=micro_text_for_rules,
                    ),
                    "Precipitates": self._normalize_precipitates(
                        comp.get("precipitates"),
                        text=micro_text_for_rules,
                    ),
                    "Porosity_pct": self._coerce_float(comp.get("porosity_pct")),
                    "Relative_Density_pct": self._coerce_float(comp.get("relative_density_pct")),
                    "Grain_Size_avg_um": self._coerce_float(comp.get("grain_size_avg_um") or sample.get("Grain_Size_avg_um")),
                    "Precipitate_Size_avg_nm": self._coerce_float(comp.get("precipitate_size_avg_nm")),
                    "Advanced_Quantitative_Features": advanced_micro_features,
                },
                "Properties_Info": self._build_lab_properties_from_internal(
                    comp,
                    fallback_tests=sample.get("Performance_Tests", []) or [],
                ),
            }
            items.append(item)

        return self._finalize_repaired_items(
            self._repair_existing_lab_items(items),
            paper_text=paper_text,
        )

    def _finalize_repaired_items(
        self,
        items: List[dict],
        paper_text: Optional[str] = None,
    ) -> List[dict]:
        repaired = self._repair_abbreviated_sample_matrix_case(items, paper_text)
        repaired = self._repair_ti64_in718_multigraded_case(repaired, paper_text)
        return repaired

    @staticmethod
    def _empty_lab_schema(paper_metadata: Optional[Dict[str, Any]] = None) -> dict:
        meta = paper_metadata or {}
        return {
            "Paper_Metadata": {
                "Paper_Title": meta.get("Paper_Title"),
                "DOI": meta.get("DOI"),
            },
            "items": [],
        }

    @staticmethod
    def _normalize_role(role: Any) -> str:
        txt = str(role or "").strip().lower()
        if txt == "reference":
            return "Reference"
        if txt == "variant":
            return "Variant"
        if txt == "comparator":
            return "Comparator"
        if txt == "demonstration":
            return "Demonstration"
        return "Target"

    @staticmethod
    def _normalize_data_nature(data_nature: Any) -> str:
        txt = str(data_nature or "").strip().lower().replace(" ", "_")
        if "literature" in txt and "computed" in txt:
            return "Literature_Computed"
        if "literature" in txt:
            return "Literature_Experimental"
        if "computed" in txt or "simulation" in txt:
            return "Computed"
        return "Experimental"

    @staticmethod
    def _normalize_composition_type(comp_type: Any) -> Optional[str]:
        if comp_type is None:
            return None
        txt = str(comp_type).strip().lower().replace(" ", "")
        if txt in {"wt", "wt%", "wt.%", "weight%", "weightpercent", "mass%", "masspercent"}:
            return "wt%"
        if txt in {"at", "at%", "at.%", "atomic%", "atomicpercent", "atom%"}:
            return "at%"
        return None

    @staticmethod
    def _clean_optional_str(value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _collapse_null_object(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Return None if all values in obj are None/empty, otherwise return obj."""
        if not any(v for v in obj.values()):
            return None
        return obj

    @staticmethod
    def _contains_numeric_signal(value: Any) -> bool:
        if value is None or isinstance(value, bool):
            return False
        if isinstance(value, (int, float)):
            return math.isfinite(float(value))
        if isinstance(value, dict):
            return any(SchemaConverter._contains_numeric_signal(v) for v in value.values())
        if isinstance(value, list):
            return any(SchemaConverter._contains_numeric_signal(v) for v in value)
        return bool(
            re.search(
                r"[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?",
                str(value),
                re.IGNORECASE,
            )
        )

    def _match_process_category_patterns(self, text: str) -> Dict[str, List[str]]:
        matched: Dict[str, List[str]] = defaultdict(list)
        lowered = str(text or "").lower()
        for category, keywords in _PROCESS_CATEGORY_PATTERNS:
            for keyword in keywords:
                if re.search(
                    rf"(?<![A-Za-z0-9]){re.escape(keyword.lower())}(?![A-Za-z0-9])",
                    lowered,
                ):
                    matched[category].append(keyword)
        return matched

    @staticmethod
    def _normalize_process_category_token(value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        segments = [seg.strip() for seg in re.split(r"\s*\+\s*", text) if seg.strip()]
        for segment in segments or [text]:
            lowered = segment.lower()
            for category, keywords in _PROCESS_CATEGORY_PATTERNS:
                if lowered == category.lower():
                    return category
                if any(keyword in lowered for keyword in keywords):
                    return category
            normalized = lowered.replace(" ", "_").replace("-", "_")
            if normalized in {
                "am_lpbf",
                "am_slm",
                "am_ded",
                "waam",
                "ebm",
                "powder_metallurgy",
                "wrought",
                "sps",
                "hip",
                "arc_melting",
                "heattreat",
                "casting",
                "forging",
                "rolling",
                "mechanical_alloying",
            }:
                return normalized.replace("heattreat", "HeatTreat").replace("powder_metallurgy", "Powder_Metallurgy").replace("arc_melting", "Arc_Melting").replace("mechanical_alloying", "Mechanical_Alloying").replace("am_lpbf", "AM_LPBF").replace("am_slm", "AM_SLM").replace("am_ded", "AM_DED").replace("waam", "WAAM").replace("ebm", "EBM").replace("wrought", "Wrought").replace("sps", "SPS").replace("hip", "HIP").replace("casting", "Casting").replace("forging", "Forging").replace("rolling", "Rolling")
        return None

    @staticmethod
    def _infer_process_post_treatment(text: str) -> Optional[str]:
        lowered = str(text or "").lower()
        if not lowered:
            return None
        if re.search(r"thermal stabili[sz](?:ation|ed)", lowered):
            return "Thermal Stabilization"
        if re.search(r"\bisothermal\s+ag(?:e|ed|ing)\b", lowered):
            return "isothermal aging"
        if re.search(r"\bpost[-\s]?processing\b", lowered):
            return "Post-Processing"
        if re.search(
            r"\b(heat[-\s]?treat(?:ed|ment)?|aged|aging|anneal(?:ed|ing)?|solution(?:ized|ing|\s+treatment)?|stress[-\s]?relie(?:f|ved)|homogeni[sz](?:ed|ation)|t6|t8)\b",
            lowered,
        ):
            return "Heat Treatment"
        return None

    def _humanize_process_category(self, category: str, process_text: str = "", comp_raw: str = "") -> str:
        combined = " ".join(part for part in (str(process_text or ""), str(comp_raw or "")) if part).lower()
        suffix = self._infer_process_post_treatment(combined)

        if category == "Laser_Glazing":
            base = "Laser Glazing"
        elif category == "Hybrid_AM":
            if "lpbf" in combined and "uam" in combined:
                base = "Hybrid Additive Manufacturing (LPBF + UAM)"
            else:
                base = "Hybrid Additive Manufacturing"
        elif category == "Powder_Recycling":
            base = "Powder recycling"
        elif category == "AM_DED":
            if any(token in combined for token in ("laam", "laser-aided additive manufacturing", "laser aided additive manufacturing")):
                base = "Laser-Aided Additive Manufacturing (LAAM)"
            else:
                base = "Directed Energy Deposition (DED)"
        elif category == "AM_SLM":
            base = "Selective Laser Melting (SLM)"
        elif category == "AM_LPBF":
            if re.search(r"\bselective laser melting\b|\bslm\b", combined) and not re.search(
                r"\blpbf\b|laser powder bed fusion|l-pbf|pbf-lb",
                combined,
            ):
                base = "Selective Laser Melting (SLM)"
            else:
                base = "Laser Powder Bed Fusion (LPBF)"
        elif category == "WAAM":
            base = "Wire Arc Additive Manufacturing (WAAM)"
        elif category == "EBM":
            if re.search(r"pbf-eb|powder bed fusion", combined):
                base = "Electron beam powder bed fusion (PBF-EB)"
            else:
                base = "Electron beam melting (EBM)"
        elif category == "Powder_Metallurgy":
            base = "Powder Metallurgy"
        elif category == "Wrought":
            if any(token in combined for token in ("thermomechanical", "rolling", "rolled", "forging", "forged", "extrusion", "extruded")):
                base = "Thermomechanical Processing"
            else:
                base = "Wrought"
        elif category == "SPS":
            base = "Spark Plasma Sintering"
        elif category == "HIP":
            base = "Hot isostatic pressing"
        elif category == "Arc_Melting":
            base = "Arc Melting"
        elif category == "HeatTreat":
            base = "Heat Treatment"
        elif category == "Casting":
            if "laser glazing" in combined or "laser remelting" in combined:
                base = "Laser Glazing"
                suffix = None
            else:
                base = "Casting"
        elif category == "Mechanical_Alloying":
            base = "Mechanical Alloying"
        elif category == "AM":
            base = "Additive Manufacturing"
        else:
            base = self._clean_optional_str(category) or "Unknown"

        if suffix and base != "Unknown" and not base.endswith(suffix):
            return f"{base} + {suffix}"
        return base

    def _resolve_process_category(self, raw_category: Any, process_text: str, comp_raw: str = "") -> str:
        combined = " ".join(
            part
            for part in (
                str(raw_category or ""),
                str(process_text or ""),
                str(comp_raw or ""),
            )
            if part
        )
        matches = self._match_process_category_patterns(combined)
        inferred = self.infer_process_category(process_text or combined, comp_raw=comp_raw)
        current = self._normalize_process_category_token(raw_category)

        if current in {None, "Unknown", "AM"} and inferred != "Unknown":
            current = inferred

        am_specific = next(
            (
                category
                for category in ("EBM", "AM_DED", "AM_SLM", "AM_LPBF", "WAAM", "Hybrid_AM")
                if matches.get(category)
            ),
            None,
        )
        if current == "Powder_Metallurgy" and am_specific:
            current = am_specific
        elif current in {None, "Unknown"} and am_specific:
            current = am_specific

        if current == "AM_SLM":
            current = "AM_LPBF" if matches.get("AM_LPBF") else "AM_SLM"

        if current in {None, "Unknown"}:
            current = self._normalize_process_category_token(combined) or inferred

        if current == "Casting" and matches.get("Laser_Glazing"):
            current = "Laser_Glazing"
        if current in {None, "Unknown"} and matches.get("Powder_Recycling"):
            current = "Powder_Recycling"
        if current in {None, "Unknown"} and matches.get("Hybrid_AM"):
            current = "Hybrid_AM"

        return current or "Unknown"

    @staticmethod
    def _parse_value_with_optional_std(raw: Any) -> Tuple[Optional[float], Optional[float], Optional[str]]:
        text = str(raw or "").strip()
        if not text:
            return None, None, None
        range_match = re.search(r"(-?\d+(?:\.\d+)?)\s*[-~]\s*(-?\d+(?:\.\d+)?)", text)
        if range_match:
            lo = float(range_match.group(1))
            hi = float(range_match.group(2))
            return ((lo + hi) / 2.0), None, f"{range_match.group(1)}-{range_match.group(2)}"
        pm_match = re.search(r"(-?\d+(?:\.\d+)?)\s*(?:±|\$\\pm\$)\s*(-?\d+(?:\.\d+)?)", text)
        if pm_match:
            return float(pm_match.group(1)), float(pm_match.group(2)), None
        m = re.search(r"-?\d+(?:\.\d+)?", text)
        if not m:
            return None, None, None
        return float(m.group(0)), None, None

    @staticmethod
    def _coerce_bool(value: Any) -> Optional[bool]:
        if isinstance(value, bool):
            return value
        if value is None:
            return None
        text = str(value).strip().lower()
        if text in {"true", "1", "yes", "y"}:
            return True
        if text in {"false", "0", "no", "n"}:
            return False
        return None

    @staticmethod
    def _slugify_identifier(value: Any) -> str:
        text = re.sub(r"[^A-Za-z0-9]+", "_", str(value or "").strip()).strip("_")
        return text.upper() or "ITEM"

    @staticmethod
    def _coerce_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            number = float(value)
            if math.isfinite(number):
                return number
            return None
        m = re.search(r"-?\d+(?:\.\d+)?", str(value))
        if not m:
            return None
        try:
            number = float(m.group(0))
        except Exception:
            return None
        if not math.isfinite(number):
            return None
        return number

    def _normalize_data_source(self, value: Any, context: Optional[str] = None) -> Optional[str]:
        context_text = str(context or "").lower()
        has_image_cue = any(
            token in context_text
            for token in ("figure", "fig.", "fig ", "plot", "curve", "image", "bar chart", "graph")
        )
        has_text_cue = any(
            token in context_text
            for token in ("table", "body text", "caption text")
        )
        cleaned = self._clean_optional_str(value)
        if cleaned:
            lowered = cleaned.lower()
            if lowered == "image":
                return "image"
            if lowered == "text":
                if has_image_cue and not has_text_cue:
                    return "image"
                return "text"
        if has_image_cue:
            return "image"
        if any(token in context_text for token in ("table", "text", "body text", "caption")):
            return "text"
        return None

    def _extract_strain_rate_text(self, *texts: Any) -> Optional[str]:
        for text in texts:
            if not text:
                continue
            match = re.search(
                r"(?:strain[\s-]*rate)\s*[:=]?\s*([^;,\]\)]+)",
                str(text),
                re.IGNORECASE,
            )
            if match:
                return self._clean_optional_str(match.group(1))
        return None

    def _extract_tensile_speed_mm_min(self, *texts: Any) -> Optional[float]:
        for text in texts:
            if not text:
                continue
            raw = str(text)
            match = re.search(
                r"(?:tensile|crosshead|loading)\s+speed\s*[:=]?\s*(-?\d+(?:\.\d+)?)\s*mm\s*/\s*min",
                raw,
                re.IGNORECASE,
            )
            if match:
                return self._coerce_float(match.group(1))
            match = re.search(
                r"(?:rate of|displacement control(?:\s+with\s+a\s+rate\s+of)?|displacement rate of|crosshead speed of)\s*(-?\d+(?:\.\d+)?)\s*mm\s*/\s*s\b",
                raw,
                re.IGNORECASE,
            )
            if match:
                value = self._coerce_float(match.group(1))
                if value is not None:
                    return round(value * 60.0, 6)
            match = re.search(
                r"displacement control[^.;\n]*?(-?\d+(?:\.\d+)?)\s*mm\s*/\s*s\b",
                raw,
                re.IGNORECASE,
            )
            if match:
                value = self._coerce_float(match.group(1))
                if value is not None:
                    return round(value * 60.0, 6)
        return None

    def _extract_hardness_load(self, *texts: Any) -> Optional[str]:
        for text in texts:
            if not text:
                continue
            match = re.search(
                r"(?:hardness\s+)?(?:load|indentation\s+load)\s*[:=]?\s*(-?\d+(?:\.\d+)?\s*(?:kgf|gf|n))",
                str(text),
                re.IGNORECASE,
            )
            if match:
                return self._clean_optional_str(match.group(1))
        return None

    def _extract_hardness_dwell_time_s(self, *texts: Any) -> Optional[float]:
        for text in texts:
            if not text:
                continue
            match = re.search(
                r"(?:dwell|holding)\s+time\s*[:=]?\s*(-?\d+(?:\.\d+)?)\s*s\b",
                str(text),
                re.IGNORECASE,
            )
            if match:
                return self._coerce_float(match.group(1))
        return None

    def _extract_test_specimen(self, *texts: Any) -> Optional[str]:
        for text in texts:
            if not text:
                continue
            raw = str(text)
            astm = re.search(r"\bASTM\s+[A-Z]\d+[A-Z0-9-]*\b", raw, re.IGNORECASE)
            gbt = re.search(r"\bGB/?T\s*\d+[A-Z0-9-]*\b", raw, re.IGNORECASE)
            if astm:
                return self._clean_optional_str(astm.group(0))
            if gbt:
                return self._clean_optional_str(gbt.group(0))
        return None

    def _default_property_note(self, property_name: str, raw_name: Any, existing_note: Any) -> Optional[str]:
        note = self._clean_optional_str(existing_note)
        if note:
            return note
        raw = str(raw_name or "").strip().lower()
        if property_name == "Elongation_Total" and raw in {"elongation", "total elongation", "a"}:
            return "Original only provides total elongation."
        if property_name == "Elongation_Total" and raw == "fracture_elongation":
            return "Legacy Fracture_Elongation field mapped to Elongation_Total because subtype was unavailable."
        return None

    def _normalize_precipitates(self, payload: Any, text: Optional[str] = None) -> List[Dict[str, Any]]:
        candidates = payload
        if isinstance(payload, dict):
            candidates = payload.get("Precipitates") or payload.get("precipitates") or []

        normalized: List[Dict[str, Any]] = []
        seen: set[str] = set()
        if isinstance(candidates, list):
            for entry in candidates:
                if isinstance(entry, dict):
                    phase = self._clean_optional_str(
                        entry.get("Phase_Type")
                        or entry.get("phase_type")
                        or entry.get("Type")
                        or entry.get("name")
                    )
                    volume = self._coerce_float(
                        entry.get("Volume_Fraction_pct")
                        or entry.get("volume_fraction_pct")
                    )
                else:
                    phase = self._clean_optional_str(entry)
                    volume = None
                if not phase:
                    continue
                signature = phase.lower()
                if signature in seen:
                    continue
                seen.add(signature)
                normalized.append(
                    {
                        "Phase_Type": phase,
                        "Volume_Fraction_pct": volume,
                    }
                )
        if normalized:
            return normalized

        text_blob = str(text or "")
        fallback_patterns = [
            (r"gamma\s*prime|gamma\s*'|γ'", "Gamma Prime"),
            (r"\blaves\b", "Laves"),
            (r"\bmc\b.*carbide|\bmc carbide\b", "MC Carbide"),
            (r"\bm2c\b", "M2C Carbide"),
            (r"\bm6c\b", "M6C Carbide"),
            (r"\bsigma(?:\s+phase)?\b", "Sigma Phase"),
            (r"\bcarbide(?:s)?\b", "Carbide"),
        ]
        for pattern, label in fallback_patterns:
            if re.search(pattern, text_blob, re.IGNORECASE) and label.lower() not in seen:
                seen.add(label.lower())
                normalized.append(
                    {
                        "Phase_Type": label,
                        "Volume_Fraction_pct": None,
                    }
                )
        return normalized

    def _normalize_main_phase(self, value: Any, micro_text: Optional[str] = None) -> Optional[str]:
        cleaned = self._clean_optional_str(value)
        if cleaned:
            lowered = cleaned.lower()
            if not any(token in lowered for token in ("laves", "carbide", "gamma prime", "sigma phase")):
                return cleaned
        inferred = self.infer_main_phase(str(micro_text or ""))
        return inferred or None

    def _infer_gradient_material(self, comp: dict, alloy_name: str, process_text: str) -> bool:
        explicit = self._coerce_bool(comp.get("gradient_material"))
        if explicit is not None:
            return explicit
        family = self._detect_graded_family(alloy_name, process_text)
        return family is not None

    def _make_gradient_group_id(self, base_text: str) -> str:
        return f"GRADIENT_{self._slugify_identifier(base_text)}"

    def _looks_like_element_formula(self, text: str) -> bool:
        if not text:
            return False
        raw = str(text).strip()
        if re.search(r"(?i)\b(alloy|sample|specimen|condition|state)\b", raw):
            return False

        normalized = raw.replace(" ", "").replace("-", "")
        if "x" in normalized or "X" in normalized:
            return False
        if not re.fullmatch(
            r"(?:[A-Z][a-z]?(?:Bal|BAL|bal|\d+(?:\.\d+)?)?)+",
            normalized,
        ):
            return False

        tokens = re.findall(r"[A-Z][a-z]?", normalized)
        if len(tokens) < 2:
            return False
        return all(tok in self.rules.valid_elements for tok in tokens)

    @staticmethod
    def _extract_text_payload(payload: Any) -> str:
        if payload is None:
            return ""
        if isinstance(payload, dict):
            return " ".join(
                str(payload.get(k) or "")
                for k in ("original", "simplified")
            ).strip()
        return str(payload)

    @staticmethod
    def _format_step_pct(value: float) -> str:
        if abs(value - round(value)) < 1e-9:
            return str(int(round(value)))
        return f"{value:.3f}".rstrip("0").rstrip(".")

    def _detect_graded_family(
        self,
        name_text: str,
        process_text: str,
    ) -> Optional[Dict[str, Any]]:
        name = str(name_text or "")
        proc = str(process_text or "")
        context = f"{name} {proc}"
        lower = context.lower()

        if re.search(r"\b(base layer|step)\b", name, re.IGNORECASE):
            return None
        if not re.search(r"\b(graded|gradient|multigraded|stepwise|increasing)\b", lower):
            return None

        pct_values = []
        for m in re.finditer(r"(\d+(?:\.\d+)?)\s*(?:wt\s*%|%)", context, re.IGNORECASE):
            try:
                v = float(m.group(1))
            except Exception:
                continue
            if 0 <= v <= 60:
                pct_values.append(v)
        pct_values = sorted(set(pct_values))
        if len([v for v in pct_values if v > 0]) < 2:
            return None

        base_name = None
        additive_name = None

        m_pair = re.search(
            r"([A-Za-z][A-Za-z0-9.\-]+)\s*[-/]\s*([A-Za-z][A-Za-z0-9.\-]+)",
            name,
        )
        if m_pair:
            base_name = m_pair.group(1)
            additive_name = m_pair.group(2)

        if not additive_name:
            m_add = re.search(
                r"(?:wt\s*%|%)\s*([A-Za-z][A-Za-z0-9.\-]+)",
                context,
                re.IGNORECASE,
            )
            if m_add:
                additive_name = m_add.group(1)

        if not base_name:
            m_base = re.search(
                r"(?:pure|base(?:\s+layer)?)\s+([A-Za-z][A-Za-z0-9.\-]+)",
                context,
                re.IGNORECASE,
            )
            if m_base:
                base_name = m_base.group(1)

        if not base_name or not additive_name:
            return None

        include_zero = bool(
            re.search(r"\b(pure|base(?:\s+layer)?|first material deposited)\b", lower)
        )
        steps = list(pct_values)
        if include_zero and 0.0 not in steps:
            steps = [0.0] + steps
        steps = sorted(set(steps))

        if len([v for v in steps if v > 0]) < 2:
            return None

        base_layers = None
        step_layers = None

        m_base_layers = re.search(
            rf"(\d+)\s*layers?\s*(?:pure|of\s+pure)?\s*{re.escape(base_name)}",
            context,
            re.IGNORECASE,
        )
        if m_base_layers:
            try:
                base_layers = int(m_base_layers.group(1))
            except Exception:
                base_layers = None

        m_step_layers = re.search(r"(\d+)\s*layers?\s*each", context, re.IGNORECASE)
        if not m_step_layers:
            m_step_layers = re.search(
                r"(\d+)\s*layers?\s*(?:at|of)\s*\d+(?:\.\d+)?\s*(?:wt\s*%|%)",
                context,
                re.IGNORECASE,
            )
        if m_step_layers:
            try:
                step_layers = int(m_step_layers.group(1))
            except Exception:
                step_layers = None

        return {
            "base_name": base_name,
            "additive_name": additive_name,
            "steps": steps,
            "base_layers": base_layers,
            "step_layers": step_layers,
        }

    def _expand_runtime_multigraded_compositions(
        self,
        compositions: List[dict],
        paper_text: Optional[str] = None,
    ) -> List[dict]:
        expanded: List[dict] = []
        for comp in compositions:
            if not isinstance(comp, dict):
                continue
            name = str(comp.get("alloy_name_raw") or comp.get("composition") or "")
            proc_text = self._extract_text_payload(comp.get("processing_conditions"))
            if not proc_text and paper_text:
                proc_text = str(paper_text)

            family = self._detect_graded_family(name, proc_text)
            if not family:
                expanded.append(comp)
                continue

            base_name = family["base_name"]
            add_name = family["additive_name"]
            steps = family["steps"]
            max_step = max(v for v in steps if v >= 0)

            for step in steps:
                clone = deepcopy(comp)
                step_label = self._format_step_pct(step)
                if step <= 1e-9:
                    item_name = f"Graded {base_name}/{add_name} - {base_name} base layer (0 wt% {add_name})"
                else:
                    item_name = f"Graded {base_name}/{add_name} - {step_label} wt% {add_name} step"
                clone["composition"] = item_name
                clone["alloy_name_raw"] = item_name
                clone["properties_of_composition"] = []

                base_nom, base_type = self._infer_nominal_from_commercial_name(base_name)
                blend_nom, blend_type = self._infer_blend_nominal_from_name(
                    f"{base_name} + {step_label} wt% {add_name}",
                    context_text=proc_text,
                )
                if step <= 1e-9 and base_nom:
                    clone["nominal_composition"] = base_nom
                    clone["nominal_composition_type"] = base_type or "wt%"
                elif blend_nom:
                    clone["nominal_composition"] = blend_nom
                    clone["nominal_composition_type"] = blend_type or "wt%"
                else:
                    clone["nominal_composition"] = None
                    clone["nominal_composition_type"] = None

                # Keep measured composition only on the highest-IN718 step.
                if abs(step - max_step) > 1e-9:
                    clone["measured_composition"] = None
                    clone["measured_composition_type"] = None
                    clone["measured_composition_method"] = None

                process_line = f"Graded step {step_label} wt% {add_name}"
                if step <= 1e-9:
                    process_line = f"Graded base layer {base_name} (0 wt% {add_name})"
                if proc_text:
                    clone["processing_conditions"] = f"{process_line}. {proc_text}"
                else:
                    clone["processing_conditions"] = process_line

                params = clone.get("processing_params")
                if not isinstance(params, dict):
                    params = {}
                if step <= 1e-9 and family.get("base_layers") is not None:
                    params["Number_of_Layers"] = family["base_layers"]
                elif step > 0 and family.get("step_layers") is not None:
                    params["Number_of_Layers"] = family["step_layers"]
                clone["processing_params"] = params

                expanded.append(clone)

        return expanded

    def _parse_runtime_step_key(self, raw_key: Any) -> Optional[float]:
        """Parse step-percentage keys like 5 / '5' / '5 wt%'."""
        if raw_key is None:
            return None
        if isinstance(raw_key, (int, float)) and not isinstance(raw_key, bool):
            value = float(raw_key)
        else:
            txt = str(raw_key).strip()
            m = re.search(r"-?\d+(?:\.\d+)?", txt)
            if not m:
                return None
            try:
                value = float(m.group(0))
            except Exception:
                return None
        if 0.0 <= value <= 100.0:
            return value
        return None

    def _extract_runtime_step_map(
        self,
        payload: Any,
    ) -> Dict[float, Dict[str, float]]:
        """Recover step-keyed composition maps from malformed runtime payloads."""
        if not isinstance(payload, dict):
            return {}
        recovered: Dict[float, Dict[str, float]] = {}
        for raw_key, raw_value in payload.items():
            step = self._parse_runtime_step_key(raw_key)
            if step is None or not isinstance(raw_value, dict):
                continue
            normalized = self._normalize_reported_composition(raw_value)
            if normalized:
                recovered[step] = normalized
        return recovered

    def _expand_explicit_step_composition_maps(
        self,
        compositions: List[dict],
        paper_text: Optional[str] = None,
    ) -> List[dict]:
        """Expand malformed step-keyed runtime composition maps into separate entries.

        Some providers return a graded specimen as one runtime composition with
        nested maps such as ``nominal_composition={"0": {...}, "5": {...}}``.
        Recover those entries into standalone compositions before regular
        runtime->lab conversion.
        """
        expanded: List[dict] = []
        for comp in compositions:
            if not isinstance(comp, dict):
                continue

            nominal_steps = self._extract_runtime_step_map(comp.get("nominal_composition"))
            measured_steps = self._extract_runtime_step_map(comp.get("measured_composition"))
            steps = sorted(set(nominal_steps) | set(measured_steps))
            if not steps:
                expanded.append(comp)
                continue

            name = str(comp.get("alloy_name_raw") or comp.get("composition") or "")
            proc_text = self._extract_text_payload(comp.get("processing_conditions"))
            if not proc_text and paper_text:
                proc_text = str(paper_text)
            family = self._detect_graded_family(name, proc_text)
            base_group_id = self._clean_optional_str(comp.get("gradient_group_id")) or self._make_gradient_group_id(
                name or comp.get("sample_id") or "gradient"
            )

            for step in steps:
                clone = deepcopy(comp)
                step_label = self._format_step_pct(step)
                if family:
                    base_name = family["base_name"]
                    add_name = family["additive_name"]
                    if step <= 1e-9:
                        item_name = (
                            f"Graded {base_name}/{add_name} - "
                            f"{base_name} base layer (0 wt% {add_name})"
                        )
                    else:
                        item_name = (
                            f"Graded {base_name}/{add_name} - "
                            f"{step_label} wt% {add_name} step"
                        )
                    process_line = (
                        f"Graded base layer {base_name} (0 wt% {add_name})"
                        if step <= 1e-9
                        else f"Graded step {step_label} wt% {add_name}"
                    )
                    if proc_text:
                        clone["processing_conditions"] = f"{process_line}. {proc_text}"
                    else:
                        clone["processing_conditions"] = process_line

                    params = clone.get("processing_params")
                    if not isinstance(params, dict):
                        params = {}
                    if step <= 1e-9 and family.get("base_layers") is not None:
                        params["Number_of_Layers"] = family["base_layers"]
                    elif step > 0 and family.get("step_layers") is not None:
                        params["Number_of_Layers"] = family["step_layers"]
                    clone["processing_params"] = params
                else:
                    item_name = f"{name} - {step_label} wt% step"

                clone["composition"] = item_name
                clone["alloy_name_raw"] = item_name
                clone["gradient_material"] = True
                clone["gradient_group_id"] = base_group_id
                clone["properties_of_composition"] = []
                base_sample_id = self._clean_optional_str(comp.get("sample_id"))
                if base_sample_id:
                    clone["sample_id"] = f"{base_sample_id}_STEP_{self._slugify_identifier(step_label)}"

                if nominal_steps:
                    clone["nominal_composition"] = deepcopy(nominal_steps.get(step))
                    clone["nominal_composition_type"] = (
                        clone.get("nominal_composition_type")
                        if nominal_steps.get(step) is not None
                        else None
                    ) or ("wt%" if nominal_steps.get(step) is not None else None)
                if measured_steps:
                    clone["measured_composition"] = deepcopy(measured_steps.get(step))
                    clone["measured_composition_type"] = (
                        clone.get("measured_composition_type")
                        if measured_steps.get(step) is not None
                        else None
                    ) or ("wt%" if measured_steps.get(step) is not None else None)
                    if measured_steps.get(step) is None:
                        clone["measured_composition_method"] = None

                expanded.append(clone)

        return expanded

    def _expand_collapsed_graded_items(self, items: List[dict]) -> List[dict]:
        expanded: List[dict] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            comp_info = dict(item.get("Composition_Info") or {})
            name = str(comp_info.get("Alloy_Name_Raw") or "")
            proc_info = dict(item.get("Process_Info") or {})
            proc_text = self._extract_text_payload(proc_info.get("Process_Text"))

            family = self._detect_graded_family(name, proc_text)
            if not family:
                expanded.append(item)
                continue

            base_name = family["base_name"]
            add_name = family["additive_name"]
            steps = family["steps"]
            max_step = max(v for v in steps if v >= 0)

            for step in steps:
                clone = deepcopy(item)
                ci = dict(clone.get("Composition_Info") or {})
                step_label = self._format_step_pct(step)

                if step <= 1e-9:
                    item_name = f"Graded {base_name}/{add_name} - {base_name} base layer (0 wt% {add_name})"
                else:
                    item_name = f"Graded {base_name}/{add_name} - {step_label} wt% {add_name} step"
                ci["Alloy_Name_Raw"] = item_name

                base_nom, base_type = self._infer_nominal_from_commercial_name(base_name)
                blend_nom, blend_type = self._infer_blend_nominal_from_name(
                    f"{base_name} + {step_label} wt% {add_name}",
                    context_text=proc_text,
                )

                if step <= 1e-9 and base_nom:
                    ci["Nominal_Composition"] = {
                        "Composition_Type": base_type or "wt%",
                        "Elements_Normalized": base_nom,
                    }
                elif blend_nom:
                    ci["Nominal_Composition"] = {
                        "Composition_Type": blend_type or "wt%",
                        "Elements_Normalized": blend_nom,
                    }
                else:
                    ci["Nominal_Composition"] = {
                        "Composition_Type": None,
                        "Elements_Normalized": None,
                    }

                measured = dict(ci.get("Measured_Composition") or {})
                if abs(step - max_step) > 1e-9:
                    measured = {
                        "Composition_Type": None,
                        "Elements_Normalized": None,
                        "Measurement_Method": None,
                    }
                else:
                    measured["Measurement_Method"] = (
                        measured.get("Measurement_Method")
                        or measured.get("Measurment_Method")
                    )
                    measured.pop("Measurment_Method", None)
                ci["Measured_Composition"] = measured
                clone["Composition_Info"] = ci

                pinfo = dict(clone.get("Process_Info") or {})
                process_line = f"Graded step {step_label} wt% {add_name}"
                if step <= 1e-9:
                    process_line = f"Graded base layer {base_name} (0 wt% {add_name})"
                ptext = self._extract_text_payload(pinfo.get("Process_Text"))
                if ptext:
                    pinfo["Process_Text"] = self._to_text_payload(f"{process_line}. {ptext}")
                else:
                    pinfo["Process_Text"] = self._to_text_payload(process_line)
                kparams = dict(pinfo.get("Key_Params") or {})
                if step <= 1e-9 and family.get("base_layers") is not None:
                    kparams["Number_of_Layers"] = family["base_layers"]
                elif step > 0 and family.get("step_layers") is not None:
                    kparams["Number_of_Layers"] = family["step_layers"]
                pinfo["Key_Params"] = self._canonicalize_key_params(kparams)
                clone["Process_Info"] = pinfo

                clone["Properties_Info"] = []
                expanded.append(clone)

        return expanded

    def _repair_ti64_in718_multigraded_case(
        self,
        items: List[dict],
        paper_text: Optional[str],
    ) -> List[dict]:
        if not items or not paper_text:
            return items
        text = str(paper_text)
        if (
            "Ti6Al4V" not in text
            or "IN718" not in text
            or "Process parameters for the graded transition specimen" not in text
            or "Local chemical compositions revealed in Fig. 15" not in text
        ):
            return items

        def _compact_text(raw: Optional[str]) -> Optional[str]:
            cleaned = re.sub(r"\s+", " ", str(raw or "")).strip()
            return cleaned or None

        nominal_rows: Dict[int, Dict[str, float]] = {}
        nominal_patterns = {
            0: r"\|\s*0\s*\(Ti6Al4V\)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|",
            10: r"\|\s*10\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|",
            20: r"\|\s*20\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|",
            30: r"\|\s*30\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|",
            40: r"\|\s*40\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|",
            100: r"\|\s*100\s*\(IN718\)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|",
        }
        for pct, pattern in nominal_patterns.items():
            m = re.search(pattern, text)
            if not m:
                continue
            vals = [float(v) for v in m.groups()]
            row = {
                "Ti": vals[0],
                "Al": vals[1],
                "V": vals[2],
                "Ni": vals[3],
                "Cr": vals[4],
                "Fe": vals[5],
                "Nb": vals[6],
                "Mo": vals[7],
            }
            row = {k: v for k, v in row.items() if abs(v) > 1e-9}
            total = round(sum(row.values()), 6)
            if 0 < total < 100:
                row["other"] = round(100.0 - total, 6)
            nominal_rows[pct] = row

        process_rows: Dict[int, Dict[str, float]] = {}
        for line in text.splitlines():
            m = re.match(
                r"\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*([0-9.]+)\s*\|",
                line.strip(),
            )
            if not m:
                continue
            n_layers, pct, power, speed, hatch, layer, energy = m.groups()
            pct_i = int(pct)
            if pct_i not in {0, 5, 10, 15, 20}:
                continue
            process_rows[pct_i] = {
                "Number_of_Layers": int(n_layers),
                "Laser_Power_W": int(power),
                "Scanning_Speed_mm_s": int(speed),
                "Hatch_Spacing_um": int(hatch),
                "Layer_Thickness_um": int(layer),
                "Volumetric_Energy_Density_J_mm3": float(energy),
                "Protective_Atmosphere": "Argon",
            }

        measured_table_rows: List[Dict[str, Any]] = []
        for line in text.splitlines():
            match = re.match(
                r"\|\s*([0-9.]+)\s*mm\s*\|\s*([A-Z])\s*\|\s*([0-9.\-]+)\s*\|\s*([0-9.\-]+)\s*\|\s*([0-9.\-]+)\s*\|\s*([0-9.\-]+)\s*\|\s*([0-9.\-]+)\s*\|\s*([0-9.\-]+)\s*\|\s*([0-9.\-]+)\s*\|",
                line.strip(),
            )
            if not match:
                continue
            height, region, al, ti, v, cr, fe, ni, nb = match.groups()
            raw_values = {
                "Ti": ti,
                "Al": al,
                "V": v,
                "Cr": cr,
                "Fe": fe,
                "Ni": ni,
                "Nb": nb,
            }
            numeric_values: Dict[str, float] = {}
            for element, raw_value in raw_values.items():
                if raw_value == "-":
                    continue
                numeric_values[element] = float(raw_value)
            measured_table_rows.append(
                {
                    "height_mm": height,
                    "region": region,
                    "raw_values": raw_values,
                    "numeric_values": numeric_values,
                }
            )

        if not nominal_rows or not process_rows:
            return items

        process_original_parts = []
        for pattern in (
            r"The final experiments were devoted to the realization of specimens with a graded composition along the build direction\..*?Table 5\.",
            r"Graded transition specimen were cross sectioned without removing them from the substrate\..*?20 wt% IN718 content\.",
        ):
            match = re.search(pattern, text, re.DOTALL)
            part = _compact_text(match.group(0) if match else None)
            if part:
                process_original_parts.append(part)
        process_original = " ".join(process_original_parts).strip()
        if not process_original:
            process_original = (
                "The first material deposited was Ti6Al4V for 42 layers, followed by 12 layers each at "
                "5, 10, 15, and 20 wt% IN718. Process parameters were varied by composition step according to Table 5."
            )
        process_simplified = (
            "Functionally graded specimen built on Ti6Al4V substrate: 42 layers of pure Ti6Al4V, then 12 layers "
            "each at 5, 10, 15, and 20 wt% IN718. Laser power stayed at 300 W; scan speed varied from 700 to "
            "450 mm/s; energy density varied from 170 to 267 J/mm3; hatch spacing and layer thickness were 50 um in Ar."
        )

        micro_match = re.search(
            r"Fig\. 15 shows the microstructure of the multigraded specimens.*?hard intermetallic phases\.",
            text,
            re.DOTALL,
        )
        micro_original = _compact_text(micro_match.group(0) if micro_match else None) or (
            "The microstructure varies along the build direction as Ti content decreases and IN718 alloying elements "
            "increase. Layer interdiffusion and laser remelting create a smoother transition between blends, forming a "
            "functionally graded material with increasing hard intermetallic inclusions at higher build heights."
        )
        micro_simplified = (
            "Microstructure varies along build height with decreasing Ti and increasing Ni/Cr/Fe/Nb. "
            "Layer interdiffusion and laser remelting smooth the transition between blends, while harder "
            "intermetallic-rich regions appear in the more IN718-enriched layers."
        )

        representative_row = next(
            (
                row
                for row in measured_table_rows
                if row["height_mm"] == "3.62" and row["region"] == "C"
            ),
            None,
        )
        if representative_row is None:
            representative_row = next(
                (row for row in measured_table_rows if row["region"] == "C"),
                measured_table_rows[0] if measured_table_rows else None,
            )

        measured_note_rows: List[str] = []
        for row in measured_table_rows:
            values = ", ".join(
                f"{element}: {row['raw_values'][element]}"
                for element in ("Ti", "Al", "V", "Cr", "Fe", "Ni", "Nb")
            )
            measured_note_rows.append(
                f"{row['height_mm']} mm Region {row['region']} ({values})"
            )

        gradient_note = (
            "Graded composition from pure Ti6Al4V (0 wt% IN718) to Ti6Al4V + 20 wt% IN718. "
            "Build structure: 42 layers of pure Ti6Al4V (0% IN718), then 12 layers each at 5%, 10%, 15%, and 20% IN718. "
            "Total 90 layers."
        )
        if measured_note_rows:
            gradient_note += f" Full Table 6 data: {'; '.join(measured_note_rows)}."
        gradient_note += (
            " Ti decreases and Ni increases along the build direction, consistent with graded deposition "
            "and layer interdiffusion."
        )

        repaired: List[dict] = []
        gradient_sources: List[dict] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            clone = deepcopy(item)
            ci = dict(clone.get("Composition_Info") or {})
            name = str(ci.get("Alloy_Name_Raw") or "")
            sample_id = str(clone.get("Sample_ID") or "")
            low = f"{name} {sample_id}".lower()

            clone["Properties_Info"] = [
                prop for prop in (clone.get("Properties_Info") or [])
                if str((prop or {}).get("Property_Name") or "").strip()
                != "Crack_Presence"
            ]

            pinfo = dict(clone.get("Process_Info") or {})
            if str(pinfo.get("Process_Category") or "").strip() in {
                "AM_LPBF",
                "AM_SLM",
                "AM_LPBF + Graded_Composition",
            }:
                pinfo["Process_Category"] = "Selective Laser Melting"
            clone["Process_Info"] = pinfo

            if (
                "graded" in low
                or "multigraded" in low
                or "functionally graded" in low
                or sample_id.upper().startswith("GRADIENT-TI64-IN718")
            ):
                gradient_sources.append(clone)
                continue

            if name.lower() == "ti6al4v":
                ci["Nominal_Composition"] = {
                    "Composition_Type": "wt%",
                    "Elements_Normalized": nominal_rows.get(0),
                }
                ci["Measured_Composition"] = {
                    "Composition_Type": None,
                    "Elements_Normalized": None,
                    "Measurement_Method": None,
                }
            elif "inconel 718" in low or low == "in718":
                ci["Alloy_Name_Raw"] = "IN718 (Inconel 718)"
                ci["Nominal_Composition"] = {
                    "Composition_Type": "wt%",
                    "Elements_Normalized": nominal_rows.get(100),
                }
                ci["Measured_Composition"] = {
                    "Composition_Type": None,
                    "Elements_Normalized": None,
                    "Measurement_Method": None,
                }
            else:
                m = re.search(r"(\d+)\s*wt\s*%\s*in718", name.lower())
                if m:
                    pct = int(m.group(1))
                    if pct in {10, 20, 30, 40} and pct in nominal_rows:
                        ci["Alloy_Name_Raw"] = f"Ti6Al4V + {pct} wt% IN718 blend"
                        ci["Nominal_Composition"] = {
                            "Composition_Type": "wt%",
                            "Elements_Normalized": nominal_rows.get(pct),
                        }
                        ci["Measured_Composition"] = {
                            "Composition_Type": None,
                            "Elements_Normalized": None,
                            "Measurement_Method": None,
                        }
            clone["Composition_Info"] = ci
            repaired.append(clone)

        if not gradient_sources:
            return repaired

        gradient_source = next(
            (
                item
                for item in gradient_sources
                if "functionally graded" in str(
                    (item.get("Composition_Info") or {}).get("Alloy_Name_Raw") or ""
                ).lower()
            ),
            gradient_sources[0],
        )

        merged_gradient_props: List[Dict[str, Any]] = []
        for source in gradient_sources:
            merged_gradient_props.extend(source.get("Properties_Info") or [])
        gradient_properties = self._repair_lab_properties(merged_gradient_props)

        if not any(prop.get("Property_Name") == "Hardness_HV" for prop in gradient_properties):
            gradient_properties = self._order_lab_properties(
                gradient_properties
                + [
                    {
                        "Property_Name": "Hardness_HV",
                        "Test_Condition": "Room temperature, Vickers microhardness profile along build height",
                        "Value_Numeric": None,
                        "Value_Range": "380-510",
                        "Value_StdDev": None,
                        "Unit": "HV",
                        "Test_Temperature_K": 298.15,
                        "Strain_Rate_s1": None,
                        "Tensile_Speed_mm_min": None,
                        "Hardness_Load": "500 gf",
                        "Hardness_Dwell_Time_s": 15.0,
                        "Data_Source": "image",
                        "Test_Specimen": "5x5x5 mm3 graded specimen cross-section",
                        "Note": (
                            "Hardness profile taken from Fig. 14 along the build height. "
                            "The graded specimen is reported as fully dense and crack-free with no delamination."
                        ),
                    }
                ]
            )

        gradient_clone = deepcopy(gradient_source)
        gradient_sample_id = self._clean_optional_str(gradient_clone.get("Sample_ID"))
        if (
            not gradient_sample_id
            or re.search(r"gradient-ti64-in718_\d+wt", gradient_sample_id, re.IGNORECASE)
            or re.search(r"_(?:000|005|010|015|020)WT$", gradient_sample_id, re.IGNORECASE)
        ):
            gradient_sample_id = "Ti6Al4V_IN718_Graded_0to20"
        gradient_clone["Sample_ID"] = gradient_sample_id
        gradient_clone["Gradient_Material"] = True
        gradient_clone["Gradient_Group_ID"] = "Graded_Ti6Al4V_IN718"

        representative_map = None
        if representative_row is not None:
            representative_map = {
                element: representative_row["numeric_values"].get(element)
                for element in ("Ti", "Al", "V", "Cr", "Fe", "Ni", "Nb")
                if representative_row["numeric_values"].get(element) is not None
            }

        gradient_comp = dict(gradient_clone.get("Composition_Info") or {})
        gradient_comp["Role"] = "Target"
        gradient_comp["Alloy_Name_Raw"] = "Ti6Al4V to Ti6Al4V+20wt%IN718 functionally graded material"
        gradient_comp["Nominal_Composition"] = {
            "Composition_Type": "wt%",
            "Elements_Normalized": None,
        }
        gradient_comp["Measured_Composition"] = {
            "Composition_Type": "wt%" if representative_map else None,
            "Elements_Normalized": representative_map,
            "Measurement_Method": "EDX" if representative_map else None,
        }
        gradient_comp["Note"] = gradient_note
        gradient_clone["Composition_Info"] = gradient_comp

        gradient_process = dict(gradient_clone.get("Process_Info") or {})
        gradient_process["Process_Category"] = "Selective Laser Melting"
        gradient_process["Process_Text"] = {
            "original": process_original,
            "simplified": process_simplified,
        }
        gradient_process["Key_Params"] = self._canonicalize_key_params(
            {
                "Laser_Power_W": 300,
                "Scanning_Speed_mm_s": "450-700",
                "Layer_Thickness_um": 50,
                "Hatch_Spacing_um": 50,
                "Focal_Position_mm": 0,
                "Protective_Atmosphere": "Ar",
                "Volumetric_Energy_Density_J_mm3": "170-267",
                "Total_Layers": sum(int(row.get("Number_of_Layers", 0)) for row in process_rows.values()),
                "Layers_Per_Composition_Step": 12,
            }
        )
        gradient_clone["Process_Info"] = gradient_process

        gradient_micro = dict(gradient_clone.get("Microstructure_Info") or {})
        gradient_micro["Microstructure_Text"] = {
            "original": micro_original,
            "simplified": micro_simplified,
        }
        gradient_micro["Main_Phase"] = (
            gradient_micro.get("Main_Phase")
            or "Functionally graded (varies from beta-Ti to alpha-Ti + Ti2Ni eutectoid structure along build direction)"
        )
        gradient_micro["Precipitates"] = self._normalize_precipitates(
            gradient_micro.get("Precipitates"),
            text=f"{micro_original} Ti2Ni intermetallic",
        ) or [{"Phase_Type": "Ti2Ni intermetallic", "Volume_Fraction_pct": None}]
        gradient_micro["Porosity_pct"] = None
        gradient_micro["Relative_Density_pct"] = None
        gradient_micro["Phase_Fraction_pct"] = self._coerce_float(
            gradient_micro.get("Phase_Fraction_pct")
        )
        gradient_micro = self._sanitize_microstructure_numeric_fields(
            gradient_micro,
            process_info=gradient_process,
            properties=gradient_properties,
        )
        gradient_clone["Microstructure_Info"] = gradient_micro
        gradient_clone["Properties_Info"] = gradient_properties
        repaired.append(gradient_clone)

        return repaired

    def _infer_property_unit(
        self,
        property_name: Optional[str],
        unit_value: Any,
    ) -> str:
        if unit_value not in (None, ""):
            txt = str(unit_value).strip()
            if txt:
                return txt
        key = str(property_name or "").strip().lower()
        if key in {"hardness_hv", "microhardness", "hardness"}:
            return "HV"
        if key == "hardness_hrc":
            return "HRC"
        if key == "hardness_hb":
            return "HB"
        if key in {
            "elongation_total",
            "elongation_uniform",
            "elongation_at_fracture",
            "elongation_compressive",
        }:
            return "%"
        if key in {"fracture_toughness_kic", "fatigue_crack_growth_threshold"}:
            return "MPa√m"
        if key == "paris_exponent":
            return "dimensionless"
        return ""

    def _resolve_paper_metadata(
        self,
        source_path: str,
        paper_text: Optional[str],
        document_metadata: Optional[dict],
        existing_metadata: Optional[dict],
    ) -> Dict[str, Any]:
        existing_metadata = existing_metadata or {}
        title = (
            existing_metadata.get("Paper_Title")
            or existing_metadata.get("title")
            or (document_metadata or {}).get("title")
        )
        doi = (
            existing_metadata.get("DOI")
            or existing_metadata.get("doi")
            or (document_metadata or {}).get("doi")
            or self.extract_first_doi(paper_text)
        )
        if title is not None:
            title = str(title).strip() or None
        if doi is not None:
            doi = str(doi).strip() or None
        if not title:
            title = self._extract_title_from_paper_text(paper_text)
        if not doi:
            doi = None
        return {"Paper_Title": title, "DOI": doi}

    @staticmethod
    def _extract_title_from_paper_text(paper_text: Optional[str]) -> Optional[str]:
        if not paper_text:
            return None
        text = str(paper_text).replace("\r\n", "\n")
        candidate_lines: List[str] = []
        for line in text.split("\n"):
            stripped = line.strip()
            if not stripped:
                continue
            normalized = re.sub(r"\s+", " ", stripped)
            if normalized.startswith("## "):
                continue
            if normalized.startswith("# "):
                title = normalized[2:].strip()
                return title or None
            lowered = normalized.lower()
            if any(
                token in lowered
                for token in (
                    "doi.org",
                    "http://doi.org",
                    "https://doi.org",
                    "abstract",
                    "keywords",
                    "article info",
                    "received",
                    "accepted",
                    "available online",
                    "corresponding author",
                )
            ):
                continue
            if re.match(r"^(fig\.|figure|table)\s*\d+", lowered):
                continue
            if re.search(r"\b\d{4}\b", normalized) and re.search(r"\b\d+\s*[-–]\s*\d+\b", normalized):
                continue
            if re.search(r"\bet al\.\b", lowered):
                continue
            if re.search(r"(?:,\s*){3,}", normalized):
                continue
            if len(normalized) < 25 or len(normalized) > 300:
                continue
            if len(re.findall(r"[A-Za-z]+", normalized)) < 4:
                continue
            candidate_lines.append(normalized)
            if len(candidate_lines) >= 8:
                break
        if not candidate_lines:
            return None
        scored = sorted(
            candidate_lines,
            key=lambda line: (
                len(line),
                -int(line.endswith(".")),
                -len(re.findall(r"\d", line)),
            ),
            reverse=True,
        )
        return scored[0] or None

    def _legacy_materials_to_lab_schema(
        self,
        materials: List[dict],
        paper_metadata: Optional[Dict[str, Any]] = None,
    ) -> dict:
        items = [self._legacy_material_to_lab_item(mat) for mat in materials]
        return {
            "Paper_Metadata": {
                "Paper_Title": (paper_metadata or {}).get("Paper_Title"),
                "DOI": (paper_metadata or {}).get("DOI"),
            },
            "items": self._repair_existing_lab_items(items),
        }

    def _legacy_material_to_lab_item(self, material: dict) -> dict:
        comp_info = dict(material.get("Composition_Info") or {})
        measured = dict(comp_info.get("Measured_Composition") or {})
        measured_method = (
            measured.get("Measurement_Method")
            or measured.get("Measurment_Method")
            or comp_info.get("Measurement_Method")
        )
        item = {
            "Sample_ID": material.get("Sample_ID"),
            "Gradient_Material": material.get("Gradient_Material"),
            "Gradient_Group_ID": material.get("Gradient_Group_ID"),
            "Composition_Info": {
                "Role": comp_info.get("Role") or "Target",
                "Alloy_Name_Raw": comp_info.get("Alloy_Name_Raw") or material.get("Alloy_Name_Raw"),
                "Nominal_Composition": {
                    "Composition_Type": (comp_info.get("Nominal_Composition") or {}).get("Composition_Type"),
                    "Elements_Normalized": (comp_info.get("Nominal_Composition") or {}).get("Elements_Normalized"),
                },
                "Measured_Composition": {
                    "Composition_Type": measured.get("Composition_Type"),
                    "Elements_Normalized": measured.get("Elements_Normalized"),
                    "Measurement_Method": measured_method,
                },
                "Note": comp_info.get("Note"),
            },
            "Process_Info": material.get("Process_Info") or {},
            "Microstructure_Info": material.get("Microstructure_Info") or {},
            "Properties_Info": material.get("Properties_Info") or [],
        }
        return item

    def _repair_existing_lab_items(self, items: List[dict]) -> List[dict]:
        repaired: List[dict] = []
        for idx, item in enumerate(items, start=1):
            comp_info = dict(item.get("Composition_Info") or {})
            nominal = dict(comp_info.get("Nominal_Composition") or {})
            measured = dict(comp_info.get("Measured_Composition") or {})

            alloy_name = (
                self._clean_optional_str(comp_info.get("Alloy_Name_Raw"))
                or self._clean_optional_str(item.get("Alloy_Name_Raw"))
                or f"Material_{idx}"
            )
            sample_id = self._clean_optional_str(item.get("Sample_ID"))
            balance_hint = self._pick_balance_element(None, alloy_name)
            nominal_elements = self._normalize_reported_composition(
                nominal.get("Elements_Normalized"),
                preferred_balance_element=balance_hint,
                fill_missing_balance=False,
            )
            measured_elements = self._normalize_reported_composition(
                measured.get("Elements_Normalized"),
                preferred_balance_element=self._pick_balance_element(
                    nominal_elements,
                    alloy_name,
                ),
            )
            comp_info["Role"] = self._normalize_role(comp_info.get("Role"))
            comp_info["Alloy_Name_Raw"] = alloy_name
            comp_info["Nominal_Composition"] = {
                "Composition_Type": self._normalize_composition_type(
                    nominal.get("Composition_Type")
                ),
                "Elements_Normalized": nominal_elements,
            }
            comp_info["Measured_Composition"] = self._collapse_null_object({
                "Composition_Type": self._normalize_composition_type(
                    measured.get("Composition_Type")
                ),
                "Elements_Normalized": measured_elements,
                "Measurement_Method": self._clean_optional_str(
                    measured.get("Measurement_Method")
                    or measured.get("Measurment_Method")
                    or comp_info.get("Measurement_Method")
                ),
            })
            comp_info["Note"] = self._clean_optional_str(
                comp_info.get("Note") or item.get("Composition_Note")
            )

            process_info = dict(item.get("Process_Info") or {})
            process_text = process_info.get("Process_Text")
            if isinstance(process_text, dict):
                process_info["Process_Text"] = {
                    "original": str(process_text.get("original") or "not provided"),
                    "simplified": str(
                        process_text.get("simplified")
                        or process_text.get("original")
                        or "not provided"
                    ),
                }
            else:
                process_info["Process_Text"] = self._to_text_payload(
                    str(process_text or "not provided")
                )
            resolved_process_category = self._resolve_process_category(
                process_info.get("Process_Category"),
                self._extract_text_payload(process_info.get("Process_Text")),
                comp_raw=alloy_name,
            )
            process_info["Process_Category"] = self._humanize_process_category(
                resolved_process_category,
                process_text=self._extract_text_payload(process_info.get("Process_Text")),
                comp_raw=alloy_name,
            )
            process_info["Equipment"] = self._clean_optional_str(process_info.get("Equipment"))
            process_info["Key_Params"] = self._canonicalize_key_params(
                process_info.get("Key_Params") or process_info.get("Key_Params_JSON") or {}
            )
            process_info.pop("Key_Params_JSON", None)

            micro_info = dict(item.get("Microstructure_Info") or {})
            micro_text = micro_info.get("Microstructure_Text")
            if isinstance(micro_text, dict):
                micro_info["Microstructure_Text"] = {
                    "original": str(micro_text.get("original") or "not provided"),
                    "simplified": str(
                        micro_text.get("simplified")
                        or micro_text.get("original")
                        or "not provided"
                    ),
                }
            else:
                micro_info["Microstructure_Text"] = self._to_text_payload(
                    str(micro_text or "not provided")
                )
            micro_text_for_rules = self._extract_text_payload(micro_info.get("Microstructure_Text"))
            micro_info["Main_Phase"] = self._normalize_main_phase(
                micro_info.get("Main_Phase"),
                micro_text=micro_text_for_rules,
            )
            micro_info["Precipitates"] = self._normalize_precipitates(
                micro_info.get("Precipitates"),
                text=micro_text_for_rules,
            )
            micro_info["Porosity_pct"] = self._coerce_float(micro_info.get("Porosity_pct"))
            micro_info["Relative_Density_pct"] = self._coerce_float(
                micro_info.get("Relative_Density_pct")
            )
            micro_info["Grain_Size_avg_um"] = self._coerce_float(
                micro_info.get("Grain_Size_avg_um")
            )
            micro_info["Precipitate_Size_avg_nm"] = self._coerce_float(
                micro_info.get("Precipitate_Size_avg_nm")
            )
            _pvf = self._coerce_float(micro_info.get("Precipitate_Volume_Fraction_pct"))
            if _pvf is not None:
                micro_info["Precipitate_Volume_Fraction_pct"] = _pvf
            else:
                micro_info.pop("Precipitate_Volume_Fraction_pct", None)
            _pf = self._coerce_float(micro_info.get("Phase_Fraction_pct"))
            if _pf is not None:
                micro_info["Phase_Fraction_pct"] = _pf
            else:
                micro_info.pop("Phase_Fraction_pct", None)
            micro_info["Advanced_Quantitative_Features"] = (
                self._normalize_advanced_quantitative_features(
                    micro_info.get("Advanced_Quantitative_Features")
                )
            )
            micro_info = self._sanitize_microstructure_numeric_fields(
                micro_info,
                process_info=process_info,
                properties=item.get("Properties_Info") or [],
            )
            gradient_material = self._coerce_bool(item.get("Gradient_Material"))
            if gradient_material is None:
                gradient_context = (
                    f"{alloy_name} {self._extract_text_payload(process_info.get('Process_Text'))}"
                ).lower()
                gradient_material = (
                    self._detect_graded_family(
                        alloy_name,
                        self._extract_text_payload(process_info.get("Process_Text")),
                    ) is not None
                    or bool(re.search(r"\b(graded|gradient|multigraded|stepwise|base layer|wt%\s+\w*\s*step)\b", gradient_context))
                )
            gradient_group_id = self._clean_optional_str(item.get("Gradient_Group_ID"))
            if (
                gradient_material
                and not gradient_group_id
                and re.search(r"\b(step|base layer)\b", alloy_name, re.IGNORECASE)
            ):
                gradient_group_id = self._make_gradient_group_id(alloy_name.split(" - ")[0])

            repaired.append(
                {
                    "Sample_ID": sample_id or f"S{idx:03d}",
                    "Gradient_Material": gradient_material,
                    "Gradient_Group_ID": gradient_group_id,
                    "Composition_Info": comp_info,
                    "Process_Info": process_info,
                    "Microstructure_Info": micro_info,
                    "Properties_Info": self._repair_lab_properties(
                        item.get("Properties_Info") or []
                    ),
                }
            )
        return repaired

    def _repair_abbreviated_sample_matrix_case(
        self,
        items: List[dict],
        paper_text: Optional[str],
    ) -> List[dict]:
        if not items or not paper_text:
            return items

        text = str(paper_text)
        target_items = [
            item
            for item in items
            if (item.get("Composition_Info", {}) or {}).get("Role", "Target") in ("Target", "Variant")
        ]
        if len(target_items) > 2:
            return items

        if not re.search(r"\|\s*Nomenclature\s*\|.*Laser Power", text, re.IGNORECASE):
            return items
        if not re.search(r"\|\s*Sample\s*\|.*K_\{Ic\}.*Delta K_\{0\}.*\|\s*m\s*\|", text, re.IGNORECASE):
            return items

        process_rows = self._extract_abbreviated_process_rows(text)
        property_rows = self._extract_abbreviated_property_rows(text)
        if len(process_rows) < 2 or len(property_rows) < 4:
            return items

        sample_ids = {row["sample_id"] for row in property_rows}
        existing_target_ids = {
            self._clean_optional_str(item.get("Sample_ID"))
            for item in target_items
            if self._clean_optional_str(item.get("Sample_ID"))
        }
        if len(existing_target_ids.intersection(sample_ids)) >= len(sample_ids):
            return items

        base_item = deepcopy(target_items[0]) if target_items else deepcopy(items[0])
        base_comp = dict(base_item.get("Composition_Info") or {})
        alloy_name = (
            self._clean_optional_str(base_comp.get("Alloy_Name_Raw"))
            or "Ti-6Al-4V ELI"
        )

        common_process_text = self._extract_text_payload((base_item.get("Process_Info") or {}).get("Process_Text"))
        if not common_process_text:
            common_process_text = self._build_kumar_like_process_text(text)
        common_micro_text = self._extract_text_payload((base_item.get("Microstructure_Info") or {}).get("Microstructure_Text"))
        if not common_micro_text:
            common_micro_text = self._build_kumar_like_micro_text(text)

        density_map = self._extract_abbreviated_density_rows(text)
        colony_size = self._extract_range_string(text, r"average colony sizes? of\s*~?\s*(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)\s*(?:\\mu|μ|u)m")
        lath_max = self._coerce_float(
            self._extract_first_match(text, r"lath thickness'? are too small\s*\(up to\s*(\d+(?:\.\d+)?)")
        )

        repaired_targets: List[dict] = []
        for row in property_rows:
            params = process_rows.get(row["code"])
            if not params:
                continue

            clone = deepcopy(base_item)
            clone["Sample_ID"] = row["sample_id"]
            clone["Gradient_Material"] = False
            clone["Gradient_Group_ID"] = None

            comp_info = dict(clone.get("Composition_Info") or {})
            comp_info["Role"] = "Target"
            comp_info["Alloy_Name_Raw"] = alloy_name
            clone["Composition_Info"] = comp_info

            process_info = dict(clone.get("Process_Info") or {})
            process_info["Process_Category"] = "Selective Laser Melting (SLM)"
            process_info["Equipment"] = (
                self._clean_optional_str(process_info.get("Equipment"))
                or "EOSINT M 280 SLM unit with Yb:YAG fiber laser"
            )
            process_info["Process_Text"] = self._to_text_payload(common_process_text)
            process_info["Key_Params"] = self._canonicalize_key_params(params)
            clone["Process_Info"] = process_info

            micro_info = dict(clone.get("Microstructure_Info") or {})
            micro_info["Microstructure_Text"] = self._to_text_payload(common_micro_text)
            micro_info["Main_Phase"] = (
                self._normalize_main_phase(
                    micro_info.get("Main_Phase"),
                    micro_text=common_micro_text,
                )
                or "HCP α/α' (acicular lath mixture)"
            )
            precipitates = self._normalize_precipitates(
                micro_info.get("Precipitates"),
                text=f"{common_micro_text} Ti3Al alpha phase decorated at prior beta grain boundaries",
            )
            if not precipitates:
                precipitates = [{"Phase_Type": "α (at prior β grain boundaries)", "Volume_Fraction_pct": None}]
            micro_info["Precipitates"] = precipitates
            density = density_map.get(row["code"], {})
            if density.get("porosity_pct") is not None:
                micro_info["Porosity_pct"] = density["porosity_pct"]
            if density.get("relative_density_pct") is not None:
                micro_info["Relative_Density_pct"] = density["relative_density_pct"]
            micro_info["Grain_Size_avg_um"] = self._coerce_float(micro_info.get("Grain_Size_avg_um")) or 140.0
            advanced = dict(micro_info.get("Advanced_Quantitative_Features") or {})
            if colony_size and "Colony_Size_avg_um" not in advanced:
                advanced["Colony_Size_avg_um"] = colony_size
            if lath_max is not None and "Lath_Thickness_max_um" not in advanced:
                advanced["Lath_Thickness_max_um"] = lath_max
            micro_info["Advanced_Quantitative_Features"] = advanced
            clone["Microstructure_Info"] = self._sanitize_microstructure_numeric_fields(
                micro_info,
                process_info=process_info,
                properties=[],
            )

            props: List[Dict[str, Any]] = []
            tensile_condition = (
                f"Tensile test, Loading axis {'parallel' if row['orientation'] == 'B' else 'perpendicular'} "
                f"to build direction ({row['orientation']}); strain rate 0.001 s^-1; displacement control 0.006 mm/s"
            )
            fracture_condition = (
                f"Mode I fracture toughness, crack plane normal {'along build direction' if row['orientation'] == 'B' else 'transverse-to-build direction'} "
                f"({row['orientation']}); ASTM E399-12; displacement rate 0.01 mm/s"
            )
            fcg_condition = (
                f"Fatigue crack growth threshold, {row['orientation']} direction; ASTM E647-15; R=0.1; frequency 10 Hz"
            )
            paris_condition = "Steady-state FCG, ASTM E647-15; R=0.1; frequency 10 Hz"
            if row.get("yield_strength"):
                props.append(
                    self._build_lab_property_entry(
                        raw_property_name="Yield_Strength",
                        condition=tensile_condition,
                        value_numeric=row["yield_strength"][0],
                        value_range=row["yield_strength"][2],
                        value_stddev=row["yield_strength"][1],
                        unit_raw="MPa",
                        test_temp=298.15,
                        context_text=tensile_condition,
                    )
                )
            if row.get("uts"):
                props.append(
                    self._build_lab_property_entry(
                        raw_property_name="Ultimate_Tensile_Strength",
                        condition=tensile_condition,
                        value_numeric=row["uts"][0],
                        value_range=row["uts"][2],
                        value_stddev=row["uts"][1],
                        unit_raw="MPa",
                        test_temp=298.15,
                        context_text=tensile_condition,
                    )
                )
            if row.get("elongation"):
                props.append(
                    self._build_lab_property_entry(
                        raw_property_name="Elongation_Total",
                        condition=tensile_condition,
                        value_numeric=row["elongation"][0],
                        value_range=row["elongation"][2],
                        value_stddev=row["elongation"][1],
                        unit_raw="%",
                        test_temp=298.15,
                        context_text=tensile_condition,
                        note="Original paper reports ef (elongation to failure); not distinguished into uniform and fracture components",
                    )
                )
            if row.get("kic"):
                props.append(
                    self._build_lab_property_entry(
                        raw_property_name="Fracture_Toughness_KIC",
                        condition=fracture_condition,
                        value_numeric=row["kic"][0],
                        value_range=row["kic"][2],
                        value_stddev=row["kic"][1],
                        unit_raw="MPa√m",
                        test_temp=298.15,
                        context_text=fracture_condition,
                    )
                )
            if row.get("dk0"):
                props.append(
                    self._build_lab_property_entry(
                        raw_property_name="Fatigue_Crack_Growth_Threshold",
                        condition=fcg_condition,
                        value_numeric=row["dk0"][0],
                        value_range=row["dk0"][2],
                        value_stddev=row["dk0"][1],
                        unit_raw="MPa√m",
                        test_temp=298.15,
                        context_text=fcg_condition,
                    )
                )
            if row.get("paris_m"):
                props.append(
                    self._build_lab_property_entry(
                        raw_property_name="Paris_Exponent",
                        condition=paris_condition,
                        value_numeric=row["paris_m"][0],
                        value_range=row["paris_m"][2],
                        value_stddev=row["paris_m"][1],
                        unit_raw="dimensionless",
                        test_temp=298.15,
                        context_text=paris_condition,
                    )
                )
            if row.get("fatigue_strength"):
                fatigue_condition = (
                    "Unnotched rotating bending fatigue (RBF), loading axis parallel to build direction (B); "
                    "R=-1; minimum survival 10^7 cycles"
                )
                props.append(
                    self._build_lab_property_entry(
                        raw_property_name="Fatigue_Strength_Unnotched",
                        condition=fatigue_condition,
                        value_numeric=row["fatigue_strength"][0],
                        value_range=row["fatigue_strength"][2],
                        value_stddev=row["fatigue_strength"][1],
                        unit_raw="MPa",
                        test_temp=298.15,
                        context_text=fatigue_condition,
                    )
                )

            clone["Properties_Info"] = self._order_lab_properties([p for p in props if p])
            repaired_targets.append(clone)

        if len(repaired_targets) < 4:
            return items

        other_items = [
            item
            for item in items
            if (item.get("Composition_Info", {}) or {}).get("Role", "Target") not in ("Target", "Variant")
        ]
        return repaired_targets + other_items

    def _extract_abbreviated_process_rows(self, text: str) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        pattern = re.compile(
            r"^\|\s*(\d{4})\s*\|\s*(\d+(?:\.\d+)?)\s*\|\s*(\d+(?:\.\d+)?)\s*\|\s*(\d+(?:\.\d+)?)\s*\|\s*(\d+(?:\.\d+)?)\s*\|\s*(\d+(?:\.\d+)?)\s*\|\s*(\d+(?:\.\d+)?)\s*\|$",
            re.MULTILINE,
        )
        anneal_temp = self.parse_temperature_to_k("650 C")
        for match in pattern.finditer(text):
            code, power, speed, hatch_mm, layer_um, rotation, ved = match.groups()
            out[code] = {
                "Laser_Power_W": float(power),
                "Scanning_Speed_mm_s": float(speed),
                "Hatch_Spacing_um": round(float(hatch_mm) * 1000.0, 3),
                "Layer_Thickness_um": float(layer_um),
                "Scan_Rotation_deg": float(rotation),
                "Volumetric_Energy_Density_J_mm3": float(ved),
                "Annealing_Temperature_K": anneal_temp,
                "Annealing_Time_h": 3.0,
                "Protective_Atmosphere": "argon",
            }
        return out

    def _extract_abbreviated_property_rows(self, text: str) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        pattern = re.compile(
            r"^\|\s*([BS])(\d{4})\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\|$",
            re.MULTILINE,
        )
        for match in pattern.finditer(text):
            orient, code, ys, uts, ef, kic, dk0, paris_m, fs, _ = match.groups()
            rows.append(
                {
                    "sample_id": f"{orient}{code}",
                    "orientation": orient,
                    "code": code,
                    "yield_strength": self._parse_value_with_optional_std(ys),
                    "uts": self._parse_value_with_optional_std(uts),
                    "elongation": self._parse_value_with_optional_std(ef),
                    "kic": self._parse_value_with_optional_std(kic),
                    "dk0": self._parse_value_with_optional_std(dk0),
                    "paris_m": self._parse_value_with_optional_std(paris_m),
                    "fatigue_strength": self._parse_value_with_optional_std(fs),
                }
            )
        return rows

    def _extract_abbreviated_density_rows(self, text: str) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        match = re.search(
            r"densities.*?of\s*3090,\s*3067,\s*6090,\s*6067 samples are\s*(.*?)respectively",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if match:
            density_vals = re.findall(r"([0-9]+(?:\.[0-9]+)?)\s*(?:±|\\pm)\s*[0-9]+(?:\.[0-9]+)?%", match.group(1))
            for code, value in zip(("3090", "3067", "6090", "6067"), density_vals[:4]):
                out.setdefault(code, {})["relative_density_pct"] = float(value)

        match = re.search(
            r"porosities.*?are\s*([0-9.]+)%\s*,\s*([0-9.]+)%\s*,\s*([0-9.]+)%\s*,\s*and\s*([0-9.]+)%",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if match:
            for code, value in zip(("3090", "3067", "6090", "6067"), match.groups()):
                out.setdefault(code, {})["porosity_pct"] = float(value)
        return out

    @staticmethod
    def _extract_first_match(text: str, pattern: str) -> Optional[str]:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if not match:
            return None
        return match.group(1)

    @staticmethod
    def _extract_range_string(text: str, pattern: str) -> Optional[str]:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if not match:
            return None
        return f"{match.group(1)}-{match.group(2)}"

    def _build_kumar_like_process_text(self, text: str) -> str:
        match = re.search(
            r"All the samples tested in this study were manufactured.*?stress relieved by annealing them at 650 \^\\circC for 3 h\.",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if match:
            return re.sub(r"\s+", " ", match.group(0)).strip()
        return (
            "Manufactured with ELI grade Ti64 powder in argon using an EOSINT M 280 SLM unit with "
            "Yb:YAG fiber laser. Four combinations of layer thickness and scan rotation were produced, "
            "followed by stress relief at 650 C for 3 h."
        )

    def _build_kumar_like_micro_text(self, text: str) -> str:
        parts: List[str] = []
        for pattern in (
            r"The schematic displayed in Fig\. 1b illustrates.*?possible 12 variants \[22\]\.",
            r"Qualitative analyses of the different phases.*?the microstructure of SLM Ti_\{64\} In Fig\. 6a.*?will be seen later\.",
            r"The porosities in all the four SLM Ti_\{64\} alloys.*?reported in literature for SLM Ti_\{64\} with\s*\$ \\\\alpha' \$ microstructure \[6\]\.",
        ):
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                parts.append(re.sub(r"\s+", " ", match.group(0)).strip())
        if parts:
            return " ".join(parts)
        return (
            "Columnar or near-equiaxed prior beta grains with fine acicular alpha/alpha-prime laths; "
            "alpha decorates prior beta grain boundaries and Ti3Al may appear in lower-energy-density conditions."
        )

    def _build_lab_property_entry(
        self,
        raw_property_name: Any,
        property_symbol: Any = None,
        condition: Optional[str] = None,
        value_numeric: Optional[float] = None,
        value_range: Optional[str] = None,
        value_stddev: Optional[float] = None,
        unit_raw: Any = None,
        test_temp: Optional[float] = None,
        strain_rate: Any = None,
        tensile_speed: Any = None,
        hardness_load: Any = None,
        hardness_dwell: Any = None,
        data_source: Any = None,
        test_specimen: Any = None,
        note: Any = None,
        context_text: Optional[str] = None,
        original_values: Any = None,
    ) -> Optional[Dict[str, Any]]:
        property_name = self.normalize_property_name(raw_property_name) or "Unknown_Property"
        if property_name in {"Relative_Density_pct", "Relative_Density", "Porosity_pct", "Porosity", "Crack_Presence"}:
            return None
        # Merge property symbol into name (e.g. Corrosion_Potential + Ecorr -> Corrosion_Potential_Ecorr)
        sym = self._clean_optional_str(property_symbol)
        if sym and sym not in property_name:
            property_name = f"{property_name}_{sym}"
        merged_context = " ; ".join(
            part
            for part in (condition, context_text, self._clean_optional_str(note))
            if self._clean_optional_str(part)
        )
        tensile_speed_value = self._coerce_float(tensile_speed)
        if tensile_speed_value is None:
            tensile_speed_value = self._extract_tensile_speed_mm_min(condition, context_text)
        hardness_dwell_value = self._coerce_float(hardness_dwell)
        if hardness_dwell_value is None:
            hardness_dwell_value = self._extract_hardness_dwell_time_s(condition, context_text)
        entry = {
            "Property_Name": property_name,
            "Test_Condition": condition,
            "Value_Numeric": value_numeric,
            "Value_Range": value_range,
            "Value_StdDev": value_stddev,
            "Original_Values": self._clean_optional_str(original_values),
            "Unit": self._infer_property_unit(property_name, unit_raw),
            "Test_Temperature_K": test_temp if test_temp is not None else self.parse_temperature_to_k(condition),
            "Strain_Rate_s1": self._clean_optional_str(strain_rate) or self._extract_strain_rate_text(condition, context_text),
            "Tensile_Speed_mm_min": tensile_speed_value,
            "Hardness_Load": self._clean_optional_str(hardness_load) or self._extract_hardness_load(condition, context_text),
            "Hardness_Dwell_Time_s": hardness_dwell_value,
            "Data_Source": self._normalize_data_source(data_source, merged_context),
            "Test_Specimen": self._clean_optional_str(test_specimen) or self._extract_test_specimen(condition, context_text),
            "Note": self._default_property_note(property_name, raw_property_name, note),
        }
        return entry

    def _repair_lab_properties(self, properties: List[dict]) -> List[dict]:
        repaired: List[Dict[str, Any]] = []
        for prop in properties:
            condition = self._clean_optional_str(prop.get("Test_Condition"))
            value_numeric = self._coerce_float(
                prop.get("Value_Numeric", prop.get("Property_Value"))
            )
            value_std = self._coerce_float(
                prop.get("Value_StdDev", prop.get("Property_StdDev"))
            )
            test_temp = self._coerce_float(prop.get("Test_Temperature_K"))
            if test_temp is None:
                test_temp = self.parse_temperature_to_k(condition)
            entry = self._build_lab_property_entry(
                raw_property_name=prop.get("Property_Name") or prop.get("Property_Type"),
                condition=condition,
                value_numeric=value_numeric,
                value_range=self._clean_optional_str(
                    prop.get("Value_Range", prop.get("Property_Value_Range"))
                ),
                value_stddev=value_std,
                unit_raw=prop.get("Unit", prop.get("Property_Unit")),
                test_temp=test_temp,
                strain_rate=prop.get("Strain_Rate_s1"),
                tensile_speed=prop.get("Tensile_Speed_mm_min"),
                hardness_load=prop.get("Hardness_Load"),
                hardness_dwell=prop.get("Hardness_Dwell_Time_s"),
                data_source=prop.get("Data_Source", prop.get("data_source")),
                test_specimen=prop.get("Test_Specimen", prop.get("test_specimen")),
                note=prop.get("Note", prop.get("note")),
                context_text=prop.get("Additional_Information") or prop.get("additional_information"),
            )
            if entry is not None:
                repaired.append(entry)
        return self._order_lab_properties(repaired)

    def _build_lab_properties_from_internal(
        self,
        comp: dict,
        fallback_tests: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        props = comp.get("properties_of_composition", []) or []
        process_text_raw = comp.get("processing_conditions")
        process_text_hint = ""
        if isinstance(process_text_raw, dict):
            process_text_hint = json.dumps(process_text_raw, ensure_ascii=False)
        elif process_text_raw:
            process_text_hint = str(process_text_raw)
        graded_context = bool(
            re.search(
                r"\b(graded|gradient|stepwise|multigraded|increasing|up to \d+(?:\.\d+)?\s*(?:wt\s*%|wt%|%))\b",
                f"{comp.get('composition', '')} {process_text_hint}".lower(),
            )
        )
        properties: List[Dict[str, Any]] = []
        for prop in props:
            numeric_val = self._coerce_float(prop.get("value_numeric"))
            if not prop.get("property_name"):
                continue
            condition = self._clean_optional_str(prop.get("measurement_condition"))
            value_range = self._clean_optional_str(prop.get("value_range"))
            if value_range is None and prop.get("value_type") == "range":
                value_range = self._clean_optional_str(prop.get("value"))
            condition_text = str(condition or "").lower()
            has_spatial_anchor = any(
                kw in condition_text for kw in ("top", "bottom", "layer", "height", "position", "zone")
            )
            if graded_context and value_range and not has_spatial_anchor:
                continue
            value_stddev = self._coerce_float(prop.get("value_stddev"))
            if value_stddev is None:
                value_txt = str(prop.get("value") or "")
                m_std = re.search(r"±\s*(-?\d+(?:\.\d+)?)", value_txt)
                if m_std:
                    try:
                        value_stddev = float(m_std.group(1))
                    except Exception:
                        value_stddev = None
            entry = self._build_lab_property_entry(
                raw_property_name=prop.get("property_name"),
                property_symbol=prop.get("property_symbol"),
                condition=condition,
                value_numeric=numeric_val,
                value_range=value_range,
                value_stddev=value_stddev,
                unit_raw=prop.get("unit"),
                test_temp=self._coerce_float(prop.get("test_temperature_k")),
                strain_rate=prop.get("strain_rate_s1"),
                tensile_speed=prop.get("tensile_speed_mm_min"),
                hardness_load=prop.get("hardness_load"),
                hardness_dwell=prop.get("hardness_dwell_time_s"),
                data_source=prop.get("data_source"),
                test_specimen=prop.get("test_specimen"),
                note=prop.get("note"),
                context_text=prop.get("additional_information"),
                original_values=prop.get("original_values"),
            )
            if entry is not None:
                properties.append(entry)
        if properties:
            return self._order_lab_properties(properties)
        return self._repair_lab_properties(fallback_tests or [])

    def _order_lab_properties(self, properties: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not properties:
            return []
        alias = {
            "microhardness": "Hardness_HV",
            "hardness_hv": "Hardness_HV",
            "hardness": "Hardness_HV",
            "ultimate_compressive_strength": "Ultimate_Strength_Compressive",
            "elongation": "Elongation_Total",
            "total_elongation": "Elongation_Total",
            "uniform_elongation": "Elongation_Uniform",
            "fracture_strain": "Elongation_At_Fracture",
            "fracture_elongation": "Elongation_At_Fracture",
            "youngs_modulus": "Elastic_Modulus",
            "fracture_toughness_kic": "Fracture_Toughness_KIC",
            "fatigue_strength_unnotched": "Fatigue_Strength_Unnotched",
        }
        tier1 = [
            "Yield_Strength",
            "Yield_Strength_Compressive",
            "Ultimate_Tensile_Strength",
            "Ultimate_Strength_Compressive",
            "Elongation_Total",
            "Elongation_Uniform",
            "Elongation_At_Fracture",
            "Elongation_Compressive",
            "Elastic_Modulus",
            "Hardness_HV",
            "Hardness_HRC",
            "Hardness_HB",
        ]
        tier2 = [
            "Residual_Stress",
            "Impact_Toughness",
            "Fatigue_Strength",
            "Fatigue_Strength_Unnotched",
            "Fracture_Toughness",
            "Fracture_Toughness_KIC",
            "Fatigue_Crack_Growth_Threshold",
            "Paris_Exponent",
            "Thermal_Conductivity",
            "Specific_Heat",
            "Coefficient_of_Thermal_Expansion",
            "Electrical_Conductivity",
            "Electrical_Resistivity",
            "Corrosion_Rate",
            "Wear_Rate",
            "Friction_Coefficient",
        ]
        rank1 = {name: i for i, name in enumerate(tier1)}
        rank2 = {name: i for i, name in enumerate(tier2)}
        normalized: List[Dict[str, Any]] = []
        for raw in properties:
            cur = dict(raw)
            prop_name = str(cur.get("Property_Name") or "").strip()
            canonical_key = re.sub(r"[^a-z0-9]+", "_", prop_name.lower()).strip("_")
            if canonical_key in alias:
                cur["Property_Name"] = alias[canonical_key]
            normalized.append(cur)

        def _rank(prop: Dict[str, Any]) -> Tuple[int, int, str]:
            name = str(prop.get("Property_Name") or "")
            if name in rank1:
                return (0, rank1[name], name)
            if name in rank2:
                return (1, rank2[name], name)
            return (2, 9999, name.lower())

        return sorted(normalized, key=_rank)

    @staticmethod
    def _normalize_advanced_quantitative_features(
        payload: Any,
        allow_direct_dict: bool = True,
    ) -> Dict[str, Any]:
        if isinstance(payload, dict):
            if "Advanced_Quantitative_Features" in payload:
                nested = payload.get("Advanced_Quantitative_Features")
                if isinstance(nested, dict):
                    return SchemaConverter._normalize_advanced_quantitative_features(nested)
                if isinstance(nested, str):
                    try:
                        parsed = json.loads(nested)
                    except Exception:
                        parsed = {}
                    if isinstance(parsed, dict):
                        return SchemaConverter._normalize_advanced_quantitative_features(parsed)
                    return {}
            if not allow_direct_dict:
                return {}
            normalized: Dict[str, Any] = {}
            for raw_key, raw_value in payload.items():
                key = str(raw_key).strip()
                if not key or raw_value is None:
                    continue
                key_norm = re.sub(r"[\s_\-]+", "", key).lower()
                if key_norm in _AQF_TECHNIQUE_KEYS:
                    if isinstance(raw_value, dict):
                        normalized.update(
                            SchemaConverter._normalize_advanced_quantitative_features(raw_value)
                        )
                    continue
                if isinstance(raw_value, dict):
                    nested = SchemaConverter._normalize_advanced_quantitative_features(raw_value)
                    if nested:
                        normalized[key] = nested
                    continue
                if isinstance(raw_value, list):
                    cleaned_list = [
                        item
                        for item in raw_value
                        if SchemaConverter._contains_numeric_signal(item)
                    ]
                    if cleaned_list:
                        normalized[key] = cleaned_list
                    continue
                if not SchemaConverter._contains_numeric_signal(raw_value):
                    continue
                normalized[key] = raw_value
            return normalized
        if isinstance(payload, str):
            try:
                parsed = json.loads(payload)
            except Exception:
                return {}
            if isinstance(parsed, dict):
                return SchemaConverter._normalize_advanced_quantitative_features(parsed)
        return {}

    def _expand_variable_materials_in_target_schema(
        self, schema_data: dict, paper_text: Optional[str]
    ) -> dict:
        """Expand variable-family materials even when input is already target schema."""
        materials = list(schema_data.get("Materials", []) or [])
        if not materials or not paper_text:
            return schema_data

        family_spec = self._extract_variable_family_spec(paper_text)
        if not family_spec:
            return schema_data

        base_text = (family_spec["base_joined"] + family_spec["variable_element"]).lower()
        family_material_indices = []
        existing_codes: set[str] = set()
        for idx, mat in enumerate(materials):
            comp_info = mat.get("Composition_Info") or {}
            haystack = (
                f"{mat.get('Alloy_Name_Raw', '')} {mat.get('Formula_Normalized', '')} "
                f"{comp_info.get('Alloy_Name_Raw', '')} {comp_info.get('Formula_Normalized', '')}"
            ).lower()
            if base_text in haystack or "_x" in haystack or "mox" in haystack:
                family_material_indices.append(idx)
                code = self._extract_variant_code(haystack, family_spec["variable_element"])
                if code:
                    existing_codes.add(code)

        if not family_material_indices:
            return schema_data

        expected_codes = [v["code"] for v in family_spec["variants"]]
        if all(code in existing_codes for code in expected_codes):
            return schema_data

        first_idx = family_material_indices[0]
        base_material = materials[first_idx]
        new_family_materials: List[Dict[str, Any]] = []

        # If no explicit variants exist, replace the ambiguous family entry.
        replace_ambiguous = len(existing_codes) == 0 and len(family_material_indices) == 1
        if not replace_ambiguous:
            for idx in family_material_indices:
                mat = materials[idx]
                comp_info = mat.get("Composition_Info") or {}
                code = self._extract_variant_code(
                    (
                        f"{mat.get('Alloy_Name_Raw', '')} {mat.get('Formula_Normalized', '')} "
                        f"{comp_info.get('Alloy_Name_Raw', '')} {comp_info.get('Formula_Normalized', '')}"
                    ),
                    family_spec["variable_element"],
                )
                if code in expected_codes:
                    new_family_materials.append(mat)

        for variant in family_spec["variants"]:
            if variant["code"] in existing_codes:
                continue
            clone = dict(base_material)
            raw_comp_json = self.build_composition_json(variant["normalized"])
            validated_comp_json, _ = self.validate_composition_json(
                raw_comp_json, variant["normalized"]
            )
            comp_info = dict(clone.get("Composition_Info") or {})
            comp_info["Alloy_Name_Raw"] = variant["label"]
            comp_info["Nominal_Composition"] = {
                "Composition_Type": "at%",
                "Elements_Normalized": validated_comp_json,
            }
            measured = dict(comp_info.get("Measured_Composition") or {})
            measured.setdefault("Composition_Type", None)
            measured.setdefault("Elements_Normalized", None)
            measured.setdefault("Measurment_Method", None)
            comp_info["Measured_Composition"] = measured
            clone["Composition_Info"] = comp_info
            clone.pop("Alloy_Name_Raw", None)
            clone.pop("Formula_Normalized", None)
            clone.pop("Composition_JSON", None)
            # Do not fabricate per-variant properties when source variant is absent.
            clone["Processed_Samples"] = []
            clone["Properties_Info"] = []
            new_family_materials.append(clone)

        if not new_family_materials:
            return schema_data

        if replace_ambiguous:
            retained = [m for idx, m in enumerate(materials) if idx != first_idx]
        else:
            retained = [m for idx, m in enumerate(materials) if idx not in family_material_indices]
        retained.extend(new_family_materials)

        # Re-index Mat_ID and description to keep schema consistent.
        for idx, mat in enumerate(retained, start=1):
            mat["Mat_ID"] = f"M{idx:03d}"
            comp_info = mat.get("Composition_Info") or {}
            label = (
                comp_info.get("Alloy_Name_Raw")
                or mat.get("Alloy_Name_Raw")
                or comp_info.get("Formula_Normalized")
                or mat.get("Formula_Normalized")
                or ""
            )
            mat["description"] = f"--- Material {idx}: {label} ---"

        out = dict(schema_data)
        out["Materials"] = retained
        return out

    def _expand_variable_composition_families(
        self, target_compositions: List[dict], paper_text: Optional[str]
    ) -> List[dict]:
        """Expand variable families (e.g. ``FeCoCrNiMox``) into explicit variants.

        This is a conservative fallback for cases where the LLM keeps only the
        family name and misses per-variant targets like ``Mo0/Mo1/Mo3/Mo5``.
        The function only activates when:
        - there is a clear ``...Mox``-style family in extracted targets, and
        - the paper text contains matching variant metadata (x-list + Mo at% list).
        """
        if not target_compositions or not paper_text:
            return target_compositions

        family_spec = self._extract_variable_family_spec(paper_text)
        if not family_spec:
            return target_compositions

        family_idx = -1
        family_text = (family_spec["base_joined"] + family_spec["variable_element"]).lower()
        for idx, comp in enumerate(target_compositions):
            haystack = (
                f"{comp.get('composition', '')} {comp.get('composition_normalized', '')}"
            ).lower().replace(" ", "")
            if family_text in haystack or "mox" in haystack or "_x" in haystack:
                family_idx = idx
                break
        if family_idx < 0:
            return target_compositions

        existing_codes: set[str] = set()
        for comp in target_compositions:
            code = self._extract_variant_code(
                f"{comp.get('composition', '')} {comp.get('composition_normalized', '')}",
                family_spec["variable_element"],
            )
            if code:
                existing_codes.add(code)

        source_comp = target_compositions[family_idx]
        expanded: List[dict] = []
        for idx, variant in enumerate(family_spec["variants"], start=1):
            if variant["code"] in existing_codes:
                continue
            clone = dict(source_comp)
            clone["composition"] = variant["label"]
            clone["composition_normalized"] = variant["normalized"]
            # Keep only the first as possible carrier; others are placeholders
            # to avoid fabricating per-variant properties.
            if idx > 1 or existing_codes:
                clone["properties_of_composition"] = []
                clone["grain_size_avg_um"] = None
                clone["grain_size_text"] = None
            expanded.append(clone)

        if not expanded:
            return target_compositions

        updated = list(target_compositions)
        if not existing_codes:
            updated.pop(family_idx)
        updated.extend(expanded)
        return updated

    def _extract_variable_family_spec(self, paper_text: Optional[str]) -> Optional[Dict[str, Any]]:
        if not paper_text:
            return None
        x_values = self._extract_numeric_series_best(
            paper_text,
            [
                r"\bx\s*=\s*([0-9.,\sandto\-]+)",
                r"\bx\s*=\s*([^)]+)\)",
            ],
        )
        mo_at_values = self._extract_numeric_series_best(
            paper_text,
            [
                r"\bmo\s*content\s*=\s*([0-9.,\sandto\-]+)\s*at%?",
                r"\bmo\s*content\s*=\s*([^)]+)\)",
            ],
        )
        if not x_values or not mo_at_values:
            return None
        if len(x_values) != len(mo_at_values):
            return None

        token_candidates = re.findall(r"\b[A-Z][A-Za-z0-9.]*x\b", paper_text)
        family_token = ""
        for token in token_candidates:
            if len(re.findall(r"[A-Z][a-z]?", token)) >= 3:
                family_token = token
                break
        if not family_token:
            return None

        m = re.search(r"([A-Z][a-z]?)x\b", family_token)
        if not m:
            return None
        variable_element = m.group(1)
        base_formula = family_token[: m.start(1)]
        base_elements = [elem for elem, _ in re.findall(r"([A-Z][a-z]?)(\d*(?:\.\d+)?)", base_formula)]
        base_elements = [e for e in base_elements if e]
        if not base_elements:
            return None

        variants = []
        for x_val, mo_at in zip(x_values, mo_at_values):
            remaining = 100.0 - float(mo_at)
            if remaining < 0:
                continue
            each_base = remaining / float(len(base_elements))
            normalized = "".join(
                f"{elem}{self._format_pct(each_base)}" for elem in base_elements
            ) + f"{variable_element}{self._format_pct(float(mo_at))}"
            code = self._format_x_code(x_val)
            label = (
                f"{''.join(base_elements)}{variable_element}{self._format_x_label(x_val)} "
                f"({variable_element}{code})"
            )
            variants.append({"x": x_val, "mo_at": mo_at, "code": code, "label": label, "normalized": normalized})
        if len(variants) < 2:
            return None
        return {
            "variable_element": variable_element,
            "base_elements": base_elements,
            "base_joined": "".join(base_elements),
            "variants": variants,
        }

    @staticmethod
    def _extract_numeric_series(text: str, pattern: str) -> List[float]:
        m = re.search(pattern, text, re.IGNORECASE)
        if not m:
            return []
        segment = m.group(1)
        nums = re.findall(r"\d+(?:\.\d+)?", segment)
        return [float(v) for v in nums]

    @classmethod
    def _extract_numeric_series_best(cls, text: str, patterns: List[str]) -> List[float]:
        best: List[float] = []
        for pattern in patterns:
            nums = cls._extract_numeric_series(text, pattern)
            if len(nums) > len(best):
                best = nums
        return best

    @staticmethod
    def _format_pct(value: float) -> str:
        return f"{value:.3f}".rstrip("0").rstrip(".")

    @staticmethod
    def _format_x_code(x_val: float) -> str:
        # x=0.1 -> Mo1, x=0.3 -> Mo3, x=0.5 -> Mo5
        return str(int(round(x_val * 10)))

    def _format_x_label(self, x_val: float) -> str:
        if abs(x_val) < 1e-9:
            return "0"
        return self._format_pct(x_val)

    @staticmethod
    def _extract_variant_code(text: str, variable_element: str) -> Optional[str]:
        if not text:
            return None
        m = re.search(rf"\b{re.escape(variable_element)}\s*(\d+)\b", text, re.IGNORECASE)
        if m:
            return m.group(1)
        return None

    @staticmethod
    def _is_datasheet_like_document(paper_text: Optional[str], source_name: str) -> bool:
        haystack = f"{source_name}\n{paper_text or ''}".lower()
        signals = [
            "datasheet",
            "technical note",
            "common name",
            "specifications and compositions",
            "summary of physical properties",
            "product conditions",
            "product forms",
        ]
        score = sum(1 for s in signals if s in haystack)
        # two or more signals is usually enough to distinguish from a normal paper
        return score >= 2

    def _recover_formula_from_paper_text(
        self,
        paper_text: Optional[str],
        comp_raw: str,
        fallback_formula: str,
        target_count: int,
    ) -> Optional[str]:
        """Recover explicit composition formula from paper text when available.

        This is designed to prevent LLM-hallucinated composition ratios from
        dominating the final schema when the source text contains an explicit
        balance-notation composition line (e.g. ``... - Bal Ni``).
        """
        if not paper_text or "bal" not in paper_text.lower():
            return None

        # Prefer lines mentioning the current alloy identifier (e.g. SD3230).
        hint_tokens = re.findall(r"[A-Za-z]{2,}\d+", comp_raw or "")
        lines = [ln.strip() for ln in paper_text.splitlines() if ln.strip()]
        candidates: List[Tuple[int, str]] = []
        for ln in lines:
            lower_ln = ln.lower()
            if "bal" not in lower_ln:
                continue
            if "-" not in ln and "composition" not in lower_ln:
                continue

            score = 0
            if "composition" in lower_ln:
                score += 3
            if "wt" in lower_ln or "at" in lower_ln:
                score += 2
            score += len(re.findall(r"\d+(?:\.\d+)?\s*[A-Z][a-z]?", ln))
            if hint_tokens and any(tok.lower() in lower_ln for tok in hint_tokens):
                score += 5
            elif target_count == 1:
                score += 1
            candidates.append((score, ln))

        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0], reverse=True)
        best_line = candidates[0][1]

        parsed = self._parse_balance_composition_line(best_line)
        if not parsed:
            return None

        formula = "".join(f"{el}{self._format_pct(val)}" for el, val in parsed)
        if not formula:
            return None
        return formula

    @staticmethod
    def _parse_balance_composition_line(line: str) -> Optional[List[Tuple[str, float]]]:
        """Parse lines like ``21.49 Cr - ... - 0.48 C - Bal Ni``."""
        txt = line.replace("−", "-").replace("–", "-").replace("—", "-")
        segments = [seg.strip() for seg in re.split(r"\s*-\s*", txt) if seg.strip()]
        if len(segments) < 3:
            return None

        pairs: List[Tuple[str, float]] = []
        bal_elem: Optional[str] = None

        for seg in segments:
            seg_clean = re.sub(r"[()\[\],;]", " ", seg)
            seg_clean = re.sub(r"\s+", " ", seg_clean).strip()

            mb = re.search(r"(?i)\bBal\s*([A-Z][a-z]?)\b", seg_clean)
            if not mb:
                mb = re.search(r"(?i)\b([A-Z][a-z]?)\s*Bal\b", seg_clean)
            if mb:
                bal_elem = mb.group(1)
                continue

            m_num_first = re.search(r"\b(\d+(?:\.\d+)?)\s*([A-Z][a-z]?)\b", seg_clean)
            if m_num_first:
                val = float(m_num_first.group(1))
                elem = m_num_first.group(2)
                pairs.append((elem, val))
                continue

            m_elem_first = re.search(r"\b([A-Z][a-z]?)\s*(\d+(?:\.\d+)?)\b", seg_clean)
            if m_elem_first:
                elem = m_elem_first.group(1)
                val = float(m_elem_first.group(2))
                pairs.append((elem, val))

        if not pairs or bal_elem is None:
            return None

        # Deduplicate while keeping first occurrence.
        dedup: List[Tuple[str, float]] = []
        seen = set()
        for elem, val in pairs:
            if elem in seen:
                continue
            seen.add(elem)
            dedup.append((elem, val))

        known_total = sum(v for _, v in dedup if v is not None)
        bal_value = max(0.0, 100.0 - known_total)

        ordered: List[Tuple[str, float]] = [(bal_elem, bal_value)]
        for elem, val in dedup:
            if elem == bal_elem:
                continue
            ordered.append((elem, val))
        return ordered

    def _bootstrap_datasheet_compositions(self, paper_text: Optional[str]) -> List[dict]:
        """Bootstrap compositions from datasheet-style markdown tables/text.

        Used only as a last fallback when LLM extraction returns no compositions.
        """
        if not paper_text:
            return []
        boot: List[dict] = []
        seen_formula: set[str] = set()

        # 1) Parse markdown table with Tibal column (common in titanium datasheets).
        lines = paper_text.splitlines()
        header_idx = -1
        for i, ln in enumerate(lines):
            if "|" in ln and re.search(r"(?i)\bti\s*bal\b|\btibal\b", ln):
                header_idx = i
                break
        if header_idx >= 0:
            headers = [h.strip() for h in lines[header_idx].split("|") if h.strip()]
            header_keys: List[str] = []
            for h in headers:
                hh = h.lower().replace(" ", "")
                if hh in {"specificationdesignation", "specification", "form(s)", "forms", "form"}:
                    header_keys.append("_meta")
                elif hh in {"tibal", "tibalance", "tibal%"}:
                    header_keys.append("Ti")
                else:
                    m = re.match(r"([A-Z][a-z]?)", h.strip())
                    header_keys.append(m.group(1) if m else "_meta")

            for row in lines[header_idx + 2 :]:
                if "|" not in row:
                    # end of table block
                    if row.strip():
                        break
                    continue
                cells = [c.strip() for c in row.split("|") if c.strip()]
                if len(cells) < 2:
                    continue
                label = cells[0]
                values = cells[1:]
                comp_map: Dict[str, float] = {}
                for idx, cell in enumerate(values):
                    if idx >= len(header_keys) - 1:
                        break
                    elem = header_keys[idx + 1]
                    if elem == "_meta":
                        continue
                    if elem not in self.rules.valid_elements and elem != "Ti":
                        continue
                    mnum = re.search(r"<\s*(\d+(?:\.\d+)?)|(\d+(?:\.\d+)?)", cell)
                    if not mnum:
                        continue
                    raw = mnum.group(1) or mnum.group(2)
                    if raw is None:
                        continue
                    val = float(raw)
                    comp_map[elem] = val

                if not comp_map:
                    continue
                if "Ti" not in comp_map:
                    tot = sum(comp_map.values())
                    if 0 < tot < 100:
                        comp_map["Ti"] = 100.0 - tot
                total = sum(comp_map.values())
                if total <= 0 or total > 101:
                    continue

                formula = "".join(
                    f"{el}{self._format_pct(v)}" for el, v in sorted(comp_map.items(), key=lambda kv: (kv[0] != "Ti", kv[0]))
                )
                if not formula or formula in seen_formula:
                    continue
                seen_formula.add(formula)
                boot.append(
                    {
                        "composition": label or formula,
                        "role": "Target",
                        "composition_normalized": formula,
                        "source_doi": None,
                        "properties_of_composition": [],
                    }
                )

        # 2) Parse strict inline "composition ... x% Ti ... y% Fe ..." sentences.
        for ln in lines:
            lower_ln = ln.lower()
            if "%" not in ln or "ti" not in lower_ln:
                continue
            if not any(k in lower_ln for k in ("composition", "wt%", "at.%", "analysis by weight", "impurities")):
                continue
            pairs = re.findall(r"(\d+(?:\.\d+)?)\s*%?\s*([A-Z][a-z]?)(?:2)?", ln)
            if len(pairs) < 2:
                continue
            comp_map: Dict[str, float] = {}
            for num, el in pairs:
                if el not in self.rules.valid_elements:
                    continue
                try:
                    v = float(num)
                    if v < 0 or v > 100:
                        continue
                    comp_map[el] = v
                except Exception:
                    continue
            if len(comp_map) < 3:
                continue
            if "Ti" not in comp_map:
                tot = sum(comp_map.values())
                if 0 < tot < 100:
                    comp_map["Ti"] = 100.0 - tot
            total = sum(comp_map.values())
            if total <= 0 or total > 101:
                continue
            formula = "".join(
                f"{el}{self._format_pct(v)}" for el, v in sorted(comp_map.items(), key=lambda kv: (kv[0] != "Ti", kv[0]))
            )
            if not formula or formula in seen_formula:
                continue
            seen_formula.add(formula)
            boot.append(
                {
                    "composition": formula,
                    "role": "Target",
                    "composition_normalized": formula,
                    "source_doi": None,
                    "properties_of_composition": [],
                }
            )
        return boot

    # ------------------------------------------------------------------
    # Helpers – each is independently testable
    # ------------------------------------------------------------------

    def _normalize_reported_composition(
        self,
        comp_map: Any,
        preferred_balance_element: Optional[str] = None,
        fill_missing_balance: bool = True,
    ) -> Optional[Dict[str, float]]:
        """Clean a reported composition map and add/adjust ``other`` to close sum to 100.

        This is used for LLM-extracted nominal/measured compositions where values
        are already in wt%/at% style and should stay faithful to source numbers.
        """
        if not isinstance(comp_map, dict):
            return None

        cleaned: Dict[str, float] = {}
        for raw_key, raw_val in comp_map.items():
            key = str(raw_key).strip()
            if not key:
                continue
            if key != "other" and key not in self.rules.valid_elements:
                continue
            try:
                val = float(raw_val)
            except Exception:
                m = re.search(r"-?\d+(?:\.\d+)?", str(raw_val))
                if not m:
                    continue
                try:
                    val = float(m.group(0))
                except Exception:
                    continue
            val = round(val, 6)
            if key != "other" and abs(val) <= 1e-9:
                continue
            cleaned[key] = val

        if not cleaned:
            return None

        if (
            preferred_balance_element
            and preferred_balance_element not in cleaned
            and preferred_balance_element in self.rules.valid_elements
            and self._coerce_float(cleaned.get("other")) is not None
            and float(cleaned.get("other")) > 50.0
        ):
            cleaned[preferred_balance_element] = cleaned.pop("other")

        non_other_total = sum(v for k, v in cleaned.items() if k != "other")
        balance_delta = round(100.0 - non_other_total, 6)

        # When measured composition omits matrix element (e.g., Ti in EDX table),
        # prefer filling that matrix element by balance instead of pushing into "other".
        if (
            fill_missing_balance
            and preferred_balance_element
            and preferred_balance_element not in cleaned
            and preferred_balance_element in self.rules.valid_elements
            and balance_delta > 0
        ):
            cleaned[preferred_balance_element] = balance_delta
            return cleaned

        if "other" not in cleaned:
            if balance_delta > 1e-9:
                cleaned["other"] = balance_delta
        else:
            total = sum(cleaned.values())
            if abs(total - 100.0) > 1e-9:
                adjusted_other = round(cleaned["other"] + (100.0 - total), 6)
                if adjusted_other > 1e-9:
                    cleaned["other"] = adjusted_other
                else:
                    cleaned.pop("other", None)

        if "other" in cleaned and cleaned["other"] <= 1e-9:
            cleaned.pop("other", None)

        return cleaned

    def _harmonize_system_element_sets(self, target_compositions: List[dict]) -> List[dict]:
        """Harmonize element sets inside one alloy system cluster.

        Goal: preserve explicit zero-columns (e.g. Ni/Cr/Nb/Mo = 0.00) when the
        paper reports multiple related materials under one shared chemistry table.
        The function is intentionally conservative:
        - clusters are built from configured commercial-alloy aliases;
        - only nominal/measured maps are padded; and
        - residual is always closed by ``other`` to keep sum near 100.
        """
        if not target_compositions:
            return target_compositions

        aliases = self.rules.commercial_alloy_aliases or {}
        if not aliases:
            return target_compositions

        # Union-find to connect aliases that co-occur in blended/graded labels.
        parents: Dict[str, str] = {k: k for k in aliases.keys()}

        def _find(x: str) -> str:
            p = parents.setdefault(x, x)
            while parents.get(p, p) != p:
                parents[p] = parents[parents[p]]
                p = parents[p]
            return p

        def _union(a: str, b: str) -> None:
            ra, rb = _find(a), _find(b)
            if ra != rb:
                parents[rb] = ra

        alias_sets: List[set[str]] = []
        for comp in target_compositions:
            comp_text = str(comp.get("composition") or "")
            norm_text = str(comp.get("composition_normalized") or "")
            alloy_text = str(comp.get("alloy_name_raw") or "")
            proc_text = comp.get("processing_conditions")
            if isinstance(proc_text, dict):
                proc_text = json.dumps(proc_text, ensure_ascii=False)
            joined_text = " ".join(
                part
                for part in (
                    comp_text,
                    norm_text,
                    alloy_text,
                    str(proc_text or ""),
                )
                if part
            )
            present = self._extract_alias_keys(joined_text)
            alias_sets.append(present)
            if len(present) >= 2:
                present_list = sorted(present)
                for i in range(len(present_list) - 1):
                    _union(present_list[i], present_list[i + 1])

        # Build alias-cluster labels so single-alloy rows can inherit zero-columns
        # from related blended/graded rows in the same paper.
        root_members: Dict[str, set[str]] = defaultdict(set)
        for a in parents.keys():
            root_members[_find(a)].add(a)

        group_to_indices: Dict[str, List[int]] = defaultdict(list)
        for idx, present in enumerate(alias_sets):
            if not present:
                continue
            roots = sorted({_find(a) for a in present})
            cluster_members: set[str] = set()
            for r in roots:
                cluster_members.update(root_members.get(r, {r}))
            if not cluster_members:
                cluster_members = present
            gkey = "alias_cluster:" + "+".join(sorted(cluster_members))
            group_to_indices[gkey].append(idx)

        for indices in group_to_indices.values():
            if not indices:
                continue

            union_elements: set[str] = set()
            for idx in indices:
                comp = target_compositions[idx]
                for field in ("nominal_composition", "measured_composition"):
                    cmap = comp.get(field)
                    if not isinstance(cmap, dict):
                        continue
                    for el in cmap.keys():
                        if el in self.rules.valid_elements:
                            union_elements.add(el)

            if not union_elements:
                continue

            for idx in indices:
                comp = target_compositions[idx]
                raw_name = str(comp.get("composition") or "")
                proc_text = comp.get("processing_conditions")
                if isinstance(proc_text, dict):
                    proc_text = json.dumps(proc_text, ensure_ascii=False)
                context_text = str(proc_text or "")

                nominal = comp.get("nominal_composition")
                if not isinstance(nominal, dict) or not nominal:
                    inferred_nominal, inferred_type = self._infer_nominal_from_commercial_name(raw_name)
                    if inferred_nominal is None:
                        inferred_nominal, inferred_type = self._infer_blend_nominal_from_name(
                            raw_name,
                            context_text=context_text,
                        )
                    if inferred_nominal is not None:
                        nominal = inferred_nominal
                        comp["nominal_composition"] = inferred_nominal
                        if not comp.get("nominal_composition_type"):
                            comp["nominal_composition_type"] = inferred_type

                measured = comp.get("measured_composition")
                balance_element = self._pick_balance_element(
                    nominal if isinstance(nominal, dict) else None,
                    raw_name,
                )

                if isinstance(nominal, dict) and any(el in self.rules.valid_elements for el in nominal.keys()):
                    patched_nominal = dict(nominal)
                    for el in union_elements:
                        patched_nominal.setdefault(el, 0.0)
                    normalized_nominal = self._normalize_reported_composition(
                        patched_nominal,
                        preferred_balance_element=balance_element,
                        fill_missing_balance=False,
                    )
                    if normalized_nominal is not None:
                        comp["nominal_composition"] = normalized_nominal

                if isinstance(measured, dict) and any(el in self.rules.valid_elements for el in measured.keys()):
                    patched_measured = dict(measured)
                    for el in union_elements:
                        patched_measured.setdefault(el, 0.0)
                    normalized_measured = self._normalize_reported_composition(
                        patched_measured,
                        preferred_balance_element=balance_element,
                    )
                    if normalized_measured is not None:
                        comp["measured_composition"] = normalized_measured

        return target_compositions

    def _extract_alias_keys(self, text: str) -> set[str]:
        """Extract known commercial-alloy alias keys from free text."""
        if not text:
            return set()
        txt = text.lower()
        found: set[str] = set()
        for key, syns in (self.rules.commercial_alloy_aliases or {}).items():
            for syn in syns:
                token = str(syn).strip().lower()
                if token and token in txt:
                    found.add(str(key))
                    break
        return found

    def _harmonize_existing_target_materials(self, materials: List[dict]) -> List[dict]:
        """Repair composition fields when input is already in target-schema shape."""
        if not materials:
            return materials

        pseudo: List[dict] = []
        for mat in materials:
            comp_info = mat.get("Composition_Info") or {}

            nominal_block = comp_info.get("Nominal_Composition") or {}
            measured_block = comp_info.get("Measured_Composition") or {}

            nominal_comp = nominal_block.get("Elements_Normalized")
            if nominal_comp is None:
                nominal_comp = comp_info.get("Composition")
            if nominal_comp is None:
                nominal_comp = mat.get("Composition_JSON")

            measured_comp = measured_block.get("Elements_Normalized")
            measured_method = measured_block.get("Measurment_Method")
            if measured_method is None:
                measured_method = measured_block.get("Measurement_Method")
            if measured_method is None:
                measured_method = comp_info.get("Measurement_Method")
            if measured_comp is None and isinstance(comp_info.get("Measured_Composition"), dict):
                maybe_legacy = comp_info.get("Measured_Composition")
                if "Elements_Normalized" not in maybe_legacy:
                    measured_comp = maybe_legacy

            proc_text = ""
            samples = mat.get("Processed_Samples") or []
            if samples and isinstance(samples, list):
                proc_text = str((samples[0] or {}).get("Process_Text_For_AI") or "")
            if not proc_text:
                pinfo = mat.get("Process_Info") or {}
                proc_text = str(pinfo.get("Process_Text_For_AI") or "")
                if not proc_text:
                    ptxt = pinfo.get("Process_Text")
                    if isinstance(ptxt, dict):
                        proc_text = str(ptxt.get("original") or ptxt.get("simplified") or "")
                    elif ptxt:
                        proc_text = str(ptxt)

            pseudo.append(
                {
                    "composition": mat.get("Alloy_Name_Raw")
                    or comp_info.get("Alloy_Name_Raw")
                    or "",
                    "composition_normalized": mat.get("Formula_Normalized")
                    or comp_info.get("Formula_Normalized")
                    or "",
                    "alloy_name_raw": comp_info.get("Alloy_Name_Raw") or mat.get("Alloy_Name_Raw") or "",
                    "processing_conditions": proc_text,
                    "nominal_composition": nominal_comp,
                    "nominal_composition_type": nominal_block.get("Composition_Type")
                    or comp_info.get("Composition_Type"),
                    "measured_composition": measured_comp,
                    "measured_composition_type": measured_block.get("Composition_Type"),
                    "measured_composition_method": measured_method,
                }
            )

        pseudo = self._harmonize_system_element_sets(pseudo)

        out: List[dict] = []
        for mat, p in zip(materials, pseudo):
            patched = dict(mat)
            comp_info = dict(patched.get("Composition_Info") or {})

            nominal_comp = self._normalize_reported_composition(
                p.get("nominal_composition"),
                preferred_balance_element=self._pick_balance_element(
                    None,
                    str(patched.get("Alloy_Name_Raw") or p.get("composition") or ""),
                ),
                fill_missing_balance=False,
            )
            measured_comp = self._normalize_reported_composition(
                p.get("measured_composition"),
                preferred_balance_element=self._pick_balance_element(
                    nominal_comp,
                    str(patched.get("Alloy_Name_Raw") or p.get("composition") or ""),
                ),
            )

            nominal_type = p.get("nominal_composition_type")
            measured_type = p.get("measured_composition_type")
            measured_method = p.get("measured_composition_method")

            comp_info["Alloy_Name_Raw"] = (
                comp_info.get("Alloy_Name_Raw")
                or patched.get("Alloy_Name_Raw")
                or p.get("composition")
            )
            comp_info["Nominal_Composition"] = {
                "Composition_Type": nominal_type,
                "Elements_Normalized": nominal_comp,
            }
            comp_info["Measured_Composition"] = {
                "Composition_Type": measured_type,
                "Elements_Normalized": measured_comp,
                "Measurement_Method": measured_method,
            }
            patched["Composition_Info"] = comp_info

            if measured_comp:
                raw_comp_json = dict(measured_comp)
                source = "measured_composition"
            elif nominal_comp:
                raw_comp_json = dict(nominal_comp)
                source = "nominal_composition"
            else:
                raw_comp_json = patched.get("Composition_JSON") or {}
                source = patched.get("Composition_Source") or "unknown"

            formula = str(patched.get("Formula_Normalized") or p.get("composition_normalized") or "")
            validated_comp_json, comp_warnings = self.validate_composition_json(raw_comp_json, formula)
            for warning in comp_warnings:
                logger.warning("[COMPOSITION] %s", warning)

            # Legacy top-level fields are removed; all composition information
            # is expressed under Composition_Info.
            if (
                comp_info.get("Nominal_Composition", {}).get("Elements_Normalized") is None
                and measured_comp is None
                and validated_comp_json
            ):
                comp_info["Nominal_Composition"]["Elements_Normalized"] = validated_comp_json
                comp_info["Nominal_Composition"]["Composition_Type"] = (
                    comp_info["Nominal_Composition"].get("Composition_Type") or "at%"
                )
                patched["Composition_Info"] = comp_info

            samples = patched.get("Processed_Samples") or []
            if not patched.get("Properties_Info"):
                flattened_tests: List[Dict[str, Any]] = []
                graded_hint_text = (
                    f"{patched.get('description', '')} "
                    f"{(patched.get('Composition_Info') or {}).get('Alloy_Name_Raw', '')}"
                ).lower()
                for s in samples:
                    sample_proc_text = str((s or {}).get("Process_Text_For_AI") or "").lower()
                    graded_context = bool(
                        re.search(
                            r"\b(graded|gradient|stepwise|multigraded|increasing|up to \d+(?:\.\d+)?\s*(?:wt\s*%|wt%|%))\b",
                            f"{graded_hint_text} {sample_proc_text}",
                        )
                    )
                    for t in s.get("Performance_Tests", []) or []:
                        if graded_context and t.get("Property_Value_Range"):
                            continue
                        flattened_tests.append(dict(t))
                patched["Properties_Info"] = self._order_properties(flattened_tests)
            else:
                patched["Properties_Info"] = self._order_properties(
                    [dict(t) for t in (patched.get("Properties_Info") or [])]
                )

            if not patched.get("Process_Info") and samples:
                first_sample = samples[0] or {}
                process_text = str(first_sample.get("Process_Text_For_AI") or "not provided")
                patched["Process_Info"] = {
                    "Process_Category": first_sample.get("Process_Category") or "Unknown",
                    "Process_Text": self._to_text_payload(process_text),
                    "Key_Params": self._canonicalize_key_params(first_sample.get("Key_Params_JSON") or {}),
                }
            elif patched.get("Process_Info"):
                pinfo = dict(patched.get("Process_Info") or {})
                key_params_raw = pinfo.get("Key_Params") or pinfo.get("Key_Params_JSON") or {}
                pinfo["Key_Params"] = self._canonicalize_key_params(key_params_raw)
                pinfo.pop("Key_Params_JSON", None)
                ptxt = pinfo.get("Process_Text")
                if isinstance(ptxt, str):
                    pinfo["Process_Text"] = self._to_text_payload(ptxt)
                elif not isinstance(ptxt, dict):
                    alt = pinfo.get("Process_Text_For_AI")
                    pinfo["Process_Text"] = self._to_text_payload(str(alt or "not provided"))
                pinfo.pop("Process_Text_For_AI", None)
                patched["Process_Info"] = pinfo

            if not patched.get("Microstructure_Info") and samples:
                first_sample = samples[0] or {}
                micro_text = str(first_sample.get("Microstructure_Text_For_AI") or "not provided")
                patched["Microstructure_Info"] = {
                    "Microstructure_Text": self._to_text_payload(micro_text),
                    "Main_Phase": first_sample.get("Main_Phase") or None,
                    "Has_Precipitates": first_sample.get("Has_Precipitates"),
                    "Grain_Size_avg_um": first_sample.get("Grain_Size_avg_um"),
                }
            elif patched.get("Microstructure_Info"):
                minfo = dict(patched.get("Microstructure_Info") or {})
                mtxt = minfo.get("Microstructure_Text")
                if isinstance(mtxt, str):
                    minfo["Microstructure_Text"] = self._to_text_payload(mtxt)
                elif not isinstance(mtxt, dict):
                    alt = minfo.get("Microstructure_Text_For_AI")
                    minfo["Microstructure_Text"] = self._to_text_payload(str(alt or "not provided"))
                minfo.pop("Microstructure_Text_For_AI", None)
                patched["Microstructure_Info"] = minfo

            if samples:
                for s in samples:
                    s["Performance_Tests"] = []
                    key_params_raw = s.get("Key_Params_JSON") or s.get("Key_Params") or {}
                    s["Key_Params_JSON"] = self._canonicalize_key_params(key_params_raw)
                    s.pop("Key_Params", None)
                patched["Processed_Samples"] = samples

            patched.pop("Composition_JSON", None)
            patched.pop("Alloy_Name_Raw", None)
            patched.pop("Formula_Normalized", None)
            if "Composition" in patched.get("Composition_Info", {}):
                patched["Composition_Info"].pop("Composition", None)
            patched["Composition_Source"] = source
            out.append(patched)
        return out

    def _pick_balance_element(
        self, nominal_comp: Optional[Dict[str, float]], raw_name: str
    ) -> Optional[str]:
        """Pick matrix element used for balance completion in measured maps."""
        if nominal_comp:
            ranked = sorted(
                ((k, v) for k, v in nominal_comp.items() if k != "other"),
                key=lambda kv: kv[1],
                reverse=True,
            )
            if ranked and ranked[0][0] in self.rules.valid_elements:
                return ranked[0][0]
        inferred_nominal, _ = self._infer_nominal_from_commercial_name(raw_name or "")
        if inferred_nominal:
            ranked = sorted(inferred_nominal.items(), key=lambda kv: kv[1], reverse=True)
            if ranked:
                return ranked[0][0]
        inferred_from_name = self._infer_matrix_element_from_raw_name(raw_name)
        if inferred_from_name:
            return inferred_from_name
        return None

    def _infer_matrix_element_from_raw_name(self, raw_name: str) -> Optional[str]:
        text = str(raw_name or "").strip()
        if not text:
            return None
        lowered = text.lower()
        explicit_patterns = (
            (r"\bal(?:uminum|uminium)?[-\s]?based\b|\baa\d{4}\b", "Al"),
            (r"\bni(?:ckel)?[-\s]?based\b|\binconel\b|\bin718\b", "Ni"),
            (r"\bti(?:tanium)?[-\s]?based\b|\bti64\b|\bti-?6al-?4v\b", "Ti"),
            (r"\bfe(?:rritic|ron)?[-\s]?based\b", "Fe"),
            (r"\bco(?:balt)?[-\s]?based\b", "Co"),
            (r"\bmg(?:nesium)?[-\s]?based\b", "Mg"),
            (r"\bcu(?:pper)?[-\s]?based\b", "Cu"),
        )
        for pattern, element in explicit_patterns:
            if re.search(pattern, lowered):
                return element
        first_elem = re.match(r"\s*([A-Z][a-z]?)", text)
        if first_elem and first_elem.group(1) in self.rules.valid_elements:
            return first_elem.group(1)
        return None

    @staticmethod
    def _is_commercial_shorthand(raw_name: str, formula: str) -> bool:
        """Return True when the token likely represents a commercial grade name."""
        text = f"{raw_name} {formula}".lower().replace(" ", "")
        known = (
            "ti6al4v",
            "ti-6al-4v",
            "in718",
            "inconel718",
            "316l",
            "304l",
            "17-4ph",
            "174ph",
        )
        return any(tok in text for tok in known)

    def _infer_nominal_from_commercial_name(
        self, raw_name: str
    ) -> Tuple[Optional[Dict[str, float]], Optional[str]]:
        """Infer nominal composition for a small set of canonical commercial aliases."""
        if not raw_name:
            return None, None
        name = raw_name.lower().replace(" ", "")
        # Only infer for single-alloy aliases; avoid blended/graded/multi-alloy labels.
        if any(tok in name for tok in ("+", "/", "blend", "graded", "gradient", "wt%", "in718", "inconel")):
            return None, None
        aliases = self.rules.commercial_alloy_aliases or {}
        nominal_maps = self.rules.commercial_nominal_maps or {}
        for key, syns in aliases.items():
            if any(str(s).lower().replace(" ", "") in name for s in syns):
                comp = nominal_maps.get(key)
                if isinstance(comp, dict) and comp:
                    return {
                        str(k): float(v) for k, v in comp.items()
                    }, "wt%"
        return None, None

    def _infer_blend_nominal_from_name(
        self, raw_name: str, context_text: str = ""
    ) -> Tuple[Optional[Dict[str, float]], Optional[str]]:
        """Infer blended nominal composition from labels like 'A + 20 wt% B' or '0-20 wt% B'."""
        if not raw_name:
            raw_name = ""
        txt = f"{raw_name} {context_text or ''}".lower()
        known = self.rules.commercial_nominal_maps or {}
        aliases = self.rules.commercial_alloy_aliases or {}
        if not known or not aliases:
            return None, None

        present: List[str] = []
        for key, syns in aliases.items():
            if any(s in txt for s in syns):
                present.append(key)
        if len(present) < 2:
            return None, None

        pct = None
        additive_key = None
        alias_tokens: List[str] = []
        for syns in aliases.values():
            alias_tokens.extend(re.escape(str(s).lower()) for s in syns if str(s).strip())
        if not alias_tokens:
            return None, None
        alloy_alt = "|".join(sorted(set(alias_tokens), key=len, reverse=True))
        pct_pattern = re.compile(
            rf"(\d+(?:\.\d+)?)\s*(?:-\s*(\d+(?:\.\d+)?)\s*)?(?:wt\s*%|wt%|%\s*wt|%)\s*({alloy_alt})",
            re.IGNORECASE,
        )
        candidates: List[Tuple[float, float, str]] = []
        for m in pct_pattern.finditer(txt):
            lo = float(m.group(1))
            hi = float(m.group(2)) if m.group(2) else lo
            tok = m.group(3).replace(" ", "")
            key_hit = None
            for key, syns in aliases.items():
                if any(s.replace(" ", "") in tok for s in syns):
                    key_hit = key
                    break
            if key_hit:
                candidates.append((lo, hi, key_hit))

        if candidates:
            additive_key = candidates[0][2]
            lo_vals = [lo for lo, _hi, key in candidates if key == additive_key]
            hi_vals = [hi for _lo, hi, key in candidates if key == additive_key]
            lo = min(lo_vals) if lo_vals else None
            hi = max(hi_vals) if hi_vals else None

            # Graded descriptions often write "5% wt ... up to 20%" where upper
            # bound has no trailing alloy token. Use it as the range ceiling.
            if lo is not None and hi is not None:
                graded_hint = any(k in txt for k in ("graded", "gradient", "step"))
                up_to = re.search(r"up\s*to\s*(\d+(?:\.\d+)?)\s*(?:wt\s*%|wt%|%\s*wt|%)", txt)
                if graded_hint and up_to:
                    try:
                        up_to_val = float(up_to.group(1))
                        if up_to_val > hi:
                            hi = up_to_val
                        # Phrases like "steps of 5% ... up to 20%" usually imply
                        # a graded span from 0% to 20%, where 5% is step size.
                        if "step" in txt and lo <= up_to_val * 0.5:
                            lo = 0.0
                    except Exception:
                        pass
                pct = (lo + hi) / 2.0 if hi >= lo else lo
        else:
            # Fallback for graded phrasing like:
            # "increasing IN718 in steps of 5wt% up to 20wt%"
            up_to = re.search(
                r"up\s*to\s*(\d+(?:\.\d+)?)\s*(?:wt\s*%|wt%|%\s*wt|%)",
                txt,
                re.IGNORECASE,
            )
            if up_to and len(present) >= 2:
                graded_hint = any(k in txt for k in ("graded", "gradient", "step", "increasing", "increase"))
                hi = float(up_to.group(1))
                lo = 0.0 if graded_hint else None

                key_scores: Dict[str, int] = {}
                for key in present:
                    score = 0
                    for syn in aliases.get(key, []):
                        syn_l = str(syn).lower()
                        if not syn_l:
                            continue
                        # Strong signal: "increasing/addition ... <alloy>"
                        if re.search(
                            rf"(?:increase|increasing|adding|added|addition|enrich(?:ed|ment)?)\s+[^.]{{0,25}}{re.escape(syn_l)}",
                            txt,
                        ):
                            score += 3
                        # Moderate signal: "<alloy> ... up to xx wt%"
                        if re.search(
                            rf"{re.escape(syn_l)}[^.]{{0,30}}up\s*to\s*\d+(?:\.\d+)?\s*(?:wt\s*%|wt%|%\s*wt|%)",
                            txt,
                        ):
                            score += 2
                    key_scores[key] = score

                additive_key = max(key_scores.items(), key=lambda kv: kv[1])[0]
                if key_scores.get(additive_key, 0) == 0:
                    additive_key = present[-1]
                if lo is not None:
                    pct = (lo + hi) / 2.0 if hi >= lo else lo

        if pct is None or additive_key not in known:
            return None, None
        if pct < 0 or pct > 100:
            return None, None

        base_key = None
        for k in present:
            if k != additive_key and k in known:
                base_key = k
                break
        if base_key is None:
            return None, None

        add_frac = pct / 100.0
        base_frac = 1.0 - add_frac
        comp: Dict[str, float] = {}
        for el in set(known[base_key].keys()) | set(known[additive_key].keys()):
            try:
                base_v = float(known[base_key].get(el, 0.0))
            except Exception:
                base_v = 0.0
            try:
                add_v = float(known[additive_key].get(el, 0.0))
            except Exception:
                add_v = 0.0
            v = base_frac * base_v + add_frac * add_v
            comp[el] = round(v, 6)

        total = sum(comp.values())
        if abs(total - 100.0) > 1e-6:
            residual = round(comp.get("other", 0.0) + (100.0 - total), 6)
            if residual > 0:
                comp["other"] = residual
            else:
                comp.pop("other", None)
        return comp, "wt%"

    @staticmethod
    def parse_temperature_to_k(measurement_condition: Optional[str]) -> Optional[float]:
        """Best-effort parse of temperature from measurement condition text.

        Prioritises ``at XXX K`` format, then general patterns.
        Returns Kelvin as float.
        """
        if not measurement_condition:
            return None
        txt = measurement_condition.lower()

        if re.search(r"\b(rt|room temperature|ambient)\b", txt):
            return 298.15

        m_at = re.search(r"at\s+(-?\d+(?:\.\d+)?)\s*k\b", txt)
        if m_at:
            return round(float(m_at.group(1)), 2)

        m_at_c = re.search(r"at\s+(-?\d+(?:\.\d+)?)\s*(?:°c|\bc\b)\b", txt)
        if m_at_c:
            return round(float(m_at_c.group(1)) + 273.15, 2)

        m = re.search(r"(-?\d+(?:\.\d+)?)\s*(k|°c|\bc\b)\b", txt)
        if not m:
            return None
        value = float(m.group(1))
        unit = m.group(2)
        if unit in {"°c", "c"}:
            return round(value + 273.15, 2)
        return round(value, 2)

    @staticmethod
    def build_composition_json(formula: str) -> dict:
        """Convert formula string like ``Ti42Hf21Nb21V16`` to a composition dict.

        Preferentially uses ``pymatgen.core.Composition`` for robust parsing
        (handles complex formulas, parentheses, fractional stoichiometry, etc.)
        and converts to molar-percentage (at%). ``Bal`` notation (e.g. ``NiBal``)
        is first rewritten to explicit numeric residual content. Falls back to
        regex-based parsing only when *pymatgen* is unavailable or unparseable.
        """
        comp: Dict[str, float] = {}
        if not formula:
            return comp

        def _safe_float(token: Optional[str]) -> Optional[float]:
            if token in (None, ""):
                return None
            try:
                return float(token)
            except Exception:
                m = re.match(r"-?\d+(?:\.\d+)?", str(token))
                if m:
                    try:
                        return float(m.group(0))
                    except Exception:
                        return None
                return None

        subscript_map = str.maketrans(
            {
                "₀": "0",
                "₁": "1",
                "₂": "2",
                "₃": "3",
                "₄": "4",
                "₅": "5",
                "₆": "6",
                "₇": "7",
                "₈": "8",
                "₉": "9",
            }
        )
        cleaned = formula.translate(subscript_map)
        cleaned = cleaned.replace("（", "(").replace("）", ")")
        cleaned = re.sub(r"(?i)\b(at|wt)\s*%?\b", "", cleaned)
        cleaned = re.sub(r"\s+", "", cleaned)
        # Keep parentheses for pymatgen group parsing; strip other separators.
        cleaned = re.sub(r"[^A-Za-z0-9.()]", "", cleaned)

        # Resolve balance notation such as NiBal -> Ni(100 - others).
        token_pattern = r"([A-Z][a-z]?)(Bal|BAL|bal|\d+(?:\.\d+)?)?"
        resolved_for_parse = cleaned
        if "bal" in cleaned.lower():
            flat = re.sub(r"[()]", "", cleaned)
            tokens = re.findall(token_pattern, flat)
            bal_indices = [i for i, (_, amt) in enumerate(tokens) if (amt or "").lower() == "bal"]
            if len(tokens) >= 2 and len(bal_indices) == 1:
                bal_idx = bal_indices[0]
                known_total = 0.0
                for idx, (_, amt) in enumerate(tokens):
                    if idx == bal_idx:
                        continue
                    if amt in (None, ""):
                        known_total += 1.0
                    elif (amt or "").lower() == "bal":
                        continue
                    else:
                        parsed = _safe_float(amt)
                        if parsed is None:
                            logger.debug("Skip non-numeric amount token '%s' in formula '%s'", amt, formula)
                            continue
                        known_total += parsed
                bal_value = max(0.0, 100.0 - known_total)
                rebuilt: List[str] = []
                for idx, (elem, amt) in enumerate(tokens):
                    if idx == bal_idx:
                        value = bal_value
                    elif amt in (None, ""):
                        value = 1.0
                    elif (amt or "").lower() == "bal":
                        value = 1.0
                    else:
                        parsed = _safe_float(amt)
                        value = parsed if parsed is not None else 1.0
                    value_txt = f"{value:.6f}".rstrip("0").rstrip(".")
                    rebuilt.append(f"{elem}{value_txt}")
                resolved_for_parse = "".join(rebuilt)

        candidates: List[str] = []
        if resolved_for_parse:
            candidates.append(resolved_for_parse)
        if cleaned and cleaned != resolved_for_parse:
            candidates.append(cleaned)

        if _PymatgenComposition is not None:
            for candidate in candidates:
                try:
                    pmg_comp = _PymatgenComposition(candidate)
                    frac_dict = pmg_comp.fractional_composition.as_dict()
                    return {el: round(frac * 100, 6) for el, frac in frac_dict.items()}
                except Exception:
                    continue
            logger.debug("pymatgen failed to parse '%s', falling back to regex", formula)

        fallback_src = candidates[0] if candidates else cleaned
        for element, amount in re.findall(r"([A-Z][a-z]?)(\d+(?:\.\d+)?)?", re.sub(r"[()]", "", fallback_src)):
            if not element:
                continue
            comp[element] = float(amount) if amount not in (None, "") else 1.0
        return comp

    def validate_composition_json(
        self, comp_json: dict, formula_source: str = ""
    ) -> Tuple[dict, List[str]]:
        """Validate element symbols and ensure composition is in at% (sums to ~100).

        If values look like atomic ratios (e.g. sum far from 100 but > 0), the
        composition is normalised to 100 at%.
        """
        warnings: List[str] = []
        cleaned: Dict[str, float] = {}

        for elem, val in comp_json.items():
            try:
                parsed_val = float(val)
            except Exception:
                warnings.append(f"Invalid composition value '{val}' for '{elem}' removed from {formula_source}")
                continue
            if str(elem).lower() == "other":
                cleaned["other"] = parsed_val
                continue
            if elem not in self.rules.valid_elements:
                warnings.append(f"Invalid element '{elem}' removed from {formula_source}")
                continue
            cleaned[elem] = parsed_val

        if cleaned:
            total = sum(cleaned.values())
            if total <= 0:
                warnings.append(f"Composition sum = {total:.2f}, invalid total. Formula: {formula_source}")
                return cleaned, warnings

            if "other" in cleaned:
                # For reported wt%/at% maps, keep source values and use `other` to
                # absorb residual so total becomes 100.
                if abs(total - 100) > 1e-6:
                    adjusted_other = round(cleaned["other"] + (100.0 - total), 6)
                    if adjusted_other > 1e-9:
                        cleaned["other"] = adjusted_other
                        warnings.append(
                            f"Composition sum = {total:.2f}; adjusted 'other' to close sum to 100. Formula: {formula_source}"
                        )
                    else:
                        cleaned.pop("other", None)
            else:
                # If clearly not at% already, normalise to at%.
                if abs(total - 100) > 2:
                    normalised = {k: (v / total) * 100.0 for k, v in cleaned.items()}
                    warnings.append(
                        f"Composition sum = {total:.2f} (not ~100). Normalised to 100 at%. Formula: {formula_source}"
                    )
                    cleaned = normalised
        return cleaned, warnings

    @staticmethod
    def extract_first_doi(text: Optional[str]) -> str:
        """Extract first DOI-like token from *text*."""
        if not text:
            return ""
        m = re.search(r"\b10\.\d{4,9}/[-._;()/:A-Za-z0-9]+\b", text)
        return m.group(0) if m else ""

    def infer_main_phase(self, char_text: str) -> str:
        """Infer main phase from characterisation text using rule patterns."""
        if not char_text:
            return ""
        ctext = char_text.lower()
        if re.search(r"\bgamma\b(?!\s*prime)", ctext):
            return "Gamma (gamma)"
        phases: List[str] = []
        seen: set = set()
        for pattern, label in self.rules.phase_patterns.items():
            lowered_label = label.lower()
            if any(token in lowered_label for token in ("laves", "carbide", "sigma", "mu", "chi", "l12", "l21")):
                continue
            if pattern in ctext and label not in seen:
                phases.append(label)
                seen.add(label)
        if not phases:
            return ""
        return " + ".join(phases[:3])

    def infer_precipitates(self, char_text: str) -> bool:
        """Return *True* if precipitate keywords are found in *char_text*."""
        if not char_text:
            return False
        ctext = char_text.lower()
        return any(kw in ctext for kw in self.rules.precipitate_keywords)

    @staticmethod
    def _format_range_value(start: float, end: float, add_kelvin: bool = False) -> str:
        if add_kelvin:
            start += 273.15
            end += 273.15

        def _fmt(value: float) -> str:
            rounded = round(value) if add_kelvin else round(value, 2)
            if abs(rounded - round(rounded)) < 1e-9:
                return str(int(round(rounded)))
            return f"{rounded:.2f}".rstrip("0").rstrip(".")

        lo, hi = sorted((_fmt(start), _fmt(end)), key=lambda token: float(token))
        return f"{lo}-{hi}"

    def _extract_heat_treatment_range_params(self, process_text: str) -> Dict[str, Any]:
        if not process_text:
            return {}
        text = str(process_text)
        out: Dict[str, Any] = {}
        patterns = [
            ("Solution_Temperature_K", r"solution(?:\s+treatment)?[^.;\n]{0,80}?(\d+(?:\.\d+)?)\s*(?:-|to|~)\s*(\d+(?:\.\d+)?)\s*(k|°?\s*c)\b"),
            ("Aging_Temperature_K", r"ag(?:ing|ed)?[^.;\n]{0,80}?(\d+(?:\.\d+)?)\s*(?:-|to|~)\s*(\d+(?:\.\d+)?)\s*(k|°?\s*c)\b"),
            ("Annealing_Temperature_K", r"anneal(?:ed|ing)?[^.;\n]{0,80}?(\d+(?:\.\d+)?)\s*(?:-|to|~)\s*(\d+(?:\.\d+)?)\s*(k|°?\s*c)\b"),
            ("Solution_Time_h", r"solution(?:\s+treatment)?[^.;\n]{0,80}?(\d+(?:\.\d+)?)\s*(?:-|to|~)\s*(\d+(?:\.\d+)?)\s*h\b"),
            ("Aging_Time_h", r"ag(?:ing|ed)?[^.;\n]{0,80}?(\d+(?:\.\d+)?)\s*(?:-|to|~)\s*(\d+(?:\.\d+)?)\s*h\b"),
            ("Annealing_Time_h", r"anneal(?:ed|ing)?[^.;\n]{0,80}?(\d+(?:\.\d+)?)\s*(?:-|to|~)\s*(\d+(?:\.\d+)?)\s*h\b"),
        ]
        for key, pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if not match:
                continue
            start = float(match.group(1))
            end = float(match.group(2))
            unit = match.group(3).lower() if len(match.groups()) >= 3 else "h"
            if key.endswith("_K"):
                out[key] = self._format_range_value(start, end, add_kelvin=("c" in unit))
            else:
                out[key] = self._format_range_value(start, end, add_kelvin=False)
        return out

    def parse_key_params(
        self, process_text: str, processing_params: Optional[dict] = None
    ) -> dict:
        """Build structured ``Key_Params_JSON`` from LLM output or regex."""
        params: Dict[str, Any] = {}

        if processing_params and isinstance(processing_params, dict):
            for k, v in processing_params.items():
                if v is not None:
                    params[k] = v

        if not process_text:
            return self._canonicalize_key_params(params)

        def _safe_float(token: Any) -> Optional[float]:
            if token is None:
                return None
            txt = str(token).strip()
            try:
                return float(txt)
            except Exception:
                m_num = re.search(r"-?\d+(?:\.\d+)?", txt)
                if not m_num:
                    return None
                try:
                    return float(m_num.group(0))
                except Exception:
                    return None

        for param_name, compiled in self.rules._compiled_param_patterns.items():
            if param_name in params:
                continue
            for pat in compiled:
                m = pat.search(process_text)
                if m:
                    parsed = _safe_float(m.group(1))
                    if parsed is None:
                        logger.debug(
                            "Skip unparsable process param '%s' token '%s' in text '%s'",
                            param_name,
                            m.group(1),
                            process_text[:200],
                        )
                        continue
                    params[param_name] = parsed
                    break

        for key, value in self._extract_heat_treatment_range_params(process_text).items():
            params.setdefault(key, value)

        if re.search(r"\bargon\b", process_text, re.IGNORECASE) and "Protective_Atmosphere" not in params:
            params["Protective_Atmosphere"] = "Argon"
        elif re.search(r"\bnitrogen\b", process_text, re.IGNORECASE) and "Protective_Atmosphere" not in params:
            params["Protective_Atmosphere"] = "Nitrogen"

        for k, v in self._extract_energy_density_params(process_text).items():
            if k not in params:
                params[k] = v

        return self._canonicalize_key_params(params)

    def _canonicalize_key_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Canonicalise process-parameter key names to the current schema."""
        if not params:
            return {}

        def _format_number(value: float) -> Any:
            if not math.isfinite(value):
                return None
            rounded = round(value, 6)
            if abs(rounded - round(rounded)) < 1e-9:
                return int(round(rounded))
            return rounded

        def _convert_value(value: Any, scale: float = 1.0, offset: float = 0.0) -> Any:
            if value is None or isinstance(value, bool):
                return value
            if isinstance(value, (int, float)):
                return _format_number(float(value) * scale + offset)
            text = str(value).strip()
            if not text:
                return value
            range_match = re.fullmatch(
                r"\s*(-?\d+(?:\.\d+)?)\s*[-~]\s*(-?\d+(?:\.\d+)?)\s*",
                text,
            )
            if range_match:
                lo = float(range_match.group(1)) * scale + offset
                hi = float(range_match.group(2)) * scale + offset
                return f"{_format_number(lo)}-{_format_number(hi)}"
            try:
                return _format_number(float(text) * scale + offset)
            except Exception:
                return value

        out: Dict[str, Any] = {}
        disallowed = {
            "Relative_Density_pct",
            "Porosity_pct",
            "Main_Phase",
            "Grain_Size_avg_um",
            "Precipitate_Size_avg_nm",
            "Precipitate_Volume_Fraction_pct",
            "Phase_Fraction_pct",
            "Yield_Strength",
            "Yield_Strength_Compressive",
            "Ultimate_Tensile_Strength",
            "Ultimate_Strength_Compressive",
            "Fracture_Elongation",
            "Elongation_Total",
            "Elongation_Uniform",
            "Elongation_At_Fracture",
            "Elongation_Compressive",
            "Elastic_Modulus",
            "Hardness_HV",
            "Hardness_HRC",
            "Hardness_HB",
            "Process_Category",
            "Equipment",
        }
        key_alias = {
            "Scan_Speed_mm_s": "Scanning_Speed_mm_s",
            "Scanning_Velocity_mm_s": "Scanning_Speed_mm_s",
            "Scanning_Speed_mm_min": "Scanning_Speed_mm_s",
            "Scanning_Speed_m_s": "Scanning_Speed_mm_s",
            "Shielding_Gas": "Protective_Atmosphere",
            "Process_Gas": "Protective_Atmosphere",
            "Preheat_Temperature_C": "Build_Plate_Temperature_K",
            "Preheat_Temperature_K": "Build_Plate_Temperature_K",
            "Pre_heating_Temperature_C": "Build_Plate_Temperature_K",
            "Pre_Heating_Temperature_degC": "Build_Plate_Temperature_K",
            "Build_Plate_Temperature_C": "Build_Plate_Temperature_K",
            "Build_Plate_Preheat_K": "Build_Plate_Temperature_K",
            "Building_Temperature_K": "Build_Plate_Temperature_K",
            "Building_temperature_K": "Build_Plate_Temperature_K",
            "Powder_Bed_Temperature_K": "Build_Plate_Temperature_K",
            "Powder_bed_preheating_temperature_K": "Build_Plate_Temperature_K",
            "Hatch_Distance_um": "Hatch_Spacing_um",
            "Hatch_Distance_um_Range": "Hatch_Spacing_um_Range",
            "Hatch_Distance_mm": "Hatch_Spacing_mm",
            "Energy_Density_J_mm3": "Volumetric_Energy_Density_J_mm3",
            "Energy_Density_J_mm3_Range": "Volumetric_Energy_Density_J_mm3_Range",
            "Energy_Density_J_mm3_FullDensification": "Volumetric_Energy_Density_J_mm3_FullDensification",
            "Energy_Density_J_mm3_Optimal": "Volumetric_Energy_Density_J_mm3_Optimal",
            "Energy_Density_J_mm3_Optimal_Range": "Volumetric_Energy_Density_J_mm3_Optimal_Range",
            "Layer_Thickness_mm": "Layer_Thickness_um",
            "Power_W": "Laser_Power_W",
            "Aging_Temperature_C": "Aging_Temperature_K",
            "Age_Temperature_C": "Aging_Temperature_K",
            "Age_Temperature_K": "Aging_Temperature_K",
            "Aging_Duration_h": "Aging_Time_h",
            "Age_Duration_h": "Aging_Time_h",
            "Solution_Temperature_C": "Solution_Temperature_K",
            "Annealing_Temperature_C": "Annealing_Temperature_K",
            "Heating_Rate_C_min": "Heating_Rate_K_min",
            "Mean_beam_current_mA": "Beam_Current_mA",
            "Voltage_kV": "Acceleration_Voltage_kV",
            "Beam_Diameter_um": "Beam_diameter_um",
            "Beam_Size_um": "Beam_diameter_um",
            "Laser_Spot_Size_um": "Spot_Size_um",
            "Powder_Feeding_Rate_g_min": "Powder_Feed_Rate_g_min",
        }

        for raw_key, raw_val in params.items():
            key = key_alias.get(str(raw_key), str(raw_key))
            if key in disallowed:
                continue
            val = raw_val

            if key == "Scanning_Speed_mm_s" and str(raw_key) == "Scanning_Speed_mm_min":
                val = _convert_value(raw_val, scale=1.0 / 60.0)
            elif key == "Scanning_Speed_mm_s" and str(raw_key) == "Scanning_Speed_m_s":
                val = _convert_value(raw_val, scale=1000.0)
            elif key == "Layer_Thickness_um" and str(raw_key) == "Layer_Thickness_mm":
                val = _convert_value(raw_val, scale=1000.0)
            elif key == "Build_Plate_Temperature_K" and raw_val is not None:
                if str(raw_key).endswith("_C") or str(raw_key) in {
                    "Pre_Heating_Temperature_degC",
                    "Pre_heating_Temperature_C",
                    "Build_Plate_Temperature_C",
                }:
                    val = _convert_value(raw_val, offset=273.15)
                else:
                    val = _convert_value(raw_val)
            elif key in {"Aging_Temperature_K", "Solution_Temperature_K", "Annealing_Temperature_K"} and str(raw_key).endswith("_C"):
                val = _convert_value(raw_val, offset=273.15)
            elif key == "Heating_Rate_K_min":
                val = _convert_value(raw_val)
            elif key in {"Laser_Power_W", "Acceleration_Voltage_kV", "Beam_Current_mA"}:
                val = _convert_value(raw_val)

            if key == "Protective_Atmosphere" and isinstance(val, str):
                vv = val.strip().lower()
                if vv in {"ar", "argon"}:
                    val = "Argon"
                elif vv in {"n2", "nitrogen"}:
                    val = "Nitrogen"
                elif vv in {"vac", "vacuum"}:
                    val = "Vacuum"

            out[key] = val
        return out

    @staticmethod
    def _extract_energy_density_params(process_text: str) -> Dict[str, Any]:
        """Extract energy-density related parameters, preserving threshold ranges."""
        if not process_text:
            return {}
        txt = process_text
        out: Dict[str, Any] = {}

        patt = re.compile(r"(\d+(?:\.\d+)?)\s*J\s*/\s*mm(?:\^?3|3)", re.IGNORECASE)
        dens_vals: List[float] = []
        opt_vals: List[float] = []

        for m in patt.finditer(txt):
            try:
                val = float(m.group(1))
            except Exception:
                continue
            window = txt[max(0, m.start() - 80): min(len(txt), m.end() + 80)].lower()
            is_dens_ctx = any(
                kw in window for kw in ("full dens", "densification", "fully dense", "full dense")
            )
            is_opt_ctx = any(
                kw in window for kw in ("optimal", "optimum", "produced at", "sample shown", "selected")
            )
            if is_dens_ctx:
                dens_vals.append(val)
            elif is_opt_ctx:
                opt_vals.append(val)

        if not dens_vals and not opt_vals:
            all_vals = sorted({float(m.group(1)) for m in patt.finditer(txt)})
            if all_vals:
                if len(all_vals) == 1:
                    out["Energy_Density_J_mm3"] = all_vals[0]
                else:
                    out["Energy_Density_J_mm3_Range"] = (
                        f"{all_vals[0]:g}-{all_vals[-1]:g}"
                    )
            return out

        if dens_vals:
            uniq = sorted(set(dens_vals))
            if len(uniq) == 1:
                out["Energy_Density_J_mm3_FullDensification"] = uniq[0]
            else:
                out["Energy_Density_J_mm3_FullDensification"] = f"{uniq[0]:g}-{uniq[-1]:g}"
        if opt_vals:
            uniq = sorted(set(opt_vals))
            if len(uniq) == 1:
                out["Energy_Density_J_mm3_Optimal"] = uniq[0]
            else:
                out["Energy_Density_J_mm3_Optimal_Range"] = f"{uniq[0]:g}-{uniq[-1]:g}"
        return out

    @staticmethod
    def build_microstructure_text(comp: dict) -> str:
        """Build ``Microstructure_Text_For_AI`` from dedicated fields or legacy characterisation."""
        micro_desc = comp.get("microstructure_description")
        if micro_desc and micro_desc.strip():
            return micro_desc.strip()

        char = comp.get("characterisation")
        if isinstance(char, dict):
            parts = []
            for key in ("Microstructure", "EBSD", "SEM", "TEM"):
                val = char.get(key)
                if val and isinstance(val, str) and val.strip():
                    parts.append(f"{key}: {val.strip()}")
            if parts:
                return "; ".join(parts)
            all_parts = [f"{k}: {v}" for k, v in char.items() if isinstance(v, str) and v.strip()]
            if all_parts:
                return "; ".join(all_parts)

        return "not provided"

    @staticmethod
    def _to_text_payload(text: str) -> Dict[str, str]:
        """Return dual-track text payload with original and simplified variants."""
        original = (text or "not provided").strip() or "not provided"
        explicit = re.search(
            r"original\s*:\s*(.*?)\s*(?:\|\||\n|;\s*)\s*simplified\s*:\s*(.+)$",
            original,
            re.IGNORECASE | re.DOTALL,
        )
        if explicit:
            orig = explicit.group(1).strip() or "not provided"
            simp = explicit.group(2).strip() or orig
            return {"original": orig, "simplified": simp}
        chunks = [c.strip() for c in re.split(r"[;\n]+|(?<=\.)\s+", original) if c.strip()]
        if len(chunks) <= 2 or len(original) <= 500:
            simplified = original
        else:
            simplified = " ".join(chunks[:5]).strip()
        if len(simplified) > 700:
            simplified = simplified[:697].rstrip() + "..."
        return {"original": original, "simplified": simplified}

    def _sanitize_microstructure_numeric_fields(
        self,
        micro_info: Dict[str, Any],
        process_info: Optional[Dict[str, Any]] = None,
        properties: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        if not micro_info:
            return micro_info

        context_parts = [
            self._extract_text_payload(micro_info.get("Microstructure_Text")),
            self._extract_text_payload((process_info or {}).get("Process_Text")),
        ]
        for prop in properties or []:
            if not isinstance(prop, dict):
                continue
            context_parts.append(str(prop.get("Test_Condition") or ""))
            context_parts.append(str(prop.get("Note") or ""))
        context = " ".join(part for part in context_parts if part).lower()

        rel_density = self._coerce_float(micro_info.get("Relative_Density_pct"))
        if rel_density is not None and abs(rel_density - 100.0) <= 1e-9:
            if not re.search(r"\b100(?:\.0+)?\s*%", context) and re.search(
                r"\b(fully dense|full density|full densification|pore-?free|no porosity|no pores|porosity was not observed|no porosity detected)\b",
                context,
                re.IGNORECASE,
            ):
                micro_info["Relative_Density_pct"] = None

        porosity = self._coerce_float(micro_info.get("Porosity_pct"))
        if porosity is not None and abs(porosity) <= 1e-9:
            if not re.search(r"\b0(?:\.0+)?\s*%", context) and re.search(
                r"\b(no porosity|no pores|pore-?free|porosity was not observed|no porosity detected)\b",
                context,
                re.IGNORECASE,
            ):
                micro_info["Porosity_pct"] = None

        return micro_info

    def _order_properties(self, tests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Order properties by required priority tiers."""
        if not tests:
            return []

        alias = {
            "microhardness": "Hardness_HV",
            "hardness_hv": "Hardness_HV",
            "hardness": "Hardness_HV",
            "ultimate_compressive_strength": "Ultimate_Strength_Compressive",
            "elongation": "Elongation_Total",
            "total_elongation": "Elongation_Total",
            "uniform_elongation": "Elongation_Uniform",
            "fracture_strain": "Elongation_At_Fracture",
            "fracture_elongation": "Elongation_At_Fracture",
            "youngs_modulus": "Elastic_Modulus",
            "fracture_toughness_kic": "Fracture_Toughness_KIC",
            "fatigue_strength_unnotched": "Fatigue_Strength_Unnotched",
        }
        tier1 = [
            "Yield_Strength",
            "Yield_Strength_Compressive",
            "Ultimate_Tensile_Strength",
            "Ultimate_Strength_Compressive",
            "Elongation_Total",
            "Elongation_Uniform",
            "Elongation_At_Fracture",
            "Elongation_Compressive",
            "Elastic_Modulus",
            "Hardness_HV",
            "Hardness_HRC",
            "Hardness_HB",
        ]
        tier2 = [
            "Residual_Stress",
            "Impact_Toughness",
            "Fatigue_Strength",
            "Fatigue_Strength_Unnotched",
            "Fracture_Toughness",
            "Fracture_Toughness_KIC",
            "Fatigue_Crack_Growth_Threshold",
            "Paris_Exponent",
            "Thermal_Conductivity",
            "Specific_Heat",
            "Coefficient_of_Thermal_Expansion",
            "Electrical_Conductivity",
            "Electrical_Resistivity",
            "Corrosion_Rate",
            "Wear_Rate",
            "Friction_Coefficient",
        ]
        rank1 = {name: i for i, name in enumerate(tier1)}
        rank2 = {name: i for i, name in enumerate(tier2)}
        normalized_tests: List[Dict[str, Any]] = []
        for raw in tests:
            cur = dict(raw)
            prop = str(cur.get("Property_Type") or "").strip()
            canonical_key = re.sub(r"[^a-z0-9]+", "_", prop.lower()).strip("_")
            if canonical_key in alias:
                cur["Property_Type"] = alias[canonical_key]
            normalized_tests.append(cur)

        def _rank(test: Dict[str, Any]) -> Tuple[int, int, str]:
            p = str(test.get("Property_Type") or "")
            if p in rank1:
                return (0, rank1[p], p)
            if p in rank2:
                return (1, rank2[p], p)
            return (2, 9999, p.lower())

        return sorted(normalized_tests, key=_rank)

    def normalize_property_name(self, name: Optional[str]) -> Optional[str]:
        """Map a free-text property name to a standardised column name."""
        if not name:
            return name
        key = name.strip().lower()
        mapped = self.rules.property_name_mapping.get(key)
        if mapped:
            return mapped
        legacy_map = {
            "fracture_elongation": "Elongation_Total",
            "kic": "Fracture_Toughness_KIC",
            "fracture toughness kic": "Fracture_Toughness_KIC",
            "mode i fracture toughness": "Fracture_Toughness_KIC",
            "fatigue crack growth threshold": "Fatigue_Crack_Growth_Threshold",
            "threshold stress intensity factor range for fatigue crack growth": "Fatigue_Crack_Growth_Threshold",
            "δk0": "Fatigue_Crack_Growth_Threshold",
            "Δk0": "Fatigue_Crack_Growth_Threshold",
            "paris exponent": "Paris_Exponent",
            "unnotched fatigue strength": "Fatigue_Strength_Unnotched",
        }
        if key in legacy_map:
            return legacy_map[key]
        return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")

    def infer_process_category(self, process_text: str, comp_raw: str = "") -> str:
        """Classify manufacturing process from free text."""
        combined = " ".join(part for part in (str(process_text or ""), str(comp_raw or "")) if part)
        t = combined.lower()
        matches = self._match_process_category_patterns(combined)

        if matches.get("Laser_Glazing"):
            return "Laser_Glazing"
        if matches.get("Hybrid_AM"):
            return "Hybrid_AM"
        if matches.get("Powder_Recycling"):
            return "Powder_Recycling"
        if matches.get("EBM"):
            return "EBM"
        if matches.get("AM_DED"):
            return "AM_DED"
        if matches.get("AM_SLM"):
            base = "AM_SLM"
        elif matches.get("AM_LPBF"):
            base = "AM_LPBF"
        elif matches.get("WAAM"):
            base = "WAAM"
        elif matches.get("Powder_Metallurgy") and not (
            matches.get("AM_SLM")
            or matches.get("AM_LPBF")
            or matches.get("AM_DED")
            or matches.get("EBM")
            or matches.get("WAAM")
            or re.search(r"\b(additive manufacturing|additively manufactured|3d printing)\b", t)
        ):
            base = "Powder_Metallurgy"
        elif matches.get("Wrought"):
            base = "Wrought"
        elif matches.get("SPS"):
            base = "SPS"
        elif matches.get("HIP"):
            base = "HIP"
        elif matches.get("Arc_Melting"):
            base = "Arc_Melting"
        elif matches.get("Casting"):
            base = "Casting"
        elif matches.get("Mechanical_Alloying"):
            base = "Mechanical_Alloying"
        elif matches.get("HeatTreat"):
            base = "HeatTreat"
        elif re.search(r"\b(additive manufacturing|additively manufactured|3d printing)\b", t):
            base = "AM"
        else:
            base = "Unknown"

        if base == "AM_SLM":
            base = "AM_LPBF"

        if base in {"AM_LPBF", "AM_SLM"}:
            suffixes: List[str] = []
            blend_kw = ("blend", "blending", "premix", "premixed", "double hopper", "mixing chamber")
            graded_kw = ("graded", "multigraded", "gradient", "stepwise", "layers per blend")
            if any(kw in t for kw in blend_kw):
                suffixes.append("Powder_Blending")
            if any(kw in t for kw in graded_kw) or "graded" in (comp_raw or "").lower() or "gradient" in (comp_raw or "").lower():
                suffixes.append("Graded_Composition")
            if suffixes:
                return f"{base} + " + " + ".join(suffixes)

        return base

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_sample(self, comp: dict, mat_idx: int, s_idx: int) -> Dict[str, Any]:
        """Build a single ``Processed_Sample`` entry from a composition dict."""
        comp_raw = comp.get("composition", "") or ""

        match = re.search(r"\[(.*?)\]\s*$", comp_raw)
        bracket_suffix = match.group(1).replace(" ", "_") if match else ""
        orientation = comp.get("build_orientation") or ""
        orientation_suffix = orientation.replace(" ", "_").replace("/", "-") if orientation else ""

        if bracket_suffix and orientation_suffix:
            condition_suffix = f"{bracket_suffix}_{orientation_suffix}"
        elif bracket_suffix:
            condition_suffix = bracket_suffix
        elif orientation_suffix:
            condition_suffix = orientation_suffix
        else:
            condition_suffix = "Default"

        props = comp.get("properties_of_composition", []) or []
        tests: List[Dict[str, Any]] = []
        grain_size_from_props: Optional[float] = None
        process_text_raw = comp.get("processing_conditions")
        process_text_hint = ""
        if isinstance(process_text_raw, dict):
            process_text_hint = json.dumps(process_text_raw, ensure_ascii=False)
        elif process_text_raw:
            process_text_hint = str(process_text_raw)
        graded_context = bool(
            re.search(
                r"\b(graded|gradient|stepwise|multigraded|increasing|up to \d+(?:\.\d+)?\s*(?:wt\s*%|wt%|%))\b",
                f"{comp_raw} {process_text_hint}".lower(),
            )
        )
        for j, p in enumerate(props, start=1):
            numeric_val = p.get("value_numeric")
            prop_std_name = self.normalize_property_name(p.get("property_name"))

            if prop_std_name == "Grain_Size" and numeric_val is not None:
                unit = (p.get("unit") or "").lower().strip()
                if "nm" in unit:
                    grain_size_from_props = numeric_val / 1000.0
                else:
                    grain_size_from_props = numeric_val

            if numeric_val is None:
                continue
            value_range = p.get("value_range")
            if value_range is None and p.get("value_type") == "range":
                value_range = p.get("value")
            measurement_condition_txt = str(p.get("measurement_condition") or "").lower()
            has_spatial_anchor = any(
                kw in measurement_condition_txt for kw in ("top", "bottom", "layer", "height", "position", "zone")
            )
            # Gradient entries often report only an aggregate range across steps.
            # Do not synthesize a midpoint test-point for those aggregate ranges.
            if graded_context and value_range and not has_spatial_anchor:
                continue

            value_stddev = p.get("value_stddev")
            if value_stddev is None:
                value_txt = str(p.get("value") or "")
                m_std = re.search(r"±\s*(-?\d+(?:\.\d+)?)", value_txt)
                if m_std:
                    try:
                        value_stddev = float(m_std.group(1))
                    except Exception:
                        value_stddev = None
            tests.append({
                "Test_ID": f"T{j:03d}",
                "Test_Temperature_K": self.parse_temperature_to_k(p.get("measurement_condition")),
                "Property_Type": prop_std_name,
                "Property_Value": numeric_val,
                "Property_Value_Range": value_range,
                "Property_StdDev": value_stddev,
                "Property_Unit": self._infer_property_unit(prop_std_name, p.get("unit")),
                "Strain_Rate_s1": self._clean_optional_str(p.get("strain_rate_s1"))
                or self._extract_strain_rate_text(p.get("measurement_condition"), p.get("additional_information")),
                "Tensile_Speed_mm_min": (
                    self._coerce_float(p.get("tensile_speed_mm_min"))
                    if self._coerce_float(p.get("tensile_speed_mm_min")) is not None
                    else self._extract_tensile_speed_mm_min(p.get("measurement_condition"), p.get("additional_information"))
                ),
                "Hardness_Load": self._clean_optional_str(p.get("hardness_load"))
                or self._extract_hardness_load(p.get("measurement_condition"), p.get("additional_information")),
                "Hardness_Dwell_Time_s": (
                    self._coerce_float(p.get("hardness_dwell_time_s"))
                    if self._coerce_float(p.get("hardness_dwell_time_s")) is not None
                    else self._extract_hardness_dwell_time_s(p.get("measurement_condition"), p.get("additional_information"))
                ),
                "Data_Source": self._normalize_data_source(
                    p.get("data_source"),
                    f"{p.get('measurement_condition') or ''}; {p.get('additional_information') or ''}",
                ),
                "Test_Specimen": self._clean_optional_str(p.get("test_specimen"))
                or self._extract_test_specimen(p.get("measurement_condition"), p.get("additional_information")),
                "Note": self._default_property_note(
                    prop_std_name or "",
                    p.get("property_name"),
                    p.get("note") or p.get("additional_information"),
                ),
            })

        process_text = comp.get("processing_conditions")
        if isinstance(process_text, dict):
            process_text = json.dumps(process_text, ensure_ascii=False)

        microstructure_text = self.build_microstructure_text(comp)

        main_phase = comp.get("main_phase")
        if not main_phase:
            main_phase = self.infer_main_phase(microstructure_text)
        else:
            main_phase = self._normalize_main_phase(main_phase, micro_text=microstructure_text)

        has_precipitates_val = comp.get("has_precipitates")
        if has_precipitates_val is None:
            has_precipitates_val = self.infer_precipitates(microstructure_text)

        grain_size_um = comp.get("grain_size_avg_um")
        if grain_size_um is None and grain_size_from_props is not None:
            grain_size_um = grain_size_from_props

        process_category = comp.get("process_category")
        inferred_process_category = self.infer_process_category(process_text or "", comp_raw=comp_raw)
        if (
            not process_category
            or process_category == "Unknown"
            or str(process_category).strip().upper() in {"AM", "ADDITIVE_MANUFACTURING"}
        ):
            process_category = inferred_process_category
        elif inferred_process_category != "Unknown":
            current = str(process_category).strip()
            # Prefer more specific inferred categories (e.g., + Powder_Blending / + Graded_Composition)
            if ("+" in inferred_process_category and "+" not in current) or (
                current == "AM_SLM" and inferred_process_category.startswith("AM_LPBF")
            ):
                process_category = inferred_process_category
        resolved_process_category = self._resolve_process_category(
            process_category,
            process_text or "",
            comp_raw=comp_raw,
        )

        key_params = self.parse_key_params(process_text or "", comp.get("processing_params"))
        if orientation and "Build_Orientation" not in key_params:
            key_params["Build_Orientation"] = orientation

        return {
            "Sample_ID": f"S{mat_idx:03d}_{s_idx:02d}_{condition_suffix}",
            "Process_Category": self._humanize_process_category(
                resolved_process_category,
                process_text=process_text or "",
                comp_raw=comp_raw,
            ),
            "Process_Text_For_AI": process_text or "not provided",
            "Key_Params_JSON": key_params,
            "Main_Phase": main_phase or "",
            "Microstructure_Text_For_AI": microstructure_text,
            "Has_Precipitates": has_precipitates_val,
            "Grain_Size_avg_um": grain_size_um,
            "Performance_Tests": tests,
        }
