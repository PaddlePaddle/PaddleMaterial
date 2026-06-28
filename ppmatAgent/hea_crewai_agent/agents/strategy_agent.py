"""CrewAI-backed strategy agent for MEL mutation planning."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Callable, Dict, List, Optional

from ppmatAgent.hea_crewai_agent.core.material import Alloy
from ppmatAgent.hea_crewai_agent.core.mel import MELExecutor, MELGenerator, MELParser
from ppmatAgent.hea_crewai_agent.core.materials_graphrag import query_materials_graph


class StrategyAgent:
    """Generate mutation MEL strings with CrewAI LLM and safe fallbacks."""

    def __init__(
        self,
        llm_model: str = "gpt-5.2",
        temperature: float = 0.2,
        max_tokens: int = 160,
        max_retries: int = 2,
        timeout: Optional[float] = 30.0,
        fallback_to_rules: bool = True,
        max_operations: int = 2,
        crewai_llm: Optional[Any] = None,
        completion_fn: Optional[Callable[..., Any]] = None,
        extra_completion_kwargs: Optional[Dict[str, Any]] = None,
        use_graphrag: bool = True,
        memory_records: Optional[List[Dict[str, Any]]] = None,
        graphrag_max_results: int = 5,
    ):
        self.llm_model = llm_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.timeout = timeout
        self.fallback_to_rules = fallback_to_rules
        self.max_operations = max_operations
        self.crewai_llm = crewai_llm
        self.completion_fn = completion_fn
        self.extra_completion_kwargs = extra_completion_kwargs or {}
        self.use_graphrag = use_graphrag
        self.memory_records = list(memory_records or [])
        self.graphrag_max_results = max(1, int(graphrag_max_results))

        self.mel_parser = MELParser()
        self.mel_executor = MELExecutor()
        self.mel_generator = MELGenerator()

        self.last_raw_response: Optional[str] = None
        self.last_rationale: Optional[str] = None
        self.last_source: str = "uninitialized"
        self.last_graphrag_context: Optional[Dict[str, Any]] = None
        self.last_graphrag_summary: Optional[Dict[str, Any]] = None
        self._last_fallback_used_graphrag: bool = False

    @classmethod
    def from_env(cls, **overrides: Any) -> "StrategyAgent":
        """Create an agent from environment variables."""
        config: Dict[str, Any] = {
            "llm_model": os.environ.get("STRATEGY_AGENT_MODEL", "gpt-5.2"),
            "temperature": float(os.environ.get("STRATEGY_AGENT_TEMPERATURE", "0.2")),
            "max_tokens": int(os.environ.get("STRATEGY_AGENT_MAX_TOKENS", "160")),
            "max_retries": int(os.environ.get("STRATEGY_AGENT_MAX_RETRIES", "2")),
            "timeout": float(os.environ.get("STRATEGY_AGENT_TIMEOUT", "30")),
        }
        config.update(overrides)

        if "crewai_llm" not in config or config["crewai_llm"] is None:
            try:
                from ppmatAgent.hea_crewai_agent.agents.llm_interface import get_crewai_llm

                config["crewai_llm"] = get_crewai_llm(
                    model=config["llm_model"],
                    temperature=config["temperature"],
                )
            except Exception:
                # Allow tests or power users to inject a custom completion_fn instead.
                pass

        return cls(**config)

    def generate_mutation(
        self,
        alloy: Alloy,
        allowed_elements: Optional[List[str]] = None,
        constraints: Optional[Dict[str, Any]] = None,
        evolution_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Return a valid MEL mutation string for the given alloy."""
        allowed_elements = allowed_elements or sorted(alloy.composition.keys())
        constraints = constraints or {}
        evolution_context = evolution_context or {}
        graphrag_context = self._retrieve_graphrag_context(
            alloy=alloy,
            allowed_elements=allowed_elements,
            constraints=constraints,
            evolution_context=evolution_context,
        )

        last_error: Optional[str] = None
        if self.crewai_llm is not None or self.completion_fn is not None:
            for _ in range(self.max_retries + 1):
                try:
                    response_text = self._query_model(
                        alloy=alloy,
                        allowed_elements=allowed_elements,
                        constraints=constraints,
                        evolution_context=evolution_context,
                        graphrag_context=graphrag_context,
                    )
                    candidate = self._extract_candidate_mel(response_text)
                    if candidate and self._is_valid_candidate(
                        alloy=alloy,
                        mel_string=candidate,
                        allowed_elements=allowed_elements,
                        constraints=constraints,
                    ):
                        self.last_raw_response = response_text
                        self.last_rationale = self._extract_rationale(response_text)
                        self.last_source = "crewai" if self.crewai_llm is not None else "custom_completion"
                        return candidate
                    last_error = f"invalid MEL candidate: {response_text!r}"
                except Exception as exc:  # pragma: no cover - defensive path
                    last_error = str(exc)
        else:
            last_error = "No LLM completion backend configured; using GraphRAG/rule fallback."

        if self.fallback_to_rules:
            fallback = self._generate_fallback_mutation(
                alloy=alloy,
                allowed_elements=allowed_elements,
                constraints=constraints,
                graphrag_context=graphrag_context,
            )
            self.last_raw_response = last_error
            self.last_rationale = "fallback-to-rules"
            self.last_source = "graphrag_fallback" if self._last_fallback_used_graphrag else "fallback"
            return fallback

        raise RuntimeError(f"StrategyAgent failed to generate a valid MEL string: {last_error}")

    def _query_model(
        self,
        alloy: Alloy,
        allowed_elements: List[str],
        constraints: Dict[str, Any],
        evolution_context: Dict[str, Any],
        graphrag_context: Optional[Dict[str, Any]],
    ) -> str:
        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {
                "role": "user",
                "content": self._build_user_prompt(
                    alloy=alloy,
                    allowed_elements=allowed_elements,
                    constraints=constraints,
                    evolution_context=evolution_context,
                    graphrag_context=graphrag_context,
                ),
            },
        ]

        if self.crewai_llm is not None:
            response = self.crewai_llm.call(messages)
            return self._extract_content(response)

        completion = self._get_completion_fn()
        request_kwargs: Dict[str, Any] = {
            "model": self.llm_model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.timeout is not None:
            request_kwargs["timeout"] = self.timeout
        request_kwargs.update(self.extra_completion_kwargs)

        response = completion(**request_kwargs)
        return self._extract_content(response)

    def _get_completion_fn(self) -> Callable[..., Any]:
        if self.completion_fn is not None:
            return self.completion_fn

        raise RuntimeError(
            "No CrewAI LLM was provided. Pass `crewai_llm` or supply a custom `completion_fn`."
        )

    def _build_system_prompt(self) -> str:
        return (
            "You are a materials-design strategy agent. "
            "Return only a MEL mutation plan with 1 or 2 operations. "
            "Allowed operations are REPLACE(source, fraction, target), "
            "ADD(element, fraction), REMOVE(element), ADJUST(element, target_fraction), "
            "and SCALE(element, scale_factor). "
            "Do not include explanations unless you append a JSON object after the MEL string. "
            "Prefer chemically sensible, conservative edits that keep the alloy valid."
        )

    def _build_user_prompt(
        self,
        alloy: Alloy,
        allowed_elements: List[str],
        constraints: Dict[str, Any],
        evolution_context: Dict[str, Any],
        graphrag_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        objective_scores = alloy.properties.get("objective_scores", {})
        payload = {
            "composition": alloy.composition,
            "properties": {
                "fitness": alloy.properties.get("fitness"),
                "objective_scores": objective_scores,
                "mixing_entropy": alloy.properties.get("mixing_entropy"),
                "mixing_enthalpy": alloy.properties.get("mixing_enthalpy"),
                "delta": alloy.properties.get("delta"),
                "estimated_strength": alloy.properties.get("estimated_strength"),
                "estimated_ductility": alloy.properties.get("estimated_ductility"),
                "estimated_hardness": alloy.properties.get("estimated_hardness"),
                "corrosion_resistance": alloy.properties.get("corrosion_resistance"),
            },
            "allowed_elements": allowed_elements,
            "constraints": constraints,
            "evolution_context": evolution_context,
            "graphrag_context": graphrag_context or {},
            "rules": [
                "Use only allowed elements.",
                "Keep every resulting concentration within the provided min/max range.",
                "Prefer improving the weakest objective score first.",
                "Use GraphRAG evidence to move toward high-fitness memory regions and away from failed/low-confidence regions when available.",
                "Output MEL only, for example: ADJUST(Cr, 0.24) + SCALE(Al, 0.90)",
            ],
        }
        return json.dumps(payload, ensure_ascii=True, indent=2)

    def _retrieve_graphrag_context(
        self,
        alloy: Alloy,
        allowed_elements: List[str],
        constraints: Dict[str, Any],
        evolution_context: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        self.last_graphrag_context = None
        self.last_graphrag_summary = None

        if not self.use_graphrag:
            return None

        query = self._build_graphrag_query(
            alloy=alloy,
            allowed_elements=allowed_elements,
            evolution_context=evolution_context,
        )
        try:
            result = query_materials_graph(
                query=query,
                memory_records=self.memory_records,
                max_results=self.graphrag_max_results,
                include_models=True,
            )
        except Exception as exc:  # pragma: no cover - defensive path
            summary = {"error": str(exc), "query": query}
            self.last_graphrag_context = summary
            self.last_graphrag_summary = summary
            return summary

        summary = self._summarize_graphrag_result(result)
        self.last_graphrag_context = result
        self.last_graphrag_summary = summary
        return summary

    def _build_graphrag_query(
        self,
        alloy: Alloy,
        allowed_elements: List[str],
        evolution_context: Dict[str, Any],
    ) -> str:
        objective_scores = alloy.properties.get("objective_scores", {})
        weakest_objective = ""
        if objective_scores:
            try:
                weakest_objective = min(objective_scores.items(), key=lambda item: item[1])[0]
            except Exception:
                weakest_objective = ""

        objectives = evolution_context.get("objectives", [])
        if isinstance(objectives, str):
            objective_text = objectives
        elif isinstance(objectives, list):
            objective_text = " ".join(str(item) for item in objectives)
        else:
            objective_text = json.dumps(objectives, ensure_ascii=False)

        composition_text = " ".join(
            f"{element}{fraction:.3f}" for element, fraction in sorted(alloy.composition.items())
        )
        return (
            "HEA composition optimization GraphRAG high fitness failed region "
            f"uncertainty surrogate model objectives {objective_text} "
            f"weakest {weakest_objective} allowed elements {' '.join(allowed_elements)} "
            f"current composition {composition_text}"
        )

    def _summarize_graphrag_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        top_compositions: List[Dict[str, Any]] = []
        top_models: List[Dict[str, Any]] = []
        for item in result.get("results", [])[: self.graphrag_max_results]:
            item_type = item.get("type")
            metadata = item.get("metadata", {}) if isinstance(item.get("metadata"), dict) else {}
            if item_type == "composition":
                top_compositions.append(
                    {
                        "label": item.get("label"),
                        "score": item.get("score"),
                        "fitness": metadata.get("fitness"),
                        "composition": metadata.get("composition", {}),
                        "content": str(item.get("content", ""))[:500],
                    }
                )
            elif item_type == "surrogate_model":
                top_models.append(
                    {
                        "label": item.get("label"),
                        "backend": metadata.get("backend"),
                        "best_for": metadata.get("best_for"),
                        "score": item.get("score"),
                    }
                )

        evidence_paths = []
        for path in result.get("evidence_paths", [])[:3]:
            neighbors = path.get("neighbors", [])
            evidence_paths.append(
                {
                    "center": path.get("center", {}),
                    "neighbors": neighbors[:3] if isinstance(neighbors, list) else [],
                }
            )

        return {
            "query": result.get("query"),
            "answer": result.get("answer"),
            "graph": result.get("graph", {}),
            "top_compositions": top_compositions,
            "top_models": top_models,
            "evidence_paths": evidence_paths,
        }

    def _extract_content(self, response: Any) -> str:
        if isinstance(response, str):
            return response

        if isinstance(response, dict):
            choices = response.get("choices", [])
            if not choices:
                raise ValueError("Completion backend response did not contain choices")
            message = choices[0].get("message", {})
            content = message.get("content")
            if content is None:
                raise ValueError("Completion backend response did not contain message content")
            return str(content)

        choices = getattr(response, "choices", None)
        if not choices:
            raise ValueError("Completion backend response did not contain choices")

        message = getattr(choices[0], "message", None)
        if message is None:
            raise ValueError("Completion backend response did not contain a message")

        content = getattr(message, "content", None)
        if content is None:
            raise ValueError("Completion backend response did not contain message content")

        return str(content)

    def _extract_candidate_mel(self, response_text: str) -> Optional[str]:
        text = response_text.strip()

        json_match = re.search(r'"mel"\s*:\s*"([^"]+)"', text)
        if json_match:
            text = json_match.group(1)

        text = text.replace("```json", "").replace("```", "").strip()
        text = self._normalize_operation_keywords(text)

        operations = re.findall(
            r"(?:REPLACE|ADD|REMOVE|ADJUST|SCALE)\([^()\n]+\)",
            text,
        )
        if not operations:
            return None

        return " + ".join(operations[: self.max_operations])

    def _extract_rationale(self, response_text: str) -> Optional[str]:
        json_match = re.search(r'"rationale"\s*:\s*"([^"]+)"', response_text)
        if json_match:
            return json_match.group(1)
        return None

    def _normalize_operation_keywords(self, text: str) -> str:
        normalized = text
        for keyword in ["replace", "add", "remove", "adjust", "scale"]:
            normalized = re.sub(
                rf"\b{keyword}\s*\(",
                keyword.upper() + "(",
                normalized,
                flags=re.IGNORECASE,
            )
        return normalized

    def _is_valid_candidate(
        self,
        alloy: Alloy,
        mel_string: str,
        allowed_elements: List[str],
        constraints: Dict[str, Any],
    ) -> bool:
        try:
            operations = self.mel_parser.parse(mel_string)
        except Exception:
            return False

        if not operations or len(operations) > self.max_operations:
            return False

        allowed_set = set(allowed_elements)
        for operation in operations:
            params = operation.params
            if operation.operation_type == "REPLACE":
                if params["source"] not in alloy.composition:
                    return False
                if params["target"] not in allowed_set:
                    return False
                if not 0.01 <= params["fraction"] <= 0.8:
                    return False
            elif operation.operation_type == "ADD":
                if params["element"] not in allowed_set:
                    return False
                if not 0.01 <= params["fraction"] <= 0.25:
                    return False
            elif operation.operation_type == "REMOVE":
                if params["element"] not in alloy.composition:
                    return False
            elif operation.operation_type == "ADJUST":
                if params["element"] not in allowed_set:
                    return False
                if not 0.01 <= params["target_fraction"] <= 0.8:
                    return False
            elif operation.operation_type == "SCALE":
                if params["element"] not in alloy.composition:
                    return False
                if not 0.5 <= params["scale_factor"] <= 1.5:
                    return False

        mutated = self.mel_executor.execute(alloy, operations)
        if mutated.composition == alloy.composition:
            return False
        if constraints and not mutated.is_valid(constraints):
            return False
        return True

    def _generate_fallback_mutation(
        self,
        alloy: Alloy,
        allowed_elements: List[str],
        constraints: Dict[str, Any],
        graphrag_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        self._last_fallback_used_graphrag = False
        for candidate in self._graphrag_rule_candidates(
            alloy,
            allowed_elements,
            constraints,
            graphrag_context,
        ):
            if self._is_valid_candidate(alloy, candidate, allowed_elements, constraints):
                self._last_fallback_used_graphrag = True
                return candidate

        for candidate in self._rule_based_candidates(alloy, allowed_elements, constraints):
            if self._is_valid_candidate(alloy, candidate, allowed_elements, constraints):
                return candidate

        for _ in range(32):
            candidate = self.mel_generator.generate_random_operation(alloy, allowed_elements)
            if self._is_valid_candidate(alloy, candidate, allowed_elements, constraints):
                return candidate

        # Last-resort best effort: let the evolution engine apply its own validity filter.
        return self.mel_generator.generate_random_operation(alloy, allowed_elements)

    def _graphrag_rule_candidates(
        self,
        alloy: Alloy,
        allowed_elements: List[str],
        constraints: Dict[str, Any],
        graphrag_context: Optional[Dict[str, Any]],
    ) -> List[str]:
        if not graphrag_context:
            return []

        min_fraction = float(constraints.get("min_concentration", 0.05))
        max_fraction = float(constraints.get("max_concentration", 0.40))
        allowed_set = set(allowed_elements)
        candidates: List[str] = []

        for item in graphrag_context.get("top_compositions", []):
            target = item.get("composition", {})
            if not isinstance(target, dict):
                continue

            deltas: List[tuple[str, float, float, float]] = []
            for element, raw_target in target.items():
                if element not in allowed_set:
                    continue
                try:
                    target_fraction = float(raw_target)
                except Exception:
                    continue
                current_fraction = float(alloy.composition.get(element, 0.0))
                delta = target_fraction - current_fraction
                if abs(delta) >= 0.025:
                    deltas.append((element, current_fraction, target_fraction, delta))

            for element, current, target_fraction, delta in sorted(
                deltas,
                key=lambda row: abs(row[3]),
                reverse=True,
            ):
                bounded_target = min(max_fraction, max(min_fraction, target_fraction))
                if delta > 0:
                    if current > 0:
                        step_target = min(
                            max_fraction,
                            max(current + 0.03, bounded_target),
                        )
                        candidates.append(f"ADJUST({element}, {step_target:.2f})")
                    else:
                        add_fraction = min(0.10, max(0.03, bounded_target))
                        candidates.append(f"ADD({element}, {add_fraction:.2f})")
                elif current > 0:
                    target_scale = bounded_target / current if current else 0.75
                    scale = min(0.95, max(0.65, target_scale))
                    candidates.append(f"SCALE({element}, {scale:.2f})")

                if len(candidates) >= self.graphrag_max_results:
                    return candidates

        return candidates

    def _rule_based_candidates(
        self,
        alloy: Alloy,
        allowed_elements: List[str],
        constraints: Dict[str, Any],
    ) -> List[str]:
        candidates: List[str] = []
        scores = alloy.properties.get("objective_scores", {})
        min_fraction = constraints.get("min_concentration", 0.05)
        max_fraction = constraints.get("max_concentration", 0.40)

        weakest_objective = None
        if scores:
            weakest_objective = min(scores.items(), key=lambda item: item[1])[0]

        if weakest_objective == "corrosion":
            candidates.extend(
                self._boost_elements(
                    alloy, allowed_elements, ["Cr", "Ni", "Mo", "Al", "Ti"], max_fraction
                )
            )
        elif weakest_objective == "mechanical":
            candidates.extend(
                self._boost_elements(
                    alloy, allowed_elements, ["Ti", "Nb", "Ta", "V", "Mo", "W"], max_fraction
                )
            )
        elif weakest_objective == "thermodynamic":
            candidates.extend(self._balance_major_elements(alloy, min_fraction, max_fraction))

        candidates.extend(self._balance_major_elements(alloy, min_fraction, max_fraction))
        return candidates

    def _boost_elements(
        self,
        alloy: Alloy,
        allowed_elements: List[str],
        preferred_elements: List[str],
        max_fraction: float,
    ) -> List[str]:
        candidates: List[str] = []
        allowed_set = set(allowed_elements)

        for element in preferred_elements:
            if element not in allowed_set:
                continue

            current = alloy.composition.get(element, 0.0)
            target = min(max_fraction, max(0.08, current + 0.05))

            if current > 0:
                candidates.append(f"ADJUST({element}, {target:.2f})")
            else:
                add_fraction = min(0.08, max(0.03, target))
                candidates.append(f"ADD({element}, {add_fraction:.2f})")

        return candidates

    def _balance_major_elements(
        self,
        alloy: Alloy,
        min_fraction: float,
        max_fraction: float,
    ) -> List[str]:
        if not alloy.composition:
            return []

        items = sorted(alloy.composition.items(), key=lambda item: item[1], reverse=True)
        dominant_element, dominant_fraction = items[0]
        weakest_element, weakest_fraction = items[-1]
        n_elements = len(items)
        target_uniform = max(min(1.0 / max(n_elements, 1), max_fraction), min_fraction)

        candidates = []
        if dominant_fraction > target_uniform + 0.02:
            scale = max(0.75, target_uniform / dominant_fraction)
            candidates.append(f"SCALE({dominant_element}, {scale:.2f})")
        if weakest_fraction < target_uniform - 0.02:
            candidates.append(f"ADJUST({weakest_element}, {target_uniform:.2f})")
        return candidates
