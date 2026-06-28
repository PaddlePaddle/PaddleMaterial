"""
Loader for domain-specific rules used by the schema converter.

Rules are stored in ``domain_rules.yaml`` so that materials scientists can
adjust extraction heuristics without touching Python code.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set

import yaml


_DEFAULT_YAML = Path(__file__).parent / "domain_rules.yaml"


@dataclass
class DomainRules:
    """Immutable container for all domain-specific extraction rules."""

    valid_elements: Set[str] = field(default_factory=set)
    phase_patterns: Dict[str, str] = field(default_factory=dict)
    precipitate_keywords: List[str] = field(default_factory=list)
    property_name_mapping: Dict[str, str] = field(default_factory=dict)
    process_category_keywords: Dict[str, List[str]] = field(default_factory=dict)
    parameter_patterns: Dict[str, List[str]] = field(default_factory=dict)
    commercial_alloy_aliases: Dict[str, List[str]] = field(default_factory=dict)
    commercial_nominal_maps: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Pre-compiled regexes (populated by _compile)
    _compiled_param_patterns: Dict[str, List[re.Pattern]] = field(
        default_factory=dict, repr=False
    )

    def _compile(self) -> None:
        """Pre-compile regex patterns for parameter extraction."""
        for param_name, patterns in self.parameter_patterns.items():
            self._compiled_param_patterns[param_name] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]

    @classmethod
    def from_yaml(cls, path: Path | None = None) -> DomainRules:
        """Load rules from a YAML file.

        Parameters
        ----------
        path : Path, optional
            Path to the YAML file.  Defaults to ``domain_rules.yaml`` next to
            this module.
        """
        if path is None:
            path = _DEFAULT_YAML
        with open(path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)

        rules = cls(
            valid_elements=set(raw.get("valid_elements", [])),
            phase_patterns=raw.get("phase_patterns", {}),
            precipitate_keywords=raw.get("precipitate_keywords", []),
            property_name_mapping=raw.get("property_name_mapping", {}),
            process_category_keywords=raw.get("process_category_keywords", {}),
            parameter_patterns=raw.get("parameter_patterns", {}),
            commercial_alloy_aliases=raw.get("commercial_alloy_aliases", {}),
            commercial_nominal_maps=raw.get("commercial_nominal_maps", {}),
        )
        rules._compile()
        return rules


# Module-level singleton so callers can ``from ppmatAgent.knowmat.domain_rules import default_rules``.
default_rules = DomainRules.from_yaml()
