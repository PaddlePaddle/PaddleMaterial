"""
多元共晶点筛选脚本

流程:
1. 在给定成分约束内随机采样候选
2. 用启发式 eutectic 适应度进行快速预筛
3. 对高分候选执行 CALPHAD 共晶扫描
4. 在最佳种子附近做局部扰动细化
5. 输出排序后的候选列表与关键热力学指标
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ppmatAgent.hea_crewai_agent.core.calphad_evaluator import CALPHADEvaluator
from ppmatAgent.hea_crewai_agent.core.evaluator import ThermodynamicEvaluator
from ppmatAgent.hea_crewai_agent.core.material import Alloy
from ppmatAgent.hea_crewai_agent.core.tdb_registry import find_best_local_tdb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a full eutectic screening workflow.")
    parser.add_argument(
        "--elements",
        default="Co,Cr,Fe,Ni,V",
        help="Comma-separated elements to screen. Default matches the bundled full CALPHAD database.",
    )
    parser.add_argument("--samples", type=int, default=400, help="Number of global random samples.")
    parser.add_argument("--prefilter-top", type=int, default=50, help="Top candidates sent to CALPHAD.")
    parser.add_argument("--final-top", type=int, default=10, help="How many final candidates to print.")
    parser.add_argument("--refine-seeds", type=int, default=5, help="Top CALPHAD seeds used for local refinement.")
    parser.add_argument(
        "--refine-samples-per-seed",
        type=int,
        default=24,
        help="Local perturbation samples generated around each seed.",
    )
    parser.add_argument("--min-concentration", type=float, default=0.10, help="Minimum atomic fraction.")
    parser.add_argument("--max-concentration", type=float, default=0.35, help="Maximum atomic fraction.")
    parser.add_argument("--dirichlet-alpha", type=float, default=2.5, help="Dirichlet alpha for global sampling.")
    parser.add_argument("--local-sigma", type=float, default=0.03, help="Local perturbation scale.")
    parser.add_argument("--temperature-points", type=int, default=31, help="CALPHAD temperature scan points.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--direct-calphad-all",
        action="store_true",
        help="Skip heuristic prefilter and run CALPHAD on all generated candidates.",
    )
    parser.add_argument(
        "--output",
        default="eutectic_screening_result.json",
        help="JSON file for the screening result.",
    )
    return parser.parse_args()


def normalize_composition(elements: List[str], fractions: np.ndarray) -> Dict[str, float]:
    total = float(np.sum(fractions))
    return {
        element: float(fraction / total)
        for element, fraction in zip(elements, fractions)
    }


def rounded_signature(composition: Dict[str, float], digits: int = 4) -> Tuple[Tuple[str, float], ...]:
    return tuple((element, round(fraction, digits)) for element, fraction in sorted(composition.items()))


def is_within_bounds(
    composition: Dict[str, float],
    min_concentration: float,
    max_concentration: float
) -> bool:
    return all(min_concentration <= fraction <= max_concentration for fraction in composition.values())


def sample_global_candidates(
    elements: List[str],
    sample_count: int,
    min_concentration: float,
    max_concentration: float,
    dirichlet_alpha: float,
    rng: np.random.Generator
) -> List[Alloy]:
    candidates: List[Alloy] = []
    seen = set()
    attempts = 0
    max_attempts = max(sample_count * 200, 5000)
    alpha = np.full(len(elements), dirichlet_alpha)

    while len(candidates) < sample_count and attempts < max_attempts:
        attempts += 1
        fractions = rng.dirichlet(alpha)
        composition = normalize_composition(elements, fractions)
        if not is_within_bounds(composition, min_concentration, max_concentration):
            continue

        signature = rounded_signature(composition)
        if signature in seen:
            continue

        seen.add(signature)
        candidates.append(Alloy(composition=composition))

    if len(candidates) < sample_count:
        raise RuntimeError(
            f"Unable to generate enough valid candidates. "
            f"Requested {sample_count}, got {len(candidates)}."
        )

    return candidates


def sample_local_variants(
    seed_alloy: Alloy,
    sample_count: int,
    min_concentration: float,
    max_concentration: float,
    sigma: float,
    rng: np.random.Generator
) -> List[Alloy]:
    elements = sorted(seed_alloy.composition.keys())
    seed_vector = np.array([seed_alloy.composition[element] for element in elements], dtype=float)
    candidates: List[Alloy] = []
    seen = set()
    attempts = 0
    max_attempts = max(sample_count * 200, 5000)

    while len(candidates) < sample_count and attempts < max_attempts:
        attempts += 1
        perturbation = rng.normal(0.0, sigma, size=len(elements))
        proposal = seed_vector + perturbation
        proposal = np.clip(proposal, 1e-6, None)
        composition = normalize_composition(elements, proposal)
        if not is_within_bounds(composition, min_concentration, max_concentration):
            continue

        signature = rounded_signature(composition)
        if signature in seen or signature == rounded_signature(seed_alloy.composition):
            continue

        seen.add(signature)
        candidates.append(Alloy(composition=composition))

    return candidates


def evaluate_simple_candidates(candidates: List[Alloy], evaluator: ThermodynamicEvaluator) -> List[Alloy]:
    for alloy in candidates:
        alloy.properties["simple_eutectic_score"] = evaluator.evaluate(alloy)
    return sorted(
        candidates,
        key=lambda alloy: alloy.properties.get("simple_eutectic_score", 0.0),
        reverse=True,
    )


def evaluate_calphad_candidates(
    candidates: List[Alloy],
    evaluator: CALPHADEvaluator,
    label: str
) -> List[Alloy]:
    total = len(candidates)
    for index, alloy in enumerate(candidates, start=1):
        alloy.properties["calphad_eutectic_score"] = evaluator.evaluate(alloy)
        print(
            f"[{label}] CALPHAD {index}/{total} "
            f"score={alloy.properties['calphad_eutectic_score']:.4f} "
            f"comp={rounded_signature(alloy.composition, digits=3)}"
        )
    return sorted(
        candidates,
        key=lambda alloy: alloy.properties.get("calphad_eutectic_score", 0.0),
        reverse=True,
    )


def merge_unique_candidates(*groups: List[Alloy]) -> List[Alloy]:
    merged: List[Alloy] = []
    seen = set()
    for group in groups:
        for alloy in group:
            signature = rounded_signature(alloy.composition)
            if signature in seen:
                continue
            seen.add(signature)
            merged.append(alloy)
    return merged


def alloy_to_record(rank: int, alloy: Alloy) -> Dict:
    props = alloy.properties
    return {
        "rank": rank,
        "alloy_id": alloy.alloy_id,
        "composition": {
            element: round(fraction, 6)
            for element, fraction in sorted(alloy.composition.items())
        },
        "simple_eutectic_score": props.get("simple_eutectic_score"),
        "calphad_eutectic_score": props.get("calphad_eutectic_score", props.get("calphad_fitness")),
        "best_temperature_K": props.get("calphad_temperature"),
        "liquidus_temp_K": props.get("calphad_liquidus_temp"),
        "solidus_temp_K": props.get("calphad_solidus_temp"),
        "freezing_range_K": props.get("calphad_freezing_range"),
        "liquidus_depression_K": props.get("calphad_liquidus_depression"),
        "best_liquid_fraction": props.get("calphad_best_liquid_fraction"),
        "reaction_score": props.get("calphad_eutectic_reaction_score"),
        "best_phases": props.get("calphad_phases", {}),
    }


def database_to_record(database: Dict) -> Dict:
    record = dict(database)
    if isinstance(record.get("supported_elements"), set):
        record["supported_elements"] = sorted(record["supported_elements"])
    return record


def print_top_candidates(candidates: List[Alloy], final_top: int) -> None:
    print()
    print("Top eutectic candidates")
    print("=" * 72)
    for index, alloy in enumerate(candidates[:final_top], start=1):
        props = alloy.properties
        composition_text = ", ".join(
            f"{element}:{fraction:.3f}"
            for element, fraction in sorted(alloy.composition.items())
        )
        print(f"{index}. score={props.get('calphad_eutectic_score', 0.0):.4f}  comp={composition_text}")
        if props.get("calphad_temperature") is not None:
            print(
                "   "
                f"T_best={props.get('calphad_temperature'):.1f} K, "
                f"liquidus={props.get('calphad_liquidus_temp')}, "
                f"solidus={props.get('calphad_solidus_temp')}, "
                f"freeze_range={props.get('calphad_freezing_range')}"
            )
        if props.get("calphad_phases"):
            phase_text = ", ".join(
                f"{phase}:{fraction:.3f}"
                for phase, fraction in sorted(props["calphad_phases"].items())
            )
            print(f"   phases={phase_text}")


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    elements = [element.strip() for element in args.elements.split(",") if element.strip()]

    print("=" * 72)
    print("多元共晶点完整筛选")
    print("=" * 72)
    print(f"Elements: {elements}")
    print(f"Global samples: {args.samples}")
    print(f"Prefilter top: {args.prefilter_top}")
    print(f"Refine seeds: {args.refine_seeds}")
    print(f"Refine samples/seed: {args.refine_samples_per_seed}")
    print(f"Direct CALPHAD all: {args.direct_calphad_all}")

    selected_db = find_best_local_tdb(elements, tdb_dir=Path("tdb_files"), allow_simplified=False)
    if selected_db is None:
        raise RuntimeError(
            f"No full pycalphad-loadable database matches {elements}. "
            "For a complete eutectic screening, use the bundled Co-Cr-Fe-Ni-V system "
            "or provide a matching full CALPHAD database."
        )

    print(f"Database: {selected_db['filename']}")

    simple_eval = ThermodynamicEvaluator(search_mode="eutectic")
    calphad_eval = CALPHADEvaluator(
        database_path=selected_db["path"],
        search_mode="eutectic",
        eutectic_num_points=args.temperature_points,
    )

    print()
    print("[1/4] Global sampling...")
    global_candidates = sample_global_candidates(
        elements=elements,
        sample_count=args.samples,
        min_concentration=args.min_concentration,
        max_concentration=args.max_concentration,
        dirichlet_alpha=args.dirichlet_alpha,
        rng=rng,
    )
    print(f"Generated {len(global_candidates)} valid global candidates.")

    print("[2/4] Simple eutectic prefilter...")
    ranked_global = evaluate_simple_candidates(global_candidates, simple_eval)
    if args.direct_calphad_all:
        prefiltered = ranked_global
        print(f"Selected all {len(prefiltered)} candidates for CALPHAD.")
    else:
        prefiltered = ranked_global[:args.prefilter_top]
        print(f"Selected top {len(prefiltered)} candidates for CALPHAD.")

    print("[3/4] CALPHAD eutectic evaluation on global top set...")
    global_calphad_ranked = evaluate_calphad_candidates(prefiltered, calphad_eval, label="global")
    seed_candidates = global_calphad_ranked[:args.refine_seeds]
    print(f"Using top {len(seed_candidates)} CALPHAD candidates as local refinement seeds.")

    print("[4/4] Local refinement around best seeds...")
    local_candidates: List[Alloy] = []
    for seed_alloy in seed_candidates:
        local_candidates.extend(
            sample_local_variants(
                seed_alloy=seed_alloy,
                sample_count=args.refine_samples_per_seed,
                min_concentration=args.min_concentration,
                max_concentration=args.max_concentration,
                sigma=args.local_sigma,
                rng=rng,
            )
        )
    print(f"Generated {len(local_candidates)} local variants.")

    local_ranked = evaluate_simple_candidates(local_candidates, simple_eval)
    if args.direct_calphad_all:
        local_prefilter = local_ranked
    else:
        local_prefilter = local_ranked[:args.prefilter_top]
    local_calphad_ranked = evaluate_calphad_candidates(local_prefilter, calphad_eval, label="local")

    final_candidates = merge_unique_candidates(global_calphad_ranked, local_calphad_ranked)
    final_candidates = sorted(
        final_candidates,
        key=lambda alloy: alloy.properties.get("calphad_eutectic_score", 0.0),
        reverse=True,
    )

    print_top_candidates(final_candidates, args.final_top)

    result = {
        "screening_mode": "eutectic",
        "elements": elements,
        "database": database_to_record(selected_db),
        "config": {
            "samples": args.samples,
            "prefilter_top": args.prefilter_top,
            "final_top": args.final_top,
            "refine_seeds": args.refine_seeds,
            "refine_samples_per_seed": args.refine_samples_per_seed,
            "min_concentration": args.min_concentration,
            "max_concentration": args.max_concentration,
            "dirichlet_alpha": args.dirichlet_alpha,
            "local_sigma": args.local_sigma,
            "temperature_points": args.temperature_points,
            "seed": args.seed,
            "direct_calphad_all": args.direct_calphad_all,
        },
        "summary": {
            "global_candidates": len(global_candidates),
            "global_calphad_evaluated": len(prefiltered),
            "local_candidates": len(local_candidates),
            "local_calphad_evaluated": len(local_prefilter),
            "final_unique_candidates": len(final_candidates),
        },
        "top_candidates": [
            alloy_to_record(rank=index, alloy=alloy)
            for index, alloy in enumerate(final_candidates[:args.final_top], start=1)
        ],
    }

    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)

    print()
    print(f"Saved result to {args.output}")


if __name__ == "__main__":
    main()
