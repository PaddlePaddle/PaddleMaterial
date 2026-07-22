"""
使用遗传算法搜索多元共晶候选。

特点:
1. 直接复用 AdaptiveEvolutionEngine
2. 仅使用纯 CALPHAD 的 eutectic 适应度
3. 可选使用筛选结果作为初始种子，加快收敛
"""

import argparse
import json
import os
import random
import sys
import copy
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ppmatAgent.hea_crewai_agent.core.calphad_evaluator import CALPHADEvaluator
from ppmatAgent.hea_crewai_agent.core.evolution_engine import AdaptiveEvolutionEngine
from ppmatAgent.hea_crewai_agent.core.material import Alloy, Population
from ppmatAgent.hea_crewai_agent.core.tdb_registry import find_best_local_tdb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Genetic algorithm search for eutectic candidates.")
    parser.add_argument("--elements", default="Co,Cr,Fe,Ni,V", help="Comma-separated elements.")
    parser.add_argument("--population-size", type=int, default=16, help="GA population size.")
    parser.add_argument("--generations", type=int, default=8, help="GA generations.")
    parser.add_argument("--mutation-rate", type=float, default=0.18, help="Initial mutation rate.")
    parser.add_argument("--crossover-rate", type=float, default=0.8, help="Crossover rate.")
    parser.add_argument("--elite-ratio", type=float, default=0.2, help="Elite preservation ratio.")
    parser.add_argument("--min-concentration", type=float, default=0.10, help="Minimum atomic fraction.")
    parser.add_argument("--max-concentration", type=float, default=0.35, help="Maximum atomic fraction.")
    parser.add_argument("--temperature-points", type=int, default=11, help="CALPHAD temperature scan points.")
    parser.add_argument(
        "--init-mode",
        choices=["screening_seeds", "balanced"],
        default="balanced",
        help="Initial population source: prior screening seeds or near-equimolar balanced alloys.",
    )
    parser.add_argument(
        "--init-jitter",
        type=float,
        default=0.03,
        help="Gaussian jitter used for balanced initialization.",
    )
    parser.add_argument(
        "--seed-json",
        default="eutectic_screening_result.json",
        help="Optional screening result JSON used to seed initial population.",
    )
    parser.add_argument("--seed-top-k", type=int, default=5, help="Number of top seed candidates to import.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--output", default="eutectic_ga_result.json", help="Output JSON path.")
    parser.add_argument(
        "--calphad-curve-every",
        type=int,
        default=5,
        help="Validate the best candidate every N generations for the CALPHAD curve.",
    )
    parser.add_argument(
        "--calphad-curve-points",
        type=int,
        default=5,
        help="Temperature scan points used for the CALPHAD curve checkpoints.",
    )
    parser.add_argument(
        "--curve-output",
        default="eutectic_ga_curves.png",
        help="Output path for the evolution curve plot.",
    )
    parser.add_argument(
        "--plot-validation-curve",
        action="store_true",
        help="Plot the extra CALPHAD validation curve. Disabled by default in pure CALPHAD mode.",
    )
    parser.add_argument(
        "--calphad-refine-levels",
        type=int,
        default=2,
        help="Adaptive refinement levels used by CALPHAD validation.",
    )
    parser.add_argument(
        "--calphad-refine-points",
        type=int,
        default=7,
        help="Intermediate points inserted per refined CALPHAD interval.",
    )
    parser.add_argument(
        "--init-pool-multiplier",
        type=int,
        default=6,
        help="Build an initial CALPHAD-prescreen pool of population_size * multiplier candidates.",
    )
    parser.add_argument(
        "--prescreen-temperature-points",
        type=int,
        default=3,
        help="Pure CALPHAD prescreen scan points used only for initialization.",
    )
    return parser.parse_args()


def composition_signature(composition: Dict[str, float], decimals: int = 6):
    """Stable composition signature used for caching deterministic CALPHAD calls."""
    return tuple(sorted(
        (element, round(float(fraction), decimals))
        for element, fraction in composition.items()
    ))


class CachedAlloyEvaluator:
    """Cache deterministic evaluator outputs and replay properties onto new Alloy objects."""

    def __init__(self, evaluator: CALPHADEvaluator):
        self.evaluator = evaluator
        self.cache: Dict = {}

    def evaluate(self, alloy: Alloy) -> float:
        signature = composition_signature(alloy.composition)
        cached = self.cache.get(signature)
        if cached is not None:
            alloy.properties.update(copy.deepcopy(cached["properties"]))
            return cached["fitness"]

        fitness = self.evaluator.evaluate(alloy)
        self.cache[signature] = {
            "fitness": fitness,
            "properties": copy.deepcopy(alloy.properties),
        }
        return fitness


class SeededAdaptiveEvolutionEngine(AdaptiveEvolutionEngine):
    """Adaptive GA engine with optional seeded initial population."""

    def __init__(
        self,
        *args,
        seed_alloys: Optional[List[Alloy]] = None,
        init_mode: str = "screening_seeds",
        init_jitter: float = 0.03,
        initial_evaluator: Optional[Callable] = None,
        init_pool_multiplier: int = 6,
        seed: int = 42,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.seed_alloys = seed_alloys or []
        self.init_mode = init_mode
        self.init_jitter = init_jitter
        self.initial_evaluator = initial_evaluator
        self.init_pool_multiplier = max(1, int(init_pool_multiplier))
        self.rng = np.random.default_rng(seed)

    def _create_balanced_alloy(self) -> Alloy:
        """Create an alloy around equal fractions for all target elements."""
        base_fraction = 1.0 / len(self.target_elements)
        proposal = np.array(
            [base_fraction + self.rng.normal(0.0, self.init_jitter) for _ in self.target_elements],
            dtype=float
        )
        proposal = np.clip(proposal, 1e-6, None)
        proposal /= np.sum(proposal)
        composition = {
            element: float(fraction)
            for element, fraction in zip(self.target_elements, proposal)
        }
        return Alloy(composition=composition)

    def _clear_prescreen_properties(self, alloy: Alloy) -> Alloy:
        """Preserve coarse prescreen score in metadata while forcing full reevaluation later."""
        prescreen_fitness = alloy.properties.get("fitness")
        if prescreen_fitness is not None:
            alloy.metadata["initial_prescreen_fitness"] = prescreen_fitness
        alloy.properties = {}
        return alloy

    def _elite_count(self) -> int:
        """Always preserve at least one elite when elite_ratio is enabled."""
        if self.population_size <= 0 or self.elite_ratio <= 0:
            return 0
        return min(self.population_size, max(1, int(self.population_size * self.elite_ratio)))

    def initialize_population(self) -> Population:
        population = Population()

        used_signatures = set()
        candidate_alloys: List[Alloy] = []
        target_candidate_count = max(self.population_size, self.population_size * self.init_pool_multiplier)

        if self.init_mode == "balanced":
            candidate_alloys.append(
                Alloy(composition={element: 1.0 / len(self.target_elements) for element in self.target_elements})
            )
            while len(candidate_alloys) < target_candidate_count:
                candidate_alloys.append(self._create_balanced_alloy())
        else:
            candidate_alloys.extend(self.seed_alloys)
            while len(candidate_alloys) < target_candidate_count:
                candidate_alloys.append(self._create_random_alloy())

        unique_candidates: List[Alloy] = []
        for alloy in candidate_alloys:
            signature = tuple(sorted((key, round(value, 4)) for key, value in alloy.composition.items()))
            if signature in used_signatures:
                continue
            if not alloy.is_valid(self.constraints):
                continue
            used_signatures.add(signature)
            unique_candidates.append(alloy)

        if self.initial_evaluator is not None:
            for alloy in unique_candidates:
                prescreen_fitness = self.initial_evaluator(alloy)
                alloy.properties["fitness"] = prescreen_fitness

            unique_candidates.sort(
                key=lambda alloy: alloy.properties.get("fitness", 0.0),
                reverse=True,
            )

        selected_alloys = unique_candidates[:self.population_size]
        for alloy in selected_alloys:
            if self.initial_evaluator is not None:
                alloy = self._clear_prescreen_properties(alloy)
            population.add_alloy(alloy)

        return population

    def _snapshot_alloy(self, alloy: Alloy) -> Alloy:
        """Create a lightweight immutable snapshot for later validation."""
        return Alloy(
            composition=dict(alloy.composition),
            properties=dict(alloy.properties),
            metadata=dict(alloy.metadata),
            generation=alloy.generation,
            parent_ids=list(alloy.parent_ids),
        )

    def evolve(self, evaluator: Callable, strategy_agent: Optional[object] = None, verbose: bool = True) -> Alloy:
        """Track best individual snapshots alongside the built-in history arrays."""
        self.best_generation_alloys: List[Alloy] = []
        self.best_generation_numbers: List[int] = []
        self.history = {
            'best_fitness': [],
            'mean_fitness': [],
            'diversity': []
        }
        population = self.initialize_population()
        self._evaluate_population(population, evaluator)
        population.update_statistics()
        initial_best = population.get_best()
        self.history['best_fitness'].append(population.statistics['max_fitness'])
        self.history['mean_fitness'].append(population.statistics['mean_fitness'])
        self.history['diversity'].append(population.statistics['diversity'])
        self.best_generation_alloys.append(self._snapshot_alloy(initial_best))
        self.best_generation_numbers.append(0)
        global_best = self._snapshot_alloy(initial_best)

        for generation in range(self.max_generations):
            self.current_generation = generation
            best = population.get_best()
            current_best_fitness = best.properties['fitness']

            if verbose:
                print(
                    f"Generation {generation}: Best={current_best_fitness:.4f}, "
                    f"MutRate={self.mutation_rate:.3f}, Diversity={population.calculate_diversity():.4f}"
                )

            if self.best_fitness_history:
                if abs(current_best_fitness - self.best_fitness_history[-1]) < 0.001:
                    self.stagnation_counter += 1
                else:
                    self.stagnation_counter = 0

            self.best_fitness_history.append(current_best_fitness)

            if self.stagnation_counter > 10:
                self.mutation_rate = min(0.3, self.mutation_rate * 1.2)
                if verbose:
                    print(f"  -> 检测到停滞,增加变异率到 {self.mutation_rate:.3f}")
                self.stagnation_counter = 0
            elif self.stagnation_counter == 0:
                self.mutation_rate = max(0.05, self.mutation_rate * 0.95)

            parents = self._selection(population)
            offspring = []

            while len(offspring) < self.population_size - int(self.population_size * self.elite_ratio):
                parent1, parent2 = random.sample(parents, 2)

                if random.random() < self.crossover_rate:
                    child = self._crossover(parent1, parent2)
                else:
                    child = random.choice([parent1, parent2])

                if random.random() < self.mutation_rate:
                    child = self._mutate(child, strategy_agent)

                if child.is_valid(self.constraints):
                    offspring.append(child)

            elites = population.get_top_k(self._elite_count())
            new_population = Population(generation=generation + 1)

            for alloy in elites + offspring:
                new_population.add_alloy(alloy)

            self._evaluate_population(new_population, evaluator)
            new_population.update_statistics()
            generation_best = new_population.get_best()
            if generation_best.properties.get('fitness', 0.0) > global_best.properties.get('fitness', 0.0):
                global_best = self._snapshot_alloy(generation_best)

            self.history['best_fitness'].append(new_population.statistics['max_fitness'])
            self.history['mean_fitness'].append(new_population.statistics['mean_fitness'])
            self.history['diversity'].append(new_population.statistics['diversity'])
            self.best_generation_alloys.append(self._snapshot_alloy(generation_best))
            self.best_generation_numbers.append(generation + 1)

            population = new_population

        best_alloy = global_best
        if verbose:
            print(f"\n进化完成! 最佳适应度: {best_alloy.properties['fitness']:.4f}")

        return best_alloy

    def get_evolution_summary(self) -> Dict:
        """Report generation count without inflating it by the initial generation-0 snapshot."""
        if not self.history['best_fitness']:
            return {
                'total_generations': 0,
                'final_best_fitness': 0,
                'initial_best_fitness': 0,
                'improvement': 0,
                'final_diversity': 0,
            }
        return {
            'total_generations': max(0, len(self.history['best_fitness']) - 1),
            'final_best_fitness': self.history['best_fitness'][-1],
            'initial_best_fitness': self.history['best_fitness'][0],
            'improvement': self.history['best_fitness'][-1] - self.history['best_fitness'][0],
            'final_diversity': self.history['diversity'][-1],
        }


def load_seed_alloys(seed_json: str, seed_top_k: int) -> List[Alloy]:
    path = Path(seed_json)
    if not path.exists():
        return []

    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    top_candidates = data.get("top_candidates", [])[:seed_top_k]
    return [Alloy(composition=item["composition"]) for item in top_candidates if "composition" in item]


def alloy_to_record(alloy: Alloy) -> Dict:
    props = alloy.properties
    return {
        "alloy_id": alloy.alloy_id,
        "composition": {
            element: round(fraction, 6)
            for element, fraction in sorted(alloy.composition.items())
        },
        "fitness": props.get("fitness"),
        "search_fitness": props.get("search_fitness", props.get("fitness")),
        "validation_calphad_fitness": props.get("validation_calphad_fitness"),
        "calphad_search_mode": props.get("calphad_search_mode"),
        "calphad_fitness": props.get("calphad_fitness"),
        "calphad_temperature": props.get("calphad_temperature"),
        "calphad_liquidus_temp": props.get("calphad_liquidus_temp"),
        "calphad_solidus_temp": props.get("calphad_solidus_temp"),
        "calphad_freezing_range": props.get("calphad_freezing_range"),
        "calphad_liquidus_depression": props.get("calphad_liquidus_depression"),
        "calphad_best_liquid_fraction": props.get("calphad_best_liquid_fraction"),
        "calphad_eutectic_reaction_score": props.get("calphad_eutectic_reaction_score"),
        "calphad_phases": props.get("calphad_phases", {}),
        "evaluation_method": props.get("evaluation_method"),
        "simple_score": props.get("simple_score"),
        "calphad_score": props.get("calphad_score"),
    }


def save_evolution_plot(history: Dict[str, List[float]], output_path: str) -> None:
    import matplotlib.pyplot as plt

    generations = range(len(history["best_fitness"]))

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    axes[0].plot(generations, history["best_fitness"], label="Best Fitness", linewidth=2)
    axes[0].plot(generations, history["mean_fitness"], label="Mean Fitness", linewidth=2, alpha=0.85)
    calphad_curve = history.get("calphad_curve", {})
    if calphad_curve.get("generations") and calphad_curve.get("fitness"):
        axes[0].plot(
            calphad_curve["generations"],
            calphad_curve["fitness"],
            label="CALPHAD Validation",
            linewidth=2,
            linestyle="--",
            marker="o",
            color="crimson",
        )
    axes[0].set_title("Eutectic Search Evolution")
    axes[0].set_xlabel("Generation")
    axes[0].set_ylabel("Fitness")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(generations, history["diversity"], color="green", linewidth=2)
    axes[1].set_title("Population Diversity")
    axes[1].set_xlabel("Generation")
    axes[1].set_ylabel("Diversity")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    elements = [element.strip() for element in args.elements.split(",") if element.strip()]

    if args.init_mode == "screening_seeds":
        seed_alloys = load_seed_alloys(args.seed_json, args.seed_top_k)
    else:
        seed_alloys = []
    constraints = {
        "max_elements": len(elements),
        "min_concentration": args.min_concentration,
        "max_concentration": args.max_concentration,
        "allowed_elements": elements,
    }

    selected_db = find_best_local_tdb(elements, allow_simplified=False)
    if selected_db is None:
        raise RuntimeError(
            f"No full pycalphad-loadable database matches {elements}. "
            "A strict eutectic GA needs a matching full CALPHAD database."
        )

    prescreen_temperature_points = max(3, min(args.prescreen_temperature_points, args.temperature_points))
    prescreen_calphad_eval = CALPHADEvaluator(
        database_path=selected_db["path"],
        search_mode="eutectic",
        eutectic_num_points=prescreen_temperature_points,
        adaptive_temperature_refinement=False,
        refinement_levels=0,
        refinement_points_per_interval=0,
    )
    calphad_eval = CALPHADEvaluator(
        database_path=selected_db["path"],
        search_mode="eutectic",
        eutectic_num_points=args.temperature_points,
        adaptive_temperature_refinement=True,
        refinement_levels=args.calphad_refine_levels,
        refinement_points_per_interval=args.calphad_refine_points,
    )
    prescreen_runner = CachedAlloyEvaluator(prescreen_calphad_eval)
    calphad_runner = CachedAlloyEvaluator(calphad_eval)
    evaluate_fn = calphad_runner.evaluate

    curve_temperature_points = None
    curve_runner = None
    if args.plot_validation_curve:
        curve_temperature_points = max(args.calphad_curve_points, args.temperature_points)
        if curve_temperature_points == args.temperature_points:
            curve_runner = calphad_runner
        else:
            calphad_curve_eval = CALPHADEvaluator(
                database_path=selected_db["path"],
                search_mode="eutectic",
                eutectic_num_points=curve_temperature_points,
                adaptive_temperature_refinement=True,
                refinement_levels=args.calphad_refine_levels,
                refinement_points_per_interval=args.calphad_refine_points,
            )
            curve_runner = CachedAlloyEvaluator(calphad_curve_eval)

    engine = SeededAdaptiveEvolutionEngine(
        population_size=args.population_size,
        max_generations=args.generations,
        target_elements=elements,
        mutation_rate=args.mutation_rate,
        crossover_rate=args.crossover_rate,
        elite_ratio=args.elite_ratio,
        constraints=constraints,
        seed_alloys=seed_alloys,
        init_mode=args.init_mode,
        init_jitter=args.init_jitter,
        initial_evaluator=prescreen_runner.evaluate,
        init_pool_multiplier=args.init_pool_multiplier,
        seed=args.seed,
    )

    print("=" * 72)
    print("遗传算法共晶搜索")
    print("=" * 72)
    print(f"Elements: {elements}")
    print(f"Database: {selected_db['filename']}")
    print("Evaluator mode: pure_calphad")
    print(f"Init mode: {args.init_mode}")
    print(f"Population size: {args.population_size}")
    print(f"Generations: {args.generations}")
    print(f"Temperature points: {args.temperature_points}")
    print(f"Initial CALPHAD prescreen points: {prescreen_temperature_points}")
    print(f"Initial candidate pool multiplier: {args.init_pool_multiplier}")
    print(f"Plot validation curve: {args.plot_validation_curve}")
    print(
        f"Adaptive validation refinement: "
        f"levels={args.calphad_refine_levels}, points={args.calphad_refine_points}"
    )
    print(f"Imported seed alloys: {len(seed_alloys)}")
    print()

    best_alloy = engine.evolve(
        evaluator=evaluate_fn,
        strategy_agent=None,
        verbose=True,
    )

    best_alloy.properties["search_fitness"] = best_alloy.properties.get("fitness")
    validation_calphad_fitness = calphad_runner.evaluate(best_alloy)
    best_alloy.properties["validation_calphad_fitness"] = validation_calphad_fitness

    calphad_curve_generations: List[int] = []
    calphad_curve_fitness: List[float] = []
    if args.plot_validation_curve and curve_runner is not None:
        generation_numbers = getattr(
            engine,
            "best_generation_numbers",
            list(range(len(getattr(engine, "best_generation_alloys", []))))
        )
        for generation_number, alloy_snapshot in zip(generation_numbers, getattr(engine, "best_generation_alloys", [])):
            if generation_number % args.calphad_curve_every != 0 and generation_number != generation_numbers[-1]:
                continue
            validation_alloy = Alloy(composition=dict(alloy_snapshot.composition))
            calphad_curve_fitness.append(curve_runner.evaluate(validation_alloy))
            calphad_curve_generations.append(generation_number)

    history_with_calphad_curve = dict(engine.history)
    if args.plot_validation_curve:
        history_with_calphad_curve["calphad_curve"] = {
            "generations": calphad_curve_generations,
            "fitness": calphad_curve_fitness,
        }

    result = {
        "mode": "eutectic_ga",
        "elements": elements,
        "database": {
            "filename": selected_db["filename"],
            "path": selected_db["path"],
        },
        "config": {
            "population_size": args.population_size,
            "generations": args.generations,
            "mutation_rate": args.mutation_rate,
            "crossover_rate": args.crossover_rate,
            "elite_ratio": args.elite_ratio,
            "temperature_points": args.temperature_points,
            "init_mode": args.init_mode,
            "init_jitter": args.init_jitter,
            "evaluator_mode": "pure_calphad",
            "min_concentration": args.min_concentration,
            "max_concentration": args.max_concentration,
            "seed_json": args.seed_json,
            "seed_top_k": args.seed_top_k,
            "imported_seed_alloys": len(seed_alloys),
            "seed": args.seed,
            "calphad_curve_every": args.calphad_curve_every,
            "calphad_curve_points": curve_temperature_points,
            "plot_validation_curve": args.plot_validation_curve,
            "calphad_refine_levels": args.calphad_refine_levels,
            "calphad_refine_points": args.calphad_refine_points,
            "init_pool_multiplier": args.init_pool_multiplier,
            "prescreen_temperature_points": prescreen_temperature_points,
        },
        "evolution_summary": engine.get_evolution_summary(),
        "history": history_with_calphad_curve,
        "best_alloy": alloy_to_record(best_alloy),
    }

    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)

    save_evolution_plot(history_with_calphad_curve, args.curve_output)

    print()
    print("Best candidate")
    print("=" * 72)
    print(alloy_to_record(best_alloy))
    print()
    print(f"Saved result to {args.output}")
    print(f"Saved curve plot to {args.curve_output}")


if __name__ == "__main__":
    main()
