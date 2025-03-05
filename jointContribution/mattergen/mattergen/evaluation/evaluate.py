from mattergen.evaluation.metrics.evaluator import MetricsEvaluator
from mattergen.evaluation.reference.reference_dataset import ReferenceDataset
# from mattergen.evaluation.utils.relaxation import relax_structures
from mattergen.evaluation.utils.structure_matcher import (
    DefaultDisorderedStructureMatcher, DisorderedStructureMatcher,
    OrderedStructureMatcher)
from pymatgen.core.structure import Structure


def evaluate(
    structures: list[Structure],
    relaxed_structures: list[Structure] | None = None,
    relax: bool = False,
    energies: (list[float] | None) = None,
    reference: (ReferenceDataset | None) = None,
    structure_matcher: (
        OrderedStructureMatcher | DisorderedStructureMatcher
    ) = DefaultDisorderedStructureMatcher(),
    save_as: (str | None) = None,
    n_failed_jobs: int = 0,
) -> dict[str, float | int]:
    """Evaluate the structures against a reference dataset."""
    if relax and energies is not None:
        raise ValueError("Cannot accept energies if relax is True.")
    if relax:
        raise NotImplementedError("Relaxing structures is currently not supported.")
        # relaxed_structures, energies = relax_structures(
        #     structures, device=device, load_path=potential_load_path
        # )
    else:
        if relaxed_structures is None:
            relaxed_structures = structures


    evaluator = MetricsEvaluator.from_structures_and_energies(
        structures=relaxed_structures,
        energies=energies,
        original_structures=structures,
        reference=reference,
        structure_matcher=structure_matcher,
        n_failed_jobs=n_failed_jobs
    )
    return evaluator.compute_metrics(
        metrics=evaluator.available_metrics, save_as=save_as, pretty_print=True
    )
