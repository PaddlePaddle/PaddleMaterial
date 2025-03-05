import json
from pathlib import Path
from typing import Literal

import fire
import numpy as np
from mattergen.common.utils.eval_utils import load_structures
from mattergen.evaluation.evaluate import evaluate
from mattergen.evaluation.utils.structure_matcher import (
    DefaultDisorderedStructureMatcher, DefaultOrderedStructureMatcher)


def main(
    structures_path: str,
    relaxed_structures_path: (str | None) = None,
    relax: bool = False,
    energies_path: (str | None) = None,
    structure_matcher: Literal["ordered", "disordered"] = "disordered",
    save_as: (str | None) = None,
):
    structures = load_structures(Path(structures_path))
    if relaxed_structures_path is not None:
        relaxed_structures = load_structures(Path(relaxed_structures_path))
    else:
        relaxed_structures = None
    energies = np.load(energies_path) if energies_path else None
    structure_matcher = (
        DefaultDisorderedStructureMatcher()
        if structure_matcher == "disordered"
        else DefaultOrderedStructureMatcher()
    )

    reference_ele = {
        'Sc', 'F', 'Pd', 'Ti', 'Nd', 'P', 'Ca', 'Ru', 'Sn', 'Sm', 'As', 'O', 'Be', 
        'Au', 'Cd', 'Pt', 'Bi', 'Y', 'Si', 'Se', 'Cu', 'Sb', 'In', 'Br', 'Hf', 'I', 
        'Ir', 'La', 'Ba', 'Er', 'Lu', 'W', 'Mo', 'Li', 'Ge', 'Pb', 'Hg', 'Tl', 'Ho', 
        'Ta', 'Co', 'Ga', 'Nb', 'Fe', 'Mg', 'B', 'N', 'Cr', 'Sr', 'Rh', 'Yb', 'Ce', 
        'Ni', 'Re', 'V', 'Os', 'H', 'Rb', 'Pr', 'Al', 'Eu', 'Cl', 'Gd', 'S', 'Ag', 
        'Mn', 'Na', 'K', 'Zn', 'Cs', 'C', 'Te', 'Tb', 'Dy', 'Tm', 'Zr'}
    new_structures = []
    index = []
    for i, structure in enumerate(structures):
        flag = True
        for site in structure:
            if site.specie.symbol not in reference_ele:
                flag = False
                break
        if flag:
            new_structures.append(structure)
            index.append(i)
    print(f'{len(structures)} -> {len(new_structures)}')
    n_failed_jobs = len(structures) - len(new_structures)
    structures = new_structures
    if energies is not None:
        energies = [energies[i] for i in index]

    metrics = evaluate(
        structures=structures,
        relaxed_structures=relaxed_structures,
        relax=relax,
        energies=energies,
        structure_matcher=structure_matcher,
        save_as=save_as,
        n_failed_jobs=n_failed_jobs,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    fire.Fire(main)
