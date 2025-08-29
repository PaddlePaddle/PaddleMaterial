import os
from hydra.utils import instantiate 

from ppmat.utils import logger

from ase.build import bulk
from pymatgen.io.ase import AseAtomsAdaptor


def get_structure_from_file(work_dir, system, predictor):
    system["file_path"] = os.path.join(work_dir, system["file_path"])
    logger.info( f"Loading structures from files: {system['file_path']}." )
    files, structures = predictor.collect_structures(file_path=system['file_path'])
    return files, structures

def get_structure_from_ase(system):
    files, structures = [], []
    for i, s in enumerate(system["structures"]):
        element = s["element"]
        atom = bulk(element)
        if "repeat" in s:
            atom = atom.repeat(s["repeat"])
        structure = AseAtomsAdaptor().get_structure(atom)
        formula = atom.get_chemical_formula()  # Get chemical formula
        structures.append(structure)
        files.append(f"structure_{i}_{formula}")
    logger.info(f"Using ASE provided structures (count: {len(structures)})")
    return files, structures

def prepare_structures(cfg, predictor):
    system = instantiate(cfg.system)
    if system["interface"] == "load_file":
        work_dir = cfg.run.work_dir
        files, structures = get_structure_from_file(work_dir, system, predictor)
    elif system["interface"] == "ase":
        files, structures = get_structure_from_ase(system)
    else:
        pass
    return files, structures