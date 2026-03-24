import gc
import multiprocessing as mp
import os
import sys

os.environ["QUACC_RESULTS_DIR"] = "/tmp"

import numpy as np  # noqa: E402
import torch  # noqa: E402
from ase.io import read  # noqa: E402
from ase.optimize import FIRE  # noqa: E402
from pymatgen.io.ase import AseAtomsAdaptor  # noqa: E402
from quacc.recipes.mlp.core import relax_job  # noqa: E402
from quacc.recipes.mlp.phonons import phonon_flow  # noqa: E402


def clean_gpu_cache():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def phonon_task(atoms):
    is_cpu = not torch.cuda.is_available()
    try:
        result1 = relax_job(
            atoms,
            method="fairchem",
            model_name="eSEN-30M-OAM",
            local_cache="./fairchem_cache/",
            cpu=is_cpu,
            opt_params={"fmax": 1e-2, "optimizer": FIRE},
            relax_cell=True,
        )
        torch.cuda.empty_cache()
    except Exception:
        return np.nan

    try:
        result2 = phonon_flow(
            result1["atoms"],
            method="fairchem",
            job_params={
                "all": dict(
                    model_name="eSEN-30M-OAM",
                    local_cache="./fairchem_cache/",
                    cpu=is_cpu,
                ),
            },
            min_lengths=10.0,
        )
        struc = AseAtomsAdaptor.get_structure(atoms)
        heat_capacity = (
            result2["results"]["thermal_properties"]["heat_capacity"][30]
            / struc.composition.weight
        )
        return heat_capacity
    except Exception:
        return np.nan


if __name__ == "__main__":
    atoms_list = read(sys.argv[1], index=":")
    with mp.Pool(processes=int(sys.argv[3])) as pool:
        results = pool.map(phonon_task, atoms_list)

    results = np.array(results, dtype=float)
    np.savetxt(sys.argv[2], results, fmt="%.6f")
