import os
import sys

os.environ["QUACC_RESULTS_DIR"] = "/tmp"
import multiprocessing as mp  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from ase.io import read  # noqa: E402
from quacc.recipes.mlp.elastic import elastic_tensor_flow  # noqa: E402


def bulk_task(atoms):
    is_cpu = not torch.cuda.is_available()
    try:
        result = elastic_tensor_flow(
            atoms,
            job_params={
                "all": dict(
                    method="fairchem",
                    model_name="eSEN-30M-OAM",
                    local_cache="./fairchem_cache/",
                    cpu=is_cpu,
                ),
            },
        )
        return result["elasticity_doc"].bulk_modulus.voigt
    except Exception:
        return np.nan


if __name__ == "__main__":
    atoms_list = read(sys.argv[1], index=":")
    with mp.Pool(processes=int(sys.argv[3])) as pool:
        results = pool.map(bulk_task, atoms_list)

    results = np.array(results, dtype=float)
    np.savetxt(sys.argv[2], results, fmt="%.6f")
