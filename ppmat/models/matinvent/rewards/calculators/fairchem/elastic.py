# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys

os.environ["QUACC_RESULTS_DIR"] = "/tmp"
import multiprocessing as mp

import numpy as np
import torch
from ase.io import read
from quacc.recipes.mlp.elastic import elastic_tensor_flow


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
