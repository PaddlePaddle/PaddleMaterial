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
import numpy as np
from pymatgen.io.cif import CifWriter
from ppmat.models.matinvent.rewards.base import Calculator


class DFTCalc(Calculator):
    def calc(self, samples, label="tmp"):
        struc_list = samples[0]
        cif_dir = os.path.join(self.root_dir, label)
        os.makedirs(cif_dir, exist_ok=True)
        for i, struc in enumerate(struc_list):
            CifWriter(struc).write_file(os.path.abspath(os.path.join(cif_dir, f"{i}.cif")))
            os.makedirs(os.path.join(self.root_dir, label, f"{i:02d}"), exist_ok=True)
        results = np.full(len(struc_list), np.nan)
        np.savetxt(os.path.join(self.root_dir, f"{label}.txt"), results, fmt="%.6f")
        return results
