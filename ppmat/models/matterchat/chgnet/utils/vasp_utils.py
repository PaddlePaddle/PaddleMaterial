# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from monty.io import reverse_readfile
from pymatgen.io.vasp.outputs import Oszicar, Vasprun

from ppmat.utils import logger

if TYPE_CHECKING:
    from pymatgen.core import Structure


def parse_vasp_dir(file_root):
    """Parse VASP output files into structures and labels."""
    try:
        oszicar = Oszicar(file_root + "/OSZICAR")
        vasprun_orig = Vasprun(file_root + "/vasprun.xml", exception_on_bad_xml=False)
        outcar_filename = file_root + "/OUTCAR"
    except Exception:
        oszicar = Oszicar(file_root + "/OSZICAR.gz")
        vasprun_orig = Vasprun(
            file_root + "/vasprun.xml.gz", exception_on_bad_xml=False
        )
        outcar_filename = file_root + "/OUTCAR.gz"

    charge = []
    mag_x = []
    mag_y = []
    mag_z = []
    header = []
    all_lines = []

    for line in reverse_readfile(outcar_filename):
        clean = line.strip()
        all_lines.append(clean)

    all_lines.reverse()
    read_charge = False
    read_mag_x = False
    read_mag_y = False
    read_mag_z = False
    mag_x_all = []
    ion_step_count = 0

    for clean in all_lines:
        if "magnetization (x)" in clean:
            ion_step_count += 1
        if read_charge or read_mag_x or read_mag_y or read_mag_z:
            if clean.startswith("# of ion"):
                header = re.split(r"\s{2,}", clean.strip())
                header.pop(0)
            else:
                m = re.match(r"\s*(\d+)\s+(([\d\.\-]+)\s+)+", clean)
                if m:
                    toks = [float(i) for i in re.findall(r"[\d\.\-]+", clean)]
                    toks.pop(0)
                    if read_charge:
                        charge.append(dict(zip(header, toks)))
                    elif read_mag_x:
                        mag_x.append(dict(zip(header, toks)))
                    elif read_mag_y:
                        mag_y.append(dict(zip(header, toks)))
                    elif read_mag_z:
                        mag_z.append(dict(zip(header, toks)))
                elif clean.startswith("tot"):
                    if ion_step_count == (len(mag_x_all) + 1):
                        mag_x_all.append(mag_x)
                    read_charge = False
                    read_mag_x = False
                    read_mag_y = False
                    read_mag_z = False
        if clean == "total charge":
            read_charge = True
            read_mag_x, read_mag_y, read_mag_z = False, False, False
        elif clean == "magnetization (x)":
            mag_x = []
            read_mag_x = True
            read_charge, read_mag_y, read_mag_z = False, False, False
        elif clean == "magnetization (y)":
            mag_y = []
            read_mag_y = True
            read_charge, read_mag_x, read_mag_z = False, False, False
        elif clean == "magnetization (z)":
            mag_z = []
            read_mag_z = True
            read_charge, read_mag_x, read_mag_y = False, False, False
        elif re.search("electrostatic", clean):
            read_charge, read_mag_x, read_mag_y, read_mag_z = (
                False,
                False,
                False,
                False,
            )

    if len(oszicar.ionic_steps) == len(mag_x_all):
        logger.warning("Unfinished OUTCAR")
        mag_x_all = mag_x_all
    elif len(oszicar.ionic_steps) == (len(mag_x_all) - 1):
        mag_x_all.pop(-1)

    n_atoms = len(vasprun_orig.ionic_steps[0]["structure"])
    dataset = {
        "structures": [i["structure"] for i in vasprun_orig.ionic_steps],
        "uncorrected_total_energies": [
            i["e_0_energy"] for i in vasprun_orig.ionic_steps
        ],
        "energies_per_atom": [
            i["e_0_energy"] / n_atoms for i in vasprun_orig.ionic_steps
        ],
        "forces": [i["forces"] for i in vasprun_orig.ionic_steps],
        "magmoms": [[i["tot"] for i in j] for j in mag_x_all],
    }
    if "stress" in vasprun_orig.ionic_steps[0]:
        dataset["stress"] = [i["stress"] for i in vasprun_orig.ionic_steps]
    else:
        dataset["stress"] = None

    return dataset


def solve_charge_by_mag(
    structure: Structure,
    default_ox: dict[str, float] | None = None,
    ox_ranges: dict[str, dict[tuple[float, float], int]] | None = None,
):
    """Solve oxidation states by magmom."""
    ox_list = []
    solved_ox = True
    default_ox = default_ox or {"Li": 1, "O": -2}
    ox_ranges = ox_ranges or {
        "Mn": {(0.5, 1.5): 2, (1.5, 2.5): 3, (2.5, 3.5): 4, (3.5, 4.2): 3, (4.2, 5): 2}
    }

    mag_key = (
        "final_magmom" if "final_magmom" in structure.site_properties else "magmom"
    )

    mag = structure.site_properties[mag_key]

    for site_i, site in enumerate(structure.sites):
        assigned = False
        if site.species_string in ox_ranges:
            for (minmag, maxmag), magox in ox_ranges[site.species_string].items():
                if mag[site_i] >= minmag and mag[site_i] < maxmag:
                    ox_list.append(magox)
                    assigned = True
                    break
        elif site.species_string in default_ox:
            ox_list.append(default_ox[site.species_string])
            assigned = True
        if not assigned:
            solved_ox = False

    if solved_ox:
        logger.debug(ox_list)
        structure.add_oxidation_state_by_site(ox_list)
        return structure
    return None
