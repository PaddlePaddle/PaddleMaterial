# Copyright (C) 2026 Suzhou National Laboratory and Baidu PaddlePaddle team
# This code was jointly developed by Suzhou National Laboratory and Baidu PaddlePaddle team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

NUM_ELEMENTS: int = 98
NUM_SPACE_GROUPS: int = 230
LATTICE_COMPONENTS: int = 6
MAX_WYCKOFF_SITES: int = 27

EPSILON: float = 1e-6
MAX_NEIGHBORS: int = 20

lattice_parameter_ranges = {
    "mp_20": {
        "min_lattice_length": 2.0,
        "max_lattice_length": 133.0,
        "min_lattice_angle": 60.0,
        "max_lattice_angle": 135.0,
    },
    "mpts_52": {
        "min_lattice_length": 0.98,
        "max_lattice_length": 189.5,
        "min_lattice_angle": 60.0,
        "max_lattice_angle": 135.0,
    },
}
max_atoms_per_dataset = {
    "mp_20": 20,
    "mpts_52": 52,
}

chemical_symbols = [
    # 0
    "X",
    # 1
    "H", "He",
    # 2
    "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    # 3
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    # 4
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr",
    # 5
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "I", "Xe",
    # 6
    "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy",
    "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt",
    "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn",
    # 7
    "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf",
    "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
    "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
]

OFFSET_LIST = [
    [-1, -1, -1], [-1, -1, 0], [-1, -1, 1],
    [-1, 0, -1],  [-1, 0, 0],  [-1, 0, 1],
    [-1, 1, -1],  [-1, 1, 0],  [-1, 1, 1],
    [0, -1, -1],  [0, -1, 0],  [0, -1, 1],
    [0, 0, -1],   [0, 0, 0],   [0, 0, 1],
    [0, 1, -1],   [0, 1, 0],   [0, 1, 1],
    [1, -1, -1],  [1, -1, 0],  [1, -1, 1],
    [1, 0, -1],   [1, 0, 0],   [1, 0, 1],
    [1, 1, -1],   [1, 1, 0],   [1, 1, 1],
]

PRETRAINED_WEIGHT_URLS = {
    "mp_20": {
        "diffusion": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/SGEquiDiff/mp_20_best_diffusion_snapshot.pdparams",
        "lattice": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/SGEquiDiff/mp_20_best_lattice_snapshot.pdparams",
        "space_group": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/SGEquiDiff/mp_20_best_space_group_snapshot.pdparams",
        "wyckoff": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/SGEquiDiff/mp_20_best_wyckoff-transformer_snapshot.pdparams",
    },
    "mpts_52": {
        "diffusion": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/SGEquiDiff/mpts_52_best_diffusion_snapshot.pdparams",
        "lattice": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/SGEquiDiff/mpts_52_best_lattice_snapshot.pdparams",
        "space_group": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/SGEquiDiff/mpts_52_best_space_group_snapshot.pdparams",
        "wyckoff": "https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/SGEquiDiff/mpts_52_best_wyckoff-transformer_snapshot.pdparams",
    },
}

SUB_MODULE_WEIGHT_MAP = {
    "atom_coord_diffusion_model": "diffusion",
    "lattice_sampler": "lattice",
    "space_group_sampler": "space_group",
    "wyckoff_and_element_sampler": "wyckoff",
}
