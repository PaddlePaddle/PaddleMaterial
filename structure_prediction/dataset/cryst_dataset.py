import sys

sys.path.append(
    "/root/host/ssd3/zhangzhimin04/workspaces_118/PP4Materials/structure_prediction/paddle_project/utils"
)
import os
import pickle

import chemparse
import numpy as np
import paddle
import paddle_aux
import pandas as pd
from common.data_utils import add_scaled_lattice_prop
from common.data_utils import preprocess
from p_tqdm import p_map
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifWriter
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pyxtal.symmetry import Group

chemical_symbols = [
    "X",
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
    "Nh",
    "Fl",
    "Mc",
    "Lv",
    "Ts",
    "Og",
]


class CrystDataset(paddle.io.Dataset):
    def __init__(
        self,
        name,
        path,
        prop,
        niggli,
        primitive,
        graph_method,
        preprocess_workers,
        lattice_scale_method,
        save_path,
        tolerance,
        use_space_group,
        use_pos_index,
        **kwargs,
    ):
        super().__init__()
        self.path = path
        self.name = name
        self.df = pd.read_csv(path)
        self.prop = prop
        self.niggli = niggli
        self.primitive = primitive
        self.graph_method = graph_method
        self.lattice_scale_method = lattice_scale_method
        self.use_space_group = use_space_group
        self.use_pos_index = use_pos_index
        self.tolerance = tolerance
        self.preprocess(save_path, preprocess_workers, prop)
        add_scaled_lattice_prop(self.cached_data, lattice_scale_method)

        self.lattice_scaler = None
        self.scaler = None

    def preprocess(self, save_path, preprocess_workers, prop):
        if os.path.exists(save_path):
            with open(save_path, "rb") as f:
                self.cached_data = pickle.load(f)
            # self.cached_data = paddle.load(path=save_path)
        else:
            cached_data = preprocess(
                self.path,
                preprocess_workers,
                niggli=self.niggli,
                primitive=self.primitive,
                graph_method=self.graph_method,
                prop_list=[prop],
                use_space_group=self.use_space_group,
                tol=self.tolerance,
            )
            with open(save_path, "wb") as f:
                pickle.dump(cached_data, f)
            # paddle.save(obj=cached_data, path=save_path)
            self.cached_data = cached_data

    def __len__(self) -> int:
        return len(self.cached_data)

    def __getitem__(self, index):
        data_dict = self.cached_data[index]
        # prop = self.scaler.transform(data_dict[self.prop])
        (
            frac_coords,
            atom_types,
            lengths,
            angles,
            edge_indices,
            to_jimages,
            num_atoms,
        ) = data_dict["graph_arrays"]

        data = dict(
            frac_coords=frac_coords.astype("float32"),
            atom_types=atom_types,
            lengths=lengths.reshape(1, -1).astype("float32"),
            angles=angles.reshape(1, -1).astype("float32"),
            edge_index=edge_indices.T,
            to_jimages=to_jimages,
            num_atoms=num_atoms,
            num_bonds=tuple(edge_indices.shape)[0],
            num_nodes=num_atoms,
        )  # y=prop.view(1, -1))

        if self.use_space_group:
            data["spacegroup"] = [data_dict["spacegroup"]]
            data["ops"] = data_dict["wyckoff_ops"]

            data["anchor_index"] = data_dict["anchors"]
        if self.use_pos_index:
            pos_dic = {}
            indexes = []
            for atom in atom_types:
                pos_dic[atom] = pos_dic.get(atom, 0) + 1
                indexes.append(pos_dic[atom] - 1)
            data["index"] = indexes
        return data

    def __repr__(self) -> str:
        return f"CrystDataset(self.name={self.name!r}, self.path={self.path!r})"


class SampleDataset(paddle.io.Dataset):
    def __init__(self, formula, num_evals):
        super().__init__()
        self.formula = formula
        self.num_evals = num_evals
        self.get_structure()

    def get_structure(self):
        self.composition = chemparse.parse_formula(self.formula)
        chem_list = []
        for elem in self.composition:
            num_int = int(self.composition[elem])
            chem_list.extend([chemical_symbols.index(elem)] * num_int)
        self.chem_list = chem_list

    def __len__(self) -> int:
        return self.num_evals

    def __getitem__(self, index):

        data = dict(
            atom_types=self.chem_list,
            num_atoms=len(self.chem_list),
            num_nodes=len(self.chem_list),
        )  # y=prop.view(1, -1))

        return data
