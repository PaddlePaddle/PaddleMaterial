import os
import pickle
import sys

import chemparse
import numpy as np
import paddle
import pandas as pd
from dataset.utils import add_scaled_lattice_prop
from dataset.utils import chemical_symbols
from dataset.utils import preprocess
from p_tqdm import p_map
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifWriter
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pyxtal.symmetry import Group
from utils import paddle_aux


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
        else:
            if not isinstance(prop, list):
                prop = [prop]
            cached_data = preprocess(
                self.path,
                preprocess_workers,
                niggli=self.niggli,
                primitive=self.primitive,
                graph_method=self.graph_method,
                prop_list=prop,
                use_space_group=self.use_space_group,
                tol=self.tolerance,
            )
            with open(save_path, "wb") as f:
                pickle.dump(cached_data, f)
            self.cached_data = cached_data

    def __len__(self) -> int:
        return len(self.cached_data)

    def __getitem__(self, index):

        data_dict = self.cached_data[index]
        (
            frac_coords,
            atom_types,
            lengths,
            angles,
            edge_indices,
            to_jimages,
            num_atoms,
        ) = data_dict["graph_arrays"]
        if isinstance(self.prop, list):
            prop_value = np.array(
                [data_dict[key] for key in self.prop], dtype=np.float32
            ).reshape(1, -1)
        else:
            prop_value = np.array(data_dict[self.prop], dtype=np.float32).reshape(1, -1)

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
            prop=prop_value,
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
    def __init__(self, formula, num_evals, lengths=None, angles=None):
        super().__init__()
        self.formula = formula
        self.num_evals = num_evals
        self.lengths = np.asarray(lengths, dtype=np.float32).reshape(1, -1)
        self.angles = np.asarray(angles, dtype=np.float32).reshape(1, -1)
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
            lengths=self.lengths,
            angles=self.angles,
        )  # y=prop.view(1, -1))

        return data


class GenDataset(paddle.io.Dataset):
    def __init__(self, total_num, property_value=None):
        super().__init__()
        distribution = [
            0.0,
            0.000865519852861625,
            0.006815968841285297,
            0.05788164016012117,
            0.06599588878069891,
            0.04619712214648924,
            0.14075516607162178,
            0.060694579681921455,
            0.09628908363085578,
            0.04013848317645786,
            0.0935843340906632,
            0.02693930542031808,
            0.08330628583793141,
            0.013307367737747485,
            0.04295142269825814,
            0.014605647517039922,
            0.03884020339716542,
            0.004976739153954344,
            0.03667640376501136,
            0.0037866493562696093,
            0.043816942551119765,
            0.000865519852861625,
            0.010278048252731797,
            0.0011900897976847343,
            0.015146597425078437,
            0.0004327599264308125,
            0.006599588878069891,
            0.0010818998160770314,
            0.007465108730931516,
            0.0010818998160770314,
            0.0038948393378773127,
            0.0003245699448231094,
            0.007897868657362328,
            0.0007573298712539219,
            0.0023801795953694686,
            0.0004327599264308125,
            0.006275018933246781,
            0.0005409499080385157,
            0.0034620794114465,
            0.0010818998160770314,
            0.006815968841285297,
            0.0,
            0.0005409499080385157,
            0.0,
            0.0007573298712539219,
            0.0,
            0.00010818998160770312,
            0.0,
            0.0004327599264308125,
            0.0,
            0.0,
            0.0,
            0.0003245699448231094,
            0.0,
            0.0,
            0.0,
            0.00021637996321540625,
            0.0,
            0.00021637996321540625,
            0.0,
            0.00021637996321540625,
            0.0,
            0.0,
            0.0,
            0.00021637996321540625,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.00010818998160770312,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.00010818998160770312,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.00010818998160770312,
            0.0,
            0.0,
            0.0,
            0.00010818998160770312,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.00010818998160770312,
        ]
        self.total_num = total_num
        self.distribution = distribution
        self.num_atoms = np.random.choice(
            len(self.distribution), total_num, p=self.distribution
        )
        self.property_value = property_value

    def __len__(self) -> int:
        return self.total_num

    def __getitem__(self, index):
        num_atom = self.num_atoms[index]
        data = dict(
            num_atoms=num_atom,
            num_nodes=num_atom,
        )  # y=prop.view(1, -1))
        if self.property_value is not None:
            prop = np.array(self.property_value, dtype=np.float32).reshape(1, -1)
            data["prop"] = prop
        return data
