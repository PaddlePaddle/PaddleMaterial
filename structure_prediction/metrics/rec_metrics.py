import itertools
import json
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd
import smact
from matminer.featurizers.composition.composite import ElementProperty
from matminer.featurizers.site.fingerprint import CrystalNNFingerprint
from p_tqdm import p_map
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.composition import Composition
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from smact.screening import pauling_test
from tqdm import tqdm

# Warning: the smact package version is 2.5.5,
# different version may cause slight differences in accuracy.

CHEMICAL_SYMBOLS = [
    # 0
    "X",
    # 1
    "H",
    "He",
    # 2
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    # 3
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    # 4
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
    # 5
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
    # 6
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
    # 7
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
]  # noqa
CrystalNNFP = CrystalNNFingerprint.from_preset("ops")
CompFP = ElementProperty.from_preset("magpie")


def smact_validity(comp, count, use_pauling_test=True, include_alloys=True):
    elem_symbols = tuple([CHEMICAL_SYMBOLS[elem] for elem in comp])
    space = smact.element_dictionary(elem_symbols)
    smact_elems = [e[1] for e in space.items()]
    electronegs = [e.pauling_eneg for e in smact_elems]
    ox_combos = [e.oxidation_states for e in smact_elems]
    if len(set(elem_symbols)) == 1:
        return True
    if include_alloys:
        is_metal_list = [(elem_s in smact.metals) for elem_s in elem_symbols]
        if all(is_metal_list):
            return True
    threshold = np.max(count)
    oxn = 1
    for oxc in ox_combos:
        oxn *= len(oxc)
    if oxn > 10000000.0:
        return False
    for ox_states in itertools.product(*ox_combos):
        stoichs = [(c,) for c in count]
        cn_e, cn_r = smact.neutral_ratios(
            ox_states, stoichs=stoichs, threshold=threshold
        )
        if cn_e:
            if use_pauling_test:
                try:
                    electroneg_OK = pauling_test(ox_states, electronegs)
                except TypeError:
                    electroneg_OK = True
            else:
                electroneg_OK = True
            if electroneg_OK:
                return True
    return False


def structure_validity(crystal, cutoff=0.5):
    dist_mat = crystal.distance_matrix
    dist_mat = dist_mat + np.diag(np.ones(tuple(dist_mat.shape)[0]) * (cutoff + 10.0))
    if dist_mat.min() < cutoff or crystal.volume < 0.1:
        return False
    else:
        return True


class Crystal(object):
    def __init__(self, crys_array_dict):
        self.frac_coords = np.array(crys_array_dict["frac_coords"])
        self.atom_types = np.array(crys_array_dict["atom_types"])
        self.lengths = np.array(crys_array_dict["lengths"])
        self.angles = np.array(crys_array_dict["angles"])
        self.dict = crys_array_dict
        if len(tuple(self.atom_types.shape)) > 1:
            self.dict["atom_types"] = np.argmax(self.atom_types, axis=-1) + 1
            self.atom_types = np.argmax(self.atom_types, axis=-1) + 1
        self.get_structure()
        self.get_composition()
        self.get_validity()
        self.get_fingerprints()

    def get_structure(self):
        if min(self.lengths.tolist()) < 0:
            self.constructed = False
            self.invalid_reason = "non_positive_lattice"
        if (
            np.isnan(self.lengths).any()
            or np.isnan(self.angles).any()
            or np.isnan(self.frac_coords).any()
        ):
            self.constructed = False
            self.invalid_reason = "nan_value"
        else:
            try:
                self.structure = Structure(
                    lattice=Lattice.from_parameters(
                        *(self.lengths.tolist() + self.angles.tolist())
                    ),
                    species=self.atom_types,
                    coords=self.frac_coords,
                    coords_are_cartesian=False,
                )
                self.constructed = True
            except Exception:
                self.constructed = False
                self.invalid_reason = "construction_raises_exception"
            if self.structure.volume < 0.1:
                self.constructed = False
                self.invalid_reason = "unrealistically_small_lattice"

    def get_composition(self):
        elem_counter = Counter(self.atom_types)
        composition = [
            (elem, elem_counter[elem]) for elem in sorted(elem_counter.keys())
        ]
        elems, counts = list(zip(*composition))
        counts = np.array(counts)
        counts = counts / np.gcd.reduce(counts)
        self.elems = elems
        self.comps = tuple(counts.astype("int").tolist())

    def get_validity(self):
        self.comp_valid = smact_validity(self.elems, self.comps)
        if self.constructed:
            self.struct_valid = structure_validity(self.structure)
        else:
            self.struct_valid = False
        self.valid = self.comp_valid and self.struct_valid

    def get_fingerprints(self):
        elem_counter = Counter(self.atom_types)
        comp = Composition(elem_counter)
        self.comp_fp = CompFP.featurize(comp)
        try:
            site_fps = [
                CrystalNNFP.featurize(self.structure, i)
                for i in range(len(self.structure))
            ]
        except Exception:
            self.valid = False
            self.comp_fp = None
            self.struct_fp = None
            return
        self.struct_fp = np.array(site_fps).mean(axis=0)


def get_gt_crys_ori(cif):
    structure = Structure.from_str(cif, fmt="cif")
    lattice = structure.lattice
    crys_array_dict = {
        "frac_coords": structure.frac_coords,
        "atom_types": np.array([_.Z for _ in structure.species]),
        "lengths": np.array(lattice.abc),
        "angles": np.array(lattice.angles),
    }
    return Crystal(crys_array_dict)


class RecMetrics:
    # the reconstruct metrics class
    def __init__(self, gt_file_path=None, stol=0.5, angle_tol=10, ltol=0.3):
        self.matcher = StructureMatcher(stol=stol, angle_tol=angle_tol, ltol=ltol)

        self.gt_file_path = gt_file_path
        if gt_file_path is not None:
            csv = pd.read_csv(self.gt_file_path)
            self.gt_crys = p_map(get_gt_crys_ori, csv["cif"])
        else:
            self.gt_crys = None

    def get_match_rate_and_rms(self, pred_crys, gt_crys):
        assert len(pred_crys) == len(gt_crys)

        def process_one(pred, gt, is_valid):
            if not is_valid:
                return None
            try:
                rms_dist = self.matcher.get_rms_dist(pred.structure, gt.structure)
                rms_dist = None if rms_dist is None else rms_dist[0]
                return rms_dist
            except Exception:
                return None

        validity = [(c1.valid and c2.valid) for c1, c2 in zip(pred_crys, gt_crys)]
        rms_dists = []
        for i in tqdm(range(len(pred_crys))):
            rms_dists.append(process_one(pred_crys[i], gt_crys[i], validity[i]))
        rms_dists = np.array(rms_dists)
        match_rate = sum(rms_dists is not None) / len(pred_crys)
        mean_rms_dist = rms_dists[rms_dists is not None].mean()
        return {"match_rate": match_rate, "rms_dist": mean_rms_dist}

    def load_jsonline(self, path):
        with open(path, "r") as f:
            lines = [json.loads(line.strip()) for line in f.readlines()]
        return lines

    def __call__(self, file_path_or_data) -> Any:
        if isinstance(file_path_or_data, str):
            data = self.load_jsonline(file_path_or_data)
        else:
            data = file_path_or_data

        prediction = [d["prediction"] for d in data]
        pred_crys = p_map(lambda x: Crystal(x), prediction)

        if self.gt_crys is None:
            ground_truth = [d["ground_truth"] for d in data]
            gt_crys = p_map(lambda x: Crystal(x), ground_truth)
        else:
            gt_crys = self.gt_crys

        recon_metrics = self.get_match_rate_and_rms(pred_crys, gt_crys)
        return recon_metrics


if __name__ == "__main__":
    metircs = RecMetrics("structure_prediction/data/mp_20/test.csv")
    print(metircs("structure_prediction/metrics/output.jsonl"))
