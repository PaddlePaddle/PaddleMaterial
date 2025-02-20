import argparse
import itertools
import json
import os
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import paddle
import pandas as pd
import smact
from matminer.featurizers.composition.composite import ElementProperty
from matminer.featurizers.site.fingerprint import CrystalNNFingerprint
from p_tqdm import p_map
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.composition import Composition
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from pyxtal import pyxtal
from smact.screening import pauling_test
from tqdm import tqdm
from utils import paddle_aux

sys.path.append(".")

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

CrystalNNFP = CrystalNNFingerprint.from_preset("ops")
CompFP = ElementProperty.from_preset("magpie")


def get_crystals_list(frac_coords, atom_types, lengths, angles, num_atoms):
    """
    args:
        frac_coords: (num_atoms, 3)
        atom_types: (num_atoms)
        lengths: (num_crystals)
        angles: (num_crystals)
        num_atoms: (num_crystals)
    """
    assert frac_coords.shape[0] == atom_types.shape[0] == num_atoms.sum()
    assert lengths.shape[0] == angles.shape[0] == num_atoms.shape[0]
    start_idx = 0
    crystal_array_list = []
    for batch_idx, num_atom in enumerate(num_atoms.tolist()):
        start_0 = frac_coords.shape[0] + start_idx if start_idx < 0 else start_idx
        cur_frac_coords = paddle.slice(
            frac_coords, [0], [start_0], [start_0 + num_atom]
        )
        start_1 = atom_types.shape[0] + start_idx if start_idx < 0 else start_idx
        cur_atom_types = paddle.slice(atom_types, [0], [start_1], [start_1 + num_atom])
        cur_lengths = lengths[batch_idx]
        cur_angles = angles[batch_idx]
        crystal_array_list.append(
            {
                "frac_coords": cur_frac_coords.detach().cpu().numpy(),
                "atom_types": cur_atom_types.detach().cpu().numpy(),
                "lengths": cur_lengths.detach().cpu().numpy(),
                "angles": cur_angles.detach().cpu().numpy(),
            }
        )
        start_idx = start_idx + num_atom
    return crystal_array_list


def smact_validity(comp, count, use_pauling_test=True, include_alloys=True):
    elem_symbols = tuple([chemical_symbols[elem] for elem in comp])
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
    compositions = []
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
        self.frac_coords = crys_array_dict["frac_coords"]
        self.atom_types = crys_array_dict["atom_types"]
        self.lengths = crys_array_dict["lengths"]
        self.angles = crys_array_dict["angles"]
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


class RecEval(object):
    def __init__(self, pred_crys, gt_crys, stol=0.5, angle_tol=10, ltol=0.3):
        assert len(pred_crys) == len(gt_crys)
        self.matcher = StructureMatcher(stol=stol, angle_tol=angle_tol, ltol=ltol)
        self.preds = pred_crys
        self.gts = gt_crys

    def get_match_rate_and_rms(self):
        def process_one(pred, gt, is_valid):
            if not is_valid:
                return None
            try:
                rms_dist = self.matcher.get_rms_dist(pred.structure, gt.structure)
                rms_dist = None if rms_dist is None else rms_dist[0]
                return rms_dist
            except Exception:
                return None

        validity = [(c1.valid and c2.valid) for c1, c2 in zip(self.preds, self.gts)]
        rms_dists = []
        for i in tqdm(range(len(self.preds))):
            rms_dists.append(process_one(self.preds[i], self.gts[i], validity[i]))
        rms_dists = np.array(rms_dists)
        match_rate = sum(rms_dists != None) / len(self.preds)
        mean_rms_dist = rms_dists[rms_dists != None].mean()
        return {"match_rate": match_rate, "rms_dist": mean_rms_dist}

    def get_metrics(self):
        metrics = {}
        metrics.update(self.get_match_rate_and_rms())
        return metrics


class RecEvalBatch(object):
    def __init__(self, pred_crys, gt_crys, stol=0.5, angle_tol=10, ltol=0.3):
        self.matcher = StructureMatcher(stol=stol, angle_tol=angle_tol, ltol=ltol)
        self.preds = pred_crys
        self.gts = gt_crys
        self.batch_size = len(self.preds)

    def get_match_rate_and_rms(self):
        def process_one(pred, gt, is_valid):
            if not is_valid:
                return None
            try:
                rms_dist = self.matcher.get_rms_dist(pred.structure, gt.structure)
                rms_dist = None if rms_dist is None else rms_dist[0]
                return rms_dist
            except Exception:
                return None

        rms_dists = []
        self.all_rms_dis = np.zeros((self.batch_size, len(self.gts)))
        for i in tqdm(range(len(self.preds[0]))):
            tmp_rms_dists = []
            for j in range(self.batch_size):
                rmsd = process_one(
                    self.preds[j][i], self.gts[i], self.preds[j][i].valid
                )
                self.all_rms_dis[j][i] = rmsd
                if rmsd is not None:
                    tmp_rms_dists.append(rmsd)
            if len(tmp_rms_dists) == 0:
                rms_dists.append(None)
            else:
                rms_dists.append(np.min(tmp_rms_dists))
        rms_dists = np.array(rms_dists)
        match_rate = sum(rms_dists != None) / len(self.preds[0])
        mean_rms_dist = rms_dists[rms_dists != None].mean()
        return {"match_rate": match_rate, "rms_dist": mean_rms_dist}

    def get_metrics(self):
        metrics = {}
        metrics.update(self.get_match_rate_and_rms())
        return metrics


def get_file_paths(root_path, task, label="", suffix="pt"):
    if args.label == "":
        out_name = f"eval_{task}.{suffix}"
    else:
        out_name = f"eval_{task}_{label}.{suffix}"
    out_name = os.path.join(root_path, out_name)
    return out_name


def get_crystal_array_list(file_path, batch_idx=0):
    data = paddle.load(file_path)
    if batch_idx == -1:
        batch_size = tuple(data["frac_coords"].shape)[0]
        crys_array_list = []
        for i in range(batch_size):
            tmp_crys_array_list = get_crystals_list(
                data["frac_coords"][i],
                data["atom_types"][i],
                data["lengths"][i],
                data["angles"][i],
                data["num_atoms"][i],
            )
            crys_array_list.append(tmp_crys_array_list)
    elif batch_idx == -2:
        crys_array_list = get_crystals_list(
            data["frac_coords"],
            data["atom_types"],
            data["lengths"],
            data["angles"],
            data["num_atoms"],
        )
    else:
        crys_array_list = get_crystals_list(
            data["frac_coords"][batch_idx],
            data["atom_types"][batch_idx],
            data["lengths"][batch_idx],
            data["angles"][batch_idx],
            data["num_atoms"][batch_idx],
        )
    if "input_data_batch" in data:
        batch = data["input_data_batch"]
        if isinstance(batch, dict):
            true_crystal_array_list = get_crystals_list(
                batch["frac_coords"],
                batch["atom_types"],
                batch["lengths"],
                batch["angles"],
                batch["num_atoms"],
            )
        else:
            true_crystal_array_list = get_crystals_list(
                batch.frac_coords,
                batch.atom_types,
                batch.lengths,
                batch.angles,
                batch.num_atoms,
            )
    else:
        true_crystal_array_list = None
    return crys_array_list, true_crystal_array_list


def main(args):
    all_metrics = {}

    recon_file_path = args.root_path

    batch_idx = -1 if args.multi_eval else 0
    crys_array_list, true_crystal_array_list = get_crystal_array_list(
        recon_file_path, batch_idx=batch_idx
    )

    gt_crys = p_map(lambda x: Crystal(x), true_crystal_array_list)

    if not args.multi_eval:
        pred_crys = p_map(lambda x: Crystal(x), crys_array_list)
    else:
        pred_crys = []
        for i in range(len(crys_array_list)):
            print(f"Processing batch {i}")
            pred_crys.append(p_map(lambda x: Crystal(x), crys_array_list[i]))
    if args.multi_eval:
        rec_evaluator = RecEvalBatch(pred_crys, gt_crys)
    else:
        rec_evaluator = RecEval(pred_crys, gt_crys)
    recon_metrics = rec_evaluator.get_metrics()
    all_metrics.update(recon_metrics)
    print(all_metrics)
    if args.label == "":
        metrics_out_file = "eval_metrics.json"
    else:
        metrics_out_file = f"eval_metrics_{args.label}.json"
    metrics_out_file = os.path.join(os.path.dirname(args.root_path), metrics_out_file)
    if Path(metrics_out_file).exists():
        with open(metrics_out_file, "r") as f:
            written_metrics = json.load(f)
            if isinstance(written_metrics, dict):
                written_metrics.update(all_metrics)
            else:
                with open(metrics_out_file, "w") as f:
                    json.dump(all_metrics, f)
        if isinstance(written_metrics, dict):
            with open(metrics_out_file, "w") as f:
                json.dump(written_metrics, f)
    else:
        with open(metrics_out_file, "w") as f:
            json.dump(all_metrics, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", required=True)
    parser.add_argument("--label", default="")
    parser.add_argument("--tasks", nargs="+", default=["csp"])
    parser.add_argument("--gt_file", default="")
    parser.add_argument("--multi_eval", action="store_true")
    args = parser.parse_args()
    main(args)
