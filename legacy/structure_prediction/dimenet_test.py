import numpy as np
import pandas as pd
from metrics.gen_metircs import prop_model_eval
from p_tqdm import p_map
from pymatgen.core.structure import Structure


def get_gt_crys_ori(cif):
    structure = Structure.from_str(cif, fmt="cif")
    lattice = structure.lattice
    crys_array_dict = {
        "frac_coords": structure.frac_coords,
        "atom_types": np.array([_.Z for _ in structure.species]),
        "lengths": np.array(lattice.abc),
        "angles": np.array(lattice.angles),
    }
    return crys_array_dict


gt_file_path = "./data/mp_20/test.csv"
csv = pd.read_csv(gt_file_path)


gt_crys = p_map(get_gt_crys_ori, csv["cif"])

gt = csv["formation_energy_per_atom"].tolist()
preds = prop_model_eval(
    cfg_path="./data/prop_models/mp20/hparams_paddle.yaml",
    weights_path="./data/prop_models/mp20/epoch=839-step=89039_paddle.pdparams",
    crystal_array_list=gt_crys,
)
print(np.abs(np.asarray(preds) - np.asarray(gt)).mean())
