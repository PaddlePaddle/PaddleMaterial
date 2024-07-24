import os
import pickle
import random

import matplotlib.pyplot as plt
import pandas as pd
import pymatgen
import tqdm
from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifParser

csv_file = "./ehull_0621.csv"
cif_path = "./cif_structure"

filter_thresh = 1

cif_names = os.listdir(cif_path)
csv_data = pd.read_csv(csv_file)
ehull_dict = {name: value for name, value in zip(csv_data["cif"], csv_data["ehull"])}
energy_dict = {name: value for name, value in zip(csv_data["cif"], csv_data["energy"])}
formula_dict = {
    name: value for name, value in zip(csv_data["cif"], csv_data["formula"])
}

random.seed(42)
random.shuffle(cif_names)


structures = []
ehulls = []
energys = []
formulas = []
material_id = []

zs = []
stds = []
for cif_name in tqdm.tqdm(cif_names):
    if not cif_name.endswith(".cif"):
        continue

    # if cif_name != 'In2Co2Te5_164_8419.vasp.cif':
    #     continue
    # import pdb;pdb.set_trace()

    ehull = ehull_dict[cif_name.replace(".cif", "")]
    cif_file = os.path.join(cif_path, cif_name)
    parser = CifParser(cif_file)
    structure = parser.get_structures()[0]
    stds.append(structure.frac_coords[:, -1].std())

    structures.append(structure.to(fmt="cif"))
    ehulls.append(ehull)
    energys.append(energy_dict[cif_name.replace(".cif", "")])
    formulas.append(formula_dict[cif_name.replace(".cif", "")])
    material_id.append(cif_name)


stds_sd = [stds[i] for i in range(len(stds)) if ehulls[i] < 0.2]
