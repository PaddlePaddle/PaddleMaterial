import os
import pickle

import pandas as pd
import pymatgen
import tqdm
from pymatgen.io.cif import CifParser

csv_file = "./data/2D_structure/ehull_0621.csv"
cif_path = "./data/2D_structure/cif_structure"

filter_thresh = 1

cif_names = os.listdir(cif_path)
csv_data = pd.read_csv(csv_file)
ehull_dict = {name: value for name, value in zip(csv_data["cif"], csv_data["ehull"])}
energy_dict = {name: value for name, value in zip(csv_data["cif"], csv_data["energy"])}
formula_dict = {
    name: value for name, value in zip(csv_data["cif"], csv_data["formula"])
}

structures = []
ehulls = []
energys = []
formulas = []
for cif_name in tqdm.tqdm(cif_names):
    if not cif_name.endswith(".cif"):
        continue
    ehull = ehull_dict[cif_name.replace(".cif", "")]
    # if abs(ehull) > 5:
    #     continue
    cif_file = os.path.join(cif_path, cif_name)
    parser = CifParser(cif_file)
    structure = parser.get_structures()[0]
    structures.append(structure)
    ehulls.append(ehull)
    energys.append(energy_dict[cif_name.replace(".cif", "")])
    formulas.append([cif_name, formula_dict[cif_name.replace(".cif", "")]])

with open("./data/2D_structure/structures_0621.pickle", "wb") as f:
    pickle.dump(structures, f)

with open("./data/2D_structure/ehulls_0621.pickle", "wb") as f:
    pickle.dump(ehulls, f)

with open("./data/2D_structure/energys_0621.pickle", "wb") as f:
    pickle.dump(energys, f)

with open("./data/2D_structure/formulas_0621.pickle", "wb") as f:
    pickle.dump(formulas, f)
