import os
import pickle
import random

import pandas as pd
import pymatgen
import tqdm
from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifParser

csv_file = "./ehull_0621.csv"
cif_path = "./cif_structure"
save_path = "./2d_structure_csv_ehull_200"
os.makedirs(save_path, exist_ok=True)

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

split = [0.9, 0.05, 0.05]


structures = []
ehulls = []
energys = []
formulas = []
material_id = []

zs = []
for cif_name in tqdm.tqdm(cif_names):
    if not cif_name.endswith(".cif"):
        continue

    # if cif_name != 'In2Co2Te5_164_8419.vasp.cif':
    #     continue
    # import pdb;pdb.set_trace()

    ehull = ehull_dict[cif_name.replace(".cif", "")]
    if ehull > 0.2:
        continue
    cif_file = os.path.join(cif_path, cif_name)
    parser = CifParser(cif_file)
    structure = parser.get_structures()[0]

    # import pdb;pdb.set_trace()
    # structure.lattice._matrix.setflags(write=True)
    # # if structure.lattice._matrix[-1][-1] < 0:
    # #     structure.lattice._matrix[-1][-1] += 25
    # # else:
    # #     structure.lattice._matrix[-1][-1] -= 25
    # structure.lattice._matrix[-1][-1] /= 10
    # structure.lattice._matrix.setflags(write=False)

    # coords = (structure.frac_coords - 0.3 )/0.4
    # coords = coords.clip(0, 1)
    # for i, site in enumerate(structure._sites):
    #     site.c = coords[i, 2]

    # species = structure.species
    # species = [s.name for s in species]
    # new_structure = Structure(lattice=structure.lattice, species=species,
    #     coords=coords,
    #     )
    # import pdb;pdb.set_trace()
    # structure = new_structure

    structures.append(structure.to(fmt="cif"))
    ehulls.append(ehull)
    energys.append(energy_dict[cif_name.replace(".cif", "")])
    formulas.append(formula_dict[cif_name.replace(".cif", "")])
    material_id.append(cif_name)

import pdb

pdb.set_trace()

train_num = int(len(material_id) * split[0])
val_num = int(len(material_id) * split[1])
test_num = len(material_id) - train_num - val_num

train_data = {
    "material_id": material_id[:train_num],
    "energy": energys[:train_num],
    "cif": structures[:train_num],
    "ehull": ehulls[:train_num],
    "formula": formulas[:train_num],
}
val_data = {
    "material_id": material_id[train_num : train_num + val_num],
    "energy": energys[train_num : train_num + val_num],
    "cif": structures[train_num : train_num + val_num],
    "ehull": ehulls[train_num : train_num + val_num],
    "formula": formulas[train_num : train_num + val_num],
}
test_data = {
    "material_id": material_id[-test_num:],
    "energy": energys[-test_num:],
    "cif": structures[-test_num:],
    "ehull": ehulls[-test_num:],
    "formula": formulas[-test_num:],
}

df = pd.DataFrame(train_data)
df.to_csv(os.path.join(save_path, "train.csv"))
df = pd.DataFrame(val_data)
df.to_csv(os.path.join(save_path, "val.csv"))
df = pd.DataFrame(test_data)
df.to_csv(os.path.join(save_path, "test.csv"))
