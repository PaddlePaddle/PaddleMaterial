import os
import pickle

import pandas as pd
import pymatgen
import tqdm
from ase.io import read
from ase.io import write
from pymatgen.io.cif import CifParser


def preprocess_mp():

    csv_file = "./data_bak/3D_structure/mp.csv"
    cif_path = "./data_bak/3D_structure/mp"

    filter_thresh = 1

    cif_names = os.listdir(cif_path)
    csv_data = pd.read_csv(csv_file)
    ehull_dict = {
        name: value
        for name, value in zip(csv_data["material_id"], csv_data["energy_above_hull"])
    }
    energy_dict = {
        name: value
        for name, value in zip(csv_data["material_id"], csv_data["energy_per_atom"])
    }
    formula_dict = {
        name: value for name, value in zip(csv_data["material_id"], csv_data["formula"])
    }
    import pdb

    pdb.set_trace()

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

    with open("./data/3D_structure/structures_mp_0708.pickle", "wb") as f:
        pickle.dump(structures, f)

    with open("./data/3D_structure/ehulls_mp_0708.pickle", "wb") as f:
        pickle.dump(ehulls, f)

    with open("./data/3D_structure/energys_mp_0708.pickle", "wb") as f:
        pickle.dump(energys, f)

    with open("./data/3D_structure/formulas_mp_0708.pickle", "wb") as f:
        pickle.dump(formulas, f)


def preprocess_oqmd():

    csv_file = "./data_bak/3D_structure/all-standard.csv"
    cif_path = "./data_bak/3D_structure/oqmd-all_cif"

    filter_thresh = 1

    cif_names = os.listdir(cif_path)
    csv_data = pd.read_csv(csv_file)
    ehull_dict = {
        str(name): value
        for name, value in zip(csv_data["oqmd_id"], csv_data["delta_e"])
    }
    energy_dict = {
        str(name): value for name, value in zip(csv_data["oqmd_id"], csv_data["energy"])
    }
    formula_dict = {
        str(name): value
        for name, value in zip(csv_data["oqmd_id"], csv_data["formula"])
    }
    import pdb

    pdb.set_trace()

    structures = []
    ehulls = []
    energys = []
    formulas = []
    for cif_name in tqdm.tqdm(cif_names):
        if not cif_name.endswith(".cif"):
            continue
        key = cif_name.replace(".cif", "").replace("OQMD-", "")
        ehull = ehull_dict[key]
        # if abs(ehull) > 5:
        #     continue
        cif_file = os.path.join(cif_path, cif_name)
        parser = CifParser(cif_file)
        structure = parser.get_structures()[0]
        structures.append(structure)
        ehulls.append(ehull)
        energys.append(energy_dict[key])
        formulas.append([cif_name, formula_dict[key]])

    with open("./data/3D_structure/structures_oqmd_0708.pickle", "wb") as f:
        pickle.dump(structures, f)

    with open("./data/3D_structure/ehulls_oqmd_0708.pickle", "wb") as f:
        pickle.dump(ehulls, f)

    with open("./data/3D_structure/energys_oqmd_0708.pickle", "wb") as f:
        pickle.dump(energys, f)

    with open("./data/3D_structure/formulas_oqmd_0708.pickle", "wb") as f:
        pickle.dump(formulas, f)


def convert_vasp_to_cif():
    mp = "./data_bak/3D_structure/oqmd-all"
    mp_save = "./data_bak/3D_structure/oqmd-all_cif"
    os.makedirs(mp_save, exist_ok=True)
    for filename in tqdm.tqdm(os.listdir(mp)):
        if filename.endswith(".vasp"):
            inputfile = os.path.join(mp, filename)
            outputfile = os.path.join(mp_save, filename.split(".")[0] + ".cif")

            try:
                vasp_file = read(inputfile, format="vasp")
                write(outputfile, vasp_file, format="cif")
            except Exception as e:
                print(f"Failed to convert {filename} due to {e}")


if __name__ == "__main__":
    preprocess_mp()
    # convert_vasp_to_cif()
    # preprocess_oqmd()
