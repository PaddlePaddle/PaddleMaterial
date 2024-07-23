import os
import pickle
import random
from collections import defaultdict

import pandas as pd
import pymatgen
import tqdm
from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifParser

csv_file = "./ehull_0621.csv"
cif_path = "/root/host/ssd3/zhangzhimin04/workspaces_118/PP4Materials/structure_prediction/data_bak/paddle_project/data/2d_structure/cif_structure"
save_path = "./2d_structure_csv_ehull_200_condition"
os.makedirs(save_path, exist_ok=True)

dist_file = "./mean_distances.csv"
property_file = "./output_data.csv"

filter_thresh = 1

cif_names = os.listdir(cif_path)

csv_data = pd.read_csv(csv_file)
dist_data = pd.read_csv(dist_file)
property_data = pd.read_csv(property_file)

csv_records = csv_data.to_dict(orient="records")
dist_records = dist_data.to_dict(orient="records")
property_records = property_data.to_dict(orient="records")

csv_records_dict = {
    data["cif"]: {k: v for k, v in data.items() if k != "cif"} for data in csv_records
}
dist_records_dict = {
    data["file"].replace(".cif", ""): {k: v for k, v in data.items() if k != "file"}
    for data in dist_records
}
property_records_dict = {
    data["cif"]: {k: v for k, v in data.items() if k != "cif"}
    for data in property_records
}

random.seed(42)
random.shuffle(cif_names)

split = [0.9, 0.05, 0.05]
result_data = defaultdict(list)

for cif_name in tqdm.tqdm(cif_names):
    if not cif_name.endswith(".cif"):
        continue

    key = cif_name.replace(".cif", "")
    ehull = csv_records_dict[key]["ehull"]
    if ehull > 0.2:
        continue
    cif_file = os.path.join(cif_path, cif_name)
    parser = CifParser(cif_file)
    structure = parser.get_structures()[0]
    sturcture = structure.to(fmt="cif")

    result_data["material_id"].append(cif_name)
    result_data["cif"].append(sturcture)

    for csv_key in csv_data.keys():
        if csv_key == "cif":
            continue
        try:
            result_data[csv_key].append(csv_records_dict[key][csv_key])
        except KeyError:
            result_data[csv_key].append("")

    for dist_key in dist_data.keys():
        if dist_key == "file":
            continue
        try:
            result_data[dist_key].append(dist_records_dict[key][dist_key])
        except KeyError:
            result_data[dist_key].append("")

    for property_key in property_data.keys():
        if property_key in ["cif", "formula", "energy"]:
            continue
        try:
            result_data[property_key].append(property_records_dict[key][property_key])
        except KeyError:
            result_data[property_key].append("")

train_num = int(len(result_data["material_id"]) * split[0])
val_num = int(len(result_data["material_id"]) * split[1])
test_num = len(result_data["material_id"]) - train_num - val_num

train_data = {key: value[:train_num] for key, value in result_data.items()}
val_data = {
    key: value[train_num : train_num + val_num] for key, value in result_data.items()
}
test_data = {key: value[-test_num:] for key, value in result_data.items()}

df = pd.DataFrame(train_data)
df.to_csv(os.path.join(save_path, "train.csv"))
df = pd.DataFrame(val_data)
df.to_csv(os.path.join(save_path, "val.csv"))
df = pd.DataFrame(test_data)
df.to_csv(os.path.join(save_path, "test.csv"))
