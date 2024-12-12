import argparse
import json

import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_path", type=str)
    parser.add_argument("--id_path", type=str)

    args = parser.parse_args()
    data = pd.read_json(args.src_path)

    with open(args.id_path, "r") as f:
        id_data = json.load(f)

    train_size = int(len(id_data["id_train"]))
    val_size = int(len(id_data["id_val"]))
    test_size = int(len(id_data["id_test"]))

    print(f"Train size: {train_size}, Val size: {val_size}, Test size: {test_size}")

    idx = list(range(len(data["structure"])))

    train_idx = []
    val_idx = []
    test_idx = []

    for i in range(len(data["structure"])):
        if data["material_id"][i] in id_data["id_train"]:
            train_idx.append(i)
        elif data["material_id"][i] in id_data["id_val"]:
            val_idx.append(i)
        else:
            test_idx.append(i)

    train_data = {
        "structure": data["structure"][train_idx].tolist(),
        "material_id": data["material_id"][train_idx].tolist(),
        "formation_energy_per_atom": data["formation_energy_per_atom"][
            train_idx
        ].tolist(),
        "band_gap": data["band_gap"][train_idx].tolist(),
        "G": data["G"][train_idx].tolist(),
        "K": data["K"][train_idx].tolist(),
    }
    val_data = {
        "structure": data["structure"][val_idx].tolist(),
        "material_id": data["material_id"][val_idx].tolist(),
        "formation_energy_per_atom": data["formation_energy_per_atom"][
            val_idx
        ].tolist(),
        "band_gap": data["band_gap"][val_idx].tolist(),
        "G": data["G"][val_idx].tolist(),
        "K": data["K"][val_idx].tolist(),
    }
    test_data = {
        "structure": data["structure"][test_idx].tolist(),
        "material_id": data["material_id"][test_idx].tolist(),
        "formation_energy_per_atom": data["formation_energy_per_atom"][
            test_idx
        ].tolist(),
        "band_gap": data["band_gap"][test_idx].tolist(),
        "G": data["G"][test_idx].tolist(),
        "K": data["K"][test_idx].tolist(),
    }

    save_file_base = args.src_path.rsplit(".", 1)[0]

    df = pd.DataFrame(train_data)
    df.to_json(f"{save_file_base}_train.json")
    print(f"Saved train data to {save_file_base}_train.json")

    df = pd.DataFrame(val_data)
    df.to_json(f"{save_file_base}_val.json")
    print(f"Saved val data to {save_file_base}_val.json")

    df = pd.DataFrame(test_data)
    df.to_json(f"{save_file_base}_test.json")
    print(f"Saved test data to {save_file_base}_test.json")
