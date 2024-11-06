import argparse
import random

import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_path", type=str)
    parser.add_argument("--ratio", type=list, default=[0.9, 0.05, 0.05])
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    data = pd.read_json(args.src_path)
    random.seed(args.seed)

    train_size = int(len(data["structure"]) * args.ratio[0])
    val_size = int(len(data["structure"]) * args.ratio[1])
    test_size = len(data["structure"]) - train_size - val_size
    print(f"Train size: {train_size}, Val size: {val_size}, Test size: {test_size}")

    idx = list(range(len(data["structure"])))
    random.shuffle(idx)
    train_idx = idx[:train_size]
    val_idx = idx[train_size : train_size + val_size]
    test_idx = idx[train_size + val_size :]

    train_data = {
        "structure": data["structure"][train_idx].tolist(),
        "material_id": data["material_id"][train_idx].tolist(),
        "formation_energy_per_atom": data["formation_energy_per_atom"][
            train_idx
        ].tolist(),
    }
    val_data = {
        "structure": data["structure"][val_idx].tolist(),
        "material_id": data["material_id"][val_idx].tolist(),
        "formation_energy_per_atom": data["formation_energy_per_atom"][
            val_idx
        ].tolist(),
    }
    test_data = {
        "structure": data["structure"][test_idx].tolist(),
        "material_id": data["material_id"][test_idx].tolist(),
        "formation_energy_per_atom": data["formation_energy_per_atom"][
            test_idx
        ].tolist(),
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
