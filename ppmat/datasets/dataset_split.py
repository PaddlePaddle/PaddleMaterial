import argparse
import os
import random

import pandas as pd

# supported file format
SUPPORTED_FORMATS = {"json": pd.read_json, "csv": pd.read_csv}


def save_data(data, path, file_format="json"):
    if file_format == "json":
        data.to_json(path, orient="records", lines=True)
    elif file_format == "csv":
        data.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported file format: {file_format}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_path", type=str, required=True, help="Path to the source data file."
    )
    parser.add_argument(
        "--ratio",
        type=float,
        nargs=3,
        default=[0.9, 0.05, 0.05],
        help="Train/Validation/Test split ratio.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=SUPPORTED_FORMATS.keys(),
        default="json",
        help="Input file format.",
    )

    args = parser.parse_args()

    # check split ratio
    if sum(args.ratio) != 1.0:
        raise ValueError("The sum of the split ratios must be 1.0.")

    # load datas
    if not os.path.exists(args.src_path):
        raise FileNotFoundError(f"Source file not found: {args.src_path}")

    print(f"Loading data from {args.src_path}...")
    data = SUPPORTED_FORMATS[args.format](args.src_path)

    # check data format to ensure at least one required column exists
    required_columns = ["structures", "smiles"]
    if not any(col in data.columns for col in required_columns):
        raise ValueError(
            f"Missing required column. At least one of {required_columns} must exist."
        )

    # set random seed
    random.seed(args.seed)

    # caculate dataset size
    total_samples = len(data)
    train_size = int(total_samples * args.ratio[0])
    val_size = int(total_samples * args.ratio[1])
    test_size = total_samples - train_size - val_size

    print(f"Train size: {train_size}, Val size: {val_size}, Test size: {test_size}")

    # shuffle data order
    print("Shuffling data...")
    idx = list(range(total_samples))
    random.shuffle(idx)

    # split index
    train_idx = idx[:train_size]
    val_idx = idx[train_size : train_size + val_size]
    test_idx = idx[train_size + val_size :]

    # split data
    print("Splitting data...")
    train_data = data.iloc[train_idx]
    val_data = data.iloc[val_idx]
    test_data = data.iloc[test_idx]

    # save data
    base_name, ext = os.path.splitext(args.src_path)

    for split_name, split_data in zip(
        ["train", "val", "test"], [train_data, val_data, test_data]
    ):
        save_path = f"{base_name}_{split_name}.{args.format}"
        print(f"Saving {split_name} data to {save_path}...")
        save_data(split_data, save_path, file_format=args.format)

    print("Data splitting complete!")
