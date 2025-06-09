import json
import os
import random


def split_dataset_by_mpid(
    input_json: str,
    output_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = 42,
) -> None:
    """
    Randomly splits a JSON dataset into train/val/test subsets based on mp-id

    Args:
        input_json: Path to input JSON file
        output_dir: Path to output directory
        train_ratio: Training set ratio (default: 0.8)
        val_ratio: Validation set ratio (default: 0.1)
        test_ratio: Test set ratio (default: 0.1), if 0, no test set will be created
        random_seed: Random seed for reproducibility (default: 42)
    """
    # check if the sum of ratios is 1
    total_ratio = train_ratio + val_ratio + test_ratio
    if not (total_ratio == 1.0):
        raise ValueError(f"Total ratio should be equal to 1 but got {total_ratio}.")

    # load the json file
    with open(input_json, "r") as f:
        data = json.load(f)

    # get all mp-ids, shuffle them a
    mp_ids = list(data.keys())
    random.seed(random_seed)
    random.shuffle(mp_ids)
    print(f"Loaded {len(mp_ids)} entries")

    # split them into train/val/test sets
    total = len(mp_ids)
    train_split = int(total * train_ratio)
    val_split = train_split + int(total * val_ratio)

    train_mpids = set(mp_ids[:train_split])
    print(f"Train set size: {len(train_mpids)}")
    val_mpids = set(mp_ids[train_split:val_split])
    print(f"Val set size: {len(val_mpids)}")
    if test_ratio > 0:
        test_mpids = set(mp_ids[val_split:])
        print(f"Test set size: {len(test_mpids)}")

    # make output dir
    os.makedirs(output_dir, exist_ok=True)

    # save function
    def save_split(mpid_set: set, filename: str):
        subset = {mpid: data[mpid] for mpid in mpid_set}
        with open(os.path.join(output_dir, filename), "w") as f:
            json.dump(subset, f, indent=2)

    save_split(train_mpids, "train.json")
    print(f"Saved training splits to {output_dir}")
    save_split(val_mpids, "val.json")
    print(f"Saved validation splits to {output_dir}")
    if test_ratio > 0:
        save_split(test_mpids, "test.json")
        print(f"Saved testing splits to {output_dir}")


# This code is used to split the Mptrj_2022.9_full.json dataset into train/val/test
# subsets based on mp-id
if __name__ == "__main__":
    split_dataset_by_mpid(
        input_json="./data/MPtrj_2022.9_full.json",
        output_dir="./data/MPtrj_2022.9_full",
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        random_seed=42,
    )
