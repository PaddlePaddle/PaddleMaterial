import argparse
import json
import os.path as osp
import random

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_path", type=str)
    parser.add_argument("--ratio", type=list, default=[0.9, 0.05, 0.05])
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    data_lines = []
    with open(args.src_path, "r") as f:
        for line in f:
            data_point = json.loads(line)
            data_lines.append(data_point)
    print("Total number of samples: {}".format(len(data_lines)))
    random.seed(args.seed)
    random.shuffle(data_lines)
    train_size = int(len(data_lines) * args.ratio[0])
    val_size = int(len(data_lines) * args.ratio[1])
    test_size = len(data_lines) - train_size - val_size
    assert train_size + val_size + test_size == len(data_lines)
    print("Train size: {}".format(train_size))
    print("Val size: {}".format(val_size))
    print("Test size: {}".format(test_size))

    data_lines_train = data_lines[:train_size]
    data_lines_val = data_lines[train_size : train_size + val_size]
    data_lines_test = data_lines[train_size + val_size :]

    save_dir = osp.dirname(args.src_path)
    train_file = osp.join(save_dir, "train.jsonl")
    with open(train_file, "w") as f:
        for dp in data_lines_train:
            f.write(json.dumps(dp) + "\n")
    val_file = osp.join(save_dir, "val.jsonl")
    with open(val_file, "w") as f:
        for dp in data_lines_val:
            f.write(json.dumps(dp) + "\n")
    test_file = osp.join(save_dir, "test.jsonl")
    with open(test_file, "w") as f:
        for dp in data_lines_test:
            f.write(json.dumps(dp) + "\n")
