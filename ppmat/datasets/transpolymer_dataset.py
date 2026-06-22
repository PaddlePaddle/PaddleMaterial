import numpy as np
import pandas as pd
import paddle
from paddle.io import Dataset


class TransPolymerCsvDataset(Dataset):
    """CSV dataset for TransPolymer downstream regression.

    The input CSV is expected to contain the polymer sequence in the first column and
    the regression target in the second column.
    """

    def __init__(
        self,
        file_path,
        tokenizer,
        blocksize=411,
        label_mean=None,
        label_std=None,
    ):
        super().__init__()
        self.data = pd.read_csv(file_path)
        self.tokenizer = tokenizer
        self.blocksize = blocksize
        self.label_mean = label_mean
        self.label_std = label_std

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        seq = str(row.iloc[0])
        label = float(row.iloc[1])
        if self.label_mean is not None and self.label_std is not None:
            label = (label - self.label_mean) / self.label_std
        encoding = self.tokenizer(
            seq,
            add_special_tokens=True,
            max_length=self.blocksize,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
        )
        return {
            "input_ids": np.asarray(encoding["input_ids"], dtype="int64"),
            "attention_mask": np.asarray(encoding["attention_mask"], dtype="int64"),
            "labels": np.asarray([label], dtype="float32"),
        }


def transpolymer_collate_fn(batch):
    return {
        "input_ids": paddle.to_tensor([item["input_ids"] for item in batch], dtype="int64"),
        "attention_mask": paddle.to_tensor(
            [item["attention_mask"] for item in batch], dtype="int64"
        ),
        "labels": paddle.to_tensor([item["labels"] for item in batch], dtype="float32"),
    }
