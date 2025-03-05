import random

import numpy as np
import paddle
from omegaconf import DictConfig
from paddle.io import DataLoader

from mattergen.common.data.collate import collate
from mattergen.common.data.collate_pp import DefaultCollator
from mattergen.common.data.dataset import CrystalDataset


def worker_init_fn(id: int):
    """
    DataLoaders workers init function.

    Initialize the numpy.random seed correctly for each worker, so that
    random augmentations between workers and/or epochs are not identical.

    If a global seed is set, the augmentations are deterministic.

    https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    """
    uint64_seed = paddle.get_rng_state()[0].current_seed()
    ss = np.random.SeedSequence([uint64_seed])
    np.random.seed(ss.generate_state(4))
    random.seed(uint64_seed)


# pytorch_lightning.LightningDataModule
class CrystDataModule:
    def __init__(
        self,
        train_dataset: CrystalDataset,
        num_workers: DictConfig,
        batch_size: DictConfig,
        val_dataset: (CrystalDataset | None) = None,
        test_dataset: (CrystalDataset | None) = None,
        **_,
    ):
        super().__init__()
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.datasets = [train_dataset, val_dataset, test_dataset]

    def train_dataloader(self, shuffle: bool = True) -> paddle.io.DataLoader:
        # return paddle.io.DataLoader(
        #     dataset=self.train_dataset,
        #     shuffle=shuffle,
        #     batch_size=self.batch_size.train,
        #     num_workers=self.num_workers.train,
        #     worker_init_fn=worker_init_fn,
        #     collate_fn=collate,
        # )
        train_dataloader = paddle.io.DataLoader(
            dataset=self.train_dataset,
            batch_sampler=paddle.io.DistributedBatchSampler(
                dataset=self.train_dataset,
                batch_size=self.batch_size.train,
                shuffle=shuffle,
                drop_last=False,
            ),
            worker_init_fn=worker_init_fn,
            # collate_fn=collate,
            collate_fn=DefaultCollator(),
            num_workers=self.num_workers.train,
            return_list=True,
        )
        return train_dataloader

    def val_dataloader(self, shuffle: bool = False) -> (DataLoader | None):
        # return (
        #     paddle.io.DataLoader(
        #         dataset=self.val_dataset,
        #         shuffle=shuffle,
        #         batch_size=self.batch_size.val,
        #         num_workers=self.num_workers.val,
        #         worker_init_fn=worker_init_fn,
        #         collate_fn=collate,
        #     )
        #     if self.val_dataset is not None
        #     else None
        # )
        if self.val_dataset is not None:
            val_dataloader = paddle.io.DataLoader(
                dataset=self.val_dataset,
                batch_sampler=paddle.io.BatchSampler(
                    dataset=self.val_dataset,
                    batch_size=self.batch_size.val,
                    shuffle=shuffle,
                    drop_last=False,
                ),
                worker_init_fn=worker_init_fn,
                # collate_fn=collate,
                collate_fn=DefaultCollator(),
                num_workers=self.num_workers.val,
                return_list=True,
            )
            return val_dataloader
        else:
            return None

    def test_dataloader(self, shuffle: bool = False) -> (DataLoader | None):
        # return (
        #     paddle.io.DataLoader(
        #         dataset=self.test_dataset,
        #         shuffle=shuffle,
        #         batch_size=self.batch_size.test,
        #         num_workers=self.num_workers.test,
        #         worker_init_fn=worker_init_fn,
        #         collate_fn=collate,
        #     )
        #     if self.test_dataset is not None
        #     else None
        # )
        if self.test_dataset is not None:
            test_dataloader = paddle.io.DataLoader(
                dataset=self.test_dataset,
                batch_sampler=paddle.io.BatchSampler(
                    dataset=self.test_dataset,
                    batch_size=self.batch_size.test,
                    shuffle=shuffle,
                    drop_last=False,
                ),
                worker_init_fn=worker_init_fn,
                # collate_fn=collate,
                collate_fn=DefaultCollator(),
                num_workers=self.num_workers.test,
                return_list=True,
            )
            return test_dataloader
        else:
            return None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(self.datasets={self.datasets!r}, self.num_workers={self.num_workers!r}, self.batch_size={self.batch_size!r})"
