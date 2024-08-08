import functools

import numpy as np
import paddle


def collate(batch, target_keys):
    """
    custom batching function because batches have variable shape
    """
    batch = batch[0]
    inputs = {}
    targets = {}
    for key in batch:
        if key in target_keys:
            targets[key] = batch[key]
        else:
            inputs[key] = batch[key]
    return inputs, targets


class DataProvider:
    """
    Parameters
    ----------
        data_container: DataContainer
            Contains the dataset.
        ntrain: int
            Number of samples in the training set.
        nval: int
            Number of samples in the validation set.
        batch_size: int
            Number of samples to process at once.
        seed: int
            Seed for drawing samples into train and val set (and shuffle).
        random_split: bool
            If True put the samples randomly into the subsets else in order.
        shuffle: bool
            If True shuffle the samples after each epoch.
        sample_with_replacement: bool
            Sample data from the dataset with replacement.
        split: str/dict
            Overwrites settings of 'ntrain', 'nval', 'random_split' and 'sample_with_replacement'.
            If of type dict the dictionary is assumed to contain the index split of the subsets.
            If split is of type str then load the index split from the .npz-file.
            Dict and split file are assumed to have keys 'train', 'val', 'test'.
    """

    def __init__(
        self,
        data_container,
        ntrain: int,
        nval: int,
        batch_size: int = 1,
        seed: int = None,
        random_split: bool = False,
        shuffle: bool = True,
        sample_with_replacement: bool = False,
        split=None,
        **kwargs,
    ):
        self.kwargs = kwargs
        self.data_container = data_container
        self._ndata = len(data_container)
        self.batch_size = batch_size
        self.seed = seed
        self.random_split = random_split
        self.shuffle = shuffle
        self.sample_with_replacement = sample_with_replacement
        self._random_state = np.random.RandomState(seed=seed)
        if split is None:
            self.nsamples, self.idx = self._random_split_data(ntrain, nval)
        else:
            self.nsamples, self.idx = self._manual_split_data(split)

    def _manual_split_data(self, split):
        if isinstance(split, (dict, str)):
            if isinstance(split, str):
                assert split.endswith(
                    ".npz"
                ), "'split' has to be a .npz file if 'split' is of type str"
                split = np.load(split)
            keys = ["train", "val", "test"]
            for key in keys:
                assert (
                    key in split.keys()
                ), f"{key} is not in {[k for k in split.keys()]}"
            idx = {key: np.array(split[key]) for key in keys}
            nsamples = {key: len(idx[key]) for key in keys}
            return nsamples, idx
        else:
            raise TypeError("'split' has to be either of type str or dict if not None.")

    def _random_split_data(self, ntrain, nval):
        nsamples = {"train": ntrain, "val": nval, "test": self._ndata - ntrain - nval}
        all_idx = np.arange(self._ndata)
        if self.random_split:
            all_idx = self._random_state.permutation(all_idx)
        if self.sample_with_replacement:
            all_idx = self._random_state.choice(all_idx, self._ndata, replace=True)
        idx = {
            "train": all_idx[0:ntrain],
            "val": all_idx[ntrain : ntrain + nval],
            "test": all_idx[ntrain + nval :],
        }
        return nsamples, idx

    def save_split(self, path):
        """
        Save the split of the samples to path.
        Data has keys 'train', 'val', 'test'.
        """
        assert isinstance(path, str)
        assert path.endswith(".npz"), "'path' has to end with .npz"
        np.savez(path, **self.idx)

    def get_dataset(self, split, batch_size=None):
        assert split in self.idx
        if batch_size is None:
            batch_size = self.batch_size
        shuffle = self.shuffle if split == "train" else False
        indices = self.idx[split]
        if shuffle:
            torch_generator = paddle.framework.core.default_cpu_generator()
            if self.seed is not None:
                torch_generator.manual_seed(self.seed)
            idx_sampler = paddle.io.SubsetRandomSampler(indices=indices)
            dataset = self.data_container
        else:
            subset = paddle.io.Subset(dataset=self.data_container, indices=indices)
            idx_sampler = paddle.io.SequenceSampler(data_source=subset)
            dataset = subset
        batch_sampler1 = paddle.io.BatchSampler(
            sampler=idx_sampler, batch_size=batch_size, drop_last=False
        )

        # NOTE: paddle.io.DataLoader only support batch_sampler parameter, but
        # Raw pytorch code pass batch_sampler to sampler parameter, and inner
        # logic is:
        #    if batch_size is not None and batch_sampler is None:
        #        batch_sampler = BatchSampler(sampler, batch_size, drop_last)
        # So we need to wrap one more BatchSampler to change it to paddlepaddle.
        batch_sampler = paddle.io.BatchSampler(
            sampler=batch_sampler1, batch_size=1, drop_last=False
        )
        # >>>>>>        dataloader = torch.utils.data.DataLoader(dataset, sampler=
        #             batch_sampler, collate_fn=functools.partial(collate,
        #             target_keys=self.data_container.targets), pin_memory=True, **
        #             self.kwargs)
        dataloader = paddle.io.DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=functools.partial(
                collate, target_keys=self.data_container.targets
            ),
            **self.kwargs,
        )
        # dataloader.auto_collate_batch = False

        def generator():
            while True:
                for inputs, targets in dataloader:
                    yield inputs, targets

        return generator()
