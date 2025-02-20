import random

import numpy as np


class Subset(object):
    """Subset of a dataset at specified indices

    Code adapted from PyTorch.

    Parameters
    ----------
    dataset
        dataset[i] should return the ith datapoint
    indices : list
        List of datapoint indices to construct the subset
    """

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, item):
        """Get the datapoint indexed by item

        Returns
        -------
        tuple
            datapoint
        """
        data = self.dataset[self.indices[item]]
        if data is None:
            return self.__getitem__(random.randint(0, len(self) - 1))
        return data

    def __len__(self):
        """Get subset size

        Returns
        -------
        int
            Number of datapoints in the subset
        """
        return len(self.indices)


def split_dataset(dataset, frac_list=None, shuffle=False, random_state=None):
    """Split dataset into training, validation and test set.

    Parameters
    ----------
    dataset
        We assume ``len(dataset)`` gives the number of datapoints and ``dataset[i]``
        gives the ith datapoint.
    frac_list : list or None, optional
        A list of length 3 containing the fraction to use for training,
        validation and test. If None, we will use [0.8, 0.1, 0.1].
    shuffle : bool, optional
        By default we perform a consecutive split of the dataset. If True,
        we will first randomly shuffle the dataset.
    random_state : None, int or array_like, optional
        Random seed used to initialize the pseudo-random number generator.
        Can be any integer between 0 and 2**32 - 1 inclusive, an array
        (or other sequence) of such integers, or None (the default).
        If seed is None, then RandomState will try to read data from /dev/urandom
        (or the Windows analogue) if available or seed from the clock otherwise.

    Returns
    -------
    list of length 3
        Subsets for training, validation and test.
    """
    from itertools import accumulate

    if frac_list is None:
        frac_list = [0.8, 0.1, 0.1]
    frac_list = np.asarray(frac_list)
    assert np.allclose(
        np.sum(frac_list), 1.0
    ), "Expect frac_list sum to 1, got {:.4f}".format(np.sum(frac_list))
    num_data = len(dataset)
    lengths = (num_data * frac_list).astype(int)
    lengths[-1] = num_data - np.sum(lengths[:-1])
    if shuffle:
        indices = np.random.RandomState(seed=random_state).permutation(num_data)
    else:
        indices = np.arange(num_data)
    return [
        Subset(dataset, indices[offset - length : offset])
        for offset, length in zip(accumulate(lengths), lengths)
    ]
