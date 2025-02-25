from __future__ import annotations

import numbers
import warnings
from collections.abc import Mapping
from collections.abc import Sequence
from typing import Any
from typing import List

import numpy as np
import paddle
import pgl

from paddle_geometric.data import Batch
from paddle_geometric.data import Data


class ConcatData(object):
    def __init__(self, data) -> None:
        self.data = data

    @staticmethod
    def batch(data_list):
        data_list = [data.data for data in data_list]
        data = np.concatenate(data_list, axis=0)
        return data

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return str(self.__dict__)


class DefaultCollator(object):
    def __call__(self, batch: List[Any]) -> Any:
        """Default_collate_fn for paddle dataloader.

        NOTE: This `default_collate_fn` is different from official `default_collate_fn`
        which specially adapt case where sample is `None` and `pgl.Graph`.

        ref: https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/io/dataloader/collate.py#L25

        Args:
            batch (List[Any]): Batch of samples to be collated.

        Returns:
            Any: Collated batch data.
        """
        sample = batch[0]
        if sample is None:
            return None
        elif isinstance(sample, np.ndarray):
            batch = np.stack(batch, axis=0)
            return batch
        elif isinstance(sample, (paddle.Tensor, paddle.framework.core.eager.Tensor)):
            return paddle.stack(batch, axis=0)
        elif isinstance(sample, numbers.Number):
            batch = np.array(batch)
            return batch
        elif isinstance(sample, (str, bytes)):
            return batch
        elif isinstance(sample, Mapping):
            return {key: self([d[key] for d in batch]) for key in sample}
        elif isinstance(sample, Sequence):
            sample_fields_num = len(sample)
            if not all(len(sample) == sample_fields_num for sample in iter(batch)):
                raise RuntimeError("Fields number not same among samples in a batch")
            return [self(fields) for fields in zip(*batch)]
        elif str(type(sample)) == "<class 'pgl.graph.Graph'>":
            # use str(type()) instead of isinstance() in case of pgl is not installed.
            graphs = pgl.Graph.batch(batch)
            graphs.tensor()
            return graphs
        elif isinstance(sample, ConcatData):
            return ConcatData.batch(batch)
        elif isinstance(sample, Data):
            attrs = set(batch[0].keys() if callable(batch[0].keys) else batch[0].keys)
            for x in batch[1:]:
                attrs.intersection_update(x.keys() if callable(x.keys) else x.keys)
            for x in batch:
                for attr in list(x.keys() if callable(x.keys) else x.keys):
                    if attr not in attrs:
                        warnings.warn(
                            f"Attribute `{attr}` is not in the intersection of attributes of the collated `Data` objects. This attribute will be dropped."
                        )
                        del x[attr]
            try:
                batch = Batch.from_data_list(batch)
            except Exception as e:
                for attr in attrs:
                    types = set(type(x[attr]) for x in batch)
                    if len(types) != 1:
                        raise ValueError(
                            f"Attribute `{attr}` has inconsistent types. Found a mix "
                            f"of {len(types)} types: `{types}`."
                        )
                    if isinstance(batch[0][attr], paddle.Tensor):
                        dtypes = set(x[attr].dtype for x in batch)
                        if len(dtypes) != 1:
                            raise ValueError(
                                f"Attribute `{attr}` has inconsistent dtypes. Found a "
                                f"mix of {len(dtypes)} dtypes: `{dtypes}`."
                            )
                raise e
            return batch
        raise TypeError(
            "batch data can only contains: paddle.Tensor, numpy.ndarray, "
            f"dict, list, number, None, pgl.Graph, but got {type(sample)}"
        )
