# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import annotations

import numbers
from collections import defaultdict
from collections.abc import Mapping
from collections.abc import Sequence
from typing import Any
from typing import Dict
from typing import List

import numpy as np
import paddle
import pgl


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


# TODO: move to new custom dataset file
class Data(object):
    def __init__(self, data: Dict):
        for key, value in data.items():
            assert isinstance(value, np.ndarray)
            setattr(self, key, value)

    @staticmethod
    def batch(data_list):
        feat = defaultdict(lambda: [])
        for data in data_list:
            for key in data.__dict__:
                feat[key].append(data.__dict__[key])

        ret_feat = {}
        for key in feat:
            ret_feat[key] = np.concatenate(feat[key], axis=0)
        return Data(ret_feat)

    def tensor(self):
        for key, value in self.__dict__.items():
            self.__dict__[key] = paddle.to_tensor(value)

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return str(self.__dict__)


# TODO: move to new custom dataset file
class DistributionNodes(object):
    def __init__(self, histogram):
        """Compute the distribution of the number of nodes in the dataset,
            and sample from this distribution.
        historgram: dict. The keys are num_nodes, the values are counts
        """
        if type(histogram) == dict:
            max_n_nodes = max(histogram.keys())
            prob = paddle.zeros(shape=max_n_nodes + 1)
            for num_nodes, count in histogram.items():
                prob[num_nodes] = count
        else:
            prob = histogram
        self.prob = prob / prob.sum()
        self.m = paddle.distribution.Categorical(prob)

    def sample_n(self, n_samples):
        idx = self.m.sample((n_samples,))
        return idx

    def log_prob(self, batch_n_nodes):
        assert len(tuple(batch_n_nodes.shape)) == 1
        p = self.prob.to(batch_n_nodes.place)
        probas = p[batch_n_nodes]
        log_p = paddle.log(x=probas + 1e-30)
        return log_p


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
        elif isinstance(sample, Data):
            data = Data.batch(batch)
            data.tensor()
            return data
        elif isinstance(sample, ConcatData):
            return ConcatData.batch(batch)
        raise TypeError(
            "batch data can only contains: paddle.Tensor, numpy.ndarray, "
            f"dict, list, number, None, pgl.Graph, but got {type(sample)}"
        )
