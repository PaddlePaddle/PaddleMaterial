import os.path as osp
from typing import Callable, Optional

import numpy as np
import paddle

from paddle_geometric.data import Data, InMemoryDataset, extract_zip
from paddle_geometric.utils import index_to_mask


class DGraphFin(InMemoryDataset):
    r"""The DGraphFin networks from the
    `"DGraph: A Large-Scale Financial Dataset for Graph Anomaly Detection"
    <https://arxiv.org/abs/2207.03579>`_ paper.
    It is a directed, unweighted dynamic graph consisting of millions of
    nodes and edges, representing a realistic user-to-user social network
    in financial industry.
    Node represents a Finvolution user, and an edge from one
    user to another means that the user regards the other user
    as the emergency contact person. Each edge is associated with a
    timestamp ranging from 1 to 821 and a type of emergency contact
    ranging from 0 to 11.

    Args:
        root (str): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`paddle_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`paddle_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        force_reload (bool, optional): Whether to re-process the dataset.
            (default: :obj:`False`)

    **STATS:**

    .. list-table::
        :widths: 10 10 10 10
        :header-rows: 1

        * - #nodes
          - #edges
          - #features
          - #classes
        * - 3,700,550
          - 4,300,999
          - 17
          - 2
    """

    url = "https://dgraph.xinye.com"

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        super().__init__(root, transform, pre_transform,
                         force_reload=force_reload)
        self.load(self.processed_paths[0])

    def download(self) -> None:
        raise RuntimeError(
            f"Dataset not found. Please download '{self.raw_file_names}' from "
            f"'{self.url}' and move it to '{self.raw_dir}'")

    @property
    def raw_file_names(self) -> str:
        return 'DGraphFin.zip'

    @property
    def processed_file_names(self) -> str:
        return 'data.pdparams'

    @property
    def num_classes(self) -> int:
        return 2

    def process(self) -> None:
        extract_zip(self.raw_paths[0], self.raw_dir, log=False)
        path = osp.join(self.raw_dir, "dgraphfin.npz")

        with np.load(path) as loader:
            x = paddle.to_tensor(loader['x'], dtype='float32')
            y = paddle.to_tensor(loader['y'], dtype='int64')
            edge_index = paddle.to_tensor(loader['edge_index'], dtype='int64')
            edge_type = paddle.to_tensor(loader['edge_type'], dtype='int64')
            edge_time = paddle.to_tensor(loader['edge_timestamp'], dtype='int64')
            train_nodes = paddle.to_tensor(loader['train_mask'], dtype='int64')
            val_nodes = paddle.to_tensor(loader['valid_mask'], dtype='int64')
            test_nodes = paddle.to_tensor(loader['test_mask'], dtype='int64')

            train_mask = index_to_mask(train_nodes, size=x.shape[0])
            val_mask = index_to_mask(val_nodes, size=x.shape[0])
            test_mask = index_to_mask(test_nodes, size=x.shape[0])
            data = Data(x=x, edge_index=edge_index.transpose([1, 0]),
                        edge_type=edge_type, edge_time=edge_time, y=y,
                        train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

        data = data if self.pre_transform is None else self.pre_transform(data)
        self.save([data], self.processed_paths[0])
