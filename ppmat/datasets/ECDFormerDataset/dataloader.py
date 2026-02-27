# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
from paddle.io import Dataset

from typing import Any, List, Sequence, Union

from paddle_geometric.data import Batch, Dataset
from paddle_geometric.data.data import BaseData
from paddle_geometric.data.datapipes import DatasetAdapter

def call(batch: List[Any]) -> Any:
    batch = [list(x) for x in zip(*batch)] # transpose
    for i in range(len(batch)): # 组Batch
        batch[i] = Batch.from_data_list(batch[i])

    batch0 = batch[0]
    batch1 = batch[1]

    # Data解包到Tensor字典
    batch_atom_bond, batch_bond_angle = batch0, batch1
    x, edge_index, edge_attr, query_mask =batch_atom_bond.x,batch_atom_bond.edge_index,batch_atom_bond.edge_attr,batch_atom_bond.query_mask
    ba_edge_index, ba_edge_attr = batch_bond_angle.edge_index,batch_bond_angle.edge_attr
    batch_data = batch_atom_bond.batch
    pos_gt = batch_atom_bond.peak_position 
    height_gt = batch_atom_bond.peak_height
    num_gt = batch_atom_bond.peak_num      
    return \
    {
    "x"               : x             ,
    "edge_index"      : edge_index    ,
    "edge_attr"       : edge_attr     ,
    "batch_data"      : batch_data    ,
    "ba_edge_index"   : ba_edge_index ,
    "ba_edge_attr"    : ba_edge_attr  ,
    "query_mask"      : query_mask
    },     \
    {
    "peak_number_gt"  : num_gt        ,
    "peak_position_gt": pos_gt        ,
    "peak_height_gt"  : height_gt
    }

class ECDFormerDataset_DataLoader(paddle.io.DataLoader):
    def __init__(
        self,
        dataset: Union[Dataset, Sequence[BaseData], DatasetAdapter],
        batch_size: int = 1,
        shuffle: bool = False,
        **kwargs,
    ):
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=call,
            **kwargs,
        )