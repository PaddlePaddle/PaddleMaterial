from typing import Callable, List, Optional

import numpy as np
import paddle

from paddle_geometric.data import InMemoryDataset, TemporalData, download_url


class MyketDataset(InMemoryDataset):
    r"""The Myket Android Application Install dataset from the
    `"Effect of Choosing Loss Function when Using T-Batching for Representation
    Learning on Dynamic Networks" <https://arxiv.org/abs/2308.06862>`_ paper.
    The dataset contains a temporal graph of application install interactions
    in an Android application market.

    Args:
        root (str): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`paddle_geometric.data.TemporalData` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`paddle_geometric.data.TemporalData` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        force_reload (bool, optional): Whether to re-process the dataset.
            (default: :obj:`False`)

    **STATS:**

    .. list-table::
        :widths: 10 10 10 10 10
        :header-rows: 1

        * - Name
          - #nodes
          - #edges
          - #features
          - #classes
        * - Myket
          - 17,988
          - 694,121
          - 33
          - 1
    """
    url = ('https://raw.githubusercontent.com/erfanloghmani/'
           'myket-android-application-market-dataset/main/data_int_index')

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        force_reload: bool = False,
    ) -> None:
        super().__init__(root, transform, pre_transform, force_reload=force_reload)
        self.load(self.processed_paths[0], data_cls=TemporalData)

    @property
    def raw_file_names(self) -> List[str]:
        return ['myket.csv', 'app_info_sample.npy']

    @property
    def processed_file_names(self) -> str:
        return 'data.pdparams'

    def download(self) -> None:
        for file_name in self.raw_file_names:
            download_url(f'{self.url}/{file_name}', self.raw_dir)

    def process(self) -> None:
        import pandas as pd

        # Read raw data
        df = pd.read_csv(self.raw_paths[0], skiprows=1, header=None)

        src = paddle.to_tensor(df[0].values, dtype='int64')
        dst = paddle.to_tensor(df[1].values, dtype='int64')
        t = paddle.to_tensor(df[2].values, dtype='int64')

        # Load node features
        x = paddle.to_tensor(np.load(self.raw_paths[1]), dtype='float32')
        msg = x[dst]

        # Adjust destination node IDs
        dst = dst + (int(src.max().numpy()) + 1)

        # Create TemporalData
        data = TemporalData(src=src, dst=dst, t=t, msg=msg)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        self.save([data], self.processed_paths[0])

    def __repr__(self) -> str:
        return f"MyketDataset({len(self)})"
