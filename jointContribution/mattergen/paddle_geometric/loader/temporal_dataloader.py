import paddle
from paddle.io import DataLoader
from typing import List

from paddle_geometric.data import TemporalData


class TemporalDataLoader(DataLoader):
    r"""A data loader which merges successive events of a
    :class:`paddle_geometric.data.TemporalData` to a mini-batch.

    Args:
        data (TemporalData): The :obj:`~paddle_geometric.data.TemporalData`
            from which to load the data.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        neg_sampling_ratio (float, optional): The ratio of sampled negative
            destination nodes to the number of positive destination nodes.
            (default: :obj:`0.0`)
        **kwargs (optional): Additional arguments of
            :class:`paddle.io.DataLoader`.
    """
    def __init__(
        self,
        data: TemporalData,
        batch_size: int = 1,
        neg_sampling_ratio: float = 0.0,
        **kwargs,
    ):
        # Remove for Paddle Lightning:
        kwargs.pop('dataset', None)
        kwargs.pop('collate_fn', None)
        kwargs.pop('shuffle', None)

        self.data = data
        self.events_per_batch = batch_size
        self.neg_sampling_ratio = neg_sampling_ratio

        if neg_sampling_ratio > 0:
            self.min_dst = int(data.dst.min())
            self.max_dst = int(data.dst.max())

        if kwargs.get('drop_last', False) and len(data) % batch_size != 0:
            arange = range(0, len(data) - batch_size, batch_size)
        else:
            arange = range(0, len(data), batch_size)

        super().__init__(arange, batch_size=1, shuffle=False, collate_fn=self, **kwargs)

    def __call__(self, arange: List[int]) -> TemporalData:
        batch = self.data[arange[0]:arange[0] + self.events_per_batch]

        n_ids = [batch.src, batch.dst]

        if self.neg_sampling_ratio > 0:
            batch.neg_dst = paddle.randint(
                low=self.min_dst,
                high=self.max_dst + 1,
                shape=(round(self.neg_sampling_ratio * batch.dst.shape[0]), ),
                dtype=batch.dst.dtype,
                device=batch.dst.device,
            )
            n_ids += [batch.neg_dst]

        batch.n_id = paddle.unique(paddle.concat(n_ids, axis=0))

        return batch
