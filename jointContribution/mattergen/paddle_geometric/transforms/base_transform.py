import copy
from abc import ABC
from typing import Any


class BaseTransform(ABC):
    r"""An abstract base class for writing transforms.

    Transforms are a general way to modify and customize
    :class:`~paddle_geometric.data.Data` or
    :class:`~paddle_geometric.data.HeteroData` objects, either by implicitly
    passing them as an argument to a :class:`~paddle_geometric.data.Dataset`, or
    by applying them explicitly to individual
    :class:`~paddle_geometric.data.Data` or
    :class:`~paddle_geometric.data.HeteroData` objects:

    .. code-block:: python

        import paddle_geometric.transforms as T
        from paddle_geometric.datasets import TUDataset

        transform = T.Compose([T.ToUndirected(), T.AddSelfLoops()])

        dataset = TUDataset(path, name='MUTAG', transform=transform)
        data = dataset[0]  # Implicitly transform data on every access.

        data = TUDataset(path, name='MUTAG')[0]
        data = transform(data)  # Explicitly transform data.
    """
    def __call__(self, data: Any) -> Any:
        # Shallow-copy the data so that we prevent in-place data modification.
        return self.forward(copy.copy(data))

    def forward(self, data: Any) -> Any:
        pass

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'
