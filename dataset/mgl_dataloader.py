
from __future__ import annotations
import paddle
import os
"""Tools to construct a dataset of DGL graphs."""
import json
from functools import partial
from typing import TYPE_CHECKING, Callable
import numpy as np
from tqdm import trange
import pgl


def collate_fn_graph(batch, include_line_graph: bool=False):
    """Merge a list of dgl graphs to form a batch."""
    line_graphs = None
    if include_line_graph:
        graphs, lattices, line_graphs, state_attr, labels = map(list, zip(*
            batch))
    else:
        graphs, lattices, state_attr, labels = map(list, zip(*batch))
    g = pgl.Graph.batch(graphs)
    labels = np.array([next(iter(d.values())) for d in labels], dtype='float32')
    state_attr = np.asarray(state_attr)
    lat = lattices[0] if g.num_graph == 1 else np.squeeze(np.asarray(lattices))
    if include_line_graph:
        l_g = dgl.batch(line_graphs)
        return g, lat, l_g, state_attr, labels
    return g.tensor(), lat, state_attr, labels

def _create_dist_sampler(dataset, dataloader_kwargs, ddp_seed):
    # Note: will change the content of dataloader_kwargs
    dist_sampler_kwargs = {"shuffle": dataloader_kwargs.get("shuffle", False)}
    dataloader_kwargs["shuffle"] = False
    # dist_sampler_kwargs["seed"] = ddp_seed
    dist_sampler_kwargs["drop_last"] = dataloader_kwargs.get("drop_last", False)
    dataloader_kwargs["drop_last"] = False
    dist_sampler_kwargs["batch_size"] = dataloader_kwargs.get("batch_size", 1)

    dataloader_kwargs.pop("batch_size")
    dataloader_kwargs.pop("shuffle")
    dataloader_kwargs.pop("drop_last")
    return paddle.io.DistributedBatchSampler(dataset, **dist_sampler_kwargs)


class GraphDataLoader(paddle.io.DataLoader):
    """Batched graph data loader.

    PyTorch dataloader for batch-iterating over a set of graphs, generating the batched
    graph and corresponding label tensor (if provided) of the said minibatch.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        The dataset to load graphs from.
    collate_fn : Function, default is None
        The customized collate function. Will use the default collate
        function if not given.
    use_ddp : boolean, optional
        If True, tells the DataLoader to split the training set for each
        participating process appropriately using
        :class:`torch.utils.data.distributed.DistributedSampler`.

        Overrides the :attr:`sampler` argument of :class:`torch.utils.data.DataLoader`.
    ddp_seed : int, optional
        The seed for shuffling the dataset in
        :class:`torch.utils.data.distributed.DistributedSampler`.

        Only effective when :attr:`use_ddp` is True.
    kwargs : dict
        Key-word arguments to be passed to the parent PyTorch
        :py:class:`torch.utils.data.DataLoader` class. Common arguments are:

          - ``batch_size`` (int): The number of indices in each batch.
          - ``drop_last`` (bool): Whether to drop the last incomplete batch.
          - ``shuffle`` (bool): Whether to randomly shuffle the indices at each epoch.

    Examples
    --------
    To train a GNN for graph classification on a set of graphs in ``dataset``:

    >>> dataloader = dgl.dataloading.GraphDataLoader(
    ...     dataset, batch_size=1024, shuffle=True, drop_last=False, num_workers=4)
    >>> for batched_graph, labels in dataloader:
    ...     train_on(batched_graph, labels)

    **With Distributed Data Parallel**

    If you are using PyTorch's distributed training (e.g. when using
    :mod:`torch.nn.parallel.DistributedDataParallel`), you can train the model by
    turning on the :attr:`use_ddp` option:

    >>> dataloader = dgl.dataloading.GraphDataLoader(
    ...     dataset, use_ddp=True, batch_size=1024, shuffle=True, drop_last=False, num_workers=4)
    >>> for epoch in range(start_epoch, n_epochs):
    ...     dataloader.set_epoch(epoch)
    ...     for batched_graph, labels in dataloader:
    ...         train_on(batched_graph, labels)
    """


    def __init__(
        self, dataset, collate_fn=None, use_ddp=False, ddp_seed=0, **kwargs
    ):
        dataloader_kwargs = {}
        for k, v in kwargs.items():
            dataloader_kwargs[k] = v

        self.use_ddp = use_ddp
        if use_ddp:
            self.dist_sampler = _create_dist_sampler(
                dataset, dataloader_kwargs, ddp_seed
            )
            dataloader_kwargs["batch_sampler"] = self.dist_sampler

        super().__init__(
            dataset=dataset, collate_fn=collate_fn, **dataloader_kwargs
        )

    def set_epoch(self, epoch):
        """Sets the epoch number for the underlying sampler which ensures all replicas
        to use a different ordering for each epoch.

        Only available when :attr:`use_ddp` is True.

        Calls :meth:`torch.utils.data.distributed.DistributedSampler.set_epoch`.

        Parameters
        ----------
        epoch : int
            The epoch number.
        """
        if self.use_ddp:
            self.dist_sampler.set_epoch(epoch)
        else:
            raise DGLError("set_epoch is only available when use_ddp is True.")


def MGLDataLoader(train_data: dgl.data.utils.Subset, val_data: dgl.data.
    utils.Subset, collate_fn: (Callable | None)=None, test_data: dgl.data.
    utils.Subset=None, **kwargs) ->tuple[GraphDataLoader, ...]:
    """Dataloader for MatGL training.

    Args:
        train_data (dgl.data.utils.Subset): Training dataset.
        val_data (dgl.data.utils.Subset): Validation dataset.
        collate_fn (Callable): Collate function.
        test_data (dgl.data.utils.Subset | None, optional): Test dataset. Defaults to None.
        **kwargs: Pass-through kwargs to dgl.dataloading.GraphDataLoader. Common ones you may want to set are
            batch_size, num_workers, use_ddp, pin_memory and generator.

    Returns:
        tuple[GraphDataLoader, ...]: Train, validation and test data loaders. Test data
            loader is None if test_data is None.
    """
    train_loader = GraphDataLoader(train_data, shuffle=False, collate_fn=
        collate_fn, **kwargs)
    val_loader = GraphDataLoader(val_data, shuffle=False, collate_fn=
        collate_fn, **kwargs)
    if test_data is not None:
        test_loader = GraphDataLoader(test_data, shuffle=False, collate_fn=
            collate_fn, **kwargs)
        return train_loader, val_loader, test_loader
    return train_loader, val_loader
