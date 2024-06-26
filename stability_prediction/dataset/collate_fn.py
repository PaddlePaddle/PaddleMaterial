from __future__ import annotations

import json
import os
from functools import partial
from typing import TYPE_CHECKING
from typing import Callable

import numpy as np
import paddle
import pgl
from tqdm import trange


def collate_fn_graph(batch):
    """Merge a list of graphs to form a batch."""
    line_graphs = None
    graphs, lattices, state_attr, labels = map(list, zip(*batch))
    g = pgl.Graph.batch(graphs)
    new_labels = {}
    for k, v in labels[0].items():
        new_labels[k] = np.array([d[k] for d in labels], dtype="float32")
    labels = new_labels
    state_attr = np.asarray(state_attr)
    lat = lattices[0] if g.num_graph == 1 else np.squeeze(np.asarray(lattices))
    return g.tensor(), lat, state_attr, labels
