# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0

"""PaddleMaterials-compatible LiFlow wrapper (propagator/corrector training and
prediction), faithful to liflow.model.modules.FlowModule (commit e6fc475).

Training forward mimics reference `training_step`: it interpolates
x_t = (1-t)*source + t*positions_2 with source = positions_1 + prior, feeds
(positions_1=original, positions_2=x_t, time per node) to the network, and
computes the velocity/data target. `predict()` is a thin no-grad network call
for the trajectory integrator, which already prepares the exact fields.
"""

import numpy as np
import paddle
from paddle import nn

from ppmat.models.common.runtime import RuntimeMixin
from ppmat.models.common.runtime import runtime_boundary
from ppmat.models.liflow.dual_painn import DualPaiNN


def _get_batch_value(batch_data, key, default=None):
    if isinstance(batch_data, dict):
        return batch_data.get(key, default)
    value = getattr(batch_data, key, default)
    return default if value is None else value


def _as_mapping(batch_data):
    if isinstance(batch_data, dict):
        data = dict(batch_data)
    else:
        data = {key: getattr(batch_data, key) for key in batch_data.keys}
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            data[key] = paddle.to_tensor(value)
    return data


def _num_atoms_and_batch(batch_data, positions_1):
    num_atoms = positions_1.shape[0]
    batch = _get_batch_value(batch_data, "batch")
    if batch is None:
        num_nodes_per_graph = _get_batch_value(batch_data, "num_atoms")
        if num_nodes_per_graph is not None:
            n = num_nodes_per_graph.reshape([-1]).astype("int64")
            batch = paddle.repeat_interleave(paddle.arange(n.shape[0]), n)
        else:
            batch = paddle.zeros([num_atoms], dtype="int64")
    return num_atoms, batch


class LiFlow(RuntimeMixin, nn.Layer):
    def __init__(
        self,
        num_features=64,
        num_radial_basis=20,
        num_layers=3,
        num_elements=77,
        r_max=5.0,
        r_offset=0.5,
        ref_temp=1000.0,
        prediction_mode="velocity",
        execution_backend="eager",
        runtime_options=None,
    ):
        super().__init__()
        self._init_runtime(execution_backend, runtime_options)
        if prediction_mode not in {"velocity", "data"}:
            raise ValueError("prediction_mode must be 'velocity' or 'data'")
        self.prediction_mode = prediction_mode
        self.network = DualPaiNN(
            num_features=num_features,
            num_radial_basis=num_radial_basis,
            num_layers=num_layers,
            num_elements=num_elements,
            r_max=r_max,
            r_offset=r_offset,
            ref_temp=ref_temp,
        )
        self.ref_temp = float(ref_temp)

    def forward(self, batch_data, return_loss=True, return_prediction=True):
        positions_1 = batch_data["positions_1"]
        positions_2 = batch_data["positions_2"]
        if isinstance(positions_1, np.ndarray):
            positions_1 = paddle.to_tensor(positions_1, dtype="float32")
        if isinstance(positions_2, np.ndarray):
            positions_2 = paddle.to_tensor(positions_2, dtype="float32")
        prior = _get_batch_value(batch_data, "prior")
        if prior is None:
            prior = paddle.zeros_like(positions_1)
        elif isinstance(prior, np.ndarray):
            prior = paddle.to_tensor(prior, dtype="float32")
        num_atoms, batch = _num_atoms_and_batch(batch_data, positions_1)

        x_cond = positions_1
        disp = positions_2 - positions_1
        source = positions_1 + prior

        time = _get_batch_value(batch_data, "time")
        if isinstance(time, np.ndarray):
            time = paddle.to_tensor(time, dtype="float32")
        time = paddle.reshape(time, [-1])
        if time.shape[0] != num_atoms:
            t_node = paddle.expand(time, [num_atoms]) if time.shape[0] == 1 else time[batch]
        else:
            t_node = time
        t_node = t_node.unsqueeze(-1)  # [n, 1]
        x_t = (1.0 - t_node) * source + t_node * positions_2
        dx_dt = positions_2 - source

        data = _as_mapping(batch_data)
        data["time"] = t_node.reshape([-1])
        data["positions_1"] = x_cond
        data["positions_2"] = x_t
        prediction = self.network(data)

        result = {"loss_dict": {}, "pred_dict": {}}
        if return_loss:
            target = dx_dt if self.prediction_mode == "velocity" else disp
            result["loss_dict"]["loss"] = paddle.mean(
                paddle.sum((prediction - target) ** 2, axis=-1)
            )
        if return_prediction:
            result["pred_dict"]["target"] = prediction
        return result

    @paddle.no_grad()
    def predict(self, batch_data):
        """Direct single vector-field call used by the integrator.

        The caller already provides positions_1 (source), positions_2 (current),
        edge_index/shifts, elements and a graph-level time scalar.
        """
        return {"target": self._runtime_predict(batch_data)}

    @runtime_boundary("predict")
    def _runtime_predict(self, batch_data):
        return self.network(batch_data)
