# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MACE 原子间势模型（PaddleMaterials 训推接口）。"""

from __future__ import annotations

from typing import Dict
from typing import Optional
from typing import Sequence

import numpy as np
import paddle
import paddle.nn as nn

from .layers import EquivariantLayer


class MACE(nn.Layer):
    """MACE 模型，适配 PaddleMaterials Trainer / Predictor。

    保留基于消息传递的能量预测主干，并提供与 CHGNet / MatterSim
    一致的 ``forward`` / ``predict`` 接口，输出 energy_per_atom、force、stress。
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_basis: int = 8,
        r_max: float = 6.0,
        num_elements: int = 89,
        property_names: Optional[Sequence[str]] = None,
        loss_type: str = "smooth_l1_loss",
        huber_loss_delta: float = 0.1,
        loss_weights_dict: Optional[Dict[str, float]] = None,
        **kwargs,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_basis = num_basis
        self.r_max = r_max
        self.num_elements = num_elements

        support_property_names = ["energy_per_atom", "force", "stress"]
        support_loss_weights = {
            "energy_per_atom": 1.0,
            "force": 1.0,
            "stress": 0.1,
        }
        if property_names is None:
            property_names = support_property_names
        for name in property_names:
            assert name in support_property_names, (
                f"{name} 不受支持，可选: {support_property_names}"
            )
        self.property_names = list(property_names)
        if loss_weights_dict is None:
            loss_weights_dict = {
                key: support_loss_weights[key] for key in self.property_names
            }
        self.loss_weights_dict = loss_weights_dict

        # 原子序数嵌入（Z 映射到 [0, num_elements)）
        self.embedding = nn.Embedding(num_elements, hidden_dim)
        self.layers = nn.LayerList(
            [EquivariantLayer(hidden_dim, num_basis, r_max) for _ in range(num_layers)]
        )
        self.output = nn.Linear(hidden_dim, 1)

        if loss_type == "mse_loss":
            self.loss_fn = nn.MSELoss()
        elif loss_type in ("smooth_l1_loss", "huber_loss"):
            self.loss_fn = nn.SmoothL1Loss(delta=huber_loss_delta)
        elif loss_type == "l1_loss":
            self.loss_fn = nn.L1Loss()
        else:
            raise ValueError(f"未知 loss_type: {loss_type}")

    def _atom_type_to_index(self, atom_types: paddle.Tensor) -> paddle.Tensor:
        """将原子序数 Z 映射为 embedding 下标。"""
        # H=1 -> 0；超出范围时截断，避免索引越界
        idx = atom_types.astype("int64").reshape([-1]) - 1
        idx = paddle.clip(idx, 0, self.num_elements - 1)
        return idx

    def _ensure_graph_tensor(self, graph):
        """保证图为 paddle Tensor 形式。"""
        if hasattr(graph, "tensor") and callable(graph.tensor):
            # 已是 batch 且含 paddle.Tensor 时，部分版本无需再转
            sample = graph.node_feat.get("atom_types")
            if sample is not None and not isinstance(sample, paddle.Tensor):
                graph = graph.tensor()
        return graph

    def _compute_bond_dist(self, graph, positions, lattice):
        """由坐标（及 PBC 偏移）重算键长，保证力/应力可对位置求导。

        构图阶段写入的 ``bond_dist`` 为常数，不能产生坐标梯度；
        这里沿用 FindPointsInSpheres 的边定义，仅替换距离计算。
        """
        edges = graph.edges.astype("int64")
        src = edges[:, 0]
        dst = edges[:, 1]
        pbc_offset = None
        if hasattr(graph, "edge_feat") and graph.edge_feat is not None:
            pbc_offset = graph.edge_feat.get("pbc_offset", None)

        if pbc_offset is not None and lattice is not None:
            pbc_offset = pbc_offset.astype("float32")
            # 边所属结构：优先 graph_edge_id，否则用源节点的图 id
            if hasattr(graph, "graph_edge_id"):
                edge_batch = graph.graph_edge_id
            else:
                edge_batch = paddle.gather(graph.graph_node_id, src)
            cell_e = paddle.gather(lattice, edge_batch)
            offset = paddle.einsum("bi,bij->bj", pbc_offset, cell_e)
            bond_vec = positions[dst] + offset - positions[src]
        else:
            bond_vec = positions[dst] - positions[src]

        bond_dist = paddle.linalg.norm(bond_vec, axis=-1)
        # 避免零距离导致径向基 / 梯度不稳定
        bond_dist = paddle.clip(bond_dist, min=1e-8)
        return bond_dist, src, dst

    def _compute_energy_force_stress(self, graph):
        """根据图计算能量、力与应力。"""
        graph = self._ensure_graph_tensor(graph)

        atom_types = graph.node_feat["atom_types"]
        positions = graph.node_feat["cart_coords"].astype("float32")
        num_atoms = graph.node_feat["num_atoms"].astype("float32").reshape([-1])

        # 力需要位置梯度
        if "force" in self.property_names:
            positions.stop_gradient = False

        lattice = graph.node_feat.get("lattice", None)
        if lattice is not None:
            lattice = lattice.astype("float32")
            if lattice.ndim == 2:
                lattice = lattice.unsqueeze(0)

        strain = None
        if "stress" in self.property_names and lattice is not None:
            # 应变张量用于应力自动微分
            strain = paddle.zeros_like(lattice)
            strain.stop_gradient = False
            lattice = paddle.matmul(
                lattice, paddle.eye(3, dtype=lattice.dtype) + strain
            )
            # 同步更新笛卡尔坐标（小应变近似）
            node_id = graph.graph_node_id
            strain_atoms = paddle.gather(strain, node_id)
            positions = paddle.einsum(
                "bi,bij->bj",
                positions,
                paddle.eye(3, dtype=positions.dtype) + strain_atoms,
            )

        indices = self._atom_type_to_index(atom_types)
        x = self.embedding(indices)

        # 用可微键长替代构图阶段的常量 bond_dist
        bond_dist, src, dst = self._compute_bond_dist(graph, positions, lattice)

        edge_indices = (src, dst)
        for layer in self.layers:
            x = layer(x, edge_indices, bond_dist)

        atomic_energies = self.output(x).squeeze(-1)  # [N]

        # 按图聚合总能量（index_add，与 CHGNet aggregate 一致，支持力的二阶反传）
        node_id = graph.graph_node_id.astype("int32")
        n_graph = int(num_atoms.shape[0])
        energy_buf = paddle.zeros([n_graph, 1], dtype=atomic_energies.dtype)
        energy_buf.stop_gradient = False
        total_energy = energy_buf.index_add(
            axis=0,
            index=node_id,
            value=atomic_energies.reshape([-1, 1]),
        ).reshape([-1])
        energy_per_atom = total_energy / paddle.clip(num_atoms, min=1.0)

        forces = None
        if "force" in self.property_names:
            # 与 CHGNet 一致：训练时 create_graph=True，便于力 loss 反传到参数
            # 若当前算子缺少二阶导，则回退 create_graph=False（力仍可计算）
            try:
                force_grad = paddle.grad(
                    outputs=[total_energy.sum()],
                    inputs=[positions],
                    create_graph=self.training,
                    retain_graph=True,
                    allow_unused=True,
                )[0]
            except RuntimeError:
                force_grad = paddle.grad(
                    outputs=[total_energy.sum()],
                    inputs=[positions],
                    create_graph=False,
                    retain_graph=True,
                    allow_unused=True,
                )[0]
            if force_grad is None:
                forces = paddle.zeros_like(positions)
            else:
                forces = -force_grad

        stress = None
        if "stress" in self.property_names and strain is not None:
            volume = paddle.linalg.det(lattice)
            try:
                stress_grad = paddle.grad(
                    outputs=[total_energy.sum()],
                    inputs=[strain],
                    create_graph=self.training,
                    retain_graph=self.training,
                    allow_unused=True,
                )[0]
            except RuntimeError:
                stress_grad = paddle.grad(
                    outputs=[total_energy.sum()],
                    inputs=[strain],
                    create_graph=False,
                    retain_graph=False,
                    allow_unused=True,
                )[0]
            if stress_grad is None:
                stress = paddle.zeros_like(lattice)
            else:
                # 转为 GPa 量级（与套件常见约定一致：1 eV/Å^3 ≈ 160.21766208 GPa）
                stress = stress_grad / volume.reshape([-1, 1, 1]) / 160.21766208

        return energy_per_atom, forces, stress

    def forward(self, data, return_loss=True, return_prediction=True):
        """套件统一前向：返回 loss_dict / pred_dict。"""
        assert return_loss or return_prediction

        energy, force, stress = self._compute_energy_force_stress(data["graph"])

        pred_dict = {}
        if "energy_per_atom" in self.property_names:
            pred_dict["energy_per_atom"] = energy
        if "force" in self.property_names and force is not None:
            pred_dict["force"] = force
        if "stress" in self.property_names and stress is not None:
            pred_dict["stress"] = stress

        loss_dict = {}
        if return_loss:
            loss = paddle.to_tensor(0.0, dtype="float32")
            for property_name, pred in pred_dict.items():
                if property_name not in data:
                    continue
                label = data[property_name]
                # 展平后过滤 NaN
                pred_flat = pred.reshape([-1])
                label_flat = label.reshape([-1]).astype(pred_flat.dtype)
                valid = ~paddle.isnan(label_flat)
                if int(valid.astype("int64").sum()) == 0:
                    continue
                loss_property = self.loss_fn(pred_flat[valid], label_flat[valid])
                loss_dict[property_name] = loss_property
                loss = loss + loss_property * self.loss_weights_dict[property_name]
            loss_dict["loss"] = loss

        prediction = pred_dict if return_prediction else {}
        return {"loss_dict": loss_dict, "pred_dict": prediction}

    def _prediction_to_numpy(self, prediction):
        """将预测结果转为 numpy，便于 Predictor 写 CSV。"""
        out = {}
        for key, value in prediction.items():
            if isinstance(value, paddle.Tensor):
                value = value.numpy()
            if key == "energy_per_atom" and isinstance(value, np.ndarray):
                value = value.reshape(-1)
                if value.size == 1:
                    value = float(value[0])
            if key == "stress" and isinstance(value, np.ndarray) and value.ndim == 3:
                value = value[0]
            out[key] = value
        return out

    def predict(self, graphs):
        """推理接口，与 CHGNet / MatterSim 保持一致。"""
        if isinstance(graphs, list):
            results = []
            for graph in graphs:
                result = self.forward(
                    {"graph": graph},
                    return_loss=False,
                    return_prediction=True,
                )
                results.append(self._prediction_to_numpy(result["pred_dict"]))
            return results

        result = self.forward(
            {"graph": graphs},
            return_loss=False,
            return_prediction=True,
        )
        return self._prediction_to_numpy(result["pred_dict"])
