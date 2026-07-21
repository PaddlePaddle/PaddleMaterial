import copy
import math
from collections.abc import Mapping

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ppmat.losses import build_loss
from ppmat.models.common.e3nn import o3


def get_arg(args, name, default):
    if isinstance(args, dict):
        return args.get(name, default)
    return getattr(args, name, default)


def pad_or_slice_last(x, target_dim):
    """Pad or slice the last dimension to target_dim."""
    current_dim = x.shape[-1]

    if current_dim == target_dim:
        return x

    if current_dim > target_dim:
        return x[..., :target_dim]

    pad_shape = list(x.shape)
    pad_shape[-1] = target_dim - current_dim
    pad = paddle.zeros(pad_shape, dtype=x.dtype)

    return paddle.concat([x, pad], axis=-1)


def equality_adjustment(equality, batch):
    """Paddle version of equality_adjustment.

    This is mainly for eval/inference. It mirrors the PyTorch loop logic.
    """
    if equality is None:
        return batch

    out = batch.clone()
    b, l1, l2 = out.shape
    flat = out.reshape([b, l1 * l2])

    for i in range(b):
        mask = equality[i]
        for j in range(l1 * l2):
            for k in range(j + 1, l1 * l2):
                if bool(mask[j, k].item()):
                    avg = (flat[i, j] + flat[i, k]) / 2.0
                    flat[i, j] = avg
                    flat[i, k] = avg

    return flat.reshape([b, l1, l2])


class SiLU(nn.Layer):
    def forward(self, x):
        return x * F.sigmoid(x)


class RBFExpansion(nn.Layer):
    """Paddle version of PyTorch RBFExpansion."""

    def __init__(self, vmin=0.0, vmax=8.0, bins=40, lengthscale=None):
        super().__init__()

        self.vmin = vmin
        self.vmax = vmax
        self.bins = bins

        centers = paddle.linspace(self.vmin, self.vmax, self.bins)
        self.register_buffer("centers", centers)

        if lengthscale is None:
            # Match PyTorch:
            # self.lengthscale = np.diff(self.centers).mean()
            # self.gamma = 1 / self.lengthscale
            if self.bins > 1:
                self.lengthscale = float((self.vmax - self.vmin) / (self.bins - 1))
            else:
                self.lengthscale = float(self.vmax - self.vmin)

            self.gamma = float(1.0 / self.lengthscale)
        else:
            self.lengthscale = float(lengthscale)
            self.gamma = float(1.0 / (self.lengthscale**2))

    def forward(self, distance):
        return paddle.exp(-self.gamma * (distance.unsqueeze(1) - self.centers) ** 2)


class ComformerConv(nn.Layer):
    """First-stage Paddle implementation of ComformerConv.

    This version focuses on matching module names and state_dict keys.
    Full message passing logic will be aligned in the next stage.
    """

    def __init__(self, in_channels, out_channels, heads=1, edge_dim=None):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.edge_dim = edge_dim if edge_dim is not None else out_channels

        self.lin_key = nn.Linear(in_channels, out_channels)
        self.lin_query = nn.Linear(in_channels, out_channels)
        self.lin_value = nn.Linear(in_channels, out_channels)
        self.lin_edge = nn.Linear(self.edge_dim, out_channels)
        self.lin_concate = nn.Linear(out_channels, out_channels)

        self.lin_msg_update = nn.Sequential(
            nn.Linear(out_channels * 3, out_channels),
            SiLU(),
            nn.Linear(out_channels, out_channels),
        )

        self.key_update = nn.Sequential(
            nn.Linear(out_channels * 3, out_channels),
            SiLU(),
            nn.Linear(out_channels, out_channels),
        )

        self.bn = nn.BatchNorm1D(out_channels)
        self.bn_att = nn.BatchNorm1D(out_channels)
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, edge_index=None, edge_attr=None):
        """Paddle implementation of PyG MessagePassing forward.

        PyG convention:
        edge_index[0] = source node j
        edge_index[1] = target node i
        """

        H, C = self.heads, self.out_channels

        if isinstance(x, (tuple, list)):
            x_src, x_dst = x
        else:
            x_src, x_dst = x, x

        query = self.lin_query(x_dst).reshape([-1, H, C])
        key = self.lin_key(x_src).reshape([-1, H, C])
        value = self.lin_value(x_src).reshape([-1, H, C])

        # Fallback path for temporary GMTNet.forward smoke tests.
        # Real graph forward should pass edge_index and edge_attr.
        if edge_index is None:
            out = value.reshape([-1, H * C])
            out = self.lin_concate(out)
            return self.softplus(x_dst + out)

        edge_index = edge_index.astype("int64")
        src = edge_index[0]
        dst = edge_index[1]

        num_nodes = x_dst.shape[0]
        num_edges = src.shape[0]

        if edge_attr is None:
            edge_attr = paddle.zeros([num_edges, self.edge_dim], dtype=x_dst.dtype)

        query_i = paddle.gather(query, dst, axis=0)
        key_i = paddle.gather(key, dst, axis=0)
        key_j = paddle.gather(key, src, axis=0)

        value_i = paddle.gather(value, dst, axis=0)
        value_j = paddle.gather(value, src, axis=0)

        edge_attr = self.lin_edge(edge_attr).reshape([-1, H, C])

        key_j = self.key_update(paddle.concat([key_i, key_j, edge_attr], axis=-1))

        alpha = (query_i * key_j) / (C**0.5)

        out = self.lin_msg_update(paddle.concat([value_i, value_j, edge_attr], axis=-1))

        alpha_bn = self.bn_att(alpha.reshape([-1, C])).reshape([-1, H, C])
        out = out * self.sigmoid(alpha_bn)

        # aggregate messages to target nodes with add aggregation
        out_nodes = paddle.zeros([num_nodes, H, C], dtype=out.dtype)
        index = dst.reshape([-1, 1])
        out_nodes = paddle.scatter_nd_add(out_nodes, index, out)

        out_nodes = out_nodes.reshape([-1, H * C])
        out_nodes = self.lin_concate(out_nodes)

        return self.softplus(x_dst + out_nodes)


class W3JBuffers(nn.Layer):
    """Container for e3nn Wigner-3j buffers.

    This is a structure/key-compatible placeholder.
    """

    def __init__(self, shapes):
        super().__init__()
        for name, shape in shapes.items():
            self.register_buffer(name, paddle.zeros(shape, dtype="float32"))


class FakeTensorProduct(nn.Layer):
    """Key-compatible placeholder for e3nn TensorProduct.

    It registers the same state_dict keys used by the PyTorch checkpoint:
    - weight
    - output_mask
    - _compiled_main_left_right._w3j_...
    """

    def __init__(self, output_mask_dim, w3j_shapes=None):
        super().__init__()

        self.register_buffer("weight", paddle.empty([0], dtype="float32"))
        self.register_buffer(
            "output_mask", paddle.ones([output_mask_dim], dtype="float32")
        )

        if w3j_shapes is not None and len(w3j_shapes) > 0:
            self._compiled_main_left_right = self.add_sublayer(
                "_compiled_main_left_right",
                W3JBuffers(w3j_shapes),
            )


class TensorProductConvLayer(nn.Layer):
    """Paddle e3nn implementation of TensorProductConvLayer."""

    def __init__(
        self,
        in_irreps,
        sh_irreps,
        out_irreps,
        n_edge_features,
        residual=True,
    ):
        super().__init__()

        self.in_irreps = in_irreps
        self.out_irreps = out_irreps
        self.sh_irreps = sh_irreps
        self.residual = residual

        self.tp = o3.FullyConnectedTensorProduct(
            in_irreps,
            sh_irreps,
            out_irreps,
            shared_weights=False,
        )

        self.fc = nn.Sequential(
            nn.Linear(n_edge_features, n_edge_features),
            nn.Softplus(),
            nn.Linear(n_edge_features, self.tp.weight_numel),
        )

    def forward(
        self, node_attr, edge_index, edge_attr, edge_sh, out_nodes=None, reduce="mean"
    ):
        edge_index = edge_index.astype("int64")

        edge_src = edge_index[0]
        edge_dst = edge_index[1]

        if out_nodes is None:
            out_nodes = node_attr.shape[0]

        node_dst = paddle.gather(node_attr, edge_dst, axis=0)
        weight = self.fc(edge_attr)

        tp_out = self.tp(node_dst, edge_sh, weight)

        out_dim = tp_out.shape[-1]

        out = paddle.zeros([out_nodes, out_dim], dtype=tp_out.dtype)
        out = paddle.scatter_nd_add(
            out,
            edge_src.reshape([-1, 1]),
            tp_out,
        )

        if reduce == "mean":
            counts = paddle.zeros([out_nodes, 1], dtype=tp_out.dtype)
            counts = paddle.scatter_nd_add(
                counts,
                edge_src.reshape([-1, 1]),
                paddle.ones([edge_src.shape[0], 1], dtype=tp_out.dtype),
            )
            out = out / paddle.clip(counts, min=1.0)

        if self.residual:
            padded = pad_or_slice_last(node_attr, out.shape[-1])
            out = out + padded

        return out


class ComformerConvEqui(nn.Layer):
    """Paddle e3nn implementation of ComformerConvEqui."""

    def __init__(
        self,
        embsize=128,
        ns=16,
        nv=2,
        residual=True,
    ):
        super().__init__()

        irrep_seq = [
            f"{ns}x0e",
            f"{ns}x0e + {nv}x1o + {nv}x2e",
            f"{ns}x0e + {nv}x1o + {nv}x1e + {nv}x2e + {nv}x2o",
            "1x0e + 1x0o + 1x1e + 1x1o + 1x2e + 1x2o + 1x3e + 1x3o",
        ]

        self.ns = ns
        self.nv = nv

        self.node_linear = nn.Linear(embsize, ns)
        self.sh = "1x0e + 1x1o + 1x2e"

        self.nlayer_1 = TensorProductConvLayer(
            in_irreps=irrep_seq[0],
            sh_irreps=self.sh,
            out_irreps=irrep_seq[1],
            n_edge_features=embsize,
            residual=residual,
        )

        self.nlayer_2 = TensorProductConvLayer(
            in_irreps=irrep_seq[1],
            sh_irreps=self.sh,
            out_irreps=irrep_seq[2],
            n_edge_features=embsize,
            residual=False,
        )

        self.nlayer_3 = TensorProductConvLayer(
            in_irreps=irrep_seq[2],
            sh_irreps=self.sh,
            out_irreps=irrep_seq[3],
            n_edge_features=embsize,
            residual=False,
        )

    def forward(self, data, node_features=None, edge_index=None, edge_features=None):
        edge_vec = data.edge_attr

        edge_irr = o3.spherical_harmonics(
            self.sh,
            edge_vec,
            normalize=True,
            normalization="component",
        )

        node_feature = self.node_linear(node_features)
        node_feature = self.nlayer_1(node_feature, edge_index, edge_features, edge_irr)
        node_feature = self.nlayer_2(node_feature, edge_index, edge_features, edge_irr)
        node_feature = self.nlayer_3(node_feature, edge_index, edge_features, edge_irr)

        return node_feature


class GradientBlock(nn.Layer):
    """Paddle e3nn eval implementation of Gradient_block.

    Note:
    create_graph=True is not supported because Paddle einsum_grad
    currently does not support higher-order grad.
    This implementation is for eval/test forward.
    """

    def __init__(self, nv=2):
        super().__init__()

        irrep_seq = [
            "1x0e + 1x0o + 1x1e + 1x1o + 1x2e + 1x2o + 1x3e + 1x3o",
            "1x1o",
        ]

        self.nv = nv
        self.sh = "1x1o"

        self.tp = o3.FullyConnectedTensorProduct(
            irrep_seq[0],
            self.sh,
            irrep_seq[1],
            internal_weights=False,
        )

        self.register_buffer(
            "constant_w",
            paddle.ones([self.tp.weight_numel], dtype="float32"),
        )

    def _training_dielectric(self, node_feature):
        input_slices = self.tp.irreps_in1.slices()
        field_mul_ir = self.tp.irreps_in2[0]
        output_mul_ir = self.tp.irreps_out[0]

        if field_mul_ir.mul != 1 or field_mul_ir.ir.l != 1:
            raise RuntimeError("GradientBlock requires a single 1o field irrep.")
        if output_mul_ir.mul != 1 or output_mul_ir.ir.l != 1:
            raise RuntimeError("GradientBlock requires a single 1o output irrep.")

        batch_size = node_feature.shape[0]
        field_dim = field_mul_ir.ir.dim
        output_dim = output_mul_ir.ir.dim

        if field_dim != 3 or output_dim != 3:
            raise RuntimeError("GradientBlock requires three-dimensional field/output.")

        dielectric = paddle.zeros(
            [batch_size, output_mul_ir.mul, output_dim, field_dim],
            dtype=node_feature.dtype,
        )
        external_weight = self.constant_w.astype(node_feature.dtype)

        for instruction_index, instruction, weight_view in self.tp.weight_views(
            external_weight,
            yield_instruction=True,
        ):
            if instruction.connection_mode != "uvw" or not instruction.has_weight:
                raise RuntimeError(
                    "GradientBlock only supports weighted uvw tensor-product paths."
                )
            if instruction.i_in2 != 0 or instruction.i_out != 0:
                raise RuntimeError(
                    "GradientBlock only supports the configured single field/output irrep."
                )

            input_mul_ir = self.tp.irreps_in1[instruction.i_in1]
            path_output_mul_ir = self.tp.irreps_out[instruction.i_out]

            if weight_view.shape != [
                input_mul_ir.mul,
                field_mul_ir.mul,
                path_output_mul_ir.mul,
            ]:
                raise RuntimeError("Unexpected tensor-product external weight shape.")

            input_feature = node_feature[:, input_slices[instruction.i_in1]]
            input_feature = input_feature.reshape(
                [batch_size, input_mul_ir.mul, input_mul_ir.ir.dim]
            )

            coupling = o3.wigner_3j(
                input_mul_ir.ir.l,
                field_mul_ir.ir.l,
                path_output_mul_ir.ir.l,
                dtype=node_feature.dtype,
                device=node_feature.place,
            )

            feature_term = input_feature.reshape(
                [batch_size, input_mul_ir.mul, input_mul_ir.ir.dim, 1, 1, 1]
            )
            coupling_term = coupling.reshape(
                [1, 1, input_mul_ir.ir.dim, field_dim, output_dim, 1]
            )
            weight_term = weight_view[:, 0, :].reshape(
                [1, input_mul_ir.mul, 1, 1, 1, path_output_mul_ir.mul]
            )

            coefficient_qkw = paddle.sum(
                feature_term * coupling_term * weight_term,
                axis=[1, 2],
            )
            coefficient_qkw = instruction.path_weight * coefficient_qkw
            coefficient_wkq = coefficient_qkw.transpose([0, 3, 2, 1])

            dielectric = dielectric + coefficient_wkq

        spherical_derivative = math.sqrt(3.0 / (4.0 * math.pi))
        dielectric = spherical_derivative * dielectric
        return dielectric.reshape([batch_size, output_dim, field_dim])

    def forward(self, node_feature):
        if self.training:
            return self._training_dielectric(node_feature)

        bs = node_feature.shape[0]

        outer_E = paddle.ones([bs, 3], dtype=node_feature.dtype)
        outer_E.stop_gradient = False

        E_ = o3.spherical_harmonics(
            self.sh,
            outer_E,
            normalize=False,
        )

        D_ = self.tp(
            node_feature,
            E_,
            self.constant_w.astype(node_feature.dtype),
        )

        dielectric = []

        for i in range(3):
            grad_outputs = paddle.zeros([bs, 3], dtype=node_feature.dtype)
            grad_outputs[:, i] = 1.0

            grad_i = paddle.grad(
                outputs=[D_],
                inputs=[outer_E],
                grad_outputs=[grad_outputs],
                create_graph=False,
                retain_graph=True,
            )[0]

            dielectric.append(grad_i)

        return paddle.stack(dielectric, axis=0).transpose([1, 0, 2])


class GMTNet(nn.Layer):
    """First-stage Paddle GMTNet with RBFExpansion and ComformerConv."""

    requires_forward_grad = True

    def __init__(self, args, loss_cfg=None):
        super().__init__()

        atom_input_features = get_arg(args, "atom_input_features", 92)
        embsize = get_arg(args, "embedding_features", 128)
        edge_features = get_arg(args, "edge_features", 512)
        output_features = get_arg(args, "output_features", 9)
        num_layers = get_arg(args, "num_layers", 2)

        self.atom_embedding = nn.Linear(atom_input_features, embsize)

        self.rbf = nn.Sequential(
            RBFExpansion(vmin=-4.0, vmax=0.0, bins=edge_features),
            nn.Linear(edge_features, embsize),
            nn.Softplus(),
        )

        self.att_layers = nn.LayerList(
            [
                ComformerConv(
                    in_channels=embsize,
                    out_channels=embsize,
                    heads=1,
                    edge_dim=embsize,
                )
                for _ in range(num_layers)
            ]
        )

        self.equi_update = ComformerConvEqui(embsize)
        self.output_block = GradientBlock()

        self.mask = get_arg(args, "use_mask", False)
        self.reduce = get_arg(args, "reduce_cell", False)

        self.etgnn_linear = nn.Linear(embsize, 1)

        if loss_cfg is None:
            self.loss_fn = None
        else:
            try:
                self.loss_fn = build_loss(copy.deepcopy(loss_cfg))
            except Exception as error:
                raise ValueError(f"Invalid loss_cfg: {error}") from error

    @staticmethod
    def _validate_mapping_tensor(name, value, trailing_shape, dtype=None):
        if not isinstance(value, paddle.Tensor):
            raise TypeError(f"Mapping key '{name}' must be a Paddle Tensor")
        if value.ndim != 3:
            raise ValueError(
                f"Mapping key '{name}' must have rank 3, got shape {list(value.shape)}"
            )
        if list(value.shape[1:]) != list(trailing_shape):
            raise ValueError(
                f"Mapping key '{name}' must have shape [B,{trailing_shape[0]},{trailing_shape[1]}], got {list(value.shape)}"
            )
        if dtype is not None and value.dtype != dtype:
            raise TypeError(
                f"Mapping key '{name}' must have dtype {dtype}, got {value.dtype}"
            )

    def _mapping_inputs(self, batch):
        for key in ("graph", "feature_mask", "matrix_equal"):
            if key not in batch:
                raise ValueError(f"Mapping input is missing required key '{key}'")
        feature_mask = batch["feature_mask"]
        matrix_equal = batch["matrix_equal"]
        self._validate_mapping_tensor("feature_mask", feature_mask, (32, 32))
        self._validate_mapping_tensor(
            "matrix_equal", matrix_equal, (9, 9), dtype=paddle.bool
        )
        if feature_mask.shape[0] != matrix_equal.shape[0]:
            raise ValueError(
                "Mapping feature_mask and matrix_equal batch sizes must match"
            )
        dielectric = batch.get("dielectric")
        if dielectric is not None:
            self._validate_mapping_tensor("dielectric", dielectric, (3, 3))
            if dielectric.shape[0] != feature_mask.shape[0]:
                raise ValueError(
                    "Mapping dielectric and feature_mask batch sizes must match"
                )
        return batch["graph"], feature_mask, matrix_equal, dielectric

    def _forward_tensor(self, data, feat_mask, equality):
        """Paddle GMTNet forward.

        Current stage:
        - atom_embedding: real
        - RBF edge feature: real
        - ComformerConv att_layers: real
        - equi_update: placeholder
        - output_block: placeholder

        Therefore this is still not final metric reproduction.
        """

        node_features = self.atom_embedding(data.x)

        edge_feat = -0.75 / paddle.norm(data.edge_attr, axis=1)
        edge_features = self.rbf(edge_feat)

        node_features = self.att_layers[0](
            node_features,
            data.edge_index,
            edge_features,
        )

        node_features = self.att_layers[1](
            node_features,
            data.edge_index,
            edge_features,
        )

        # Placeholder currently returns node_features unchanged.
        node_features = self.equi_update(
            data,
            node_features,
            data.edge_index,
            edge_features,
        )

        # Manual global mean pooling by data.batch.
        if hasattr(data, "batch"):
            batch = data.batch.astype("int64")
        else:
            batch = paddle.zeros([node_features.shape[0]], dtype="int64")

        num_graphs = int(paddle.max(batch).item()) + 1

        crystal_features = paddle.zeros(
            [num_graphs, node_features.shape[1]],
            dtype=node_features.dtype,
        )

        crystal_features = paddle.scatter_nd_add(
            crystal_features,
            batch.reshape([-1, 1]),
            node_features,
        )

        counts = paddle.zeros([num_graphs, 1], dtype=node_features.dtype)
        counts = paddle.scatter_nd_add(
            counts,
            batch.reshape([-1, 1]),
            paddle.ones([node_features.shape[0], 1], dtype=node_features.dtype),
        )

        crystal_features = crystal_features / paddle.clip(counts, min=1.0)

        if self.mask and feat_mask is not None:
            crystal_features = paddle.bmm(
                feat_mask,
                crystal_features.unsqueeze(-1),
            ).squeeze(-1)

        outputs = self.output_block(crystal_features)

        if equality is not None:
            outputs = equality_adjustment(equality, outputs)

        return outputs

    def forward(self, data, feat_mask=None, equality=None):
        if isinstance(data, Mapping):
            if feat_mask is not None or equality is not None:
                raise ValueError(
                    "Mapping input must not be combined with feat_mask or equality arguments"
                )
            graph, feature_mask, matrix_equal, dielectric = self._mapping_inputs(data)
            prediction = self._forward_tensor(graph, feature_mask, matrix_equal)
            result = {"pred_dict": {"dielectric": prediction}}
            if dielectric is None:
                return result
            if self.loss_fn is None:
                raise ValueError(
                    "Dielectric labels were provided, but no loss function was explicitly configured. No default loss is available."
                )
            result["loss_dict"] = {"loss": self.loss_fn(prediction, dielectric)}
            return result
        if feat_mask is None or equality is None:
            raise ValueError(
                "Raw GMTNet calls require both feat_mask and equality arguments"
            )
        return self._forward_tensor(data, feat_mask, equality)

    def predict(self, data):
        """Predict dielectric tensors from label-free GMTNet Mapping inputs."""
        if isinstance(data, list):
            predictions = [self.forward(item)["pred_dict"]["dielectric"] for item in data]
            return {"pred_dict": {"dielectric": paddle.concat(predictions, axis=0)}}
        return self.forward(data)
