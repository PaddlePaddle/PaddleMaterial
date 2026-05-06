"""GNN 通用子模块：Swish、GraphNorm、VPA 聚合、FourierLinear 等。"""
import math
from typing import Optional

import paddle
import paddle.nn as nn
from ppmat.models.sgequidiff.scatter_utils import safe_scatter as paddle_scatter_scatter


class Swish(nn.Layer):
    """
    Swish 激活函数。
    """
    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        return nn.functional.silu(x) / 0.6

class VariancePreservingAggregation(nn.Layer):
    """方差保持聚合: vpa(X) = sum(X) / sqrt(|X|)。"""

    def forward(
        self,
        src: paddle.Tensor,
        index: paddle.Tensor,
        dim_size: Optional[int] = None,
    ) -> paddle.Tensor:
        """(n_edges, hidden_dim) -> (n_nodes, hidden_dim)"""
        if dim_size is None:
            dim_size = int(index.max().item()) + 1

        sum_agg = paddle_scatter_scatter(
            src, index, dim=0, dim_size=dim_size, reduce="sum"
        )  # (n_nodes, hidden_dim)
        counts = paddle_scatter_scatter(
            paddle.ones([src.shape[0]], dtype=src.dtype),
            index,
            dim=0,
            dim_size=dim_size,
            reduce="sum",
        )  # (n_nodes,)
        return paddle.nan_to_num(sum_agg / paddle.sqrt(counts).unsqueeze(-1))

class FourierLinear(nn.Layer):
    """将 3D 点特征化为 Fourier 特征。"""

    def __init__(
        self,
        input_dim: int,
        num_fourier_frequencies: int,
        scale: float,
        output_dim: int,
        num_layers: int = 1,
        use_bias: bool = False,
    ):
        super().__init__()
        assert num_layers >= 1
        self.num_fourier_frequencies = num_fourier_frequencies
        self.scale = scale
        self.output_dim = output_dim
        self.num_layers = num_layers

        if self.scale > 0:
            self.fourier_freqs = paddle.create_parameter(
                shape=[input_dim, num_fourier_frequencies],
                dtype="float32",
                default_initializer=nn.initializer.Normal(std=scale),
            )
            self.fourier_freqs.stop_gradient = True
            in_dim = input_dim + 2 * num_fourier_frequencies
            self.layer = nn.Linear(in_dim, output_dim, bias_attr=use_bias)
        else:
            in_dim = input_dim
            self.layer = nn.Linear(in_dim, output_dim, bias_attr=use_bias)
        self.weight = self.layer.weight
        if num_layers > 1:
            self.layers = nn.LayerList(
                [nn.Linear(output_dim, output_dim) for _ in range(num_layers - 1)]
            )

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        if self.scale > 0:
            with paddle.no_grad():
                v = 2 * math.pi * x @ self.fourier_freqs
                v = paddle.concat([x, v.sin(), v.cos()], axis=-1)
        else:
            v = x
        v = self.layer(v)
        if self.num_layers > 1:
            for layer in self.layers:
                v = nn.functional.silu(layer(v)) + v
        return v

class GraphNorm(nn.Layer):
    """图归一化层。"""

    def __init__(self, in_channels: int, eps: float = 1e-5):
        super().__init__()
        self.in_channels = in_channels
        self.eps = eps
        self.weight = self.create_parameter(
            [in_channels],
            default_initializer=nn.initializer.Constant(1.0),
        )
        self.bias = self.create_parameter(
            [in_channels],
            default_initializer=nn.initializer.Constant(0.0),
        )
        self.mean_scale = self.create_parameter(
            [in_channels],
            default_initializer=nn.initializer.Constant(1.0),
        )

    def forward(
        self,
        x: paddle.Tensor,
        map_node_to_graph: paddle.Tensor,
        num_graphs: int,
    ) -> paddle.Tensor:
        """(num_nodes, in_channels) -> (num_nodes, in_channels)"""

        sorted_order = paddle.argsort(map_node_to_graph)
        sorted_map = map_node_to_graph[sorted_order]
        sorted_x = x[sorted_order]

        mean = paddle_scatter_scatter(
            sorted_x, sorted_map, dim=0, dim_size=num_graphs, reduce="mean"
        )  # (num_graphs, in_channels)

        out = x - mean[map_node_to_graph] * self.mean_scale

        sorted_out = out[sorted_order]
        var = paddle_scatter_scatter(
            sorted_out ** 2, sorted_map, dim=0, dim_size=num_graphs, reduce="mean"
        )  # (num_graphs, in_channels)

        std = (var + self.eps).sqrt()[map_node_to_graph].clip(min=1.0)
        return self.weight * out / std + self.bias

class Envelope(nn.Layer):
    """
    平滑截断包络函数，取自 GemNet。
    """

    def __init__(self, cutoff_radius: float, exponent: int = 5):
        super().__init__()
        assert exponent > 0
        self.cutoff_radius = cutoff_radius
        self.p = exponent
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = -self.p * (self.p + 1) / 2

    def forward(self, inputs: paddle.Tensor) -> paddle.Tensor:
        d_scaled = inputs / self.cutoff_radius
        env_val = (
            1
            + self.a * d_scaled ** self.p
            + self.b * d_scaled ** (self.p + 1)
            + self.c * d_scaled ** (self.p + 2)
        )
        return paddle.where(
            inputs < self.cutoff_radius,
            env_val,
            paddle.zeros_like(d_scaled),
        )
