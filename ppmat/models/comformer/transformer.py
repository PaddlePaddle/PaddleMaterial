import math
from typing import Optional
from typing import Tuple
from typing import Union

import paddle

from ppmat.models.comformer.message_passing.message_passing import MessagePassing


class ComformerConv(MessagePassing):
    _alpha: paddle.Tensor

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        beta: bool = False,
        dropout: float = 0.0,
        edge_dim: Optional[int] = None,
        bias: bool = True,
        root_weight: bool = True,
        **kwargs
    ):
        kwargs.setdefault("aggr", "add")
        super(ComformerConv, self).__init__(node_dim=0, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.beta = beta and root_weight
        self.root_weight = root_weight
        self.concat = concat
        self.dropout = dropout
        self.edge_dim = edge_dim
        self._alpha = None
        if isinstance(in_channels, int):
            in_channels = in_channels, in_channels
        self.lin_key = paddle.nn.Linear(
            in_features=in_channels[0], out_features=heads * out_channels
        )
        self.lin_query = paddle.nn.Linear(
            in_features=in_channels[1], out_features=heads * out_channels
        )
        self.lin_value = paddle.nn.Linear(
            in_features=in_channels[0], out_features=heads * out_channels
        )
        self.lin_edge = paddle.nn.Linear(
            in_features=edge_dim, out_features=heads * out_channels
        )
        self.lin_concate = paddle.nn.Linear(
            in_features=heads * out_channels, out_features=out_channels
        )
        self.lin_msg_update = paddle.nn.Sequential(
            paddle.nn.Linear(in_features=out_channels * 3, out_features=out_channels),
            paddle.nn.Silu(),
            paddle.nn.Linear(in_features=out_channels, out_features=out_channels),
        )
        self.softplus = paddle.nn.Softplus()
        self.silu = paddle.nn.Silu()
        self.key_update = paddle.nn.Sequential(
            paddle.nn.Linear(in_features=out_channels * 3, out_features=out_channels),
            paddle.nn.Silu(),
            paddle.nn.Linear(in_features=out_channels, out_features=out_channels),
        )
        self.bn = paddle.nn.BatchNorm1D(num_features=out_channels)
        self.bn_att = paddle.nn.BatchNorm1D(num_features=out_channels)
        self.sigmoid = paddle.nn.Sigmoid()

    def forward(
        self,
        x: paddle.Tensor,
        edge_index,
        edge_attr=None,
        return_attention_weights=None,
    ):
        H, C = self.heads, self.out_channels
        if isinstance(x, paddle.Tensor):
            x = (x, x)
        query = self.lin_query(x[1]).reshape([-1, H, C])
        key = self.lin_key(x[0]).reshape([-1, H, C])
        value = self.lin_value(x[0]).reshape([-1, H, C])
        out = self.propagate(
            edge_index,
            query=query,
            key=key,
            value=value,
            edge_attr=edge_attr,
            size=None,
        )
        out = out.reshape([-1, self.heads * self.out_channels])
        out = self.lin_concate(out)
        return self.softplus(x[1] + self.bn(out))

    def message(
        self,
        query_i: paddle.Tensor,
        key_i: paddle.Tensor,
        key_j: paddle.Tensor,
        value_j: paddle.Tensor,
        value_i: paddle.Tensor,
        edge_attr: paddle.Tensor,
        index: paddle.Tensor,
        ptr: paddle.Tensor,
        size_i: Optional[int],
    ) -> paddle.Tensor:
        edge_attr = self.lin_edge(edge_attr).reshape(
            [-1, self.heads, self.out_channels]
        )
        key_j = self.key_update(paddle.concat(x=(key_i, key_j, edge_attr), axis=-1))
        alpha = query_i * key_j / math.sqrt(self.out_channels)
        out = self.lin_msg_update(
            paddle.concat(x=(value_i, value_j, edge_attr), axis=-1)
        )
        out = out * self.sigmoid(
            self.bn_att(alpha.reshape([-1, self.out_channels])).reshape(
                [-1, self.heads, self.out_channels]
            )
        )
        return out


class ComformerConv_edge(paddle.nn.Layer):
    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        beta: bool = False,
        dropout: float = 0.0,
        edge_dim: Optional[int] = None,
        bias: bool = True,
        root_weight: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.beta = beta and root_weight
        self.root_weight = root_weight
        self.concat = concat
        self.dropout = dropout
        self.edge_dim = edge_dim
        if isinstance(in_channels, int):
            in_channels = in_channels, in_channels
        self.lemb = paddle.nn.Embedding(num_embeddings=3, embedding_dim=32)
        self.embedding_dim = 32
        self.lin_key = paddle.nn.Linear(
            in_features=in_channels[0], out_features=heads * out_channels
        )
        self.lin_query = paddle.nn.Linear(
            in_features=in_channels[1], out_features=heads * out_channels
        )
        self.lin_value = paddle.nn.Linear(
            in_features=in_channels[0], out_features=heads * out_channels
        )
        self.lin_key_e1 = paddle.nn.Linear(
            in_features=in_channels[0], out_features=heads * out_channels
        )
        self.lin_value_e1 = paddle.nn.Linear(
            in_features=in_channels[0], out_features=heads * out_channels
        )
        self.lin_key_e2 = paddle.nn.Linear(
            in_features=in_channels[0], out_features=heads * out_channels
        )
        self.lin_value_e2 = paddle.nn.Linear(
            in_features=in_channels[0], out_features=heads * out_channels
        )
        self.lin_key_e3 = paddle.nn.Linear(
            in_features=in_channels[0], out_features=heads * out_channels
        )
        self.lin_value_e3 = paddle.nn.Linear(
            in_features=in_channels[0], out_features=heads * out_channels
        )
        self.lin_edge = paddle.nn.Linear(
            in_features=edge_dim, out_features=heads * out_channels, bias_attr=False
        )
        self.lin_edge_len = paddle.nn.Linear(
            in_features=in_channels[0] + self.embedding_dim, out_features=in_channels[0]
        )
        self.lin_concate = paddle.nn.Linear(
            in_features=heads * out_channels, out_features=out_channels
        )
        self.lin_msg_update = paddle.nn.Sequential(
            paddle.nn.Linear(in_features=out_channels * 3, out_features=out_channels),
            paddle.nn.Silu(),
            paddle.nn.Linear(in_features=out_channels, out_features=out_channels),
        )
        self.silu = paddle.nn.Silu()
        self.softplus = paddle.nn.Softplus()
        self.key_update = paddle.nn.Sequential(
            paddle.nn.Linear(in_features=out_channels * 3, out_features=out_channels),
            paddle.nn.Silu(),
            paddle.nn.Linear(in_features=out_channels, out_features=out_channels),
        )
        self.bn_att = paddle.nn.BatchNorm1D(num_features=out_channels)
        self.bn = paddle.nn.BatchNorm1D(num_features=out_channels)
        self.sigmoid = paddle.nn.Sigmoid()

    def forward(
        self,
        edge: paddle.Tensor,
        edge_nei_len: paddle.Tensor = None,
        edge_nei_angle: paddle.Tensor = None,
    ):
        H, C = self.heads, self.out_channels
        if isinstance(edge, paddle.Tensor):
            edge = (edge, edge)

        query_x = (
            self.lin_query(edge[1])
            .reshape([-1, H, C])
            .unsqueeze(axis=1)
            .tile(repeat_times=[1, 3, 1, 1])
        )
        key_x = (
            self.lin_key(edge[0])
            .reshape([-1, H, C])
            .unsqueeze(axis=1)
            .tile(repeat_times=[1, 3, 1, 1])
        )
        value_x = (
            self.lin_value(edge[0])
            .reshape([-1, H, C])
            .unsqueeze(axis=1)
            .tile(repeat_times=[1, 3, 1, 1])
        )

        key_y = paddle.concat(
            x=(
                self.lin_key_e1(edge_nei_len[:, 0, :]).reshape([-1, 1, H, C]),
                self.lin_key_e2(edge_nei_len[:, 1, :]).reshape([-1, 1, H, C]),
                self.lin_key_e3(edge_nei_len[:, 2, :]).reshape([-1, 1, H, C]),
            ),
            axis=1,
        )
        value_y = paddle.concat(
            x=(
                self.lin_value_e1(edge_nei_len[:, 0, :]).reshape([-1, 1, H, C]),
                self.lin_value_e2(edge_nei_len[:, 1, :]).reshape([-1, 1, H, C]),
                self.lin_value_e3(edge_nei_len[:, 2, :]).reshape([-1, 1, H, C]),
            ),
            axis=1,
        )
        edge_xy = self.lin_edge(edge_nei_angle).reshape([-1, 3, H, C])
        key = self.key_update(paddle.concat(x=(key_x, key_y, edge_xy), axis=-1))
        alpha = query_x * key / math.sqrt(self.out_channels)
        out = self.lin_msg_update(paddle.concat(x=(value_x, value_y, edge_xy), axis=-1))
        out = out * self.sigmoid(
            self.bn_att(alpha.reshape([-1, self.out_channels])).reshape(
                [-1, 3, self.heads, self.out_channels]
            )
        )
        out = out.reshape([-1, 3, self.heads * self.out_channels])
        out = self.lin_concate(out)
        out = out.sum(axis=1)
        return self.softplus(edge[1] + self.bn(out))
