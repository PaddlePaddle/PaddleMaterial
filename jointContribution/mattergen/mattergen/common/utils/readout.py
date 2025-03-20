from abc import ABC
from abc import abstractmethod
from typing import List

import paddle
from paddle_scatter import scatter
# from paddle_scatter import scatter_softmax
from typing_extensions import Literal

from paddle_utils import *  # noqa


class MLP(paddle.nn.Layer):
    def __init__(
        self,
        input_dim: int,
        out_dim: int,
        hidden_layer_dims: List[int],
        activation=paddle.nn.ReLU(),
    ):
        super().__init__()
        layers = []
        cur_hidden_dim = input_dim
        for hidden_layer_dim in hidden_layer_dims:
            layers.append(
                paddle.nn.Linear(in_features=cur_hidden_dim, out_features=hidden_layer_dim)  # noqa
            )
            layers.append(activation)
            cur_hidden_dim = hidden_layer_dim
        layers.append(paddle.nn.Linear(in_features=cur_hidden_dim, out_features=out_dim))  # noqa
        self._layers = paddle.nn.Sequential(*layers)

    def forward(self, inputs):
        return self._layers(inputs)


class GraphReadout(paddle.nn.Layer, ABC):
    def __init__(self, node_dim: int, out_dim: int):
        """
        Args:
            node_dim: Dimension of each node node representation.
            out_dim: Dimension of the graph representation to produce.
        """
        super().__init__()
        self._node_dim = node_dim
        self._out_dim = out_dim

    @abstractmethod
    def forward(
        self,
        node_embeddings: paddle.Tensor,
        node_to_graph_id: paddle.Tensor,
        num_graphs: int,
    ) -> paddle.Tensor:
        """
        Args:
            node_embeddings: representations of individual graph nodes. A float tensor
                of shape [num_nodes, self.node_dim].
            node_to_graph_id: int tensor of shape [num_nodes], assigning a graph_id to
                each node.
            num_graphs: int scalar, giving the number of graphs in the batch.

        Returns:
            float tensor of shape [num_graphs, out_dim]
        """
        pass


class CombinedGraphReadout(GraphReadout):
    def __init__(self, node_dim: int, out_dim: int, num_heads: int, head_dim: int):
        """
        See superclass for first few parameters.

        Args:
            num_heads: Number of independent heads to use for independent weights.
            head_dim: Size of the result of each independent head.
            num_mlp_layers: Number of layers in the MLPs used to compute per-head
            weights and outputs.
        """
        super().__init__(node_dim, out_dim)
        self._num_heads = num_heads
        self._head_dim = head_dim
        self._weighted_mean_pooler = MultiHeadWeightedGraphReadout(
            node_dim=node_dim,
            out_dim=out_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            weighting_type="weighted_mean",
        )
        self._weighted_sum_pooler = MultiHeadWeightedGraphReadout(
            node_dim=node_dim,
            out_dim=out_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            weighting_type="weighted_sum",
        )
        self._max_pooler = UnweightedGraphReadout(
            node_dim=node_dim, out_dim=out_dim, pooling_type="max"
        )
        self._combination_layer = paddle.nn.Linear(
            in_features=3 * out_dim, out_features=out_dim, bias_attr=False
        )

    def forward(
        self,
        node_embeddings: paddle.Tensor,
        node_to_graph_id: paddle.Tensor,
        num_graphs: int,
    ) -> paddle.Tensor:
        mean_graph_repr = self._weighted_mean_pooler(
            node_embeddings, node_to_graph_id, num_graphs
        )  # noqa
        sum_graph_repr = self._weighted_sum_pooler(
            node_embeddings, node_to_graph_id, num_graphs
        )  # noqa
        max_graph_repr = self._max_pooler(node_embeddings, node_to_graph_id, num_graphs)  # noqa
        raw_graph_repr = paddle.concat(
            x=(mean_graph_repr, sum_graph_repr, max_graph_repr), axis=1
        )  # noqa
        return self._combination_layer(paddle.nn.functional.relu(x=raw_graph_repr))


class MultiHeadWeightedGraphReadout(GraphReadout):
    def __init__(
        self,
        node_dim: int,
        out_dim: int,
        num_heads: int,
        head_dim: int,
        weighting_type: Literal["weighted_sum", "weighted_mean"],
        num_mlp_layers: int = 1,
    ):
        """
        See superclass for first few parameters.

        Args:
            num_heads: Number of independent heads to use for independent weights.
            head_dim: Size of the result of each independent head.
            weighting_type: Type of weighting to use, either "weighted_sum" (weights
                are in [0, 1], obtained through a logistic sigmoid) or "weighted_mean"
                (weights are in [0, 1] and sum up to 1 for each graph, obtained through
                a softmax).
            num_mlp_layers: Number of layers in the MLPs used to compute per-head
                weights and outputs.
        """
        super().__init__(node_dim, out_dim)
        self._num_heads = num_heads
        self._head_dim = head_dim
        if weighting_type not in ("weighted_sum", "weighted_mean"):
            raise ValueError(f"Unknown weighting type {weighting_type}!")
        self._weighting_type = weighting_type
        self._scoring_module = MLP(
            input_dim=self._node_dim,
            hidden_layer_dims=[self._head_dim * num_heads] * num_mlp_layers,
            out_dim=num_heads,
        )
        self._transformation_mlp = MLP(
            input_dim=self._node_dim,
            hidden_layer_dims=[self._head_dim * num_heads] * num_mlp_layers,
            out_dim=num_heads * head_dim,
        )
        self._combination_layer = paddle.nn.Linear(
            in_features=num_heads * head_dim, out_features=out_dim, bias_attr=False
        )

    def forward(
        self,
        node_embeddings: paddle.Tensor,
        node_to_graph_id: paddle.Tensor,
        num_graphs: int,
    ) -> paddle.Tensor:
        scores = self._scoring_module(node_embeddings)
        if self._weighting_type == "weighted_sum":
            weights = paddle.nn.functional.sigmoid(x=scores)
        elif self._weighting_type == "weighted_mean":
            # weights = scatter_softmax(scores, index=node_to_graph_id, dim=0)
            raise NotImplementedError()
        else:
            raise ValueError(f"Unknown weighting type {self._weighting_type}!")
        values = self._transformation_mlp(node_embeddings)
        values = values.view(-1, self._num_heads, self._head_dim)
        weighted_values = weights.unsqueeze(axis=-1) * values
        per_graph_values = paddle.zeros(
            shape=(num_graphs, self._num_heads * self._head_dim)
        )  # noqa
        per_graph_values.index_add_(
            axis=0,
            index=node_to_graph_id,
            value=weighted_values.view(-1, self._num_heads * self._head_dim),
        )
        return self._combination_layer(per_graph_values)


class UnweightedGraphReadout(GraphReadout):
    def __init__(
        self,
        node_dim: int,
        out_dim: int,
        pooling_type: Literal["min", "max", "sum", "mean"],
    ):
        """
        See superclass for first few parameters.

        Args:
            pooling_type: Type of pooling to use. One of "min", "max", "sum" and "mean".
        """
        super().__init__(node_dim, out_dim)
        self._pooling_type = pooling_type
        if pooling_type not in ("min", "max", "sum", "mean"):
            raise ValueError(f"Unknown weighting type {self.pooling_type}!")
        self._combination_layer = paddle.nn.Linear(
            in_features=self._node_dim, out_features=out_dim, bias_attr=False
        )

    def forward(
        self,
        node_embeddings: paddle.Tensor,
        node_to_graph_id: paddle.Tensor,
        num_graphs: int,
    ) -> paddle.Tensor:
        per_graph_values = scatter(
            src=node_embeddings,
            index=node_to_graph_id,
            dim=0,
            dim_size=num_graphs,
            reduce=self._pooling_type,
        )
        return self._combination_layer(per_graph_values)
