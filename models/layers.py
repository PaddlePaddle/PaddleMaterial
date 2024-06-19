from __future__ import annotations
import sys
import paddle
from typing import TYPE_CHECKING, Any, Callable, Literal
from collections.abc import Sequence
import math
from enum import Enum
import paddle.nn as nn

from pgl.math import segment_sum, segment_softmax, segment_pool

class MLP(paddle.nn.Layer):
    """An implementation of a multi-layer perceptron."""

    def __init__(self, dims: Sequence[int], activation: (Callable[[paddle.Tensor],
        paddle.Tensor] | None)=None, activate_last: bool=False, bias_last:
        bool=True) ->None:
        """:param dims: Dimensions of each layer of MLP.
        :param activation: Activation function.
        :param activate_last: Whether to apply activation to last layer.
        :param bias_last: Whether to apply bias to last layer.
        """
        super().__init__()
        self._depth = len(dims) - 1
        self.layers = paddle.nn.LayerList()
        bias_attr = paddle.ParamAttr(initializer=paddle.nn.initializer.XavierNormal())
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            if i < self._depth - 1:
                self.layers.append(paddle.nn.Linear(in_features=in_dim,
                    out_features=out_dim, bias_attr=bias_attr))
                if activation is not None:
                    self.layers.append(activation)
            else:
                if bias_last:
                    bias_last = bias_attr
                self.layers.append(paddle.nn.Linear(in_features=in_dim,
                    out_features=out_dim, bias_attr=bias_last))
                if activation is not None and activate_last:
                    self.layers.append(activation)

    # def __repr__(self):
    #     dims = []
    #     for layer in self.layers:
    #         if isinstance(layer, paddle.nn.Linear):
    #             dims.append(f'{layer.in_features} → {layer.out_features}')
    #         else:
    #             dims.append(layer.__class__.__name__)
    #     return f"MLP({', '.join(dims)})"

    @property
    def last_linear(self) ->(Linear | None):
        """:return: The last linear layer."""
        for layer in reversed(self.layers):
            if isinstance(layer, paddle.nn.Linear):
                return layer
        raise RuntimeError

    @property
    def depth(self) ->int:
        """Returns depth of MLP."""
        return self._depth

    @property
    def in_features(self) ->int:
        """Return input features of MLP."""
        return self.layers[0].in_features

    @property
    def out_features(self) ->int:
        """Returns output features of MLP."""
        for layer in reversed(self.layers):
            if isinstance(layer, paddle.nn.Linear):
                return layer.out_features
        raise RuntimeError

    def forward(self, inputs):
        """Applies all layers in turn.

        :param inputs: Input tensor
        :return: Output tensor
        """
        x = inputs
        for layer in self.layers:
            x = layer(x)
        return x


class SoftPlus2(paddle.nn.Layer):
    """SoftPlus2 activation function:
    out = log(exp(x)+1) - log(2)
    softplus function that is 0 at x=0, the implementation aims at avoiding overflow.
    """

    def __init__(self) ->None:
        """Initializes the SoftPlus2 class."""
        super().__init__()
        self.ssp = paddle.nn.Softplus()

    def forward(self, x: paddle.Tensor) ->paddle.Tensor:
        """Evaluate activation function given the input tensor x.

        Args:
            x (torch.tensor): Input tensor

        Returns:
            out (torch.tensor): Output tensor
        """
        return self.ssp(x) - math.log(2.0)


class ActivationFunction(Enum):
    """Enumeration of optional activation functions."""
    softplus2 = SoftPlus2


class EmbeddingBlock(paddle.nn.Layer):
    """Embedding block for generating node, bond and state features."""

    def __init__(self, degree_rbf: int, activation: paddle.nn.Layer,
        dim_node_embedding: int, dim_edge_embedding: (int | None)=None,
        dim_state_feats: (int | None)=None, ntypes_node: (int | None)=None,
        include_state: bool=False, ntypes_state: (int | None)=None,
        dim_state_embedding: (int | None)=None):
        """
        Args:
            degree_rbf (int): number of rbf
            activation (nn.Module): activation type
            dim_node_embedding (int): dimensionality of node features
            dim_edge_embedding (int): dimensionality of edge features
            dim_state_feats: dimensionality of state features
            ntypes_node: number of node labels
            include_state: Whether to include state embedding
            ntypes_state: number of state labels
            dim_state_embedding: dimensionality of state embedding.
        """
        super().__init__()
        self.include_state = include_state
        self.ntypes_state = ntypes_state
        self.dim_node_embedding = dim_node_embedding
        self.dim_edge_embedding = dim_edge_embedding
        self.dim_state_feats = dim_state_feats
        self.ntypes_node = ntypes_node
        self.dim_state_embedding = dim_state_embedding
        self.activation = activation

        self.layer_node_embedding = paddle.nn.Embedding(num_embeddings=
            ntypes_node, embedding_dim=dim_node_embedding)

        if dim_edge_embedding is not None:
            dim_edges = [degree_rbf, dim_edge_embedding]
            self.layer_edge_embedding = MLP(dim_edges, activation=
                activation, activate_last=True)

    def forward(self, node_attr, edge_attr, state_attr):
        """Output embedded features.

        Args:
            node_attr: node attribute
            edge_attr: edge attribute
            state_attr: state attribute

        Returns:
            node_feat: embedded node features
            edge_feat: embedded edge features
            state_feat: embedded state features
        """
        if self.ntypes_node is not None:
            node_feat = self.layer_node_embedding(node_attr)
        else:
            node_feat = self.layer_node_embedding(node_attr.to('float32'))
        if self.dim_edge_embedding is not None:
            edge_feat = self.layer_edge_embedding(edge_attr.to('float32'))
        else:
            edge_feat = edge_attr
        if self.include_state is True:
            state_feat = state_attr
        else:
            state_feat = None
        return node_feat, edge_feat, state_feat




class MEGNetGraphConv(paddle.nn.Layer):
    """A MEGNet graph convolution layer in DGL."""

    def __init__(self, edge_func: paddle.nn.Layer, node_func: paddle.nn.
        Layer, state_func: paddle.nn.Layer) ->None:
        """
        Args:
            edge_func: Edge update function.
            node_func: Node update function.
            state_func: Global state update function.
        """
        super().__init__()
        self.edge_func = edge_func
        self.node_func = node_func
        self.state_func = state_func

    @staticmethod
    def from_dims(edge_dims: list[int], node_dims: list[int], state_dims:
        list[int], activation: paddle.nn.Layer) ->MEGNetGraphConv:
        """Create a MEGNet graph convolution layer from dimensions.

        Args:
            edge_dims (list[int]): Edge dimensions.
            node_dims (list[int]): Node dimensions.
            state_dims (list[int]): State dimensions.
            activation (Module): Activation function.

        Returns:
            MEGNetGraphConv: MEGNet graph convolution layer.
        """
        edge_update = MLP(edge_dims, activation, activate_last=True)
        node_update = MLP(node_dims, activation, activate_last=True)
        attr_update = MLP(state_dims, activation, activate_last=True)
        return MEGNetGraphConv(edge_update, node_update, attr_update)

    def _edge_udf(self, edges: dgl.udf.EdgeBatch):
        vi = edges.src['v']
        vj = edges.dst['v']
        u = edges.src['u']
        eij = edges.data.pop('e')
        inputs = paddle.hstack(x=[vi, vj, eij, u])
        mij = {'mij': self.edge_func(inputs)}
        return mij

    def edge_update_(self, graph: dgl.DGLGraph) ->paddle.Tensor:
        """Perform edge update.

        Args:
            graph: Input g

        Returns:
            Output tensor for edges.
        """
        graph.apply_edges(self._edge_udf)
        graph.edata['e'] = graph.edata.pop('mij')
        return graph.edata['e']

    def node_update_(self, graph: dgl.DGLGraph) ->paddle.Tensor:
        """Perform node update.

        Args:
            graph: Input g

        Returns:
            Output tensor for nodes.
        """
        graph.update_all(fn.copy_e('e', 'e'), fn.mean('e', 've'))
        ve = graph.ndata.pop('ve')
        v = graph.ndata.pop('v')
        u = graph.ndata.pop('u')
        inputs = paddle.hstack(x=[v, ve, u])
        graph.ndata['v'] = self.node_func(inputs)
        return graph.ndata['v']

    def state_update_(self, graph: dgl.DGLGraph, state_feat: paddle.Tensor
        ) ->paddle.Tensor:
        """Perform attribute (global state) update.

        Args:
            graph: Input g
            state_feat: Input attributes

        Returns:
            Output tensor for attributes
        """
        u_edge = dgl.readout_edges(graph, feat='e', op='mean')
        u_vertex = dgl.readout_nodes(graph, feat='v', op='mean')
        u_edge = paddle.squeeze(x=u_edge)
        u_vertex = paddle.squeeze(x=u_vertex)
        inputs = paddle.hstack(x=[state_feat.squeeze(), u_edge, u_vertex])
        state_feat = self.state_func(inputs)
        return state_feat

    def edge_update(self, graph, node_feat, edge_feat, state_feat):
        vi = node_feat[graph.edges[:, 0]]
        vj = node_feat[graph.edges[:, 1]]
        batch_num_nodes = graph._graph_node_index
        batch_num_nodes = batch_num_nodes[1:] - batch_num_nodes[:-1]
        u = paddle.repeat_interleave(state_feat, batch_num_nodes, axis=0)
        u = u[graph.edges[:, 0]]
        edge_feat = paddle.concat([vi, vj, edge_feat, u], axis=1)
        edge_feat = self.edge_func(edge_feat)
        return edge_feat

    def node_update(self, graph, node_feat, edge_feat, state_feat):

        batch_num_nodes = graph._graph_node_index
        batch_num_nodes = batch_num_nodes[1:] - batch_num_nodes[:-1]

        src, dst, eid = graph.sorted_edges(sort_by='dst')
        node_feat_e = paddle.geometric.segment_mean(edge_feat[eid], dst)

        u = paddle.repeat_interleave(state_feat, batch_num_nodes, axis=0)
        v = node_feat
        node_feat = paddle.concat([node_feat, node_feat_e, u], axis=1)
        node_feat = self.node_func(node_feat)
        return node_feat

    def state_update(self, graph, node_feat, edge_feat, state_feat):

        batch_num_edges = graph._graph_edge_index
        batch_num_edges = batch_num_edges[1:] - batch_num_edges[:-1]
        seg_ids = []
        for i in range(len(batch_num_edges)):
            num_edges = batch_num_edges[i]
            seg_ids.append(paddle.full(shape=[num_edges], fill_value=i, dtype='int32'))
        seg_ids = paddle.concat(seg_ids)
        u_edge_feat = paddle.geometric.segment_mean(edge_feat, seg_ids)
        
        batch_num_nodes = graph._graph_node_index
        batch_num_nodes = batch_num_nodes[1:] - batch_num_nodes[:-1]
        seg_ids = []
        for i in range(len(batch_num_nodes)):
            num_nodes = batch_num_nodes[i]
            seg_ids.append(paddle.full(shape=[num_nodes], fill_value=i, dtype='int32'))
        seg_ids = paddle.concat(seg_ids)
        u_node_feat = paddle.geometric.segment_mean(node_feat, seg_ids)
        state = paddle.concat([state_feat, u_edge_feat, u_node_feat], axis=1)
        state_feat = self.state_func(state)
        return state_feat

    def forward(self, graph: dgl.DGLGraph, edge_feat: paddle.Tensor,
        node_feat: paddle.Tensor, state_feat: paddle.Tensor) ->tuple[paddle
        .Tensor, paddle.Tensor, paddle.Tensor]:
        """Perform sequence of edge->node->attribute updates.

        Args:
            graph: Input g
            edge_feat: Edge features
            node_feat: Node features
            state_feat: Graph attributes (global state)

        Returns:
            (edge features, node features, graph attributes)
        """
        edge_feat = self.edge_update(graph, node_feat, edge_feat, state_feat)
        node_feat = self.node_update(graph, node_feat, edge_feat, state_feat)
        state_feat = self.state_update(graph, node_feat, edge_feat, state_feat)
        return edge_feat, node_feat, state_feat
        # with graph.local_scope():
        #     graph.edata['e'] = edge_feat
        #     graph.ndata['v'] = node_feat
        #     graph.ndata['u'] = dgl.broadcast_nodes(graph, state_feat)
        #     edge_feat = self.edge_update_(graph)
        #     node_feat = self.node_update_(graph)
        #     state_feat = self.state_update_(graph, state_feat)
        # return edge_feat, node_feat, state_feat


class MEGNetBlock(paddle.nn.Layer):
    """A MEGNet block comprising a sequence of update operations."""

    def __init__(self, dims: list[int], conv_hiddens: list[int], act:
        paddle.nn.Layer, dropout: (float | None)=None, skip: bool=True) ->None:
        """
        Init the MEGNet block with key parameters.

        Args:
            dims: Dimension of dense layers before graph convolution.
            conv_hiddens: Architecture of hidden layers of graph convolution.
            act: Activation type.
            dropout: Randomly zeroes some elements in the input tensor with given probability (0 < x < 1) according
                to a Bernoulli distribution.
            skip: Residual block.
        """
        super().__init__()
        self.has_dense = len(dims) > 1
        self.activation = act
        conv_dim = dims[-1]
        out_dim = conv_hiddens[-1]
        mlp_kwargs = {'dims': dims, 'activation': self.activation,
            'activate_last': True, 'bias_last': True}
        self.edge_func = MLP(**mlp_kwargs
            ) if self.has_dense else paddle.nn.Identity()
        self.node_func = MLP(**mlp_kwargs
            ) if self.has_dense else paddle.nn.Identity()
        self.state_func = MLP(**mlp_kwargs
            ) if self.has_dense else paddle.nn.Identity()
        edge_in = 2 * conv_dim + conv_dim + conv_dim
        node_in = out_dim + conv_dim + conv_dim
        attr_in = out_dim + out_dim + conv_dim
        self.conv = MEGNetGraphConv.from_dims(edge_dims=[edge_in, *
            conv_hiddens], node_dims=[node_in, *conv_hiddens], state_dims=[
            attr_in, *conv_hiddens], activation=self.activation)
        self.dropout = paddle.nn.Dropout(p=dropout) if dropout else None
        self.skip = skip

    def forward(self, graph: dgl.DGLGraph, edge_feat: paddle.Tensor,
        node_feat: paddle.Tensor, state_feat: paddle.Tensor) ->tuple[paddle
        .Tensor, paddle.Tensor, paddle.Tensor]:
        """MEGNetBlock forward pass.

        Args:
            graph (dgl.DGLGraph): A DGLGraph.
            edge_feat (Tensor): Edge features.
            node_feat (Tensor): Node features.
            state_feat (Tensor): Graph attributes (global state).

        Returns:
            tuple[Tensor, Tensor, Tensor]: Updated (edge features,
                node features, graph attributes)
        """
        inputs = edge_feat, node_feat, state_feat
        edge_feat = self.edge_func(edge_feat)
        node_feat = self.node_func(node_feat)
        state_feat = self.state_func(state_feat)
        edge_feat, node_feat, state_feat = self.conv(graph, edge_feat,
            node_feat, state_feat)
        if self.dropout:
            edge_feat = self.dropout(edge_feat)
            node_feat = self.dropout(node_feat)
            state_feat = self.dropout(state_feat)
        if self.skip:
            edge_feat = edge_feat + inputs[0]
            node_feat = node_feat + inputs[1]
            state_feat = state_feat + inputs[2]
        return edge_feat, node_feat, state_feat



def broadcast_nodes(graph, graph_feat, *, ntype=None):
    """Generate a node feature equal to the graph-level feature :attr:`graph_feat`.

    The operation is similar to ``numpy.repeat`` (or ``torch.repeat_interleave``).
    It is commonly used to normalize node features by a global vector. For example,
    to normalize node features across graph to range :math:`[0~1)`:

    >>> g = dgl.batch([...])  # batch multiple graphs
    >>> g.ndata['h'] = ...  # some node features
    >>> h_sum = dgl.broadcast_nodes(g, dgl.sum_nodes(g, 'h'))
    >>> g.ndata['h'] /= h_sum  # normalize by summation

    Parameters
    ----------
    graph : DGLGraph
        The graph.
    graph_feat : tensor
        The feature to broadcast. Tensor shape is :math:`(B, *)` for batched graph,
        where :math:`B` is the batch size.

    ntype : str, optional
        Node type. Can be omitted if there is only one node type.

    Returns
    -------
    Tensor
        The node features tensor with shape :math:`(N, *)`, where :math:`N` is the
        number of nodes.

    Examples
    --------

    >>> import dgl
    >>> import torch as th

    Create two :class:`~dgl.DGLGraph` objects and initialize their
    node features.

    >>> g1 = dgl.graph(([0], [1]))                    # Graph 1
    >>> g2 = dgl.graph(([0, 1], [1, 2]))              # Graph 2
    >>> bg = dgl.batch([g1, g2])
    >>> feat = th.rand(2, 5)
    >>> feat
    tensor([[0.4325, 0.7710, 0.5541, 0.0544, 0.9368],
            [0.2721, 0.4629, 0.7269, 0.0724, 0.1014]])

    Broadcast feature to all nodes in the batched graph, feat[i] is broadcast to nodes
    in the i-th example in the batch.

    >>> dgl.broadcast_nodes(bg, feat)
    tensor([[0.4325, 0.7710, 0.5541, 0.0544, 0.9368],
            [0.4325, 0.7710, 0.5541, 0.0544, 0.9368],
            [0.2721, 0.4629, 0.7269, 0.0724, 0.1014],
            [0.2721, 0.4629, 0.7269, 0.0724, 0.1014],
            [0.2721, 0.4629, 0.7269, 0.0724, 0.1014]])

    Broadcast feature to all nodes in the single graph (the feature tensor shape
    to broadcast should be :math:`(1, *)`).

    >>> feat0 = th.unsqueeze(feat[0], 0)
    >>> dgl.broadcast_nodes(g1, feat0)
    tensor([[0.4325, 0.7710, 0.5541, 0.0544, 0.9368],
            [0.4325, 0.7710, 0.5541, 0.0544, 0.9368]])

    See Also
    --------
    broadcast_edges
    """
    if tuple(graph_feat.shape)[0
        ] != graph.batch_size and graph.batch_size == 1:
        dgl_warning(
            'For a single graph, use a tensor of shape (1, *) for graph_feat. The support of shape (*) will be deprecated.'
            )
        graph_feat = paddle.unsqueeze(x=graph_feat, axis=0)
    return paddle.repeat_interleave(x=graph_feat, repeats=graph.
        batch_num_nodes(ntype), axis=0)


# class Set2Set(paddle.nn.Layer):
#     """Set2Set operator from `Order Matters: Sequence to sequence for sets
#     <https://arxiv.org/pdf/1511.06391.pdf>`__

#     For each individual graph in the batch, set2set computes

#     .. math::
#         q_t &= \\mathrm{LSTM} (q^*_{t-1})

#         \\alpha_{i,t} &= \\mathrm{softmax}(x_i \\cdot q_t)

#         r_t &= \\sum_{i=1}^N \\alpha_{i,t} x_i

#         q^*_t &= q_t \\Vert r_t

#     for this graph.

#     Parameters
#     ----------
#     input_dim : int
#         The size of each input sample.
#     n_iters : int
#         The number of iterations.
#     n_layers : int
#         The number of recurrent layers.

#     Examples
#     --------
#     The following example uses PyTorch backend.

#     >>> import dgl
#     >>> import torch as th
#     >>> from dgl.nn import Set2Set
#     >>>
#     >>> g1 = dgl.rand_graph(3, 4)  # g1 is a random graph with 3 nodes and 4 edges
#     >>> g1_node_feats = th.rand(3, 5)  # feature size is 5
#     >>> g1_node_feats
#     tensor([[0.8948, 0.0699, 0.9137, 0.7567, 0.3637],
#             [0.8137, 0.8938, 0.8377, 0.4249, 0.6118],
#             [0.5197, 0.9030, 0.6825, 0.5725, 0.4755]])
#     >>>
#     >>> g2 = dgl.rand_graph(4, 6)  # g2 is a random graph with 4 nodes and 6 edges
#     >>> g2_node_feats = th.rand(4, 5)  # feature size is 5
#     >>> g2_node_feats
#     tensor([[0.2053, 0.2426, 0.4111, 0.9028, 0.5658],
#             [0.5278, 0.6365, 0.9990, 0.2351, 0.8945],
#             [0.3134, 0.0580, 0.4349, 0.7949, 0.3891],
#             [0.0142, 0.2709, 0.3330, 0.8521, 0.6925]])
#     >>>
#     >>> s2s = Set2Set(5, 2, 1)  # create a Set2Set layer(n_iters=2, n_layers=1)

#     Case 1: Input a single graph

#     >>> s2s(g1, g1_node_feats)
#         tensor([[-0.0235, -0.2291,  0.2654,  0.0376,  0.1349,  0.7560,  0.5822,  0.8199,
#                   0.5960,  0.4760]], grad_fn=<CatBackward>)

#     Case 2: Input a batch of graphs

#     Build a batch of DGL graphs and concatenate all graphs' node features into one tensor.

#     >>> batch_g = dgl.batch([g1, g2])
#     >>> batch_f = th.cat([g1_node_feats, g2_node_feats], 0)
#     >>>
#     >>> s2s(batch_g, batch_f)
#     tensor([[-0.0235, -0.2291,  0.2654,  0.0376,  0.1349,  0.7560,  0.5822,  0.8199,
#               0.5960,  0.4760],
#             [-0.0483, -0.2010,  0.2324,  0.0145,  0.1361,  0.2703,  0.3078,  0.5529,
#               0.6876,  0.6399]], grad_fn=<CatBackward>)

#     Notes
#     -----
#     Set2Set is widely used in molecular property predictions, see
#     `dgl-lifesci's MPNN example <https://github.com/awslabs/dgl-lifesci/blob/
#     ecd95c905479ec048097777039cf9a19cfdcf223/python/dgllife/model/model_zoo/
#     mpnn_predictor.py>`__
#     on how to use DGL's Set2Set layer in graph property prediction applications.
#     """

#     def __init__(self, input_dim, n_iters, n_layers):
#         super(Set2Set, self).__init__()
#         self.input_dim = input_dim
#         self.output_dim = 2 * input_dim
#         self.n_iters = n_iters
#         self.n_layers = n_layers
#         self.lstm = paddle.nn.LSTM(input_size=self.output_dim, hidden_size=
#             self.input_dim, num_layers=n_layers, time_major=not False,
#             direction='forward')
#         self.reset_parameters()

#     def reset_parameters(self):
#         """Reinitialize learnable parameters."""
#         self.lstm.reset_parameters()

#     def forward(self, graph, feat):
#         """
#         Compute set2set pooling.

#         Parameters
#         ----------
#         graph : DGLGraph
#             The input graph.
#         feat : torch.Tensor
#             The input feature with shape :math:`(N, D)` where  :math:`N` is the
#             number of nodes in the graph, and :math:`D` means the size of features.

#         Returns
#         -------
#         torch.Tensor
#             The output feature with shape :math:`(B, D)`, where :math:`B` refers to
#             the batch size, and :math:`D` means the size of features.
#         """
#         with graph.local_scope():
#             batch_size = graph.batch_size
#             h = paddle.zeros(shape=(self.n_layers, batch_size, self.
#                 input_dim), dtype=feat.dtype), paddle.zeros(shape=(self.
#                 n_layers, batch_size, self.input_dim), dtype=feat.dtype)
#             q_star = paddle.zeros(shape=[batch_size, self.output_dim],
#                 dtype=feat.dtype)
#             for _ in range(self.n_iters):
#                 q, h = self.lstm(q_star.unsqueeze(axis=0), h)
#                 q = q.view(batch_size, self.input_dim)
#                 e = (feat * broadcast_nodes(graph, q)).sum(axis=-1, keepdim
#                     =True)
#                 graph.ndata['e'] = e
#                 alpha = softmax_nodes(graph, 'e')
#                 graph.ndata['r'] = feat * alpha
#                 readout = sum_nodes(graph, 'r')
#                 q_star = paddle.concat(x=[q, readout], axis=-1)
#             return q_star

#     def extra_repr(self):
#         """Set the extra representation of the module.
#         which will come into effect when printing the model.
#         """
#         summary = 'n_iters={n_iters}'
#         return summary.format(**self.__dict__)



class Set2Set(nn.Layer):
    """Implementation of Graph Global Pooling "Set2Set".
    
    Reference Paper: ORDER MATTERS: SEQUENCE TO SEQUENCE 

    Args:
        input_dim (int): dimentional size of input
        n_iters: number of iteration
        n_layers: number of LSTM layers
    Return:
        output_feat: output feature of set2set pooling with shape [batch, 2*dim].
    """

    def __init__(self, input_dim, n_iters, n_layers=1):
        super(Set2Set, self).__init__()
        self.input_dim = input_dim
        self.output_dim = 2 * input_dim
        self.n_iters = n_iters
        self.n_layers = n_layers
        self.lstm = paddle.nn.LSTM(
            input_size=self.output_dim,
            hidden_size=self.input_dim,
            num_layers=n_layers,
            time_major=True)

    def forward(self, graph, x):
        """Forward function of Graph Global Pooling "Set2Set".
        
        Args:
            graph: the graph object from (:code:`Graph`)
            x: A tensor with shape (num_nodes, feature_size).
        Return:
            output_feat: A tensor with shape (num_nodes, output_size).
        """
        graph_id = graph.graph_node_id
        batch_size = graph_id.max() + 1
        h = (
            paddle.zeros((self.n_layers, batch_size, self.input_dim)),
            paddle.zeros((self.n_layers, batch_size, self.input_dim)), )
        q_star = paddle.zeros((batch_size, self.output_dim))
        for _ in range(self.n_iters):
            q, h = self.lstm(q_star.unsqueeze(0), h)
            q = q.reshape((batch_size, self.input_dim))
            e = (x * q.index_select(
                graph_id, axis=0)).sum(axis=-1, keepdim=True)
            a = segment_softmax(e, graph_id)
            r = segment_sum(a * x, graph_id)
            q_star = paddle.concat([q, r], axis=-1)

        return q_star


class EdgeSet2Set(paddle.nn.Layer):
    """Implementation of Set2Set."""

    def __init__(self, input_dim: int, n_iters: int, n_layers: int) ->None:
        """:param input_dim: The size of each input sample.
        :param n_iters: The number of iterations.
        :param n_layers: The number of recurrent layers.
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = 2 * input_dim
        self.n_iters = n_iters
        self.n_layers = n_layers
        self.lstm = paddle.nn.LSTM(input_size=self.output_dim, hidden_size=
            self.input_dim, num_layers=n_layers, time_major=not False,
            direction='forward')
    #     self.reset_parameters()

    # def reset_parameters(self):
    #     """Reinitialize learnable parameters."""
    #     self.lstm.reset_parameters()

    def forward(self, g: DGLGraph, feat: paddle.Tensor):
        """Defines the computation performed at every call.

        :param g: Input graph
        :param feat: Input features.
        :return: One hot vector
        """
        with g.local_scope():
            batch_size = g.batch_size
            h = paddle.zeros(shape=(self.n_layers, batch_size, self.
                input_dim), dtype=feat.dtype), paddle.zeros(shape=(self.
                n_layers, batch_size, self.input_dim), dtype=feat.dtype)
            q_star = paddle.zeros(shape=[batch_size, self.output_dim],
                dtype=feat.dtype)
            for _ in range(self.n_iters):
                q, h = self.lstm(q_star.unsqueeze(axis=0), h)
                q = q.view(batch_size, self.input_dim)
                e = (feat * broadcast_edges(g, q)).sum(dim=-1, keepdim=True)
                g.edata['e'] = e
                alpha = softmax_edges(g, 'e')
                g.edata['r'] = feat * alpha
                readout = sum_edges(g, 'r')
                q_star = paddle.concat(x=[q, readout], axis=-1)
            return q_star

    def forward(self, graph, x):
        """Forward function of Graph Global Pooling "Set2Set".
        
        Args:
            graph: the graph object from (:code:`Graph`)
            x: A tensor with shape (num_nodes, feature_size).
        Return:
            output_feat: A tensor with shape (num_nodes, output_size).
        """
        graph_id = graph.graph_edge_id
        batch_size = graph_id.max() + 1
        h = (
            paddle.zeros((self.n_layers, batch_size, self.input_dim)),
            paddle.zeros((self.n_layers, batch_size, self.input_dim)), )
        q_star = paddle.zeros((batch_size, self.output_dim))
        for _ in range(self.n_iters):
            q, h = self.lstm(q_star.unsqueeze(0), h)
            q = q.reshape((batch_size, self.input_dim))
            e = (x * q.index_select(
                graph_id, axis=0)).sum(axis=-1, keepdim=True)
            a = segment_softmax(e, graph_id)
            r = segment_sum(a * x, graph_id)
            q_star = paddle.concat([q, r], axis=-1)

        return q_star