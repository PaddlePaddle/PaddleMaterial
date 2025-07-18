import copy
import warnings
from typing import Dict, List, Optional, Union

import paddle
import paddle.nn.functional as F
from paddle import Tensor
from paddle.nn import Layer

import paddle_geometric
from paddle_geometric import is_compiling
from paddle_geometric.typing import EdgeType, NodeType, OptTensor
from paddle_geometric.utils import cumsum, scatter

class ToHeteroLinear(Layer):
    def __init__(
        self,
        module: Layer,
        types: Union[List[NodeType], List[EdgeType]],
    ):
        from paddle_geometric.nn import HeteroLinear, Linear

        super().__init__()

        self.types = types

        if isinstance(module, Linear):
            in_channels = module.in_channels
            out_channels = module.out_channels
            bias = module.bias is not None

        elif isinstance(module, paddle.nn.Linear):
            in_channels = module.in_features
            out_channels = module.out_features
            bias = module.bias is not None

        else:
            raise ValueError(f"Expected 'Linear' module (got '{type(module)}'")

        self.hetero_module = HeteroLinear(
            in_channels,
            out_channels,
            num_types=len(types),
            is_sorted=True,
            bias=bias,
        )

    def fused_forward(self, x: Tensor, type_vec: Tensor) -> Tensor:
        return self.hetero_module(x, type_vec)

    def dict_forward(
        self,
        x_dict: Dict[Union[NodeType, EdgeType], Tensor],
    ) -> Dict[Union[NodeType, EdgeType], Tensor]:
        if not paddle_geometric.typing.WITH_PYG_LIB or is_compiling():
            return {
                key:
                F.linear(x_dict[key], self.hetero_module.weight[i].T()) +
                self.hetero_module.bias[i]
                for i, key in enumerate(self.types)
            }

        x = paddle.concat([x_dict[key] for key in self.types], axis=0)
        sizes = [x_dict[key].shape[0] for key in self.types]
        type_vec = paddle.arange(len(self.types), device=x.device)
        size = paddle.to_tensor(sizes, device=x.device)
        type_vec = type_vec.tile([size])
        outs = self.hetero_module(x, type_vec).split(sizes)
        return {key: out for key, out in zip(self.types, outs)}

    def forward(
        self,
        x: Union[Tensor, Dict[Union[NodeType, EdgeType], Tensor]],
        type_vec: Optional[Tensor] = None,
    ) -> Union[Tensor, Dict[Union[NodeType, EdgeType], Tensor]]:
        if isinstance(x, dict):
            return self.dict_forward(x)

        elif isinstance(x, Tensor) and type_vec is not None:
            return self.fused_forward(x, type_vec)

        raise ValueError(f"Encountered invalid forward types in "
                         f"'{self.__class__.__name__}'")


class ToHeteroMessagePassing(Layer):
    def __init__(
        self,
        module: Layer,
        node_types: List[NodeType],
        edge_types: List[NodeType],
        aggr: str = 'sum',
    ):
        from paddle_geometric.nn import HeteroConv, MessagePassing

        super().__init__()

        self.node_types = node_types
        self.node_type_to_index = {key: i for i, key in enumerate(node_types)}
        self.edge_types = edge_types

        if not isinstance(module, MessagePassing):
            raise ValueError(f"Expected 'MessagePassing' module "
                             f"(got '{type(module)}'")

        if (not hasattr(module, 'reset_parameters')
                and sum([p.numel() for p in module.parameters()]) > 0):
            warnings.warn(f"'{module}' will be duplicated, but its parameters "
                          f"cannot be reset. To suppress this warning, add a "
                          f"'reset_parameters()' method to '{module}'")

        convs = {edge_type: copy.deepcopy(module) for edge_type in edge_types}
        self.hetero_module = HeteroConv(convs, aggr)
        self.hetero_module.reset_parameters()

    def fused_forward(self, x: Tensor, edge_index: Tensor, node_type: Tensor,
                      edge_type: Tensor) -> Tensor:
        node_sizes = scatter(paddle.ones_like(node_type), node_type, dim=0,
                             dim_size=len(self.node_types), reduce='sum')
        edge_sizes = scatter(paddle.ones_like(edge_type), edge_type, dim=0,
                             dim_size=len(self.edge_types), reduce='sum')

        ptr = cumsum(node_sizes)

        xs = x.split(node_sizes.tolist())
        x_dict = {node_type: x for node_type, x in zip(self.node_types, xs)}

        edge_indices = edge_index.clone().split(edge_sizes.tolist(), axis=1)
        for (src, _, dst), index in zip(self.edge_types, edge_indices):
            index[0] -= ptr[self.node_type_to_index[src]]
            index[1] -= ptr[self.node_type_to_index[dst]]

        edge_index_dict = {
            edge_type: edge_index
            for edge_type, edge_index in zip(self.edge_types, edge_indices)
        }

        out_dict = self.hetero_module(x_dict, edge_index_dict)
        return paddle.concat([out_dict[key] for key in self.node_types], axis=0)

    def dict_forward(
        self,
        x_dict: Dict[NodeType, Tensor],
        edge_index_dict: Dict[EdgeType, Tensor],
        **kwargs,
    ) -> Dict[NodeType, Tensor]:
        return self.hetero_module(x_dict, edge_index_dict, **kwargs)

    def forward(
        self,
        x: Union[Tensor, Dict[NodeType, Tensor]],
        edge_index: Union[Tensor, Dict[EdgeType, Tensor]],
        node_type: OptTensor = None,
        edge_type: OptTensor = None,
        **kwargs,
    ) -> Union[Tensor, Dict[NodeType, Tensor]]:
        if isinstance(x, dict) and isinstance(edge_index, dict):
            return self.dict_forward(x, edge_index, **kwargs)

        elif (isinstance(x, Tensor) and isinstance(edge_index, Tensor)
              and node_type is not None and edge_type is not None):

            if len(kwargs) > 0:
                raise ValueError("Additional forward arguments not yet "
                                 "supported in fused mode")

            return self.fused_forward(x, edge_index, node_type, edge_type)

        raise ValueError(f"Encountered invalid forward types in "
                         f"'{self.__class__.__name__}'")
