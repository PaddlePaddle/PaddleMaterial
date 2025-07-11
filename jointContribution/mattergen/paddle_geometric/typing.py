import inspect
import os
import sys
import typing
import warnings
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import paddle
from paddle import Tensor

WITH_PADDLE_DEV = True


WITH_WINDOWS = os.name == 'nt'

MAX_INT64 = paddle.iinfo(paddle.int64).max

INDEX_DTYPES: Set[paddle.dtype] = {
    paddle.int32,
    paddle.int64,
}


try:
    import pyg_lib  # noqa
    WITH_PYG_LIB = True
    WITH_GMM = hasattr(pyg_lib.ops, 'grouped_matmul')
    WITH_SEGMM = hasattr(pyg_lib.ops, 'segment_matmul')
    if WITH_SEGMM and 'pytest' in sys.modules and paddle.device.is_compiled_with_cuda():
        try:
            x = paddle.randn([3, 4], place='gpu')
            ptr = paddle.to_tensor([0, 2, 3], place='gpu')
            weight = paddle.randn([2, 4, 4], place='gpu')
            out = pyg_lib.ops.segment_matmul(x, ptr, weight)
        except RuntimeError:
            WITH_GMM = False
            WITH_SEGMM = False
    WITH_SAMPLED_OP = hasattr(pyg_lib.ops, 'sampled_add')
    WITH_SOFTMAX = hasattr(pyg_lib.ops, 'softmax_csr')
    WITH_INDEX_SORT = hasattr(pyg_lib.ops, 'index_sort')
    WITH_METIS = hasattr(pyg_lib, 'partition')
    WITH_EDGE_TIME_NEIGHBOR_SAMPLE = ('edge_time' in inspect.signature(
        pyg_lib.sampler.neighbor_sample).parameters)
    WITH_WEIGHTED_NEIGHBOR_SAMPLE = ('edge_weight' in inspect.signature(
        pyg_lib.sampler.neighbor_sample).parameters)
except Exception as e:
    if not isinstance(e, ImportError):  # pragma: no cover
        warnings.warn(f"An issue occurred while importing 'pyg-lib'. "
                      f"Disabling its usage. Stacktrace: {e}")
    pyg_lib = object
    WITH_PYG_LIB = False
    WITH_GMM = False
    WITH_SEGMM = False
    WITH_SAMPLED_OP = False
    WITH_SOFTMAX = False
    WITH_INDEX_SORT = False
    WITH_METIS = False
    WITH_EDGE_TIME_NEIGHBOR_SAMPLE = False
    WITH_WEIGHTED_NEIGHBOR_SAMPLE = False

try:
    import paddle_scatter  # noqa
    WITH_PADDLE_SCATTER = True
except Exception as e:
    if not isinstance(e, ImportError):  # pragma: no cover
        warnings.warn(f"An issue occurred while importing 'paddle-scatter'. "
                      f"Disabling its usage. Stacktrace: {e}")
    paddle_scatter = object
    WITH_PADDLE_SCATTER = False

try:
    import paddle_cluster  # noqa
    WITH_PADDLE_CLUSTER = True
    WITH_PADDLE_CLUSTER_BATCH_SIZE = 'batch_size' in paddle_cluster.knn.__doc__
except Exception as e:
    if not isinstance(e, ImportError):  # pragma: no cover
        warnings.warn(f"An issue occurred while importing 'paddle-cluster'. "
                      f"Disabling its usage. Stacktrace: {e}")
    WITH_PADDLE_CLUSTER = False
    WITH_PADDLE_CLUSTER_BATCH_SIZE = False

    class PaddleCluster:
        def __getattr__(self, key: str) -> Any:
            raise ImportError(f"'{key}' requires 'paddle-cluster'")

    paddle_cluster = PaddleCluster()

try:
    import paddle_sparse  # noqa
    from paddle_sparse import SparseStorage, SparseTensor
    WITH_PADDLE_SPARSE = True
except Exception as e:
    if not isinstance(e, ImportError):  # pragma: no cover
        warnings.warn(f"An issue occurred while importing 'paddle-sparse'. "
                      f"Disabling its usage. Stacktrace: {e}")
    WITH_PADDLE_SPARSE = False

    class SparseStorage:  # type: ignore
        def __init__(
            self,
            row: Optional[Tensor] = None,
            rowptr: Optional[Tensor] = None,
            col: Optional[Tensor] = None,
            value: Optional[Tensor] = None,
            sparse_sizes: Optional[Tuple[Optional[int], Optional[int]]] = None,
            rowcount: Optional[Tensor] = None,
            colptr: Optional[Tensor] = None,
            colcount: Optional[Tensor] = None,
            csr2csc: Optional[Tensor] = None,
            csc2csr: Optional[Tensor] = None,
            is_sorted: bool = False,
            trust_data: bool = False,
        ):
            raise ImportError("'SparseStorage' requires 'paddle-sparse'")

        def value(self) -> Optional[Tensor]:
            raise ImportError("'SparseStorage' requires 'paddle-sparse'")

        def rowcount(self) -> Tensor:
            raise ImportError("'SparseStorage' requires 'paddle-sparse'")

    class SparseTensor:  # type: ignore
        def __init__(
            self,
            row: Optional[Tensor] = None,
            rowptr: Optional[Tensor] = None,
            col: Optional[Tensor] = None,
            value: Optional[Tensor] = None,
            sparse_sizes: Optional[Tuple[Optional[int], Optional[int]]] = None,
            is_sorted: bool = False,
            trust_data: bool = False,
        ):
            raise ImportError("'SparseTensor' requires 'paddle-sparse'")

        @classmethod
        def from_edge_index(
            self,
            edge_index: Tensor,
            edge_attr: Optional[Tensor] = None,
            sparse_sizes: Optional[Tuple[Optional[int], Optional[int]]] = None,
            is_sorted: bool = False,
            trust_data: bool = False,
        ) -> 'SparseTensor':
            raise ImportError("'SparseTensor' requires 'paddle-sparse'")

        @property
        def storage(self) -> SparseStorage:
            raise ImportError("'SparseTensor' requires 'paddle-sparse'")

        @classmethod
        def from_dense(self, mat: Tensor,
                       has_value: bool = True) -> 'SparseTensor':
            raise ImportError("'SparseTensor' requires 'paddle-sparse'")

        def size(self, dim: int) -> int:
            raise ImportError("'SparseTensor' requires 'paddle-sparse'")

        def nnz(self) -> int:
            raise ImportError("'SparseTensor' requires 'paddle-sparse'")

        def is_cuda(self) -> bool:
            raise ImportError("'SparseTensor' requires 'paddle-sparse'")

        def has_value(self) -> bool:
            raise ImportError("'SparseTensor' requires 'paddle-sparse'")

        def set_value(self, value: Optional[Tensor],
                      layout: Optional[str] = None) -> 'SparseTensor':
            raise ImportError("'SparseTensor' requires 'paddle-sparse'")

        def fill_value(self, fill_value: float,
                       dtype: Optional[paddle.dtype] = None) -> 'SparseTensor':
            raise ImportError("'SparseTensor' requires 'paddle-sparse'")

        def coo(self) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
            raise ImportError("'SparseTensor' requires 'paddle-sparse'")

        def csr(self) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
            raise ImportError("'SparseTensor' requires 'paddle-sparse'")

        def requires_grad(self) -> bool:
            raise ImportError("'SparseTensor' requires 'paddle-sparse'")

        def to_paddle_sparse_csr_tensor(
            self,
            dtype: Optional[paddle.dtype] = None,
        ) -> Tensor:
            raise ImportError("'SparseTensor' requires 'paddle-sparse'")

    class paddle_sparse:  # type: ignore
        @staticmethod
        def matmul(src: SparseTensor, other: Tensor,
                   reduce: str = "sum") -> Tensor:
            raise ImportError("'matmul' requires 'paddle-sparse'")

        @staticmethod
        def sum(src: SparseTensor, dim: Optional[int] = None) -> Tensor:
            raise ImportError("'sum' requires 'paddle-sparse'")

        @staticmethod
        def mul(src: SparseTensor, other: Tensor) -> SparseTensor:
            raise ImportError("'mul' requires 'paddle-sparse'")

        @staticmethod
        def set_diag(src: SparseTensor, values: Optional[Tensor] = None,
                     k: int = 0) -> SparseTensor:
            raise ImportError("'set_diag' requires 'paddle-sparse'")

        @staticmethod
        def fill_diag(src: SparseTensor, fill_value: float,
                      k: int = 0) -> SparseTensor:
            raise ImportError("'fill_diag' requires 'paddle-sparse'")

        @staticmethod
        def masked_select_nnz(src: SparseTensor, mask: Tensor,
                              layout: Optional[str] = None) -> SparseTensor:
            raise ImportError("'masked_select_nnz' requires 'paddle-sparse'")

try:
    import paddle_frame  # noqa
    WITH_PADDLE_FRAME = True
    from paddle_frame import TensorFrame
except Exception:
    paddle_frame = object
    WITH_PADDLE_FRAME = False

    class TensorFrame:  # type: ignore
        pass

class MockPaddleCSCTensor:
    def __init__(
        self,
        edge_index: Tensor,
        edge_attr: Optional[Tensor] = None,
        size: Optional[Union[int, Tuple[int, int]]] = None,
    ):
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.size = size

    def t(self) -> Tensor:
        from paddle_geometric.utils import to_paddle_csr_tensor
        size = self.size
        return to_paddle_csr_tensor(
            self.edge_index.flip([0]),
            self.edge_attr,
            size[::-1] if isinstance(size, (tuple, list)) else size,
        )


# Types for accessing data ####################################################

# Node-types are denoted by a single string, e.g.: `data['paper']`:
NodeType = str

# Edge-types are denotes by a triplet of strings, e.g.:
# `data[('author', 'writes', 'paper')]
EdgeType = Tuple[str, str, str]

NodeOrEdgeType = Union[NodeType, EdgeType]

DEFAULT_REL = 'to'
EDGE_TYPE_STR_SPLIT = '__'


class EdgeTypeStr(str):
    r"""A helper class to construct serializable edge types by merging an edge
    type tuple into a single string.
    """
    def __new__(cls, *args: Any) -> 'EdgeTypeStr':
        if isinstance(args[0], (list, tuple)):
            # Unwrap `EdgeType((src, rel, dst))` and `EdgeTypeStr((src, dst))`:
            args = tuple(args[0])

        if len(args) == 1 and isinstance(args[0], str):
            arg = args[0]  # An edge type string was passed.

        elif len(args) == 2 and all(isinstance(arg, str) for arg in args):
            # A `(src, dst)` edge type was passed - add `DEFAULT_REL`:
            arg = EDGE_TYPE_STR_SPLIT.join((args[0], DEFAULT_REL, args[1]))

        elif len(args) == 3 and all(isinstance(arg, str) for arg in args):
            # A `(src, rel, dst)` edge type was passed:
            arg = EDGE_TYPE_STR_SPLIT.join(args)

        else:
            raise ValueError(f"Encountered invalid edge type '{args}'")

        return str.__new__(cls, arg)

    def to_tuple(self) -> EdgeType:
        r"""Returns the original edge type."""
        out = tuple(self.split(EDGE_TYPE_STR_SPLIT))
        if len(out) != 3:
            raise ValueError(f"Cannot convert the edge type '{self}' to a "
                             f"tuple since it holds invalid characters")
        return out


# There exist some short-cuts to query edge-types (given that the full triplet
# can be uniquely reconstructed, e.g.:
# * via str: `data['writes']`
# * via Tuple[str, str]: `data[('author', 'paper')]`
QueryType = Union[NodeType, EdgeType, str, Tuple[str, str]]

Metadata = Tuple[List[NodeType], List[EdgeType]]

# A representation of a feature tensor
FeatureTensorType = Union[Tensor, np.ndarray]

# A representation of an edge index, following the possible formats:
#   * COO: (row, col)
#   * CSC: (row, colptr)
#   * CSR: (rowptr, col)
EdgeTensorType = Tuple[Tensor, Tensor]

# Types for message passing ###################################################

Adj = Union[Tensor, SparseTensor]
OptTensor = Optional[Tensor]
PairTensor = Tuple[Tensor, Tensor]
OptPairTensor = Tuple[Tensor, Optional[Tensor]]
PairOptTensor = Tuple[Optional[Tensor], Optional[Tensor]]
Size = Optional[Tuple[int, int]]
NoneType = Optional[Tensor]

MaybeHeteroNodeTensor = Union[Tensor, Dict[NodeType, Tensor]]
MaybeHeteroAdjTensor = Union[Tensor, Dict[EdgeType, Adj]]
MaybeHeteroEdgeTensor = Union[Tensor, Dict[EdgeType, Tensor]]

# Types for sampling ##########################################################

InputNodes = Union[OptTensor, NodeType, Tuple[NodeType, OptTensor]]
InputEdges = Union[OptTensor, EdgeType, Tuple[EdgeType, OptTensor]]

# Serialization ###############################################################

if hasattr(paddle, "serialization"):
    paddle.serialization.add_safe_globals([
        SparseTensor,
        SparseStorage,
        TensorFrame,
        MockPaddleCSCTensor,
        EdgeTypeStr,
    ])
