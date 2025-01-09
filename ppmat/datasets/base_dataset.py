import warnings
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import paddle
import pandas as pd


class GraphData:
    """
    Basic Graph Data
    """

    def __init__(
        self,
        x: Optional[paddle.Tensor] = None,
        edge_index: Optional[paddle.Tensor] = None,
        edge_attr: Optional[paddle.Tensor] = None,
        y: Optional[Union[paddle.Tensor, int, float]] = None,
        **kwargs,
    ):
        if x is not None:
            self.x = x
        if edge_index is not None:
            self.edge_index = edge_index
        if edge_attr is not None:
            self.edge_attr = edge_attr
        if y is not None:
            self.y = y
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __str__(self):
        attributes = []
        for key, value in self.__dict__.items():
            if isinstance(value, int) or value.shape == []:
                shape = [1]
            else:
                shape = value.shape
            attributes.append(f"{key}={shape}")
        return f"{self.__class__.__name__}({', '.join(attributes)})"


def subgraph(
    subset: Union[paddle.Tensor, List[int]],
    edge_index: paddle.Tensor,
    edge_attr: paddle.Tensor = None,
    relabel_nodes: bool = False,
    num_nodes: Optional[int] = None,
    *,
    return_edge_mask: bool = False,
) -> Union[Tuple[paddle.Tensor], Tuple[paddle.Tensor]]:
    r"""Returns the induced subgraph of :obj:`(edge_index, edge_attr)`
    containing the nodes in :obj:`subset`.
    """

    if isinstance(subset, (list, tuple)):
        subset = paddle.to_tensor(subset, dtype=paddle.int64)

    assert subset.dtype == paddle.bool, "subset.dtype should be paddle.bool now."

    num_nodes = subset.shape[0]
    node_mask = subset
    node_mask_int = node_mask.astype("int64")
    subset = paddle.nonzero(node_mask_int).reshape([-1])
    edge_mask = node_mask_int[edge_index[0]] & node_mask_int[edge_index[1]]
    edge_index = paddle.gather(
        edge_index, paddle.nonzero(edge_mask).reshape([-1]), axis=1
    )
    edge_attr = (
        paddle.gather(edge_attr, paddle.nonzero(edge_mask).reshape([-1]), axis=0)
        if edge_attr is not None
        else None
    )

    if relabel_nodes:
        edge_index_mapped, _ = map_index(
            src=edge_index.reshape([-1]),
            index=subset,
            max_index=num_nodes,
            inclusive=True,
        )
        edge_index = edge_index_mapped.reshape([2, -1])

    if return_edge_mask:
        return edge_index, edge_attr, edge_mask
    else:
        return edge_index, edge_attr


def map_index(
    src: paddle.Tensor,
    index: paddle.Tensor,
    max_index: Optional[Union[int, paddle.Tensor]] = None,
    inclusive: bool = False,
) -> Tuple[paddle.Tensor, Optional[paddle.Tensor]]:
    if src.dtype in [paddle.float32, paddle.float64]:
        raise ValueError(f"Expected 'src' to be an index (got '{src.dtype}')")
    if index.dtype in [paddle.float32, paddle.float64]:
        raise ValueError(f"Expected 'index' to be an index (got '{index.dtype}')")
    if str(src.place) != str(index.place):
        raise ValueError(
            "Both 'src' and 'index' must be on the same device "
            f"(got '{src.place}' and '{index.place}')"
        )

    if max_index is None:
        max_index = paddle.maximum(src.max(), index.max()).item()

    # Thresholds may need to be adjusted based on memory constraints
    THRESHOLD = 40_000_000 if src.place.is_gpu_place() else 10_000_000
    if max_index <= THRESHOLD:
        if inclusive:
            assoc = paddle.empty((max_index + 1,), dtype=src.dtype)
        else:
            assoc = paddle.full((max_index + 1,), -1, dtype=src.dtype)
        assoc = assoc.scatter(index, paddle.arange(index.numel(), dtype=src.dtype))
        out = assoc.gather(src)

        if inclusive:
            return out, None
        else:
            mask = out != -1
            return out[mask], mask

    WITH_CUDF = False
    if src.place.is_gpu_place():
        try:
            import cudf

            WITH_CUDF = True
        except ImportError:
            warnings.warn(
                "Using CPU-based processing within 'map_index' which may "
                "cause slowdowns and device synchronization. "
                "Consider installing 'cudf' to accelerate computation"
            )

    if not WITH_CUDF:
        src_np = src.cpu().numpy()
        index_np = index.cpu().numpy()
        left_ser = pd.Series(src_np, name="left_ser")
        right_ser = pd.Series(
            index=index_np, data=np.arange(0, len(index_np)), name="right_ser"
        )

        result = pd.merge(
            left_ser, right_ser, how="left", left_on="left_ser", right_index=True
        )
        out_numpy = result["right_ser"].values

        out = paddle.to_tensor(out_numpy, place=src.place)

        if inclusive:
            return out, None
        else:
            mask = out != -1
            return out[mask], mask

    else:
        left_ser = cudf.Series(src.numpy(), name="left_ser")
        right_ser = cudf.Series(
            index=index.numpy(),
            data=cudf.RangeIndex(0, len(index.numpy())),
            name="right_ser",
        )

        result = cudf.merge(
            left_ser,
            right_ser,
            how="left",
            left_on="left_ser",
            right_index=True,
            sort=True,
        )

        if inclusive:
            out = paddle.to_tensor(result["right_ser"].to_numpy(), dtype=src.dtype)
        else:
            out = paddle.to_tensor(
                result["right_ser"].fillna(-1).to_numpy(), dtype=src.dtype
            )

        out = out[src.argsort().argsort()]  # Restore original order.

        if inclusive:
            return out, None
        else:
            mask = out != -1
            return out[mask], mask


# def BaseDataset:
#     def __init__(self, cfg, datasets):
