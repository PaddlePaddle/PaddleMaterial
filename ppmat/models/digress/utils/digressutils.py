import warnings
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import paddle

# import paddle.sparse as sparse
import pandas as pd
from pgl.math import segment_sum


class PlaceHolder:
    """
    辅助类，用于封装 (X, E, y) 并提供 type_as / mask 等方法。
    """

    def __init__(self, X, E, y):
        self.X = X
        self.E = E
        self.y = y

    def type_as(self, x: paddle.Tensor):
        self.X = self.X.astype(x.dtype)
        self.E = self.E.astype(x.dtype)
        self.y = self.y.astype(x.dtype)
        return self

    def mask(self, node_mask, collapse=False):
        # x_mask = node_mask.unsqueeze(-1)
        x_mask = paddle.unsqueeze(node_mask, axis=-1).astype(self.X.dtype)  # (bs, n, 1)
        e_mask1 = paddle.unsqueeze(x_mask, axis=2)  # (bs, n, 1, 1)
        e_mask2 = paddle.unsqueeze(x_mask, axis=1)  # (bs, 1, n, 1)

        if collapse:
            # self.X = torch.argmax(self.X, dim=-1)
            self.X = paddle.argmax(self.X, axis=-1)  # (bs,n)
            self.E = paddle.argmax(self.E, axis=-1)  # (bs,n,n)

            # self.X[node_mask == 0] = -1
            zero_mask = node_mask == 0
            self.X = paddle.where(
                zero_mask, paddle.full_like(self.X, fill_value=-1), self.X
            )

            # e_mask => (bs,n,n) shape (由 e_mask1 * e_mask2 => (bs,n,n,1)?)
            e_mask = paddle.squeeze(e_mask1 * e_mask2, axis=-1)  # (bs,n,n)
            self.E = paddle.where(
                e_mask == 0, paddle.full_like(self.E, fill_value=-1), self.E
            )
        else:
            # self.X = self.X * x_mask
            self.X = self.X * x_mask
            # self.E = self.E * e_mask1 * e_mask2
            self.E = self.E * e_mask1 * e_mask2

            if not paddle.allclose(self.E, paddle.transpose(self.E, perm=[0, 2, 1, 3])):
                raise ValueError("E is not symmetric after masking.")
        return self


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


def to_dense(x, edge_index, edge_attr, batch):
    """
    将稀疏图数据转换为密集格式 (PaddlePaddle 版)
    Args:
        x (paddle.Tensor): 节点特征矩阵 (N, F)
        edge_index (paddle.Tensor): 边索引矩阵 (2, E)
        edge_attr (paddle.Tensor): 边属性矩阵 (E, D)
        batch (paddle.Tensor): 节点批次索引 (N,)
    Returns:
        PlaceHolder: 包含密集化后的节点特征矩阵和邻接矩阵
    """
    X, node_mask = to_dense_batch(x=x, batch=batch)
    # 移除自环
    edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)

    # # 将边索引转换为稀疏矩阵形式
    # values = edge_attr
    # shape = [num_nodes, num_nodes, edge_attr.shape[1]]
    # # 构建稀疏邻接矩阵
    # indices = edge_index
    # E = sparse.sparse_coo_tensor(indices, values, shape)
    # E = sparse.to_dense(E)

    max_num_nodes = X.shape[1]
    E = to_dense_adj(
        edge_index=edge_index,
        batch=batch,
        edge_attr=edge_attr,
        max_num_nodes=max_num_nodes,
    )
    E = encode_no_edge(E)

    return PlaceHolder(X=X, E=E, y=None), node_mask


def to_dense_batch(x, batch, fill_value=0, max_num_nodes=None, batch_size=None):
    """Transfrom a batch of graphs to a dense node feature tensor and
       provide the mask  holing the positions of dummy nodes

    Args:
        x (paddle.tensor): The feature map of nodes
        batch (pgl.Graph): The graph holing the graph node id
        fill_value (bool): The value of dummy nodes. Default: 0.
        max_node_nodes: The dimension of nodes in dense batch. Default: None.
        batch_size (int, optional): The batch size. Default: None.

    Returns:

        out (paddle.tensor): Returns a dense node feature tensor
            (shape = [batch_size,max_num_nodes,-1])
        mask (paddle.tensor): Return a mask indicating the position of
            dummy nodes (shape = [batch_size, max_num_nodes])

    """
    if batch is None and max_num_nodes is None:
        mask = paddle.ones(shape=[1, x.shape[0]], dtype="bool")
        return paddle.unsqueeze(x, axis=0), mask

    if batch is None:
        batch = paddle.zeros(shape=[x.shape[0]], dtype="int64")

    if batch_size is None:
        batch_size = (batch.max().item()) + 1

    num_nodes = segment_sum(paddle.ones([x.shape[0]]), batch)
    cum_nodes = paddle.concat([paddle.zeros([1]), num_nodes.cumsum(0)]).astype(
        batch.dtype
    )

    if max_num_nodes is None:
        max_num_nodes = int(num_nodes.max())

    idx = paddle.arange(batch.shape[0], dtype=batch.dtype)
    idx = (idx - cum_nodes[batch]) + (batch * max_num_nodes)

    size = [batch_size * max_num_nodes] + list(x.shape)[1:]
    out = paddle.full(size, fill_value).astype(x.dtype)
    out = paddle.scatter(out, idx, x)
    out = out.reshape([batch_size, max_num_nodes] + list(x.shape)[1:])

    mask = paddle.zeros(batch_size * max_num_nodes, dtype=paddle.bool)
    mask[idx] = 1
    mask = mask.reshape([batch_size, max_num_nodes])

    return out, mask


def remove_self_loops(edge_index, edge_attr=None):
    mask = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, mask]

    if edge_attr is not None:
        edge_attr = edge_attr[mask]

    return edge_index, edge_attr


def to_dense_adj(
    edge_index,
    batch=None,
    edge_attr=None,
    max_num_nodes=None,
    batch_size=None,
):
    if batch is None:
        max_index = int(edge_index.max()) + 1 if edge_index.numel() > 0 else 0
        batch = paddle.zeros(shape=[max_index], dtype="int64")

    if batch_size is None:
        batch_size = int(batch.max()) + 1 if batch.numel() > 0 else 1

    one = paddle.ones_like(batch, dtype=paddle.float32)
    num_nodes = segment_sum(one, batch)
    cum_nodes = paddle.concat([paddle.zeros([1]), num_nodes.cumsum(0)]).astype(
        edge_index.dtype
    )

    idx0 = batch[edge_index[0]].astype(edge_index.dtype)
    idx1 = edge_index[0] - cum_nodes[batch][edge_index[0]]
    idx2 = edge_index[1] - cum_nodes[batch][edge_index[1]]

    if max_num_nodes is None:
        max_num_nodes = int(num_nodes.max())
    elif (idx1.numel() > 0 and idx1.max() >= max_num_nodes) or (
        idx2.numel() > 0 and idx2.max() >= max_num_nodes
    ):
        mask = (idx1 < max_num_nodes) & (idx2 < max_num_nodes)
        idx0 = idx0[mask]
        idx1 = idx1[mask]
        idx2 = idx2[mask]
        edge_attr = None if edge_attr is None else edge_attr[mask]

    if edge_attr is None:
        edge_attr = paddle.ones(shape=[idx0.numel()], dtype=edge_index.dtype)

    size = [batch_size, max_num_nodes, max_num_nodes]
    size.extend(list(edge_attr.shape[1:]))
    flattened_size = batch_size * max_num_nodes * max_num_nodes

    idx = idx0 * max_num_nodes * max_num_nodes + idx1 * max_num_nodes + idx2
    adj_partial = segment_sum(edge_attr, idx)
    adj = paddle.zeros([flattened_size, edge_attr.shape[1]], dtype=paddle.float32)
    index = paddle.arange(idx.max() + 1)
    adj[index] = adj_partial
    adj = paddle.reshape(adj, size)

    return adj


def encode_no_edge(E):
    assert len(E.shape) == 4
    if E.shape[-1] == 0:
        return E
    no_edge = paddle.sum(E, axis=3) == 0
    first_elt = E[:, :, :, 0]
    first_elt = paddle.where(no_edge, paddle.ones_like(first_elt), first_elt)
    E[:, :, :, 0] = first_elt
    diag = paddle.eye(E.shape[1], dtype="int32").unsqueeze(0).tile([E.shape[0], 1, 1])
    diag = diag.astype("bool")
    E = paddle.where(diag.unsqueeze(-1), paddle.zeros_like(E), E)
    return E


def return_empty(x, shape=None):
    if shape is not None:
        return paddle.empty(shape, dtype="float32")
    return paddle.empty(x.shape, dtype="float32")


# ===========================
# 测试代码
# ===========================
if __name__ == "__main__":
    import paddle

    # from paddle import sparse
    # 创建测试数据
    x = paddle.arange(15).reshape([5, 3])  # 5个节点，每个节点3维特征
    edge_index = paddle.to_tensor([[0, 1, 2], [1, 2, 0]], dtype="int64")
    edge_attr = paddle.ones([3, 2]) * 2  # 3条边，每条边2维特征
    batch = paddle.to_tensor([0, 0, 1, 1, 1], dtype="int64")

    # 测试 to_dense 函数
    placeholder, node_mask = to_dense(x, edge_index, edge_attr, batch)
    print("X Shape:", placeholder.X.shape)
    print("E Shape:", placeholder.E.shape)
    print("Node Mask:", node_mask.shape)
