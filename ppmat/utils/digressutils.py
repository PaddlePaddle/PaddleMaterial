import paddle
import paddle.sparse as sparse


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
        x_mask = paddle.unsqueeze(node_mask, axis=-1)  # (bs, n, 1)
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

            # assert torch.allclose(self.E, torch.transpose(self.E, 1, 2))
            # Paddle => 需要改成 paddle.allclose(self.E, paddle.transpose(self.E, perm=[0,2,1]))
            if not paddle.allclose(self.E, paddle.transpose(self.E, perm=[0, 2, 1, 3])):
                raise ValueError("E is not symmetric after masking.")
        return self



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
    # 计算最大节点数
    num_nodes = x.shape[0]
    max_num_nodes = batch.max().item() + 1

    # 初始化密集化节点特征矩阵
    X = paddle.zeros([max_num_nodes, num_nodes, x.shape[1]])
    node_mask = paddle.zeros([max_num_nodes, num_nodes], dtype='bool')

    for i in range(max_num_nodes):
        mask = (batch == i)
        X[i, :mask.sum(), :] = x[mask]
        node_mask[i, :mask.sum()] = True

    # 移除自环
    edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)

    # 将边索引转换为稀疏矩阵形式
    values = edge_attr
    shape = [num_nodes, num_nodes, edge_attr.shape[1]]

    # 构建稀疏邻接矩阵
    indices = edge_index
    E = sparse.sparse_coo_tensor(indices, values, shape)
    E = sparse.to_dense(E)
    E = encode_no_edge(E)

    return PlaceHolder(X=X, E=E, y=None), node_mask


def remove_self_loops(edge_index, edge_attr):
    """
    移除自环 (即从 i 到 i 的边)
    """
    mask = edge_index[0] != edge_index[1]  # 仅保留非自环的边
    return edge_index[:, mask], edge_attr[mask]

def encode_no_edge(E):
    """
    将无边特征的元素编码为特定的标识符
    """
    assert len(E.shape) == 3  # E: (num_nodes, num_nodes, edge_dim)
    
    # 如果没有边特征，直接返回
    if E.shape[-1] == 0:
        return E

    # 计算无边的掩码
    no_edge = paddle.sum(E, axis=2) == 0

    # 设置第一个边特征为1（用于无边）
    first_elt = E[:, :, 0]
    first_elt[no_edge] = 1
    E[:, :, 0] = first_elt

    # 移除自环（对角线置0）
    diag = paddle.eye(E.shape[0], dtype='bool')
    E = paddle.where(diag.unsqueeze(-1), paddle.zeros_like(E), E)

    return E


# ===========================
# 测试代码
# ===========================
if __name__ == '__main__':
    import paddle
    from paddle import sparse

    # 创建测试数据
    x = paddle.randn([5, 3])  # 5个节点，每个节点3维特征
    edge_index = paddle.to_tensor([[0, 1, 2], [1, 2, 0]], dtype='int64')
    edge_attr = paddle.randn([3, 2])  # 3条边，每条边2维特征
    batch = paddle.to_tensor([0, 0, 1, 1, 1], dtype='int64')

    # 测试 to_dense 函数
    placeholder, node_mask = to_dense(x, edge_index, edge_attr, batch)
    print("X Shape:", placeholder.X.shape)
    print("E Shape:", placeholder.E.shape)
    print("Node Mask:", node_mask.shape)
