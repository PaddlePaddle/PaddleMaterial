import paddle
import paddle.nn.functional as F
from ppmat.utils.digressutils import PlaceHolder

# ==============
# 一些辅助函数
# ==============

def paddle_mode(tensor, axis=1):
    """
    近似替代 torch.mode(...).values 的功能，返回给定 axis 上出现次数最多的元素。
    注：Paddle 当前无直接 mode API，这里用一种简化写法:
        1. 将 tensor 转为 numpy
        2. 调用 scipy.stats 或 numpy 的方法找 mode
        3. 再转回到 paddle.Tensor
    如果您的场景对性能要求高，可自行实现更高效的 Paddle 原生统计。
    """
    import numpy as np
    data_np = tensor.numpy()
    # 计算每行的众数
    # 如果您有 scipy 可这样:
    # from scipy.stats import mode
    # m = mode(data_np, axis=axis, keepdims=False).mode
    # 这里纯 numpy 实现:
    bs = data_np.shape[0]
    modes = []
    for i in range(bs):
        vals, counts = np.unique(data_np[i], return_counts=True)
        max_count_idx = np.argmax(counts)
        modes.append(vals[max_count_idx])
    modes_np = np.array(modes).reshape([-1])  # shape (bs,)
    return paddle.to_tensor(modes_np, dtype=tensor.dtype)


def round_to_decimals(tensor, decimals=3):
    factor = 10 ** decimals
    return paddle.round(tensor * factor) / factor


# ======================================
# 1. DummyExtraFeatures (Paddle 版本)
# ======================================
class DummyExtraFeatures:
    def __init__(self):
        """ This class does not compute anything, just returns empty tensors."""

    def __call__(self, noisy_data):
        X = noisy_data['X_t']
        E = noisy_data['E_t']
        y = noisy_data['y_t']

        # 对应 torch 中 X.new_zeros(...):
        empty_x = paddle.zeros(shape=X.shape[:-1] + [0], dtype=X.dtype)
        empty_e = paddle.zeros(shape=E.shape[:-1] + [0], dtype=E.dtype)
        empty_y = paddle.zeros(shape=[y.shape[0], 0], dtype=y.dtype)

        return PlaceHolder(X=empty_x, E=empty_e, y=empty_y)


# ======================================
# 2. ExtraFeatures (Paddle 版本)
# ======================================
class ExtraFeatures:
    def __init__(self, extra_features_type, dataset_info):
        self.max_n_nodes = dataset_info.max_n_nodes
        self.ncycles = NodeCycleFeatures()
        self.features_type = extra_features_type
        if extra_features_type in ['eigenvalues', 'all']:
            self.eigenfeatures = EigenFeatures(mode=extra_features_type)

    def __call__(self, noisy_data):
        # n: (bs,1)
        mask_sum = paddle.sum(noisy_data['node_mask'], axis=1, keepdim=False)  # (bs,)
        n = paddle.unsqueeze(mask_sum / self.max_n_nodes, axis=1)              # (bs,1)

        # x_cycles, y_cycles: (bs, ?)
        x_cycles, y_cycles = self.ncycles(noisy_data)  # (bs, n_cycles)

        if self.features_type == 'cycles':
            E = noisy_data['E_t']
            extra_edge_attr = paddle.zeros(shape=E.shape[:-1] + [0], dtype=E.dtype)

            # 等效于 torch.hstack((n, y_cycles)) => concat along axis=1
            # 假设 n shape (bs,1), y_cycles shape (bs,k)
            # => result shape (bs, 1+k)
            y_stacked = paddle.concat([n, y_cycles], axis=1)

            return PlaceHolder(X=x_cycles, E=extra_edge_attr, y=y_stacked)

        elif self.features_type == 'eigenvalues':
            eigenfeatures = self.eigenfeatures(noisy_data)
            E = noisy_data['E_t']
            extra_edge_attr = paddle.zeros(shape=E.shape[:-1] + [0], dtype=E.dtype)

            n_components, batched_eigenvalues = eigenfeatures  # (bs,1), (bs,10)

            # hstack => concat along axis=1
            y_stacked = paddle.concat([n, y_cycles, n_components, batched_eigenvalues], axis=1)

            return PlaceHolder(X=x_cycles, E=extra_edge_attr, y=y_stacked)

        elif self.features_type == 'all':
            eigenfeatures = self.eigenfeatures(noisy_data)
            E = noisy_data['E_t']
            extra_edge_attr = paddle.zeros(shape=E.shape[:-1] + [0], dtype=E.dtype)

            n_components, batched_eigenvalues, nonlcc_indicator, k_lowest_eigvec = eigenfeatures
            # X = concat [x_cycles, nonlcc_indicator, k_lowest_eigvec] along last dim
            X_cat = paddle.concat([x_cycles, nonlcc_indicator, k_lowest_eigvec], axis=-1)

            # y = hstack => concat along axis=1
            y_stacked = paddle.concat([n, y_cycles, n_components, batched_eigenvalues], axis=1)

            return PlaceHolder(X=X_cat, E=extra_edge_attr, y=y_stacked)

        else:
            raise ValueError(f"Features type {self.features_type} not implemented")


# ======================================
# 3. NodeCycleFeatures (Paddle 版本)
# ======================================
class NodeCycleFeatures:
    def __init__(self):
        self.kcycles = KNodeCycles()

    def __call__(self, noisy_data):
        # adj_matrix: (bs, n, n), 取 E_t[...,1:] 并在最后一维 sum => shape (bs, n, n)
        E_t = noisy_data['E_t']
        adj_matrix = paddle.sum(E_t[..., 1:], axis=-1).astype('float32')  # (bs, n, n)

        x_cycles, y_cycles = self.kcycles.k_cycles(adj_matrix=adj_matrix)  # (bs, n_cycles)

        # x_cycles 与 node_mask 对应位置相乘
        node_mask = paddle.unsqueeze(noisy_data['node_mask'], axis=-1)  # (bs, n, 1)
        x_cycles = x_cycles.astype(adj_matrix.dtype) * node_mask

        # Avoid large values when the graph is dense
        x_cycles = x_cycles / 10
        y_cycles = y_cycles / 10

        # 类似 x_cycles[x_cycles > 1] = 1
        # Paddle 不支持直接 in-place boolean mask；需要先构造mask再赋值
        bool_mask_x = x_cycles > 1
        bool_mask_y = y_cycles > 1
        x_cycles = paddle.where(bool_mask_x, paddle.ones_like(x_cycles), x_cycles)
        y_cycles = paddle.where(bool_mask_y, paddle.ones_like(y_cycles), y_cycles)

        return x_cycles, y_cycles


# ======================================
# 4. EigenFeatures (Paddle 版本)
# ======================================
class EigenFeatures:
    """
    Code taken from : https://github.com/Saro00/DGN/blob/master/models/pytorch/eigen_agg.py
    """
    def __init__(self, mode):
        """ mode: 'eigenvalues' or 'all' """
        self.mode = mode

    def __call__(self, noisy_data):
        E_t = noisy_data['E_t']
        mask = noisy_data['node_mask']
        A = paddle.sum(E_t[..., 1:], axis=-1).astype('float32')  # (bs, n, n)
        A = A * paddle.unsqueeze(mask, axis=1) * paddle.unsqueeze(mask, axis=2)

        L = compute_laplacian(A, normalize=False)

        # 添加正则化项以防止计算失败
        n_ = L.shape[-1]
        eps_eye = paddle.eye(n_, dtype=L.dtype) * 1e-6
        L = L + eps_eye
        # 强制对称化
        L = (L + paddle.transpose(L, perm=[0, 2, 1])) / 2

        # 构造对 mask 外节点的惩罚项
        mask_diag = paddle.eye(n_, dtype=L.dtype) * (2 * n_)
        mask_diag = paddle.unsqueeze(mask_diag, axis=0)  # (1, n, n)
        # (~mask) => paddle.logical_not
        mask_bool = mask.astype('bool')
        mask_diag = mask_diag * paddle.logical_not(paddle.unsqueeze(mask_bool, 1)) \
                             * paddle.logical_not(paddle.unsqueeze(mask_bool, 2))

        L = L * paddle.unsqueeze(mask, axis=1) * paddle.unsqueeze(mask, axis=2) + mask_diag

        if self.mode == 'eigenvalues':
            # paddle.linalg.eigvalsh => (bs, n)
            eigvals = paddle.linalg.eigvalsh(L)  # (bs, n)
            sum_mask = paddle.sum(mask, axis=1, keepdim=True)
            eigvals = eigvals.astype(A.dtype) / sum_mask  # (bs,n)

            n_connected_comp, batch_eigenvalues = get_eigenvalues_features(eigvals)
            return (n_connected_comp.astype(A.dtype),
                    batch_eigenvalues.astype(A.dtype))

        elif self.mode == 'all':
            try:
                eigvals, eigvectors = paddle.linalg.eigh(L)  # (bs,n), (bs,n,n)
            except Exception as e:
                print(f"Warning: Eigen decomposition failed with linalg.eigh: {e}")
                # 回退到 SVD
                U, S, Vh = paddle.linalg.svd(L)
                eigvals = S
                eigvectors = paddle.transpose(Vh, perm=[0, 2, 1])
                print("Using SVD as fallback method.")

            sum_mask = paddle.sum(mask, axis=1, keepdim=True)
            eigvals = eigvals.astype(A.dtype) / sum_mask
            eigvectors = eigvectors * paddle.unsqueeze(mask, axis=2) * paddle.unsqueeze(mask, axis=1)

            # Retrieve eigenvalues features
            n_connected_comp, batch_eigenvalues = get_eigenvalues_features(eigvals)

            # Retrieve eigenvectors features
            nonlcc_indicator, k_lowest_eigvec = get_eigenvectors_features(
                vectors=eigvectors, node_mask=mask, n_connected=n_connected_comp
            )
            return n_connected_comp, batch_eigenvalues, nonlcc_indicator, k_lowest_eigvec
        else:
            raise NotImplementedError(f"Mode {self.mode} is not implemented")


# ======================================
# 5. compute_laplacian (Paddle 版本)
# ======================================
def compute_laplacian(adjacency, normalize: bool):
    """
    adjacency : batched adjacency matrix (bs, n, n)
    normalize: can be None, 'sym' or 'rw'
    """
    diag = paddle.sum(adjacency, axis=-1)  # (bs, n)
    n = diag.shape[-1]
    D = paddle.diag_embed(diag)            # (bs, n, n)
    combinatorial = D - adjacency          # (bs, n, n)

    if not normalize:
        return (combinatorial + paddle.transpose(combinatorial, perm=[0, 2, 1])) / 2

    diag0 = diag.clone()
    diag = paddle.where(diag == 0, paddle.to_tensor(1e-12, dtype=diag.dtype), diag)
    diag_norm = 1.0 / paddle.sqrt(diag)    # (bs, n)
    D_norm = paddle.diag_embed(diag_norm)  # (bs, n, n)
    eye_n = paddle.unsqueeze(paddle.eye(n, dtype=adjacency.dtype), axis=0)  # (1, n, n)

    L = eye_n - D_norm @ adjacency @ D_norm
    # 对于 diag0 == 0 的节点，令其对应位置 = 0
    zero_mask = (diag0 == 0).astype(L.dtype)  # (bs, n)
    # 需要 broadcast 到 (bs,n,n)，可先 unsqueeze
    zero_mask_2d = paddle.unsqueeze(zero_mask, axis=-1)  # (bs, n, 1)
    zero_mask_2d = paddle.matmul(zero_mask_2d, paddle.ones_like(zero_mask_2d).transpose([0, 2, 1]))
    # 将 L 中对应位置 = 0
    L = paddle.where(zero_mask_2d > 0, paddle.zeros_like(L), L)
    return (L + paddle.transpose(L, perm=[0, 2, 1])) / 2


# ======================================
# 6. get_eigenvalues_features (Paddle 版本)
# ======================================
def get_eigenvalues_features(eigenvalues, k=5):
    """
    eigenvalues: (bs, n)
    k: num of non zero eigenvalues to keep
    """
    ev = eigenvalues
    bs, n = ev.shape
    # n_connected_components = (ev < 1e-5).sum(dim=-1)
    n_connected_components = paddle.sum(ev < 1e-5, axis=-1)  # (bs,)

    # 断言: 这里可能需要自己处理报错 or 做一个检查
    # assert (n_connected_components > 0).all(), "some assert..."

    to_extend = int(paddle.max(n_connected_components).numpy()[0]) + k - n
    if to_extend > 0:
        # 相当于 torch.hstack([...]) => paddle.concat([...], axis=1)
        fill_val = paddle.full(shape=[bs, to_extend], fill_value=2.0, dtype=ev.dtype)
        ev_extended = paddle.concat([ev, fill_val], axis=1)  # (bs, n+to_extend)
    else:
        ev_extended = ev

    # indices => shape (bs,k)
    #   range(k) + n_connected_components.unsqueeze(1)
    range_k = paddle.arange(k, dtype='int64')  # (k,)
    range_k = paddle.unsqueeze(range_k, axis=0)  # (1,k)
    indices = range_k + paddle.unsqueeze(n_connected_components.astype('int64'), axis=1)  # (bs,k)
    # gather => 需要 index 的 shape 与 ev_extended 一致
    # paddle.gather(ev_extended, axis=1, index=indices) 要写个 batch_gather
    first_k_ev = batch_gather_2d(ev_extended, indices)

    n_connected_components = paddle.unsqueeze(n_connected_components, axis=-1)  # (bs,1)
    return n_connected_components, first_k_ev


def batch_gather_2d(data, index):
    """
    仿照 PyTorch 的 gather(dim=1)，对 2D data (bs, m)，
    根据同样 shape (bs, k) 的 index，返回 (bs, k)
    """
    bs, m = data.shape
    _, k = index.shape
    row_idx = paddle.arange(bs, dtype='int64')
    row_idx = paddle.unsqueeze(row_idx, axis=-1)  # (bs,1)
    row_idx = paddle.expand(row_idx, [bs, k])     # (bs,k)

    # 将 row_idx 和 index 拼成 (bs*k, 2)
    flat_indices = paddle.stack([row_idx.flatten(), index.flatten()], axis=1)
    # 按照 flat_indices 在 data 上取值
    gathered = paddle.gather_nd(data, flat_indices)  # (bs*k,)
    # reshape 回 (bs,k)
    gathered = paddle.reshape(gathered, [bs, k])
    return gathered


# ======================================
# 7. get_eigenvectors_features (Paddle 版本)
# ======================================
def get_eigenvectors_features(vectors, node_mask, n_connected, k=2):
    """
    vectors: (bs, n, n) => eigenvectors (in columns)
    returns:
      not_lcc_indicator: (bs, n, 1)
      k_lowest_eigvec:   (bs, n, k)
    """
    bs, n = vectors.shape[0], vectors.shape[1]

    # first_ev => 取第 0 列 => vectors[:, :, 0]  shape (bs,n)
    first_ev = vectors[:, :, 0]
    # round to 3 decimals
    first_ev = round_to_decimals(first_ev, decimals=3)
    # 与 node_mask 相乘
    first_ev = first_ev * node_mask

    # 加一些随机数到 mask 里，防止 0 变成众数
    # random => paddle.randn([bs,n]) * (1 - node_mask)  (若 node_mask 是 float 0/1)
    random_part = paddle.randn([bs, n], dtype=first_ev.dtype)
    random_part = random_part * (1.0 - node_mask)
    first_ev = first_ev + random_part

    # 计算每行的众数
    most_common = paddle_mode(first_ev, axis=1)  # (bs,)

    # 构造 mask => ~ (first_ev == most_common.unsqueeze(1))
    # 先判断相等, shape (bs,n)
    mc_expanded = paddle.unsqueeze(most_common, axis=1)  # (bs,1)
    eq_mask = paddle.equal(first_ev, mc_expanded)
    mask = paddle.logical_not(eq_mask)
    # not_lcc_indicator => (mask * node_mask).unsqueeze(-1)
    # 这里 node_mask 若是 float，需要先转 bool
    node_mask_bool = (node_mask > 0.5)
    combined_mask = paddle.logical_and(mask, node_mask_bool)  # (bs,n)
    not_lcc_indicator = paddle.unsqueeze(combined_mask.astype('float32'), axis=-1)

    # 拿到前 k 个非零特征向量
    to_extend = int(paddle.max(n_connected).numpy()[0]) + k - n
    if to_extend > 0:
        extension = paddle.zeros(shape=[bs, n, to_extend], dtype=vectors.dtype)
        vectors = paddle.concat([vectors, extension], axis=2)  # (bs, n, n+to_extend)

    # indices => shape (bs, 1, k)
    range_k = paddle.arange(k, dtype='int64')  # (k,)
    range_k = paddle.unsqueeze(range_k, axis=[0, 1])  # (1,1,k)
    n_connected_i64 = paddle.unsqueeze(n_connected.astype('int64'), axis=2)  # (bs,1,1)
    # 广播 => (bs,1,k)
    range_k = paddle.expand(range_k, [n_connected_i64.shape[0], 1, k])
    indices = range_k + n_connected_i64  # (bs,1,k)
    # expand 到 (bs,n,k)
    indices = paddle.expand(indices, [bs, n, k])  # (bs,n,k)

    # gather => 需要 batch gather
    # vectors shape (bs,n,n+to_extend)
    # indices shape (bs,n,k)
    # 在最后一维 gather => axis=2
    k_lowest_eigvec = batch_gather_3d(vectors, indices)  # (bs,n,k)

    # 乘以 node_mask
    k_lowest_eigvec = k_lowest_eigvec * paddle.unsqueeze(node_mask, axis=2)  # (bs,n,k)
    return not_lcc_indicator, k_lowest_eigvec


def batch_gather_3d(data, index):
    """
    仿 torch.gather(dim=2).
    data:   (bs,n,m)
    index:  (bs,n,k), each element in [0, m-1]
    output: (bs,n,k)
    """
    bs, n, m = data.shape
    _, _, k = index.shape

    # 将 (bs,n,k) 展平 => (bs*n*k)
    data_flat = paddle.reshape(data, [bs*n, m])     # (bs*n, m)
    index_flat = paddle.reshape(index, [bs*n, k])   # (bs*n, k)

    # 构造行号: 0 ~ bs*n-1
    row_idx = paddle.arange(0, bs*n, dtype='int64')
    row_idx = paddle.unsqueeze(row_idx, axis=-1)  # (bs*n, 1)
    row_idx = paddle.expand(row_idx, [bs*n, k])   # (bs*n, k)

    gather_idx = paddle.stack([row_idx.flatten(), index_flat.flatten()], axis=1)  # (bs*n*k, 2)
    out_flat = paddle.gather_nd(data_flat, gather_idx)       # (bs*n*k,)
    out = paddle.reshape(out_flat, [bs, n, k])               # (bs,n,k)
    return out


# ======================================
# 8. batch_trace, batch_diagonal (Paddle 版本)
# ======================================
def batch_trace(X):
    """
    X: shape (bs, n, n)
    返回每个样本的 trace => shape (bs,)
    """
    diag = paddle.diagonal(X, axis1=-2, axis2=-1)  # (bs, n)
    return paddle.sum(diag, axis=-1)


def batch_diagonal(X):
    """
    X: shape (bs, n, n)
    返回其对角线 => (bs, n)
    """
    return paddle.diagonal(X, axis1=-2, axis2=-1)


# ======================================
# 9. KNodeCycles (Paddle 版本)
# ======================================
class KNodeCycles:
    """ Builds cycle counts for each node in a graph.
    """

    def __init__(self):
        super().__init__()

    def calculate_kpowers(self):
        self.k1_matrix = self.adj_matrix.astype('float32')
        self.d = paddle.sum(self.adj_matrix, axis=-1)  # (bs,n)
        self.k2_matrix = paddle.matmul(self.k1_matrix, self.adj_matrix.astype('float32'))
        self.k3_matrix = paddle.matmul(self.k2_matrix, self.adj_matrix.astype('float32'))
        self.k4_matrix = paddle.matmul(self.k3_matrix, self.adj_matrix.astype('float32'))
        self.k5_matrix = paddle.matmul(self.k4_matrix, self.adj_matrix.astype('float32'))
        self.k6_matrix = paddle.matmul(self.k5_matrix, self.adj_matrix.astype('float32'))

    def k3_cycle(self):
        c3 = batch_diagonal(self.k3_matrix)
        x3 = (c3 / 2.0).unsqueeze(-1).astype('float32')
        y3 = (paddle.sum(c3, axis=-1) / 6.0).unsqueeze(-1).astype('float32')
        return x3, y3

    def k4_cycle(self):
        diag_a4 = batch_diagonal(self.k4_matrix)  # (bs,n)
        c4 = diag_a4 - self.d * (self.d - 1) - paddle.sum(
            paddle.matmul(self.adj_matrix, paddle.unsqueeze(self.d, axis=-1)), axis=-1
        )
        x4 = (c4 / 2.0).unsqueeze(-1).astype('float32')
        y4 = (paddle.sum(c4, axis=-1) / 8.0).unsqueeze(-1).astype('float32')
        return x4, y4

    def k5_cycle(self):
        diag_a5 = batch_diagonal(self.k5_matrix)  # (bs,n)
        triangles = batch_diagonal(self.k3_matrix) / 2.0  # (bs,n)

        joint_cycles = self.k2_matrix * self.adj_matrix  # (bs,n,n)
        prod = paddle.squeeze(paddle.matmul(joint_cycles, paddle.unsqueeze(self.d, axis=-1)), axis=-1)
        prod2 = paddle.squeeze(paddle.matmul(self.adj_matrix, paddle.unsqueeze(triangles, axis=-1)), axis=-1)

        c5 = diag_a5 - prod - 4.0 * self.d * triangles - prod2 + 10.0 * triangles
        x5 = (c5 / 2.0).unsqueeze(-1).astype('float32')
        y5 = (paddle.sum(c5, axis=-1) / 10.0).unsqueeze(-1).astype('float32')
        return x5, y5

    def k6_cycle(self):
        term_1_t = batch_trace(self.k6_matrix)
        term_2_t = batch_trace(paddle.pow(self.k3_matrix, 2.0))
        term3_t = paddle.sum(self.adj_matrix * paddle.pow(self.k2_matrix, 2.0), axis=[-2, -1])
        d_t4 = batch_diagonal(self.k2_matrix)
        a_4_t = batch_diagonal(self.k4_matrix)
        term_4_t = paddle.sum(d_t4 * a_4_t, axis=-1)
        term_5_t = batch_trace(self.k4_matrix)
        term_6_t = batch_trace(self.k3_matrix)
        term_7_t = paddle.sum(paddle.pow(batch_diagonal(self.k2_matrix), 3.0), axis=-1)
        term8_t = paddle.sum(self.k3_matrix, axis=[-2, -1])
        term9_t = paddle.sum(paddle.pow(batch_diagonal(self.k2_matrix), 2.0), axis=-1)
        term10_t = batch_trace(self.k2_matrix)

        c6_t = (term_1_t - 3.0 * term_2_t + 9.0 * term3_t - 6.0 * term_4_t
                + 6.0 * term_5_t - 4.0 * term_6_t + 4.0 * term_7_t
                + 3.0 * term8_t - 12.0 * term9_t + 4.0 * term10_t)
        y6 = (c6_t / 12.0).unsqueeze(-1).astype('float32')
        return None, y6

    def k_cycles(self, adj_matrix, verbose=False):
        """
        adj_matrix: (bs, n, n)
        return: (kcyclesx, kcyclesy)
        """
        self.adj_matrix = adj_matrix
        self.calculate_kpowers()

        k3x, k3y = self.k3_cycle()
        assert paddle.all(k3x >= -0.1)

        k4x, k4y = self.k4_cycle()
        assert paddle.all(k4x >= -0.1)

        k5x, k5y = self.k5_cycle()
        assert paddle.all(k5x >= -0.1), k5x

        _, k6y = self.k6_cycle()
        assert paddle.all(k6y >= -0.1)

        kcyclesx = paddle.concat([k3x, k4x, k5x], axis=-1)  # (bs, n, 3)
        kcyclesy = paddle.concat([k3y, k4y, k5y, k6y], axis=-1)  # (bs, ncycles?)
        return kcyclesx, kcyclesy
