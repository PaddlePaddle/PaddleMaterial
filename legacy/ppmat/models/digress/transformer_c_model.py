import math

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ppmat.utils.digressutils import PlaceHolder
from ppmat.models.digress.layer import Xtoy, Etoy, masked_softmax
from ppmat.models.digress import diffusion_utils

class XEyTransformerLayer(nn.Layer):
    """ 
    Paddle 版本的 Transformer 层，用于同时更新节点 (X)、边 (E) 和全局 (y) 特征。
    
    dx, de, dy: 节点/边/全局特征维度
    n_head: 多头注意力的头数
    dim_ffX, dim_ffE, dim_ffy: FFN 维度
    dropout: Dropout 概率
    layer_norm_eps: LayerNorm 的 eps
    """
    def __init__(self, dx: int, de: int, dy: int, n_head: int,
                 dim_ffX: int = 2048, dim_ffE: int = 128, dim_ffy: int = 2048,
                 dropout: float = 0.1, layer_norm_eps: float = 1e-5, 
                 device=None, dtype=None):
        super().__init__()

        # Paddle 中若要指定 device、dtype，可用 to_tensor 或 set_device 等操作，
        # 这里先保留接口，代码示例中暂不显式使用。

        self.self_attn = NodeEdgeBlock(dx, de, dy, n_head)

        # FFN for X
        self.linX1 = nn.Linear(dx, dim_ffX)
        self.linX2 = nn.Linear(dim_ffX, dx)
        self.normX1 = nn.LayerNorm(dx, epsilon=layer_norm_eps)
        self.normX2 = nn.LayerNorm(dx, epsilon=layer_norm_eps)
        self.dropoutX1 = nn.Dropout(dropout)
        self.dropoutX2 = nn.Dropout(dropout)
        self.dropoutX3 = nn.Dropout(dropout)

        # FFN for E
        self.linE1 = nn.Linear(de, dim_ffE)
        self.linE2 = nn.Linear(dim_ffE, de)
        self.normE1 = nn.LayerNorm(de, epsilon=layer_norm_eps)
        self.normE2 = nn.LayerNorm(de, epsilon=layer_norm_eps)
        self.dropoutE1 = nn.Dropout(dropout)
        self.dropoutE2 = nn.Dropout(dropout)
        self.dropoutE3 = nn.Dropout(dropout)

        # FFN for y
        self.lin_y1 = nn.Linear(dy, dim_ffy)
        self.lin_y2 = nn.Linear(dim_ffy, dy)
        self.norm_y1 = nn.LayerNorm(dy, epsilon=layer_norm_eps)
        self.norm_y2 = nn.LayerNorm(dy, epsilon=layer_norm_eps)
        self.dropout_y1 = nn.Dropout(dropout)
        self.dropout_y2 = nn.Dropout(dropout)
        self.dropout_y3 = nn.Dropout(dropout)

        self.activation = F.relu

    def forward(self, X: paddle.Tensor, E: paddle.Tensor, y: paddle.Tensor, node_mask: paddle.Tensor):
        """
        X: (bs, n, dx)
        E: (bs, n, n, de)
        y: (bs, dy)
        node_mask: (bs, n) 
        return: newX, newE, new_y
        """
        newX, newE, new_y = self.self_attn(X, E, y, node_mask=node_mask)

        # --- residual & norm for X ---
        newX_d = self.dropoutX1(newX)
        X = self.normX1(X + newX_d)

        # --- residual & norm for E ---
        newE_d = self.dropoutE1(newE)
        E = self.normE1(E + newE_d)

        # --- residual & norm for y ---
        new_y_d = self.dropout_y1(new_y)
        y = self.norm_y1(y + new_y_d)

        # --- feed-forward for X ---
        ff_outputX = self.linX2(self.dropoutX2(self.activation(self.linX1(X))))
        ff_outputX = self.dropoutX3(ff_outputX)
        X = self.normX2(X + ff_outputX)

        # --- feed-forward for E ---
        ff_outputE = self.linE2(self.dropoutE2(self.activation(self.linE1(E))))
        ff_outputE = self.dropoutE3(ff_outputE)
        E = self.normE2(E + ff_outputE)

        # --- feed-forward for y ---
        ff_output_y = self.lin_y2(self.dropout_y2(self.activation(self.lin_y1(y))))
        ff_output_y = self.dropout_y3(ff_output_y)
        y = self.norm_y2(y + ff_output_y)

        return X, E, y


class NodeEdgeBlock(nn.Layer):
    """
    自注意力层 + 对边特征的更新
    """
    def __init__(self, dx, de, dy, n_head):
        super().__init__()
        assert dx % n_head == 0, f"dx: {dx} -- n_head: {n_head}"
        self.dx = dx
        self.de = de
        self.dy = dy
        self.df = dx // n_head
        self.n_head = n_head

        # Attention Q,K,V
        self.q = nn.Linear(dx, dx)
        self.k = nn.Linear(dx, dx)
        self.v = nn.Linear(dx, dx)

        # FiLM E -> X
        self.e_add = nn.Linear(de, dx)
        self.e_mul = nn.Linear(de, dx)

        # FiLM y -> E
        self.y_e_mul = nn.Linear(dy, dx)  # 注意原代码注释说这里是 dx 而非 de
        self.y_e_add = nn.Linear(dy, dx)

        # FiLM y -> X
        self.y_x_mul = nn.Linear(dy, dx)
        self.y_x_add = nn.Linear(dy, dx)

        # process y
        self.y_y = nn.Linear(dy, dy)
        self.x_y = Xtoy(dx, dy)
        self.e_y = Etoy(de, dy)

        # Output layers
        self.x_out = nn.Linear(dx, dx)
        self.e_out = nn.Linear(dx, de)
        self.y_out = nn.Sequential(
            nn.Linear(dy, dy),
            nn.ReLU(),
            nn.Linear(dy, dy)
        )

    def forward(self, X, E, y, node_mask):
        """
        X: (bs, n, dx)
        E: (bs, n, n, de)
        y: (bs, dy)
        node_mask: (bs, n)
        return: newX, newE, newY
        """
        bs, n, _ = X.shape
        x_mask = paddle.unsqueeze(node_mask, axis=-1)            # (bs, n, 1)
        e_mask1 = paddle.unsqueeze(x_mask, axis=2)               # (bs, n, 1, 1)
        e_mask2 = paddle.unsqueeze(x_mask, axis=1)               # (bs, 1, n, 1)

        # 1. Map X to Q,K
        Q = self.q(X) * x_mask
        K = self.k(X) * x_mask
        diffusion_utils.assert_correctly_masked(Q, x_mask)

        # 2. Reshape to (bs, n, n_head, df)
        Q = paddle.reshape(Q, [Q.shape[0], Q.shape[1], self.n_head, self.df])
        K = paddle.reshape(K, [K.shape[0], K.shape[1], self.n_head, self.df])

        # Q => (bs, n, 1, n_head, df) or (bs, 1, n, n_head, df) 根据原逻辑
        Q = paddle.unsqueeze(Q, axis=2)   # shape: (bs, n, 1, n_head, df)
        K = paddle.unsqueeze(K, axis=1)   # shape: (bs, 1, n, n_head, df)

        # 计算注意力分数 Y
        Y = Q * K
        Y = Y / math.sqrt(Y.shape[-1])
        diffusion_utils.assert_correctly_masked(Y, paddle.unsqueeze(e_mask1 * e_mask2, axis=-1))

        # E1, E2
        E1 = self.e_mul(E) * e_mask1 * e_mask2  # (bs, n, n, dx)
        E1 = paddle.reshape(E1, [E.shape[0], E.shape[1], E.shape[2], self.n_head, self.df])

        E2 = self.e_add(E) * e_mask1 * e_mask2
        E2 = paddle.reshape(E2, [E.shape[0], E.shape[1], E.shape[2], self.n_head, self.df])

        # Incorporate edge features
        Y = Y * (E1 + 1.0) + E2

        # Incorporate y to E
        newE = paddle.flatten(Y, start_axis=3)          # => (bs, n, n, dx)
        ye_add = self.y_e_add(y).unsqueeze(1).unsqueeze(1)  # (bs,1,1,dx)
        ye_mul = self.y_e_mul(y).unsqueeze(1).unsqueeze(1)  # (bs,1,1,dx)
        newE = ye_add + (ye_mul + 1.0) * newE

        # Output E
        newE = self.e_out(newE) * e_mask1 * e_mask2
        diffusion_utils.assert_correctly_masked(newE, e_mask1 * e_mask2)

        # compute attn => (bs, n, n, n_head, df)
        # softmax over dim=2
        softmax_mask = paddle.expand(e_mask2, [bs, n, n, self.n_head])  
        attn = masked_softmax(Y, softmax_mask, dim=2)  # shape still (bs,n,n,n_head,df)

        V = self.v(X) * x_mask
        V = paddle.reshape(V, [V.shape[0], V.shape[1], self.n_head, self.df])
        V = paddle.unsqueeze(V, axis=1)  # (bs,1,n,n_head,df)

        # Weighted sum
        weighted_V = attn * V
        weighted_V = paddle.sum(weighted_V, axis=2)
        weighted_V = paddle.flatten(weighted_V, start_axis=2)  # (bs, n, dx)

        # Incorporate y -> X
        yx_add = self.y_x_add(y).unsqueeze(1)  # (bs,1,dx)
        yx_mul = self.y_x_mul(y).unsqueeze(1)
        newX = yx_add + (yx_mul + 1.0) * weighted_V

        # Output X
        newX = self.x_out(newX) * x_mask
        diffusion_utils.assert_correctly_masked(newX, x_mask)

        # Process y
        y_ = self.y_y(y)
        e_y_ = self.e_y(E, e_mask1, e_mask2)
        x_y_ = self.x_y(X, x_mask)
        new_y = y_ + x_y_ + e_y_
        new_y = self.y_out(new_y)

        return newX, newE, new_y


class GraphTransformer_C(nn.Layer):
    """
    n_layers : int -- number of layers
    dims : dict -- contains dimensions for each feature type
    """
    def __init__(self, n_layers: int, input_dims: dict, hidden_mlp_dims: dict, 
                 hidden_dims: dict, output_dims: dict,
                 act_fn_in=nn.ReLU(), act_fn_out=nn.ReLU()):
        super().__init__()
        self.n_layers = n_layers
        self.out_dim_X = output_dims['X']
        self.out_dim_E = output_dims['E']
        self.out_dim_y = output_dims['y']

        self.mlp_in_X = nn.Sequential(
            nn.Linear(input_dims['X'], hidden_mlp_dims['X']), act_fn_in,
            nn.Linear(hidden_mlp_dims['X'], hidden_dims['dx']), act_fn_in
        )

        self.mlp_in_E = nn.Sequential(
            nn.Linear(input_dims['E'], hidden_mlp_dims['E']), act_fn_in,
            nn.Linear(hidden_mlp_dims['E'], hidden_dims['de']), act_fn_in
        )

        self.mlp_in_y = nn.Sequential(
            nn.Linear(input_dims['y'], hidden_mlp_dims['y']), act_fn_in,
            nn.Linear(hidden_mlp_dims['y'], hidden_dims['dy']), act_fn_in
        )

        # 用 LayerList 存放多层 Transformer
        self.tf_layers = nn.LayerList([
            XEyTransformerLayer(dx=hidden_dims['dx'],
                                de=hidden_dims['de'],
                                dy=hidden_dims['dy'],
                                n_head=hidden_dims['n_head'],
                                dim_ffX=hidden_dims['dim_ffX'],
                                dim_ffE=hidden_dims['dim_ffE'])
            for _ in range(n_layers)
        ])

        self.mlp_out_X = nn.Sequential(
            nn.Linear(hidden_dims['dx'], hidden_mlp_dims['X']), act_fn_out,
            nn.Linear(hidden_mlp_dims['X'], 512)
        )

        # 其他输出层（如 E, y）在原示例里注释了，可自行添加
        # self.mlp_out_E = ...
        # self.mlp_out_y = ...

    def forward(self, X, E, y, node_mask):
        """
        X: (bs, n, input_dims['X'])
        E: (bs, n, n, input_dims['E'])
        y: (bs, input_dims['y'])
        node_mask: (bs, n)
        """
        bs, n = X.shape[0], X.shape[1]

        # 构建对角 mask (如需用)
        diag_mask = paddle.eye(n, dtype='bool')  # (n, n)
        diag_mask = paddle.logical_not(diag_mask)
        diag_mask = diag_mask.unsqueeze(0).unsqueeze(-1).expand([bs, -1, -1, -1])  # (bs,n,n,1)

        # 保存原来的 X/E/y 部分给 skip-connection（如果需要）
        X_to_out = X[..., :self.out_dim_X]
        E_to_out = E[..., :self.out_dim_E]
        y_to_out = y[..., :self.out_dim_y]

        # MLP in
        new_E = self.mlp_in_E(E)
        # 对称化
        E_t = paddle.transpose(new_E, perm=[0, 2, 1, 3])
        new_E = (new_E + E_t) / 2.0

        X = self.mlp_in_X(X)
        Y = self.mlp_in_y(y)

        after_in = PlaceHolder(X, E=new_E, y=Y).mask(node_mask)
        X, E, Y = after_in.X, after_in.E, after_in.y

        # 多层 Transformer
        for layer in self.tf_layers:
            X, E, Y = layer(X, E, Y, node_mask)

        # Output
        X = self.mlp_out_X(X)                # (bs, n, 512)
        X_mean = paddle.mean(X, axis=1)      # (bs, 512)

        # 如果还需要输出 E,y，可以在此添加 mlp_out_E, mlp_out_y
        # 并做对称化/加 skip connection 等

        return X_mean
