import sys
sys.path.append('/home/liuxuwei01/SpectrumMol/src_paddle/utils')
import paddle_aux
import paddle
import math
from src import utils
from src.diffusion import diffusion_utils
from src.models.layer import Xtoy, Etoy, masked_softmax


class XEyTransformerLayer(paddle.nn.Layer):
    """ Transformer that updates node, edge and global features
        d_x: node features
        d_e: edge features
        dz : global features
        n_head: the number of heads in the multi_head_attention
        dim_feedforward: the dimension of the feedforward network model after self-attention
        dropout: dropout probablility. 0 to disable
        layer_norm_eps: eps value in layer normalizations.
    """

    def __init__(self, dx: int, de: int, dy: int, n_head: int, dim_ffX: int
        =2048, dim_ffE: int=128, dim_ffy: int=2048, dropout: float=0.1,
        layer_norm_eps: float=1e-05, device=None, dtype=None) ->None:
        kw = {'device': device, 'dtype': dtype}
        super().__init__()
        self.self_attn = NodeEdgeBlock(dx, de, dy, n_head, **kw)
>>>>>>        self.linX1 = torch.nn.linear(dx, dim_ffX, **kw)
>>>>>>        self.linX2 = torch.nn.linear(dim_ffX, dx, **kw)
>>>>>>        self.normX1 = torch.nn.modules.normalization.LayerNorm(dx, eps=
            layer_norm_eps, **kw)
>>>>>>        self.normX2 = torch.nn.modules.normalization.LayerNorm(dx, eps=
            layer_norm_eps, **kw)
>>>>>>        self.dropoutX1 = torch.nn.modules.dropout.Dropout(dropout)
>>>>>>        self.dropoutX2 = torch.nn.modules.dropout.Dropout(dropout)
>>>>>>        self.dropoutX3 = torch.nn.modules.dropout.Dropout(dropout)
>>>>>>        self.linE1 = torch.nn.linear(de, dim_ffE, **kw)
>>>>>>        self.linE2 = torch.nn.linear(dim_ffE, de, **kw)
>>>>>>        self.normE1 = torch.nn.modules.normalization.LayerNorm(de, eps=
            layer_norm_eps, **kw)
>>>>>>        self.normE2 = torch.nn.modules.normalization.LayerNorm(de, eps=
            layer_norm_eps, **kw)
>>>>>>        self.dropoutE1 = torch.nn.modules.dropout.Dropout(dropout)
>>>>>>        self.dropoutE2 = torch.nn.modules.dropout.Dropout(dropout)
>>>>>>        self.dropoutE3 = torch.nn.modules.dropout.Dropout(dropout)
>>>>>>        self.lin_y1 = torch.nn.linear(dy, dim_ffy, **kw)
>>>>>>        self.lin_y2 = torch.nn.linear(dim_ffy, dy, **kw)
>>>>>>        self.norm_y1 = torch.nn.modules.normalization.LayerNorm(dy, eps=
            layer_norm_eps, **kw)
>>>>>>        self.norm_y2 = torch.nn.modules.normalization.LayerNorm(dy, eps=
            layer_norm_eps, **kw)
>>>>>>        self.dropout_y1 = torch.nn.modules.dropout.Dropout(dropout)
>>>>>>        self.dropout_y2 = torch.nn.modules.dropout.Dropout(dropout)
>>>>>>        self.dropout_y3 = torch.nn.modules.dropout.Dropout(dropout)
        self.activation = paddle.nn.functional.relu

    def forward(self, X: paddle.Tensor, E: paddle.Tensor, y, node_mask:
        paddle.Tensor):
        """ Pass the input through the encoder layer.
            X: (bs, n, d)
            E: (bs, n, n, d)
            y: (bs, dy)
            node_mask: (bs, n) Mask for the src keys per batch (optional)
            Output: newX, newE, new_y with the same shape.
        """
        newX, newE, new_y = self.self_attn(X, E, y, node_mask=node_mask)
        newX_d = self.dropoutX1(newX)
        X = self.normX1(X + newX_d)
        newE_d = self.dropoutE1(newE)
        E = self.normE1(E + newE_d)
        new_y_d = self.dropout_y1(new_y)
        y = self.norm_y1(y + new_y_d)
        ff_outputX = self.linX2(self.dropoutX2(self.activation(self.linX1(X))))
        ff_outputX = self.dropoutX3(ff_outputX)
        X = self.normX2(X + ff_outputX)
        ff_outputE = self.linE2(self.dropoutE2(self.activation(self.linE1(E))))
        ff_outputE = self.dropoutE3(ff_outputE)
        E = self.normE2(E + ff_outputE)
        ff_output_y = self.lin_y2(self.dropout_y2(self.activation(self.
            lin_y1(y))))
        ff_output_y = self.dropout_y3(ff_output_y)
        y = self.norm_y2(y + ff_output_y)
        return X, E, y


class NodeEdgeBlock(paddle.nn.Layer):
    """ Self attention layer that also updates the representations on the edges. """

    def __init__(self, dx, de, dy, n_head, **kwargs):
        super().__init__()
        assert dx % n_head == 0, f'dx: {dx} -- nhead: {n_head}'
        self.dx = dx
        self.de = de
        self.dy = dy
        self.df = int(dx / n_head)
        self.n_head = n_head
>>>>>>        self.q = torch.nn.linear(dx, dx)
>>>>>>        self.k = torch.nn.linear(dx, dx)
>>>>>>        self.v = torch.nn.linear(dx, dx)
>>>>>>        self.e_add = torch.nn.linear(de, dx)
>>>>>>        self.e_mul = torch.nn.linear(de, dx)
>>>>>>        self.y_e_mul = torch.nn.linear(dy, dx)
>>>>>>        self.y_e_add = torch.nn.linear(dy, dx)
>>>>>>        self.y_x_mul = torch.nn.linear(dy, dx)
>>>>>>        self.y_x_add = torch.nn.linear(dy, dx)
>>>>>>        self.y_y = torch.nn.linear(dy, dy)
        self.x_y = Xtoy(dx, dy)
        self.e_y = Etoy(de, dy)
>>>>>>        self.x_out = torch.nn.linear(dx, dx)
>>>>>>        self.e_out = torch.nn.linear(dx, de)
        self.y_out = paddle.nn.Sequential(paddle.nn.Linear(in_features=dy,
            out_features=dy), paddle.nn.ReLU(), paddle.nn.Linear(
            in_features=dy, out_features=dy))

    def forward(self, X, E, y, node_mask):
        """
        :param X: bs, n, d        node features
        :param E: bs, n, n, d     edge features
        :param y: bs, dz           global features
        :param node_mask: bs, n
        :return: newX, newE, new_y with the same shape.
        """
        bs, n, _ = tuple(X.shape)
        x_mask = node_mask.unsqueeze(axis=-1)
        e_mask1 = x_mask.unsqueeze(axis=2)
        e_mask2 = x_mask.unsqueeze(axis=1)
        Q = self.q(X) * x_mask
        K = self.k(X) * x_mask
        diffusion_utils.assert_correctly_masked(Q, x_mask)
        Q = Q.reshape((Q.shape[0], Q.shape[1], self.n_head, self.df))
        K = K.reshape((K.shape[0], K.shape[1], self.n_head, self.df))
        Q = Q.unsqueeze(axis=2)
        K = K.unsqueeze(axis=1)
        Y = Q * K
        Y = Y / math.sqrt(Y.shape[-1])
        diffusion_utils.assert_correctly_masked(Y, (e_mask1 * e_mask2).
            unsqueeze(axis=-1))
        E1 = self.e_mul(E) * e_mask1 * e_mask2
        E1 = E1.reshape((E.shape[0], E.shape[1], E.shape[2], self.n_head,
            self.df))
        E2 = self.e_add(E) * e_mask1 * e_mask2
        E2 = E2.reshape((E.shape[0], E.shape[1], E.shape[2], self.n_head,
            self.df))
        Y = Y * (E1 + 1) + E2
        newE = Y.flatten(start_axis=3)
        ye1 = self.y_e_add(y).unsqueeze(axis=1).unsqueeze(axis=1)
        ye2 = self.y_e_mul(y).unsqueeze(axis=1).unsqueeze(axis=1)
        newE = ye1 + (ye2 + 1) * newE
        newE = self.e_out(newE) * e_mask1 * e_mask2
        diffusion_utils.assert_correctly_masked(newE, e_mask1 * e_mask2)
        softmax_mask = e_mask2.expand(shape=[-1, n, -1, self.n_head])
        attn = masked_softmax(Y, softmax_mask, dim=2)
        V = self.v(X) * x_mask
        V = V.reshape((V.shape[0], V.shape[1], self.n_head, self.df))
        V = V.unsqueeze(axis=1)
        weighted_V = attn * V
        weighted_V = weighted_V.sum(axis=2)
        weighted_V = weighted_V.flatten(start_axis=2)
        yx1 = self.y_x_add(y).unsqueeze(axis=1)
        yx2 = self.y_x_mul(y).unsqueeze(axis=1)
        newX = yx1 + (yx2 + 1) * weighted_V
        newX = self.x_out(newX) * x_mask
        diffusion_utils.assert_correctly_masked(newX, x_mask)
        y = self.y_y(y)
        e_y = self.e_y(E, e_mask1, e_mask2)
        x_y = self.x_y(X, x_mask)
        new_y = y + x_y + e_y
        new_y = self.y_out(new_y)
        return newX, newE, new_y


class GraphTransformer(paddle.nn.Layer):
    """
    n_layers : int -- number of layers
    dims : dict -- contains dimensions for each feature type
    """

    def __init__(self, n_layers: int, input_dims: dict, hidden_mlp_dims:
        dict, hidden_dims: dict, output_dims: dict, act_fn_in: paddle.nn.
        ReLU(), act_fn_out: paddle.nn.ReLU()):
        super().__init__()
        self.n_layers = n_layers
        self.out_dim_X = output_dims['X']
        self.out_dim_E = output_dims['E']
        self.out_dim_y = output_dims['y']
        self.mlp_in_X = paddle.nn.Sequential(paddle.nn.Linear(in_features=
            input_dims['X'], out_features=hidden_mlp_dims['X']), act_fn_in,
            paddle.nn.Linear(in_features=hidden_mlp_dims['X'], out_features
            =hidden_dims['dx']), act_fn_in)
        self.mlp_in_E = paddle.nn.Sequential(paddle.nn.Linear(in_features=
            input_dims['E'], out_features=hidden_mlp_dims['E']), act_fn_in,
            paddle.nn.Linear(in_features=hidden_mlp_dims['E'], out_features
            =hidden_dims['de']), act_fn_in)
        self.mlp_in_y = paddle.nn.Sequential(paddle.nn.Linear(in_features=
            input_dims['y'], out_features=hidden_mlp_dims['y']), act_fn_in,
            paddle.nn.Linear(in_features=hidden_mlp_dims['y'], out_features
            =hidden_dims['dy']), act_fn_in)
        self.tf_layers = paddle.nn.LayerList(sublayers=[XEyTransformerLayer
            (dx=hidden_dims['dx'], de=hidden_dims['de'], dy=hidden_dims[
            'dy'], n_head=hidden_dims['n_head'], dim_ffX=hidden_dims[
            'dim_ffX'], dim_ffE=hidden_dims['dim_ffE']) for i in range(
            n_layers)])
        self.mlp_out_X = paddle.nn.Sequential(paddle.nn.Linear(in_features=
            hidden_dims['dx'], out_features=hidden_mlp_dims['X']),
            act_fn_out, paddle.nn.Linear(in_features=hidden_mlp_dims['X'],
            out_features=output_dims['X']))
        self.mlp_out_E = paddle.nn.Sequential(paddle.nn.Linear(in_features=
            hidden_dims['de'], out_features=hidden_mlp_dims['E']),
            act_fn_out, paddle.nn.Linear(in_features=hidden_mlp_dims['E'],
            out_features=output_dims['E']))
        self.mlp_out_y = paddle.nn.Sequential(paddle.nn.Linear(in_features=
            hidden_dims['dy'], out_features=hidden_mlp_dims['y']),
            act_fn_out, paddle.nn.Linear(in_features=hidden_mlp_dims['y'],
            out_features=output_dims['y']))

    def forward(self, X, E, y, node_mask):
        bs, n = tuple(X.shape)[0], tuple(X.shape)[1]
        diag_mask = paddle.eye(num_rows=n)
        diag_mask = ~diag_mask.astype(dtype=E.dtype).astype(dtype='bool')
        diag_mask = diag_mask.unsqueeze(axis=0).unsqueeze(axis=-1).expand(shape
            =[bs, -1, -1, -1])
        X_to_out = X[..., :self.out_dim_X]
        E_to_out = E[..., :self.out_dim_E]
        y_to_out = y[..., :self.out_dim_y]
        new_E = self.mlp_in_E(E)
        new_E = (new_E + new_E.transpose(perm=paddle_aux.transpose_aux_func
            (new_E.ndim, 1, 2))) / 2
        X = self.mlp_in_X(X)
        y = self.mlp_in_y(y)
        after_in = utils.PlaceHolder(X, E=new_E, y=y).mask(node_mask)
        X, E, y = after_in.X, after_in.E, after_in.y
        for layer in self.tf_layers:
            X, E, y = layer(X, E, y, node_mask)
        X = self.mlp_out_X(X)
        E = self.mlp_out_E(E)
        y = self.mlp_out_y(y)
        X = X + X_to_out
        E = (E + E_to_out) * diag_mask
        y = y + y_to_out
        E = 1 / 2 * (E + paddle.transpose(x=E, perm=paddle_aux.
            transpose_aux_func(E.ndim, 1, 2)))
        return utils.PlaceHolder(X=X, E=E, y=y).mask(node_mask)
