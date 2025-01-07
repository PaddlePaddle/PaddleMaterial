import paddle
from .transformer_model import GraphTransformer
from .transformer_c_model import GraphTransformer_C


class molecularGT(paddle.nn.Layer):

    def __init__(
        self, 
        n_layers_GT: int, 
        input_dims: dict, 
        hidden_mlp_dims:dict, 
        hidden_dims: dict, 
        output_dims: dict, 
        act_fn_in: paddle.nn.ReLU(), 
        act_fn_out: paddle.nn.ReLU()
    ):
        super().__init__()
        self.GT = GraphTransformer(
            n_layers=n_layers_GT, 
            input_dims=input_dims, 
            hidden_mlp_dims=hidden_mlp_dims, 
            hidden_dims=hidden_dims, 
            output_dims=output_dims, 
            act_fn_in=act_fn_in,
            act_fn_out=act_fn_out
        )
        self.con_input_dim = input_dims
        self.con_input_dim['X'] = input_dims['X'] - 8
        self.con_input_dim['y'] = 1024
        self.con_output_dim = output_dims
        self.conditionEn = GraphTransformer_C(
            n_layers=n_layers_GT,
            input_dims=self.con_input_dim, 
            hidden_mlp_dims=hidden_mlp_dims,
            hidden_dims=hidden_dims, 
            output_dims=self.con_output_dim,
            act_fn_in=act_fn_in, 
            act_fn_out=act_fn_out
        )

    def forward(self, X, E, y, node_mask, X_condition, E_condtion):
        y_condition = paddle.zeros(shape=[X.shape[0], 1024]).cuda(blocking=True)
        conditionVec = self.conditionEn(X_condition, E_condtion,y_condition, node_mask)
        y = paddle.hstack(x=(y, conditionVec)).astype(dtype='float32')
        return self.GT(X, E, y, node_mask)
