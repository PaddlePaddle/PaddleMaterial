import sys
sys.path.append('/home/liuxuwei01/SpectrumMol/src_paddle/utils')
import paddle_aux
import paddle
from .transformer_model import GraphTransformer
from .transformer_c_model import GraphTransformer_C
from src.models.model.encoder import Encoder


class contrastGT(paddle.nn.Layer):

    def __init__(self, n_layers_GT: int, input_dims: dict, hidden_mlp_dims:
        dict, hidden_dims: dict, output_dims: dict, act_fn_in: paddle.nn.
        ReLU(), act_fn_out: paddle.nn.ReLU(), enc_voc_size, max_len,
        d_model, ffn_hidden, n_head, n_layers_TE, drop_prob, device):
        super().__init__()
        self.transEn = Encoder(enc_voc_size=enc_voc_size, max_len=max_len,
            d_model=d_model, ffn_hidden=ffn_hidden, n_head=n_head, n_layers
            =n_layers_TE, drop_prob=drop_prob, device=device)
        self.linear_layer = paddle.nn.Linear(in_features=max_len * d_model,
            out_features=512)
        self.con_input_dim = input_dims
        self.con_input_dim['X'] = input_dims['X'] - 8
        self.con_input_dim['y'] = 1024
        self.con_output_dim = output_dims
        self.conditionEn = GraphTransformer_C(n_layers=n_layers_GT,
            input_dims=self.con_input_dim, hidden_mlp_dims=hidden_mlp_dims,
            hidden_dims=hidden_dims, output_dims=self.con_output_dim,
            act_fn_in=act_fn_in, act_fn_out=act_fn_out)
        self.device = device
        checkpoint = paddle.load(path=str(
            '/home/liuxuwei01/molecular2molecular/src/epoch-438.ckpt'))
        state_dict = checkpoint['state_dict']
        print(state_dict.keys())
        conditionEn_state_dict = {k[len('model.conditionEn.'):]: v for k, v in
            state_dict.items() if k.startswith('model.conditionEn.')}
        self.conditionEn.set_state_dict(state_dict=conditionEn_state_dict)
        print('conditionEn parameters loaded successfully.')
        for param in self.conditionEn.parameters():
            param.stop_gradient = not False
        self.conditionEn.eval()

    def make_src_mask(self, src):
        src_mask = (src != 0).unsqueeze(axis=1).unsqueeze(axis=2)
        return src_mask

    def forward(self, X, E, y, node_mask, X_condition, E_condtion, conditionVec
        ):
        assert isinstance(conditionVec, paddle.Tensor
            ), 'conditionVec should be a tensor, but got type {}'.format(type
            (conditionVec))
        srcMask = self.make_src_mask(conditionVec).to(self.device)
        conditionVecNmr = self.transEn(conditionVec, srcMask)
        conditionVecNmr = conditionVecNmr.view(conditionVecNmr.shape[0], -1)
        conditionVecNmr = self.linear_layer(conditionVecNmr)
        y_condition = paddle.zeros(shape=[X_condition.shape[0], 1024]).cuda(
            blocking=True)
        conditionVecM = self.conditionEn(X_condition, E_condtion,
            y_condition, node_mask)
        return conditionVecM, conditionVecNmr
