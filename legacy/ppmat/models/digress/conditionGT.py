import sys
sys.path.append('/home/liuxuwei01/SpectrumMol/src_paddle/utils')
import paddle_aux
import paddle
from .transformer_model import GraphTransformer
from src.models.model.encoder import Encoder


class ConditionGT(paddle.nn.Layer):

    def __init__(self, n_layers_GT: int, input_dims: dict, hidden_mlp_dims:
        dict, hidden_dims: dict, output_dims: dict, act_fn_in: paddle.nn.
        ReLU(), act_fn_out: paddle.nn.ReLU(), enc_voc_size, max_len,
        d_model, ffn_hidden, n_head, n_layers_TE, drop_prob, device):
        super().__init__()
        self.GT = GraphTransformer(n_layers=n_layers_GT, input_dims=
            input_dims, hidden_mlp_dims=hidden_mlp_dims, hidden_dims=
            hidden_dims, output_dims=output_dims, act_fn_in=act_fn_in,
            act_fn_out=act_fn_out)
        self.transEn = Encoder(enc_voc_size=enc_voc_size, max_len=max_len,
            d_model=d_model, ffn_hidden=ffn_hidden, n_head=n_head, n_layers
            =n_layers_TE, drop_prob=drop_prob, device=device)
        self.linear_layer = paddle.nn.Linear(in_features=max_len * d_model,
            out_features=512)
        self.device = device
        checkpoint = paddle.load(path=str(
            '/home/liuxuwei01/molecular2molecular/src/epoch-438.ckpt'))
        state_dict = checkpoint['state_dict']
        GT_state_dict = {k[len('model.GT.'):]: v for k, v in state_dict.
            items() if k.startswith('model.GT.')}
        self.GT.set_state_dict(state_dict=GT_state_dict)
        checkpoint = paddle.load(path=str(
            '/home/liuxuwei01/molecular2molecular/src/epoch-35.ckpt'))
        state_dict = checkpoint['state_dict']
        linear_layer_state_dict = {k[len('model.linear_layer.'):]: v for k,
            v in state_dict.items() if k.startswith('model.linear_layer.')}
        self.linear_layer.set_state_dict(state_dict=linear_layer_state_dict)
        checkpoint = paddle.load(path=str(
            '/home/liuxuwei01/molecular2molecular/src/epoch-35.ckpt'))
        state_dict = checkpoint['state_dict']
        transEn_state_dict = {k[len('model.transEn.'):]: v for k, v in
            state_dict.items() if k.startswith('model.transEn.')}
        self.transEn.set_state_dict(state_dict=transEn_state_dict)

    def make_src_mask(self, src):
        src_mask = (src != 0).unsqueeze(axis=1).unsqueeze(axis=2)
        return src_mask

    def forward(self, X, E, y, node_mask, conditionVec):
        assert isinstance(conditionVec, paddle.Tensor
            ), 'conditionVec should be a tensor, but got type {}'.format(type
            (conditionVec))
        srcMask = self.make_src_mask(conditionVec).to(self.device)
        conditionVec = self.transEn(conditionVec, srcMask)
        conditionVec = conditionVec.view(conditionVec.shape[0], -1)
        conditionVec = self.linear_layer(conditionVec)
        y = paddle.hstack(x=(y, conditionVec)).astype(dtype='float32')
        return self.GT(X, E, y, node_mask)
