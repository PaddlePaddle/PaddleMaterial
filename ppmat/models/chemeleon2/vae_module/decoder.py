import math
import paddle
import paddle.nn as nn

from ppmat.models.chemeleon2.common.scatter import scatter_mean
from ..common.batch_utils import to_dense_batch


def get_index_embedding(indices, emb_dim, max_len=2048):
    K = paddle.arange(emb_dim // 2)
    pos_embedding_sin = paddle.sin(
        indices.unsqueeze(-1) * math.pi / (max_len ** (2 * K / emb_dim))
    )
    pos_embedding_cos = paddle.cos(
        indices.unsqueeze(-1) * math.pi / (max_len ** (2 * K / emb_dim))
    )
    pos_embedding = paddle.concat([pos_embedding_sin, pos_embedding_cos], axis=-1)
    return pos_embedding


class TransformerDecoder(nn.Layer):
    def __init__(
        self,
        atom_type_predict=True,
        max_num_elements=100,
        d_model=1024,
        nhead=8,
        dim_feedforward=2048,
        activation="gelu",
        dropout=0.0,
        norm_first=True,
        bias=True,
        num_layers=6,
    ):
        super().__init__()

        self.max_num_elements = max_num_elements
        self.d_model = d_model
        self.num_layers = num_layers
        self.atom_type_predict = atom_type_predict

        if activation == "gelu":
            act_fn = nn.GELU(approximate='tanh')
        elif activation == "relu":
            act_fn = nn.ReLU()
        else:
            act_fn = nn.GELU(approximate='tanh')

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            normalize_before=norm_first,
        )
        
        layer_norm = nn.LayerNorm(d_model)
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=layer_norm,
        )
        
        for layer in self.transformer.layers:
            layer.activation = act_fn

        if atom_type_predict:
            self.atom_types_head = nn.Linear(d_model, max_num_elements, bias_attr=True)
        self.frac_coords_head = nn.Linear(d_model, 3, bias_attr=False)
        self.lattice_head = nn.Linear(d_model, 6, bias_attr=False)

    @property
    def hidden_dim(self):
        return self.d_model

    def forward(self, encoded_batch):
        x = encoded_batch["x"]

        x += get_index_embedding(encoded_batch["token_idx"], self.d_model)

        x_dense, token_mask = to_dense_batch(x, encoded_batch["batch"])

        # Check if there is any padding (mask not all True)
        has_padding = not token_mask.cast('bool').all()

        if has_padding:
            # Create 4D attention mask [batch_size, num_heads, seq_len, seq_len]
            # Use additive mask (float with -inf for masked positions)
            batch_size, seq_len, _ = x_dense.shape
            num_heads = 8  # Must match nhead in TransformerEncoderLayer
            attn_mask = paddle.zeros([batch_size, num_heads, seq_len, seq_len], dtype='float32')
            for b in range(batch_size):
                for h in range(num_heads):
                    for j in range(seq_len):
                        if not token_mask[b, j]:  # padding position
                            attn_mask[b, h, :, j] = -1e9
        else:
            # No padding, use None to avoid triggering different code path
            attn_mask = None

        x_out = self.transformer(x_dense, src_mask=attn_mask)

        x = x_out[token_mask]

        x_global = scatter_mean(x, encoded_batch["batch"], dim=0)

        if self.atom_type_predict:
            atom_types_out = self.atom_types_head(x)
        else:
            atom_types_out = None

        lattices_out = self.lattice_head(x_global)

        frac_coords_out = self.frac_coords_head(x)

        result = {
            "atom_types": atom_types_out,
            "lattices": lattices_out,
            "lengths": lattices_out[:, :3],
            "angles": lattices_out[:, 3:],
            "frac_coords": frac_coords_out,
        }
        return result
