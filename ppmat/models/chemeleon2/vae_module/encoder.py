import math
import paddle
import paddle.nn as nn
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


class TransformerEncoder(nn.Layer):
    def __init__(
        self,
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
        self.atom_type_embedder = nn.Embedding(max_num_elements, d_model)
        self.lattices_embedder = nn.Sequential(
            nn.Linear(9, d_model, bias_attr=False),
            nn.Silu(),
            nn.Linear(d_model, d_model),
        )
        self.frac_coords_embedder = nn.Sequential(
            nn.Linear(3, d_model, bias_attr=False),
            nn.Silu(),
            nn.Linear(d_model, d_model),
        )

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

    @property
    def hidden_dim(self):
        return self.d_model

    def forward(self, batch):
        atom_types = batch.atom_types
        lattices = batch.lattices
        frac_coords = batch.frac_coords
        token_idx = batch.token_idx
        batch_idx = batch.batch
        num_atoms = batch.num_atoms

        x = self.atom_type_embedder(atom_types)
        x += self.lattices_embedder(lattices.reshape([-1, 9]))[batch_idx]
        x += self.frac_coords_embedder(frac_coords)

        x += get_index_embedding(token_idx, self.d_model)

        x_dense, token_mask = to_dense_batch(x, batch_idx)

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

        return {
            "x": x,
            "num_atoms": num_atoms,
            "batch": batch_idx,
            "token_idx": token_idx,
        }
