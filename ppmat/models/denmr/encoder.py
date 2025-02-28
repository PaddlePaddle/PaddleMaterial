import paddle

from .blocks.encoder_layer import EncoderLayer
from .embedding.transformer_embedding import TransformerEmbedding

"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""


class Encoder(paddle.nn.Layer):
    def __init__(
        self,
        enc_voc_size,
        max_len,
        d_model,
        ffn_hidden,
        n_head,
        n_layers,
        drop_prob,
    ):
        super().__init__()
        self.emb = TransformerEmbedding(
            d_model=d_model,
            max_len=max_len,
            vocab_size=enc_voc_size,
            drop_prob=drop_prob,
        )
        self.layers = paddle.nn.LayerList(
            sublayers=[
                EncoderLayer(
                    d_model=d_model,
                    ffn_hidden=ffn_hidden,
                    n_head=n_head,
                    drop_prob=drop_prob,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, x, src_mask):
        x = self.emb(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        return x
