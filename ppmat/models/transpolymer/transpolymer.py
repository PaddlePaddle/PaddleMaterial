from copy import deepcopy

import paddle
import paddle.nn as nn

from ppmat.models.transpolymer.modeling import RobertaConfig
from ppmat.models.transpolymer.modeling import RobertaModel


class TransPolymerRegressor(nn.Layer):
    """TransPolymer encoder with a regression head for polymer property prediction."""

    def __init__(
        self,
        pretrained_model_path=None,
        vocab_size=50265,
        hidden_size=768,
        intermediate_size=3072,
        max_position_embeddings=514,
        num_attention_heads=12,
        num_hidden_layers=6,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        drop_rate=0.1,
        resize_vocab_size=None,
    ):
        super().__init__()
        if pretrained_model_path:
            encoder = RobertaModel.from_pretrained(pretrained_model_path)
        else:
            config = RobertaConfig(
                vocab_size=vocab_size,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                max_position_embeddings=max_position_embeddings,
                num_attention_heads=num_attention_heads,
                num_hidden_layers=num_hidden_layers,
                hidden_dropout_prob=hidden_dropout_prob,
                attention_probs_dropout_prob=attention_probs_dropout_prob,
            )
            encoder = RobertaModel(config=config)

        encoder.config.hidden_dropout_prob = hidden_dropout_prob
        encoder.config.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.encoder = deepcopy(encoder)
        if resize_vocab_size is not None:
            self.encoder.resize_token_embeddings(resize_vocab_size)

        hidden_size = self.encoder.config.hidden_size
        self.regressor = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.Silu(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return self.regressor(cls_embedding)
