from copy import deepcopy

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import pandas as pd

from ppmat.models.transpolymer.modeling import RobertaConfig
from ppmat.models.transpolymer.modeling import RobertaModel
from ppmat.models.transpolymer.tokenizer import PolymerSmilesTokenizer


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
        tokenizer_name_or_path="roberta-base",
        vocab_sup_file=None,
        blocksize=411,
        smiles_key="smiles",
        use_token_cache=True,
        property_name="Conductivity [S/cm]",
        data_mean=0.0,
        data_std=1.0,
        loss_type="mse_loss",
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
        self.tokenizer = PolymerSmilesTokenizer.from_pretrained(
            tokenizer_name_or_path, max_len=blocksize
        )
        if vocab_sup_file is not None:
            vocab_sup = pd.read_csv(vocab_sup_file, header=None).values.flatten()
            self.tokenizer.add_tokens(vocab_sup.tolist())
        if resize_vocab_size is None:
            resize_vocab_size = len(self.tokenizer)
        if resize_vocab_size is not None:
            self.encoder.resize_token_embeddings(resize_vocab_size)
        self.blocksize = blocksize
        self.smiles_key = smiles_key
        self.use_token_cache = use_token_cache
        self._token_cache = {}
        if isinstance(property_name, list):
            self.property_name = property_name[0]
        else:
            self.property_name = property_name
        self.register_buffer(tensor=paddle.to_tensor(data_mean), name="data_mean")
        self.register_buffer(tensor=paddle.to_tensor(data_std), name="data_std")
        if loss_type == "mse_loss":
            self.loss_fn = F.mse_loss
        elif loss_type == "l1_loss":
            self.loss_fn = F.l1_loss
        else:
            raise ValueError(f"Unknown loss type {loss_type}.")

        hidden_size = self.encoder.config.hidden_size
        self.regressor = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.Silu(),
            nn.Linear(hidden_size, 1),
        )

    def normalize(self, tensor):
        return (tensor - self.data_mean) / self.data_std

    def unnormalize(self, tensor):
        return tensor * self.data_std + self.data_mean

    def _forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return self.regressor(cls_embedding)

    def _encode_smiles(self, smiles):
        if self.use_token_cache and smiles in self._token_cache:
            return self._token_cache[smiles]

        encoding = self.tokenizer(
            smiles,
            add_special_tokens=True,
            max_length=self.blocksize,
            return_token_type_ids=False,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
        )
        if self.use_token_cache:
            self._token_cache[smiles] = encoding
        return encoding

    def _build_model_inputs(self, data):
        if "input_ids" in data and "attention_mask" in data:
            return data["input_ids"], data["attention_mask"]

        smiles_batch = data[self.smiles_key]
        if isinstance(smiles_batch, str):
            smiles_batch = [smiles_batch]
        encodings = [self._encode_smiles(str(smiles)) for smiles in smiles_batch]
        input_ids = paddle.to_tensor(
            [encoding["input_ids"] for encoding in encodings], dtype="int64"
        )
        attention_mask = paddle.to_tensor(
            [encoding["attention_mask"] for encoding in encodings], dtype="int64"
        )
        return input_ids, attention_mask

    def forward(
        self,
        data,
        attention_mask=None,
        return_loss=True,
        return_prediction=True,
    ):
        if not isinstance(data, dict):
            return self._forward(data, attention_mask)

        assert (
            return_loss or return_prediction
        ), "At least one of return_loss or return_prediction must be True."
        input_ids, attention_mask = self._build_model_inputs(data)
        pred = self._forward(input_ids, attention_mask)

        loss_dict = {}
        if return_loss:
            label = self.normalize(data[self.property_name])
            loss_dict["loss"] = self.loss_fn(input=pred, label=label)

        pred_dict = {}
        if return_prediction:
            pred_dict[self.property_name] = self.unnormalize(pred)

        return {"loss_dict": loss_dict, "pred_dict": pred_dict}

    def predict(self, data):
        if isinstance(data, str):
            data = {self.smiles_key: [data]}
        elif isinstance(data, (list, tuple)):
            data = {self.smiles_key: list(data)}

        output = self.forward(data, return_loss=False, return_prediction=True)
        return output["pred_dict"]
