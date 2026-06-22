import json
import math
import os
from dataclasses import asdict, dataclass
from typing import Optional, Tuple

import paddle
import paddle.nn as nn
import paddle.nn.functional as F


@dataclass
class RobertaConfig:
    vocab_size: int = 50265
    hidden_size: int = 768
    num_hidden_layers: int = 6
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 514
    type_vocab_size: int = 1
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    pad_token_id: int = 1
    bos_token_id: int = 0
    eos_token_id: int = 2
    position_embedding_type: str = "absolute"

    @classmethod
    def from_dict(cls, values):
        fields = cls.__dataclass_fields__
        return cls(**{k: v for k, v in values.items() if k in fields})

    @classmethod
    def from_pretrained(cls, path):
        with open(os.path.join(path, "config.json"), "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

    def to_dict(self):
        result = asdict(self)
        result["model_type"] = "roberta"
        return result

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        with open(os.path.join(save_directory, "config.json"), "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)


class ModelOutput:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __getitem__(self, item):
        values = tuple(v for v in self.__dict__.values() if v is not None)
        return values[item]

    def __iter__(self):
        return iter(tuple(v for v in self.__dict__.values() if v is not None))


def _normal_init(layer, config):
    if isinstance(layer, nn.Linear):
        nn.initializer.Normal(mean=0.0, std=config.initializer_range)(layer.weight)
        if layer.bias is not None:
            nn.initializer.Constant(0.0)(layer.bias)
    elif isinstance(layer, nn.Embedding):
        nn.initializer.Normal(mean=0.0, std=config.initializer_range)(layer.weight)
        padding_idx = getattr(layer, "_padding_idx", None)
        if padding_idx is not None:
            with paddle.no_grad():
                weight = layer.weight.numpy()
                weight[padding_idx] = 0
                layer.weight.set_value(weight)
    elif isinstance(layer, nn.LayerNorm):
        nn.initializer.Constant(1.0)(layer.weight)
        nn.initializer.Constant(0.0)(layer.bias)


def init_weights(layer, config):
    for sublayer in [layer] + list(layer.sublayers()):
        _normal_init(sublayer, config)


def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
    mask = paddle.cast(input_ids != padding_idx, "int64")
    incremental_indices = (paddle.cumsum(mask, axis=1) + past_key_values_length) * mask
    return incremental_indices + padding_idx


class RobertaEmbeddings(nn.Layer):
    def __init__(self, config):
        super().__init__()
        # Paddle masks padding_idx outputs to zero at runtime, while HuggingFace
        # PyTorch returns the loaded padding row. Keep runtime behavior aligned
        # with the source checkpoint and handle padding only in the attention mask.
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size, padding_idx=config.pad_token_id)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.padding_idx = config.pad_token_id
        self.position_embedding_type = config.position_embedding_type

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        if position_ids is None:
            if input_ids is not None:
                position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx)
            else:
                input_shape = inputs_embeds.shape[:-1]
                seq_length = input_shape[1]
                position_ids = paddle.arange(self.padding_idx + 1, seq_length + self.padding_idx + 1, dtype="int64")
                position_ids = position_ids.unsqueeze(0).expand(input_shape)

        if input_ids is not None:
            input_shape = input_ids.shape
        else:
            input_shape = inputs_embeds.shape[:-1]

        if token_type_ids is None:
            token_type_ids = paddle.zeros(input_shape, dtype="int64")
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        embeddings = inputs_embeds + self.token_type_embeddings(token_type_ids)
        if self.position_embedding_type == "absolute":
            embeddings = embeddings + self.position_embeddings(position_ids)
        embeddings = self.LayerNorm(embeddings)
        return self.dropout(embeddings)


class RobertaSelfAttention(nn.Layer):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_shape = list(x.shape[:-1]) + [self.num_attention_heads, self.attention_head_size]
        x = x.reshape(new_shape)
        return x.transpose([0, 2, 1, 3])

    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        attention_scores = paddle.matmul(query_layer, key_layer, transpose_y=True)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        attention_probs = F.softmax(attention_scores, axis=-1)
        attention_probs = self.dropout(attention_probs)

        context_layer = paddle.matmul(attention_probs, value_layer)
        context_layer = context_layer.transpose([0, 2, 1, 3])
        context_layer = context_layer.reshape([context_layer.shape[0], context_layer.shape[1], self.all_head_size])
        return (context_layer, attention_probs) if output_attentions else (context_layer,)


class RobertaSelfOutput(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return self.LayerNorm(hidden_states + input_tensor)


class RobertaAttention(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.self = RobertaSelfAttention(config)
        self.output = RobertaSelfOutput(config)

    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        self_outputs = self.self(hidden_states, attention_mask, output_attentions)
        attention_output = self.output(self_outputs[0], hidden_states)
        return (attention_output,) + self_outputs[1:]


class RobertaIntermediate(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = F.gelu

    def forward(self, hidden_states):
        return self.intermediate_act_fn(self.dense(hidden_states))


class RobertaOutput(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return self.LayerNorm(hidden_states + input_tensor)


class RobertaLayer(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.attention = RobertaAttention(config)
        self.intermediate = RobertaIntermediate(config)
        self.output = RobertaOutput(config)

    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        self_attention_outputs = self.attention(hidden_states, attention_mask, output_attentions)
        attention_output = self_attention_outputs[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return (layer_output,) + self_attention_outputs[1:]


class RobertaEncoder(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.layer = nn.LayerList([RobertaLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask=None, output_attentions=False, output_hidden_states=False):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        for layer_module in self.layer:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_outputs = layer_module(hidden_states, attention_mask, output_attentions)
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        return ModelOutput(last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_self_attentions)


class RobertaPooler(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        return self.activation(self.dense(hidden_states[:, 0]))


class RobertaModel(nn.Layer):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__()
        self.config = config
        self.embeddings = RobertaEmbeddings(config)
        self.encoder = RobertaEncoder(config)
        self.pooler = RobertaPooler(config) if add_pooling_layer else None
        init_weights(self, config)

    @classmethod
    def from_pretrained(cls, path, add_pooling_layer=True):
        config = RobertaConfig.from_pretrained(path)
        model = cls(config, add_pooling_layer=add_pooling_layer)
        state_path = os.path.join(path, "model_state.pdparams")
        if os.path.exists(state_path):
            raw_state = paddle.load(state_path)
            current_state = model.state_dict()
            loaded_state = {}
            for key, value in raw_state.items():
                model_key = key[len("roberta.") :] if key.startswith("roberta.") else key
                if model_key in current_state and list(current_state[model_key].shape) == list(value.shape):
                    loaded_state[model_key] = value
            current_state.update(loaded_state)
            model.set_state_dict(current_state)
            print(f"Loaded {len(loaded_state)} RobertaModel tensors from {state_path}")
        return model

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def resize_token_embeddings(self, new_num_tokens):
        old_embeddings = self.embeddings.word_embeddings
        old_num_tokens, embedding_dim = old_embeddings.weight.shape
        if new_num_tokens == old_num_tokens:
            return old_embeddings
        new_embeddings = nn.Embedding(new_num_tokens, embedding_dim)
        _normal_init(new_embeddings, self.config)
        num_to_copy = min(old_num_tokens, new_num_tokens)
        with paddle.no_grad():
            weight = new_embeddings.weight.numpy()
            weight[:num_to_copy] = old_embeddings.weight.numpy()[:num_to_copy]
            new_embeddings.weight.set_value(weight)
        self.embeddings.word_embeddings = new_embeddings
        self.config.vocab_size = new_num_tokens
        return new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        output_attentions=False,
        output_hidden_states=False,
    ):
        if input_ids is not None:
            input_shape = input_ids.shape
        else:
            input_shape = inputs_embeds.shape[:-1]
        if attention_mask is None:
            attention_mask = paddle.ones(input_shape, dtype="int64")

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = paddle.cast(extended_attention_mask, paddle.get_default_dtype())
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids=input_ids, token_type_ids=token_type_ids, position_ids=position_ids, inputs_embeds=inputs_embeds)
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        pooled_output = self.pooler(encoder_outputs.last_hidden_state) if self.pooler is not None else None
        return ModelOutput(
            last_hidden_state=encoder_outputs.last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        self.config.save_pretrained(save_directory)
        paddle.save(self.state_dict(), os.path.join(save_directory, "model_state.pdparams"))


class RobertaLMHead(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, epsilon=config.layer_norm_eps)
        self.bias = self.create_parameter(
            shape=[config.vocab_size],
            dtype=paddle.get_default_dtype(),
            default_initializer=nn.initializer.Constant(0.0),
        )

    def forward(self, features, decoder_weight):
        x = self.dense(features)
        x = F.gelu(x)
        x = self.layer_norm(x)
        return paddle.matmul(x, decoder_weight, transpose_y=True) + self.bias


class RobertaForMaskedLM(nn.Layer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.lm_head = RobertaLMHead(config)
        init_weights(self.lm_head, config)

    @classmethod
    def from_pretrained(cls, path):
        config = RobertaConfig.from_pretrained(path)
        model = cls(config)
        state_path = os.path.join(path, "model_state.pdparams")
        if os.path.exists(state_path):
            raw_state = paddle.load(state_path)
            current_state = model.state_dict()
            loaded_state = {
                key: value
                for key, value in raw_state.items()
                if key in current_state and list(current_state[key].shape) == list(value.shape)
            }
            current_state.update(loaded_state)
            model.set_state_dict(current_state)
            print(f"Loaded {len(loaded_state)} RobertaForMaskedLM tensors from {state_path}")
        return model

    def resize_token_embeddings(self, new_num_tokens):
        self.roberta.resize_token_embeddings(new_num_tokens)
        old_bias = self.lm_head.bias
        old_num_tokens = old_bias.shape[0]
        if new_num_tokens == old_num_tokens:
            return self.roberta.get_input_embeddings()
        new_bias = self.lm_head.create_parameter(
            shape=[new_num_tokens],
            dtype=paddle.get_default_dtype(),
            default_initializer=nn.initializer.Constant(0.0),
        )
        num_to_copy = min(old_num_tokens, new_num_tokens)
        with paddle.no_grad():
            bias = new_bias.numpy()
            bias[:num_to_copy] = old_bias.numpy()[:num_to_copy]
            new_bias.set_value(bias)
        self.lm_head.bias = new_bias
        self.config.vocab_size = new_num_tokens
        return self.roberta.get_input_embeddings()

    def forward(self, input_ids=None, attention_mask=None, labels=None, output_attentions=False, output_hidden_states=False):
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        logits = self.lm_head(outputs.last_hidden_state, self.roberta.embeddings.word_embeddings.weight)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.reshape([-1, self.config.vocab_size]), labels.reshape([-1]), ignore_index=-100)
        return ModelOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        self.config.save_pretrained(save_directory)
        paddle.save(self.state_dict(), os.path.join(save_directory, "model_state.pdparams"))
