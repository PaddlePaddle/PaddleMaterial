import paddle

from ppmat.models.transpolymer.modeling import RobertaConfig
from ppmat.models.transpolymer.modeling import RobertaModel
from ppmat.models.transpolymer.transpolymer import TransPolymerRegressor


def test_transpolymer_model_forward():
    config = RobertaConfig(
        vocab_size=128,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=32,
        type_vocab_size=1,
    )
    model = RobertaModel(config)
    input_ids = paddle.randint(0, 128, shape=[2, 16], dtype="int64")
    attention_mask = paddle.ones([2, 16], dtype="int64")
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    assert list(outputs.last_hidden_state.shape) == [2, 16, 32]


def test_transpolymer_regressor_forward():
    model = TransPolymerRegressor(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        max_position_embeddings=32,
        num_attention_heads=4,
        num_hidden_layers=1,
        resize_vocab_size=128,
    )
    input_ids = paddle.randint(0, 128, shape=[2, 16], dtype="int64")
    attention_mask = paddle.ones([2, 16], dtype="int64")
    pred = model(input_ids, attention_mask)
    assert list(pred.shape) == [2, 1]
