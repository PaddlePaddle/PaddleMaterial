from enum import Enum
from typing import List, Optional, Union

import paddle
import paddle.nn.functional as F
from paddle import Tensor


class PoolingStrategy(Enum):
    MEAN = 'mean'
    LAST = 'last'
    CLS = 'cls'


class SentenceTransformer(paddle.nn.Layer):
    def __init__(
        self,
        model_name: str,
        pooling_strategy: Union[PoolingStrategy, str] = 'mean',
    ) -> None:
        super().__init__()

        self.model_name = model_name
        self.pooling_strategy = PoolingStrategy(pooling_strategy)

        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)

        emb = out[0]  # First element contains all token embeddings.
        if self.pooling_strategy == PoolingStrategy.MEAN:
            emb = mean_pooling(emb, attention_mask)
        elif self.pooling_strategy == PoolingStrategy.LAST:
            emb = last_pooling(emb, attention_mask)
        else:
            assert self.pooling_strategy == PoolingStrategy.CLS
            emb = emb[:, 0, :]

        emb = F.normalize(emb, p=2, axis=1)
        return emb

    @property
    def device(self) -> paddle.device:
        return next(iter(self.model.parameters())).place

    @paddle.no_grad()
    def encode(
        self,
        text: List[str],
        batch_size: Optional[int] = None,
        output_device: Optional[Union[str, str]] = None,
    ) -> Tensor:
        is_empty = len(text) == 0
        text = ['dummy'] if is_empty else text

        batch_size = len(text) if batch_size is None else batch_size

        embs: List[Tensor] = []
        for start in range(0, len(text), batch_size):
            token = self.tokenizer(
                text[start:start + batch_size],
                padding=True,
                truncation=True,
                return_tensors='pd',
            )

            emb = self(
                input_ids=token.input_ids.astype(paddle.int64).to(self.device),
                attention_mask=token.attention_mask.to(self.device),
            ).to(output_device)

            embs.append(emb)

        out = paddle.concat(embs, axis=0) if len(embs) > 1 else embs[0]
        out = out[:0] if is_empty else out
        return out

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(model_name={self.model_name})'


def mean_pooling(emb: Tensor, attention_mask: Tensor) -> Tensor:
    mask = attention_mask.unsqueeze(-1).expand(emb.shape).to(emb.dtype)
    return (emb * mask).sum(axis=1) / mask.sum(axis=1).clip(min=1e-9)


def last_pooling(emb: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = paddle.sum(attention_mask[:, -1]) == attention_mask.shape[0]
    if left_padding:
        return emb[:, -1]

    seq_indices = paddle.sum(attention_mask, axis=1) - 1
    return emb[paddle.arange(emb.shape[0], device=emb.device), seq_indices]
