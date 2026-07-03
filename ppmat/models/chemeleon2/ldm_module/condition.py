# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from enum import Enum
import paddle
import paddle.nn as nn


class ConditionType(Enum):
    COMPOSITION = "composition"
    CHEMICAL_SYSTEM = "chemical_system"
    VALUE = "float"
    CATEGORICAL = "categorical"


class ConditionModule(nn.Layer):
    def __init__(
        self,
        condition_type,
        hidden_dim,
        drop_prob,
        stats=None,
        **kwargs,
    ):
        super().__init__()
        self.condition_type = condition_type
        self.target_condition = list(condition_type.keys())
        self.hidden_dim = hidden_dim
        self.drop_prob = drop_prob
        self.stats = stats if stats is not None else {}

        self.encoders = nn.LayerDict()
        for cond_name, cond_type in condition_type.items():
            if cond_type == ConditionType.COMPOSITION.value:
                self.encoders[cond_name] = CompositionEncoder(
                    in_dim=100,
                    hidden_dim=hidden_dim,
                )
            elif cond_type == ConditionType.CHEMICAL_SYSTEM.value:
                self.encoders[cond_name] = ChemicalSystemEncoder(
                    in_dim=100,
                    hidden_dim=hidden_dim,
                )
            elif cond_type == ConditionType.VALUE.value:
                _stats = self.stats.get(cond_name, {})
                self.encoders[cond_name] = ValueEncoder(
                    hidden_dim=hidden_dim,
                    mean=_stats.get("mean", None),
                    std=_stats.get("std", None),
                )
            elif cond_type == ConditionType.CATEGORICAL.value:
                assert "num_classes" in kwargs, (
                    "num_classes must be provided when using CLASS condition type"
                )
                self.encoders[cond_name] = CategoricalEncoder(
                    in_dim=kwargs["num_classes"],
                    hidden_dim=hidden_dim,
                )
            else:
                raise ValueError(f"Unsupported condition type: {cond_type}")

        self.proj = nn.Sequential(
            nn.Linear(len(self.encoders) * hidden_dim, hidden_dim, bias_attr=True),
            nn.Silu(),
            nn.Linear(hidden_dim, hidden_dim, bias_attr=True),
        )

    def forward(self, batch_y, training=True):
        target_conditions = list(batch_y.keys())
        assert set(target_conditions) == set(self.target_condition), (
            f"Expected conditions {self.target_condition}, but got {target_conditions}"
        )

        batch_size = len(list(batch_y.values())[0])
        if training:
            assert self.drop_prob >= 0
            drop_mask = paddle.rand([batch_size]) < self.drop_prob
        else:
            drop_mask = paddle.concat([
                paddle.zeros([batch_size]),
                paddle.ones([batch_size])
            ])
            drop_mask = drop_mask.astype('bool')
            batch_y = {k: _duplicate(v) for k, v in batch_y.items()}

        cond_embeds = []
        for cond_name, encoder in self.encoders.items():
            y = batch_y[cond_name]
            embed = encoder(y, drop_mask)
            cond_embeds.append(embed)

        cond_embeds = paddle.concat(cond_embeds, axis=-1)
        cond_embeds = self.proj(cond_embeds)
        return cond_embeds


def _duplicate(v):
    if isinstance(v, paddle.Tensor):
        return paddle.concat([v, v])
    elif isinstance(v, list):
        return v * 2
    else:
        raise ValueError(f"Unsupported type: {type(v)}")


class BaseEncoder(nn.Layer):
    def __init__(
        self,
        *,
        in_dim,
        hidden_dim,
        preprocess,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim

        self.preprocess = preprocess
        self.embedding = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias_attr=True),
            nn.Silu(),
            nn.Linear(hidden_dim, hidden_dim, bias_attr=True),
        )
        self.null_embed = paddle.create_parameter(
            shape=[1, in_dim],
            dtype='float32',
            default_initializer=nn.initializer.Normal()
        )

    def forward(self, y, drop_mask=None):
        y = self.preprocess(y)
        if drop_mask is not None:
            y = paddle.where(
                drop_mask.unsqueeze(-1).expand_as(y),
                self.null_embed.expand_as(y),
                y
            )
        return self.embedding(y)


class ValueEncoder(BaseEncoder):
    def __init__(self, hidden_dim, mean=None, std=None):
        self.mean = mean
        self.std = std
        
        def preprocess(batch):
            if isinstance(batch, list):
                batch = paddle.to_tensor(batch, dtype='float32')
            elif not isinstance(batch, paddle.Tensor):
                batch = paddle.to_tensor(batch, dtype='float32')
            
            batch = batch.astype('float32').unsqueeze(-1)
            if self.mean is not None and self.std is not None:
                batch = (batch - self.mean) / self.std
            return batch

        super().__init__(
            in_dim=1,
            hidden_dim=hidden_dim,
            preprocess=preprocess,
        )


class CategoricalEncoder(BaseEncoder):
    def __init__(self, in_dim, hidden_dim):
        self.num_classes = in_dim
        
        def preprocess(batch):
            if isinstance(batch, list):
                idx = paddle.to_tensor(batch, dtype='int64')
            elif not isinstance(batch, paddle.Tensor):
                idx = paddle.to_tensor(batch, dtype='int64')
            else:
                idx = batch.astype('int64')
            
            return paddle.nn.functional.one_hot(idx, num_classes=self.num_classes).astype('float32')

        super().__init__(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            preprocess=preprocess,
        )


class _ElementEncoder(BaseEncoder):
    def __init__(self, in_dim, hidden_dim, fill_fn):
        self._in_dim = in_dim
        self._fill = fill_fn
        def preprocess(batch):
            vals = [self._to_embeds(s) for s in batch]
            return paddle.concat(vals, axis=0)
        super().__init__(in_dim=in_dim, hidden_dim=hidden_dim, preprocess=preprocess)

    def _to_embeds(self, s):
        from pymatgen.core import Element
        v = paddle.zeros([self._in_dim])
        self._fill(v, s, Element)
        return v.unsqueeze(0)


class CompositionEncoder(_ElementEncoder):
    def __init__(self, in_dim, hidden_dim):
        def fill(v, comp_str, Element):
            from pymatgen.core import Composition
            comp = Composition(comp_str).reduced_composition
            for el, amt in comp.get_el_amt_dict().items():
                v[Element(el).Z] = float(amt)
            v = v / v.sum() if v.sum() > 0 else v
        super().__init__(in_dim, hidden_dim, fill)


class ChemicalSystemEncoder(_ElementEncoder):
    def __init__(self, in_dim, hidden_dim):
        def fill(v, cs, Element):
            for el in cs.split("-"):
                v[Element(el).Z] = 1.0
        super().__init__(in_dim, hidden_dim, fill)



