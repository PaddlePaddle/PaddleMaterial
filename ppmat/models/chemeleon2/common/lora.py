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

import paddle.nn as nn

__all__ = [
    "LoRALayer",
    "LoRALinear",
    "apply_lora_to_linear",
    "get_lora_parameters",
    "merge_lora_weights",
    "print_trainable_parameters",
]


class LoRALayer(nn.Layer):
    def __init__(self, in_features, out_features, rank=8, alpha=16, dropout=0.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.lora_A = self.create_parameter(shape=[in_features, rank], default_initializer=nn.initializer.KaimingUniform())
        self.lora_B = self.create_parameter(shape=[rank, out_features], default_initializer=nn.initializer.Constant(0.0))
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

    def forward(self, x):
        if self.dropout is not None:
            x = self.dropout(x)
        return (x @ self.lora_A @ self.lora_B) * self.scaling


class LoRALinear(nn.Layer):
    def __init__(self, original_layer, rank=8, alpha=16, dropout=0.0):
        super().__init__()
        self.original_layer = original_layer
        self.original_layer.stop_gradient = True
        in_features = original_layer.weight.shape[0]
        out_features = original_layer.weight.shape[1]
        self.lora = LoRALayer(in_features, out_features, rank, alpha, dropout)

    def forward(self, x):
        return self.original_layer(x) + self.lora(x)


def apply_lora_to_linear(module, rank=8, alpha=16, dropout=0.0, target_modules=None):
    if target_modules is None:
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "fc1", "fc2"]
    for name, child in module.named_children():
        if isinstance(child, nn.Linear) and any(target in name for target in target_modules):
            setattr(module, name, LoRALinear(child, rank, alpha, dropout))
        else:
            apply_lora_to_linear(child, rank, alpha, dropout, target_modules)
    return module


def get_lora_parameters(module):
    return [param for name, param in module.named_parameters() if 'lora' in name.lower()]


def merge_lora_weights(module):
    for name, child in module.named_children():
        if isinstance(child, LoRALinear):
            original_weight = child.original_layer.weight
            lora_weight = child.lora.lora_A @ child.lora.lora_B * child.lora.scaling
            merged_weight = original_weight + lora_weight
            merged_layer = nn.Linear(original_weight.shape[0], original_weight.shape[1], bias_attr=child.original_layer.bias is not None)
            merged_layer.weight.set_value(merged_weight)
            if child.original_layer.bias is not None:
                merged_layer.bias.set_value(child.original_layer.bias)
            setattr(module, name, merged_layer)
        else:
            merge_lora_weights(child)
    return module


def print_trainable_parameters(module):
    trainable_params = sum(p.numel() for p in module.parameters() if not p.stop_gradient)
    all_params = sum(p.numel() for p in module.parameters())
    print(f"trainable params: {trainable_params:,} || all params: {all_params:,} || trainable%: {100 * trainable_params / all_params:.2f}%")
