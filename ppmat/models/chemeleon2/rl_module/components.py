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

from abc import ABC, abstractmethod
import paddle
import paddle.nn as nn


class RewardComponent(ABC, nn.Layer):
    required_metrics = []

    def __init__(
        self,
        weight=1.0,
        normalize_fn=None,
        eps=1e-4,
    ):
        super().__init__()
        self.weight = weight
        self.normalize_fn = normalize_fn
        self.eps = eps

    @abstractmethod
    def compute(self, **kwargs):
        pass

    def forward(self, **kwargs):
        rewards = self.compute(**kwargs)
        if self.normalize_fn:
            rewards = self._normalize(rewards)

        return rewards * self.weight

    def _normalize(self, rewards):
        if self.normalize_fn == "norm":
            rewards = self.normalize(rewards, eps=self.eps)
        elif self.normalize_fn == "std":
            rewards = self.standardize(rewards, eps=self.eps)
        elif self.normalize_fn == "subtract_mean":
            rewards = rewards - rewards.mean()
        elif self.normalize_fn == "clip":
            rewards = paddle.clip(rewards, min=-1.0, max=1.0)
        elif self.normalize_fn is None:
            pass
        else:
            raise ValueError(
                f"Unknown normalization type: {self.normalize_fn}. Use 'norm', 'std', 'clip', or None."
            )
        return rewards

    def normalize(self, rewards, eps=1e-4):
        return (rewards - rewards.min()) / (rewards.max() - rewards.min() + eps)

    def standardize(self, rewards, eps=1e-4):
        return (rewards - rewards.mean()) / (rewards.std() + eps)


class CustomReward(RewardComponent):
    def compute(self, gen_structures, **kwargs):
        return paddle.zeros([len(gen_structures)])


class CreativityReward(RewardComponent):
    required_metrics = ["unique", "novel"]

    def compute(self, gen_structures, metrics_obj, **kwargs):
        from collections import defaultdict
        
        reference_structures = metrics_obj._reference_structures
        metrics_results = metrics_obj._results
        
        ref_structures_by_formula = defaultdict(list)
        for ref_structure in reference_structures + gen_structures:
            ref_structures_by_formula[ref_structure.reduced_formula].append(
                ref_structure
            )
        
        rewards = []
        for i, gen_structure in enumerate(gen_structures):
            u, v = metrics_results["unique"][i], metrics_results["novel"][i]
            if u and v:
                r = 1.0
            elif not u and not v:
                r = 0.0
            else:
                matching_refs = ref_structures_by_formula.get(
                    gen_structure.reduced_formula, []
                )
                r = 0.5
            rewards.append(r)
        
        return paddle.to_tensor(rewards).astype('float32')


class EnergyReward(RewardComponent):
    required_metrics = ["e_above_hull"]

    def compute(self, gen_structures, metrics_obj, **kwargs):
        metrics_results = metrics_obj._results
        
        r_energy = paddle.to_tensor(metrics_results["e_above_hull"]).astype('float32')
        r_energy = paddle.where(paddle.isnan(r_energy), paddle.to_tensor(1.0), r_energy)
        r_energy = paddle.clip(r_energy, min=0.0, max=1.0)
        r_energy = r_energy * -1.0
        return r_energy


class StructureDiversityReward(RewardComponent):
    required_metrics = ["structure_diversity"]

    def compute(self, gen_structures, metrics_obj, device, **kwargs):
        raise NotImplementedError(
            "StructureDiversityReward requires real feature extractor. "
            "Override this method and provide gen_structure_features."
        )


class CompositionDiversityReward(RewardComponent):
    required_metrics = ["composition_diversity"]

    def compute(self, gen_structures, metrics_obj, device, **kwargs):
        raise NotImplementedError(
            "CompositionDiversityReward requires real feature extractor. "
            "Override this method and provide gen_composition_features."
        )


class PredictorReward(RewardComponent):
    required_metrics = []

    def __init__(self, predictor, target_value=None, **kwargs):
        super().__init__(**kwargs)
        self.predictor = predictor
        self.target_value = target_value

    def compute(self, gen_structures, **kwargs):
        raise NotImplementedError(
            "PredictorReward requires a real predictor model. "
            "Pass a predictor to __init__ or override this method."
        )


def mmd_reward(z_gen, z_ref):
    def poly_k(z, y, deg=3):
        d = z.shape[-1]
        return (z @ y.T / d + 1) ** deg
    
    M, N = len(z_gen), len(z_ref)
    
    k_gg = poly_k(z_gen, z_gen)
    k_rr = poly_k(z_ref, z_ref)
    k_gr = poly_k(z_gen, z_ref)
    
    if N <= 1 or M <= 1:
        return {'r': paddle.to_tensor(0.0), 'r_indiv': paddle.zeros([M])}

    R_term = (k_rr.sum() - k_rr.trace()) / (N * (N - 1))
    G = k_gg.sum() - k_gg.trace()
    C = k_gr.sum()
    mmd_full = G / (M * (M - 1)) + R_term - 2 * C / (M * N)
    
    S = k_gg.sum(axis=1) - k_gg.diagonal()
    T = k_gr.sum(axis=1)
    
    Mp = M - 1
    Ap = Mp * (Mp - 1)
    mmd_drop = (G - 2 * S) / Ap + R_term - 2 * (C - T) / (Mp * N)
    
    r_indiv = mmd_drop - mmd_full
    return {'r': -mmd_full, 'r_indiv': r_indiv}
