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

from collections import defaultdict
from functools import partial

import paddle
import paddle.nn as nn

from ppmat.models.chemeleon2.ldm_module.diffusion import create_diffusion
from ppmat.models.chemeleon2.ldm_module.ldm import LDMModule


class RLModule(nn.Layer):
    def __init__(
        self,
        ldm_ckpt_path,
        rl_configs,
        reward_fn,
        sampling_configs,
        optimizer=None,
        scheduler=None,
        vae_ckpt_path=None,
    ):
        super().__init__()

        self.clip_ratio = rl_configs.get("clip_ratio", 0.2)
        self.kl_weight = rl_configs.get("kl_weight", 0.1)
        self.entropy_weight = rl_configs.get("entropy_weight", 0.01)
        self.num_group_samples = rl_configs.get("num_group_samples", 4)
        self.group_reward_norm = rl_configs.get("group_reward_norm", True)
        self.num_inner_batch = rl_configs.get("num_inner_batch", 1)
        self.reward_fn = reward_fn
        self.sampling_configs = sampling_configs
        self.optimizer_config = optimizer
        self.scheduler_config = scheduler

        if ldm_ckpt_path is not None:
            checkpoint = paddle.load(ldm_ckpt_path)
            self.ldm = LDMModule(**checkpoint.get('model_config', {}))
            self.ldm.set_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded LDM from {ldm_ckpt_path}")
            
            self.ldm.vae.eval()
            for param in self.ldm.vae.parameters():
                param.stop_gradient = True
        
        self.use_cfg = self.ldm.use_cfg
        
        if sampling_configs.get('sampler') == "ddim":
            timestep_respacing = "ddim" + str(sampling_configs.get('sampling_steps', 50))
        else:
            timestep_respacing = str(sampling_configs.get('sampling_steps', 50))
        
        diffusion_configs = self.ldm.diffusion_configs.copy()
        diffusion_configs['timestep_respacing'] = timestep_respacing
        self.sampling_diffusion = create_diffusion(**diffusion_configs)

    @paddle.no_grad()
    def rollout(self, batch):
        result = self.ldm.sample(batch, **{**self.sampling_configs, 'collect_trajectory': True})
        if isinstance(result, dict):
            trajectory = result.get('trajectory', {})
        else:
            trajectory = result
        return result, trajectory

    def compute_rewards(self, batch_gen):
        num_samples = batch_gen.num_graphs
        rewards = self.reward_fn(batch_gen)
        if self.group_reward_norm:
            group_rewards_norm = []
            for i in range(0, num_samples, self.num_group_samples):
                group_reward = rewards[i : i + self.num_group_samples]
                group_reward_norm = self.reward_fn.normalize(group_reward)
                group_rewards_norm.append(group_reward_norm)
            rewards_norm = paddle.concat(group_rewards_norm, axis=0)
        else:
            rewards_norm = self.reward_fn.normalize(rewards)
        return rewards, rewards_norm

    def calculate_loss(self, zs, log_probs, advantages, mask, y=None):
        sampler_step_fn = (
            partial(self.sampling_diffusion.ddim_sample, eta=self.sampling_configs.get('eta', 0.0))
            if self.sampling_configs.get('sampler') == "ddim"
            else self.sampling_diffusion.p_sample
        )
        indices = list(range(self.sampling_diffusion.num_timesteps))[::-1]
        
        if self.use_cfg:
            assert y is not None
            y = self.ldm.condition_module(y, training=False)
            y.stop_gradient = True
        
        res = defaultdict(int)
        for i, t in enumerate(indices):
            z = zs[i]
            old_log_probs = log_probs[i]
            old_log_probs.stop_gradient = True
            
            if self.use_cfg:
                z = paddle.concat([z, z], axis=0)
                mask = paddle.concat([mask, mask], axis=0)
            
            model_kwargs = {
                'mask': mask,
                'y': y,
            }
            if self.use_cfg:
                model_kwargs['cfg_scale'] = self.sampling_configs.get('cfg_scale', 1.0)
            
            t_tensor = paddle.full([z.shape[0]], t, dtype='int64')
            out = sampler_step_fn(
                model=(
                    self.ldm.denoiser.forward_with_cfg
                    if self.use_cfg
                    else self.ldm.denoiser.forward
                ),
                x=z,
                t=t_tensor,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )
            
            if self.use_cfg:
                out['mean'] = paddle.chunk(out['mean'], 2, axis=0)[0]
                out['std'] = paddle.chunk(out['std'], 2, axis=0)[0]
                mask = paddle.chunk(mask, 2, axis=0)[0]
            
            current_log_probs = _calculate_log_prob(
                zs[i + 1], out['mean'], out['std'], mask
            )
            
            if (t_tensor == 0).all() and (out['std'] == 0).all():
                continue
            
            log_ratio = current_log_probs - old_log_probs
            ratio = paddle.exp(log_ratio)
            clipped_ratio = paddle.clip(
                ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio
            )
            surrogate_objective = paddle.minimum(
                ratio * advantages, clipped_ratio * advantages
            )
            
            kl_div_log_ratio = old_log_probs - current_log_probs
            kl_div = kl_div_log_ratio.exp() - 1 - kl_div_log_ratio
            
            entropy = -current_log_probs
            
            policy_loss = (
                -surrogate_objective
                + self.kl_weight * kl_div
                - self.entropy_weight * entropy
            )
            loss = policy_loss.mean()
            scaled_loss = loss / len(indices)
            
            scaled_loss.backward()
            
            res['scaled_loss'] += scaled_loss.detach().item()
            res['loss'] += loss.detach().item()
            res['surrogate_objective'] += surrogate_objective.mean().detach().item()
            res['kl_div'] += kl_div.mean().detach().item()
            res['entropy'] += entropy.mean().detach().item()
            res['log_ratio'] = log_ratio.mean().detach().item()
            res['ratio'] = ratio.mean().detach().item()
        
        return res

    def get_config(self):
        return {
            'clip_ratio': self.clip_ratio,
            'kl_weight': self.kl_weight,
            'entropy_weight': self.entropy_weight,
            'num_group_samples': self.num_group_samples,
            'group_reward_norm': self.group_reward_norm,
            'num_inner_batch': self.num_inner_batch,
            'sampling_configs': self.sampling_configs,
        }


def _broadcast_mask(mask, z):
    while len(mask.shape) < len(z.shape):
        mask = mask.unsqueeze(-1)
    return mask.expand_as(z)


def _calculate_log_prob(x, mean, std, mask, reduce='mean'):
    log_prob = -0.5 * (paddle.log(2 * paddle.to_tensor(3.14159265359) * std**2) + ((x - mean) / std) ** 2)
    
    if mask is not None:
        log_prob = log_prob * _broadcast_mask(mask, x)
    
    reduced_dim = list(range(1, x.ndim))
    if reduce == 'sum':
        log_prob = log_prob.sum(axis=reduced_dim)
    elif reduce == 'mean':
        if mask is not None:
            summed = log_prob.sum(axis=reduced_dim)
            counts = _broadcast_mask(mask, x).sum(axis=reduced_dim).clip(min=1e-6)
            log_prob = summed / counts
        else:
            log_prob = log_prob.mean(axis=reduced_dim)
    elif reduce == 'none':
        pass
    return log_prob
