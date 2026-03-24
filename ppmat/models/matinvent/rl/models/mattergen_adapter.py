# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
MatterGen adapter for RL training.

This module provides an adapter wrapper that adds RL-specific methods to MatterGen
without modifying the base class or using monkey patching. Uses composition pattern
for cleaner architecture.

Replaces the monkey patching approach in compat.py.
"""

from typing import Dict
from typing import List

import paddle


class MatterGenRLAdapter:
    """MatterGen adapter that adds RL training capabilities.

    This adapter wraps a MatterGen model and provides RL-specific methods
    required by the MatInvent RL training pipeline. It uses composition
    instead of inheritance or monkey patching.

    Usage:
        base_model = MatterGen(...)
        adapted_model = MatterGenRLAdapter(base_model)

        # The adapter delegates all standard operations to the base model
        output = adapted_model.forward(batch)
        sample_output = adapted_model.sample(batch_data)

        # RL-specific methods are available through the adapter
        noisy_batch, clean_batch, t = adapted_model.add_noise(batch, timestep)
    """

    def __init__(self, base_model):
        """Initialize the adapter with a base MatterGen model.

        Args:
            base_model: An instance of MatterGen or MatterGenWithCondition
        """
        self._base_model = base_model

    def __getattr__(self, name: str):
        """Proxy all undefined attributes to the base model.

        This allows the adapter to transparently delegate all standard
        MatterGen methods (forward, sample, etc.) to the wrapped model.
        """
        return getattr(self._base_model, name)

    @property
    def decoder(self):
        """Access the decoder from the base model."""
        return self._base_model.decoder

    @property
    def model(self):
        """Access the model (denoiser) from the base model."""
        return self._base_model.model

    @property
    def lattice_scheduler(self):
        """Access the lattice scheduler from the base model."""
        return self._base_model.lattice_scheduler

    @property
    def coord_scheduler(self):
        """Access the coord scheduler from the base model."""
        return self._base_model.coord_scheduler

    @property
    def atom_scheduler(self):
        """Access the atom scheduler from the base model."""
        return self._base_model.atom_scheduler

    @property
    def num_train_timesteps(self):
        """Access num_train_timesteps from the base model."""
        return self._base_model.num_train_timesteps

    @property
    def max_t(self):
        """Access max_t from the base model."""
        return getattr(self._base_model, "max_t", 1.0)

    @property
    def time_dim(self):
        """Access time_dim from the base model."""
        return getattr(self._base_model, "time_dim", 256)

    @property
    def lattice_loss_weight(self):
        """Access lattice_loss_weight from the base model."""
        return getattr(self._base_model, "lattice_loss_weight", 1.0)

    @property
    def coord_loss_weight(self):
        """Access coord_loss_weight from the base model."""
        return getattr(self._base_model, "coord_loss_weight", 0.1)

    @property
    def atom_loss_weight(self):
        """Access atom_loss_weight from the base model."""
        return getattr(self._base_model, "atom_loss_weight", 1.0)

    @property
    def d3pm_hybrid_lambda(self):
        """Access d3pm_hybrid_lambda from the base model."""
        return getattr(self._base_model, "d3pm_hybrid_lambda", None)

    def noise_level_encoding(self, t: paddle.Tensor) -> paddle.Tensor:
        """Get noise level encoding from the base model.

        Args:
            t: Timestep tensor

        Returns:
            Noise level encoding tensor
        """
        if hasattr(self._base_model, "noise_level_encoding"):
            return self._base_model.noise_level_encoding(t)
        else:
            # Fallback: use sinusoidal embedding
            from ppmat.models.common.sinusoidal_embedding import SinusoidalEmbedding

            return SinusoidalEmbedding(dim=self.time_dim)(t)

    # MatterGen 不实现 add_noise / calc_sample_loss / calc_kl_reg；
    # 这些计算由 MatInvent._add_noise_to_model / _calc_sample_loss_from_model /
    # _calc_kl_reg_from_models 中的 MatterGen 专用代码路径完成。
    # DiffCSP adapter 实现了这些方法，因此 mat_invent.py 中的委托逻辑会直接调用；
    # MatterGen 没有这些方法，则走 mat_invent.py 中的通用路径。

    def train(self) -> None:
        """Set the model to training mode."""
        self._base_model.train()

    def eval(self) -> None:
        """Set the model to evaluation mode."""
        self._base_model.eval()

    def parameters(self):
        """Return model parameters."""
        return self._base_model.parameters()

    def state_dict(self) -> Dict:
        """Return the model state dict."""
        return self._base_model.state_dict()

    def set_state_dict(self, state_dict: Dict) -> None:
        """Set the model state dict."""
        self._base_model.set_state_dict(state_dict)

    def named_parameters(self):
        """Return named model parameters."""
        return self._base_model.named_parameters()

    def __call__(self, *args, **kwargs):
        """Forward pass through the base model."""
        return self._base_model(*args, **kwargs)


class MatterGenAdapterFactory:
    """Factory for creating adapted MatterGen models.

    This factory provides a clean interface for creating RL-adapted
    MatterGen models while preserving the original model's weights.
    """

    @staticmethod
    def create_adapter(base_model) -> MatterGenRLAdapter:
        """Create an RL adapter for a MatterGen model.

        Args:
            base_model: An instance of MatterGen or MatterGenWithCondition

        Returns:
            MatterGenRLAdapter wrapping the base model

        Example:
            >>> base_model = MatterGen(...)
            >>> adapted = MatterGenAdapterFactory.create_adapter(base_model)
            >>> # Use adapted.model for all operations
        """
        return MatterGenRLAdapter(base_model)

    @staticmethod
    def create_adapted_batch(
        models: List, adapter_class: type = MatterGenRLAdapter
    ) -> List:
        """Create adapters for a batch of models.

        Args:
            models: List of MatterGen instances
            adapter_class: Adapter class to use (for testing/customization)

        Returns:
            List of adapted models
        """
        return [adapter_class(model) for model in models]


def create_matinvent_adapter(model) -> MatterGenRLAdapter:
    """Convenience function to create an adapter.

    This is the main entry point for creating RL-adapted MatterGen models.

    Args:
        model: MatterGen or MatterGenWithCondition instance

    Returns:
        MatterGenRLAdapter instance

    Example:
        >>> from ppmat.models.mattergen.mattergen import MatterGen
        >>> from ppmat.models.matinvent.rl.models.mattergen_adapter import (  # noqa
        ...     create_matinvent_adapter,
        ... )
        >>>
        >>> base_model = MatterGen(...)
        >>> agent = create_matinvent_adapter(base_model)
        >>> prior = create_matinvent_adapter(base_model)
    """
    return MatterGenAdapterFactory.create_adapter(model)
