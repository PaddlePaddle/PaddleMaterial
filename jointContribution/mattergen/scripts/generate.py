import os
from pathlib import Path
from typing import Literal
import fire

import numpy as np
import random

import paddle

from mattergen.common.data.types import TargetProperty
from mattergen.common.utils.data_classes import MatterGenCheckpointInfo
from mattergen.generator import CrystalGenerator
from hydra.utils import instantiate
from typing import Any, Mapping, TypeVar
T = TypeVar("T")

def maybe_instantiate(
    instance_or_config: (T | Mapping), expected_type=None, **kwargs
) -> T:
    """
    If instance_or_config is a mapping with a _target_ field, instantiate it.
    Otherwise, return it as is.
    """
    if isinstance(instance_or_config, Mapping) and "_target_" in instance_or_config:
        instance = instantiate(instance_or_config, **kwargs)
    else:
        instance = instance_or_config
    assert expected_type is None or isinstance(
        instance, expected_type
    ), f"Expected {expected_type}, got {type(instance)}"
    return instance

### checked
def main(
    output_path: str,
    model_path: str,
    batch_size: int = 64,
    num_batches: int = 1,
    config_overrides: (list[str] | None) = None,
    checkpoint_epoch: (Literal["best", "last"] | int) = "last",
    properties_to_condition_on: (TargetProperty | None) = None,
    sampling_config_path: (str | None) = None,
    sampling_config_name: str = "default",
    sampling_config_overrides: (list[str] | None) = None,
    record_trajectories: bool = True,
    diffusion_guidance_factor: (float | None) = None,
    strict_checkpoint_loading: bool = True,
    num_atoms_distribution: str = "ALEX_MP_20",
):
    """
    Evaluate diffusion model against molecular metrics.

    Args:
        model_path: Path to DiffusionLightningModule checkpoint directory.
        output_path: Path to output directory.
        config_overrides: Overrides for the model config, e.g., `model.num_layers=3 model.hidden_dim=128`.
        properties_to_condition_on: Property value to draw conditional sampling with respect to. When this value is an empty dictionary (default), unconditional samples are drawn.
        sampling_config_path: Path to the sampling config file. (default: None, in which case we use `DEFAULT_SAMPLING_CONFIG_PATH` from explorers.common.utils.utils.py)
        sampling_config_name: Name of the sampling config (corresponds to `{sampling_config_path}/{sampling_config_name}.yaml` on disk). (default: default)
        sampling_config_overrides: Overrides for the sampling config, e.g., `condition_loader_partial.batch_size=32`.
        load_epoch: Epoch to load from the checkpoint. If None, the best epoch is loaded. (default: None)
        record: Whether to record the trajectories of the generated structures. (default: True)
        strict_checkpoint_loading: Whether to raise an exception when not all parameters from the checkpoint can be matched to the model.

    NOTE: When specifying dictionary values via the CLI, make sure there is no whitespace between the key and value, e.g., `--properties_to_condition_on={key1:value1}`.
    """
    # set seed
    # seed = 42
    # paddle.seed(seed=seed)
    # np.random.seed(seed)
    # random.seed(seed)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    sampling_config_overrides = sampling_config_overrides or []
    config_overrides = config_overrides or []
    properties_to_condition_on = properties_to_condition_on or {}
    checkpoint_info = MatterGenCheckpointInfo(
        model_path=Path(model_path).resolve(),
        load_epoch=checkpoint_epoch,
        config_overrides=config_overrides,
        strict_checkpoint_loading=strict_checkpoint_loading,
    )
    _sampling_config_path = (
        Path(sampling_config_path) if sampling_config_path is not None else None
    )
    model = maybe_instantiate(checkpoint_info.config.lightning_module.diffusion_module)

    # for name, m in model.named_sublayers():
    #     if isinstance(m, paddle.nn.Linear):
    #         print(f"'diffusion_module.{name}',")
    state_dict = paddle.load(checkpoint_info.checkpoint_path)
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    missing_keys, unexpected_keys = model.set_state_dict(state_dict)
    if len(missing_keys) > 0:
        raise ValueError(f"Missing keys: {missing_keys}")
    if len(unexpected_keys) > 0:
        raise ValueError(f"Unexpected keys: {unexpected_keys}")
    
    generator = CrystalGenerator(
        checkpoint_info=checkpoint_info,
        properties_to_condition_on=properties_to_condition_on,
        batch_size=batch_size,
        num_batches=num_batches,
        sampling_config_name=sampling_config_name,
        sampling_config_path=_sampling_config_path,
        sampling_config_overrides=sampling_config_overrides,
        record_trajectories=record_trajectories,
        num_atoms_distribution=num_atoms_distribution,
        diffusion_guidance_factor=diffusion_guidance_factor
        if diffusion_guidance_factor is not None
        else 0.0,
        _model=model,
    )
    generator.generate(output_dir=Path(output_path))


if __name__ == "__main__":
    fire.Fire(main)


# PYTHONPATH=$PWD python scripts/generate.py results/ checkpoints/mattergen_base --batch_size=16 --num_batches 1
# PYTHONPATH=$PWD python scripts/generate.py results_mp20/ outputs/08-54-29/output --batch_size=100 --num_batches 100