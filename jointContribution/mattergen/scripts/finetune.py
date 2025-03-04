import json
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path

import os
import os.path as osp
import random

import numpy as np

import hydra
import omegaconf
import paddle

import paddle.distributed as dist
from mattergen.common.utils.data_classes import MatterGenCheckpointInfo
from mattergen.common.utils.globals import MODELS_PROJECT_ROOT
from mattergen.diffusion.run import maybe_instantiate
from omegaconf import DictConfig, OmegaConf, open_dict

from mattergen.diffusion.trainer import TrainerDiffusion
from mattergen.common.data.callback import SetPropertyScalers

from mattergen.utils import logger
from mattergen.common.data.utils import set_signal_handlers
from paddle_utils import *

if dist.get_world_size() > 1:
    dist.fleet.init(is_collective=True)


def init_adapter_lightningmodule_from_pretrained(
    adapter_cfg: DictConfig, lightning_module_cfg: DictConfig
):
    assert adapter_cfg.model_path is not None, "model_path must be provided."
    model_path = Path(hydra.utils.to_absolute_path(adapter_cfg.model_path))
    ckpt_info = MatterGenCheckpointInfo(model_path, adapter_cfg.load_epoch)
    ckpt_path = ckpt_info.checkpoint_path
    version_root_path = Path(ckpt_path).relative_to(model_path).parents[1]
    config_path = model_path / version_root_path
    if (config_path / "config.yaml").exists():
        pretrained_cfg_path = config_path
    else:
        pretrained_cfg_path = config_path.parent.parent
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    with hydra.initialize_config_dir(
        str(pretrained_cfg_path.absolute()), version_base="1.1"
    ):
        pretrained_cfg = hydra.compose(config_name="config")
    diffusion_module_cfg = deepcopy(pretrained_cfg.lightning_module.diffusion_module)
    denoiser_cfg = diffusion_module_cfg.model
    with open_dict(adapter_cfg.adapter):
        for k, v in denoiser_cfg.items():
            if k != "_target_" and k != "property_embeddings_adapt":
                adapter_cfg.adapter[k] = v
            if k == "property_embeddings":
                for field in v:
                    if field in adapter_cfg.adapter.property_embeddings_adapt:
                        adapter_cfg.adapter.property_embeddings_adapt.remove(field)
        adapter_cfg.adapter.gemnet[
            "_target_"
        ] = "mattergen.common.gemnet.gemnet_ctrl.GemNetTCtrl"
        adapter_cfg.adapter.gemnet.condition_on_adapt = list(
            adapter_cfg.adapter.property_embeddings_adapt
        )
    with open_dict(diffusion_module_cfg):
        diffusion_module_cfg.model = adapter_cfg.adapter
    with open_dict(lightning_module_cfg):
        lightning_module_cfg.diffusion_module = diffusion_module_cfg
    
    model = maybe_instantiate(diffusion_module_cfg)
    # lightning_module = hydra.utils.instantiate(lightning_module_cfg)

    ckpt: dict = paddle.load(path=str(ckpt_path))
    if 'state_dict' in ckpt:
        pretrained_dict: OrderedDict = ckpt["state_dict"]
    else:
        pretrained_dict: OrderedDict = ckpt
    scratch_dict: OrderedDict = model.state_dict()
    scratch_dict.update(
        (k, pretrained_dict[k]) for k in scratch_dict.keys() & pretrained_dict.keys()
    )
    model.set_state_dict(state_dict=scratch_dict)
    if not adapter_cfg.full_finetuning:
        for name, param in model.named_parameters():
            if name in set(pretrained_dict.keys()):
                out_1 = param
                out_1.stop_gradient = True
                out_1
    return model, lightning_module_cfg


@hydra.main(
    config_path=str(MODELS_PROJECT_ROOT / "conf"),
    config_name="finetune",
    version_base="1.1",
)
def mattergen_finetune(cfg: omegaconf.DictConfig):


    set_signal_handlers()
    logger.init_logger(
        log_file=osp.join(cfg.trainer.output_dir, f"{cfg.trainer.mode}.log")
    )
    seed = cfg.trainer.seed
    if seed is not None:
        paddle.seed(seed=seed)
        np.random.seed(seed)
        random.seed(seed)
    logger.info(f"Seeding everything with {seed}")

    # paddle.set_float32_matmul_precision("high")
    datamodule = maybe_instantiate(cfg.data_module)

    model, lightning_module_cfg = init_adapter_lightningmodule_from_pretrained(
        cfg.adapter, cfg.lightning_module
    )
    with open_dict(cfg):
        cfg.lightning_module = lightning_module_cfg
    config_as_dict = OmegaConf.to_container(cfg, resolve=True)
    print(json.dumps(config_as_dict, indent=4))

    if dist.get_rank() == 0:
        os.makedirs(cfg.trainer.output_dir, exist_ok=True)
        OmegaConf.save(config_as_dict, osp.join(cfg.trainer.output_dir, "config.yaml"))


    optimizer_cfg = cfg.lightning_module.optimizer_partial
    optimizer_cfg = OmegaConf.to_container(optimizer_cfg, resolve=True)
    optimizer_cfg.update(
        dict(
            model_list=model, 
            epochs=cfg.trainer.max_epochs, 
            iters_per_epoch=len(datamodule.train_dataloader())
        )
    )

    optimizer, lr_scheduler = maybe_instantiate(optimizer_cfg)

    set_property_scalers = SetPropertyScalers()
    set_property_scalers.on_fit_start(
        train_dataloader=datamodule.train_dataloader(), model=model
    )

    trainer = TrainerDiffusion(
        config=cfg,
        model = model,
        train_dataloader=datamodule.train_dataloader(),
        val_dataloader=datamodule.val_dataloader(),
        test_dataloader=datamodule.test_dataloader(),
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
    )
    if cfg.trainer.mode == "train":
        trainer.train()
    elif cfg.trainer.mode == "eval":
        trainer.eval()
    elif cfg.trainer.mode == "test":
        trainer.test()




if __name__ == "__main__":
    mattergen_finetune()
