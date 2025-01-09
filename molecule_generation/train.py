import argparse
import os
import os.path as osp

import paddle.distributed as dist
from omegaconf import OmegaConf

from ppmat.datasets import build_dataloader
from ppmat.datasets import set_signal_handlers
from ppmat.datasets import CHnmr_dataset
from ppmat.models.digress.extra_features_graph import DummyExtraFeatures, ExtraFeatures
from ppmat.metrics.molecular_metrics_discrete import TrainMolecularMetricsDiscrete
from ppmat.metrics.molecular_metrics import SamplingMolecularMetrics
from ppmat.utils.visualization import MolecularVisualization
from ppmat.models.digress.base_model import (
    MolecularGraphTransformer,
    ContrastGraphTransformer,
    ConditionGraphTransformer
)

from ppmat.optimizer import build_optimizer
from ppmat.trainer.trainer_multimodal import TrainerCLIP
from ppmat.trainer.trainer_multimodal import TrainerGraph
from ppmat.trainer.trainer_multimodal import TrainerMultiModal
from ppmat.utils import logger
from ppmat.utils import misc

if dist.get_world_size() > 1:
    dist.fleet.init(is_collective=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="./molecule_generation/configs/digress_CHnmr.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--mode", type=str, default="train", choices=["train", "eval", "test"]
    )
    parser.add_argument(
        "--step", type=int, default=1, help="Step to perform multimodel training"
    )
    args, dynamic_args = parser.parse_known_args()

    config = OmegaConf.load(args.config)
    cli_config = OmegaConf.from_dotlist(dynamic_args)
    config = OmegaConf.merge(config, cli_config)

    if dist.get_rank() == 0:
        os.makedirs(config["Global"]["output_dir"], exist_ok=True)
        config_name = os.path.basename(args.config)
        OmegaConf.save(config, osp.join(config["Global"]["output_dir"], config_name))

    config = OmegaConf.to_container(config, resolve=True)

    logger.init_logger(
        log_file=osp.join(config["Global"]["output_dir"], f"{args.mode}.log")
    )
    seed = config["Global"].get("seed", 42)
    misc.set_random_seed(seed)
    logger.info(f"Set random seed to {seed}")

    # load dataloader from config
    #train_data_cfg = config["Dataset"]["train"]
    #train_loader = build_dataloader(train_data_cfg)
    #val_data_cfg = config["Dataset"]["val"]
    #val_loader = build_dataloader(val_data_cfg)
    #test_data_cfg = config["Dataset"]["test"]
    #test_loader = build_dataloader(test_data_cfg)

    # build datasetinfo
    #dataset_infos = CHnmr_dataset.CHnmrinfos(datamodule=train_loader, cfg=config)
    #train_smiles = CHnmr_dataset.get_train_smiles(cfg=config, train_dataloader=train_loader,
    #                                                    dataset_infos=dataset_infos, evaluate_dataset=False)
    
    # extra features
    if config["Model"]["model_setting"]["extra_features"] is not None:
        extra_features = ExtraFeatures(config["Model"]["model_setting"]["extra_features"], dataset_info=dataset_infos)
        domain_features = ExtraFeatures(config["Model"]["model_setting"]["extra_features"], dataset_infos=dataset_infos)
    else:
        extra_features = DummyExtraFeatures()
        domain_features = DummyExtraFeatures()
    
    #dataset_infos.compute_input_output_dims(datamodule=train_loader, extra_features=extra_features, domain_features=domain_features, conditionDim=config.model.model_setting.conditdim)
    train_metrics = TrainMolecularMetricsDiscrete(dataset_infos)
    sampling_metrics = SamplingMolecularMetrics(dataset_infos, train_smiles)
    visualization_tools = MolecularVisualization(config.Dataset.train.remove_h, dataset_infos=dataset_infos)
    
    model_kwargs = {'dataset_infos': dataset_infos,
                    'train_metrics': train_metrics,
                    'sampling_metrics': sampling_metrics,
                    'visualization_tools': visualization_tools,
                    'extra_features': extra_features,
                    'domain_features': domain_features}

    # initialize trainer
    if args.step == 1:
        model = MolecularGraphTransformer(config["Model"], **model_kwargs)
        # build optimizer and learning rate scheduler from config
        optimizer, lr_scheduler = build_optimizer(
            config["Optimizer"], model, config["Global"]["epochs"], len(train_loader)
        )
        trainer = TrainerGraph(
            config,
            model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            test_dataloader=test_loader,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            metric_class=train_metrics,
        )

    elif args.mode == 2:
        model = ContrastGraphTransformer(config["Model"], **model_kwargs)
        # build optimizer and learning rate scheduler from config
        optimizer, lr_scheduler = build_optimizer(
            config["Optimizer"], model, config["Global"]["epochs"], len(train_loader)
        )
        trainer = TrainerCLIP(
            config,
            model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            test_dataloader=test_loader,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            metric_class=train_metrics,
        )

    elif args.mode == 3:
        model = ConditionGraphTransformer(config["Model"], **model_kwargs)
        # build optimizer and learning rate scheduler from config
        optimizer, lr_scheduler = build_optimizer(
            config["Optimizer"], model, config["Global"]["epochs"], len(train_loader)
        )
        trainer = TrainerMultiModal(
            config,
            model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            test_dataloader=test_loader,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            metric_class=train_metrics,
        )
    if args.mode == "train":
        trainer.train()
    elif args.mode == "eval":
        if dist.get_rank() == 0:
            loss_dict = trainer.eval()
    elif args.mode == "test":
        if dist.get_rank() == 0:
            result, metric_dict = trainer.test()
