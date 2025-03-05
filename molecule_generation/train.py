import argparse
import os
import os.path as osp

import paddle.distributed as dist
from omegaconf import OmegaConf

from ppmat.datasets import build_dataloader
from ppmat.datasets.CHnmr_dataset import CHnmrinfos
from ppmat.datasets.CHnmr_dataset import DataLoaderCollection
from ppmat.datasets.CHnmr_dataset import get_train_smiles

# from ppmat.datasets import set_signal_handlers
from ppmat.metrics.molecular_metrics import SamplingMolecularMetrics
from ppmat.metrics.molecular_metrics_discrete import TrainMolecularMetricsDiscrete
from ppmat.models.denmr.base_model import ContrastiveModel
from ppmat.models.denmr.base_model import DiffusionPriorModel
from ppmat.models.denmr.base_model import MolecularGraphTransformer
from ppmat.models.denmr.base_model import MultiModalDecoder
from ppmat.models.denmr.diffusion_prior import DiffusionPriorNetwork
from ppmat.models.denmr.extra_features_graph import DummyExtraFeatures
from ppmat.models.denmr.extra_features_graph import ExtraFeatures
from ppmat.models.denmr.extra_features_molecular_graph import ExtraMolecularFeatures
from ppmat.optimizer import build_optimizer
from ppmat.trainer.trainer_multimodal import TrainerCLIP
from ppmat.trainer.trainer_multimodal import TrainerDiffPrior
from ppmat.trainer.trainer_multimodal import TrainerDiffGraphFormer
from ppmat.trainer.trainer_multimodal import TrainerMMDecoder
from ppmat.utils import logger
from ppmat.utils import misc
from ppmat.utils.visualization import MolecularVisualization

if dist.get_world_size() > 1:
    dist.fleet.init(is_collective=True)

if __name__ == "__main__":
    # parse arguments
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

    # load config
    config = OmegaConf.load(args.config)
    cli_config = OmegaConf.from_dotlist(dynamic_args)
    config = OmegaConf.merge(config, cli_config)

    # create output dir
    if dist.get_rank() == 0:
        os.makedirs(config["Tracker"]["save"]["output_dir"], exist_ok=True)
        config_name = os.path.basename(args.config)
        OmegaConf.save(
            config, osp.join(config["Tracker"]["save"]["output_dir"], config_name)
        )

    # convert config to dict
    config = OmegaConf.to_container(config, resolve=True)

    # set logger
    logger.init_logger(
        log_file=osp.join(config["Tracker"]["save"]["output_dir"], f"{args.mode}.log")
    )
    seed = config["Trainer"].get("seed", 42)
    misc.set_random_seed(seed)
    logger.info(f"Set random seed to {seed}")

    # load dataloader from config
    train_data_cfg = config["Dataset"]["train"]
    train_loader = build_dataloader(train_data_cfg)
    val_data_cfg = config["Dataset"]["val"]
    val_loader = build_dataloader(val_data_cfg)
    test_data_cfg = config["Dataset"]["test"]
    test_loader = build_dataloader(test_data_cfg)
    dataloaders = DataLoaderCollection(train_loader, val_loader, test_loader)

    # build datasetinfo
    dataset_infos = CHnmrinfos(dataloaders=dataloaders, cfg=config)
    train_smiles = get_train_smiles(
        cfg=config["Dataset"]["train"],
        dataloader=train_loader,
        dataset_infos=dataset_infos,
        evaluate_dataset=False,
    )
    # extra features
    if config["Model"]["diffusion_model"]["extra_features"] is not None:
        extra_features = ExtraFeatures(
            config["Model"]["diffusion_model"]["extra_features"],
            dataset_infos=dataset_infos,
        )
        domain_features = ExtraMolecularFeatures(
            dataset_infos=dataset_infos,
        )
    else:
        extra_features = DummyExtraFeatures()
        domain_features = DummyExtraFeatures()
    # build datasetinfo
    dataset_infos.compute_input_output_dims(
        dataloader=train_loader,
        extra_features=extra_features,
        domain_features=domain_features,
        conditionDim=config["Model"]["diffusion_model"]["conditdim"],
    )
    # custom training metrics
    train_metrics = TrainMolecularMetricsDiscrete(dataset_infos)
    # custom sampling metrics
    sampling_metrics = SamplingMolecularMetrics(dataset_infos, train_smiles)
    # visualization tools
    visualization_tools = MolecularVisualization(
        config["Dataset"]["train"]["dataset"]["remove_h"], 
        dataset_infos=dataset_infos,
        output_dir=config["Tracker"]["save"]["output_dir"],
    )
    # build model configures
    model_kwargs = {
        "dataset_infos": dataset_infos,
        "train_metrics": train_metrics,
        "sampling_metrics": sampling_metrics,
        "visualization_tools": visualization_tools,
        "extra_features": extra_features,
        "domain_features": domain_features,
    }

    # initialize trainer
    if args.step == 1:
        # build model from config
        model = MolecularGraphTransformer(config["Model"], **model_kwargs)
        # build optimizer and learning rate scheduler from config
        optimizer, lr_scheduler = build_optimizer(
            config["Trainer"]["Optimizer"],
            model,
            config["Trainer"]["epochs"],
            len(train_loader),
        )
        # build trainers
        trainer = TrainerDiffGraphFormer(
            config,
            model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            test_dataloader=test_loader,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            metric_class=train_metrics,
        )

    elif args.step == 2:
        # build model from config
        clip_model = ContrastiveModel(**config["Model"])
        # build optimizer and learning rate scheduler from config
        optimizer, lr_scheduler = build_optimizer(
            config["Trainer"]["Optimizer"],
            clip_model,
            config["Trainer"]["epochs"],
            len(train_loader),
        )
        # build trainer
        trainer = TrainerCLIP(
            config,
            clip_model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            test_dataloader=test_loader,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            metric_class=train_metrics,
        )

    elif args.step == 3:
        # build clip from config and freeze clip encoder
        clip = ContrastiveModel(**config["Model"]["CLIP"])
        for param in clip.graph_encoder.parameters():
            param.stop_gradient = True
        clip.graph_encoder.eval()
        for param in clip.text_encoder.parameters():
            param.stop_gradient = True
        clip.text_encoder.eval()
        # build prior model from config
        prior_network = DiffusionPriorNetwork(**config["Model"]["prior_network"])
        # build diffuison prior
        model = DiffusionPriorModel(
            config=config["Model"],
            model=prior_network,
            clip=clip,
        )
        # build optimizer and learning rate scheduler from config
        optimizer, lr_scheduler = build_optimizer(
            config["Trainer"]["Optimizer"],
            model,
            config["Trainer"]["epochs"],
            len(train_loader),
        )
        # build trainer
        trainer = TrainerDiffPrior(
            config,
            model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            test_dataloader=test_loader,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
        )

    elif args.step == 4:
        # build model from config
        model = MultiModalDecoder(config["Model"], **model_kwargs)
        # build optimizer and learning rate scheduler from config
        optimizer, lr_scheduler = build_optimizer(
            config["Trainer"]["Optimizer"],
            model,
            config["Trainer"]["epochs"],
            len(train_loader),
        )
        # build trainer
        trainer = TrainerMMDecoder(
            config,
            model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            test_dataloader=test_loader,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            metric_class=train_metrics,
        )

    # begin to train or eval or test
    if args.mode == "train":
        trainer.train()
    elif args.mode == "eval":
        if dist.get_rank() == 0:
            loss_dict = trainer.eval()
    elif args.mode == "test":
        if dist.get_rank() == 0:
            result, metric_dict = trainer.test()
