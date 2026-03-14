# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import os.path as osp

import paddle.distributed as dist
from omegaconf import OmegaConf

from ppmat.datasets import build_dataloader
from ppmat.datasets import set_signal_handlers
from ppmat.models import build_model
from ppmat.optimizer import build_optimizer
from ppmat.utils import logger
from ppmat.utils import misc

from spectrum_elucidation.ecformer.trainer import ECDFormerTrainer


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="ECDFormer for ECD Spectrum Prediction")
    parser.add_argument(
        "-c", "--config",
        type=str,
        default="./spectrum_elucidation/ecformer/configs/ecd.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint path",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Only run evaluation on validation set",
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Only run evaluation on test set",
    )
    parser.add_argument(
        "--predict",
        type=str,
        default=None,
        help="Path to data for prediction (inference mode)",
    )

    args, dynamic_args = parser.parse_known_args()

    # Load configuration
    config = OmegaConf.load(args.config)
    cli_config = OmegaConf.from_dotlist(dynamic_args)
    config = OmegaConf.merge(config, cli_config)
    
    # Override Global configuration based on command line arguments
    if args.eval_only:
        config.Global.do_train = False
        config.Global.do_eval = True
        config.Global.do_test = False
    elif args.test_only:
        config.Global.do_train = False
        config.Global.do_eval = False
        config.Global.do_test = True
    elif args.predict is not None:
        config.Global.do_train = False
        config.Global.do_eval = False
        config.Global.do_test = False
        config.Global.do_predict = True
        config.Dataset.predict.data_path = args.predict

    # Save configuration
    if dist.get_rank() == 0:
        os.makedirs(config.Trainer.output_dir, exist_ok=True)
        config_name = os.path.basename(args.config)
        OmegaConf.save(config, osp.join(config.Trainer.output_dir, config_name))

    # Convert to dictionary
    config = OmegaConf.to_container(config, resolve=True)

    # Initialize logging
    logger_path = osp.join(config["Trainer"]["output_dir"], "run.log")
    logger.init_logger(log_file=logger_path)
    logger.info(f"Logger saved to {logger_path}")
    logger.info(f"Config: {config}")

    # Set random seed
    seed = config["Trainer"].get("seed", 42)
    misc.set_random_seed(seed)
    logger.info(f"Set random seed to {seed}")

    # Set signal handlers
    set_signal_handlers()

    # Build data loaders
    dataloaders = {}
    
    if config["Global"].get("do_train", True):
        train_cfg = config["Dataset"].get("train")
        assert train_cfg is not None, "train dataset must be defined when do_train is True"
        dataloaders["train"] = build_dataloader(train_cfg)
        logger.info(f"Train dataset loaded, size: {len(dataloaders['train'].dataset)}")
    
    if config["Global"].get("do_eval", False) or config["Global"].get("do_train", True):
        val_cfg = config["Dataset"].get("val")
        if val_cfg is not None:
            dataloaders["val"] = build_dataloader(val_cfg)
            logger.info(f"Validation dataset loaded, size: {len(dataloaders['val'].dataset)}")
        else:
            logger.info("No validation dataset defined.")
    
    if config["Global"].get("do_test", False):
        test_cfg = config["Dataset"].get("test")
        assert test_cfg is not None, "test dataset must be defined when do_test is True"
        dataloaders["test"] = build_dataloader(test_cfg)
        logger.info(f"Test dataset loaded, size: {len(dataloaders['test'].dataset)}")
    
    if config["Global"].get("do_predict", False):
        predict_cfg = config["Dataset"].get("predict")
        assert predict_cfg is not None, "predict dataset must be defined when do_predict is True"
        dataloaders["predict"] = build_dataloader(predict_cfg)
        logger.info(f"Prediction dataset loaded, size: {len(dataloaders['predict'].dataset)}")

    # Build model
    model_cfg = config["Model"]
    model = build_model(model_cfg)
    logger.info(f"Model built: {model_cfg['__class_name__']}")
    
    # Print model parameters count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if not p.stop_gradient)
    logger.info(f"Total parameters: {total_params / 1e6:.2f}M")
    logger.info(f"Trainable parameters: {trainable_params / 1e6:.2f}M")

    # Build optimizer and learning rate scheduler
    optimizer = None
    lr_scheduler = None
    
    if config.get("Optimizer") is not None and config["Global"].get("do_train", True):
        assert dataloaders.get("train") is not None, "train_loader must be defined when optimizer is defined"
        assert config["Trainer"].get("max_epochs") is not None, "max_epochs must be defined"
        
        optimizer, lr_scheduler = build_optimizer(
            config["Optimizer"],
            model,
            config["Trainer"]["max_epochs"],
            len(dataloaders["train"]),
        )
        logger.info(f"Optimizer built: {config['Optimizer']['__class_name__']}")

    # Build trainer
    trainer = ECDFormerTrainer(
        config=config["Trainer"],
        model=model,
        train_dataloader=dataloaders.get("train"),
        val_dataloader=dataloaders.get("val"),
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
    )

    # Resume from checkpoint
    if args.resume is not None:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        save_load.load_checkpoint(
            args.resume,
            model,
            optimizer,
            trainer.scaler,
        )

    # Execute training/evaluation/prediction
    if config["Global"].get("do_train", True):
        logger.info("Starting training...")
        trainer.train()
    
    if config["Global"].get("do_eval", False):
        logger.info("Evaluating on validation set...")
        if "val" in dataloaders:
            time_info, loss_info, metric_info = trainer.eval(dataloaders["val"])
            
            # Print detailed metrics
            msg = "Validation Results:"
            for key, meter in metric_info.items():
                msg += f" | {key}: {meter.avg:.6f}"
            logger.info(msg)
        else:
            logger.warning("No validation dataloader found, skipping evaluation.")
    
    if config["Global"].get("do_test", False):
        logger.info("Evaluating on test set...")
        if "test" in dataloaders:
            time_info, loss_info, metric_info = trainer.eval(dataloaders["test"])
            
            msg = "Test Results:"
            for key, meter in metric_info.items():
                msg += f" | {key}: {meter.avg:.6f}"
            logger.info(msg)
        else:
            logger.warning("No test dataloader found, skipping test evaluation.")
    
    if config["Global"].get("do_predict", False):
        logger.info("Running prediction...")
        if "predict" in dataloaders:
            results = trainer.predict(dataloaders["predict"])
            
            # Save prediction results
            import json
            output_path = osp.join(config["Trainer"]["output_dir"], "predictions.json")
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Predictions saved to {output_path}")
        else:
            logger.warning("No prediction dataloader found, skipping prediction.")


if __name__ == "__main__":
    main()