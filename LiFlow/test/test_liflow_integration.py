import os

import paddle
import pytest
from omegaconf import OmegaConf

from ppmat.datasets import build_dataloader
from ppmat.models import build_model
from ppmat.optimizer import build_optimizer
from ppmat.trainer.base_trainer import BaseTrainer
from ppmat.models.liflow import LiFlow
from ppmat.utils import save_load


def test_optimizer_step():
    model = LiFlow(num_features=8, num_radial_basis=4, num_layers=1, num_elements=10)
    optimizer = paddle.optimizer.Adam(learning_rate=1e-3, parameters=model.parameters())
    batch = {
        "positions_1": paddle.zeros([2, 3]),
        "positions_2": paddle.ones([2, 3]),
        "prior": paddle.zeros([2, 3]),
        "elements": paddle.to_tensor([1, 2]),
        "num_atoms": paddle.to_tensor([2]),
        "time": paddle.to_tensor([0.5]),
        "temp": paddle.to_tensor([800.0]),
    }
    loss = model(batch)["loss_dict"]["loss"]
    loss.backward()
    optimizer.step()
    optimizer.clear_grad()


def test_two_steps_save_reload_and_continue(tmp_path):
    model = LiFlow(num_features=8, num_radial_basis=4, num_layers=1, num_elements=10)
    optimizer = paddle.optimizer.Adam(learning_rate=1e-3, parameters=model.parameters())
    batch = {
        "positions_1": paddle.zeros([2, 3]),
        "positions_2": paddle.ones([2, 3]),
        "prior": paddle.zeros([2, 3]),
        "elements": paddle.to_tensor([1, 2]),
        "num_atoms": paddle.to_tensor([2]),
        "time": paddle.to_tensor([0.5]),
        "temp": paddle.to_tensor([800.0]),
    }
    for _ in range(2):
        model(batch)["loss_dict"]["loss"].backward()
        optimizer.step()
        optimizer.clear_grad()

    prefix = str(tmp_path / "checkpoints" / "latest")
    save_load.save_checkpoint(
        model, optimizer, {"global_step": 2}, output_dir=str(tmp_path), prefix="latest"
    )

    reloaded = LiFlow(num_features=8, num_radial_basis=4, num_layers=1, num_elements=10)
    reloaded_optimizer = paddle.optimizer.Adam(
        learning_rate=1e-3, parameters=reloaded.parameters()
    )
    state = save_load.load_checkpoint(
        prefix, reloaded, reloaded_optimizer
    )
    assert state["global_step"] == 2
    reloaded(batch)["loss_dict"]["loss"].backward()
    reloaded_optimizer.step()
    reloaded_optimizer.clear_grad()


@pytest.mark.parametrize(
    "config_name,prediction_mode,noise_scale",
    [
        ("liflow_universal_propagator.yaml", "velocity", 0.0),
        ("liflow_universal_corrector.yaml", "data", 0.05),
    ],
)
def test_yaml_basetrainer_two_steps(tmp_path, config_name, prediction_mode, noise_scale):
    root = os.path.dirname(os.path.dirname(__file__))
    config_path = os.path.join(
        root, "molecular_dynamics_integrator", "liflow", "configs", config_name
    )
    config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    config["Trainer"].update(
        {
            "output_dir": str(tmp_path / prediction_mode),
            "max_epochs": 1,
            "max_iter": 2,
            "save_freq": 10,
            "log_freq": 100,
            "start_eval_epoch": 100,
            "eval_freq": 100,
            "compute_metric_during_train": False,
        }
    )
    dataset_params = config["Dataset"]["train"]["dataset"]["__init_params__"]
    dataset_params.update(
        {
            "path": os.path.join(root, "test", "fixtures", "liflow", "dataset_mini"),
            "cache_path": str(tmp_path / "cache" / prediction_mode),
            "index_file": "train_800K.csv",
            "time_delay_steps": 100,
            "random_time": False,
            "noise_scale": noise_scale,
        }
    )
    config["Dataset"]["train"]["sampler"]["__init_params__"].update(
        {"batch_size": 1, "shuffle": False}
    )
    config["Model"]["__init_params__"]["num_features"] = 8
    config["Model"]["__init_params__"]["num_radial_basis"] = 4
    config["Model"]["__init_params__"]["num_layers"] = 1
    config["Model"]["__init_params__"]["num_elements"] = 77
    config["Model"]["__init_params__"]["prediction_mode"] = prediction_mode

    model = build_model(config["Model"])
    loader = build_dataloader(config["Dataset"]["train"])
    optimizer, scheduler = build_optimizer(
        config["Optimizer"], model, config["Trainer"]["max_epochs"], len(loader)
    )
    trainer = BaseTrainer(
        config["Trainer"], model, train_dataloader=loader, optimizer=optimizer,
        lr_scheduler=scheduler,
    )
    trainer.train()
    assert trainer.state.global_step == 1
    save_load.save_checkpoint(
        model, optimizer, {"global_step": trainer.state.global_step},
        output_dir=str(tmp_path), prefix=prediction_mode,
    )
    assert os.path.exists(str(tmp_path / "checkpoints" / f"{prediction_mode}.pdparams"))
