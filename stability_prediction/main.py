from __future__ import annotations

import argparse
import os
import pickle
import shutil
import warnings
import zipfile
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import paddle
import paddle.distributed as dist
import paddle.distributed.fleet as fleet
import pandas as pd
import yaml
from dataset.mgl_dataloader import collate_fn_graph
from dataset.mgl_dataset import MGLDataset
from dataset.utils import split_dataset
from models.megnet import MEGNetPlus
from pymatgen.core import Structure
from tqdm import tqdm
from utils._bond import BondExpansion
from utils.default_elements import DEFAULT_ELEMENTS
from utils.ext_pymatgen import Structure2Graph
from utils.ext_pymatgen import get_element_list
from utils.logger import init_logger
from utils.misc import set_random_seed

# To suppress warnings for clearer output
warnings.simplefilter("ignore")

if dist.get_world_size() > 1:
    fleet.init(is_collective=True)


def load_dataset() -> tuple[list[Structure], list[str], list[float]]:
    """Raw data loading function.

    Returns:
        tuple[list[Structure], list[str], list[float]]: structures, mp_id, Eform_per_atom
    """
    # if not os.path.exists("mp.2018.6.1.json"):
    #     f = RemoteFile("https://figshare.com/ndownloader/files/15087992")
    #     with zipfile.ZipFile(f.local_path) as zf:
    #         zf.extractall(".")
    data = pd.read_json("data/mp.2018.6.1.json")
    structures = []
    mp_ids = []

    for mid, structure_str in tqdm(zip(data["material_id"], data["structure"])):
        struct = Structure.from_str(structure_str, fmt="cif")
        structures.append(struct)
        mp_ids.append(mid)
        if len(mp_ids) >= 100:
            break
    return structures, mp_ids, data["formation_energy_per_atom"].tolist()


def load_dataset_from_pickle(
    structures_path,
    ehull_path,
    energy_path=None,
    ehull_clip=None,
    energy_clip=None,
    **kwargs,
):
    with open(structures_path, "rb") as f:
        structures = pickle.load(f)
    with open(ehull_path, "rb") as f:
        ehulls = pickle.load(f)
    if energy_path is not None:
        with open(energy_path, "rb") as f:
            energys = pickle.load(f)
    else:
        energys = None

    if ehull_clip:
        ehulls = np.asarray(ehulls)
        ehulls = ehulls.clip(ehull_clip[0], ehull_clip[1])
        ehulls = ehulls.tolist()
    if energy_clip and energys is not None:
        energys = np.asarray(energys)
        energys = energys.clip(energy_clip[0], energy_clip[1])
        energys = energys.tolist()

    return structures, ehulls, energys


def get_dataloader(cfg):
    # structures, mp_ids, ehulls = load_dataset()
    # structures = structures[:100]
    # ehulls = ehulls[:100]

    structures, ehulls, energys = load_dataset_from_pickle(**cfg["dataset"])
    # structures = structures[:100]
    # ehulls = ehulls[:100]

    # get element types in the dataset
    elem_list = get_element_list(structures)
    elem_list = DEFAULT_ELEMENTS
    # setup a graph converter
    converter = Structure2Graph(
        element_types=elem_list, cutoff=cfg["dataset"]["cutoff"]
    )
    # convert the raw dataset into MEGNetDataset
    labels = (
        {"ehull": ehulls, "energy": energys}
        if energys is not None
        else {"ehull": ehulls}
    )
    mp_dataset = MGLDataset(
        structures=structures,
        labels=labels,
        converter=converter,
        save_dir="",
    )

    train_data, val_data, test_data = split_dataset(
        mp_dataset,
        frac_list=cfg["dataset"]["split_list"],
        shuffle=True,
        random_state=42,
    )

    train_loader = paddle.io.DataLoader(
        train_data,
        batch_sampler=paddle.io.DistributedBatchSampler(
            train_data,
            batch_size=cfg["batch_size"],
            shuffle=True,
        ),
        collate_fn=collate_fn_graph,
        num_workers=0,
    )
    val_loader = paddle.io.DataLoader(
        val_data,
        batch_sampler=paddle.io.DistributedBatchSampler(
            val_data,
            batch_size=cfg["batch_size"],
        ),
        collate_fn=collate_fn_graph,
    )
    test_loader = paddle.io.DataLoader(
        test_data,
        batch_sampler=paddle.io.DistributedBatchSampler(
            test_data,
            batch_size=cfg["batch_size"],
        ),
        collate_fn=collate_fn_graph,
    )

    return train_loader, val_loader, test_loader, elem_list


def get_model(cfg, elem_list):
    # define the bond expansion
    bond_expansion = BondExpansion(
        rbf_type="Gaussian", initial=0.0, final=5.0, num_centers=100, width=0.5
    )
    # setup the architecture of MEGNet model
    model_cfg = cfg["model"]
    model_cfg.update({"bond_expansion": bond_expansion, "element_types": elem_list})
    model = MEGNetPlus(**model_cfg)
    # model.set_dict(paddle.load('data/paddle_weight.pdparams'))

    if dist.get_world_size() > 1:
        model = fleet.distributed_model(model)

    return model


def get_optimizer(cfg, model):

    lr_scheduler = paddle.optimizer.lr.CosineAnnealingDecay(**cfg["lr_cfg"])
    optimizer = paddle.optimizer.Adam(
        parameters=model.parameters(),
        learning_rate=lr_scheduler,
        epsilon=1e-08,
        weight_decay=0.0,
    )
    if dist.get_world_size() > 1:
        optimizer = fleet.distributed_optimizer(optimizer)
    return optimizer, lr_scheduler


def train_epoch(model, loader, loss_fn, metric_fn, optimizer, loss_weight, epoch, log):
    model.train()
    total_loss = defaultdict(list)
    total_metric = defaultdict(list)
    total_num_data = 0
    for idx, batch_data in enumerate(loader):
        graph, _, state_attr, labels = batch_data
        batch_size = state_attr.shape[0]

        preds = model(graph, state_attr)

        keys = labels.keys()
        keys = sorted(keys)
        train_loss = 0.0

        msg = ""
        for i, key in enumerate(keys):
            label = labels[key]
            if len(preds.shape) > 1:
                pred = preds[:, i]
            else:
                pred = preds

            loss = loss_fn(pred, label)
            metric = metric_fn(pred, label)

            total_loss[key].append(loss * batch_size)
            total_metric[key].append(metric * batch_size)
            if key in loss_weight.keys():
                train_loss += loss * loss_weight[key]
            else:
                train_loss += loss
            msg += f" | {key}_loss: {loss.item():.6f} | {key}_mae: {metric.item():.6f}"

        train_loss.backward()
        optimizer.step()
        optimizer.clear_grad()

        total_num_data += batch_size

        if paddle.distributed.get_rank() == 0 and (
            idx % 10 == 0 or idx == len(loader) - 1
        ):
            message = "train: epoch %d | step %d | lr %.6f" % (
                epoch,
                idx,
                optimizer.get_lr(),
            )
            message += msg
            log.info(message)

    total_loss = {key: sum(total_loss[key]) / total_num_data for key in keys}
    total_metric = {key: sum(total_metric[key]) / total_num_data for key in keys}
    return total_loss, total_metric


@paddle.no_grad()
def eval_epoch(model, loader, loss_fn, metric_fn, log):
    model.eval()
    total_loss = defaultdict(list)
    total_metric = defaultdict(list)
    total_preds = defaultdict(list)
    total_labels = defaultdict(list)

    total_num_data = 0
    for idx, batch_data in enumerate(loader):
        graph, _, state_attr, labels = batch_data
        batch_size = state_attr.shape[0]

        preds = model(graph, state_attr)

        keys = labels.keys()
        keys = sorted(keys)

        msg = ""
        for i, key in enumerate(keys):
            label = labels[key]
            if len(preds.shape) > 1:
                pred = preds[:, i]
            else:
                pred = preds

            loss = loss_fn(pred, label)
            metric = metric_fn(pred, label)

            total_loss[key].append(loss * batch_size)
            total_metric[key].append(metric * batch_size)

            total_preds[key].extend(pred.tolist())
            total_labels[key].extend(label.tolist())

        total_num_data += batch_size
    total_loss = {key: sum(total_loss[key]) / total_num_data for key in keys}
    total_metric = {key: sum(total_metric[key]) / total_num_data for key in keys}
    return total_loss, total_metric, total_preds, total_labels


def train(cfg):
    log = init_logger(log_file=os.path.join(cfg["save_path"], "train.log"))
    train_loader, val_loader, test_loader, elem_list = get_dataloader(cfg)

    model = get_model(cfg, elem_list)
    optimizer, lr_scheduler = get_optimizer(cfg, model)

    loss_fn = paddle.nn.functional.mse_loss
    metric_fn = paddle.nn.functional.l1_loss

    loss_weight = cfg.get("loss_weight", {})

    global_step = 0
    best_metric = float("inf")

    for epoch in range(cfg["epochs"]):
        train_loss, train_metric = train_epoch(
            model, train_loader, loss_fn, metric_fn, optimizer, loss_weight, epoch, log
        )
        lr_scheduler.step()

        if paddle.distributed.get_rank() == 0:
            eval_loss, eval_metric, total_preds, total_labels = eval_epoch(
                model, val_loader, loss_fn, metric_fn, log
            )
            msg = ""
            for key in train_loss.keys():
                msg += f", train_{key}_loss: {train_loss[key].item():.6f}"
                msg += f", train_{key}_mae: {train_metric[key].item():.6f}"
            for key in eval_loss.keys():
                msg += f", eval_{key}_loss: {eval_loss[key].item():.6f}"
                msg += f", eval_{key}_mae: {eval_metric[key].item():.6f}"

            log.info(f"epoch: {epoch}" + msg)

            if eval_metric["ehull"] < best_metric:
                best_metric = eval_metric["ehull"]
                paddle.save(
                    model.state_dict(), "{}/best.pdparams".format(cfg["save_path"])
                )
                log.info("Saving best checkpoint at {}".format(cfg["save_path"]))

            paddle.save(
                model.state_dict(), "{}/latest.pdparams".format(cfg["save_path"])
            )
            if epoch % 500 == 0:
                paddle.save(
                    model.state_dict(),
                    "{}/epoch_{}.pdparams".format(cfg["save_path"], epoch),
                )
    if paddle.distributed.get_rank() == 0:
        test_loss, test_metric, total_preds, total_labels = eval_epoch(
            model, test_loader, loss_fn, metric_fn, log
        )
        msg = ""
        for key in test_loss.keys():
            msg += f", test_{key}_loss: {test_loss[key].item():.6f}"
            msg += f", test_{key}_mae: {test_metric[key].item():.6f}"
        log.info(f"epoch: {epoch}" + msg)


def evaluate(cfg):
    log = init_logger(log_file=os.path.join(cfg["save_path"], "evaluate.log"))
    train_loader, val_loader, test_loader, elem_list = get_dataloader(cfg)

    model = get_model(cfg, elem_list)
    optimizer, lr_scheduler = get_optimizer(cfg, model)

    loss_fn = paddle.nn.functional.mse_loss
    metric_fn = paddle.nn.functional.l1_loss

    test_loss, test_metric, total_preds, total_labels = eval_epoch(
        model, val_loader, loss_fn, metric_fn, log
    )
    msg = ""
    for key in test_loss.keys():
        msg += f", eval_{key}_loss: {test_loss[key].item():.6f}"
        msg += f", eval_{key}_mae: {test_metric[key].item():.6f}"
    log.info(f"epoch: 0" + msg)


def test(cfg):
    log = init_logger(log_file=os.path.join(cfg["save_path"], "test.log"))
    train_loader, val_loader, test_loader, elem_list = get_dataloader(cfg)

    model = get_model(cfg, elem_list)
    optimizer, lr_scheduler = get_optimizer(cfg, model)

    loss_fn = paddle.nn.functional.mse_loss
    metric_fn = paddle.nn.functional.l1_loss

    test_loss, test_metric, total_preds, total_labels = eval_epoch(
        model, test_loader, loss_fn, metric_fn, log
    )
    msg = ""
    for key in test_loss.keys():
        msg += f", test_{key}_loss: {test_loss[key].item():.6f}"
        msg += f", test_{key}_mae: {test_metric[key].item():.6f}"
    log.info("epoch: 0" + msg)

    data = {}
    for key in total_preds.keys():
        data[f"pred_{key}"] = total_preds[key]
        data[f"label_{key}"] = total_labels[key]
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(cfg["save_path"], "predictions.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="./configs/megnet_2d.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--mode", type=str, default="train", choices=["train", "eval", "test"]
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    if paddle.distributed.get_rank() == 0:
        os.makedirs(cfg["save_path"], exist_ok=True)
        shutil.copy(args.config, cfg["save_path"])

    set_random_seed(cfg.get("seed", 42))

    if args.mode == "train":
        train(cfg)
    elif args.mode == "eval":
        evaluate(cfg)
    elif args.mode == "test":
        test(cfg)
    else:
        raise ValueError("Unknown mode: {}".format(args.mode))
