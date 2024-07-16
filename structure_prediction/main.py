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
from dataset.cryst_dataset import CrystDataset
from dataset.cryst_dataset import GenDataset
from dataset.cryst_dataset import SampleDataset
from models.diffusion import CSPDiffusion
from models.diffusion import CSPDiffusionWithType
from p_tqdm import p_map
from pymatgen.core import Structure
from tqdm import tqdm
from utils.logger import init_logger
from utils.misc import set_random_seed

# To suppress warnings for clearer output
warnings.simplefilter("ignore")
import time

from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifWriter
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pyxtal.symmetry import Group

if dist.get_world_size() > 1:
    fleet.init(is_collective=True)


def collate_fn_graph(batch):
    new_batch = {}
    keys = [
        "edge_index",
        "y",
        "batch",
        "ptr",
        "frac_coords",
        "atom_types",
        "lengths",
        "angles",
        "to_jimages",
        "num_atoms",
        "num_bonds",
        "num_nodes",
    ]
    for key in keys:
        if key not in batch[0]:
            continue
        if key in ["edge_index"]:
            new_batch[key] = np.concatenate([x[key] for x in batch], axis=1)
        elif key in ["frac_coords", "atom_types", "lengths", "angles", "to_jimages"]:
            new_batch[key] = np.concatenate([x[key] for x in batch], axis=0)
        elif key in ["num_atoms", "num_bonds"]:
            new_batch[key] = np.array([x[key] for x in batch])
        elif key in ["num_nodes"]:
            new_batch[key] = np.array([x[key] for x in batch]).sum()

    graph_idxs = []
    for i in range(len(batch)):
        graph_idxs.extend([i] * batch[i]["num_atoms"])
    new_batch["batch"] = np.array(graph_idxs, dtype="int64")
    new_batch["num_graphs"] = len(batch)

    return new_batch


def get_model(cfg):
    # setup the architecture of MEGNet model
    model_cfg = cfg["model"]
    model_name = model_cfg.pop("__name__", None)
    if model_name == "CSPDiffusionWithType":
        model = CSPDiffusionWithType(**model_cfg)
    else:
        model = CSPDiffusion(**model_cfg)
    # model.set_dict(paddle.load('data/paddle_weight.pdparams'))

    if dist.get_world_size() > 1:
        model = fleet.distributed_model(model)

    return model


def get_dataloader(cfg):

    train_data = CrystDataset(**cfg["dataset"]["train"])
    val_data = CrystDataset(**cfg["dataset"]["val"])
    test_data = CrystDataset(**cfg["dataset"]["test"])

    train_loader = paddle.io.DataLoader(
        train_data,
        batch_sampler=paddle.io.DistributedBatchSampler(
            train_data,
            batch_size=cfg["batch_size"],
            shuffle=True,
        ),
        collate_fn=collate_fn_graph,
        num_workers=cfg["num_workers"],
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

    return train_loader, val_loader, test_loader


def get_sample_dataloader(cfg):

    sample_data = SampleDataset(**cfg["dataset"]["sample"])

    sample_loader = paddle.io.DataLoader(
        sample_data,
        batch_sampler=paddle.io.DistributedBatchSampler(
            sample_data,
            batch_size=cfg["batch_size"],
        ),
        collate_fn=collate_fn_graph,
    )

    return sample_loader


def get_gen_dataloader(cfg):

    gen_data = GenDataset(**cfg["dataset"]["generation"])

    gen_loader = paddle.io.DataLoader(
        gen_data,
        batch_sampler=paddle.io.DistributedBatchSampler(
            gen_data,
            batch_size=cfg["batch_size"],
        ),
        collate_fn=collate_fn_graph,
    )

    return gen_loader


def get_optimizer(cfg, model):
    lr_scheduler = paddle.optimizer.lr.ReduceOnPlateau(**cfg["lr_cfg"])
    if cfg.get("grad_clip") is not None:
        clip = paddle.nn.ClipGradByValue(max=cfg.get("grad_clip"))
    else:
        clip = None
    optimizer = paddle.optimizer.Adam(
        parameters=model.parameters(),
        learning_rate=lr_scheduler,
        epsilon=1e-08,
        weight_decay=0.0,
        grad_clip=clip,
    )
    if dist.get_world_size() > 1:
        optimizer = fleet.distributed_optimizer(optimizer)
    return optimizer, lr_scheduler


def train_epoch(
    model,
    loader,
    optimizer,
    epoch,
    log,
):
    model.train()
    total_loss = defaultdict(list)
    total_num_data = 0

    for idx, batch_data in enumerate(loader):
        batch_size = batch_data["num_graphs"]
        losses = model(batch_data)

        train_loss = losses["loss"]
        train_loss.backward()
        optimizer.step()
        optimizer.clear_grad()

        for key, value in losses.items():
            total_loss[key].append(value * batch_size)
        total_num_data += batch_size

        if paddle.distributed.get_rank() == 0 and (
            idx % 10 == 0 or idx == len(loader) - 1
        ):
            msg = ""
            for key, value in losses.items():
                msg += " | %s: %.4f" % (key, value.item())
            message = "train: epoch %d | step %d | lr %.6f" % (
                epoch,
                idx,
                optimizer.get_lr(),
            )
            message += msg
            log.info(message)
    total_loss = {
        key: sum(total_loss[key]) / total_num_data for key in total_loss.keys()
    }
    return total_loss


@paddle.no_grad()
def eval_epoch(model, loader, log):
    model.eval()
    total_loss = defaultdict(list)
    total_num_data = 0
    for idx, batch_data in enumerate(loader):
        batch_size = batch_data["num_graphs"]

        losses = model(batch_data)

        for key, value in losses.items():
            total_loss[key].append(value * batch_size)
        total_num_data += batch_size
    total_loss = {
        key: sum(total_loss[key]) / total_num_data for key in total_loss.keys()
    }

    return total_loss


def train(cfg):
    log = init_logger(log_file=os.path.join(cfg["save_path"], "train.log"))
    train_loader, val_loader, test_loader = get_dataloader(cfg)

    model = get_model(cfg)

    optimizer, lr_scheduler = get_optimizer(cfg, model)

    global_step = 0
    best_metric = float("inf")

    for epoch in range(cfg["epochs"]):
        train_loss = train_epoch(model, train_loader, optimizer, epoch, log)

        if paddle.distributed.get_rank() == 0:
            eval_loss = eval_epoch(model, val_loader, log)
            lr_scheduler.step(eval_loss["loss"])

            msg = ""
            for key in train_loss.keys():
                msg += f", train_{key}_loss: {train_loss[key].item():.6f}"
            for key in eval_loss.keys():
                msg += f", eval_{key}_loss: {eval_loss[key].item():.6f}"

            log.info(f"epoch: {epoch}" + msg)

            if eval_loss["loss"] < best_metric:
                best_metric = eval_loss["loss"]
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


def lattices_to_params_shape(lattices):
    lengths = paddle.sqrt(x=paddle.sum(x=lattices**2, axis=-1))
    angles = paddle.zeros_like(x=lengths)
    for i in range(3):
        j = (i + 1) % 3
        k = (i + 2) % 3
        angles[..., i] = paddle.clip(
            x=paddle.sum(x=lattices[..., j, :] * lattices[..., k, :], axis=-1)
            / (lengths[..., j] * lengths[..., k]),
            min=-1.0,
            max=1.0,
        )
    angles = paddle.acos(x=angles) * 180.0 / np.pi
    return lengths, angles


def diffusion(loader, model, step_lr):
    frac_coords = []
    num_atoms = []
    atom_types = []
    lattices = []
    input_data_list = []
    for idx, batch in enumerate(loader):
        outputs, traj = model.sample(batch, step_lr=step_lr)
        frac_coords.append(outputs["frac_coords"].detach().cpu())
        num_atoms.append(outputs["num_atoms"].detach().cpu())
        atom_types.append(outputs["atom_types"].detach().cpu())
        lattices.append(outputs["lattices"].detach().cpu())
    frac_coords = paddle.concat(x=frac_coords, axis=0)
    num_atoms = paddle.concat(x=num_atoms, axis=0)
    atom_types = paddle.concat(x=atom_types, axis=0)
    lattices = paddle.concat(x=lattices, axis=0)
    lengths, angles = lattices_to_params_shape(lattices)
    return frac_coords, atom_types, lattices, lengths, angles, num_atoms


def get_crystals_list(frac_coords, atom_types, lengths, angles, num_atoms):
    """
    args:
        frac_coords: (num_atoms, 3)
        atom_types: (num_atoms)
        lengths: (num_crystals)
        angles: (num_crystals)
        num_atoms: (num_crystals)
    """
    assert frac_coords.shape[0] == atom_types.shape[0] == num_atoms.sum()
    assert lengths.shape[0] == angles.shape[0] == num_atoms.shape[0]
    start_idx = 0
    crystal_array_list = []
    for batch_idx, num_atom in enumerate(num_atoms.tolist()):
        start_0 = frac_coords.shape[0] + start_idx if start_idx < 0 else start_idx
        cur_frac_coords = paddle.slice(
            frac_coords, [0], [start_0], [start_0 + num_atom]
        )
        start_1 = atom_types.shape[0] + start_idx if start_idx < 0 else start_idx
        cur_atom_types = paddle.slice(atom_types, [0], [start_1], [start_1 + num_atom])
        cur_lengths = lengths[batch_idx]
        cur_angles = angles[batch_idx]
        crystal_array_list.append(
            {
                "frac_coords": cur_frac_coords.detach().cpu().numpy(),
                "atom_types": cur_atom_types.detach().cpu().numpy(),
                "lengths": cur_lengths.detach().cpu().numpy(),
                "angles": cur_angles.detach().cpu().numpy(),
            }
        )
        start_idx = start_idx + num_atom
    return crystal_array_list


def get_pymatgen(crystal_array):
    frac_coords = crystal_array["frac_coords"]
    atom_types = crystal_array["atom_types"]
    lengths = crystal_array["lengths"]
    angles = crystal_array["angles"]
    try:
        structure = Structure(
            lattice=Lattice.from_parameters(*(lengths.tolist() + angles.tolist())),
            species=atom_types,
            coords=frac_coords,
            coords_are_cartesian=False,
        )
        return structure
    except:
        return None


def test(cfg):
    log = init_logger(log_file=os.path.join(cfg["save_path"], "test.log"))
    train_loader, val_loader, test_loader = get_dataloader(cfg)

    model = get_model(cfg)

    step_lr = cfg["sample_step_lr"]
    num_evals = cfg.get("num_evals", 1)
    frac_coords = []
    num_atoms = []
    atom_types = []
    lattices = []
    input_data_list = []
    start_time = time.time()
    for idx, batch in enumerate(test_loader):
        batch_all_frac_coords = []
        batch_all_lattices = []
        batch_frac_coords, batch_num_atoms, batch_atom_types = [], [], []
        batch_lattices = []
        for eval_idx in range(num_evals):
            log.info(
                f"batch {idx} / {len(test_loader)}, sample {eval_idx} / {num_evals}"
            )
            outputs, traj = model.sample(batch, step_lr=step_lr)
            batch_frac_coords.append(outputs["frac_coords"].detach().cpu())
            batch_num_atoms.append(outputs["num_atoms"].detach().cpu())
            batch_atom_types.append(outputs["atom_types"].detach().cpu())
            batch_lattices.append(outputs["lattices"].detach().cpu())
        frac_coords.append(paddle.stack(x=batch_frac_coords, axis=0))
        num_atoms.append(paddle.stack(x=batch_num_atoms, axis=0))
        atom_types.append(paddle.stack(x=batch_atom_types, axis=0))
        lattices.append(paddle.stack(x=batch_lattices, axis=0))
        input_data_list.append(batch)
    frac_coords = paddle.concat(x=frac_coords, axis=1)
    num_atoms = paddle.concat(x=num_atoms, axis=1)
    atom_types = paddle.concat(x=atom_types, axis=1)
    lattices = paddle.concat(x=lattices, axis=1)
    lengths, angles = lattices_to_params_shape(lattices)

    input_data_batch = {}
    keys = [
        "edge_index",
        "y",
        "batch",
        "ptr",
        "frac_coords",
        "atom_types",
        "lengths",
        "angles",
        "to_jimages",
        "num_atoms",
        "num_bonds",
        "num_nodes",
    ]
    for key in keys:
        if key not in input_data_list[0]:
            continue
        if key in ["edge_index"]:
            input_data_batch[key] = paddle.concat(
                [x[key] for x in input_data_list], axis=1
            )
        elif key in ["frac_coords", "atom_types", "lengths", "angles", "to_jimages"]:
            input_data_batch[key] = paddle.concat(
                [x[key] for x in input_data_list], axis=0
            )
        elif key in ["num_atoms", "num_bonds"]:
            input_data_batch[key] = paddle.concat([x[key] for x in input_data_list])
        elif key in ["num_nodes"]:
            input_data_batch[key] = paddle.to_tensor(
                [x[key] for x in input_data_list]
            ).sum()

    paddle.save(
        obj={
            "eval_setting": cfg,
            "input_data_batch": input_data_batch,
            "frac_coords": frac_coords,
            "num_atoms": num_atoms,
            "atom_types": atom_types,
            "lattices": lattices,
            "lengths": lengths,
            "angles": angles,
            "time": time.time() - start_time,
        },
        path=os.path.join(cfg["save_path"], "test.pt"),
    )


def sample(cfg):
    log = init_logger(log_file=os.path.join(cfg["save_path"], "sample.log"))
    model = get_model(cfg)

    sample_loader = get_sample_dataloader(cfg)

    formula = cfg["dataset"]["sample"]["formula"]
    tar_dir = os.path.join(cfg["save_path"], formula)
    os.makedirs(tar_dir, exist_ok=True)

    frac_coords, atom_types, lattices, lengths, angles, num_atoms = diffusion(
        sample_loader, model, cfg["sample_step_lr"]
    )
    crystal_list = get_crystals_list(
        frac_coords, atom_types, lengths, angles, num_atoms
    )
    strcuture_list = p_map(get_pymatgen, crystal_list)
    for i, structure in enumerate(strcuture_list):
        tar_file = os.path.join(tar_dir, f"{formula}_{i + 1}.cif")
        if structure is not None:
            writer = CifWriter(structure)
            writer.write_file(tar_file)
        else:
            print(f"{i + 1} Error Structure.")


def generation(cfg):
    log = init_logger(log_file=os.path.join(cfg["save_path"], "generation.log"))
    model = get_model(cfg)

    sample_loader = get_gen_dataloader(cfg)

    tar_dir = os.path.join(cfg["save_path"], "generation")
    os.makedirs(tar_dir, exist_ok=True)

    frac_coords, atom_types, lattices, lengths, angles, num_atoms = diffusion(
        sample_loader, model, cfg["sample_step_lr"]
    )
    atom_types = paddle.to_tensor([row.argmax() + 1 for row in atom_types])
    crystal_list = get_crystals_list(
        frac_coords, atom_types, lengths, angles, num_atoms
    )
    strcuture_list = p_map(get_pymatgen, crystal_list)
    for i, structure in enumerate(strcuture_list):
        formula = structure.formula.replace(" ", "-")
        tar_file = os.path.join(tar_dir, f"{i + 1}_{formula}.cif")
        if structure is not None:
            writer = CifWriter(structure)
            writer.write_file(tar_file)
        else:
            print(f"{i + 1} Error Structure.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="./configs/diffcsp_2d.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--mode", type=str, default="train", choices=["train", "test", "sample", "gen"]
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    if paddle.distributed.get_rank() == 0:
        os.makedirs(cfg["save_path"], exist_ok=True)
        try:
            shutil.copy(args.config, cfg["save_path"])
        except shutil.SameFileError:
            pass

    set_random_seed(cfg.get("seed", 42))

    if args.mode == "train":
        train(cfg)
    elif args.mode == "test":
        test(cfg)
    elif args.mode == "sample":
        sample(cfg)
    elif args.mode == "gen":
        generation(cfg)
    else:
        raise ValueError("Unknown mode: {}".format(args.mode))
