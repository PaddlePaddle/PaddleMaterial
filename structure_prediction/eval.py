from __future__ import annotations

import argparse
import json
import os
import shutil

import numpy as np
import paddle
import paddle.distributed as dist
import paddle.distributed.fleet as fleet
import yaml
from dataset.cryst_dataset import CrystDataset
from dataset.cryst_dataset import GenDataset
from metrics.gen_metircs import GenMetrics
from metrics.rec_metrics import RecMetrics
from models.diffusion import CSPDiffusion
from models.diffusion import CSPDiffusionWithType
from models.diffusion_pp import CSPDiffusionPP
from models.diffusion_with_guidance import CSPDiffusionWithGuidance
from models.diffusion_with_guidance_d3pm import CSPDiffusionWithGuidanceD3PM
from models.mattergen import MatterGen
from models.mattergen import MatterGenWithGuidance
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure
from utils import logger
from utils.crystal import lattices_to_params_shape
from utils.misc import set_random_seed

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
        "prop",
        "anchor_index",
        "ops_inv",
        "ops",
        "spacegroup",
    ]
    for key in keys:
        if key not in batch[0]:
            continue
        if key in ["edge_index"]:
            cumulative_length = 0
            result_arrays_edge_index = []
            for x in batch:
                new_array = x[key] + cumulative_length
                result_arrays_edge_index.append(new_array)
                cumulative_length += x["num_atoms"]
            new_batch[key] = np.concatenate(result_arrays_edge_index, axis=1)
        elif key in [
            "frac_coords",
            "atom_types",
            "lengths",
            "angles",
            "to_jimages",
            "prop",
            "ops",
            "ops_inv",
            "spacegroup",
        ]:
            new_batch[key] = np.concatenate([x[key] for x in batch], axis=0)
        elif key in [
            "anchor_index",
        ]:
            cumulative_length = 0
            result_arrays_anchor_index = []
            for x in batch:
                new_array = x[key] + cumulative_length
                result_arrays_anchor_index.append(new_array)
                cumulative_length += len(x[key])
            new_batch[key] = np.concatenate(result_arrays_anchor_index, axis=0)
        elif key in [
            "num_atoms",
            "num_bonds",
        ]:
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
    elif model_name == "CSPDiffusionWithGuidance":
        model = CSPDiffusionWithGuidance(**model_cfg)
    elif model_name == "CSPDiffusionWithGuidanceD3PM":
        model = CSPDiffusionWithGuidanceD3PM(**model_cfg)
    elif model_name == "CSPDiffusionPP":
        model = CSPDiffusionPP(**model_cfg)
    elif model_name == "MatterGen":
        model = MatterGen(**model_cfg)
    elif model_name == "MatterGenWithGuidance":
        model = MatterGenWithGuidance(**model_cfg)
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
            shuffle=False,
        ),
        collate_fn=collate_fn_graph,
    )
    test_loader = paddle.io.DataLoader(
        test_data,
        batch_sampler=paddle.io.DistributedBatchSampler(
            test_data,
            batch_size=cfg["batch_size"],
            shuffle=False,
        ),
        collate_fn=collate_fn_graph,
    )

    return train_loader, val_loader, test_loader


def diffusion(loader, model, step_lr):
    frac_coords = []
    num_atoms = []
    atom_types = []
    lattices = []
    for idx, batch in enumerate(loader):
        print(f"{idx}/{len(loader)}")
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
    except Exception as e:
        print(f"pymatgen error: {e}")
        return None


@paddle.no_grad()
def eval_csp(cfg):
    # csp 任务
    train_loader, val_loader, test_loader = get_dataloader(cfg)
    model = get_model(cfg)
    model.eval()
    step_lr = cfg["sample_step_lr"]
    num_evals = cfg.get("num_evals", 1)
    frac_coords = []
    num_atoms = []
    atom_types = []
    lattices = []
    input_data_list = []
    for idx, batch in enumerate(test_loader):
        batch_frac_coords, batch_num_atoms, batch_atom_types = [], [], []
        batch_lattices = []
        for eval_idx in range(num_evals):
            logger.info(
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

    batch_idx = 0
    crys_array_list = get_crystals_list(
        frac_coords[batch_idx],
        atom_types[batch_idx],
        lengths[batch_idx],
        angles[batch_idx],
        num_atoms[batch_idx],
    )
    true_crystal_array_list = get_crystals_list(
        input_data_batch["frac_coords"],
        input_data_batch["atom_types"],
        input_data_batch["lengths"],
        input_data_batch["angles"],
        input_data_batch["num_atoms"],
    )
    # crys_array_list 和 true_crystal_array_list中的元素组成list，其中每个元素是一个dict
    results = [
        {"prediction": crys_array_list[i], "ground_truth": true_crystal_array_list[i]}
        for i in range(len(true_crystal_array_list))
    ]
    # 将results转换为jsonl格式，并写入文件
    with open("output.jsonl", "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    metric_fn = RecMetrics()
    metrics = metric_fn(results)
    print(metrics)


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
        cur_frac_coords = frac_coords[start_idx : start_idx + num_atom]
        cur_atom_types = atom_types[start_idx : start_idx + num_atom]
        cur_lengths = lengths[batch_idx]
        cur_angles = angles[batch_idx]

        crystal_array_list.append(
            {
                "frac_coords": cur_frac_coords.detach().cpu().numpy().tolist(),
                "atom_types": cur_atom_types.detach().cpu().numpy().tolist(),
                "lengths": cur_lengths.detach().cpu().numpy().tolist(),
                "angles": cur_angles.detach().cpu().numpy().tolist(),
            }
        )
        start_idx = start_idx + num_atom
    return crystal_array_list


def get_crystals_list2(frac_coords, atom_types, lengths, angles, num_atoms):
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


@paddle.no_grad()
def eval_ab_generation(cfg):
    model = get_model(cfg)
    model.eval()
    sample_loader = get_gen_dataloader(cfg)
    metric_fn = GenMetrics(gt_file_path=cfg["dataset"]["test"]["path"])

    tar_dir = os.path.join(cfg["save_path"], "generation")
    os.makedirs(tar_dir, exist_ok=True)

    frac_coords, atom_types, lattices, lengths, angles, num_atoms = diffusion(
        sample_loader, model, cfg["sample_step_lr"]
    )
    if atom_types.dim() != 1:
        atom_types = paddle.to_tensor([row.argmax() + 1 for row in atom_types])
    else:
        atom_types = atom_types + 1
    crystal_list = get_crystals_list(
        frac_coords, atom_types, lengths, angles, num_atoms
    )

    results = [{"prediction": crystal_list[i]} for i in range(len(crystal_list))]
    metric = metric_fn(results)
    print(metric)

    # strcuture_list = p_map(get_pymatgen, crystal_list)
    # for i, structure in enumerate(strcuture_list):
    #     formula = structure.formula.replace(" ", "-")
    #     tar_file = os.path.join(tar_dir, f"{i + 1}_{formula}.cif")
    #     if structure is not None:
    #         writer = CifWriter(structure)
    #         writer.write_file(tar_file)
    #     else:
    #         logger.info(f"{i + 1} Error Structure.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="./configs/diffcsp_2d.yaml",
        help="Path to config file",
    )
    parser.add_argument("--mode", type=str, default="test", choices=["csp", "gen"])
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
    logger.init_logger(log_file=os.path.join(cfg["save_path"], f"{args.mode}.log"))

    if args.mode == "csp":
        eval_csp(cfg)
    elif args.mode == "gen":
        eval_ab_generation(cfg)
    else:
        raise ValueError("Unknown mode: {}".format(args.mode))
