#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PaddleMaterials - Interatomic Potentials Training Script
训练入口：python train.py --config configs/sevennet/sevennet_hfo2.yaml
"""

import argparse
import os
import sys
import random
import numpy as np
import paddle
import paddle.nn.functional as F
from paddle.io import Dataset, DataLoader
from tqdm import tqdm
from ase.io import read
from ase.neighborlist import neighbor_list

# 添加项目路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from ppmat.models.sevennet import PurePaddleSevenNet


def parse_args():
    parser = argparse.ArgumentParser(description="Train interatomic potential model")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    return parser.parse_args()


def load_config(config_path):
    import yaml
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _extract_energy(atoms):
    for key in ["y_energy", "energy", "free_energy", "total_energy"]:
        if key in atoms.info:
            try:
                return float(atoms.info[key])
            except Exception:
                pass
    try:
        return float(atoms.get_potential_energy())
    except Exception:
        pass
    raise ValueError("Cannot find energy label from atoms object")


def _extract_forces(atoms):
    if "y_force" in atoms.arrays:
        try:
            return np.array(atoms.arrays["y_force"], dtype=np.float32)
        except Exception:
            pass
    try:
        return np.array(atoms.get_forces(), dtype=np.float32)
    except Exception:
        pass
    raise ValueError("Cannot find force labels from atoms object")


def _atoms_to_graph(atoms, cutoff):
    positions = atoms.get_positions().astype("float32")
    numbers = atoms.get_atomic_numbers().astype("int64")
    i, j = neighbor_list("ij", atoms, cutoff)
    edge_index = np.vstack([i, j]).astype("int64")

    total_energy = float(_extract_energy(atoms))
    forces = _extract_forces(atoms)
    num_atoms = len(numbers)
    energy_per_atom = total_energy / max(num_atoms, 1)

    return {
        "z": numbers,
        "pos": positions,
        "edge_index": edge_index,
        "energy": np.array([total_energy], dtype="float32"),
        "energy_per_atom": np.array([energy_per_atom], dtype="float32"),
        "forces": forces.astype("float32"),
        "num_nodes": np.array([num_atoms], dtype="int64"),
    }


class GraphDataset(Dataset):
    def __init__(self, graphs):
        super().__init__()
        self.graphs = graphs

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        g = self.graphs[idx]
        return {
            "z": paddle.to_tensor(g["z"], dtype="int64"),
            "pos": paddle.to_tensor(g["pos"], dtype="float32"),
            "edge_index": paddle.to_tensor(g["edge_index"], dtype="int64"),
            "energy": paddle.to_tensor(g["energy"], dtype="float32"),
            "energy_per_atom": paddle.to_tensor(g["energy_per_atom"], dtype="float32"),
            "forces": paddle.to_tensor(g["forces"], dtype="float32"),
            "num_nodes": paddle.to_tensor(g["num_nodes"], dtype="int64"),
        }


def _collate_graphs(batch):
    return batch


def _split_graphs(graphs, valid_ratio=0.1, seed=1):
    rng = random.Random(seed)
    idx = list(range(len(graphs)))
    rng.shuffle(idx)
    n_valid = max(1, int(len(graphs) * valid_ratio))
    valid_idx = idx[:n_valid]
    train_idx = idx[n_valid:]
    return [graphs[i] for i in train_idx], [graphs[i] for i in valid_idx]


def _compute_stats(graphs):
    energies = np.array([float(g["energy_per_atom"][0]) for g in graphs], dtype=np.float32)
    return float(np.mean(energies)), float(max(np.std(energies), 1e-8))


def _run_epoch(model, loader, energy_mean, energy_std, force_weight, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    
    total_loss, total_e_loss, total_f_loss, count = 0.0, 0.0, 0.0, 0

    for batch in tqdm(loader, leave=False):
        if is_train:
            optimizer.clear_grad()

        losses, e_losses, f_losses = [], [], []

        for graph in batch:
            pos = graph["pos"].detach().clone()
            pos.stop_gradient = False

            out = model({
                "z": graph["z"],
                "pos": pos,
                "edge_index": graph["edge_index"],
            })
            pred_total = out["total_energy"].reshape([1])

            num_atoms = paddle.cast(graph["num_nodes"].reshape([1]), "float32")
            pred_per_atom = pred_total / num_atoms
            true_per_atom = graph["energy_per_atom"].reshape([1])

            true_norm = (true_per_atom - energy_mean) / energy_std
            pred_norm = (pred_per_atom - energy_mean) / energy_std
            e_loss = F.mse_loss(pred_norm, true_norm)

            pred_forces = -paddle.grad(
                outputs=[pred_total],
                inputs=[pos],
                create_graph=False,
                retain_graph=False,
            )[0]
            f_loss = F.mse_loss(pred_forces, graph["forces"])

            loss = e_loss + force_weight * f_loss
            losses.append(loss)
            e_losses.append(e_loss.detach())
            f_losses.append(f_loss.detach())

        loss = paddle.stack(losses).mean()
        if is_train:
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item())
        total_e_loss += float(paddle.stack(e_losses).mean().item())
        total_f_loss += float(paddle.stack(f_losses).mean().item())
        count += 1

    return {
        "loss": total_loss / max(count, 1),
        "e_loss": total_e_loss / max(count, 1),
        "f_loss": total_f_loss / max(count, 1),
    }


def build_model(config):
    model_cfg = config["model"]
    return PurePaddleSevenNet(
        num_species=model_cfg.get("num_species", 100),
        hidden_dim=model_cfg.get("hidden_dim", 128),
        num_message_layers=model_cfg.get("num_message_layers", 4),
        num_rbf=model_cfg.get("num_rbf", 32),
        cutoff=model_cfg.get("cutoff", 5.0),
    )


def build_optimizer(config, model):
    optim_cfg = config["optimizer"]
    lr = optim_cfg.get("lr", 0.001)
    return paddle.optimizer.Adam(learning_rate=lr, parameters=model.parameters())


def main():
    args = parse_args()
    config = load_config(args.config)
    
    # 设置种子
    seed = config["trainer"].get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)
    
    # 设置设备
    device = config["trainer"].get("device", "auto")
    if device == "auto":
        device = "gpu" if paddle.is_compiled_with_cuda() else "cpu"
    paddle.set_device(device)
    print(f"Using device: {device}")
    
    # 创建输出目录
    save_dir = args.output_dir or config["log"].get("save_dir", "./output")
    os.makedirs(save_dir, exist_ok=True)
    print(f"Output directory: {save_dir}")
    
    # 加载数据
    print("Loading dataset...")
    atoms_list = read(config["dataset"]["path"], index=":")
    if not isinstance(atoms_list, list):
        atoms_list = [atoms_list]
    print(f"Loaded {len(atoms_list)} structures")
    
    # 构建图数据集
    cutoff = config["dataset"]["cutoff"]
    graphs = [_atoms_to_graph(atoms, cutoff) for atoms in tqdm(atoms_list)]
    train_graphs, valid_graphs = _split_graphs(
        graphs, 
        valid_ratio=config["dataset"].get("valid_ratio", 0.1),
        seed=seed
    )
    print(f"Train: {len(train_graphs)}, Valid: {len(valid_graphs)}")
    
    # 计算统计信息
    energy_mean, energy_std = _compute_stats(train_graphs)
    print(f"Energy mean: {energy_mean:.6f}, std: {energy_std:.6f}")
    
    # 构建数据加载器
    train_dataset = GraphDataset(train_graphs)
    valid_dataset = GraphDataset(valid_graphs)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["dataloader"]["batch_size"],
        shuffle=config["dataloader"].get("shuffle", True),
        collate_fn=_collate_graphs,
        num_workers=config["dataloader"].get("num_workers", 0),
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config["dataloader"]["batch_size"],
        shuffle=False,
        collate_fn=_collate_graphs,
        num_workers=config["dataloader"].get("num_workers", 0),
    )
    
    # 构建模型和优化器
    model = build_model(config)
    optimizer = build_optimizer(config, model)
    print(f"Model parameters: {sum(p.size for p in model.parameters())}")
    
    # 训练循环
    epoch = config["trainer"]["epoch"]
    force_weight = config["loss"].get("force_loss_weight", 0.1)
    best_valid = float("inf")
    
    for ep in range(1, epoch + 1):
        print(f"\nEpoch {ep}/{epoch}")
        
        train_stats = _run_epoch(model, train_loader, energy_mean, energy_std, force_weight, optimizer)
        valid_stats = _run_epoch(model, valid_loader, energy_mean, energy_std, force_weight)
        
        print(f"Train: loss={train_stats['loss']:.6f}, e_loss={train_stats['e_loss']:.6f}, f_loss={train_stats['f_loss']:.6f}")
        print(f"Valid: loss={valid_stats['loss']:.6f}, e_loss={valid_stats['e_loss']:.6f}, f_loss={valid_stats['f_loss']:.6f}")
        
        # 保存模型
        save_obj = {
            "model_state_dict": model.state_dict(),
            "energy_mean": energy_mean,
            "energy_std": energy_std,
            "config": config,
        }
        
        if valid_stats["loss"] < best_valid:
            best_valid = valid_stats["loss"]
            paddle.save(save_obj, os.path.join(save_dir, "best_model.pdparams"))
            print(f"Best model saved")
        
        if ep % config["log"].get("save_interval", 1) == 0:
            paddle.save(save_obj, os.path.join(save_dir, f"model_epoch_{ep}.pdparams"))
    
    paddle.save(save_obj, os.path.join(save_dir, "model_final.pdparams"))
    print(f"\nTraining finished. Final model saved to {save_dir}")


if __name__ == "__main__":
    main()
