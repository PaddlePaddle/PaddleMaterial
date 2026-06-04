#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PaddleMaterials - Interatomic Potentials Inference Script
推理入口：python infer.py --model model.pdparams --structure structure.extxyz
"""

import argparse
import os
import sys
import numpy as np
import paddle
from ase.io import read
from ase.neighborlist import neighbor_list

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from ppmat.models.sevennet import PurePaddleSevenNet


def parse_args():
    parser = argparse.ArgumentParser(description="Infer interatomic potential model")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--structure", type=str, required=True, help="Path to structure file")
    parser.add_argument("--cutoff", type=float, default=5.0, help="Cutoff radius")
    return parser.parse_args()


def atoms_to_graph(atoms, cutoff):
    positions = atoms.get_positions().astype("float32")
    numbers = atoms.get_atomic_numbers().astype("int64")
    i, j = neighbor_list("ij", atoms, cutoff)
    edge_index = np.vstack([i, j]).astype("int64")

    return {
        "z": paddle.to_tensor(numbers, dtype="int64"),
        "pos": paddle.to_tensor(positions, dtype="float32"),
        "edge_index": paddle.to_tensor(edge_index, dtype="int64"),
    }


def predict(model, atoms, cutoff, energy_mean, energy_std):
    graph = atoms_to_graph(atoms, cutoff)
    num_atoms = len(atoms)

    with paddle.no_grad():
        out = model(graph)
        raw_total = float(out["total_energy"].numpy())
        pred_total = raw_total * energy_std + energy_mean
        pred_per_atom = pred_total / num_atoms

    return pred_total, pred_per_atom


def main():
    args = parse_args()
    
    # 加载模型
    print(f"Loading model: {args.model}")
    checkpoint = paddle.load(args.model)
    state_dict = checkpoint["model_state_dict"]
    energy_mean = float(checkpoint.get("energy_mean", 0.0))
    energy_std = float(checkpoint.get("energy_std", 1.0))
    
    # 获取配置
    config = checkpoint.get("config", {})
    model_cfg = config.get("model", {})
    
    # 创建模型
    model = PurePaddleSevenNet(
        num_species=model_cfg.get("num_species", 100),
        hidden_dim=model_cfg.get("hidden_dim", 128),
        num_message_layers=model_cfg.get("num_message_layers", 4),
        num_rbf=model_cfg.get("num_rbf", 32),
        cutoff=model_cfg.get("cutoff", args.cutoff),
    )
    model.set_state_dict(state_dict)
    model.eval()
    
    print(f"Model loaded successfully")
    print(f"Energy mean: {energy_mean:.6f}, std: {energy_std:.6f}")
    
    # 读取结构
    print(f"\nReading structure: {args.structure}")
    atoms_list = read(args.structure, index=":")
    if not isinstance(atoms_list, list):
        atoms_list = [atoms_list]
    print(f"Loaded {len(atoms_list)} structures")
    
    # 预测
    print("\n=== Prediction Results ===")
    for idx, atoms in enumerate(atoms_list):
        pred_total, pred_per_atom = predict(model, atoms, args.cutoff, energy_mean, energy_std)
        
        ref_energy = None
        for key in ["y_energy", "energy", "free_energy", "total_energy"]:
            if key in atoms.info:
                try:
                    ref_energy = float(atoms.info[key])
                    break
                except Exception:
                    pass
        
        print(f"\nStructure {idx + 1}:")
        print(f"  Atoms: {len(atoms)}")
        print(f"  Predicted total energy: {pred_total:.6f} eV")
        print(f"  Predicted per-atom energy: {pred_per_atom:.6f} eV/atom")
        
        if ref_energy is not None:
            print(f"  Reference energy: {ref_energy:.6f} eV")
            print(f"  Error: {abs(pred_total - ref_energy):.6f} eV")
    
    print("\nInference completed!")


if __name__ == "__main__":
    main()
