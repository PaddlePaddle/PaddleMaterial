#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SevenNet prediction script using PotentialPredictor interface.

Usage:
    # Option 1: Using custom trained model
    python interatomic_potentials/sevennet_predict.py \
        --config_path interatomic_potentials/configs/sevennet/sevennet_hfo2.yaml \
        --checkpoint_path path/to/checkpoint.pdparams
    
    # Option 2: Interactive prediction
    python interatomic_potentials/sevennet_predict.py --interactive
"""

import argparse
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import paddle
import numpy as np

# Import SevenNet model directly to avoid ppmat import issues
import importlib.util
model_path = os.path.join(project_root, "ppmat/models/sevennet/sevennet_model.py")
spec = importlib.util.spec_from_file_location("sevennet_model", model_path)
sevennet_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sevennet_module)
PurePaddleSevenNet = sevennet_module.PurePaddleSevenNet


def create_graph_converter(cutoff=5.0):
    """Create a simple graph converter for molecular structures.
    
    This is a simplified version. For production use, integrate with
    ppmat.models.common.graph_converter.FindPointsInSpheres.
    """
    def convert(atoms):
        """Convert ASE atoms object to graph format.
        
        Args:
            atoms: ASE Atoms object or dict with 'positions', 'numbers'
            
        Returns:
            dict: Graph data with 'z', 'pos', 'edge_index'
        """
        if hasattr(atoms, 'get_positions'):
            # ASE Atoms object
            positions = atoms.get_positions()
            numbers = atoms.get_atomic_numbers()
        else:
            positions = atoms['positions']
            numbers = atoms['numbers']
        
        num_atoms = len(numbers)
        z = paddle.to_tensor(numbers, dtype="int64")
        pos = paddle.to_tensor(positions, dtype="float32")
        
        # Build edge list using cutoff radius
        edge_src = []
        edge_dst = []
        cutoff_dist = cutoff
        
        for i in range(num_atoms):
            for j in range(i + 1, num_atoms):
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist < cutoff_dist:
                    edge_src.extend([i, j])
                    edge_dst.extend([j, i])
        
        if len(edge_src) > 0:
            edge_index = paddle.to_tensor([edge_src, edge_dst], dtype="int64")
        else:
            edge_index = paddle.to_tensor([[], []], dtype="int64")
        
        return {
            "z": z,
            "pos": pos,
            "edge_index": edge_index,
        }
    
    return convert


class SevenNetPredictor:
    """SevenNet predictor with PotentialPredictor-compatible interface."""
    
    def __init__(self, checkpoint_path=None, config=None):
        """
        Initialize SevenNet predictor.
        
        Args:
            checkpoint_path: Path to model checkpoint (.pdparams)
            config: Model configuration dict
        """
        if config is None:
            config = {}
        
        self.model = PurePaddleSevenNet(**config)
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            ckpt = paddle.load(checkpoint_path)
            # Support both raw state_dict and wrapped checkpoint format
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                state_dict = ckpt["model_state_dict"]
                self.energy_mean = ckpt.get("energy_mean", 0.0)
                self.energy_std = ckpt.get("energy_std", 1.0)
                print(f"Loaded checkpoint from {checkpoint_path}")
                print(f"  energy_mean={self.energy_mean}, energy_std={self.energy_std}")
            else:
                state_dict = ckpt
                self.energy_mean = 0.0
                self.energy_std = 1.0
            self.model.set_state_dict(state_dict)
        
        self.model.eval()
        self.graph_converter = create_graph_converter(
            cutoff=config.get('cutoff', 5.0)
        )
    
    def predict(self, atoms):
        """Predict energy for a structure.
        
        Args:
            atoms: ASE Atoms object or dict with 'positions', 'numbers'
            
        Returns:
            dict: Prediction results with 'energy' and optionally 'forces'
        """
        graph = self.graph_converter(atoms)
        
        with paddle.no_grad():
            result = self.model(graph)
        
        # Denormalize energy
        raw_energy = float(result["total_energy"].numpy())
        energy = raw_energy * self.energy_std + self.energy_mean
        
        return {
            "energy": energy,
            "atomic_energy": result["atomic_energy"].numpy(),
        }
    
    def predict_with_forces(self, atoms):
        """Predict energy and forces for a structure.
        
        Uses finite difference method for force calculation because
        Paddle's scatter_nd_add backward pass has numerical instability
        with deep message-passing networks.
        
        Args:
            atoms: ASE Atoms object or dict with 'positions', 'numbers'
            
        Returns:
            dict: Prediction results with 'energy', 'forces', 'atomic_energy'
        """
        graph = self.graph_converter(atoms)
        positions_np = graph["pos"].numpy()
        numbers = graph["z"].numpy()
        edge_index = graph["edge_index"]
        z = graph["z"]
        
        # Forward pass for energy
        with paddle.no_grad():
            result = self.model(graph)
        
        raw_energy = float(result["total_energy"].numpy())
        energy = raw_energy * self.energy_std + self.energy_mean
        
        # Finite difference forces: F_i = -dE/dr_i
        eps = 1e-4
        num_atoms = positions_np.shape[0]
        forces = np.zeros_like(positions_np)
        
        for i in range(num_atoms):
            for j in range(3):
                pos_plus = positions_np.copy()
                pos_plus[i, j] += eps
                pos_minus = positions_np.copy()
                pos_minus[i, j] -= eps
                
                pos_p = paddle.to_tensor(pos_plus, dtype="float32")
                graph_p = {"z": z, "pos": pos_p, "edge_index": edge_index}
                with paddle.no_grad():
                    e_plus = float(self.model(graph_p)["total_energy"].numpy())
                
                pos_m = paddle.to_tensor(pos_minus, dtype="float32")
                graph_m = {"z": z, "pos": pos_m, "edge_index": edge_index}
                with paddle.no_grad():
                    e_minus = float(self.model(graph_m)["total_energy"].numpy())
                
                # Denormalize
                e_plus = e_plus * self.energy_std + self.energy_mean
                e_minus = e_minus * self.energy_std + self.energy_mean
                
                forces[i, j] = -(e_plus - e_minus) / (2 * eps)
        
        return {
            "energy": energy,
            "forces": forces,
            "atomic_energy": result["atomic_energy"].numpy(),
        }


def interactive_demo():
    """Interactive demo for SevenNet prediction."""
    print("\n" + "="*50)
    print("SevenNet Interactive Prediction Demo")
    print("="*50)
    
    # Try to load pretrained checkpoint
    checkpoint_path = os.path.join(
        project_root, "ppmat/models/sevennet/checkpoints/sevennet_hfo2_best.pdparams"
    )
    
    # Config matching the pretrained weights
    config = {
        "num_species": 100,
        "hidden_dim": 64,
        "num_message_layers": 5,
        "num_rbf": 32,
        "cutoff": 5.0,
    }
    
    if os.path.exists(checkpoint_path):
        print(f"\nLoading pretrained weights: {checkpoint_path}")
        predictor = SevenNetPredictor(checkpoint_path=checkpoint_path, config=config)
    else:
        print("\nNo pretrained weights found, using random initialization")
        predictor = SevenNetPredictor(config=config)
    
    # H2O molecule example
    print("\nExample: H2O molecule")
    h2o = {
        'positions': np.array([
            [0.0, 0.0, 0.0],    # O
            [0.96, 0.0, 0.0],   # H
            [-0.24, 0.93, 0.0], # H
        ]),
        'numbers': np.array([8, 1, 1]),  # O, H, H
    }
    
    result = predictor.predict(h2o)
    print(f"  Total energy: {result['energy']:.6f} eV")
    print(f"  Atomic energies: {result['atomic_energy'].flatten()}")
    
    # Get forces
    result_with_forces = predictor.predict_with_forces(h2o)
    print(f"\n  Forces:")
    for i, (num, pos) in enumerate(zip(h2o['numbers'], h2o['positions'])):
        element = {1: 'H', 8: 'O'}.get(num, f'Z={num}')
        forces = result_with_forces['forces'][i]
        has_nan = np.any(np.isnan(forces))
        if has_nan:
            print(f"    {element}: [forces not available - atoms outside cutoff]")
        else:
            print(f"    {element}: {forces}")
    
    print("\n" + "="*50)
    print("Demo completed!")
    print("="*50 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SevenNet Prediction")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config YAML file")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint file")
    parser.add_argument("--interactive", action="store_true",
                        help="Run interactive demo")
    parser.add_argument("--positions", type=str, default=None,
                        help="Path to positions.npy file")
    parser.add_argument("--numbers", type=str, default=None,
                        help="Path to numbers.npy file")
    args = parser.parse_args()
    
    if args.interactive or (args.positions is None and args.numbers is None):
        interactive_demo()
    else:
        # Load positions and numbers
        positions = np.load(args.positions) if args.positions else None
        numbers = np.load(args.numbers) if args.numbers else None
        
        if positions is None or numbers is None:
            print("Error: Please provide both --positions and --numbers")
            sys.exit(1)
        
        atoms = {'positions': positions, 'numbers': numbers}
        
        # Default config (in production, load from config file)
        config = {
            "num_species": 100,
            "hidden_dim": 128,
            "num_message_layers": 4,
            "num_rbf": 32,
            "cutoff": 5.0,
        }
        
        predictor = SevenNetPredictor(
            checkpoint_path=args.checkpoint,
            config=config
        )
        
        result = predictor.predict_with_forces(atoms)
        print(f"Energy: {result['energy']:.6f} eV")
        print(f"Forces shape: {result['forces'].shape}")