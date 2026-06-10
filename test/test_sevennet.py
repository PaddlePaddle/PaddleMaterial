#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test file for SevenNet model

Usage:
    python test/test_sevennet.py
    pytest test/test_sevennet.py -v
"""

import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import paddle
import numpy as np


def import_model():
    """Import SevenNet model dynamically to avoid ppmat dependencies"""
    import importlib.util
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(project_root, "ppmat/models/sevennet/sevennet_model.py")
    
    spec = importlib.util.spec_from_file_location("sevennet_model", model_path)
    sevennet_model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sevennet_model)
    
    return sevennet_model.PurePaddleSevenNet


def test_forward_pass():
    """Test basic forward pass"""
    PurePaddleSevenNet = import_model()
    
    model = PurePaddleSevenNet(
        num_species=100,
        hidden_dim=16,
        num_message_layers=2,
        num_rbf=16,
        cutoff=4.0,
    )
    
    z = paddle.to_tensor([1, 8, 1], dtype="int64")
    pos = paddle.to_tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype="float32")
    edge_index = paddle.to_tensor([[0, 1, 1, 2, 0, 2], [1, 0, 2, 1, 2, 0]], dtype="int64")
    
    graph = {"z": z, "pos": pos, "edge_index": edge_index}
    out = model(graph)
    
    assert "total_energy" in out, "Missing total_energy"
    assert "atomic_energy" in out, "Missing atomic_energy"
    assert out["total_energy"].shape == (), f"Expected scalar, got {out['total_energy'].shape}"
    assert out["atomic_energy"].shape == (3, 1), f"Expected (3, 1), got {out['atomic_energy'].shape}"
    assert not paddle.isnan(out["total_energy"]), "Total energy is NaN"
    assert not paddle.isinf(out["total_energy"]), "Total energy is Inf"
    
    print("✓ test_forward_pass passed")


def test_backward_pass():
    """Test backward pass"""
    PurePaddleSevenNet = import_model()
    
    model = PurePaddleSevenNet(
        num_species=100,
        hidden_dim=16,
        num_message_layers=2,
        num_rbf=16,
        cutoff=4.0,
    )
    
    z = paddle.to_tensor([1, 8, 1], dtype="int64")
    pos = paddle.to_tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype="float32")
    pos.stop_gradient = False
    edge_index = paddle.to_tensor([[0, 1, 1, 2, 0, 2], [1, 0, 2, 1, 2, 0]], dtype="int64")
    
    graph = {"z": z, "pos": pos, "edge_index": edge_index}
    out = model(graph)
    loss = out["total_energy"]
    loss.backward()
    
    assert pos.grad is not None, "Position gradient is None"
    assert pos.grad.shape == pos.shape, "Gradient shape mismatch"
    assert not paddle.any(paddle.isnan(pos.grad)), "Gradient contains NaN"
    assert not paddle.any(paddle.isinf(pos.grad)), "Gradient contains Inf"
    
    print("✓ test_backward_pass passed")


def test_force_computation():
    """Test force computation via gradient"""
    PurePaddleSevenNet = import_model()
    
    model = PurePaddleSevenNet(
        num_species=100,
        hidden_dim=16,
        num_message_layers=2,
        num_rbf=16,
        cutoff=4.0,
    )
    
    z = paddle.to_tensor([1, 8, 1], dtype="int64")
    pos = paddle.to_tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype="float32")
    pos.stop_gradient = False
    edge_index = paddle.to_tensor([[0, 1, 1, 2, 0, 2], [1, 0, 2, 1, 2, 0]], dtype="int64")
    
    graph = {"z": z, "pos": pos, "edge_index": edge_index}
    out = model(graph)
    forces = -paddle.grad(
        outputs=[out["total_energy"]],
        inputs=[pos],
        create_graph=False,
        retain_graph=False,
    )[0]
    
    assert forces.shape == pos.shape, f"Forces shape mismatch: {forces.shape} vs {pos.shape}"
    assert not paddle.any(paddle.isnan(forces)), "Forces contain NaN"
    assert not paddle.any(paddle.isinf(forces)), "Forces contain Inf"
    
    print("✓ test_force_computation passed")


def test_model_reproducibility():
    """Test model reproducibility with same seed"""
    PurePaddleSevenNet = import_model()
    
    paddle.seed(42)
    model1 = PurePaddleSevenNet(
        num_species=100,
        hidden_dim=16,
        num_message_layers=2,
        num_rbf=16,
        cutoff=4.0,
    )
    
    paddle.seed(42)
    model2 = PurePaddleSevenNet(
        num_species=100,
        hidden_dim=16,
        num_message_layers=2,
        num_rbf=16,
        cutoff=4.0,
    )
    
    z = paddle.to_tensor([1, 8, 1], dtype="int64")
    pos = paddle.to_tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype="float32")
    edge_index = paddle.to_tensor([[0, 1, 1, 2, 0, 2], [1, 0, 2, 1, 2, 0]], dtype="int64")
    graph = {"z": z, "pos": pos, "edge_index": edge_index}
    
    out1 = model1(graph)
    out2 = model2(graph)
    
    diff = paddle.abs(out1["total_energy"] - out2["total_energy"])
    assert diff < 1e-6, f"Models not reproducible: diff={diff}"
    
    print("✓ test_model_reproducibility passed")


def test_empty_graph():
    """Test behavior with empty graph"""
    PurePaddleSevenNet = import_model()
    
    model = PurePaddleSevenNet(
        num_species=100,
        hidden_dim=16,
        num_message_layers=2,
        num_rbf=16,
        cutoff=4.0,
    )
    
    z = paddle.to_tensor([], dtype="int64")
    pos = paddle.to_tensor([], dtype="float32").reshape([0, 3])
    edge_index = paddle.to_tensor([[], []], dtype="int64")
    
    graph = {"z": z, "pos": pos, "edge_index": edge_index}
    out = model(graph)
    
    assert out["total_energy"].shape == ()
    
    print("✓ test_empty_graph passed")


if __name__ == "__main__":
    print("Running SevenNet tests...\n")
    
    try:
        test_forward_pass()
        test_backward_pass()
        test_force_computation()
        test_model_reproducibility()
        test_empty_graph()
        
        print("\n✅ All tests passed!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)