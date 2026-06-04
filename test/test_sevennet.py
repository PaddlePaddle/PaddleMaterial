#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test for SevenNet model"""
import pytest
import paddle
import numpy as np


import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ppmat.models.sevennet import PurePaddleSevenNet


def test_sevennet_forward():
    """Test SevenNet forward pass"""
    model = PurePaddleSevenNet(
        num_species=100,
        hidden_dim=32,
        num_message_layers=3,
        num_rbf=32,
        cutoff=4.0,
    )
    
    z = paddle.to_tensor([1, 8, 1], dtype="int64")
    pos = paddle.to_tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype="float32")
    edge_index = paddle.to_tensor([[0, 1, 1, 2, 0, 2], [1, 0, 2, 1, 2, 0]], dtype="int64")
    
    graph = {"z": z, "pos": pos, "edge_index": edge_index}
    out = model(graph)
    
    assert "total_energy" in out
    assert "atomic_energy" in out
    assert out["total_energy"].shape == ()
    assert out["atomic_energy"].shape == (3, 1)
    print("✓ test_sevennet_forward passed")


def test_sevennet_batch():
    """Test SevenNet with batch-like input"""
    model = PurePaddleSevenNet(
        num_species=100,
        hidden_dim=64,
        num_message_layers=4,
        num_rbf=64,
        cutoff=5.0,
    )
    

    np.random.seed(42)
    z = paddle.to_tensor(np.random.randint(1, 10, size=10), dtype="int64")
    pos = paddle.to_tensor(np.random.randn(10, 3).astype("float32"), dtype="float32")
    
  
    edge_index = []
    for i in range(10):
        for j in range(i+1, min(i+5, 10)):
            edge_index.append([i, j])
            edge_index.append([j, i])
    edge_index = paddle.to_tensor(np.array(edge_index).T, dtype="int64")
    
    graph = {"z": z, "pos": pos, "edge_index": edge_index}
    out = model(graph)
    
    assert not paddle.any(paddle.isnan(out["total_energy"]))
    assert not paddle.any(paddle.isnan(out["atomic_energy"]))
    print("✓ test_sevennet_batch passed")


def test_sevennet_gradients():
    """Test SevenNet gradient computation"""
    model = PurePaddleSevenNet(
        num_species=100,
        hidden_dim=32,
        num_message_layers=3,
        num_rbf=32,
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
    
    assert forces.shape == pos.shape
    assert not paddle.any(paddle.isnan(forces))
    print("✓ test_sevennet_gradients passed")


def test_sevennet_reproducibility():
    """Test SevenNet reproducibility"""
    paddle.seed(42)
    
    model1 = PurePaddleSevenNet(
        num_species=100,
        hidden_dim=32,
        num_message_layers=3,
        num_rbf=32,
        cutoff=4.0,
    )
    
    paddle.seed(42)
    model2 = PurePaddleSevenNet(
        num_species=100,
        hidden_dim=32,
        num_message_layers=3,
        num_rbf=32,
        cutoff=4.0,
    )
    
    
    for (name1, param1), (name2, param2) in zip(
        model1.named_parameters(), model2.named_parameters()
    ):
        assert paddle.allclose(param1, param2), f"Params {name1} differ"
    
    print("✓ test_sevennet_reproducibility passed")


if __name__ == "__main__":
    test_sevennet_forward()
    test_sevennet_batch()
    test_sevennet_gradients()
    test_sevennet_reproducibility()
    print("\n✓✓✓ All SevenNet tests passed!")
