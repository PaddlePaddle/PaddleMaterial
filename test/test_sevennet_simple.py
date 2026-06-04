#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Simple test for SevenNet model"""
import sys
import os
import paddle
import numpy as np

# 添加项目路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# 直接导入模型文件
import importlib.util
model_path = os.path.join(PROJECT_ROOT, "ppmat/models/sevennet/sevennet_model.py")
spec = importlib.util.spec_from_file_location("sevennet_model", model_path)
sevennet_model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sevennet_model)


def test_sevennet():
    print("Testing SevenNet model...")
    
    # 创建模型
    model = sevennet_model.PurePaddleSevenNet(
        num_species=100,
        hidden_dim=32,
        num_message_layers=3,
        num_rbf=32,
        cutoff=4.0,
    )
    
    # 创建测试输入
    z = paddle.to_tensor([1, 8, 1], dtype="int64")
    pos = paddle.to_tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype="float32")
    edge_index = paddle.to_tensor([[0, 1, 1, 2, 0, 2], [1, 0, 2, 1, 2, 0]], dtype="int64")
    
    graph = {"z": z, "pos": pos, "edge_index": edge_index}
    
    # 前向传播
    out = model(graph)
    
    print(f"✓ Forward pass OK")
    print(f"  Total energy: {float(out['total_energy'].item()):.6f} eV")
    
    # 测试梯度
    pos_grad = paddle.to_tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype="float32")
    pos_grad.stop_gradient = False
    graph_grad = {"z": z, "pos": pos_grad, "edge_index": edge_index}
    
    out_grad = model(graph_grad)
    forces = -paddle.grad(
        outputs=[out_grad["total_energy"]],
        inputs=[pos_grad],
        create_graph=False,
        retain_graph=False,
    )[0]
    
    print(f"✓ Gradient computation OK")
    print(f"  Forces shape: {forces.shape}")
    
    return True


if __name__ == "__main__":
    try:
        test_sevennet()
        print("\n✅ All tests passed!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
