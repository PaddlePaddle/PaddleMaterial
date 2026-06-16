# MACE-MP-0

MACE-MP-0 是一种基于等变图神经网络的通用原子间势函数，在 Materials Project Trajectory (MPtrj) 数据集上训练。

## Quick Start

### 训练

```bash
python interatomic_potentials/train.py \
    --config=interatomic_potentials/configs/mace/mace_mp0_medium.yaml
```

### 推理

```bash
python interatomic_potentials/predict.py \
    --config=interatomic_potentials/configs/mace/mace_mp0_medium.yaml \
    --model_path=checkpoints/mace_mp0_best.pdparams \
    --structure_file=data/test.cif
```

## Model Architecture

MACE (Many-body Atomic Cluster Expansion) 采用高阶等变消息传递架构：
- **等变性**: 旋转、平移、置换不变性
- **高阶关联**: 单一层捕捉 3-body 和 4-body 相互作用
- **元素覆盖**: 89 种元素 (H 至 Bi)

## Dataset

使用套件集成的 **MPtrj** 数据集：
- 规模: ~1.58M 结构，146K+ 材料
- 理论级别: PBE+U
- 目标: 能量、力、应力

详见 [DATASET.md](DATASET.md)。

## Configuration

| Parameter | Value |
|-----------|-------|
| hidden_dim | 128 |
| num_layers | 2 |
| num_basis | 8 |
| r_max | 6.0 |
| batch_size | 8 |
| max_epochs | 200 |
| force_weight | 100.0 |

## Evaluation

| Metric | Target |
|--------|--------|
| Energy MAE | 1e-4 |
| Force MAE | 1e-4 |
| Stress MAE | 1e-4 |
| Loss Consistency | 一致 |

## References

- Batatia et al., arXiv:2401.00096 (2024)
- https://github.com/ACEsuit/mace
- https://github.com/ACEsuit/mace-foundations/releases
