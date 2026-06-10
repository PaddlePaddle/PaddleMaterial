# MACE-MP-0 模型复现

## 任务简介
MACE-MP-0 是一种基于等变图神经网络的通用原子间势函数（Universal Interatomic Potential），在 Materials Project Trajectory（MPtrj）数据集上训练，覆盖 89 种元素，能够以 DFT 精度预测能量、力和应力。

## 复现目标
对齐原始 MACE-MP-0 模型（`mace-torch 0.3.5`, `2023-12-03-mace-128-L1_epoch-199.model`）的推理精度，评估指标包括能量 MAE、力 MAE、Convex Hull Distance MAE 等。

## 模型简介
MACE（Many-body Atomic Cluster Expansion）采用高阶等变消息传递架构，单一层内即可捕捉三体和四体关联，主要特点包括：
- **等变性**：旋转、平移、置换不变性内建于架构
- **高阶关联**：单一层捕捉 3-body 和 4-body 相互作用
- **元素覆盖**：支持 89 种元素（H 至 Bi，含镧系）
- **计算效率**：CPU 推理友好，单卡 GPU 可扩展至百万原子体系

## 数据集

MACE-MP-0 在 **Materials Project Trajectory（MPtrj）** 数据集上训练，包含 1.58M 结构、89 种元素。

完整的数据集说明、下载链接、预处理脚本和验证方法请参见 [DATASET.md](DATASET.md)。

## 环境依赖
- PaddlePaddle 3.2.2 及以上（官方正式发布版本）
- Python 3.9 及以上
- numpy >= 1.25.0
- ase >= 3.22.1
- pymatgen >= 2023.7.14

## 训练命令
```bash
python interatomic_potentials/train.py \
    --config=interatomic_potentials/configs/mace/mace_mp0_medium.yaml
```

关键超参数（对齐原始 MACE-MP-0 Medium 配置）：
| 参数 | 值 | 说明 |
|------|-----|------|
| hidden_dim | 128 | 隐藏层维度 |
| num_layers | 2 | 等变消息传递层数 |
| num_basis | 8 | 径向基函数数量 |
| r_max | 6.0 Å | 图构建截断半径 |
| num_elements | 89 | 最大原子序数 |
| batch_size | 8 | 每批次结构数 |
| learning_rate | 1.0e-3 | 初始学习率（Cosine 衰减至 1.0e-6） |
| force_weight | 100.0 | 力损失权重 |
| max_epochs | 200 | 训练轮数 |

## 评测命令
```bash
python interatomic_potentials/evaluate.py \
    --config=interatomic_potentials/configs/mace/mace_mp0_medium.yaml \
    --model_path=checkpoints/mace_mp0_best.pdparams \
    --test_set=mptrj_test
```

## 推理命令
```bash
python interatomic_potentials/predict.py \
    --config=interatomic_potentials/configs/mace/mace_mp0_medium.yaml \
    --model_path=checkpoints/mace_mp0_best.pdparams \
    --structure_file=data/test.cif
```

## 评估指标与对齐精度

### 验收标准

#### 1. 单卡前向精度对齐
使用相同的输入结构和原子位置，对比 PaddlePaddle 复现版本与原始 PyTorch 版本的前向输出：
- **能量对齐**：体系总能量 diff 控制在 **1e-4** 量级（eV）
- **力对齐**：原子力矢量 diff 控制在 **1e-4** 量级（eV/Å）
- **应力对齐**：应力张量 diff 控制在 **1e-4** 量级（kBar）

#### 2. 反向对齐（训练一致性）
使用相同的训练配置和数据划分，对比训练过程：
- **Loss 一致性**：训练 2 轮以上，每轮 loss 与原始实现一致
- **梯度对齐**：模型参数梯度 diff 控制在 **1e-4** 量级

#### 3. 监督类任务精度对齐
在 MPtrj 测试集和 Matbench Discovery 基准上评估：
- **能量 MAE**：与原始 MACE-MP-0 的 metric 误差控制在 **1%** 以内
- **力 MAE**：与原始 MACE-MP-0 的 metric 误差控制在 **1%** 以内
- **应力 MAE**：与原始 MACE-MP-0 的 metric 误差控制在 **1%** 以内

#### 4. 性能优化
对比飞桨深度学习编译器（CINN）开启前后的训推性能：
- **推理加速**：单卡推理速度平均提升 **30%** 以上
- **训练加速**：单卡训练速度平均提升 **30%** 以上

### 评估方式
| 评估项 | 测试集 | 评估指标 | 对齐标准 |
|--------|--------|----------|----------|
| 前向精度 | MPtrj 随机采样结构 | 能量/力/应力 diff | 1e-4 |
| 反向对齐 | MPtrj 训练集前 2 轮 | Loss diff | 一致 |
| 监督精度 | MPtrj 测试集 + Matbench Discovery | MAE 相对误差 | < 1% |
| 性能优化 | 标准 MD 模拟任务 | 训推速度提升 | > 30% |

## 关键配置说明
配置文件 `mace_mp0_medium.yaml` 包含以下关键参数：
- `model.hidden_dim=128`：对齐原始 Medium 模型
- `model.num_layers=2`：对齐原始 2 层等变层配置
- `model.r_max=6.0`：图构建截断半径，与原始一致
- `loss.force_weight=100.0`：力损失权重，与原始训练配置一致
- `optimizer.lr_scheduler=CosineAnnealingLR`：学习率余弦衰减
- `trainer.max_epochs=200`：训练轮数，与原始 checkpoint 对应

## 参考论文与来源
- **原始论文**：Batatia et al., "A foundation model for atomistic materials chemistry", arXiv:2401.00096 (2024)
- **原始代码**：https://github.com/ACEsuit/mace
- **预训练权重**：https://github.com/ACEsuit/mace-foundations/releases
- **Matbench Discovery 基准**：https://matbench-discovery.materialsproject.org/models/mace-mp-0
