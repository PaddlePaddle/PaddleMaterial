# GDI-NN: Gibbs-Duhem-Informed Neural Networks for Binary Activity Coefficient Prediction

## 任务简介

GDI-NN (Gibbs-Duhem-Informed Neural Networks) 是一种基于热力学约束的图神经网络模型，用于预测二元溶剂混合物的活度系数 (activity coefficient, γ)。活度系数是描述非理想溶液中组分偏离理想行为的重要热力学参数，广泛应用于化工过程设计和模拟。

本复现任务实现了 GDI-NN 论文中的核心模型，包括：
- **SolvGNN**：基于图神经网络的溶剂活度系数预测模型
- **SolvGNNxMLP**：SolvGNN 变体，将组成信息在 MLP 层引入
- **GEGNN**：Excess Gibbs Free Energy GNN
- **MCM**：Multi-Component Model，基于嵌入的模型（不使用分子图）


## 模型 / 数据集简介

### 模型架构

| 模型 | 描述 | 输入 | 特点 |
|------|------|------|------|
| SolvGNN | 基础图神经网络模型 | 分子图 + 组成 | 组成信息在全局卷积前引入 |
| SolvGNNxMLP | SolvGNN 变体 | 分子图 + 组成 | 组成信息在 MLP 层引入 |
| GEGNN | Excess Gibbs GNN | 分子图 + 组成 | 预测超额 Gibbs 自由能 |
| MCM | 多组分模型 | 溶剂 ID + 组成 | 基于嵌入，不使用分子图 |

### 数据集

**数据来源**：Qin et al. (2023) 创建的二元活度系数数据集

**数据格式** (GDI-NN 格式 CSV)：
```csv
job_id,solv1,solv2,solv1_x,solv2_x,solv1_gamma,solv2_gamma,warnings,solv1_smiles,solv2_smiles,solv1_name,solv2_name,tpsa_binary_avg
0,solvent_587,solvent_604,0.1,0.9,0.47175935,0.00025148,,CN,CC(=O)CC(C)C,METHYL AMINE,METHYL ISOBUTYL KETONE,2
```

**字段说明**：
- `solv1`, `solv2`：溶剂 ID（格式：solvent_xxx）
- `solv1_x`, `solv2_x`：摩尔分数（x₁ + x₂ = 1）
- `solv1_gamma`, `solv2_gamma`：**ln(γ)** 值（注意：存储的是自然对数值）
- `solv1_smiles`, `solv2_smiles`：SMILES 字符串
- `tpsa_binary_avg`：拓扑极性表面积

**溶剂列表格式**：
```csv
solvent_name,solvent_id,smiles_can
"1,1,1-TRICHLOROETHANE",solvent_1,CC(Cl)(Cl)Cl
```

## 数据准备方式

从 [GDI-NN 原项目](https://git.rwth-aachen.de/avt-svt/public/GDI-NN) 获取数据文件：

原始数据文件位于 GDI-NN/data/ 目录，将数据文件放置到指定目录，并更新配置文件中的路径：

```bash
# 目录结构示例
data
├── all_systems_comp_range_step5e-2.csv # 其他数据集，包含温度信息，不使用
├── output_binary_with_inf_all.csv # 主二元活度系数数据
├── output_binary_with_inf_all_extra.csv # 包含额外列，如 names, cas_number等，不使用
└── solvent_list.csv # 溶剂列表
```

## 环境依赖

GDI-NN 核心依赖 `rdkit` `pgl`，PaddleMaterials 正常安装后即可使用。

## 训练命令

```bash
cd /path/to/PaddleMaterials

python property_prediction/train.py \
    -c property_prediction/configs/gdinn/solvgnn_binary_gamma.yaml
```

### 关键训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `max_epochs` | 200 | 最大训练轮数 |
| `batch_size` | 8 | 批次大小 |
| `learning_rate` | 0.001 | 学习率 |
| `hidden_dim` | 256 | 隐藏层维度 |
| `pinn_lambda` | 1.0 | Gibbs-Duhem 损失权重 |

## 评测命令

在配置文件中设置：
```yaml
Global:
  do_train: false
  do_eval: true
```

然后运行：
```bash
python property_prediction/train.py \
    -c property_prediction/configs/gdinn/solvgnn_binary_gamma.yaml \
    Trainer.pretrained_model_path=/path/to/best.pdparams
```

## 关键配置说明

### 模型配置

```yaml
Model:
  __class_name__: "SolvGNN"
  __init_params__:
    in_dim: 74              # 原子特征维度（CanonicalAtomFeaturizer）
    hidden_dim: 256         # 隐藏层维度
    n_classes: 1            # 输出类别数
    mlp_dropout_rate: 0.0   # MLP dropout 率
    mlp_activation: "relu"  # MLP 激活函数
    mpnn_activation: "relu" # MPNN 激活函数
    num_step_message_passing: 1  # 消息传递步数
    pinn_lambda: 1.0        # Gibbs-Duhem 损失权重
```

### 损失函数配置

```yaml
Loss:
  __class_name__: "GibbsDuhemLoss"
  __init_params__:
    lambda_gd: 1.0          # Gibbs-Duhem 损失权重
    loss_type: "mse"        # 损失类型
```

### 数据配置

```yaml
Dataset:
  train:
    dataset:
      __class_name__: "BinaryActivityDataset"
      __init_params__:
        path: "train_binary.csv"           # 训练数据路径
        solvent_list_path: "solvent_list.csv"  # 溶剂列表路径
        add_self_loop: true                # 是否添加自环
        preload_graphs: false              # 是否预加载图
    sampler:
      __class_name__: "BatchSampler"
      __init_params__:
        batch_size: 8
        shuffle: false
        drop_last: false
    loader:
      num_workers: 0
      collate_fn: "BinaryActivityCollator"
```

### 优化器配置

```yaml
Optimizer:
  __class_name__: "Adam"
  __init_params__:
    lr:
      __class_name__: "Step"
      __init_params__:
        learning_rate: 0.001
        step_size: 50       # 每 50 个 epoch 衰减
        gamma: 0.5          # 衰减因子
        by_epoch: true
```

## 参考结果

### 复现目标

复现 GDI-NN 论文中 SolvGNN 模型在二元活度系数预测任务上的性能。

### 所用数据集版本

- **数据集**：output_binary_with_inf_all.csv （280000 条样本中抽取 35374 条训练样本）
- **溶剂列表**：solvent_list.csv （700 种溶剂）

### 关键超参数

| 参数 | 值 |
|------|-----|
| 训练轮数 | 10 |
| 批次大小 | 256 |
| 学习率 | 0.001 |
| 隐藏层维度 | 256 |
| GD 损失权重 | 1.0 |

### 评测指标

- **MSE (Mean Squared Error)**：预测损失
- **GD Loss**：Gibbs-Duhem 约束损失

### 当前复现状态

```shell
[2026/04/21 14:39:28] ppmat INFO: Train: Epoch [1/10] | reader_cost: 0.005887 | batch_cost: 0.308326 | loss(loss): 0.534919 | pred_loss(loss): 0.523145 | gd_loss(loss): 0.011774 | gamma1(metric): 1.405649 | gamma2(metric): 1.047136
...
[2026/04/21 14:40:11] ppmat INFO: Train: Epoch [2/10] | reader_cost: 0.001740 | batch_cost: 0.296518 | loss(loss): 0.399445 | pred_loss(loss): 0.382759 | gd_loss(loss): 0.016685 | gamma1(metric): 1.775483 | gamma2(metric): 1.026949
...
[2026/04/21 14:41:34] ppmat INFO: Train: Epoch [4/10] | reader_cost: 0.000948 | batch_cost: 0.290668 | loss(loss): 0.262740 | pred_loss(loss): 0.250712 | gd_loss(loss): 0.012027 | gamma1(metric): 2.300859 | gamma2(metric): 1.003786
...
[2026/04/21 14:41:34] ppmat INFO: Train: Epoch [4/10] | reader_cost: 0.000948 | batch_cost: 0.290668 | loss(loss): 0.262740 | pred_loss(loss): 0.250712 | gd_loss(loss): 0.012027 | gamma1(metric): 2.300859 | gamma2(metric): 1.003786
...
[2026/04/21 14:42:17] ppmat INFO: Train: Epoch [5/10] | reader_cost: 0.000968 | batch_cost: 0.302154 | loss(loss): 0.197765 | pred_loss(loss): 0.185370 | gd_loss(loss): 0.012395 | gamma1(metric): 2.704353 | gamma2(metric): 1.004758
...
[2026/04/21 14:42:59] ppmat INFO: Train: Epoch [6/10] | reader_cost: 0.002687 | batch_cost: 0.293904 | loss(loss): 0.159561 | pred_loss(loss): 0.150231 | gd_loss(loss): 0.009330 | gamma1(metric): 3.123730 | gamma2(metric): 1.003311
...
[2026/04/21 14:43:41] ppmat INFO: Train: Epoch [7/10] | reader_cost: 0.003025 | batch_cost: 0.292735 | loss(loss): 0.124270 | pred_loss(loss): 0.116944 | gd_loss(loss): 0.007326 | gamma1(metric): 3.461181 | gamma2(metric): 1.002554
...
[2026/04/21 14:44:22] ppmat INFO: Train: Epoch [8/10] | reader_cost: 0.001628 | batch_cost: 0.293956 | loss(loss): 0.106535 | pred_loss(loss): 0.100849 | gd_loss(loss): 0.005686 | gamma1(metric): 3.570331 | gamma2(metric): 1.002322
...
[2026/04/21 14:45:05] ppmat INFO: Train: Epoch [9/10] | reader_cost: 0.001512 | batch_cost: 0.299606 | loss(loss): 0.092665 | pred_loss(loss): 0.087616 | gd_loss(loss): 0.005049 | gamma1(metric): 3.801551 | gamma2(metric): 1.001646
...
[2026/04/21 14:45:47] ppmat INFO: Train: Epoch [10/10] | reader_cost: 0.001908 | batch_cost: 0.295166 | loss(loss): 0.078702 | pred_loss(loss): 0.074575 | gd_loss(loss): 0.004127 | gamma1(metric): 3.935121 | gamma2(metric): 1.001604

```

结论： loss 正在显著且稳定地下降

## 参考论文或来源链接

### 论文

1. **GDI-NN 论文**：
   - Rittig, J. G., Felton, K. C., Lapkin, A. A., & Mitsos, A. (2023). Gibbs-Duhem-Informed Neural Networks for Binary Activity Coefficient Prediction. *Digital Discovery*, 2(6), 1752-1767.
   - DOI: [10.1039/D3DD00103B](https://doi.org/10.1039/D3DD00103B)

### 代码仓库

- **GDI-NN 原项目**：https://git.rwth-aachen.de/avt-svt/public/GDI-NN
- **原始 SolvGNN**：https://github.com/zavalab/ML/tree/master/SolvGNN
