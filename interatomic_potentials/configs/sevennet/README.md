# SevenNet - Interatomic Potential Model

## 任务简介

SevenNet 是一个基于图神经网络的机器学习原子间势函数模型，用于预测分子和材料的能量与力。

## 模型简介

SevenNet 模型结构包括：
- **原子嵌入层**: 将原子序数映射为特征向量
- **径向基函数 (RBF)**: 编码原子间距离信息
- **消息传递块**: 多层图神经网络进行信息聚合
- **能量预测头**: 预测原子能量并求和得到总能量

## 环境依赖

```bash
# 安装 PaddlePaddle
pip install paddlepaddle==3.3.1

# 安装依赖
pip install ase numpy tqdm pyyaml
```

## 数据准备

数据集格式：extxyz 格式，包含原子坐标、能量和力标签。

示例数据位于：`tests/data/systems/hfo2.extxyz`

## 训练命令

```bash
# 使用默认配置训练
cd interatomic_potentials
python train.py --config configs/sevennet/sevennet_hfo2.yaml

# 指定输出目录
python train.py --config configs/sevennet/sevennet_hfo2.yaml --output-dir ./output/sevennet_hfo2
```

## 推理命令

```bash
cd interatomic_potentials
python infer.py --model ./output/sevennet_hfo2/model_final.pdparams --structure tests/data/systems/hfo2.extxyz
```

## 配置说明

### 模型配置 (`model`)
- `type`: 模型类型，固定为 "sevennet"
- `num_species`: 原子种类数，默认 100
- `hidden_dim`: 隐藏层维度，默认 128
- `num_message_layers`: 消息传递层数，默认 4
- `num_rbf`: 径向基函数数量，默认 32
- `cutoff`: 截断半径（Å），默认 5.0

### 数据集配置 (`dataset`)
- `type`: 数据集类型，固定为 "extxyz"
- `path`: 数据集路径
- `cutoff`: 截断半径
- `valid_ratio`: 验证集比例，默认 0.1

### 训练配置 (`trainer`)
- `epoch`: 训练轮数
- `per_epoch`: 每多少轮保存一次
- `seed`: 随机种子
- `device`: 设备，"auto"、"gpu" 或 "cpu"

## 参考结果

使用 `sevennet_hfo2.yaml` 配置训练 HfO2 数据集：
- 训练轮数：2
- 初始训练损失：~888
- 最终验证损失：~794

## 目录结构

```
interatomic_potentials/
├── train.py              # 训练入口脚本
├── infer.py              # 推理入口脚本
└── configs/
    └── sevennet/
        ├── sevennet_hfo2.yaml        # 默认配置
        ├── sevennet_hfo2_large.yaml  # 大型配置
        └── README.md                 # 本文件
```

## 代码结构

```
ppmat/
└── models/
    └── sevennet/
        ├── __init__.py
        └── sevennet_model.py  # SevenNet 模型实现
```

## 参考

本模型参考了 SevenNet 系列原子间势函数的设计理念。
