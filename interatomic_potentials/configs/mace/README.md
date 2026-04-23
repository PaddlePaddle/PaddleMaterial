# MACE Model for Interatomic Potentials

## 任务简介
MACE (Many-body Atomic Cluster Expansion) 是一种基于原子聚类展开的机器学习原子间势函数模型，用于预测分子和晶体的能量和力。

## 模型简介
MACE 模型采用等变图神经网络架构，能够处理不同原子类型和结构的能量和力预测。其主要特点包括：
- 等变性：保持旋转、平移和置换不变性
- 局部性：只考虑近邻原子的相互作用
- 表达能力强：能够捕捉复杂的原子间相互作用

## 数据准备方式
MACE 支持与原始 MACE 仓库相同的数据格式，数据应以 XYZ 格式提供，并包含能量和力信息。

### 数据格式示例：
```
2
Energy=-10.0
O 0.0 0.0 0.0
H 0.0 0.0 1.0
```

### 数据放置：
将训练和测试数据放在 `data/` 目录中，例如：
- `data/train.xyz` - 训练数据
- `data/test.xyz` - 测试数据
- `data/valid.xyz` - 验证数据

### 样本数据：
`data/sample_data/` 目录中提供了一个小的样本数据集，用于测试目的。

## 环境依赖
- PaddlePaddle 3.2.2 及以上
- Python 3.7 及以上
- 其他依赖：numpy, ase

## 训练命令
```bash
python interatomic_potentials/train.py \
    --config=interatomic_potentials/configs/mace/mace.yaml
```

## 评测命令
```bash
python interatomic_potentials/evaluate.py \
    --config=interatomic_potentials/configs/mace/mace.yaml \
    --model_path=checkpoints/mace_best.pdparams
```

## 推理命令
```bash
python interatomic_potentials/predict.py \
    --config=interatomic_potentials/configs/mace/mace.yaml \
    --model_path=checkpoints/mace_best.pdparams \
    --input_file=data/test.xyz \
    --output_file=predictions.json
```

## 关键配置说明
配置文件 `mace.yaml` 包含以下关键参数：
- `model.hidden_dim`：隐藏层维度
- `model.num_layers`：网络层数
- `model.num_basis`：径向基函数数量
- `model.r_max`：截断半径
- `model.num_elements`：元素数量
- `optimizer.learning_rate`：学习率
- `loss.force_weight`：力损失权重

## 参考结果
在样本数据集上的参考结果：
- 能量预测 MAE：< 0.1 eV
- 力预测 MAE：< 0.5 eV/Å

## 参考论文或来源链接
- MACE: Many-body Atomic Cluster Expansion for Molecular Properties
- https://github.com/ACEsuit/mace
