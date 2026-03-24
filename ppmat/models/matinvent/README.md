# MatInvent

MatInvent 是一个结合扩散模型与强化学习的材料生成框架，用于定向设计具有特定性质的晶体材料。

## 功能特点

- **扩散模型集成**：支持 MatterGen 和 DiffCSP 作为生成器 backbone
- **强化学习优化**：通过奖励函数引导生成过程，实现定向材料设计
- **多属性评估**：支持多种材料性质计算器，如密度、价格、丰度等
- **内存管理**：包含回放缓冲区和长期记忆模块，提升训练效率

## 目录结构

```
matinvent/
├── rl/                 # 强化学习核心代码
│   ├── models/         # 模型套件实现
│   ├── samplers.py     # 采样器
│   └── mat_invent.py   # MatInvent 主类
├── rewards/            # 奖励系统
│   ├── calculators/    # 性质计算器
│   └── reward.py       # 奖励函数
├── memory/             # 内存管理
│   ├── replay_buffer.py # 回放缓冲区
│   └── ltm.py          # 长期记忆
├── rl_train.py         # 强化学习训练脚本
└── rl_wrapper.py       # 包装器，用于集成到现有训练流程
```

## 核心组件

1. **MatInvent**：强化学习训练的核心类，实现了采样、评估、微调的完整循环
2. **奖励系统**：计算材料性质并转换为奖励信号
3. **模型套件**：封装了 MatterGen 和 DiffCSP 模型的加载和使用
4. **内存模块**：存储和管理生成的材料结构

## 使用方法

### 强化学习训练

```bash
# 使用统一入口执行 RL 训练
python structure_generation/train.py \
    --config structure_generation/configs/matinvent/matinvent_mattergen.yaml
```

### 采样

```bash
# 使用训练好的模型进行采样
python structure_generation/sample.py \
    --config_path structure_generation/configs/matinvent/matinvent_mattergen.yaml \
    --checkpoint_path output/matinvent_mattergen/models/final/model.pdparams \
    --save_path results/matinvent_mattergen \
    --mode by_dataloader
```

## 配置文件

配置文件位于 `structure_generation/configs/matinvent/` 目录，包含以下主要部分：

- **Global**：全局设置
- **Model**：模型配置
- **RL**：强化学习参数
- **Sample**：采样设置

## 支持的性质计算器

- **PyMatGen**：密度、价格、丰度等基本性质
- **SynScore**：可合成性评分
- **FairChem**：基于 FairChem 的性质预测
- **DFT**：基于密度泛函理论的性质计算
