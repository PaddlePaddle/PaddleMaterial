# MatInvent

论文：[MatInvent: Combining Diffusion Models with Reinforcement Learning for Crystal Invention](https://arxiv.org/abs/2503.09787)

## 简介

MatInvent 是一个结合了扩散模型和强化学习微调循环的晶体生成框架。它使用预训练的扩散模型（MatterGen 或 DiffCSP）作为骨干网络，生成候选晶体结构，通过性质计算器对其进行评分，并使用奖励加权损失对生成器进行微调。经过几轮迭代后，生成器能够产生匹配用户指定性质目标的新稳定晶体。

## 配置文件

| 配置文件 | 骨干网络 | 默认奖励目标 |
|--------|----------|-----------------------|
| `matinvent_mattergen.yaml` | MatterGen | band_gap (升序) |
| `matinvent_diffcsp.yaml`   | DiffCSP   | formation_energy (降序) |

两个配置文件都遵循与正常训练配置相同的结构，但增加了一个额外的 `RL:` 部分。

## 关键 RL 参数

| 参数 | 描述 |
|-----------|-------------|
| `rl_epoch` | 运行多少轮 RL 外循环 |
| `topk_ratio` | 每轮保留用于微调的顶级结构比例 |
| `sample_cfg.num_samples` | 每轮生成多少个结构 |
| `finetune_cfg.lr` | 微调学习率 |
| `reward_cfg.prop_cfg` | 目标性质列表（名称、方向、数值范围） |
| `replay_cfg.capacity` | 经验回放缓冲区最大容量 |
| `div_filter_cfg.method` | 多样性过滤方法（`composition` 或 `rmsd`） |

## 使用方法

### 1. RL 微调

```bash
# MatterGen 骨干网络
python structure_generation/train.py \
    --config structure_generation/configs/matinvent/matinvent_mattergen.yaml

# DiffCSP 骨干网络
python structure_generation/train.py \
    --config structure_generation/configs/matinvent/matinvent_diffcsp.yaml
```

### 2. 采样

```bash
# 使用预训练模型
python structure_generation/sample.py \
    --config_path structure_generation/configs/matinvent/matinvent_mattergen.yaml \
    --checkpoint_path ./output/matinvent_diffcsp/models/final/model.pdparams \
    --save_path results/matinvent_mattergen \
    --mode by_dataloader

# 使用 RL 微调后的模型 
python structure_generation/sample.py \
    --config_path structure_generation/configs/matinvent/matinvent_mattergen.yaml \
    --checkpoint_path output/matinvent_mattergen/models/final/model.pdparams \
    --save_path results/matinvent_mattergen_rl \
    --mode by_dataloader
```

对于 DiffCSP 骨干网络，只需将配置文件替换为 `matinvent_diffcsp.yaml`。
