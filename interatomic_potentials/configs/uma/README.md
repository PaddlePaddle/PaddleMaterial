# UMA

[UMA: A Family of Universal Models for Atoms](https://github.com/facebookresearch/fairchem/tree/main/src/fairchem/core/models/uma)

## 简介

UMA 是一个面向原子体系的通用机器学习势能模型。本迁移版本在
PaddleMaterials 中提供了基于 eSCN 的 UMA backbone、直接能量/力预测
head，以及接入 `interatomic_potentials/train.py` 的训练流程。

当前主配置面向 OMat24 `rattled-500` S2EF 任务：从 ASELMDB 结构数据中预测
体系总能量和原子力。

## 数据集

示例配置使用 OMat24 `rattled-500` ASELMDB 数据：

| 划分 | 默认路径 | 默认使用样本数 |
| :-- | :-- | --: |
| Train | `./data/omat24/train/rattled-500/train.aselmdb` | 128 |
| Val | `./data/omat24/val/rattled-500/val.aselmdb` | 64 |
| Test | `./data/omat24/test/rattled-500/test.aselmdb` | 64 |

标签字段如下：

| 字段 | 含义 |
| :-- | :-- |
| `energy` | 结构总能量 |
| `forces` | 原子力，形状为 `[num_atoms, 3]` |

数据可放在 `./data/omat24` 下，也可以通过命令行覆盖
`Dataset.*.dataset.__init_params__.src` 指向本地实际路径。

## 环境依赖

除 PaddleMaterials 常规依赖外，UMA 数据读取还需要：

```bash
pip install ase lmdb e3nn omegaconf
```

`ppmat/models/uma/Jd.pt` 保存 UMA 旋转模块需要的预计算 Wigner-d 系数。如果该
文件放在其他位置，可以在启动训练前设置 `PPMAT_UMA_JD_PATH`。

## 训练

单卡训练：

```bash
python interatomic_potentials/train.py -c interatomic_potentials/configs/uma/uma_omat24_r500_s2ef.yaml
```

多卡训练：

```bash
python -m paddle.distributed.launch --gpus="0,1,2,3" interatomic_potentials/train.py -c interatomic_potentials/configs/uma/uma_omat24_r500_s2ef.yaml
```

如需使用完整本地 split，可以删除配置中的 `select_args.limit`，或通过命令行覆盖
对应字段。

## OC20 数据准备

OC20 官方 S2EF `200k` 训练包可以直接下载，不需要额外授权：

```bash
mkdir -p ./data/oc20/raw
curl -L https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_train_200K.tar -o ./data/oc20/raw/s2ef_train_200K.tar
tar -xf ./data/oc20/raw/s2ef_train_200K.tar -C ./data/oc20/raw
```

官方包内数据是 `*.extxyz.xz` 和 `*.txt.xz`。当前 UMA 数据适配器默认读取
ASELMDB，因此本目录提供了一个轻量转换脚本，将 OC20 S2EF extxyz 流式转换为
UMA 可直接读取的 `.aselmdb`：

```bash
python interatomic_potentials/configs/uma/prepare_oc20_s2ef_aselmdb.py \
  --raw-dir ./data/oc20/raw/s2ef_train_200K/s2ef_train_200K \
  --out-dir ./data/oc20/uma_aselmdb \
  --train 1000 \
  --val 100 \
  --test 100
```

转换后目录结构如下：

```text
data/oc20/uma_aselmdb/
  train/train.aselmdb
  val/val.aselmdb
  test/test.aselmdb
```

OC20 单域训练配置：

```bash
python interatomic_potentials/train.py -c interatomic_potentials/configs/uma/uma_oc20_200k_s2ef.yaml
```

说明：这里的 val/test 是从官方 `s2ef_train_200K` 中切出的轻量验证划分，主要用于
快速验证 OC20 task 的数据加载和训练链路。如需严格 benchmark，应下载官方
`s2ef_val_id.tar`、OOD validation 或 test split，并按相同方式转换。

### 半天预算验证数据

全量 OMat24/OC20 训练数据量很大。为了在单卡预算内完成更具代表性的迁移验证，
本目录提供了固定预算数据准备脚本和配置：

| 配置 | Train | Val | Test | Epoch |
| :-- | --: | --: | --: | --: |
| `uma_omat24_r500_budget_s2ef.yaml` | OMat24 rattled-500 train 50k | OMat24 rattled-500 val offset 0, 5k | OMat24 rattled-500 val offset 5k, 5k | 3 |
| `uma_oc20_50k_budget_s2ef.yaml` | OC20 S2EF train 50k | OC20 S2EF val-id offset 0, 5k | OC20 S2EF val-id offset 5k, 5k | 2 |

准备数据：

```bash
bash interatomic_potentials/configs/uma/prepare_budget_data.sh ./data
```

脚本会下载并整理：

| 数据 | 来源 | 本地目录 |
| :-- | :-- | :-- |
| OMat24 rattled-500 train | 官方 OMat24 train split | `./data/omat24/train/rattled-500` |
| OMat24 rattled-500 validation | 官方 2024-12-20 修正版 validation split | `./data/omat24/val/rattled-500` |
| OC20 S2EF train 200K | 官方 OC20 S2EF train 200K | `./data/oc20/raw/s2ef_train_200K` |
| OC20 S2EF val-id | 官方 OC20 S2EF val-id | `./data/oc20/raw/s2ef_val_id` |
| OC20 budget ASELMDB | 从 train/val-id 流式转换 | `./data/oc20/uma_budget_aselmdb` |

如果默认 `python` 环境缺少 `ase` 或 `lmdb`，可以显式指定：

```bash
PYTHON_BIN=/path/to/python bash interatomic_potentials/configs/uma/prepare_budget_data.sh ./data
```

预算训练：

```bash
python interatomic_potentials/train.py -c interatomic_potentials/configs/uma/uma_omat24_r500_budget_s2ef.yaml
python interatomic_potentials/train.py -c interatomic_potentials/configs/uma/uma_oc20_50k_budget_s2ef.yaml
```

这两个实验定位为 medium-scale migration validation：用于验证 PaddleMaterials
UMA 的数据读取、task/domain 传播、训练收敛、评估和 checkpoint 保存，不作为
fairchem 原论文规模 benchmark 的完整复现。

### 半天预算验证结果

以下实验在单卡 RTX 4090 上使用 PaddlePaddle `3.3.0` 环境完成，开启 AMP O1，
随机种子为 `42`。两个单域实验均使用 50k train / 5k
validation / 5k test 的固定预算划分，指标为日志中记录的 `energy(metric)`、
`forces(metric)` 和加权 `loss(loss)`。

| 实验 | 配置 | Epoch | 可训练参数 | 运行时间 | Best checkpoint |
| :-- | :-- | --: | --: | :-- | :-- |
| OMat24 rattled-500 | `uma_omat24_r500_budget_s2ef.yaml` | 3 | 6.33M | 约 1h09m | `output/uma_omat24_r500_budget_s2ef/checkpoints/best.pdparams` |
| OC20 S2EF | `uma_oc20_50k_budget_s2ef.yaml` | 2 | 6.34M | 约 55m | `output/uma_oc20_50k_budget_s2ef/checkpoints/best.pdparams` |
| Multi-task short-run | `uma_multitask_budget_smoke.yaml` | 1 | 6.34M | < 1 min | `output/uma_multitask_budget_smoke/checkpoints/best.pdparams` |

单域训练的 epoch 汇总 loss 持续下降，说明迁移后的 forward、loss、backward 和
optimizer 链路均能稳定学习：

| 实验 | Train energy(loss) | Train forces(loss) | Train loss(loss) |
| :-- | :-- | :-- | :-- |
| OMat24 epoch 1 -> 2 -> 3 | 6.627979 -> 4.304747 -> 3.176400 | 0.483082 -> 0.310235 -> 0.257879 | 80.772244 -> 52.354506 -> 39.500356 |
| OC20 epoch 1 -> 2 | 51.754490 -> 19.917693 | 0.181425 -> 0.178013 | 522.987643 -> 204.517324 |

最终评估结果如下：

| 实验 | Val energy(metric) | Val forces(metric) | Val loss(loss) | Test energy(metric) | Test forces(metric) | Test loss(loss) |
| :-- | --: | --: | --: | --: | --: | --: |
| OMat24 rattled-500 | 11.780332 | 0.412024 | 127.072415 | 11.963885 | 0.331205 | 128.924666 |
| OC20 S2EF | 16.703999 | 0.182028 | 172.312066 | 16.996420 | 0.161219 | 175.239833 |
| Multi-task short-run | 237.356260 | 0.493668 | 2385.158658 | 183.573112 | 1.271681 | 1850.638359 |

其中 multi-task short-run 使用真实 OC20 小样本和 OMat24 子集作为多 task proxy，
目的是验证 `task_name` 经过 dataset、collate、task embedding、forward、loss、
backward、eval/test 和 checkpoint 保存链路，不作为真实多域科学 benchmark。

## 多任务验证

fairchem 原版 UMA 支持多个 task/domain，例如 `oc20`、`omat`、`omol`、`odac`
和 `omc`。在本次 PaddleMaterials 迁移中，UMA 数据适配器提供了 `task_name`
字段；collate 后该字段会作为 `dataset` 传入模型。当
`use_dataset_embedding=True` 时，它会驱动 UMA 的 dataset/task embedding 路径。

当前提供三个小规模验证配置：

| 配置 | 目的 |
| :-- | :-- |
| `uma_multitask_omat_subsets_forward.yaml` | 在同一个 dataloader 中检查多个 `task_name` 的 forward/eval 路径 |
| `uma_multitask_omat_subsets_train.yaml` | 检查混合 task 的短程训练、loss、backward 和 trainer 兼容性 |
| `uma_multitask_budget_smoke.yaml` | 使用预算数据做 5-task 短训，验证 eval/test 和 checkpoint 保存 |

这些配置使用真实 OC20 小样本作为 `oc20` task，并使用不同 OMat24 subset
作为 `omat`、`omol`、`odac` 和 `omc` 的 proxy。它们是轻量机制验证：
用于确认 task 名可以正确经过数据读取、collate、task embedding、forward、loss
和 backward。由于 `omol`、`odac` 和 `omc` 的真实官方数据需要授权，这里不代表
这些真实数据域上的科学指标复现。

多任务 forward/eval 验证：

```bash
python interatomic_potentials/train.py -c interatomic_potentials/configs/uma/uma_multitask_omat_subsets_forward.yaml
```

混合 task 短程训练验证：

```bash
python interatomic_potentials/train.py -c interatomic_potentials/configs/uma/uma_multitask_omat_subsets_train.yaml
```

预算数据 multi-task short-run 验证：

```bash
python interatomic_potentials/train.py -c interatomic_potentials/configs/uma/uma_multitask_budget_smoke.yaml
```

如果要做更强的多域验证，可以将 OMat subset 路径替换为真实 domain 数据：

| `task_name` | 应用域 | 示例路径 |
| :-- | :-- | :-- |
| `oc20` | 催化体系 | `./data/uma_multidomain/oc20/train.aselmdb` |
| `omat` | 无机材料 | `./data/uma_multidomain/omat/train.aselmdb` |
| `omol` | 分子体系 | `./data/uma_multidomain/omol/train.aselmdb` |
| `odac` | MOF 体系 | `./data/uma_multidomain/odac/train.aselmdb` |
| `omc` | 分子晶体 | `./data/uma_multidomain/omc/train.aselmdb` |

验证结论可按如下层级理解：

| 验证层级 | 能证明什么 | 不应过度声称什么 |
| :-- | :-- | :-- |
| 单域 OMat24 | PaddleMaterials UMA 基础 train/eval/test 链路可用 | 完整多域 UMA 行为 |
| 多 task forward | `task_name` 传播和 task embedding 选择逻辑可用 | 完整 MoE 正确性 |
| 混合 task loss alignment | 多域训练路径、loss 和 backward 稳定 | expert routing 分布完全正确 |

如需显式验证 MoE，还应进一步记录或对比 expert routing/mixing weights、active
expert 数量，或者 expert 参数调用情况。多数据集 loss 对齐是有价值的行为证据，
但更严谨的表述应是“间接验证迁移行为一致”，而不是“完整证明 MoE 机制正确”。

## 验证

```bash
python interatomic_potentials/train.py -c interatomic_potentials/configs/uma/uma_omat24_r500_s2ef.yaml Global.do_train=False Global.do_eval=True Global.do_test=False Trainer.pretrained_model_path="path/to/checkpoint.pdparams"
```

## 测试

```bash
python interatomic_potentials/train.py -c interatomic_potentials/configs/uma/uma_omat24_r500_s2ef.yaml Global.do_train=False Global.do_eval=False Global.do_test=True Trainer.pretrained_model_path="path/to/checkpoint.pdparams"
```

## 关键配置

当前配置参考 fairchem UMA small K4L2 direct-training 设置，并映射到本次
PaddleMaterials 迁移中已经打通的训练链路：

| 配置项 | 取值 |
| :-- | :-- |
| Backbone 层数 | 4 |
| `lmax` / `mmax` | 2 / 2 |
| hidden / sphere / edge channels | 128 |
| distance basis | 64 |
| cutoff | 6.0 |
| max neighbors | 30 |
| optimizer | AdamW |
| learning rate | `8e-4`，cosine decay 到 `8e-6` |
| weight decay | `1e-3` |
| loss weights | energy 10.0，forces 30.0 |

当前实现使用单任务适配器 `ppmat.models.uma.escn_md.UMASingleTaskModel`，
并设置 `freeze_backbone=False` 进行全量训练。

## 参考结果

以下结果来自 `uma_omat24_r500_s2ef.yaml` 默认小样本设置，在本地 OMat24
`rattled-500` split 上运行得到：

| 划分 | Energy MAE | Force MAE | Weighted loss |
| :-- | --: | --: | --: |
| Train epoch 1 | 29.499472 | 1.765254 | 347.952337 |
| Train epoch 6 | 13.526471 | 1.013610 | 165.672998 |
| Val epoch 6 | 15.380561 | 1.007013 | 180.619805 |
| Test epoch 6 | 15.120827 | 0.634044 | 184.816584 |

该结果用于说明迁移链路和小规模训练流程可用，不是完整论文规模复现实验。
当前 PaddleMaterials 版本已对齐 UMA K4L2 主干结构和直接能量/力训练路径，但尚未
完整覆盖 fairchem 的多数据集 MoE/task embedding 训练栈。

## 引用

```bibtex
@misc{fairchem_uma,
  title = {UMA: A Family of Universal Models for Atoms},
  author = {FAIR Chemistry Team},
  howpublished = {\url{https://github.com/facebookresearch/fairchem}},
  year = {2025}
}
```
