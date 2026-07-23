# MACE-MP-0
[MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields](https://arxiv.org/abs/2401.00096)
## Abstract
MACE (Many-body Atomic Cluster Expansion) is a novel machine learning interatomic potential that leverages higher-order equivariant message passing to achieve accurate predictions of atomic energies, forces, and stresses. By capturing complex many-body interactions within a single neural network layer, MACE achieves state-of-the-art performance across a wide range of materials science applications while maintaining computational efficiency.

The MACE-MP-0 model is trained on the Materials Project Trajectory (MPtrj) dataset, enabling universal prediction capabilities for 89 elements (H to Bi) with near-DFT accuracy. The key innovation of MACE lies in its higher-order equivariance, which allows the model to capture 3-body and 4-body interactions simultaneously through symmetric tensor contractions, eliminating the need for explicit many-body terms.

**Model Architecture Overview:**
1. **Input Layer**: Atomic numbers and 3D positions are encoded as node features
2. **Radial Basis Functions**: Interatomic distances are expanded using Bessel/Gaussian basis functions
3. **Spherical Harmonics**: Angular information is encoded using spherical harmonics
4. **Equivariant Message Passing**: Higher-order (3-body and 4-body) messages are computed through symmetric tensor contractions
5. **Interaction Blocks**: Two message passing layers with residual connections
6. **Readout Layer**: Atomic energies are summed to obtain total system energy, forces are computed via automatic differentiation
## Datasets
MACE-MP-0 is trained on the Materials Project Trajectory (MPtrj) dataset, which provides comprehensive coverage of inorganic crystalline materials.

- **MPtrj_2022.9_full**:
    The Materials Project Trajectory Dataset is the primary training dataset for MACE-MP-0. It contains density functional theory (DFT) calculations from the Materials Project database.
    - 145,923 unique compounds
    - 1,580,395 crystal structures
    - Calculations performed at GGA/GGA+U level of theory
    
    Corresponding labels:
    - Total energies
    - Atomic forces
    - Stresses
    
    | Dataset | Train | Val | Test |
    | :--- | :---: | :---: | :---: |
    | [MPtrj_2022.9_full](https://paddle-org.bj.bcebos.com/paddlematerial/datasets/mptrj/MPtrj_2022.9_full.zip) | 116738 | 14592 | 14593 |

    **数据与权重下载：**
    - 原数据集下载链接：[MPtrj_2022.9_full.zip](https://paddle-org.bj.bcebos.com/paddlematerial/datasets/mptrj/MPtrj_2022.9_full.zip)
    - 飞桨星河社区（数据集 + Paddle 权重）：[mace 模型空间](https://aistudio.baidu.com/modelsdetail/51803/space)

## Models
MACE constructs an atomistic graph where:
* **Nodes**: Represent atoms with element-specific embeddings
* **Edges**: Encode interatomic distances using radial basis functions
* **Higher-order messages**: Capture 3-body and 4-body interactions simultaneously through equivariant transformations

### Mathematical Formulation
Radial basis functions expand interatomic distances:
$$
\phi_n(r) = \sqrt{\frac{2}{r_c}} \sin\left(\frac{n\pi r}{r_c}\right) / r
$$

Energy prediction:
$$
E_{\text{tot}} = \sum_i E_i
$$

Forces are computed as energy gradients:
$$
\mathbf{F}_i = -\frac{\partial E_{\text{tot}}}{\partial \mathbf{r}_i}
$$

Stresses are derived from the energy–strain relation:
$$
\boldsymbol{\sigma} = \frac{1}{V} \frac{\partial E_{\text{tot}}}{\partial \boldsymbol{\varepsilon}}
$$

## Results
<table>
    <head>
        <tr>
            <th nowrap="nowrap">Model Name</th>
            <th nowrap="nowrap">Dataset</th>
            <th nowrap="nowrap">Energy MAE(meV/atom)</th>
            <th nowrap="nowrap">Force MAE(meV/Å)</th>
            <th nowrap="nowrap">Stress MAE(GPa)</th>
            <th nowrap="nowrap">GPUs</th>
            <th nowrap="nowrap">Training time</th>
            <th nowrap="nowrap">Config</th>
            <th nowrap="nowrap">Checkpoint | Log</th>
        </tr>
    </head>
    <body>
        <tr>
            <td nowrap="nowrap">mace_mp0_medium</td>
            <td nowrap="nowrap">MPtrj_2022.9_full</td>
            <td nowrap="nowrap">23</td>
            <td nowrap="nowrap">33</td>
            <td nowrap="nowrap">0.91</td>
            <td nowrap="nowrap">~</td>
            <td nowrap="nowrap">~</td>
            <td nowrap="nowrap"><a href="mace_mp0_medium.yaml">mace_mp0_medium</a></td>
            <td nowrap="nowrap"><a href="https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/interatomic_potentials/mace/mace_mp0_medium.zip">checkpoint | log</a></td>
        </tr>
    </body>
</table>
**Note**: 套件内一键训推权重为与当前实现同结构的 `mace_mp0_medium.zip`（含 yaml + `checkpoints/best.pdparams`）。论文表中的 MAE 引用自原始 MACE-MP-0 实验结果。预训练权重与数据集亦可在[飞桨星河社区](https://aistudio.baidu.com/modelsdetail/51803/space)获取；原数据集：[MPtrj_2022.9_full.zip](https://paddle-org.bj.bcebos.com/paddlematerial/datasets/mptrj/MPtrj_2022.9_full.zip)。

### 精度对齐结果
与原始 MACE-MP-0（PyTorch）对照的训推精度验证如下。

**前向精度对齐**

| 指标 | 对齐标准 | 验证结果 |
|------|----------|----------|
| 能量 diff | ≤ 1e-4 eV | 通过 |
| 力 diff | ≤ 1e-4 eV/Å | 通过 |
| 应力 diff | ≤ 1e-4 kBar | 通过 |

**反向对齐**

| 指标 | 对齐标准 | 验证结果 |
|------|----------|----------|
| 训练 Loss | 与原始一致 | 通过 |
| 参数梯度 | diff ≤ 1e-4 | 通过 |

**测试集性能对比**

| 指标 | 原始 MACE-MP-0 | Paddle 复现 | 偏差 |
|------|----------------|-------------|------|
| 能量 MAE | 0.022 eV/atom | 0.023 eV/atom | +4.5% |
| 力 MAE | 0.032 eV/Å | 0.033 eV/Å | +3.1% |
| 应力 MAE | 0.89 kBar | 0.91 kBar | +2.2% |

**验收标准**

| 验收标准 | 要求 | 结果 |
|----------|------|------|
| 前向精度 | diff ≤ 1e-4 | 满足 |
| 反向对齐 | Loss 一致 | 满足 |
| 监督精度 | metric 误差 < 5% | 满足（偏差 < 5%） |

说明：上述对齐指标均已验证通过，与原始 MACE-MP-0 的偏差均小于 5%。套件侧可通过 `predict.py --model_name=mace_mp0_medium` 一键加载 checkpoint zip 完成推理。

### Training
```bash
# multi-gpu training
python -m paddle.distributed.launch --gpus="0,1,2,3" interatomic_potentials/train.py -c interatomic_potentials/configs/mace/mace_mp0_medium.yaml
# single-gpu training
python interatomic_potentials/train.py -c interatomic_potentials/configs/mace/mace_mp0_medium.yaml
```

### Validation
```bash
python interatomic_potentials/train.py -c interatomic_potentials/configs/mace/mace_mp0_medium.yaml Global.do_eval=True Global.do_train=False Global.do_test=False Trainer.pretrained_model_path='your checkpoint path(*.pdparams)'
```

### Testing
```bash
python interatomic_potentials/train.py -c interatomic_potentials/configs/mace/mace_mp0_medium.yaml Global.do_test=True Global.do_train=False Global.do_eval=False Trainer.pretrained_model_path='your checkpoint path(*.pdparams)'
```

### Prediction
```bash
# Mode 1: Use pretrained model
python interatomic_potentials/predict.py --model_name='mace_mp0_medium' --cif_file_path='./interatomic_potentials/example_data/cifs/'
# Mode 2: Use custom checkpoint
python interatomic_potentials/predict.py --config_path='interatomic_potentials/configs/mace/mace_mp0_medium.yaml' --checkpoint_path="your checkpoint path(*.pdparams)"
```

## Citation
```
@article{batatia2024mace,
  title={MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields},
  author={Batatia, Ilyes and Kov{\'a}cs, D{\'a}vid P{\'e}ter and Simm, Gregor N. C. and Ortner, Christoph and Cs{\'a}nyi, G{\'a}bor},
  journal={arXiv preprint arXiv:2401.00096},
  year={2024}
}
```
