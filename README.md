# PaddleScience-Material

<p align="center">
 <img src="docs/logo.png" align="middle" width = "600"/>
<p align="center">

## Introduction

晶体材料具有对称和周期性的结构，具有多种性能，广泛应用于从电子设备到能源应用的各个领域。为了发现晶体材料，传统的实验和计算方法通常既耗时又昂贵。近年来，由于晶体材料数据的爆炸性增长，数据驱动的材料发现引起了人们的极大兴趣。特别是，最近的进展利用了深度学习的表达能力来模拟晶体材料中高度复杂的原子系统，为快速准确的材料发现开辟了新的途径。PaddleScience-Material是一个基于PaddlePaddle的材料科学工具包，旨在帮助研究人员更高效地探索、发现和开发新的晶体材料。




已整理好的数据、模型可从[此处](https://pan.baidu.com/s/1payB2J7uJE8nOSa_wVSHLw?pwd=13k6)下载。

## 1. Property prediction

[基于GNN的二维材料稳定性预测](stability_prediction/README.md)

<div align="center">
    <img src="docs/flow.svg" width="900">
</div>

### Results

#### Datasets:

- MP2018.6.1

    The original dataset can download from [here](https://figshare.com/ndownloader/files/15087992).
    For the convenience of training, we divided it into a ratio of 0.9:0.05:0.05，you can download it from [here](https://pan.baidu.com/s/1Y6ye2hu3y0v9ofDs06tejg?pwd=n3my)

    |    Dataset   | train | val  | test |
    | :----------: | :---: | :--: | :--: |
    | MP2018.6.1   | 62315 | 3461 | 3463 |

- MP20
    The MP20 dataset can download from [here](https://github.com/jiaor17/DiffCSP/tree/main/data/mp_20).

    |    Dataset   | train | val  | test |
    | :----------: | :---: | :--: | :--: |
    | MP20   | 27136 | 9047 | 9046 |

#### Task 1: formation energy per atom


|    Model     | Dataset | MAE(test dataset) | config    | Checkpoint |
| :----------: | :---------------: | :---------------: | :-------: |  :-------: |
| MegNet       | MP18           | 0.034           | [megnet_mp18](property_prediction/configs/megnet_mp18.yaml) | [checkpoint](https://pan.baidu.com/s/128VPZFjBmhObyJSkoCxUxA?pwd=kv82) |
| DimeNet++       | MP18           | 0.030           | [dimenet_mp18](property_prediction/configs/dimenet_mp18.yaml) | [checkpoint](https://pan.baidu.com/s/1QdafA1DSQ9yj9UzgXTNmiA?pwd=ke3x) |
| MegNet       | MP20           | 0.028           | [megnet_mp20](property_prediction/configs/megnet_mp20.yaml) | [checkpoint](https://pan.baidu.com/s/15BRj5_-N1yw767vldm8qFg?pwd=bmat) |
| DimeNet++       | MP20           | 0.023           | [dimenet_mp20](property_prediction/configs/dimenet_mp20.yaml) | [checkpoint](https://pan.baidu.com/s/17SkyrvOOsoSgdsWAr3fwIA?pwd=bnnn) |
| GemNet       | MP20           | 0.023           | [gemnet_mp20](property_prediction/configs/gemnet_mp20.yaml) | [checkpoint](https://pan.baidu.com/s/1MWkFZ9xkyTfinf8pwt5SxA?pwd=8k68) |



## 2. Structure prediction


[基于扩散模型的二维材料结构生成](structure_prediction/README.md)

<div align="center">
    <img src="docs/diff_arch.png" width="900">
</div>


### Results

#### Task 1: Stable Structure Prediction

|    Model     | # of samples | Dataset  | Match rate | RMSE   | Config                         | Checkpoint |
| :----------: | :----------: | :-------: | :--------: | :----: | :----------------------------: | :--------: |
| diffcsp-paper| 1            | mp_20 | 51.49          | 0.0631 | - | - |
| diffcsp      | 1            | mp_20 | 54.53          | 0.0547 | [diffcsp_mp20](structure_prediction/configs/diffcsp_mp20.yaml) | [checkpoint](https://pan.baidu.com/s/1aBhac-ctdBe1WWv09hVq7g?pwd=awi4) |


#### Task 2: Ab Initio Crystal Generation

|             Model          |Dataset|samples|Validity<br>Struc. / Comp.| COV<br>R / P| $d_\rho$ | $d_E$  | $d_{ele}$ | Config                         | Checkpoint |
| :------------------------: | :---: | :----:| :----------------------: | :---------: | :------: | :----: | :-------: | :----------------------------: | :--------: |
| diffcsp<br>(one-hot)       | mp_20 | 1000  | 99.95 / 84.51            |99.61 / 99.32| 0.2069   | 0.0659 | 0.4193    | [diffcsp_mp20_with_type](structure_prediction/configs/diffcsp_mp20_with_type.yaml) | [checkpoint](https://pan.baidu.com/s/1JiniNkRb2Rb_sGNhrKpU_w?pwd=1ath) |
| diffcsp<br>(one-hot)       | mp_20 | 5000  | 99.95 / 84.51            |99.61 / 99.32| 0.1170   | 0.0723 | 0.3988    | [diffcsp_mp20_with_type](structure_prediction/configs/diffcsp_mp20_with_type.yaml) | [checkpoint](https://pan.baidu.com/s/1JiniNkRb2Rb_sGNhrKpU_w?pwd=1ath) |
| diffcsp<br>(d3pm-uniform)  | mp_20 |  1000 | 99.95 / 82.19            |99.51 / 99.73| 0.1582   | 0.1292 | 0.1360    | [diffcsp_mp20_d3pm](structure_prediction/configs/diffcsp_mp20_d3pm.yaml) | [checkpoint](https://pan.baidu.com/s/1qU8TDMAhrrkp12zMuXg6VA?pwd=2jcc) |
| diffcsp<br>(d3pm-uniform)  | mp_20 |  5000 | 99.95 / 82.19            |99.51 / 99.73| 0.1651   | 0.0939 | 0.1310    | [diffcsp_mp20_d3pm](structure_prediction/configs/diffcsp_mp20_d3pm.yaml) | [checkpoint](https://pan.baidu.com/s/1qU8TDMAhrrkp12zMuXg6VA?pwd=2jcc) |
| diffcsp<br>(d3pm-absorbing)  | mp_20 |  1000 | 99.89 / 82.66          |99.91 / 99.47| 0.2093   | 0.1186 | 0.1460    | [diffcsp_mp20_d3pm_absorbing](structure_prediction/configs/diffcsp_mp20_d3pm_absorbing.yaml) | [checkpoint](https://pan.baidu.com/s/1HbIrhvjLNk83aIon8Pz_5Q?pwd=815m) |
| diffcsp<br>(d3pm-absorbing)  | mp_20 |  5000 | 99.89 / 82.66          |99.91 / 99.47| 0.1809   | 0.0988 | 0.1262    | [diffcsp_mp20_d3pm_absorbing](structure_prediction/configs/diffcsp_mp20_d3pm_absorbing.yaml) | [checkpoint](https://pan.baidu.com/s/1HbIrhvjLNk83aIon8Pz_5Q?pwd=815m) |

The term "samples" refers to the valid samples of generated structures utilized for calculating $d_\rho$, $d_E$  and  $d_{ele}$.

# Install

Please refer to the installation [document](install.md) for environment configuration.

# Acknowledgements

This repo referenced the code of the following repos: [PaddleScience](https://github.com/PaddlePaddle/PaddleScience), [Matgl](https://github.com/materialsvirtuallab/matgl), [CDVAE](https://github.com/txie-93/cdvae), [DiffCSP](https://github.com/jiaor17/DiffCSP)
