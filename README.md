# PaddleScience-Material

<p align="center">
 <img src="docs/logo.png" align="middle" width = "600"/>
<p align="center">

## 简介

晶体材料具有对称和周期性的结构，具有多种性能，广泛应用于从电子设备到能源应用的各个领域。为了发现晶体材料，传统的实验和计算方法通常既耗时又昂贵。近年来，由于晶体材料数据的爆炸性增长，数据驱动的材料发现引起了人们的极大兴趣。特别是，最近的进展利用了深度学习的表达能力来模拟晶体材料中高度复杂的原子系统，为快速准确的材料发现开辟了新的途径。PaddleScience-Material是一个基于PaddlePaddle的材料科学工具包，旨在帮助研究人员更高效地探索、发现和开发新的晶体材料。


## 任务类型

已整理好的数据、模型可从[此处](https://pan.baidu.com/s/1payB2J7uJE8nOSa_wVSHLw?pwd=13k6)下载。

### 1. 性质预测

预测晶体材料的性质是一个复杂而广泛的领域，涉及理论和实验方法。这项任务的复杂性归因于所涉及的众多变量和相互作用。最近，深度学习技术极大地推进了晶体材料的研究，越来越多的研究人员开发了复杂的模型来预测材料的性质。这些方法利用深度神经网络从数据集中捕获晶体结构和属性信息之间的映射关系，使其能够对各种材料属性进行预测。

[基于GNN的二维材料稳定性预测](stability_prediction/README.md)

<div align="center">
    <img src="docs/flow.svg" width="900">
</div>


### 2. 结构生成

传统上，新型晶体材料的发现在很大程度上依赖于直觉、试错实验和偶然性。然而，可能的晶体结构的化学空间是非常巨大的，因此仅通过物理合成和表征进行详尽的探索是不可行的。近年来，GANs、VAEs、扩散、流匹配、Transformer等深度生成模型为加速晶体材料的发现提供了一条有前景的新途径。通过从大型数据集中学习，它们展示了学习控制晶体材料结构-性质关系的潜在模式和规则的潜力。


[基于扩散模型的二维材料结构生成](structure_prediction/README.md)

<div align="center">
    <img src="docs/diff_arch.png" width="900">
</div>

## 环境安装

请参考[安装文档](install.md)进行环境配置。


## Results


### Task: Ab Initio Crystal Generation

|    Model     |  Dataset  | Struc. Validity | Comp. Validity | COV-R | COV-P | $d_\rho$ | $d_E$  | $d_{ele}$ | config                         | Checkpoint |
| :----------: | :-------: | :-------------: | :------------: | :---: | :---: | :------: | :----: | :-------: | :----------------------------: | :--------: |
| diffcsp(one-hot) | mp_20 | 99.95           | 84.51          | 99.61 | 99.32 | 0.2069   | 0.0659 | 0.4193    | [diffcsp_mp20_with_type](structure_prediction/configs/diffcsp_mp20_with_type.yaml) | [checkpoint](https://pan.baidu.com/s/1JiniNkRb2Rb_sGNhrKpU_w?pwd=1ath) |
