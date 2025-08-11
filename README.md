# PaddleMaterials

<p align="center">
 <img src="docs/ppmat_logo.png" align="middle" width = "400"/>
<p align="center">

## 🚀 Introduction

**PaddleMaterials** is a data-mechanism dual-driven, development and deployment of foundation model, end to end toolkit based on PaddlePaddle deep learning framework for materials science and engineering. It is designed to help researchers more efficiently build AI4Materials foundation models and explore, discover, and develop new materials based on deployed pretrained models. It has supported inorganic materials and part of organic molecules, and will support more types of materials including polymers, organic molecules, catalysts, and so on. It has supported some representative models including the equivalent graph networks-based model, diffusion model, multi-modal model, and will support more kinds of deep learing models and agents works related to AI4Material fields in the feature.

<p align="left">
 <img src="docs/overview_en.png" align="middle" width = "1000"/>
<p align="left">

**Inorganic materials**, characterized by their symmetrical and periodic structures, exhibit a wide range of properties and are widely applied in various fields, from electronic devices to energy applications. Traditional experimental and computational methods for discovering crystalline materials are often time-consuming and expensive. Data-driven approaches to material discovery have the power to model the highly complex atomic systems within crystalline materials, paving the way for rapid and accurate material discovery.

**Organic materials**, distinguished by covalently linked, directionally bonded networks, mainly defined as a carbon–hydrogen or carbon–carbon bond chemical compound. These traits support core applications including flexible displays, organic photovoltaics, high-energy-density battery electrodes, advanced separation membranes, catalyts. The vast compositional and conformational space of organic molecules makes trial-and-error synthesis and ab-initio simulations slow and costly. Data-driven methods that fuse high-throughput datasets, graph-based representations, and deep generative models rapidly learn structure–property links, enabling fast virtual screening and rational design for more agile, sustainable advances in organic materials.

## 📣 News

🔥 **2025.07.01**: The Suzhou Laboratory has established a novel framework based on PaddleMaterials, combining an active learning workflow with conditional-diffusion-based structure generation, thereby achieving unprecedented expansion of two-dimensional material databases. For more information, please refer to [ML2DDB](./research/ML2DDB/README.md).

## 📑 Task
- [MLIP-Machine Learning Interatomic Potential](interatomic_potentials/README.md)
- [IOMPP-Inorganic Material Property Prediction](property_prediction/README.md)
- [IOMSG-Inorganic Material Structure Generation](structure_generation/README.md)

## 🔧 Installation

Please refer to the installation [document](Install.md) for environment configuration.


## ⚡ Get Started

s offers multiple built-in models that can be directly used for inference. Taking the `megnet_mp2018_train_60k_e_form` model as an example (a MEGNet model trained on the MP2018 dataset for material formation energy prediction), use the following command for inference:
```bash
python property_prediction/predict.py --model_name='megnet_mp2018_train_60k_e_form' --weights_name='best.pdparams' --cif_file_path='./property_prediction/example_data/cifs/' --save_path='result.csv'
```

<table>
    <thead>
        <tr>
            <th>Parameter</th>
            <th>Description</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>--model_name</td>
            <td>Name of the built-in model</td>
        </tr>
        <tr>
            <td>--weights_name</td>
            <td>Weights file name</td>
        </tr>
        <tr>
            <td>--cif_file_path</td>
            <td>Path to CIF files for prediction</td>
        </tr>
        <tr>
            <td>--save_path</td>
            <td>Path to save prediction results</td>
        </tr>
    </tbody>
</table>

For more information on how to use PaddleMaterials to train and fine tune a model, please refer to the [documentation](get_started.md).


## 👩‍👩‍👧‍👦 Cooperation

<p align="left">
 <img src="docs/suzhoulab.png" align="middle" width = "200"/>
 <img src="docs/zhonghua.jpeg" align="middle" width = "240"/>
<p align="left">

## 👩‍👩‍👧‍👦 Community

Join PaddleMaterials WeChat group to disscuss with us!

<p align="left">
 <img src="docs/wechat_group.png" align="middle" width = "200"/>
<p align="left">

## 🔄 Feedback

We sincerely invite you to spare a moment from your busy schedule to share your [feedback](https://paddle.wjx.cn/vm/rXyQwB2.aspx#).

## 📜 License

PaddleMaterials is licensed under the [Apache License 2.0](LICENSE).


## 🎓 Citation


    @misc{paddlematerials2025,
    title={PaddleMaterials, a deep learning toolkit based on PaddlePaddle for material science.},
    author={PaddleMaterials Contributors},
    howpublished = {\url{https://github.com/PaddlePaddle/PaddleMaterials}},
    year={2025}
    }


## Acknowledgements

This repository references the code from the following repositories:
[PaddleScience](https://github.com/PaddlePaddle/PaddleScience),
[Matgl](https://github.com/materialsvirtuallab/matgl),
[CDVAE](https://github.com/txie-93/cdvae),
[DiffCSP](https://github.com/jiaor17/DiffCSP),
[MatterGen](https://github.com/microsoft/mattergen),
[MatterSim](https://github.com/microsoft/mattersim),
[CHGNet](https://github.com/CederGroupHub/chgnet),
[AIRS](https://github.com/divelab/AIRS),
etc.
