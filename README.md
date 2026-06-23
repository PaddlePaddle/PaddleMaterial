# PaddleMaterials

<p align="center">
 <img src="docs/ppmat_logo.png" align="middle" width = "400"/>
<p align="center">

## 🚀 Introduction

**PaddleMaterials** is an end-to-end AI4Materials toolkit built on the **PaddlePaddle** deep learning framework. Designed as a data-mechanism dual-driven platform for developing and deploying foundation models in materials science, **PPMat** enables researchers to efficiently build AI models and accelerate material discovery using pretrained models.

<p align="left">
 <img src="docs/overview_en.png" align="middle" width = "1000"/>
<p align="left">

### Core Capabilities

| Task | Description | Typical Applications |
|------|-------------|---------------------|
| **Property Prediction (PP)** | Predict material properties from structure | Formation energy, band gap, elastic moduli |
| **Structure Generation (SG)** | Generate novel crystal structures | High-throughput screening, inverse design |
| **Interatomic Potential (IP)** | Replace DFT with ML potentials | Molecular dynamics, large-scale simulations |
| **Electronic Structure (ES)** | Predict electronic properties | Band structure, density of states |
| **Spectrum Elucidation (SE)** | Reconstruct structures from spectra | NMR structure elucidation |
| **Spectrum Enhancement (SPEN)** | Enhance microscopy and spectrum signals | STEM image enhancement, denoising |

### Supported Materials

- **Inorganic Crystals** - Well-supported with multiple datasets and pretrained models
- **Organic Molecules** - Support for small molecule datasets and property prediction
- *Polymers, catalysts, and amorphous materials are under development*

### Why PaddleMaterials?

- ✅ **Rich Pretrained Models** - 50+ pretrained models ready for inference
- ✅ **Multi-Task Integration** - Unified framework across PP, SG, MLIP, MLES, SE, SPEN
- ✅ **Domestic Hardware Support** - Full support for MetaX GPUs and NVIDIA GPUs
- ✅ **Production-Ready** - Distributed training, mixed precision, checkpoint recovery

### 📑 Support Tasks

| Task | Description | Link |
|------|-------------|------|
| **Property Prediction (PP)** | Predict formation energy, band gap, elastic properties | [README](property_prediction/README.md) |
| **Structure Generation (SG)** | Generate new crystal structures with diffusion models | [README](structure_generation/README.md) |
| **Interatomic Potential (IP)** | DFT-accurate potentials for molecular dynamics | [README](interatomic_potentials/README.md) |
| **Electronic Structure (ES)** | Predict electronic structure properties | [README](electronic_structure/README.md) |
| **Spectrum Elucidation (SE)** | Reconstruct molecular structures from NMR spectra | [README](spectrum_elucidation/README.md) |
| **Spectrum Enhancement (SPEN)** | Enhance microscopy and spectrum signals | [README](spectrum_enhancement/README.md) |

### 🎯 Available Pretrained Models

| Task | Models | Dataset |
|------|--------|---------|
| **Property Prediction** | MEGNet, iComformer, DimeNet++ | MP2018, MP2024, JARVIS |
| **Structure Generation** | MatterGen, DiffCSP | MP20, ALEX |
| **Interatomic Potentials** | CHGNet, MatterSim | MPTRJ |
| **Electronic Structure** | InfGCN | Custom datasets |
| **Spectrum Elucidation** | DiffNMR | MSD_NMR |
| **Spectrum Enhancement** | SFIN | HAADF/BF STEM image datasets |

Full model list: See [MODEL_REGISTRY](ppmat/models/__init__.py#L75)

---

## ⚡ Get Started

### 🔧 Installation

Please refer to the installation [document](Install.md) for your hardware environment. See [SupportedHardwareList](./docs/multi_device.md) for more multi-hardware adaptation information.

---

### Mimi Inference

#### Property Prediction

Predict material formation energy using a pretrained MEGNet model:

```bash
python property_prediction/predict.py \
    --model_name='megnet_mp2018_train_60k_e_form' \
    --weights_name='best.pdparams' \
    --cif_file_path='./property_prediction/example_data/cifs/' \
    --save_path='result.csv'
```

#### Structure Generation

Generate novel crystal structures:

```bash
python structure_generation/predict.py \
    --model_name='mattergen_mp20' \
    --num_structures=100 \
    --save_path='generated_structures/'
```

#### Interatomic Potentials

Run molecular dynamics with ML potentials:

```bash
python interatomic_potentials/run_md.py 
    --model_name='mattersim_1M' 
    --structure_path='input.cif' 
    --temperature=300
```

#### Electronic Structure

Run prediction of elcutorninc density:

```bash
python interatomic_potentials/run_md.py 
    --model_name='mattersim_1M' 
    --structure_path='input.cif' 
    --temperature=300
```

#### Spectrum Elucidation

Run NMR spectrum elucidate:

```bash
python spectrum_elucidation/sample.py 
    --config_path='spectrum_elucidation/configs/diffnmr/DiffNMR.yaml' 
    --weights_name='DiffNMR_nless15_best.pdparams' 
    --save_path='result_diffnmr_nless15/' 
    --checkpoint_path="pretrained"
```

#### Spectrum Enhancement

Run prediction of elcutorninc density:

```bash
python spectrum_enhancement/predict.py 
    --model_name sfin_haadf_enhance 
    --split val
```

---

### Train Your Own Model

For training and fine-tuning, refer to the [documentation](get_started.md).

---

## 👩‍👩‍👧‍👦 Contributors & Cooperation & Community

Thanks to all contributors who have helped build PaddleMaterials！
<a href="https://github.com/PaddlePaddle/PaddleMaterials/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=PaddlePaddle/PaddleMaterials" />
</a>

Thanks for the following organiziton for cooprative support!
<p align="left">
 <img src="docs/logo_SZNL_2.jpeg" align="middle" width = "240"/>
 <img src="docs/logo_SinochemDI_2.jpeg" align="middle" width = "240"/>
 <img src="docs/logo_MetaX.png" align="middle" width = "240"/>
<p align="left">

[![Star History Chart](https://api.star-history.com/svg?repos=PaddlePaddle/PaddleMaterials&type=date&legend=top-left)](https://www.star-history.com/#PaddlePaddle/PaddleMaterilas&type=date&legend=top-left)

Join the PaddleMaterials WeChat group to discuss with us!
<p align="left">
 <img src="docs/wechat_group.png" align="middle" width = "100"/>
<p align="left">

## Contribute to PaddleMaterials

For developer, please refer to [architecture](docs/ARCHITECTURE_ch.md).

---

## 📜 License

PaddleMaterials is licensed under the [Apache License 2.0](LICENSE).

---

## 🎓 Citation

```bibtex
@misc{paddlematerials2025,
  title={PaddleMaterials, a deep learning toolkit based on PaddlePaddle for material science.},
  author={PaddleMaterials Contributors},
  howpublished = {\url{https://github.com/PaddlePaddle/PaddleMaterials}},
  year={2025}
}
```

---

## Acknowledgements

This repository references code from the following projects:

[PaddleScience](https://github.com/PaddlePaddle/PaddleScience) |
[Matgl](https://github.com/materialsvirtuallab/matgl) |
[CDVAE](https://github.com/txie-93/cdvae) |
[DiffCSP](https://github.com/jiaor17/DiffCSP) |
[MatterGen](https://github.com/microsoft/mattergen) |
[MatterSim](https://github.com/microsoft/mattersim) |
[CHGNet](https://github.com/CederGroupHub/chgnet) |
[AIRS](https://github.com/divelab/AIRS)
