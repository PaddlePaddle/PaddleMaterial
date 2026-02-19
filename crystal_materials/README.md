# CM-Crystal Materials

## 1.Introduction

Crystal Materials (CM) is a task category in PaddleMaterials for crystal-related model workflows.
Current support includes microscopy enhancement for crystalline materials with SFIN.

The supported model is:

- **SFIN**: *Noise Calibration and Spatial-Frequency Interactive Network for STEM Image Enhancement* (CVPR 2025)  
  Paper: https://arxiv.org/pdf/2504.02555

## 2.Models Matrix

| **Supported Functions**                           | **[SFIN](./configs/sfin/README.md)** |
| ------------------------------------------------- | :-----------------------------------: |
| **Crystal Microscopy**                            |                                       |
| STEM image denoising/enhancement                  |                  ✅                   |
| **ML Capabilities · Training**                    |                                       |
| Single-GPU                                        |                  ✅                   |
| Distributed training                              |                  ✅                   |
| Mixed precision (AMP)                             |                  —                    |
| Fine-tuning                                       |                  ✅                   |
| **ML Capabilities · Evaluation**                  |                                       |
| PSNR                                              |                  ✅                   |
| SSIM                                              |                  ✅                   |
| **Datasets**                                      |                                       |
| Paired STEM `noisy` / `gt_enhance` image datasets |                  ✅                   |

## 3.Quick Start

Run commands from the `PaddleMaterials` root directory.  
If `ppmat` is not installed in your environment, install once:

```bash
pip install -e . --no-build-isolation
```

Train:

```bash
python crystal_materials/train.py \
  -c crystal_materials/configs/sfin/sfin_tem_enhance.yaml
```

Predict with provided SFIN checkpoint:

```bash
python crystal_materials/predict.py \
  --config_path crystal_materials/configs/sfin/sfin_tem_enhance.yaml \
  --checkpoint_path ../sfin/checkpoints/sfin_he_500.pdparams \
  --input_dir ../sfin/data_test/noisy \
  --output_dir ./output/sfin_tem_enhance/predictions
```

Evaluate PSNR/SSIM:

```bash
python crystal_materials/evaluate.py \
  --gt_dir ../sfin/data_test/gt_enhance \
  --pred_dir ./output/sfin_tem_enhance/predictions
```

For model details, dataset format, and tiny smoke-test commands, see `crystal_materials/configs/sfin/README.md`.
