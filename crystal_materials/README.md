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

**Notice**:🌟 represent originate research work published from paddlematerials toolkit