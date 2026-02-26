# Spectrum Enhancement

## 1.Introduction

Spectrum Enhancement (SE) focuses on enhancing and denoising spectral and microscopy data for crystalline materials. Leveraging advanced deep learning techniques, SE aims to recover high-quality signals from noisy observations, enabling more accurate analysis of material properties at the atomic scale. This task is particularly valuable for STEM (Scanning Transmission Electron Microscopy) image processing, where noise reduction can significantly improve the visualization of crystal structures and defects.

Current SFIN cases support:
- HAADF mode: `enhance` and `detect`
- BF mode: `enhance` and `detect`

## 2.Models Matrix

| **Supported Functions**                           | **[SFIN](./configs/sfin/README.md)** |
| ------------------------------------------------- | :-----------------------------------: |
| **Microscopy Enhancement**                        |                                       |
| STEM image denoising/enhancement                  |                  ✅                   |
| STEM image detection-target restoration           |                  ✅                   |
| **ML Capabilities · Training**                    |                                       |
| Single-GPU                                        |                  ✅                   |
| Distributed training                              |                  ✅                   |
| Mixed precision (AMP)                             |                  —                    |
| Fine-tuning                                       |                  ✅                   |
| **ML Capabilities · Evaluation**                  |                                       |
| PSNR                                              |                  ✅                   |
| SSIM                                              |                  ✅                   |
| **Datasets**                                      |                                       |
| HAADF/BF paired `noisy` / `gt_enhance` / `gt_detect` datasets | ✅ |

## 3.Configurations

| Config | Mode | Target |
| --- | --- | --- |
| `sfin_tem_enhance.yaml` | HAADF (TEM) | `gt_enhance` |
| `sfin_tem_detect.yaml` | HAADF (TEM) | `gt_detect` |
| `sfin_bf_enhance.yaml` | BF | `gt_enhance` |
| `sfin_bf_detect.yaml` | BF | `gt_detect` |

**Notice**: 🌟 represents original research work published from PaddleMaterials toolkit
