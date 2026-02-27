# SPEN-Spectrum Enhancement

## 1.Introduction

Spectrum Enhancement (SE) focuses on enhancing and denoising spectral and microscopy data. Leveraging advanced deep learning techniques, SE aims to recover high-quality signals from noisy observations, enabling more accurate analysis of material properties at the atomic scale. This task is particularly valuable for STEM (Scanning Transmission Electron Microscopy) image processing, where noise reduction can significantly improve the visualization of crystal structures and defects.

## 2.Framework Support Matrix

| **Supported Functions**             | **Support** |
| ----------------------------------- | :---------: |
| **ML Capabilities · Training**      |             |
| Single-GPU                          |      ✅     |
| Distributed training                |      ✅     |
| Mixed precision (AMP)               |      —      |
| Fine-tuning                         |      ✅     |
| **ML Capabilities · Predict**       |             |
| Standard inference                  |      ✅     |
| Distributed inference               |      —      |
| **Data Pipeline**                   |             |
| Local dataset loading               |      ✅     |
| Auto dataset download               |      ✅     |
| **Task Workflow**                   |             |
| Training / Evaluation / Prediction  |      ✅     |

## 3.Model README

- [SFIN](./configs/sfin/README.md)
