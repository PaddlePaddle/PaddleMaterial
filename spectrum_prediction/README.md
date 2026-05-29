# ECFormer - Spectrum Prediction

## 1. Introduction

ECFormer is a Transformer-based model for predicting Electronic Circular Dichroism (ECD) and Infrared (IR) spectra from molecular structures. The model uses a geometric-enhanced graph neural network (GeoGNN) to extract molecular features and a Transformer encoder to predict peak properties (number, position, and symbol).

## 2. Model Matrix

| **Supported Functions** | **ECD** | **IR** |
|------------------------|---------|--------|
| Peak Number Prediction | ✅ | ✅ |
| Peak Position Prediction | ✅ | ✅ |
| Peak Symbol/Intensity Prediction | ✅ | ✅ |
| Single-GPU Training | ✅ | ✅ |
| Inference | ✅ | ✅ |

## 3. Datasets

- **ECD Dataset**: 22,190 chiral molecules with calculated ECD spectra
- **IR Dataset**: Infrared spectra dataset for molecular analysis

## 4. Results

### Precision Alignment Results

| Metric | Torch Output | Paddle Output | Diff | Status |
|--------|-------------|---------------|------|--------|
| Peak Number | [1e+08, ...] | [1e+08, ...] | < 1e-10 | ✅ Aligned |
| Peak Position | [19.25, ...] | [19.25, ...] | < 1e-10 | ✅ Aligned |
| Peak Height | [-0.5, ...] | [-0.5, ...] | < 1e-10 | ✅ Aligned |

**Alignment Method**:
- Weights set to 0.5 (binary exact representation)
- Float64 precision (CUDA fp64)
- Real data input (avoid Embedding overflow)
- Tolerance: 1e-10

## 5. Training

```bash
# ECD task - single GPU
python spectrum_prediction/train.py -c spectrum_prediction/configs/ecd.yaml

# IR task - single GPU
python spectrum_prediction/train.py -c spectrum_prediction/configs/ir.yaml
```

## 6. Inference

```bash
# ECD inference
python spectrum_prediction/train.py -c spectrum_prediction/configs/ecd.yaml --eval-only

# IR inference
python spectrum_prediction/train.py -c spectrum_prediction/configs/ir.yaml --eval-only
```

## 7. Model Architecture

```
Input: Molecular Graph (atoms, bonds, angles)
    ↓
GeoGNN (Geometry-enhanced Graph Neural Network)
    ↓
Graph Pooling
    ↓
Transformer Encoder
    ↓
Prediction Heads:
    - Peak Number (classification)
    - Peak Position (classification)
    - Peak Symbol/Intensity (classification/regression)
```

## 8. Citation

```bibtex
@article{li2025decoupled,
  title={Decoupled peak property learning for efficient and interpretable electronic circular dichroism spectrum prediction},
  author={Li, Hao and Long, Da and Yuan, Li and Wang, Yu and Tian, Yonghong and Wang, Xinchang and Mo, Fanyang},
  journal={Nature Computational Science},
  pages={1--11},
  year={2025},
  publisher={Nature Publishing Group US New York}
}
```
