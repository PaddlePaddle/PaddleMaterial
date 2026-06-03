# ECFormer Model

## Overview

ECFormer is a deep learning model for predicting ECD (Electronic Circular Dichroism) and IR (Infrared) spectra from molecular structures. The model architecture consists of:

1. **GeoGNN** - Geometry-enhanced Graph Neural Network for molecular feature extraction
2. **Transformer Encoder** - For learning peak property features
3. **Prediction Heads** - For predicting peak number, position, and symbol/intensity

## Architecture

```
Molecular Graph (atoms, bonds, bond angles)
    ↓
GeoGNN Encoder
    ├── Atom Encoder
    ├── Bond Encoder
    ├── Bond Float Encoder (RBF)
    ├── Bond Angle Encoder (RBF)
    └── GIN Convolution Layers (x5)
    ↓
Graph Pooling (Attention)
    ↓
Transformer Encoder (x2 layers)
    ↓
Prediction Heads:
    ├── Peak Number (Linear → 9 classes)
    ├── Peak Position (Linear → 20 classes)
    └── Peak Height (Linear → 2 classes for ECD / 1 for IR)
```

## Key Components

### GeoGNN
- Uses dual graph structure (atom-bond graph + bond-angle graph)
- Geometry-enhanced with RBF encoding for bond lengths and angles
- 5-layer GIN convolution for message passing

### Transformer Encoder
- 2 layers with 4 attention heads
- Dropout: 0.1
- Batch first: True
- d_model: 128

### Prediction Heads
- **Peak Number**: Classification (max 9 peaks for ECD, 15 for IR)
- **Peak Position**: Classification (20 position classes for ECD, 36 for IR)
- **Peak Symbol**: Classification (2 classes: positive/negative for ECD)
- **Peak Intensity**: Regression (1 value for IR)

## Usage

```python
from ppmat.models.ecformer import ECFormerECD, ECFormerIR

# ECD model
ecd_model = ECFormerECD(
    full_atom_feature_dims=[119, 9, 12, 14, 17, 9, 14, 2, 10],
    full_bond_feature_dims=[8, 23, 3],
    emb_dim=128,
    num_layers=5,
    num_heads=4,
    tf_layers=2,
    max_peaks=9,
    num_position_classes=20,
    height_classes=2,
)

# IR model
ir_model = ECFormerIR(
    full_atom_feature_dims=[119, 9, 12, 14, 17, 9, 14, 2, 10],
    full_bond_feature_dims=[8, 23, 3],
    emb_dim=128,
    num_layers=5,
    num_heads=4,
    tf_layers=2,
    max_peaks=15,
    num_position_classes=36,
)
```

## Precision Alignment

Alignment test results (weights set to 0.5, float64 precision, CUDA fp64):

| Metric | Status | Diff |
|--------|--------|------|
| Peak Number | ✅ Aligned | < 1e-10 |
| Peak Position | ✅ Aligned | < 1e-10 |
| Peak Height | ✅ Aligned | < 1e-10 |

## References

- Paper: [ECDFormer](https://arxiv.org/abs/2401.03403)
- Original implementation: [GitHub](https://github.com/HowardLi1984/ECDFormer)
