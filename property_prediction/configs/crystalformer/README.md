# CrystalFormer

[CrystalFormer: Infinitely Connected Attention for Periodic Structure Encoding](https://arxiv.org/abs/2403.04745)

## Abstract

CrystalFormer is a lattice-aware Transformer for crystal property prediction. It computes pairwise minimum-image distances considering periodic lattice translations and uses Gaussian RBF features to augment multi-head attention with crystallographic distance bias. A global pooling followed by an MLP regression head predicts scalar properties (formation energy, band gap, etc.).

## Model

CrystalFormer encodes atom features via a linear projection, computes pairwise distances across periodic images within a configurable lattice range, and expands distances using Gaussian radial basis functions. These distance features bias the attention weights in a stack of Transformer encoder layers. After encoding, atom representations are pooled (max or mean) and passed through an MLP head for property regression.

### Implementation Notes

This is a **simplified re-implementation** of the original CrystalFormer paper. Key differences from Taniai et al. (2024):

| Aspect | Original Paper | This Implementation |
|--------|---------------|---------------------|
| Periodic images | Infinite sum via Ewald decomposition | Truncated lattice enumeration |
| Distance encoding | Dual α/β Ewald domain features | Single-domain Gaussian RBF |
| Attention | Gaussian-decayed attention kernel | Standard softmax + distance bias |
| Complexity | O(N²·K) where K = Ewald terms | O(N²·(2r+1)³) where r = lattice_range |

### `lattice_range` Parameter

The `lattice_range` parameter (default: 2) controls how many periodic images are considered. A value of `r` generates `(2r+1)³` lattice translations (e.g., r=2 → 125 images). Larger values improve accuracy for large unit cells at the cost of memory and compute. For most materials with < 20 atoms per cell, the default is sufficient.

## Training

```bash
# single-gpu training
python property_prediction/train.py -c property_prediction/configs/crystalformer/crystalformer_mp2018_train_60k_e_form.yaml

# multi-gpu training
python -m paddle.distributed.launch --gpus="0,1,2,3" property_prediction/train.py -c property_prediction/configs/crystalformer/crystalformer_mp2018_train_60k_e_form.yaml
```

## Validation

```bash
python property_prediction/train.py -c property_prediction/configs/crystalformer/crystalformer_mp2018_train_60k_e_form.yaml Global.do_eval=True Global.do_train=False Global.do_test=False Trainer.pretrained_model_path='your model path(*.pdparams)'
```

## Testing

```bash
python property_prediction/train.py -c property_prediction/configs/crystalformer/crystalformer_mp2018_train_60k_e_form.yaml Global.do_test=True Global.do_train=False Global.do_eval=False Trainer.pretrained_model_path='your model path(*.pdparams)'
```

## Citation

```
@inproceedings{taniai2024crystalformer,
  title={CrystalFormer: Infinitely Connected Attention for Periodic Structure Encoding},
  author={Taniai, Tatsunori and Igarashi, Ryo and Suzuki, Yuta and Koide, Naoya and Saito, Kotaro and Tanaka, Koji},
  booktitle={International Conference on Learning Representations},
  year={2024}
}
```
