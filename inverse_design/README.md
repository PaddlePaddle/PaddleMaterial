# Inverse Design

This directory contains models for **inverse materials design** — generating novel materials with target properties using generative models.

## AlloyGAN

**Paper**: *Inverse Materials Design by Large Language Model-Assisted Generative Framework*
(Hao et al., arXiv:2502.18127, 2025)

**Repository**: https://github.com/photon-git/AlloyGAN

AlloyGAN uses Conditional Generative Adversarial Networks (CGAN) to inversely design metallic glass alloys with desired glass-forming ability (GFA) properties.

### Quick Start

```bash
# 1. Prepare dataset (downloads PDF, parses, generates CSV)
pip install pdfplumber requests
python tools/prepare_alloy_data.py --output_dir ./data/alloy/

# 2. Train CGAN (primary model — 10K iterations, ~5 min on CPU)
python inverse_design/train.py -c inverse_design/configs/alloygan/alloygan_cgan.yaml

# 3. Train standard GAN (optional comparison)
python inverse_design/train.py -c inverse_design/configs/alloygan/alloygan_gan.yaml
```

### Models

| Model | Architecture | Noise Dim | Conditions | Paper Wasserstein Distance |
|-------|-------------|-----------|------------|---------------------------|
| AlloyGAN | G(100→512→40), D(40→1024→1) | 100 | None | 0.48 (Cu) |
| AlloyCGAN | G(31→512→40), D(66→1024→1) | 5 | 26-dim (Tg,Tx,Tl + 23 GFA) | **0.41** (Cu) |

### Dataset

The dataset consists of 1,302 metallic glass alloy entries from 200+ published papers, collected via LLM-assisted text mining. Each entry has:
- **40 element composition fractions** (atomic %)
- **3 thermal transition temperatures**: Tg (glass), Tx (crystallization), Tl (liquidus)
- **23 glass-forming ability (GFA) criteria** (derived from Tg/Tx/Tl)

For training, only Cu (156), Fe (257), Ti (167), and Zr (213) subsets are used (793 entries total).

### Results

Paper-reported metrics (Cu subset):

| Model | Wasserstein Distance ↓ | Trg MAE ↓ | Trg MSE ↓ | Trg R² ↑ |
|-------|----------------------|-----------|-----------|----------|
| GAN | 0.48 | — | — | — |
| GAN+ | 0.60 | — | — | — |
| **CGAN** | **0.41** | **0.0074** | **0.0058** | **0.8030** |

### Configuration

Override any config value from the command line:

```bash
# Change training iterations
python inverse_design/train.py -c <config>.yaml Trainer.generator_iters=5000

# Use different dataset categories
python inverse_design/train.py -c <config>.yaml Dataset.train.dataset.__init_params__.categories='[Cu,Zr]'
```

### References

```bibtex
@article{hao2025alloygan,
  title={Inverse Materials Design by Large Language Model-Assisted Generative Framework},
  author={Hao, Yun and Fan, Che and Ye, Beilin and Lu, Wenhao and Lu, Zhen and Zhao, Peilin and Gao, Zhifeng and Wu, Qingyao and Liu, Yanhui and Wen, Tongqi},
  journal={arXiv preprint arXiv:2502.18127},
  year={2025}
}
```
