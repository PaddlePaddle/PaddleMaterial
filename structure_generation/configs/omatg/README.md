# OMatG

[Open Materials Generation with Stochastic Interpolants](https://openreview.net/forum?id=gHGrzxFujU) (ICML 2025) /
[All that structure matches does not glitter](https://openreview.net/forum?id=ig9ujp50D4) (NeurIPS 2025)

## Abstract

A state-of-the-art generative model for crystal structure prediction and *de novo* generation of inorganic crystals.
OMatG implements the [stochastic interpolants (SIs) framework](https://arxiv.org/abs/2303.08797) that bridges samples
from a base distribution to the target data distribution. A stochastic interpolant
$x_t = \alpha(t)\,x_0 + \beta(t)\,x_1 + \gamma(t)\,z$ evolves base samples $x_0$ into data samples $x_1$ over time
$t\in[0,1]$. The time-dependent density is realized via deterministic (ODE) or stochastic (SDE) sampling, requiring
only a learned velocity field $b^\theta(t, x)$ (and optionally a denoiser $z^\theta(t, x)$ for SDE).

OMatG defines a crystalline material by its unit cell ($\mathbf{L}\in\mathbb{R}^{3\times3}$), fractional coordinates
($\mathbf{X}\in[0,1)^{3\times N}$ with periodic boundary conditions), and discrete atomic species
($\mathbf{A}\in\mathbb{Z}^N_{>0}$). The SI framework handles the continuous variables $\{\mathbf{X}, \mathbf{L}\}$
while discrete species $\mathbf{A}$ are treated with [discrete flow matching](https://arxiv.org/abs/2402.04997).

Two crystal generation modes are supported:
1. **CSP** (crystal structure prediction): atomic species are fixed; only coordinates and lattice vectors evolve.
2. **DNG** (*de novo* generation): all species are masked at start and evolve together with structure.

## Model Architecture

OMatG consists of the following core components:

- **CSPNet**: Crystal structure prediction network using message-passing GNN architecture, supporting fully-connected (fc) or k-NN (knn) edge construction.
- **Stochastic Interpolants**: SI framework supporting ODE/SDE integration, with multiple interpolant types (Linear/Trigonometric/EncDec/VESBD/VPSBD).
- **IndependentSampler**: Independent distribution sampler for position/lattice/species base distributions.

## Datasets

### Included Datasets

Several standard material datasets are included as LMDB files:

| Dataset | Structures | Max Atoms | Description |
|---------|-----------:|:---------:|-------------|
| MP-20 | 45,229 | 20 | [Materials Project](https://pubs.aip.org/aip/apm/article/1/1/011002/119685) structures |
| MPTS-52 | 40,476 | 52 | [Chronological MP split](https://joss.theoj.org/papers/10.21105/joss.05618) |
| Perov-5 | 18,928 | 5 | [Perovskite dataset](https://pubs.rsc.org/en/content/articlelanding/2012/ee/c2ee22341d) |
| Alex-MP-20 | 675,204 | — | Consolidated [Alexandria](https://arxiv.org/abs/2210.00579) + MP-20 |


### Supported Datasets

Pretrained weights are available for 5 dataset × mode combinations.
Training configs are provided for `mp_20` (CSP + DNG); other datasets
(`perov_5_csp`, `mpts_52_csp`, `alex_mp_20_csp`) can be used by changing
the `file_path` in the dataset section of the config.

| Dataset | Mode | Variants | Weight Index |
|---------|:----:|:--------:|--------------|
| mp_20_csp | CSP | 11 | `build_omatg_model("mp_20_csp", variant)` |
| mp_20_dng | DNG | 11 | `build_omatg_model("mp_20_dng", variant)` |
| perov_5_csp | CSP | 11 | `build_omatg_model("perov_5_csp", variant)` |
| mpts_52_csp | CSP | 8 | `build_omatg_model("mpts_52_csp", variant)` |
| alex_mp_20_csp | CSP | 11 | `build_omatg_model("alex_mp_20_csp", variant)` |

See `ppmat/models/omatg/__init__.py` (`OMATG_WEIGHTS` dict) for all weight URLs.

## Configuration Files

| File | Mode | Dataset | Description |
|------|:----:|---------|-------------|
| `omatg_mp20_csp.yaml` | CSP | MP-20 | Simplified MSE (fixed lr) |
| `omatg_mp20_csp_linear_ode.yaml` | CSP | MP-20 | Linear-ODE (Cosine LR) |
| `omatg_mp20_dng_linear_sde.yaml` | DNG | MP-20 | Linear-SDE (species prediction + WeightDecay) |
| `omatg_mp20_csp_sample.yaml` | CSP | MP-20 | Sampling by number of atoms |
| `omatg_mp20_dng_sample.yaml` | DNG | MP-20 | DNG sampling (species prediction) |

## Results

<table>
    <tr>
        <th nowrap>Model</th>
        <th nowrap>Dataset</th>
        <th nowrap>Mode</th>
        <th nowrap>Interpolant</th>
        <th nowrap>Config (Train / Sample)</th>
        <th nowrap>Weight</th>
    </tr>
    <tr>
        <td nowrap>mp_20_csp</td><td nowrap>MP-20</td><td nowrap>CSP</td><td nowrap>Linear-ODE</td>
        <td nowrap><a href="omatg_mp20_csp_linear_ode.yaml">train</a> / <a href="omatg_mp20_csp_sample.yaml">sample</a></td>
        <td nowrap><a href="https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mp_20_csp/Linear-ODE.pdparams">weight</a></td>
    </tr>
    <tr>
        <td nowrap>mp_20_dng</td><td nowrap>MP-20</td><td nowrap>DNG</td><td nowrap>Linear-SDE</td>
        <td nowrap><a href="omatg_mp20_dng_linear_sde.yaml">train</a> / <a href="omatg_mp20_dng_sample.yaml">sample</a></td>
        <td nowrap><a href="https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mp_20_dng/Linear-SDE.pdparams">weight</a></td>
    </tr>
    <tr>
        <td nowrap>perov_5_csp</td><td nowrap>Perov-5</td><td nowrap>CSP</td><td nowrap>EncDec-ODE-Gamma</td>
        <td nowrap>use mp_20 CSP config, change <code>file_path</code> to <code>./data/perov_5/</code></td>
        <td nowrap><a href="https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_perov_5_csp/EncDec-ODE-Gamma.pdparams">weight</a></td>
    </tr>
    <tr>
        <td nowrap>mpts_52_csp</td><td nowrap>MPTS-52</td><td nowrap>CSP</td><td nowrap>EncDec-ODE-Gamma</td>
        <td nowrap>use mp_20 CSP config, change <code>file_path</code> to <code>./data/mpts_52/</code></td>
        <td nowrap><a href="https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_mpts_52_csp/EncDec-ODE-Gamma.pdparams">weight</a></td>
    </tr>
    <tr>
        <td nowrap>alex_mp_20_csp</td><td nowrap>Alex-MP-20</td><td nowrap>CSP</td><td nowrap>EncDec-ODE-Gamma</td>
        <td nowrap>use mp_20 CSP config, change <code>file_path</code> to <code>./data/alex_mp_20/</code></td>
        <td nowrap><a href="https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/structure_generation/OMatG/omatg_alex_mp_20_csp/EncDec-ODE-Gamma.pdparams">weight</a></td>
    </tr>
</table>

## Training

```bash
# Smoke training (1 epoch, small batch, no eval)
python structure_generation/train.py \
    -c structure_generation/configs/omatg/omatg_mp20_csp.yaml \
    Trainer.max_epochs=1 \
    Trainer.do_eval=False \
    Dataset.train.sampler.__init_params__.batch_size=32 \
    Dataset.val.sampler.__init_params__.batch_size=32

# Full training (requires data/mp_20/ dataset)
python structure_generation/train.py \
    -c structure_generation/configs/omatg/omatg_mp20_csp_linear_ode.yaml \
    Trainer.max_epochs=500 \
    Trainer.output_dir=./output/omatg_mp20_csp
```

## Validation

```bash
# Evaluate on the validation split using a saved checkpoint
python structure_generation/train.py \
    -c structure_generation/configs/omatg/omatg_mp20_csp_linear_ode.yaml \
    Global.do_train=False \
    Global.do_eval=True \
    Trainer.pretrained_model_path=./output/omatg_mp20_csp/checkpoints
```

## Testing

```bash
# Evaluate on the test split
python structure_generation/train.py \
    -c structure_generation/configs/omatg/omatg_mp20_csp_linear_ode.yaml \
    Global.do_train=False \
    Global.do_test=True \
    Global.do_eval=False \
    Trainer.pretrained_model_path=./output/omatg_mp20_csp/checkpoints
```

## Sample

```bash
# Sample by number of atoms
python structure_generation/sample.py \
    --config_path structure_generation/configs/omatg/omatg_mp20_csp_sample.yaml \
    --checkpoint_path ./output/omatg_mp20_csp/checkpoints/best.pdparams \
    --mode by_num_atoms \
    --num_atoms 8 \
    --save_path ./results/omatg_samples

# Sample by chemical formula
python structure_generation/sample.py \
    --config_path structure_generation/configs/omatg/omatg_mp20_csp_sample.yaml \
    --checkpoint_path ./output/omatg_mp20_csp/checkpoints/best.pdparams \
    --mode by_chemical_formula \
    --chemical_formula LiMnO2 \
    --save_path ./results/omatg_samples

# Batch sample by dataloader
python structure_generation/sample.py \
    --config_path structure_generation/configs/omatg/omatg_mp20_csp.yaml \
    --checkpoint_path ./output/omatg_mp20_csp/checkpoints/best.pdparams \
    --mode by_dataloader \
    --save_path ./results/omatg_samples
```



## Citation

```bibtex
@inproceedings{
    hoellmer2025,
    title={Open Materials Generation with Stochastic Interpolants},
    author={Philipp H{\"o}llmer and Thomas Egg and Maya Martirossyan and Eric
    Fuemmeler and Zeren Shui and Amit Gupta and Pawan Prakash and Adrian
    Roitberg and Mingjie Liu and George Karypis and Mark Transtrum and Richard
    Hennig and Ellad B. Tadmor and Stefano Martiniani},
    booktitle={Forty-second International Conference on Machine Learning},
    year={2025},
    url={https://openreview.net/forum?id=gHGrzxFujU},
    archivePrefix={arXiv},
    eprint={2502.02582},
    primaryClass={cs.LG},
}
```

```bibtex
@inproceedings{
    martirossyan2025,
    title={All that structure matches does not glitter},
    author={Maya Martirossyan and Thomas Egg and Philipp H{\"o}llmer 
    and George Karypis and Mark Transtrum and Adrian Roitberg 
    and Mingjie Liu and Richard Hennig and Ellad B. Tadmor and Stefano Martiniani},
    booktitle={Thirty-Ninth Annual Conference on Neural Information Processing Systems},
    year={2025},
    url={https://openreview.net/forum?id=ig9ujp50D4},
    archivePrefix={arXiv},
    eprint={2509.12178},
    primaryClass={cs.LG},
}
```
