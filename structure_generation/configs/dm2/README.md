# DM2

[DM2: Diffusion Models for Disordered Materials](https://arxiv.org/abs/2507.05024)
generates amorphous structures by learning a denoising vector field over periodic
atomistic graphs. The PaddleMaterials implementation follows the official DM2
training recipe:

- build a large-cutoff periodic graph from ASE structures;
- perturb atom positions with Gaussian rattle noise;
- downselect edges to the model cutoff;
- train a NequIP-style equivariant denoiser to predict the Cartesian displacement;
- iteratively denoise random or noisy structures for sampling.

## Data

Use `DM2StructureDataset` for ASE-readable files such as `lammps-data`, `extxyz`,
or `cif`. The dataset encodes species as contiguous ids for embedding and keeps
the original atomic numbers for writing sampled structures.

The Paddle implementation adapts the official DM2/graphite denoising design
from `digital-synthesis-lab/DM2` (MIT licensed).

For conditional DM2, pass per-structure `cooling_rates`; by default the dataset
stores `log10(cooling_rate)`, matching the official conditional demo.

## Training

Update `paths`, `species`, and `cooling_rates` in `dm2_sio2.yaml`, then run:

```bash
python structure_generation/train.py -c structure_generation/configs/dm2/dm2_sio2.yaml
```

For unconditional training, keep `Model.__init_params__.denoiser_cfg.use_condition`
as `false` and omit `cooling_rates` from the dataset.

## Sampling

Load a trained checkpoint through `structure_generation/sample.py` and provide a
DM2 graph batch from the sample dataset section. Sampling returns structure-array
dicts compatible with `BuildStructure(format="array")`.
