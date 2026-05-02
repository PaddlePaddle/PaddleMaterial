# DM2

[DM2: Diffusion Models for Disordered Materials](https://arxiv.org/abs/2507.05024)
generates amorphous structures by learning an equivariant denoising vector field over
periodic atomistic graphs. This PaddleMaterials implementation adapts the official
`digital-synthesis-lab/DM2` demo recipe and integrates it with the PPMat trainer,
sampler, dataset, scheduler and metric builders.

## Coverage

| System | Config | Condition | Metric target |
| --- | --- | --- | --- |
| a-SiO2 | [dm2_sio2_unconditional.yaml](dm2_sio2_unconditional.yaml) | none | RDF Wasserstein, Si-O coordination |
| a-SiO2 | [dm2_sio2_conditional.yaml](dm2_sio2_conditional.yaml) | cooling rate | RDF Wasserstein, Si-O coordination |
| Cu50Zr50 | [dm2_cuzr.yaml](dm2_cuzr.yaml) | none | RDF Wasserstein |

`dm2_sio2.yaml` is kept as a compact backward-compatible SiO2 config.

## Data And Assets

Use `DM2StructureDataset` for ASE-readable periodic structures such as
`lammps-data`, `extxyz` or `cif`. The dataset encodes atomic species as contiguous
ids for the embedding table and preserves the original atomic numbers for sampled
structure export.

Expected local layout:

```text
data/dm2/
  sio2/
    unconditional/{train,val,sample,reference}/
    conditional/{train,val,sample,reference}/
  cuzr/{train,val,sample,reference}/
```

The official PyTorch DM2 repository provides demo SiO2 training files, SiO2 random
initial structures and three pretrained checkpoints:

- `gen-a-sio2-uncond-v1.pt`
- `gen-a-sio2-cond-v1.pt`
- `gen-cu50zr50-v1.pt`

For final Hackathon acceptance, upload the original dataset files, converted
Paddle checkpoints and training logs through the PaddleMaterials maintainers'
requested channel, then put the provided Baidu/BOS links here and in the PR:

| Asset | Link |
| --- | --- |
| a-SiO2 unconditional data/checkpoint/log | TODO |
| a-SiO2 conditional data/checkpoint/log | TODO |
| Cu50Zr50 data/checkpoint/log | TODO |

## Training

```bash
python structure_generation/train.py -c structure_generation/configs/dm2/dm2_sio2_unconditional.yaml
python structure_generation/train.py -c structure_generation/configs/dm2/dm2_sio2_conditional.yaml
python structure_generation/train.py -c structure_generation/configs/dm2/dm2_cuzr.yaml
```

The default training recipe follows the official demos: large-cutoff graph
construction, Gaussian rattle noise, cutoff edge down-selection, AdamW with
learning rate `2e-4`, and displacement MSE loss.

## Sampling And Metrics

```bash
python structure_generation/sample.py \
  --config_path structure_generation/configs/dm2/dm2_sio2_unconditional.yaml \
  --checkpoint_path ./output/dm2_sio2_unconditional/checkpoints/latest.pdparams \
  --mode by_dataloader \
  --save_path ./output/dm2_sio2_unconditional/samples

python structure_generation/sample.py \
  --config_path structure_generation/configs/dm2/dm2_sio2_unconditional.yaml \
  --checkpoint_path ./output/dm2_sio2_unconditional/checkpoints/latest.pdparams \
  --mode compute_metric
```

DM2 sampling uses graph batches from the `Sample.data` section. `by_num_atoms` and
`by_chemical_formula` are not the recommended DM2 entrypoints because DM2 needs an
initial periodic graph, not only a composition.

## Validation Requirements

Hackathon final review requires evidence beyond smoke tests:

| Item | Requirement | Artifact |
| --- | --- | --- |
| Forward parity | Paddle vs official PyTorch score/displacement diff <= 1e-6 | parity script output or screenshot |
| Backward parity | same data, 2+ epochs, aligned loss curve | train logs |
| Sampling quality | generation metric error within 5% | RDF/coordination report |
| Systems | a-SiO2 unconditional, a-SiO2 conditional, Cu50Zr50 | config + checkpoint + metric |

This config directory provides the runnable PPMat entrypoints and metric hooks for
those artifacts. The numeric reproduction results must be produced with the final
uploaded data and checkpoints.

## Citation

```bibtex
@article{yang2025generative,
  title={A generative diffusion model for amorphous materials},
  author={Yang, Kai and Schwalbe-Koda, Daniel},
  journal={arXiv:2507.05024},
  year={2025}
}
```
