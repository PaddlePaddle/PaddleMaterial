# UMA

[UMA: A Family of Universal Models for Atoms](https://github.com/facebookresearch/fairchem/tree/main/src/fairchem/core/models/uma)

## Abstract

UMA is a family of universal machine learning interatomic potentials for
atomistic systems. The PaddleMaterials implementation provides a Paddle-based
UMA/eSCN model for structure-to-energy-and-force tasks. It follows the
PaddleMaterials interatomic-potential workflow: datasets are built under
`ppmat.datasets`, graphs are constructed through `Global.graph_converter` and
`Dataset.*.build_graph_cfg`, and training/evaluation runs through
`interatomic_potentials/train.py`.

## Datasets

UMA uses atomistic structures with total energy and atomic force labels.

| Field | Description |
| :-- | :-- |
| `energy` | Total structure energy |
| `forces` | Atomic forces with shape `[num_atoms, 3]` |

The current configs use ASELMDB data. `UMAAseDBDataset` is implemented in
`ppmat.datasets.uma_dataset` and supports graph construction through
`build_graph_cfg`.

### OMat24

The OMat24 `rattled-500` split is used for material-domain S2EF training.

| Dataset | Split | Download |
| :-- | :-- | :-- |
| OMat24 rattled-500 | train | [official source](https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/omat/train/rattled-500.tar.gz) |
| OMat24 rattled-500 | val | [official source](https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241220/omat/val/rattled-500.tar.gz) |

Default paths:

```text
data/omat24/train/rattled-500
data/omat24/val/rattled-500
data/omat24/test/rattled-500
```

### OC20 S2EF

OC20 S2EF is used for catalyst-domain S2EF validation. The official files are
distributed in compressed `extxyz` format and should be converted to ASELMDB
before using the provided UMA configs.

| Dataset | Split | Download |
| :-- | :-- | :-- |
| OC20 S2EF 200K | train | [official source](https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_train_200K.tar) |
| OC20 S2EF | val-id | [official source](https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_val_id.tar) |

Default converted paths:

```text
data/oc20/uma_aselmdb/train/train.aselmdb
data/oc20/uma_aselmdb/val/val.aselmdb
data/oc20/uma_aselmdb/test/test.aselmdb
```

## Models

UMA constructs local atomic neighborhoods and applies an eSCN-style equivariant
message-passing backbone. Atomic numbers, positions, periodic cells, charges,
spins, and optional task names are embedded into the model. The graph converter
builds neighbor edges and periodic cell offsets, then the backbone updates
spherical node features through equivariant interaction blocks. Prediction heads
map the learned atomic representations to total energy and atomic forces.

The PaddleMaterials implementation exposes:

| Component | PaddleMaterials entry |
| :-- | :-- |
| Model | `ppmat.models.uma.escn_md.UMASingleTaskModel` |
| Dataset | `ppmat.datasets.uma_dataset.UMAAseDBDataset` |
| Graph converter | `ppmat.models.uma.uma_graph_converter.UMAGraphConverter` |

Precomputed Wigner-d coefficients required by the UMA rotation module are stored
as Paddle tensors in `ppmat/models/uma/Jd.pdparams`.

## Results

<table>
    <head>
        <tr>
            <th nowrap="nowrap">Model Name</th>
            <th nowrap="nowrap">Dataset</th>
            <th nowrap="nowrap">Energy MAE</th>
            <th nowrap="nowrap">Force MAE</th>
            <th nowrap="nowrap">GPUs</th>
            <th nowrap="nowrap">Training time</th>
            <th nowrap="nowrap">Config</th>
            <th nowrap="nowrap">Checkpoint | Log</th>
        </tr>
    </head>
    <body>
        <tr>
            <td nowrap="nowrap">uma_omat24_r500_budget_s2ef</td>
            <td nowrap="nowrap">OMat24 rattled-500 budget</td>
            <td nowrap="nowrap">11.780332</td>
            <td nowrap="nowrap">0.412024</td>
            <td nowrap="nowrap">1 x RTX 4090</td>
            <td nowrap="nowrap">~1h09m</td>
            <td nowrap="nowrap"><a href="../../../test/uma/uma_omat24_r500_budget_s2ef.yaml">config</a></td>
            <td nowrap="nowrap">To be released</td>
        </tr>
        <tr>
            <td nowrap="nowrap">uma_oc20_50k_budget_s2ef</td>
            <td nowrap="nowrap">OC20 S2EF budget</td>
            <td nowrap="nowrap">16.703999</td>
            <td nowrap="nowrap">0.182028</td>
            <td nowrap="nowrap">1 x RTX 4090</td>
            <td nowrap="nowrap">~55m</td>
            <td nowrap="nowrap"><a href="../../../test/uma/uma_oc20_50k_budget_s2ef.yaml">config</a></td>
            <td nowrap="nowrap">To be released</td>
        </tr>
    </body>
</table>

The table reports fixed-budget migration validation results. Full-scale UMA
benchmark reproduction is not claimed in this PR.

## Training

```bash
python interatomic_potentials/train.py -c interatomic_potentials/configs/uma/uma_omat24_r500_s2ef.yaml
```

```bash
python interatomic_potentials/train.py -c interatomic_potentials/configs/uma/uma_oc20_200k_s2ef.yaml
```

## Evaluation

```bash
python interatomic_potentials/train.py -c interatomic_potentials/configs/uma/uma_omat24_r500_s2ef.yaml Global.do_train=False Global.do_eval=True Global.do_test=False Trainer.pretrained_model_path="path/to/checkpoint.pdparams"
```

## Testing

```bash
python interatomic_potentials/train.py -c interatomic_potentials/configs/uma/uma_omat24_r500_s2ef.yaml Global.do_train=False Global.do_eval=False Global.do_test=True Trainer.pretrained_model_path="path/to/checkpoint.pdparams"
```

## Citations

```bibtex
@misc{fairchem_uma,
  title = {UMA: A Family of Universal Models for Atoms},
  author = {FAIR Chemistry Team},
  howpublished = {\url{https://github.com/facebookresearch/fairchem}},
  year = {2025}
}
```
