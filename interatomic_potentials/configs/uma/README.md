# UMA

[UMA: A Family of Universal Models for Atoms](https://github.com/facebookresearch/fairchem/tree/main/src/fairchem/core/models/uma)

## Abstract

UMA is a family of universal machine learning interatomic potentials for
atomistic systems. The PaddleMaterials implementation provides a Paddle-based
UMA/eSCN backbone, direct energy and force prediction heads, ASELMDB data
loading, single-domain training, and lightweight multi-task validation through
the standard `interatomic_potentials/train.py` workflow.

The default UMA configurations in this directory target S2EF-style training:
given atomic numbers, periodic structures, and positions, the model predicts the
total energy and per-atom forces.

## Datasets

UMA reads ASELMDB files containing atomistic structures and the following labels:

| Field | Description |
| :-- | :-- |
| `energy` | Total structure energy |
| `forces` | Atomic forces with shape `[num_atoms, 3]` |

### OMat24

The OMat24 `rattled-500` split is used by the default material-domain training
configuration.

| Dataset | Train | Val | Test | Download |
| :-- | --: | --: | --: | :-- |
| OMat24 rattled-500 | configurable | configurable | configurable | [train](https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241018/omat/train/rattled-500.tar.gz), [val](https://dl.fbaipublicfiles.com/opencatalystproject/data/omat/241220/omat/val/rattled-500.tar.gz) |

Default paths:

```text
data/omat24/train/rattled-500/train.aselmdb
data/omat24/val/rattled-500/val.aselmdb
data/omat24/test/rattled-500/test.aselmdb
```

### OC20 S2EF

The OC20 S2EF 200K and val-id splits are used for OC20-domain validation.
Official OC20 files are distributed as compressed `extxyz` files, so this
directory provides a converter that writes UMA-compatible ASELMDB files.

```bash
mkdir -p ./data/oc20/raw
curl -L https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_train_200K.tar -o ./data/oc20/raw/s2ef_train_200K.tar
curl -L https://dl.fbaipublicfiles.com/opencatalystproject/data/s2ef_val_id.tar -o ./data/oc20/raw/s2ef_val_id.tar
tar -xf ./data/oc20/raw/s2ef_train_200K.tar -C ./data/oc20/raw
tar -xf ./data/oc20/raw/s2ef_val_id.tar -C ./data/oc20/raw
```

```bash
python interatomic_potentials/configs/uma/prepare_oc20_s2ef_aselmdb.py \
  --raw-dir ./data/oc20/raw/s2ef_train_200K/s2ef_train_200K \
  --val-raw-dir ./data/oc20/raw/s2ef_val_id/s2ef_val_id \
  --out-dir ./data/oc20/uma_aselmdb \
  --train 50000 \
  --val 5000 \
  --test 5000
```

The converted directory follows this layout:

```text
data/oc20/uma_aselmdb/
  train/train.aselmdb
  val/val.aselmdb
  test/test.aselmdb
```

### Budget Validation Data

For migration validation on a single GPU, the following script prepares fixed
OMat24 and OC20 subsets:

```bash
bash interatomic_potentials/configs/uma/prepare_budget_data.sh ./data
```

| Config | Train | Val | Test |
| :-- | :-- | :-- | :-- |
| `uma_omat24_r500_budget_s2ef.yaml` | OMat24 rattled-500 train 50k | OMat24 rattled-500 val offset 0, 5k | OMat24 rattled-500 val offset 5k, 5k |
| `uma_oc20_50k_budget_s2ef.yaml` | OC20 S2EF train 50k | OC20 S2EF val-id offset 0, 5k | OC20 S2EF val-id offset 5k, 5k |

The required Python packages, including `ase`, `lmdb`, and `omegaconf`, are
listed in the PaddleMaterials `requirements.txt`. UMA uses the e3nn-compatible
operators implemented in `ppmat.models.common.e3nn`.

## Models

The PaddleMaterials UMA model builds neighbor graphs from periodic atomistic
structures, embeds atom types and optional dataset/task names, and applies an
eSCN-style equivariant message-passing backbone. The model predicts total
energy directly and trains forces as supervised vector targets.

The default single-domain model is
`ppmat.models.uma.escn_md.UMASingleTaskModel`.

| Setting | Value |
| :-- | :-- |
| Backbone | UMA/eSCN |
| Layers | 4 |
| `lmax` / `mmax` | 2 / 2 |
| Hidden / sphere / edge channels | 128 |
| Distance basis | 64 |
| Cutoff | 6.0 |
| Max neighbors | 30 |
| Optimizer | AdamW |
| Learning rate | `8e-4` with cosine decay to `8e-6` |
| Weight decay | `1e-3` |
| Loss weights | energy 10.0, forces 30.0 |

`ppmat/models/uma/Jd.pt` stores precomputed Wigner-d coefficients used by UMA
rotation modules. If the file is placed outside the packaged model directory,
set `PPMAT_UMA_JD_PATH` before running training or evaluation.

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
            <td nowrap="nowrap"><a href="uma_omat24_r500_budget_s2ef.yaml">config</a></td>
            <td nowrap="nowrap">To be released</td>
        </tr>
        <tr>
            <td nowrap="nowrap">uma_oc20_50k_budget_s2ef</td>
            <td nowrap="nowrap">OC20 S2EF budget</td>
            <td nowrap="nowrap">16.703999</td>
            <td nowrap="nowrap">0.182028</td>
            <td nowrap="nowrap">1 x RTX 4090</td>
            <td nowrap="nowrap">~55m</td>
            <td nowrap="nowrap"><a href="uma_oc20_50k_budget_s2ef.yaml">config</a></td>
            <td nowrap="nowrap">To be released</td>
        </tr>
    </body>
</table>

The table reports validation metrics from fixed budget subsets. These results
are intended to verify the Paddle migration, data loading, training, evaluation,
and checkpoint workflows, rather than to reproduce the full-scale fairchem UMA
benchmark.

## Training

Single-GPU training:

```bash
python interatomic_potentials/train.py -c interatomic_potentials/configs/uma/uma_omat24_r500_s2ef.yaml
```

Multi-GPU training:

```bash
python -m paddle.distributed.launch --gpus="0,1,2,3" interatomic_potentials/train.py -c interatomic_potentials/configs/uma/uma_omat24_r500_s2ef.yaml
```

Budget validation:

```bash
python interatomic_potentials/train.py -c interatomic_potentials/configs/uma/uma_omat24_r500_budget_s2ef.yaml
python interatomic_potentials/train.py -c interatomic_potentials/configs/uma/uma_oc20_50k_budget_s2ef.yaml
```

Multi-task smoke validation:

```bash
python interatomic_potentials/train.py -c interatomic_potentials/configs/uma/uma_multitask_budget_smoke.yaml
```

The multi-task smoke configuration uses real OC20 samples and OMat24 subsets as
task proxies to validate task-name propagation, dataset embedding, forward,
loss, backward, evaluation, and checkpoint saving.

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
