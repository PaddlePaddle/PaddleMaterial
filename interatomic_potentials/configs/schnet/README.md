# SchNet

[SchNet: A continuous-filter convolutional neural network for modeling quantum interactions](https://arxiv.org/abs/1706.08566)

## Abstract

SchNet is an end-to-end continuous-filter convolutional neural network for atomistic systems. It models quantum interactions directly from atomic numbers and coordinates, using distance-based filters with smooth cutoffs and atom-wise readout. In PaddleMaterials, this implementation targets paper-style small-molecule regression benchmarks and keeps training/inference interfaces consistent with the `interatomic_potentials` suite.

## Datasets

This SchNet case currently covers:

- QM9 (`U0`) with paper-aligned split setup
- MD17 (ethanol example, energy + force supervision)
- ISO17 (reference to test_other generalization)

Dataset sources:

- **QM9 (small molecules)**:
  - Original source: https://figshare.com/ndownloader/files/3195389
  - PaddleMaterials packaged mirror (used by SchNet config):  
    `https://paddle-org.bj.bcebos.com/paddlematerials/datasets/qm9/qm9.tar.gz`
  - PaddleMaterials raw mirror (fallback / non-packaged path):  
    `https://paddle-org.bj.bcebos.com/paddlematerials/datasets/qm9/dsgdb9nsd.xyz.tar.bz2`
- **MD17 / rMD17 (small molecules, energy + force)**:
  - Original source: https://www.quantum-machine.org/datasets/
  - PaddleMaterials package mirror:  
    `https://paddle-org.bj.bcebos.com/paddlematerials/datasets/MD17/md17.tar.gz`
  - Recommended format in this repo: `*.npz` with keys `R/Z/E/F`
- **ISO17 (isomer generalization benchmark)**:
  - Original source: https://www.quantum-machine.org/datasets/
  - PaddleMaterials package mirror:  
    `https://paddle-org.bj.bcebos.com/paddlematerials/datasets/ISO17/iso17.tar.gz`
  - Recommended format in this repo: `iso17.npz` with keys `R/Z/E` and optional `isomer_ids`

## Models

The Paddle implementation follows the SchNet interaction design:

- Gaussian RBF expansion for inter-atomic distances
- Cosine cutoff for locality
- Continuous-filter interaction blocks with residual updates
- Atom-wise readout for molecular properties

Config files:

- `schnet_qm9_lumo.yaml`: paper-aligned QM9 `U0` case (filename kept for compatibility)
- `schnet_md17_ethanol.yaml`: MD17 ethanol energy-force joint training
- `schnet_iso17.yaml`: ISO17 reference->test_other setting

Training hyperparameters in these SchNet configs are aligned to
SchNet-master/paper style (instead of current schnetpack defaults):

- Optimizer: `Adam`
- LR schedule: exponential decay (`lr=1e-3`, decay steps `100000`, gamma `0.96`)
- Global-step stop: `max_iter=5000000`
- Step-based validation/save: `eval_interval_steps=5000`, `save_interval_steps=50000`
- QM9 train batch: `32`, val/test batch: `100`
- QM9 split file: `split_qm9_110k_1k_seed42.npz` (`num_train=110000`, `num_val=1000`)

## Results

<table>
    <head>
        <tr>
            <th nowrap="nowrap">Model Name</th>
            <th nowrap="nowrap">Dataset</th>
            <th nowrap="nowrap">Task</th>
            <th nowrap="nowrap">Metric</th>
            <th nowrap="nowrap">GPUs</th>
            <th nowrap="nowrap">Training time</th>
            <th nowrap="nowrap">Config</th>
            <th nowrap="nowrap">Checkpoint | Log</th>
        </tr>
    </head>
    <body>
        <tr>
            <td nowrap="nowrap">schnet_qm9_u0</td>
            <td nowrap="nowrap">QM9</td>
            <td nowrap="nowrap">U0 regression</td>
            <td nowrap="nowrap">MAE: to_be_filled</td>
            <td nowrap="nowrap">~</td>
            <td nowrap="nowrap">~</td>
            <td nowrap="nowrap"><a href="schnet_qm9_lumo.yaml">schnet_qm9_lumo.yaml</a></td>
            <td nowrap="nowrap"><a href="to_be_filled">checkpoint | log</a></td>
        </tr>
        <tr>
            <td nowrap="nowrap">schnet_md17_ethanol</td>
            <td nowrap="nowrap">MD17</td>
            <td nowrap="nowrap">Energy + Force</td>
            <td nowrap="nowrap">MAE: to_be_filled</td>
            <td nowrap="nowrap">~</td>
            <td nowrap="nowrap">~</td>
            <td nowrap="nowrap"><a href="schnet_md17_ethanol.yaml">schnet_md17_ethanol.yaml</a></td>
            <td nowrap="nowrap"><a href="to_be_filled">checkpoint | log</a></td>
        </tr>
        <tr>
            <td nowrap="nowrap">schnet_iso17</td>
            <td nowrap="nowrap">ISO17</td>
            <td nowrap="nowrap">Energy regression</td>
            <td nowrap="nowrap">MAE: to_be_filled</td>
            <td nowrap="nowrap">~</td>
            <td nowrap="nowrap">~</td>
            <td nowrap="nowrap"><a href="schnet_iso17.yaml">schnet_iso17.yaml</a></td>
            <td nowrap="nowrap"><a href="to_be_filled">checkpoint | log</a></td>
        </tr>
    </body>
</table>

## Alignment Checklist

Use the following acceptance criteria during migration:

- Single-card forward alignment: logits diff around `1e-4` (generative models `1e-6`)
- Backward alignment: after at least 2 epochs, train loss trend is consistent
- Supervised task metric error: within `1%`

Quick local checks:

```bash
python test/run_schnet_torch_alignment.py
python test/run_schnet_data_metric_alignment.py
```

Dataset preparation from original/torch sources:

```bash
python test/prepare_schnet_paper_datasets.py --data_root ./data --tasks all
```

## Resource Links (Baidu)

Per PaddleMaterials delivery requirement, fill cloud links here after PR/WeChat handoff:

- Dataset package links:
  - QM9: `https://paddle-org.bj.bcebos.com/paddlematerials/datasets/qm9/qm9.tar.gz`
  - MD17: `https://paddle-org.bj.bcebos.com/paddlematerials/datasets/MD17/md17.tar.gz`
  - ISO17: `https://paddle-org.bj.bcebos.com/paddlematerials/datasets/ISO17/iso17.tar.gz`
- Pretrained model link (Baidu): `to_be_filled`
- Training/inference raw log link (Baidu): `to_be_filled`

### Training

```bash
# multi-gpu training (QM9)
python -m paddle.distributed.launch --gpus="0,1,2,3" interatomic_potentials/train.py -c interatomic_potentials/configs/schnet/schnet_qm9_lumo.yaml

# single-gpu training (QM9)
python interatomic_potentials/train.py -c interatomic_potentials/configs/schnet/schnet_qm9_lumo.yaml

# MD17 (ethanol)
python interatomic_potentials/train.py -c interatomic_potentials/configs/schnet/schnet_md17_ethanol.yaml

# ISO17
python interatomic_potentials/train.py -c interatomic_potentials/configs/schnet/schnet_iso17.yaml
```

### Validation

```bash
python interatomic_potentials/train.py -c interatomic_potentials/configs/schnet/schnet_qm9_lumo.yaml Global.do_eval=True Global.do_train=False Global.do_test=False Trainer.pretrained_model_path='your checkpoint path(*.pdparams)'
```

### Testing

```bash
python interatomic_potentials/train.py -c interatomic_potentials/configs/schnet/schnet_qm9_lumo.yaml Global.do_test=True Global.do_train=False Global.do_eval=False Trainer.pretrained_model_path='your checkpoint path(*.pdparams)'
```

### Prediction

```bash
# Mode 1: load from local config/checkpoint
python interatomic_potentials/predict.py --config_path='interatomic_potentials/configs/schnet/schnet_qm9_lumo.yaml' --checkpoint_path='your checkpoint path(*.pdparams)' --cif_file_path='./interatomic_potentials/example_data/cifs/'
```

## Citation

```
@inproceedings{schutt2017schnet,
  title={SchNet: A continuous-filter convolutional neural network for modeling quantum interactions},
  author={Sch{\"u}tt, Kristof T and Kindermans, Pieter-Jan and Sauceda, Felix A and Chmiela, Stefan and Tkatchenko, Alexandre and M{\"u}ller, Klaus-Robert},
  booktitle={Advances in Neural Information Processing Systems},
  volume={30},
  year={2017}
}
```
