# GPWNO

[Gaussian Plane-Wave Neural Operator for Electron Density Estimation](https://arxiv.org/abs/2402.04278)

Official implementation: [seongsukim-ml/GPWNO](https://github.com/seongsukim-ml/GPWNO)

## Abstract

GPWNO is an equivariant neural operator for electron-density estimation. It combines atom-centered Gaussian basis functions with a plane-wave neural operator, allowing the model to capture both local atomic environments and global Fourier-space density patterns. The model follows the same electron-density prediction setting as InfGCN and was originally built on top of a similar data pipeline.

---

## Model Description

### Overview

Given atom types and atomic coordinates, GPWNO predicts a continuous electron-density field. The model first learns local equivariant features around atoms, then refines the field with a plane-wave operator in Fourier space.

The main components are:

- Gaussian atom-centered density basis
- SE(3)-equivariant coefficient learning
- Plane-wave neural operator refinement
- Optional periodic-boundary support for crystalline systems

### Method

#### 1) Gaussian basis expansion

The electron density is represented with atom-centered basis functions:

$$
\hat{\rho}(x) =
\sum_i \sum_{n,l,m}
c_{i,nlm}\,g_n(\|x-r_i\|)\,Y_{lm}(\widehat{x-r_i})
$$

where $g_n$ is a radial Gaussian basis and $Y_{lm}$ is a spherical harmonic basis. The learnable part is the coefficient tensor $c_{i,nlm}$.

#### 2) Equivariant local message passing

GPWNO learns basis coefficients with equivariant message passing over the atomic graph. Edge features are built from interatomic distances and angular information, preserving rotation and translation behavior needed for density prediction.

#### 3) Plane-wave neural operator

After local coefficient learning, GPWNO applies a Fourier-space operator to improve global consistency:

$$
\rho_{\mathrm{out}} = \mathcal{F}^{-1}
\left(
W(k)\,\mathcal{F}(\rho_{\mathrm{local}})
\right)
$$

The Fourier module is controlled mainly by `num_fourier`, `num_fourier_time`, `width`, `fourier_mode`, `padding`, and `using_ff`.

#### 4) Training objective and metric

The model is trained with point-wise density regression on sampled grid points. The commonly reported metric is Normalized Mean Absolute Error:

$$
\mathrm{NMAE} =
\frac{\sum_i |\hat{\rho}(x_i)-\rho(x_i)|}
{\sum_i |\rho(x_i)|}
$$

---

## Dataset Description

GPWNO uses the same electron-density datasets and dataloaders as InfGCN.

- **MD17_EC**: Small-molecule molecular dynamics electron-density data. The supported molecules include benzene, ethanol, phenol, resorcinol, ethane, and malonaldehyde. [Data](https://paddle-org.bj.bcebos.com/paddlematerials/datasets/MD17_ES/md17_es.tar.gz).
- **MP_EC (cubic)**: Materials Project-style crystal electron densities stored as `*.json.xz` under `./data/data_mp`. [Data](https://paddle-org.bj.bcebos.com/paddlematerials/datasets/MP_ES/mp_es.tar), [atom dictionary](https://paddle-org.bj.bcebos.com/paddlematerials/datasets/MP_ES/crystal.json), [split file](https://paddle-org.bj.bcebos.com/paddlematerials/datasets/MP_ES/crystal_data_split.json).

---

## Key Configuration

The PaddleMaterials configs follow the original GPWNO hyper-parameter choices:

| Dataset | Original model profile | Main differences |
| --- | --- | --- |
| MD17_EC | `GPWNO_MD.yaml` | `num_spherical=3`, `num_fourier=40`, `padding=4`, `use_max_cell=true`, `equivariant_frame=true`, `residual=false` |
| MP_EC | `GPWNO_pbc.yaml` | `pbc=true`, `num_fourier=20`, `padding=0`, `use_max_cell=false`, `max_cell_size=324`, `equivariant_frame=false` |

---

## Results

<table>
    <thead>
        <tr>
            <th nowrap="nowrap">Model Name</th>
            <th nowrap="nowrap">Dataset</th>
            <th nowrap="nowrap">Normalized MAE of Density</th>
            <th nowrap="nowrap">GPUs</th>
            <th nowrap="nowrap">Training time</th>
            <th nowrap="nowrap">Config</th>
            <th nowrap="nowrap">Checkpoint | Log</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td nowrap="nowrap">gpwno_md17_benzene</td>
            <td nowrap="nowrap">MD17_EC_Benzene</td>
            <td nowrap="nowrap">3.5567% (val, 10 epochs)</td>
            <td nowrap="nowrap">1</td>
            <td nowrap="nowrap">15hour33min</td>
            <td nowrap="nowrap"><a href="../../../electronic_structure/configs/gpwno/gpwno_md17_benzene.yaml">gpwno_md17_benzene</a></td>
            <td nowrap="nowrap">TBD</td>
        </tr>
        <tr>
            <td nowrap="nowrap">gpwno_md17_ethane</td>
            <td nowrap="nowrap">MD17_EC_Ethane</td>
            <td nowrap="nowrap">4.8339% (test)</td>
            <td nowrap="nowrap">1</td>
            <td nowrap="nowrap">52hour18min</td>
            <td nowrap="nowrap"><a href="../../../electronic_structure/configs/gpwno/gpwno_md17_ethane.yaml">gpwno_md17_ethane</a></td>
            <td nowrap="nowrap">TBD</td>
        </tr>
        <tr>
            <td nowrap="nowrap">gpwno_mp</td>
            <td nowrap="nowrap">MP_EC (cubic)</td>
            <td nowrap="nowrap">37.8910%</td>
            <td nowrap="nowrap">1</td>
            <td nowrap="nowrap">37hour46min</td>
            <td nowrap="nowrap"><a href="../../../electronic_structure/configs/gpwno/gpwno_mp.yaml">gpwno_mp</a></td>
            <td nowrap="nowrap">TBD</td>
        </tr>
    </tbody>
</table>

Note: The MD17_EC_Benzene result is reported from the epoch-10 validation of the local run `gpwno_md17_benzene_t_20260527_182931_s_42`; the corresponding checkpoint is `checkpoints/best.pdparams`. The run was intentionally recorded as a 10-epoch result.

Note: The MD17_EC_Ethane result is reported from the final test set evaluation of the local run `gpwno_md17_ethane_t_20260529_204928_s_42`. The final validation NMAE is `4.8302%`, and the final test NMAE is `4.8339%`.

Note: The MP_EC result is reported from the final test set evaluation of the local run `gpwno_mp_resume_t_20260528_204842_s_42`. The final validation NMAE is `34.8928%`, and the final test NMAE is `37.8910%`.

Note: Checkpoint and log links are reserved as `TBD`. The local checkpoints and logs are not submitted in this PR; they will be uploaded separately by the project maintainers and replaced with Paddle-hosted links after conversion.

---

## Command

### Data preparation

The datasets are downloaded automatically by the dataset classes when the configured root directory is missing and `auto_download` is enabled.

```bash
# MD17_EC will be prepared under ./data/data_md
python electronic_structure/train.py -c electronic_structure/configs/gpwno/gpwno_md17_ethane.yaml Global.do_train=False Global.do_eval=False Global.do_test=True
```

### Training

```bash
# single-gpu training
python electronic_structure/train.py -c electronic_structure/configs/gpwno/gpwno_md17_ethane.yaml

# multi-gpu training
python -m paddle.distributed.launch --gpus="0,1,2,3" electronic_structure/train.py -c electronic_structure/configs/gpwno/gpwno_md17_benzene.yaml
```

### Validation

```bash
python electronic_structure/train.py -c electronic_structure/configs/gpwno/gpwno_md17_ethane.yaml Global.do_train=False Global.do_eval=True Global.do_test=False Trainer.pretrained_model_path='path/to/model.pdparams'
```

### Testing

```bash
python electronic_structure/train.py -c electronic_structure/configs/gpwno/gpwno_md17_ethane.yaml Global.do_train=False Global.do_eval=False Global.do_test=True Trainer.pretrained_model_path='path/to/model.pdparams'
```

### Prediction

```bash
python electronic_structure/predict.py -c electronic_structure/configs/gpwno/gpwno_md17_ethane.yaml Trainer.pretrained_model_path='path/to/model.pdparams'
```

---

## Citation

```bibtex
@inproceedings{kim2024gaussian,
  title={Gaussian Plane-Wave Neural Operator for Electron Density Estimation},
  author={Kim, Seongsu and Kim, Dong Hwan and Lee, Woo Youn and Han, Seungwu},
  booktitle={International Conference on Machine Learning},
  year={2024}
}
```

```bibtex
@article{fandel2023infgcn,
  title={InfGCN: Equivariant Neural Operator Learning with Graphon Convolution},
  author={Fandel, Dorian and von Rudorff, Guido Falk and von Lilienfeld, O. Anatole},
  journal={arXiv preprint arXiv:2311.10908},
  year={2023}
}
```
