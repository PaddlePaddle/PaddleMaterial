# GPWNO

[Gaussian Plane-Wave Neural Operator for Electron Density Estimation](https://arxiv.org/abs/2402.04278)

## Abstract

GPWNO is an equivariant neural operator for electron-density estimation. It combines atom-centered Gaussian basis functions with a plane-wave neural operator, allowing the model to capture both local atomic environments and global Fourier-space density patterns.

<div align="center">
  <img src="../../docs/gpwno_overview.png" alt="GPWNO overview" width="95%">
</div>

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

**MD17_EC.** This molecular-dynamics electron-density dataset contains six small molecules: ethanol, benzene, phenol, resorcinol, ethane, and malonaldehyde. The first four molecules use 1,000 training geometries and 500 test geometries; ethane and malonaldehyde use 2,000 and 400, respectively. Each molecule is trained independently. [Download the PaddleMaterials package](https://paddle-org.bj.bcebos.com/paddlematerials/datasets/MD17_ES/md17_es.tar.gz).

**QM9_EC.** QM9 contains 133,884 small molecules composed of C, H, O, N, and F. Its electron-density benchmark contains 123,835 training samples, 50 validation samples, and 10,000 test samples. The PaddleMaterials reader uses the metadata and split files together with the compressed charge-density files.

**MP_EC.** The Materials Project benchmark contains electron densities for inorganic crystals. After invalid files are filtered, the structures are grouped into seven crystal systems. This PR provides the cubic-system configuration. [Download densities](https://paddle-org.bj.bcebos.com/paddlematerials/datasets/MP_ES/mp_es.tar), [atom dictionary](https://paddle-org.bj.bcebos.com/paddlematerials/datasets/MP_ES/crystal.json), and [split file](https://paddle-org.bj.bcebos.com/paddlematerials/datasets/MP_ES/crystal_data_split.json).

---

## Key Configuration

The PaddleMaterials configs follow the original GPWNO hyper-parameter choices:

| Dataset | Original model profile | Main differences |
| --- | --- | --- |
| MD17_EC | `GPWNO_MD.yaml` | `num_spherical=3`, `num_fourier=40`, `padding=4`, `use_max_cell=true`, `equivariant_frame=true`, `residual=false` |
| QM9_EC | `GPWNO_QM9.yaml` | `num_spherical=4`, `num_fourier=40`, `padding=4`, `use_max_cell=true`, `equivariant_frame=true`, `residual=true` |
| MP_EC | `GPWNO_pbc.yaml` | `pbc=true`, `num_fourier=20`, `padding=0`, `use_max_cell=false`, `max_cell_size=324`, `equivariant_frame=false` |

---

## Results

The paper reports test NMAE of **2.45%** for MD benzene, **3.67%** for MD ethane, **0.73%** for QM9, and **4.32%** for cubic MP. These are the reproduction targets. The runs below used shorter or otherwise incomplete training schedules and are retained only as implementation smoke-test results; they do not constitute a reproduction of the paper metrics. New checkpoint and log links should be published only after the corresponding test command reaches the paper target with the aligned step budget.

<table>
    <thead>
        <tr>
            <th nowrap="nowrap">Model Name</th>
            <th nowrap="nowrap">Dataset</th>
            <th nowrap="nowrap">Smoke-test NMAE</th>
            <th nowrap="nowrap">Paper test NMAE</th>
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
            <td nowrap="nowrap">2.45%</td>
            <td nowrap="nowrap">1</td>
            <td nowrap="nowrap">15hour33min</td>
            <td nowrap="nowrap"><a href="../../../electronic_structure/configs/gpwno/gpwno_md17_benzene.yaml">gpwno_md17_benzene</a></td>
            <td nowrap="nowrap">TBD</td>
        </tr>
        <tr>
            <td nowrap="nowrap">gpwno_md17_ethane</td>
            <td nowrap="nowrap">MD17_EC_Ethane</td>
            <td nowrap="nowrap">4.8339% (test)</td>
            <td nowrap="nowrap">3.67%</td>
            <td nowrap="nowrap">1</td>
            <td nowrap="nowrap">52hour18min</td>
            <td nowrap="nowrap"><a href="../../../electronic_structure/configs/gpwno/gpwno_md17_ethane.yaml">gpwno_md17_ethane</a></td>
            <td nowrap="nowrap">TBD</td>
        </tr>
        <tr>
            <td nowrap="nowrap">gpwno_qm9</td>
            <td nowrap="nowrap">QM9_EC</td>
            <td nowrap="nowrap">7.4001% (test, 2 epochs)</td>
            <td nowrap="nowrap">0.73%</td>
            <td nowrap="nowrap">3</td>
            <td nowrap="nowrap">36hour33min</td>
            <td nowrap="nowrap"><a href="../../../electronic_structure/configs/gpwno/gpwno_qm9.yaml">gpwno_qm9</a></td>
            <td nowrap="nowrap">TBD</td>
        </tr>
        <tr>
            <td nowrap="nowrap">gpwno_mp</td>
            <td nowrap="nowrap">MP_EC (cubic)</td>
            <td nowrap="nowrap">37.8910%</td>
            <td nowrap="nowrap">4.32%</td>
            <td nowrap="nowrap">1</td>
            <td nowrap="nowrap">37hour46min</td>
            <td nowrap="nowrap"><a href="../../../electronic_structure/configs/gpwno/gpwno_mp.yaml">gpwno_mp</a></td>
            <td nowrap="nowrap">TBD</td>
        </tr>
    </tbody>
</table>

Note: The MD17_EC_Benzene smoke-test result is from epoch-10 validation of `gpwno_md17_benzene_t_20260527_182931_s_42`; it is not directly comparable to the paper's test result.

Note: The MD17_EC_Ethane smoke-test result is from the test set evaluation of `gpwno_md17_ethane_t_20260529_204928_s_42`. The final validation NMAE is `4.8302%`, and the final test NMAE is `4.8339%`.

Note: The QM9_EC smoke-test result is from the two-epoch test set evaluation of `gpwno_qm9_t_20260626_164201_s_42`, which resumed from `gpwno_qm9_t_20260625_221247_s_42/checkpoints/latest`. The best validation NMAE is `7.0865%`, and the final test NMAE is `7.4001%`.

Note: The MP_EC smoke-test result is from the test set evaluation of `gpwno_mp_resume_t_20260528_204842_s_42`. The final validation NMAE is `34.8928%`, and the final test NMAE is `37.8910%`.

---

## Command

### Data preparation

The datasets are downloaded automatically by the dataset classes when the configured root directory is missing and `auto_download` is enabled.

```bash
# MD17_EC will be prepared under ./data/data_md
python electronic_structure/train.py -c electronic_structure/configs/gpwno/gpwno_md17_ethane.yaml Global.do_train=False Global.do_eval=False Global.do_test=True

# QM9_EC will be prepared under ./data/data_qm9 after the dataset package is downloaded and extracted
python electronic_structure/train.py -c electronic_structure/configs/gpwno/gpwno_qm9.yaml Global.do_train=False Global.do_eval=False Global.do_test=True
```

### Training

```bash
# single-gpu training
python electronic_structure/train.py -c electronic_structure/configs/gpwno/gpwno_md17_ethane.yaml

# multi-gpu training
python -m paddle.distributed.launch --gpus="0,1,2,3" electronic_structure/train.py -c electronic_structure/configs/gpwno/gpwno_md17_benzene.yaml

# QM9_EC multi-gpu training
python -m paddle.distributed.launch --gpus="0,1,2" electronic_structure/train.py -c electronic_structure/configs/gpwno/gpwno_qm9.yaml

# resume training from checkpoint
python -m paddle.distributed.launch --gpus="0,1,2" electronic_structure/train.py -c electronic_structure/configs/gpwno/gpwno_qm9.yaml Trainer.resume_from_checkpoint='path/to/checkpoints/latest'
```

### Validation

```bash
python electronic_structure/train.py -c electronic_structure/configs/gpwno/gpwno_md17_ethane.yaml Global.do_train=False Global.do_eval=True Global.do_test=False Trainer.pretrained_model_path='path/to/model.pdparams'

python electronic_structure/train.py -c electronic_structure/configs/gpwno/gpwno_qm9.yaml Global.do_train=False Global.do_eval=True Global.do_test=False Trainer.pretrained_model_path='path/to/checkpoints/best.pdparams'
```

### Testing

```bash
python electronic_structure/train.py -c electronic_structure/configs/gpwno/gpwno_md17_ethane.yaml Global.do_train=False Global.do_eval=False Global.do_test=True Trainer.pretrained_model_path='path/to/model.pdparams'

python electronic_structure/train.py -c electronic_structure/configs/gpwno/gpwno_qm9.yaml Global.do_train=False Global.do_eval=False Global.do_test=True Trainer.pretrained_model_path='path/to/checkpoints/best.pdparams'
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
  author={Kim, Seongsu and Ahn, Sungsoo},
  booktitle={International Conference on Machine Learning},
  year={2024}
}
```
