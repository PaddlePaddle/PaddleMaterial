# SFIN

[Noise Calibration and Spatial-Frequency Interactive Network for STEM Image Enhancement](https://arxiv.org/pdf/2504.02555)

## Abstract

SFIN is a CNN-based model for STEM image restoration. It targets paired reconstruction from noisy grayscale inputs and supports two experimental modes (HAADF and BF), each with two training targets (`enhance`, `detect`).

---

## Model Description

SFIN takes `noisy` as input and predicts one target image (`gt_enhance` or `gt_detect`).

- Noise calibration module for robust denoising.
- Spatial-frequency interaction blocks for detail recovery.
- Multi-scale feature extraction.

Training objective (L1):

$$
\mathcal{L} = \left\| \hat{I} - I_{gt} \right\|_1
$$

Evaluation metrics:
- PSNR
- SSIM

---

## Configurations

| Config | Mode | Target | Output Dir |
| --- | --- | --- | --- |
| `sfin_tem_enhance.yaml` | HAADF (TEM) | `gt_enhance` | `./output/sfin_tem_enhance` |
| `sfin_tem_detect.yaml` | HAADF (TEM) | `gt_detect` | `./output/sfin_tem_detect` |
| `sfin_bf_enhance.yaml` | BF | `gt_enhance` | `./output/sfin_bf_enhance` |
| `sfin_bf_detect.yaml` | BF | `gt_detect` | `./output/sfin_bf_detect` |

Key training settings are shared across the four configs unless noted:

| Setting | Value |
| --- | --- |
| Optimizer | Adam |
| Learning rate | `2.0e-4` |
| LR scheduler | MultiStepDecay, milestones `[250, 400, 425, 450, 475]`, gamma `0.5` |
| Batch size | `8` for training, `1` for validation/test |
| Loss | L1 |
| Metric | PSNR |
| Epochs | `500` |

---

## Dataset Description

SFIN uses the HAADF and BF STEM image datasets released with the reference
implementation. The original format contains paired noisy inputs and ground
truth targets. PaddleMaterials loads the same paired directory structure through
`STEMImageDataset`. The released data provide train and test splits; the SFIN
configs use the released test split for both validation and testing.

### Format

Expected paired directory format:

```text
data/                      # HAADF train
  noisy/
  gt_enhance/
  gt_detect/

data_test/                 # HAADF val/test
  noisy/
  gt_enhance/
  gt_detect/

bf_data/                   # BF train
  noisy/
  gt_enhance/
  gt_detect/

bf_data_test/              # BF val/test
  noisy/
  gt_enhance/
  gt_detect/
```

### Download Links

| Dataset | Train | Test | Link |
| :---: | :---: | :---: | :---: |
| HAADF train dataset | 1000 | - | [haadf_data.zip](https://paddle-org.bj.bcebos.com/paddlematerials/datasets/SFIN_datasets/haadf_data.zip) |
| BF train dataset | 1000 | - | [bf_data.zip](https://paddle-org.bj.bcebos.com/paddlematerials/datasets/SFIN_datasets/bf_data.zip) |
| HAADF test dataset | - | 100 | [haadf_data_test.zip](https://paddle-org.bj.bcebos.com/paddlematerials/datasets/SFIN_datasets/haadf_data_test.zip) |
| BF test dataset | - | 100 | [bf_data_test.zip](https://paddle-org.bj.bcebos.com/paddlematerials/datasets/SFIN_datasets/bf_data_test.zip) |

### Auto Download Behavior

`STEMImageDataset` supports local loading + auto download:

- If `data_path` exists locally, data is loaded directly.
- If `data_path` is missing and `download=True`, URL is inferred by `data_path` basename:
  - `data` -> `haadf_data.zip`
  - `data_test` -> `haadf_data_test.zip`
  - `bf_data` -> `bf_data.zip`
  - `bf_data_test` -> `bf_data_test.zip`
- Archive formats `zip` and `tar.gz` are both supported.

---

## Environment

Run from the PaddleMaterials root directory:

```bash
pip install -r requirements.txt
pip install -e . --no-build-isolation
```

The implementation is expected to run with PaddlePaddle official release
packages. The submitted training logs were generated on one V100-32GB GPU.

---

## Results

The table tracks the four supported SFIN configurations. Unless otherwise
specified, metrics use the Torch/Paddle alignment protocol below:

```text
raw tensor -> global PSNR / SSIM
```

<table>
    <thead>
        <tr>
            <th nowrap="nowrap">Model Name</th>
            <th nowrap="nowrap">Dataset</th>
            <th nowrap="nowrap">Target</th>
            <th nowrap="nowrap">Paddle PSNR / SSIM</th>
            <th nowrap="nowrap">Torch PSNR / SSIM</th>
            <th nowrap="nowrap">Status</th>
            <th nowrap="nowrap">GPUs</th>
            <th nowrap="nowrap">Training time</th>
            <th nowrap="nowrap">Config</th>
            <th nowrap="nowrap">Checkpoint | Log</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td nowrap="nowrap">sfin_tem_enhance</td>
            <td nowrap="nowrap">HAADF test</td>
            <td nowrap="nowrap">gt_enhance</td>
            <td nowrap="nowrap">37.440395 / 0.967452</td>
            <td nowrap="nowrap">37.085558 / 0.958724</td>
            <td nowrap="nowrap">aligned, PSNR rel. diff 0.957%</td>
            <td nowrap="nowrap">1 (V100-32GB)</td>
            <td nowrap="nowrap">~21.5 hours</td>
            <td nowrap="nowrap"><a href="sfin_tem_enhance.yaml">sfin_tem_enhance</a></td>
            <td nowrap="nowrap"><a href="https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/spectrum_enhancement/sfin/sfin_he_500.zip">checkpoint | log</a></td>
        </tr>
        <tr>
            <td nowrap="nowrap">sfin_tem_detect</td>
            <td nowrap="nowrap">HAADF test</td>
            <td nowrap="nowrap">gt_detect</td>
            <td nowrap="nowrap">26.013702 / 0.964492</td>
            <td nowrap="nowrap">25.919255 / 0.963871</td>
            <td nowrap="nowrap">aligned, PSNR rel. diff 0.364%</td>
            <td nowrap="nowrap">1 (V100-32GB)</td>
            <td nowrap="nowrap">~21.2 hours</td>
            <td nowrap="nowrap"><a href="sfin_tem_detect.yaml">sfin_tem_detect</a></td>
            <td nowrap="nowrap"><a href="https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/spectrum_enhancement/sfin/sfin_hd.pdparams">checkpoint</a> | <a href="https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/spectrum_enhancement/sfin/sfin_hd_run.log">log</a></td>
        </tr>
        <tr>
            <td nowrap="nowrap">sfin_bf_enhance</td>
            <td nowrap="nowrap">BF test</td>
            <td nowrap="nowrap">gt_enhance</td>
            <td nowrap="nowrap">31.499178 / pending SSIM</td>
            <td nowrap="nowrap">31.507519 / 0.989180</td>
            <td nowrap="nowrap">epoch 294 selected, pending full checkpoint verification</td>
            <td nowrap="nowrap">1 (V100-32GB)</td>
            <td nowrap="nowrap">pending</td>
            <td nowrap="nowrap"><a href="sfin_bf_enhance.yaml">sfin_bf_enhance</a></td>
            <td nowrap="nowrap">pending BCE upload</td>
        </tr>
        <tr>
            <td nowrap="nowrap">sfin_bf_detect</td>
            <td nowrap="nowrap">BF test</td>
            <td nowrap="nowrap">gt_detect</td>
            <td nowrap="nowrap">23.826540 / 0.943309</td>
            <td nowrap="nowrap">23.820280 / 0.943306</td>
            <td nowrap="nowrap">aligned, PSNR rel. diff 0.026%</td>
            <td nowrap="nowrap">1 (V100-32GB)</td>
            <td nowrap="nowrap">~21.3 hours</td>
            <td nowrap="nowrap"><a href="sfin_bf_detect.yaml">sfin_bf_detect</a></td>
            <td nowrap="nowrap"><a href="https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/spectrum_enhancement/sfin/sfin_bd_epoch_500.pdparams">checkpoint</a> | <a href="https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/spectrum_enhancement/sfin/sfin_bd_run.log">log</a></td>
        </tr>
    </tbody>
</table>

For HAADF enhance, the original SFIN-main metric protocol
(`clip -> uint8 -> per-image average`) was also verified on 100 HAADF test
images: Paddle `38.739602 / 0.962186`, Torch `38.420514 / 0.958101`.

For HAADF detect, the original SFIN-main metric protocol was also verified on
100 HAADF test images: Paddle `28.768660 / 0.963753`, Torch
`28.671135 / 0.963961`.

For BF enhance, `epoch_500.pdparams` is not a strict alignment checkpoint. Under
the raw/global protocol, epoch 294 has PSNR `31.499178`, which is closest to
the Torch reference PSNR `31.507519`. The config therefore stops at epoch 294.
Under the original SFIN-main metric protocol, the epoch 500 Paddle checkpoint is
`34.011838 / 0.989638` and Torch is `32.295260 / 0.988141`.

Note: each config stores its default inference weight URL in
`Predict.checkpoint_path`. HAADF enhance, HAADF detect, and BF detect URLs are
available now; BF enhance needs its BCE URL updated after upload.

---

## Command

### Training

```bash
# HAADF enhance
python spectrum_enhancement/train.py \
  -c spectrum_enhancement/configs/sfin/sfin_tem_enhance.yaml

# HAADF detect
python spectrum_enhancement/train.py \
  -c spectrum_enhancement/configs/sfin/sfin_tem_detect.yaml

# BF enhance
python spectrum_enhancement/train.py \
  -c spectrum_enhancement/configs/sfin/sfin_bf_enhance.yaml

# BF detect
python spectrum_enhancement/train.py \
  -c spectrum_enhancement/configs/sfin/sfin_bf_detect.yaml
```

### Prediction

```bash
# Mode 1: use checkpoint URL from Predict.checkpoint_path.
python spectrum_enhancement/predict.py \
  --config_path spectrum_enhancement/configs/sfin/sfin_tem_enhance.yaml
```

Use the matching config for the other tasks:

| Task | Config | Default data root | Default checkpoint |
| --- | --- | --- | --- |
| HAADF enhance | `sfin_tem_enhance.yaml` | `./data_test` | available |
| HAADF detect | `sfin_tem_detect.yaml` | `./data_test` | available |
| BF enhance | `sfin_bf_enhance.yaml` | `./bf_data_test` | pending BCE upload |
| BF detect | `sfin_bf_detect.yaml` | `./bf_data_test` | available |

```bash
# Mode 2: override data/checkpoint/output paths.
python spectrum_enhancement/predict.py \
  --config_path spectrum_enhancement/configs/sfin/sfin_bf_detect.yaml \
  --checkpoint_path https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/spectrum_enhancement/sfin/sfin_bd_epoch_500.pdparams \
  --data_path ./bf_data_test \
  --output_dir ./output/sfin_predictions
```

---

## References

- Reference implementation: [HeasonLee/SFIN](https://github.com/HeasonLee/SFIN)
- Paper: [Noise Calibration and Spatial-Frequency Interactive Network for STEM Image Enhancement](https://arxiv.org/pdf/2504.02555)

---

## Citation

```bibtex
@inproceedings{li2025sfin,
  title={Noise Calibration and Spatial-Frequency Interactive Network for STEM Image Enhancement},
  author={Li, Hesong and Wu, Ziqi and Shao, Ruiwen and Zhang, Tao and Fu, Ying},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2025}
}
```
