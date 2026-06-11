# SFIN

[Noise Calibration and Spatial-Frequency Interactive Network for STEM Image Enhancement](https://arxiv.org/pdf/2504.02555)

## Abstract

Scanning transmission electron microscopy (STEM) images often suffer from
severe noise and missing structural details under low-dose acquisition.
SFIN introduces a noise calibration and spatial-frequency interaction network
for paired STEM image restoration. PaddleMaterials provides four SFIN configs
covering HAADF and BF inputs, with `gt_enhance` and `gt_detect` as the two
supervised targets.

## Datasets:

SFIN uses the paired HAADF and BF datasets released with the reference
implementation. Each sample contains a noisy grayscale input and one or more
ground-truth target images. The released test split is used for both validation
and testing in the submitted configs.

Expected directory structure:

```text
data/                      # HAADF train
  noisy/
  gt_enhance/
  gt_detect/

data_test/                 # HAADF validation/test
  noisy/
  gt_enhance/
  gt_detect/

bf_data/                   # BF train
  noisy/
  gt_enhance/
  gt_detect/

bf_data_test/              # BF validation/test
  noisy/
  gt_enhance/
  gt_detect/
```

| Dataset | Train | Val/Test | Link |
| :---: | :---: | :---: | :---: |
| HAADF train | 1000 | - | [haadf_data.zip](https://paddle-org.bj.bcebos.com/paddlematerials/datasets/SFIN_datasets/haadf_data.zip) |
| HAADF test | - | 100 | [haadf_data_test.zip](https://paddle-org.bj.bcebos.com/paddlematerials/datasets/SFIN_datasets/haadf_data_test.zip) |
| BF train | 1000 | - | [bf_data.zip](https://paddle-org.bj.bcebos.com/paddlematerials/datasets/SFIN_datasets/bf_data.zip) |
| BF test | - | 100 | [bf_data_test.zip](https://paddle-org.bj.bcebos.com/paddlematerials/datasets/SFIN_datasets/bf_data_test.zip) |

`STEMImageDataset` supports automatic download through the common
`ppmat.utils.download` utilities. When `data_path` is missing and
`download=True`, the URL is inferred from the `data_path` basename:
`data`, `data_test`, `bf_data`, or `bf_data_test`.

## Model

SFIN takes `noisy` as input and predicts one configured target image
(`gt_enhance` or `gt_detect`). The Paddle implementation keeps the model
interface compatible with the PaddleMaterials trainer by returning
`loss_dict` and `pred_dict` during training/evaluation.

Training objective:

```math
\mathcal{L} = \left\| \hat{I} - I_{gt} \right\|_1
```

The four configs share the same network and optimizer settings unless noted:

| Config | Mode | Target | Epochs | Output Dir |
| :---: | :---: | :---: | :---: | :---: |
| [sfin_tem_enhance.yaml](sfin_tem_enhance.yaml) | HAADF | `gt_enhance` | 500 | `./output/sfin_tem_enhance` |
| [sfin_tem_detect.yaml](sfin_tem_detect.yaml) | HAADF | `gt_detect` | 500 | `./output/sfin_tem_detect` |
| [sfin_bf_enhance.yaml](sfin_bf_enhance.yaml) | BF | `gt_enhance` | 294 | `./output/sfin_bf_enhance` |
| [sfin_bf_detect.yaml](sfin_bf_detect.yaml) | BF | `gt_detect` | 500 | `./output/sfin_bf_detect` |

| Setting | Value |
| :---: | :---: |
| Optimizer | Adam |
| Learning rate | `2.0e-4` |
| LR scheduler | MultiStepDecay, milestones `[250, 400, 425, 450, 475]`, gamma `0.5` |
| Batch size | 8 for training, 1 for validation/test |
| Loss | L1 |
| Metric | PSNR |

## Results

Unless otherwise noted, the table reports the raw/global alignment protocol:

```text
raw tensor -> global PSNR / SSIM
```

<table>
    <head>
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
    </head>
    <body>
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
            <td nowrap="nowrap">31.595643 / 0.992208</td>
            <td nowrap="nowrap">31.507519 / 0.989180</td>
            <td nowrap="nowrap">aligned, PSNR rel. diff 0.280%</td>
            <td nowrap="nowrap">1 (V100-32GB)</td>
            <td nowrap="nowrap">~12.7 hours</td>
            <td nowrap="nowrap"><a href="sfin_bf_enhance.yaml">sfin_bf_enhance</a></td>
            <td nowrap="nowrap"><a href="https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/spectrum_enhancement/sfin/sfin_be.pdparams">checkpoint</a> | <a href="https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/spectrum_enhancement/sfin/sfin_be_run.log">log</a></td>
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
    </body>
</table>

For HAADF enhance, the original SFIN-main metric protocol
(`clip -> uint8 -> per-image average`) was also verified on 100 HAADF test
images: Paddle `38.739602 / 0.962186`, Torch `38.420514 / 0.958101`.

For HAADF detect, the original SFIN-main metric protocol was also verified on
100 HAADF test images: Paddle `28.768660 / 0.963753`, Torch
`28.671135 / 0.963961`.

For BF enhance, the original SFIN-main metric protocol was also verified on 100
BF test images: Paddle `32.663727 / 0.987594`, Torch
`32.295260 / 0.988141`.

For BF detect, the original SFIN-main metric protocol was also verified on 100
BF test images: Paddle `25.816507 / 0.942792`, Torch
`25.816014 / 0.943252`.

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

### Validation

```bash
# Use Global.do_eval=True and provide a checkpoint path.
python spectrum_enhancement/train.py \
  -c spectrum_enhancement/configs/sfin/sfin_tem_enhance.yaml \
  Global.do_train=False \
  Global.do_eval=True \
  Global.do_test=False \
  Trainer.pretrained_model_path='path/to/model.pdparams'
```

### Testing

```bash
# Evaluate on the test dataset.
python spectrum_enhancement/train.py \
  -c spectrum_enhancement/configs/sfin/sfin_tem_enhance.yaml \
  Global.do_train=False \
  Global.do_eval=False \
  Global.do_test=True \
  Trainer.pretrained_model_path='path/to/model.pdparams'
```

### Prediction

Use a config whose `Predict.checkpoint_path` is available:

```bash
# Mode 1: use checkpoint URL from Predict.checkpoint_path.
python spectrum_enhancement/predict.py \
  --config_path spectrum_enhancement/configs/sfin/sfin_tem_enhance.yaml
```

Or override the checkpoint and data path explicitly:

```bash
# Mode 2: custom config + checkpoint.
python spectrum_enhancement/predict.py \
  --config_path spectrum_enhancement/configs/sfin/sfin_bf_detect.yaml \
  --checkpoint_path https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/spectrum_enhancement/sfin/sfin_bd_epoch_500.pdparams \
  --data_path ./bf_data_test \
  --output_dir ./output/sfin_predictions
```

## References

- Reference implementation: [HeasonLee/SFIN](https://github.com/HeasonLee/SFIN)
- Paper: [Noise Calibration and Spatial-Frequency Interactive Network for STEM Image Enhancement](https://arxiv.org/pdf/2504.02555)

## Citation

```bibtex
@inproceedings{li2025sfin,
  title={Noise Calibration and Spatial-Frequency Interactive Network for STEM Image Enhancement},
  author={Li, Hesong and Wu, Ziqi and Shao, Ruiwen and Zhang, Tao and Fu, Ying},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2025}
}
```
