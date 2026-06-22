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
implementation. Each sample contains a noisy grayscale input and one
ground-truth target image for the configured task. The submitted configs use
the released training split for optimization, the released test split for
validation, and keep testing as an explicit standalone step.

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
`ppmat.utils.download` utilities. When `path` is missing and
`auto_download=True`, the URL is inferred from the `path` basename:
`data`, `data_test`, `bf_data`, or `bf_data_test`.

## Model

SFIN uses an explicit PaddleMaterials data contract:

- dataset input key: config-controlled by `input_name` (default: `noisy`)
- dataset target key: config-controlled by `target_name`
  (`gt_enhance` or `gt_detect`)
- model prediction key: same as configured target key

The Paddle implementation follows the common trainer interface and returns
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
| [sfin_bf_enhance.yaml](sfin_bf_enhance.yaml) | BF | `gt_enhance` | 434 | `./output/sfin_bf_enhance` |
| [sfin_bf_detect.yaml](sfin_bf_detect.yaml) | BF | `gt_detect` | 500 | `./output/sfin_bf_detect` |

| Setting | Value |
| :---: | :---: |
| Optimizer | Adam |
| Learning rate | `2.0e-4` |
| LR scheduler | MultiStepDecay, milestones `[250, 400, 425, 450, 475]`, gamma `0.5` |
| Batch size | 8 for training, 1 for validation/test |
| Loss | L1 |
| Metric | PSNR, SSIM |

## Metric

The configs report PSNR and SSIM using the PaddleMaterials raw/global protocol.
Predictions and targets are evaluated as raw tensors with value range
`[0, 255]`. For `N` images with shape `C x H x W`, the global mean squared
error is computed over all pixels:

```math
\mathrm{MSE}_{global}
= \frac{1}{NCHW}
\sum_{n=1}^{N}\sum_{c=1}^{C}\sum_{h=1}^{H}\sum_{w=1}^{W}
\left(\hat{I}_{nchw} - I_{nchw}\right)^2
```

```math
\mathrm{PSNR}_{global}
= 10\log_{10}
\left(
\frac{L^2}{\max(\mathrm{MSE}_{global}, \epsilon)}
\right),
\quad L=255,\ \epsilon=10^{-12}
```

SSIM is computed on raw tensors using an `11 x 11` Gaussian window with
`sigma=1.5`:

```math
\mathrm{SSIM}(x,y)
=
\frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}
{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}
```

where `C1=(0.01L)^2`, `C2=(0.03L)^2`, and `L=255`. The reported SSIM is the
mean value of the SSIM map over all evaluated images.

## Results

<table>
    <head>
        <tr>
            <th nowrap="nowrap">Model Name</th>
            <th nowrap="nowrap">Dataset</th>
            <th nowrap="nowrap">Target</th>
            <th nowrap="nowrap">PSNR / SSIM(Test dataset)</th>
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
            <td nowrap="nowrap">1 (V100-32GB)</td>
            <td nowrap="nowrap">~21.2 hours</td>
            <td nowrap="nowrap"><a href="sfin_tem_detect.yaml">sfin_tem_detect</a></td>
            <td nowrap="nowrap"><a href="https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/spectrum_enhancement/sfin/sfin_hd.pdparams">checkpoint</a> | <a href="https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/spectrum_enhancement/sfin/sfin_hd_run.log">log</a></td>
        </tr>
        <tr>
            <td nowrap="nowrap">sfin_bf_enhance</td>
            <td nowrap="nowrap">BF test</td>
            <td nowrap="nowrap">gt_enhance</td>
            <td nowrap="nowrap">31.339841 / 0.992708</td>
            <td nowrap="nowrap">1 (V100-32GB)</td>
            <td nowrap="nowrap">~19.1 hours</td>
            <td nowrap="nowrap"><a href="sfin_bf_enhance.yaml">sfin_bf_enhance</a></td>
            <td nowrap="nowrap"><a href="https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/spectrum_enhancement/sfin/sfin_be.pdparams">checkpoint</a> | <a href="https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/spectrum_enhancement/sfin/sfin_be_run.log">log</a></td>
        </tr>
        <tr>
            <td nowrap="nowrap">sfin_bf_detect</td>
            <td nowrap="nowrap">BF test</td>
            <td nowrap="nowrap">gt_detect</td>
            <td nowrap="nowrap">23.826540 / 0.943309</td>
            <td nowrap="nowrap">1 (V100-32GB)</td>
            <td nowrap="nowrap">~21.3 hours</td>
            <td nowrap="nowrap"><a href="sfin_bf_detect.yaml">sfin_bf_detect</a></td>
            <td nowrap="nowrap"><a href="https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/spectrum_enhancement/sfin/sfin_bd_epoch_500.pdparams">checkpoint</a> | <a href="https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/spectrum_enhancement/sfin/sfin_bd_run.log">log</a></td>
        </tr>
    </body>
</table>

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
# Mode 1: use checkpoint URL from Predict.checkpoint_path and build the
# Dataset.test dataloader from the config.
python spectrum_enhancement/predict.py \
  --config_path spectrum_enhancement/configs/sfin/sfin_tem_enhance.yaml
```

Or override the checkpoint and data path explicitly:

```bash
# Mode 2: custom config + checkpoint + local noisy-image directory.
python spectrum_enhancement/predict.py \
  --config_path spectrum_enhancement/configs/sfin/sfin_bf_detect.yaml \
  --checkpoint_path https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/spectrum_enhancement/sfin/sfin_bd_epoch_500.pdparams \
  --input_path ./bf_data_test/noisy \
  --split test \
  --output_dir ./output/sfin_predictions
```

When `--input_path` is provided, prediction only reads noisy input images and
does not require the target sub-directory to exist. Without `--input_path`,
prediction uses the configured `Dataset.<split>` branch through the common
dataset factory/dataloader flow.

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
