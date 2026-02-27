# SFIN

[Noise Calibration and Spatial-Frequency Interactive Network for STEM Image Enhancement](https://arxiv.org/pdf/2504.02555)

## 1.Introduction

SFIN is a CNN-based model for STEM image restoration. It targets paired reconstruction from noisy grayscale inputs and supports two experimental modes (HAADF and BF), each with two training targets (`enhance`, `detect`).

## 2.Model Description

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

## 3.Configurations

| Config | Mode | Target | Output Dir |
| --- | --- | --- | --- |
| `sfin_tem_enhance.yaml` | HAADF (TEM) | `gt_enhance` | `./output/sfin_tem_enhance` |
| `sfin_tem_detect.yaml` | HAADF (TEM) | `gt_detect` | `./output/sfin_tem_detect` |
| `sfin_bf_enhance.yaml` | BF | `gt_enhance` | `./output/sfin_bf_enhance` |
| `sfin_bf_detect.yaml` | BF | `gt_detect` | `./output/sfin_bf_detect` |

## 4.Dataset

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

## 5.Results

<table>
    <thead>
        <tr>
            <th nowrap="nowrap">Model Name</th>
            <th nowrap="nowrap">Dataset</th>
            <th nowrap="nowrap">PSNR (dB)</th>
            <th nowrap="nowrap">SSIM</th>
            <th nowrap="nowrap">GPUs</th>
            <th nowrap="nowrap">Training time</th>
            <th nowrap="nowrap">Config</th>
            <th nowrap="nowrap">Checkpoint | Log</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td nowrap="nowrap">sfin_tem_enhance</td>
            <td nowrap="nowrap">STEM Enhancement</td>
            <td nowrap="nowrap">38.74</td>
            <td nowrap="nowrap">0.9622</td>
            <td nowrap="nowrap">1 (V100-32GB)</td>
            <td nowrap="nowrap">~21.5 hours</td>
            <td nowrap="nowrap"><a href="sfin_tem_enhance.yaml">sfin_tem_enhance</a></td>
            <td nowrap="nowrap"><a href="https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/spectrum_enhancement/sfin/sfin_he_500.zip">checkpoint | log</a></td>
        </tr>
    </tbody>
</table>

## 6.Command

Run from `PaddleMaterials` root:

```bash
pip install -e . --no-build-isolation
```

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

### Evaluation

```bash
python spectrum_enhancement/train.py \
  -c spectrum_enhancement/configs/sfin/sfin_tem_enhance.yaml \
  Global.do_train=False Global.do_eval=True Global.do_test=True \
  Trainer.pretrained_model_path='path/to/model.pdparams'
```

### Prediction

```bash
python spectrum_enhancement/predict.py \
  --config_path spectrum_enhancement/configs/sfin/sfin_tem_enhance.yaml \
  --checkpoint_path https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/spectrum_enhancement/sfin/sfin_he_500.pdparams \
  --data_path ./data_test \
  --output_dir ./output/sfin_predictions
```

## 7.Citation

```bibtex
@inproceedings{li2025sfin,
  title={Noise Calibration and Spatial-Frequency Interactive Network for STEM Image Enhancement},
  author={Li, Hesong and Wu, Ziqi and Shao, Ruiwen and Zhang, Tao and Fu, Ying},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2025}
}
```
