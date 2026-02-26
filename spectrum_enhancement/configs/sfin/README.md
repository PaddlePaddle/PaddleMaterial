# SFIN

[Noise Calibration and Spatial-Frequency Interactive Network for STEM Image Enhancement](https://arxiv.org/pdf/2504.02555)

## Abstract

We propose SFIN, a novel CNN-based model for STEM (Scanning Transmission Electron Microscopy) image enhancement in crystal materials microscopy. The model introduces a noise calibration mechanism and spatial-frequency interactive learning to effectively denoise and enhance grayscale STEM images. Given a noisy input image, SFIN produces an enhanced output with improved signal-to-noise ratio while preserving fine structural details critical for materials characterization.

---

## Model Description

SFIN is designed for paired image restoration tasks. The model takes a noisy grayscale STEM image as input and outputs the enhanced image. Key features include:

- **Noise Calibration**: Adaptive noise estimation and calibration module
- **Spatial-Frequency Interaction**: Joint learning in both spatial and frequency domains
- **Multi-scale Processing**: Hierarchical feature extraction across multiple scales

### Training Objective

The model is trained using L1 loss between the predicted enhanced image and the ground truth:

$$
\mathcal{L} = \left\| \hat{I}_{enhance} - I_{gt} \right\|_1
$$

### Evaluation Metrics

- **PSNR (Peak Signal-to-Noise Ratio)**: Measures reconstruction quality
- **SSIM (Structural Similarity Index)**: Measures perceptual similarity

---

## Dataset

### Format

SFIN supports four training settings (HAADF/BF × enhance/detect).  
Expected directory examples:

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

### Configurations

| Config | Mode | Target | Train Path | Val/Test Path |
| --- | --- | --- | --- | --- |
| `sfin_tem_enhance.yaml` | HAADF (TEM) | `gt_enhance` | `./data` | `./data_test` |
| `sfin_tem_detect.yaml` | HAADF (TEM) | `gt_detect` | `./data` | `./data_test` |
| `sfin_bf_enhance.yaml` | BF | `gt_enhance` | `./bf_data` | `./bf_data_test` |
| `sfin_bf_detect.yaml` | BF | `gt_detect` | `./bf_data` | `./bf_data_test` |

### Download

| Dataset | Train | Test | Link |
| :---: | :---: | :---: | :---: |
| HAADF train dataset | 1000 | - | [haadf_data.zip](https://paddle-org.bj.bcebos.com/paddlematerials/datasets/SFIN_datasets/haadf_data.zip) |
| BF train dataset | 1000 | - | [bf_data.zip](https://paddle-org.bj.bcebos.com/paddlematerials/datasets/SFIN_datasets/bf_data.zip) |
| HAADF test dataset | - | 100 | [haadf_data_test.zip](https://paddle-org.bj.bcebos.com/paddlematerials/datasets/SFIN_datasets/haadf_data_test.zip) |
| BF test dataset | - | 100 | [bf_data_test.zip](https://paddle-org.bj.bcebos.com/paddlematerials/datasets/SFIN_datasets/bf_data_test.zip) |

> Note: BF configs expect `./bf_data` and `./bf_data_test` under `PaddleMaterials` root.
> The YAML files already include dataset `url`; if local data is missing, `STEMImageDataset` will auto-download and extract (`zip` or `tar.gz`).

---

## Results

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

---

## Command

Run commands from the `PaddleMaterials` root directory.  
If `ppmat` is not installed in your environment, install once:

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

# multi GPU training (example: HAADF enhance)
python -m paddle.distributed.launch --gpus="0,1,2,3" \
  spectrum_enhancement/train.py \
  -c spectrum_enhancement/configs/sfin/sfin_tem_enhance.yaml
```

### Evaluation

```bash
# switch config file to evaluate other settings
python spectrum_enhancement/train.py \
  -c spectrum_enhancement/configs/sfin/sfin_tem_enhance.yaml \
  Global.do_train=False Global.do_eval=True Global.do_test=True \
  Trainer.pretrained_model_path='path/to/model.pdparams'
```

### Prediction

```bash
# switch config file to predict with other settings
python spectrum_enhancement/predict.py \
  --config_path spectrum_enhancement/configs/sfin/sfin_tem_enhance.yaml \
  --checkpoint_path https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/spectrum_enhancement/sfin/sfin_he_500.pdparams \
  --data_path ./data_test \
  --output_dir ./output/sfin_predictions
```

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
