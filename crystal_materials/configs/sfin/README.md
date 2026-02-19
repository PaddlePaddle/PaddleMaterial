# SFIN

[Noise Calibration and Spatial-Frequency Interactive Network for STEM Image Enhancement](https://arxiv.org/pdf/2504.02555)

## 1.Introduction

SFIN is a CNN model for STEM image enhancement in crystal materials microscopy.
Input is a noisy grayscale STEM image (`noisy`), and output is the enhanced grayscale image (`gt_enhance` target during training).

## 2.Dataset Format

The default config `sfin_tem_enhance.yaml` expects paired images under:

```text
../sfin/
  data/
    noisy/
      0.png
      ...
    gt_enhance/
      0.png
      ...
  data_test/
    noisy/
      0.png
      ...
    gt_enhance/
      0.png
      ...
  checkpoints/
    sfin_he_500.pdparams
```

Notes:
- `STEMImageDataset` uses `strict_index_naming: True` by default, so paired files should follow indexed names (such as `0.png`, `1.png`).
- `Dataset.train.dataset.__init_params__.data_path` points to `../sfin/data`.
- `Dataset.val/test.dataset.__init_params__.data_path` points to `../sfin/data_test`.

## 3.Command

Run commands from the `PaddleMaterials` root directory.  
If `ppmat` is not installed in your environment, install once:

```bash
pip install -e . --no-build-isolation
```

Single GPU training:

```bash
python crystal_materials/train.py \
  -c crystal_materials/configs/sfin/sfin_tem_enhance.yaml
```

Multi GPU training:

```bash
python -m paddle.distributed.launch --gpus="0,1,2,3" \
  crystal_materials/train.py \
  -c crystal_materials/configs/sfin/sfin_tem_enhance.yaml
```

Tiny smoke training with 2 images (for quick validation):

```bash
python crystal_materials/train.py \
  -c crystal_materials/configs/sfin/sfin_tem_enhance.yaml \
  Trainer.max_epochs=1 \
  Trainer.save_freq=1 \
  Trainer.log_freq=1 \
  Trainer.eval_freq=1 \
  Global.do_test=False \
  Dataset.train.dataset.__init_params__.data_path='../sfin/data_test' \
  Dataset.train.dataset.__init_params__.data_count=2 \
  Dataset.train.sampler.__init_params__.batch_size=1 \
  Dataset.train.sampler.__init_params__.drop_last=False \
  Dataset.val.dataset.__init_params__.data_count=2
```

Eval only with a checkpoint:

```bash
python crystal_materials/train.py \
  -c crystal_materials/configs/sfin/sfin_tem_enhance.yaml \
  Global.do_train=False Global.do_eval=True Global.do_test=False \
  Trainer.pretrained_model_path='path/to/model.pdparams'
```

Predict with provided SFIN checkpoint:

```bash
python crystal_materials/predict.py \
  --config_path crystal_materials/configs/sfin/sfin_tem_enhance.yaml \
  --checkpoint_path ../sfin/checkpoints/sfin_he_500.pdparams \
  --input_dir ../sfin/data_test/noisy \
  --output_dir ./output/sfin_tem_enhance/predictions
```

Evaluate PSNR + SSIM on prediction folder:

```bash
python crystal_materials/evaluate.py \
  --gt_dir ../sfin/data_test/gt_enhance \
  --pred_dir ./output/sfin_tem_enhance/predictions
```

## 4.Citation

```bibtex
@article{Li2025SFIN,
  title={Noise Calibration and Spatial-Frequency Interactive Network for STEM Image Enhancement},
  author={Li, Hesong and Wu, Ziqi and Shao, Ruiwen and Zhang, Tao and Fu, Ying},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2025}
}
```
