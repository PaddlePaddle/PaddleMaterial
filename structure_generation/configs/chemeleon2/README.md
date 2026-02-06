# Chemeleon2 配置文件说明

本目录包含 Chemeleon2 模型训练和采样的配置文件。

## 配置文件列表

### 训练配置

1. train_vae.yaml - VAE模块的训练
2. train_ldm.yaml - LDM模块的训练
3. sample.yaml - 采样生成配置文件



## 使用方法

### 训练 VAE

```bash
python structure_generation/train.py \
    --config structure_generation/configs/chemeleon2/train_vae.yaml
```


### 训练 LDM

```bash
python structure_generation/train.py \
    --config structure_generation/configs/chemeleon2/train_ldm.yaml
```

## 验证VAE权重
```bash
python structure_generation/train.py \
    -c structure_generation/configs/chemeleon2/train_vae.yaml \
    Global.do_eval=False \
    Global.do_train=False \
    Global.do_test=True \
    Trainer.pretrained_model_path=output/chemeleon2_vae/checkpoints/latest.pdparams
```

## 验证LDM权重
```bash
python structure_generation/train.py \
    -c structure_generation/configs/chemeleon2/train_ldm.yaml \
    Global.do_eval=False \
    Global.do_train=False \
    Global.do_test=True \
    Trainer.pretrained_model_path=output/chemeleon2_ldm/checkpoints/latest.pdparams
```


### 验证VAE权重

```bash
python structure_generation/train.py \
    -c structure_generation/configs/chemeleon2/train_vae.yaml \
    Global.do_eval=True \
    Global.do_train=False \
    Global.do_test=False \
    Trainer.pretrained_model_path=output/chemeleon2_vae/checkpoints/latest.pdparams
```

### 验证LDM权重
```bash
python structure_generation/train.py \
    -c structure_generation/configs/chemeleon2/train_ldm.yaml \
    Global.do_eval=True \
    Global.do_train=False \
    Global.do_test=False \
    Trainer.pretrained_model_path=output/chemeleon2_ldm/checkpoints/latest.pdparams
```



### 晶体采样

```bash
# vae的模型地址在 yaml里约定
python structure_generation/sample.py \
    --config structure_generation/configs/chemeleon2/sample.yaml \
    --checkpoint_path test-for-weight/converted_weights/ldm_paddle.pdparams
```

