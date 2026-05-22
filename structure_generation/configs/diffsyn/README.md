# DiffSyn

[Generative Design of Inorganic Compounds Using a Deep Conditional Diffusion Model](https://github.com/eltonpan/zeosyn_gen)

## Abstract

DiffSyn uses a conditional 1D Gaussian diffusion process to generate zeolite synthesis conditions. A 1D U-Net denoiser with classifier-free guidance generates synthesis parameter sequences conditioned on zeolite/OSDA features. The model supports DDPM and DDIM sampling with cosine or linear noise schedules.

## Model

DiffSyn wraps a 1D U-Net in a DDPM/DDIM diffusion framework. The U-Net processes temporal sequences of synthesis parameters, using sinusoidal time embeddings, condition MLPs for classifier-free guidance, and multi-scale ResNet blocks with linear attention. During training, the model learns to denoise corrupted sequences; at inference, it progressively denoises random Gaussian noise into valid synthesis conditions.

## Training

```bash
# single-gpu training
python structure_generation/train.py -c structure_generation/configs/diffsyn/diffsyn_zeolite.yaml

# multi-gpu training
python -m paddle.distributed.launch --gpus="0,1,2,3" structure_generation/train.py -c structure_generation/configs/diffsyn/diffsyn_zeolite.yaml
```

## Validation

```bash
python structure_generation/train.py -c structure_generation/configs/diffsyn/diffsyn_zeolite.yaml Global.do_eval=True Global.do_train=False Global.do_test=False Trainer.pretrained_model_path='path/to/model.pdparams'
```

## Testing

```bash
python structure_generation/train.py -c structure_generation/configs/diffsyn/diffsyn_zeolite.yaml Global.do_eval=False Global.do_train=False Global.do_test=True Trainer.pretrained_model_path='path/to/model.pdparams'
```

## Citation

```
@article{pan2024diffsyn,
  title={Generative design of inorganic compounds using a deep conditional diffusion model},
  author={Pan, Elton and Kwon, Soonhyoung and Xie, Tian},
  year={2024}
}
```
