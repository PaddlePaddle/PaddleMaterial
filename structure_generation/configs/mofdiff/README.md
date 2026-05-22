# MOFDiff

[MOFDiff: Coarse-Grained Diffusion for Metal-Organic Framework Generation](https://arxiv.org/abs/2310.10732)

## Abstract

MOFDiff is a coarse-grained diffusion model for generating Metal-Organic Frameworks (MOFs). It uses a three-stage pipeline consisting of: (1) a graph encoder with VAE bottleneck for learning latent representations, (2) a Variance Preserving (VP) diffusion process for generating building-block types and coordinates, and (3) a lattice predictor that maps latents to 6 lattice parameters.

## Model

MOFDiff operates on coarse-grained MOF representations where each building block is a single node. The encoder aggregates per-node features into a graph-level latent via a VAE bottleneck. A VP diffusion process with a GNN-based denoiser generates building-block type embeddings and 3D coordinates. A separate MLP predicts lattice parameters (3 lengths + 3 angles) from the latent. The total loss combines reconstruction, KL divergence, coordinate denoising, type denoising, lattice, and num-BBs classification terms.

## Training

```bash
# single-gpu training
python structure_generation/train.py -c structure_generation/configs/mofdiff/mofdiff_bw20k.yaml

# multi-gpu training
python -m paddle.distributed.launch --gpus="0,1,2,3" structure_generation/train.py -c structure_generation/configs/mofdiff/mofdiff_bw20k.yaml
```

## Validation

```bash
python structure_generation/train.py -c structure_generation/configs/mofdiff/mofdiff_bw20k.yaml Global.do_eval=True Global.do_train=False Global.do_test=False Trainer.pretrained_model_path='path/to/model.pdparams'
```

## Testing

```bash
python structure_generation/train.py -c structure_generation/configs/mofdiff/mofdiff_bw20k.yaml Global.do_eval=False Global.do_train=False Global.do_test=True Trainer.pretrained_model_path='path/to/model.pdparams'
```

## Citation

```
@inproceedings{yao2024mofdiff,
  title={Coarse-Grained Diffusion for Metal-Organic Framework Generation},
  author={Yao, Xiang and Mao, Nannan and Zhao, Yili and Chen, Chang and Usman, Muhammad and Tao, Dacheng},
  booktitle={International Conference on Learning Representations},
  year={2024}
}
```
