# NewtonNet

[NewtonNet: a Newtonian message passing network for deep learning of interatomic potentials and forces](https://doi.org/10.1039/D2DD00008C)

## Abstract

NewtonNet is a Newtonian message-passing neural network for learning interatomic potentials and forces. It operates on atomic graphs with radial Bessel basis functions and polynomial cutoff envelopes. The architecture uses a sequence of interaction layers that update both atom-level and force-level representations, enabling accurate prediction of energies and atomic forces while maintaining energy conservation.

## Model

NewtonNet constructs a radius graph from atomic positions and computes pairwise radial basis features. Interaction layers iteratively refine atom representations using Newtonian message passing that preserves physical symmetries. Per-atom energies are summed for the total molecular energy, and forces are obtained as negative energy gradients with respect to positions.

## Training

```bash
# single-gpu training
python interatomic_potentials/train.py -c interatomic_potentials/configs/newtonnet/newtonnet_qm9_energy.yaml

# multi-gpu training
python -m paddle.distributed.launch --gpus="0,1,2,3" interatomic_potentials/train.py -c interatomic_potentials/configs/newtonnet/newtonnet_qm9_energy.yaml
```

## Validation

```bash
python interatomic_potentials/train.py -c interatomic_potentials/configs/newtonnet/newtonnet_qm9_energy.yaml Global.do_eval=True Global.do_train=False Global.do_test=False Trainer.pretrained_model_path='your model path(*.pdparams)'
```

## Testing

```bash
python interatomic_potentials/train.py -c interatomic_potentials/configs/newtonnet/newtonnet_qm9_energy.yaml Global.do_test=True Global.do_train=False Global.do_eval=False Trainer.pretrained_model_path='your model path(*.pdparams)'
```

## Citation

```
@article{haghighatlari2022newtonnet,
  title={NewtonNet: a Newtonian message passing network for deep learning of interatomic potentials and forces},
  author={Haghighatlari, Mojtaba and Li, Jie and Guan, Xingyi and Zhang, Oufan and Das, Akshaya and Stein, Christopher J and Heiber, Farnaz and Liu, Tiantian and Head-Gordon, Martin and Bertels, Luke and others},
  journal={Digital Discovery},
  volume={1},
  number={3},
  pages={333--343},
  year={2022},
  publisher={Royal Society of Chemistry}
}
```
