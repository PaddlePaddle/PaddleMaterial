# wD-MPNN

[A graph representation of molecular ensembles for polymer property prediction](https://doi.org/10.1039/D2SC02839E)

## Abstract

Weighted Directed Message Passing Neural Network (wD-MPNN) is a graph neural network that operates on molecular graphs where atoms are nodes and bonds are edges. It uses directed message passing along bonds to learn molecular representations and predict scalar molecular properties. The architecture consists of an MPNEncoder for graph-level embedding followed by a feed-forward network for regression.

## Model

wD-MPNN encodes molecules by passing directed messages along bonds in a molecular graph. Each message-passing step aggregates neighbor information weighted by learned edge features, producing an atom-level representation that is then pooled (mean/sum/norm) into a fixed-size molecular fingerprint. A multi-layer FFN maps this fingerprint to the target property.

## Training

```bash
# single-gpu training
python property_prediction/train.py -c property_prediction/configs/wd_mpnn/wd_mpnn_qm9_homo.yaml

# multi-gpu training
python -m paddle.distributed.launch --gpus="0,1,2,3" property_prediction/train.py -c property_prediction/configs/wd_mpnn/wd_mpnn_qm9_homo.yaml
```

## Validation

```bash
python property_prediction/train.py -c property_prediction/configs/wd_mpnn/wd_mpnn_qm9_homo.yaml Global.do_eval=True Global.do_train=False Global.do_test=False Trainer.pretrained_model_path='your model path(*.pdparams)'
```

## Testing

```bash
python property_prediction/train.py -c property_prediction/configs/wd_mpnn/wd_mpnn_qm9_homo.yaml Global.do_test=True Global.do_train=False Global.do_eval=False Trainer.pretrained_model_path='your model path(*.pdparams)'
```

## Citation

```
@article{aldeghi2022graph,
  title={A graph representation of molecular ensembles for polymer property prediction},
  author={Aldeghi, Matteo and Coley, Connor W.},
  journal={Chemical Science},
  volume={13},
  number={35},
  pages={10486--10498},
  year={2022},
  publisher={Royal Society of Chemistry},
  doi={10.1039/D2SC02839E}
}
```
