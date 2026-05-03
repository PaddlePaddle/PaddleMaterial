# TrinityLLM

[TrinityLLM: A Molecular Language Model for Property Prediction](https://github.com/IBM/molformer)

## Abstract

TrinityLLM is a MoLFormer-based SMILES language model for polymer and molecular property prediction. It tokenizes SMILES strings using a regex-based tokenizer, encodes them through a Transformer encoder with Rotary Position Embeddings (RoPE), and predicts scalar properties via a feedforward regression head with skip connections.

## Model

TrinityLLM builds on the MoLFormer architecture. Input SMILES strings are tokenized and embedded via a learned token embedding. A stack of Transformer encoder layers with RoPE provides contextualized representations. A mean-pooled sequence representation is passed through a property prediction head (two-layer FFN with residual connections) to produce the final scalar output.

## Training

```bash
# single-gpu training
python property_prediction/train.py -c property_prediction/configs/trinityllm/trinityllm_polymer_tg.yaml

# multi-gpu training
python -m paddle.distributed.launch --gpus="0,1,2,3" property_prediction/train.py -c property_prediction/configs/trinityllm/trinityllm_polymer_tg.yaml
```

## Validation

```bash
python property_prediction/train.py -c property_prediction/configs/trinityllm/trinityllm_polymer_tg.yaml Global.do_eval=True Global.do_train=False Global.do_test=False Trainer.pretrained_model_path='your model path(*.pdparams)'
```

## Testing

```bash
python property_prediction/train.py -c property_prediction/configs/trinityllm/trinityllm_polymer_tg.yaml Global.do_test=True Global.do_train=False Global.do_eval=False Trainer.pretrained_model_path='your model path(*.pdparams)'
```

## Citation

```
@inproceedings{ross2022large,
  title={Large-scale chemical language representations capture molecular structure and properties},
  author={Ross, Jerret and Belgodere, Brian and Chenthamarakshan, Vijil and Padhi, Inkit and Mroueh, Youssef and Das, Payel},
  booktitle={Nature Machine Intelligence},
  volume={4},
  pages={1256--1264},
  year={2022}
}
```
