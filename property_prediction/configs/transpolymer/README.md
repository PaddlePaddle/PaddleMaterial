# TransPolymer for Polymer Property Prediction

## Overview

TransPolymer is a RoBERTa-style language model for polymer sequence representation
learning. This configuration finetunes a converted Paddle checkpoint on the PE-I
polymer electrolyte conductivity benchmark.

Reference implementation and paper:

- Paper: TransPolymer: a Transformer-based language model for polymer property predictions
- Original implementation: https://github.com/ChangwenXu98/TransPolymer

## Dataset

The PE-I benchmark predicts `Conductivity [S/cm]`.

Dataset files and converted Paddle checkpoint are packed together:

- Download link: https://pan.baidu.com/s/1wPVmTP1H0x3qSZi8HV0r9g
- Extraction code: `1227`

Archive layout:

```text
transpolymer_artifacts/data/train_PE_I.csv
transpolymer_artifacts/data/test_PE_I.csv
transpolymer_artifacts/data/vocab/vocab_sup_PE_I.csv
transpolymer_artifacts/ckpt/pretrain.pt/config.json
transpolymer_artifacts/ckpt/pretrain.pt/model_state.pdparams
```

Place or copy the extracted files under the PaddleMaterials root as:

```text
data/train_PE_I.csv
data/test_PE_I.csv
data/vocab/vocab_sup_PE_I.csv
ckpt/pretrain.pt/config.json
ckpt/pretrain.pt/model_state.pdparams
```

The train/test split follows the original TransPolymer repository. The training
split contains augmented PE-I training entries, while `test_PE_I.csv` is the
held-out PE-I test split. The supplementary vocabulary file adds PE-I-specific
tokens before finetuning.

## Pretrained Checkpoint

The archive above already contains the converted Paddle checkpoint. If starting
from the original PyTorch checkpoint, it can be converted with:

```bash
python tools/convert_torch_ckpt_to_paddle.py \
  --torch_model ../TransPolymer-pytorch/ckpt/pretrain.pt/pytorch_model.bin \
  --torch_config ../TransPolymer-pytorch/ckpt/pretrain.pt/config.json \
  --output_dir ckpt/pretrain.pt
```

The resulting directory should contain:

```text
ckpt/pretrain.pt/config.json
ckpt/pretrain.pt/model_state.pdparams
```

## Environment

Verified environment:

```text
Python 3.10
paddlepaddle-gpu 3.3.1, CUDA 11.8 wheel
RTX 4090, NVIDIA driver 550.142
```

## Training

Run from PaddleMaterials root:

```bash
python property_prediction/transpolymer_train.py \
  -c property_prediction/configs/transpolymer/transpolymer_pe_i_finetune.yaml
```

## Configuration

Key options are defined in `transpolymer_pe_i_finetune.yaml`:

- `Tokenizer.blocksize`: maximum sequence length after tokenization.
- `Tokenizer.vocab_sup_file`: supplementary PE-I vocabulary file.
- `Model.pretrained_model_path`: converted Paddle checkpoint directory.
- `Optimizer.lr_rate`: encoder learning rate.
- `Optimizer.lr_rate_reg`: regression head learning rate.
- `Trainer.max_epochs`: maximum finetuning epochs.

## Reference Results

Validated local Paddle result after tokenizer and optimizer fixes:

```text
PE-I test RMSE: 0.8993
PE-I test R2:   0.4508
```

Local PyTorch baseline in the same server environment:

```text
PE-I test RMSE: 0.9813
PE-I test R2:   0.3461
```

Paper reference result:

```text
PE-I test RMSE: approximately 0.67
PE-I test R2:   approximately 0.69
```

The remaining gap to the paper result should be documented in the PR if exact
paper-level reproduction is not achieved.

## Notes

This contribution includes a transitional task-specific training entry because
PaddleMaterials' current generic property prediction trainer is graph-model
oriented, while TransPolymer consumes tokenized polymer strings. The model and
dataset are still placed under `ppmat/models` and `ppmat/datasets` so they can
be further integrated into the unified trainer later.
