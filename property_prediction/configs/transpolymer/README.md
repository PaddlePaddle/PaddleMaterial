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

Expected files:

```text
data/train_PE_I.csv
data/test_PE_I.csv
data/vocab/vocab_sup_PE_I.csv
```

The train/test split follows the original TransPolymer repository. Dataset files
and pretrained checkpoints should be uploaded to BCE by the reviewer and the
download links should be filled in here before merge.

## Pretrained Checkpoint

The original PyTorch checkpoint can be converted with:

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
python property_prediction/train.py \
  -c property_prediction/configs/transpolymer/transpolymer_pe_i_finetune.yaml
```

## Configuration

Key options are defined in `transpolymer_pe_i_finetune.yaml`:

- `Dataset.*.dataset.__init_params__.blocksize`: maximum sequence length after tokenization.
- `Dataset.*.dataset.__init_params__.vocab_sup_file`: supplementary PE-I vocabulary file.
- `Model.__init_params__.pretrained_model_path`: converted Paddle checkpoint directory.
- `Optimizer.lr.__init_params__.learning_rate`: finetuning learning rate.
- `Trainer.max_epochs`: maximum finetuning epochs.

## Results

<table>
    <head>
        <tr>
            <th nowrap="nowrap">Model Name</th>
            <th nowrap="nowrap">Dataset</th>
            <th nowrap="nowrap">Property</th>
            <th nowrap="nowrap">RMSE / R2(Test dataset)</th>
            <th nowrap="nowrap">GPUs</th>
            <th nowrap="nowrap">Training time</th>
            <th nowrap="nowrap">Config</th>
            <th nowrap="nowrap">Checkpoint | Log</th>
        </tr>
    </head>
    <body>
        <tr>
            <td nowrap="nowrap">transpolymer_pe_i_finetune</td>
            <td nowrap="nowrap">PE-I</td>
            <td nowrap="nowrap">Conductivity [S/cm]</td>
            <td nowrap="nowrap">0.8993 / 0.4508</td>
            <td nowrap="nowrap">1</td>
            <td nowrap="nowrap">~3 hours</td>
            <td nowrap="nowrap"><a href="transpolymer_pe_i_finetune.yaml">transpolymer_pe_i_finetune</a></td>
            <td nowrap="nowrap"><a href="https://pan.baidu.com/s/1SB2KP7zYkWkBF7Z1Q1EG8w">checkpoint | log</a> (code: 1227)</td>
        </tr>
    </body>
</table>
