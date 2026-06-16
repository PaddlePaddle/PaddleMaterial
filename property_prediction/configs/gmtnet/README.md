# GMTNet for Dielectric Tensor Prediction

PaddlePaddle implementation of GMTNet (ICML 2024). Predicts dielectric tensors from crystal structures.

## Requirements
- paddlepaddle>=2.6.2
- numpy, pickle

## Dataset
Place your preprocessed data file `paddle_dielectric_data.pkl` in the same directory as the config.

## Training
```bash
python train.py

