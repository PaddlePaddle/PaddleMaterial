# SevenNet Model Configuration

## Overview

SevenNet is a message-passing graph neural network for interatomic potential prediction.

## Pretrained Models

SevenNet provides multiple pretrained models:

| Model | Description | Training Dataset | Performance (CPS) |
|-------|-------------|------------------|-------------------|
| **SevenNet-Omni** (Recommended) | Universal potential, 15 datasets | 15 open ab initio datasets | 0.849 |
| SevenNet-Omni-i8 | Higher capacity (Nlayer=8) | 15 datasets | 0.859 |
| SevenNet-Omni-i12 | Highest capacity (Nlayer=12) | 15 datasets | 0.873 |
| SevenNet-MF-ompa | Multi-fidelity learning | MPtrj, sAlex, OMat24 | 0.845 |
| SevenNet-omat | OMat24 only | OMat24 | κSRME: 0.221 |
| SevenNet-l3i5 | MPtrj with lmax=3 | MPtrj | 0.714 |
| SevenNet-0 | Fastest inference | MPtrj | F1: 0.67 |

## Using Pretrained Models

### Step 1: Download Model

When official provides URLs, download the pretrained model:

```python
from ppmat.utils import download

model_name = "sevennet_omni"  # or other model name
model_path = download.get_weights_path_from_url(MODEL_REGISTRY[model_name])
```

### Step 2: Load Model

```python
import paddle
from ppmat.models import PurePaddleSevenNet

# Load checkpoint
state_dict = paddle.load("path/to/checkpoint.pdparams")

# Create model
model = PurePaddleSevenNet(
    num_species=100,
    hidden_dim=128,
    num_message_layers=5,
    num_rbf=32,
    cutoff=5.0,
)

# Load weights
model.set_state_dict(state_dict)
model.eval()
```

### Step 3: Predict

```python
# Create graph from structure
graph = {
    "z": paddle.to_tensor([1, 8, 1], dtype="int64"),  # H, O, H
    "pos": paddle.to_tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype="float32"),
    "edge_index": paddle.to_tensor([[0, 1], [1, 0]], dtype="int64"),
}

# Predict
result = model(graph)
energy = result["total_energy"]
```

## Training Datasets

The official training datasets include:

- **MPtrj** (Materials Project Trajectory): https://figshare.com/articles/dataset/Materials_Project_Trjectory_MPtrj_Dataset/23713842
- **OMat24**: https://huggingface.co/datasets/fairchem/OMAT24
- **sAlex**: https://huggingface.co/datasets/fairchem/OMAT24

## References

- [SevenNet Official Repository](https://github.com/MDIL-SNU/SevenNet)
- [SevenNet Documentation](https://sevennet.readthedocs.io/)
- [Pretrained Models Guide](https://sevennet.readthedocs.io/en/latest/user_guide/pretrained.html)