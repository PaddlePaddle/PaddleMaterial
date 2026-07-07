# MatterChat

[A conversational crystal-material AI based on BLIP-2 + Mistral-7B + CHGNet](https://arxiv.org/abs/2502.13107)

## Abstract

MatterChat is a multimodal large language model for crystal materials. It adopts the BLIP-2 architecture to bridge a crystal structure encoder (CHGNet) with a large language model (Mistral-7B-Instruct) via a Q-Former, enabling natural-language question answering about a given crystal structure. Given a CIF file and a text prompt, MatterChat answers questions such as the chemical formula, space group, stability, and band gap. It is an inference / dialogue model, not a structure generator: it does not produce new crystal structures, but reasons about an input structure through language.

---

## Model Description

### Overview
MatterChat follows the BLIP-2 three-module design, adapted from image-text to crystal-text:

```
CIF (pymatgen Structure)
  -> CHGNet encoder          : [N_atom, 64] per-atom embeddings
  -> Q-Former (BERT + cross) : [num_query_token, 768] query embeddings
  -> llm_proj (Linear)       : [num_query_token, 4096] LLM input embeddings
  -> Mistral-7B LLM          : generated text answer
```

A crystal is represented by its unit cell:
- atom types: $A = (a_1,\ldots,a_N)$
- fractional coordinates: $X = (x_1,\ldots,x_N),\; x_i \in [0,1)^3$
- lattice: $L \in \mathbb{R}^{3 \times 3}$

### Method

#### 1) CHGNet material encoder
CHGNet encodes the crystal graph (atoms, bonds, angles) into per-atom embeddings of dimension 64. The crystal graph is constructed via a radius-based neighbor search with periodic boundary conditions:

$$
\tilde{X} = X + L^\top k, \qquad k \in \mathbb{Z}^3
$$

$$
d_{ij} = \|\tilde{x}_i - x_j\|_2, \qquad \text{kept if } d_{ij} < r_{\text{cut}}
$$

The encoder is frozen during all training stages.

#### 2) Q-Former (cross-attention bridge)
A BERT-style transformer with cross-attention layers. A fixed set of learnable query tokens $Q \in \mathbb{R}^{N_q \times d}$ attend to the CHGNet embeddings $E \in \mathbb{R}^{N \times 64}$:

$$
Q' = \text{CrossAttn}(Q,\; E,\; E) = \text{softmax}\!\left(\frac{Q W_q (E W_k)^\top}{\sqrt{d_k}}\right) E W_v
$$

producing a compact material representation of shape $[N_q, 768]$ that is independent of the atom count $N$.

#### 3) Projection + Mistral LLM
A linear layer maps Q-Former outputs to the LLM hidden size (4096):

$$
Z = \text{llm\_proj}(Q') W + b, \qquad Z \in \mathbb{R}^{N_q \times 4096}
$$

The projected embeddings $Z$ are concatenated with tokenized prompt embeddings $T$ and fed to Mistral-7B-Instruct, which generates the answer autoregressively:

$$
P(y_t \mid y_{<t},\; Z,\; T) = \text{softmax}(\text{LLM}([Z;\,T;\,y_{<t}]))
$$

#### 4) Three-stage training
Following BLIP-2, MatterChat is trained in stages with progressive unfreezing:

| Stage | Trainable | Frozen |
|-------|-----------|--------|
| 1 | Q-Former | CHGNet, LLM (not yet implemented) |
| 2 | Q-Former + llm_proj | CHGNet, LLM |
| 3 | Q-Former + llm_proj + LoRA | CHGNet, LLM (LoRA adapters only) |


---

## Dataset Description

MatterChat is trained on instruction-tuning data for crystal Q&A. Each sample pairs a crystal structure (pymatgen `Structure`) with a question and its answer:

- `structure`: pymatgen `Structure` object (serialized as dict for the dataloader)
- `text_input`: the question, e.g. "what is the chemical formula of this material?"
- `text_output`: the answer, e.g. "Si"

The dataset class `MTDataset` supports loading from pickle material files + JSON question/answer files, or falls back to a small built-in demo set (Si, GaN).

---

## Results

| Model Name | Dataset | Config | Checkpoint / Log |
| --- | --- | --- | --- |
| matterchat_full | Crystal Q&A instruction data | [matterchat_full.yaml](matterchat_full.yaml) | [checkpoint / log](https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/llm/MatterChat/matterchat_full.zip) |

Dialogue test results (greedy decoding, float32):

| Crystal | Chemical Formula | Space Group | Stable | Band Gap |
| --- | --- | --- | --- | --- |
| Si | Si | Cmcm | No | 0.68795 |
| GaN | GaN | P6_3mc | No | 1.68300 |
| LiCoO2 (YZnSe2) | YZnSe2 | Fd-3m | No | 0.23871 |

---

## Command

### Inference
MatterChat is a dialogue model, not a structure generator. It does **not** use
`structure_generation/sample.py` (StructureSampler), which expects `model.sample()`
and outputs CIF files. Instead, load the model and call `chat()`.

```bash
# Mode 1: Use a pre-trained model (downloads automatically via MODEL_REGISTRY)
python -c "
from ppmat.models import build_model_from_name
from pymatgen.io.cif import CifParser

model, _ = build_model_from_name('matterchat_full')
model.to('gpu').eval()

struct = CifParser('Si.cif').get_structures()[0]
print(model.chat(struct, 'what is the chemical formula of this material?'))
# -> 'The chemical formula of this material is Si.'
"
```

```bash
# Mode 2: Use a custom config file and checkpoint
python -c "
from omegaconf import OmegaConf
from pymatgen.io.cif import CifParser
from ppmat.models import build_model
from ppmat.utils import save_load

config = OmegaConf.to_container(OmegaConf.load('structure_generation/configs/matterchat/matterchat_full.yaml'), resolve=True)
model = build_model(config['Model'])
save_load.load_pretrain(model, './matterchat_full/')
model.to('gpu').eval()

struct = CifParser('Si.cif').get_structures()[0]
print(model.chat(struct, 'what is the chemical formula of this material?'))
"
```


---

## Citation
```
@misc{matterchat2025,
  title={MatterChat: A conversational crystal-material AI based on BLIP-2.},
  author={PaddleMaterials Contributors},
  year={2025}
}
@article{li2023blip,
  title={Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models},
  author={Li, Junnan and Li, Dongxu and Savarese, Silvio and Hoi, Steven},
  journal={arXiv preprint arXiv:2301.12597},
  year={2023}
}
```
