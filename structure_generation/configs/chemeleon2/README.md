# Chemeleon2

[Guiding Generative Models to Uncover Diverse and Novel Crystals via Reinforcement Learning](https://arxiv.org/abs/2511.07158)

## Abstract

The discovery of novel crystalline materials with targeted properties remains a central challenge in computational materials science. Generative models offer a promising route to accelerate this process, yet existing approaches often struggle to simultaneously achieve high novelty, thermodynamic stability, and compositional diversity. Here, we present **Chemeleon2**, a reinforcement learning framework built on latent diffusion models for crystal structure generation. Chemeleon2 implements a three-stage sequential pipeline—Variational Autoencoder (VAE), Latent Diffusion Model (LDM), and Reinforcement Learning (RL)—where each stage builds upon the learned representations of the previous. The RL stage employs Group Relative Policy Optimization (GRPO) with a modular, multi-objective reward system to steer generation toward desired material properties. Chemeleon2 supports de novo generation (DNG), composition-specified prediction (CSP), and text-to-structure prediction (TSP), and provides a simple Python interface for defining custom reward functions targeting arbitrary material properties such as band gap, density, or thermodynamic stability.

---

## Model Description

### Overview

Chemeleon2 represents a crystal structure by its unit cell:
- atom types: $A = (a_1, \ldots, a_N)$, where $a_i \in \{1, \ldots, 100\}$ (atomic number)
- fractional coordinates: $X = (x_1, \ldots, x_N)$, $x_i \in [0,1)^3$
- lattice parameters: lengths $(a, b, c)$ and angles $(\alpha, \beta, \gamma)$

The three-stage pipeline is strictly sequential: the VAE is trained first to learn a continuous latent space, the LDM is then trained to generate in that latent space, and finally the RL module fine-tunes the LDM denoiser using reward signals.

### Method

#### 1) Stage 1: Variational Autoencoder (VAE)

The VAE compresses crystal structures into a continuous latent space of dimension $L=8$ per atom, enabling the LDM to operate in a low-dimensional, well-structured space.

**Encoder** (`TransformerEncoder`, 8 layers, $d_\text{model}=512$, 8 heads): Atom type embeddings and fractional coordinate projections are summed and passed through transformer self-attention layers to produce per-atom representations of shape $(B_n, d_\text{model})$. A linear projection `quant_conv` maps these to mean $\mu$ and log-variance $\log\sigma^2$:

$$
(\mu, \log\sigma^2) = \text{quant\_conv}(\text{Encoder}(A, X)) \in \mathbb{R}^{B_n \times 2L}
$$

**Reparameterization**: Latent vectors are sampled via:

$$
z = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I), \quad z \in \mathbb{R}^{B_n \times L}
$$

**Decoder** (`TransformerDecoder`, 8 layers, $d_\text{model}=512$, 8 heads): The latent $z$ is projected back to $d_\text{model}$ via `post_quant_conv` and decoded into four prediction heads: atom type logits, lattice lengths $(a,b,c)$, lattice angles $(\alpha,\beta,\gamma)$, and fractional coordinates $(x,y,z)$.

**Training objective**:

$$
\mathcal{L}_\text{VAE} = \lambda_A \mathcal{L}_\text{CE}(A, \hat{A}) + \lambda_L \mathcal{L}_\text{MSE}(L, \hat{L}) + \lambda_\alpha \mathcal{L}_\text{MSE}(\alpha, \hat{\alpha}) + \lambda_X \mathcal{L}_\text{MSE}(X, \hat{X}) + \lambda_\text{KL} \cdot \text{KL}(q(z|x) \| \mathcal{N}(0,I))
$$

where $\lambda_\text{KL} = 10^{-5}$ is kept small to prevent posterior collapse. The VAE is frozen (parameters fixed, gradients disabled) for all subsequent stages.

#### 2) Stage 2: Latent Diffusion Model (LDM)

The LDM learns to generate crystal structures by performing Gaussian diffusion entirely within the VAE's latent space, avoiding the complexity of diffusing directly over discrete atom types and periodic coordinates.

**Forward process**: Latent vectors $z_0 \in \mathbb{R}^{B_n \times L}$ are reshaped to dense batches $(B, N, L)$ with a padding mask, then corrupted over $T=1000$ timesteps with a linear noise schedule:

$$
q(z_t | z_0) = \mathcal{N}\!\left(z_t;\, \sqrt{\bar{\alpha}_t}\, z_0,\, (1 - \bar{\alpha}_t) I\right)
$$

**Denoiser** (`DiT`, Diffusion Transformer): A Vision Transformer-based architecture with `hidden_size=768`, `depth=12`, `num_heads=12` processes the noisy dense latent $(B, N, L)$ with timestep $t$ and optional condition $y$, using Adaptive LayerNorm (AdaLN) conditioning and masked self-attention to handle variable-length structures:

$$
\epsilon_\theta = \text{DiT}(z_t,\, t,\, \text{mask},\, y)
$$

**Training objective** (simple diffusion loss):

$$
\mathcal{L}_\text{LDM} = \mathbb{E}_{z_0, \epsilon, t}\!\left[\|\epsilon - \epsilon_\theta(z_t, t, \text{mask}, y)\|^2\right]
$$

**Sampling**: Both DDPM and DDIM samplers are supported. DDIM with 50 steps is the default for efficient generation. The final latent $z_0$ is decoded by the frozen VAE decoder to recover the crystal structure.

**Conditional generation**: A `ConditionModule` embeds composition (CSP) or scalar property (TSP) conditions into a vector $y \in \mathbb{R}^{L_y}$. Classifier-Free Guidance (CFG) is applied at sampling time:

$$
\epsilon_\text{cfg} = \epsilon_\theta(\cdot \mid \varnothing) + w \cdot \bigl(\epsilon_\theta(\cdot \mid y) - \epsilon_\theta(\cdot \mid \varnothing)\bigr)
$$

where $w$ is the guidance scale (default 2.0). LoRA (Low-Rank Adaptation) is supported for parameter-efficient fine-tuning of the DiT on labeled datasets.

#### 3) Stage 3: Reinforcement Learning (RL) with GRPO

The RL module fine-tunes the LDM denoiser using Group Relative Policy Optimization (GRPO) to maximize expected rewards from a modular reward system. The VAE and condition module remain frozen throughout.

**Policy**: The LDM denoiser $\pi_\theta$ defines a Gaussian policy over latent transitions. Log-probabilities of trajectories $\{z_T, z_{T-1}, \ldots, z_0\}$ are computed as:

$$
\log \pi_\theta(\tau) = \sum_{t=0}^{T-1} \log \mathcal{N}(z_t;\, \mu_\theta(z_{t+1}, t),\, \sigma_t^2 I)
$$

**GRPO objective**: For each batch, $G$ trajectories are sampled per composition (group). Advantages $A_t$ are computed by normalizing rewards within each group. The clipped surrogate loss is:

$$
\mathcal{L}_\text{GRPO} = -\mathbb{E}\!\left[\min\!\left(r_t(\theta)\, A_t,\; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\, A_t\right)\right] + \beta\, D_\text{KL} - \gamma\, H
$$

where $r_t(\theta) = \pi_\theta / \pi_{\theta_\text{old}}$ is the probability ratio, $\epsilon$ is the clipping parameter, $\beta$ controls KL penalty, and $\gamma$ controls entropy bonus.

**Reward system**: A modular `ReinforceReward` aggregates multiple `RewardComponent` objects with configurable weights:

| Component | Purpose |
|-----------|---------|
| `CreativityReward` | Reward unique (AMD-based) and novel structures |
| `EnergyReward` | Penalize high energy above convex hull (MACE-Torch) |
| `StructureDiversityReward` | Maximize MMD between generated and reference structure embeddings |
| `CompositionDiversityReward` | Maximize MMD between generated and reference composition embeddings |
| `PredictorReward` | Optimize toward a target property value using a trained ML predictor |

Custom reward components can be defined by subclassing `RewardComponent` and configuring them via YAML.

---

## Dataset Description

### Dataset Contents

#### 1) MP-20

MP-20 is a benchmark subset of Materials Project structures containing **up to 20 atoms per unit cell**. It is widely used for fair comparison across crystal generative models. The dataset includes property labels (band gap, energy above hull) and is split into train/val/test sets.

#### 2) Alex-MP-20

Alex-MP-20 is a larger-scale dataset combining Materials Project and Alexandria structures, filtered to **≤ 20 atoms per unit cell**. It provides a broader chemical space for pretraining and is used for the primary Chemeleon2 models. Stability is assessed using energy above hull after DFT relaxation.

#### 3) Labeled Datasets for Conditional Generation (Optional)

For CSP or property-conditioned generation, labeled datasets are required. Each sample contains $(A, X, L)$ plus a condition label $c$ such as:
- Scalar property targets (e.g., band gap, bulk modulus)
- Chemical formula constraints (CSP)
- Text descriptions (TSP)

### Data Format

Each structure sample minimally provides:
- `atom_types`: length-$N$ list of atomic numbers
- `frac_coords`: $N \times 3$ fractional coordinates in $[0,1)$
- `lengths`: lattice vector lengths $(a, b, c)$
- `angles`: lattice angles $(\alpha, \beta, \gamma)$

Optional fields include `num_atoms`, `band_gap`, `e_above_hull`, and other property labels.

---

## Results

Pre-trained model checkpoints are available via [HuggingFace Hub](https://huggingface.co/hspark1212/chemeleon2-checkpoints) and can be loaded automatically using the `${hub:...}` resolver.

| Model Name | Dataset | Stage | Config |
|---|---|---|---|
| `mp_20_vae` | MP-20 | VAE | `experiment=mp_20/vae_dng` |
| `alex_mp_20_vae` | Alex-MP-20 | VAE | `experiment=alex_mp_20/vae_dng` |
| `mp_20_ldm_base` | MP-20 | LDM | `experiment=mp_20/ldm_base` |
| `alex_mp_20_ldm_base` | Alex-MP-20 | LDM | `experiment=alex_mp_20/ldm_base` |
| `mp_20_ldm_rl` | MP-20 | RL (DNG) | `custom_reward=rl_dng` |
| `alex_mp_20_ldm_rl` | Alex-MP-20 | RL (DNG) | `custom_reward=rl_dng` |

Pre-computed benchmark structures (10,000 generated structures per model) for de novo generation are available in `benchmarks/dng/`:

| Benchmark File | Model | Dataset |
|---|---|---|
| `chemeleon2_rl_dng_mp_20.json.gz` | RL-DNG | MP-20 |
| `chemeleon2_rl_dng_alex_mp_20.json.gz` | RL-DNG | Alex-MP-20 |

Evaluation metrics (computed against MP-20 reference via `src/evaluate.py`):

| Metric | Base LDM (expected) | RL-DNG (expected) |
|---|---|---|
| Unique | 0.90 – 0.95 | 0.95 – 0.98 |
| Novel | 0.70 – 0.80 | 0.85 – 0.95 |
| Stable (`e_above_hull < 0.1 eV`) | 0.02 – 0.05 | 0.05 – 0.10 |
| Composition Validity | 0.90 – 0.95 | 0.95 – 0.98 |

---

## Command

### Training

```bash
# Stage 1: VAE training (mp20 dataset)
# multi-gpu training (example with 8 GPUs)
python -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" structure_generation/train.py -c structure_generation/configs/chemeleon2/chemeleon2_mp20_vae.yaml
# single-gpu training
python structure_generation/train.py -c structure_generation/configs/chemeleon2/chemeleon2_mp20_vae.yaml

# Stage 2: LDM training (mp20 dataset, requires pre-trained VAE)
# Before training, set vae_ckpt_path in the yaml to the VAE checkpoint path.
# multi-gpu training (example with 8 GPUs)
python -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" structure_generation/train.py -c structure_generation/configs/chemeleon2/chemeleon2_mp20_ldm.yaml
# single-gpu training
python structure_generation/train.py -c structure_generation/configs/chemeleon2/chemeleon2_mp20_ldm.yaml
```

### Validation

```bash
# Adjust program behavior on the fly using command-line parameters without modifying the configuration file directly.
# Example: --Global.do_eval=True

# VAE validation
python structure_generation/train.py -c structure_generation/configs/chemeleon2/chemeleon2_mp20_vae.yaml Global.do_eval=True Global.do_train=False Global.do_test=False Trainer.pretrained_model_path='path/to/vae_model.pdparams'

# LDM validation
python structure_generation/train.py -c structure_generation/configs/chemeleon2/chemeleon2_mp20_ldm.yaml Global.do_eval=True Global.do_train=False Global.do_test=False Trainer.pretrained_model_path='path/to/ldm_model.pdparams'
```

### Testing

```bash
# This command is used to evaluate the model's performance on the test dataset.

# VAE testing
python structure_generation/train.py -c structure_generation/configs/chemeleon2/chemeleon2_mp20_vae.yaml Global.do_eval=False Global.do_train=False Global.do_test=True Trainer.pretrained_model_path='path/to/vae_model.pdparams'

# LDM testing
python structure_generation/train.py -c structure_generation/configs/chemeleon2/chemeleon2_mp20_ldm.yaml Global.do_eval=False Global.do_train=False Global.do_test=True Trainer.pretrained_model_path='path/to/ldm_model.pdparams'
```

### Sample

```bash
# This command is used to predict the crystal structure using a trained model.
# Mode 1: Use a pre-trained model (downloads automatically via MODEL_REGISTRY).
# Mode 2: Use a custom configuration file and checkpoint.
# Results are saved to the folder specified by --save_path (default: result).

# Mode 1: Auto-download (requires local MODEL_REGISTRY entry or internet access)
python structure_generation/sample.py --model_name='chemeleon2_ldm' --weights_name='latest.pdparams' --save_path='result_chemeleon2_ldm/' --mode='by_num_atoms' --num_atoms=20

# Mode 2: Custom checkpoint
python structure_generation/sample.py --config_path='structure_generation/configs/chemeleon2/chemeleon2_mp20_sample.yaml' --checkpoint_path='./output/chemeleon2_ldm/checkpoints/latest.pdparams' --save_path='result_chemeleon2_ldm/' --mode='by_num_atoms' --num_atoms=20

# Quick forward pass test (no training data required)
python -c "
from ppmat.models import build_model_from_name
import paddle
model, config = build_model_from_name('chemeleon2_ldm')
batch = {
    'structure_array': {
        'atom_types': paddle.randint(0, 100, [640]),
        'num_atoms': paddle.full([32], 20, dtype='int64'),
        'frac_coords': paddle.rand([640, 3]),
        'lengths': paddle.rand([32, 3]) * 10 + 5,
        'angles': paddle.rand([32, 3]) * 60 + 60,
    }
}
out = model(batch)
print(f'LDM forward loss: {float(out[\"loss_dict\"][\"total_loss\"]):.4f}')
"
```

---

## Citation

```
@article{Park2025chemeleon2,
  title={Guiding Generative Models to Uncover Diverse and Novel Crystals via Reinforcement Learning},
  author={Hyunsoo Park and Aron Walsh},
  year={2025},
  url={https://arxiv.org/abs/2511.07158}
}
```