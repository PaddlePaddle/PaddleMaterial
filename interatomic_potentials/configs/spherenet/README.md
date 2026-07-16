# SphereNet

[Spherical Message Passing for 3D Molecular Graphs](https://arxiv.org/abs/2102.05013) (ICLR 2021)

## Abstract

We propose the spherical message passing (SMP) scheme for 3D molecular graphs,
which leverages **distance, angle, and torsion** information simultaneously to
uniquely identify the relative positions of atoms in 3D space. Previous
methods such as SchNet (distance-only) and DimeNet++ (distance + angle) suffer
from equivariance ambiguity because multiple spatial configurations can map
to the same pairwise distances or angles. By incorporating torsion angles
(dihedral angles), SphereNet resolves this ambiguity and achieves
state-of-the-art results on the QM9 and MD17 benchmarks.

<p align="center">
  <img src="../../docs/SphereNet.png" alt="SphereNet Architecture" width="80%"/>
  <br/>
  <em>Figure 1: SphereNet architecture.</em>
</p>

## Datasets

### MD17

The MD17 dataset contains DFT molecular dynamics trajectories for 8 small
organic molecules. Each configuration includes the total energy (kcal/mol) and
atomic forces (kcal/mol/Å).

| Molecule      | Train | Val  | Test | Atoms |
|---------------|------:|-----:|-----:|------:|
| Aspirin       | 1000  | 1000 | 209762 | 21    |
| Benzene       | 1000  | 1000 | 625983 | 12    |
| Ethanol       | 1000  | 1000 | 553092 | 9     |
| Malonaldehyde | 1000  | 1000 | 991237 | 9     |
| Naphthalene   | 1000  | 1000 | 324250 | 18    |
| Salicylic     | 1000  | 1000 | 318231 | 16    |
| Toluene       | 1000  | 1000 | 440790 | 15    |
| Uracil        | 1000  | 1000 | 131770 | 12    |

**Data format**: Each molecule is stored as a single `.npz` file with keys
`E` (energies), `F` (forces), `R` (positions), and `z` (atomic numbers).

The configs expect each molecule file under `./data/md17/` and split indices
under `./data/md17/splits/<molecule>_<split>_idx.npy`. For example, the
aspirin config reads `./data/md17/md17_aspirin.npz`.

Each MD17 config stores the training-split total-energy mean in the SphereNet
model with `data_std: 1.0`. The model learns centered energy, restores physical
energy internally, and differentiates the restored energy to obtain forces.
Dataset labels therefore remain in their original physical units.

## Model

SphereNet is a spherical message passing neural network for 3D molecular
graphs. It represents each molecule as a graph where nodes correspond to
atoms, and directed edges encode interatomic interactions within a cutoff
radius. The model builds a hierarchy of geometric features and propagates
information using spherical message passing.

### Geometric embedding hierarchy

SphereNet constructs three levels of geometric embeddings to capture the
full 3D structure:

**1. Radial (distance) embeddings** — For each directed edge $j \to i$, the
interatomic distance $d_{ji}$ is expanded using a radial basis function
(RBF) composed with a smooth envelope.

**2. Angular (spherical) embeddings** — For each triplet $k \to j \to i$,
the bond angle $\theta_{kji}$ is expanded together with the distance
$d_{kj}$ using spherical Bessel functions combined with Legendre
polynomials (spherical Fourier-Bessel basis).

**3. Torsional embeddings** — For each quadruplet $l \to k \to j \to i$,
the torsion (dihedral) angle $\tau_{lkji}$ together with distances $d_{lk}$
and $d_{kj}$ is expanded using a 3D spherical Fourier-Bessel basis.

## Results

> **Checkpoint notice:** The linked MD17 checkpoints were trained before
> SphereNet force gradients were enabled. They remain available for artifact
> compatibility, but must not be used for force prediction. Retrain with the
> current configs and replace the packages before reporting MLIP results.

<table>
    <head>
        <tr>
            <th nowrap="nowrap">Model Name</th>
            <th nowrap="nowrap">Dataset</th>
            <th nowrap="nowrap">Property</th>
            <th nowrap="nowrap">MAE</th>
            <th nowrap="nowrap">GPUs</th>
            <th nowrap="nowrap">Training time</th>
            <th nowrap="nowrap">Config</th>
            <th nowrap="nowrap">Checkpoint | Log</th>
        </tr>
    </head>
    <body>
        <tr>
            <td nowrap="nowrap">spherenet_md17_aspirin</td>
            <td nowrap="nowrap">MD17</td>
            <td nowrap="nowrap">Energy (kcal/mol) / Force (kcal/mol/Å)</td>
            <td nowrap="nowrap">0.26 / 0.44</td>
            <td nowrap="nowrap">1</td>
            <td nowrap="nowrap">~6 h</td>
            <td nowrap="nowrap"><a href="./spherenet_md17_aspirin.yaml">config</a></td>
            <td nowrap="nowrap"><a href="https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/interatomic_potentials/spherenet/spherenet_md17_aspirin.zip">checkpoint | log</a></td>
        </tr>
        <tr>
            <td nowrap="nowrap">spherenet_md17_benzene_old</td>
            <td nowrap="nowrap">MD17</td>
            <td nowrap="nowrap">Energy (kcal/mol) / Force (kcal/mol/Å)</td>
            <td nowrap="nowrap">0.14 / 0.21</td>
            <td nowrap="nowrap">1</td>
            <td nowrap="nowrap">~3 h</td>
            <td nowrap="nowrap"><a href="./spherenet_md17_benzene_old.yaml">config</a></td>
            <td nowrap="nowrap"><a href="https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/interatomic_potentials/spherenet/spherenet_md17_benzene_old.zip">checkpoint | log</a></td>
        </tr>
        <tr>
            <td nowrap="nowrap">spherenet_md17_ethanol</td>
            <td nowrap="nowrap">MD17</td>
            <td nowrap="nowrap">Energy (kcal/mol) / Force (kcal/mol/Å)</td>
            <td nowrap="nowrap">0.10 / 0.23</td>
            <td nowrap="nowrap">1</td>
            <td nowrap="nowrap">~2 h</td>
            <td nowrap="nowrap"><a href="./spherenet_md17_ethanol.yaml">config</a></td>
            <td nowrap="nowrap"><a href="https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/interatomic_potentials/spherenet/spherenet_md17_ethanol.zip">checkpoint | log</a></td>
        </tr>
        <tr>
            <td nowrap="nowrap">spherenet_md17_malonaldehyde</td>
            <td nowrap="nowrap">MD17</td>
            <td nowrap="nowrap">Energy (kcal/mol) / Force (kcal/mol/Å)</td>
            <td nowrap="nowrap">0.17 / 0.32</td>
            <td nowrap="nowrap">1</td>
            <td nowrap="nowrap">~2 h</td>
            <td nowrap="nowrap"><a href="./spherenet_md17_malonaldehyde.yaml">config</a></td>
            <td nowrap="nowrap"><a href="https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/interatomic_potentials/spherenet/spherenet_md17_malonaldehyde.zip">checkpoint | log</a></td>
        </tr>
        <tr>
            <td nowrap="nowrap">spherenet_md17_naphthalene</td>
            <td nowrap="nowrap">MD17</td>
            <td nowrap="nowrap">Energy (kcal/mol) / Force (kcal/mol/Å)</td>
            <td nowrap="nowrap">0.16 / 0.26</td>
            <td nowrap="nowrap">1</td>
            <td nowrap="nowrap">~5 h</td>
            <td nowrap="nowrap"><a href="./spherenet_md17_naphthalene.yaml">config</a></td>
            <td nowrap="nowrap"><a href="https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/interatomic_potentials/spherenet/spherenet_md17_naphthalene.zip">checkpoint | log</a></td>
        </tr>
        <tr>
            <td nowrap="nowrap">spherenet_md17_salicylic</td>
            <td nowrap="nowrap">MD17</td>
            <td nowrap="nowrap">Energy (kcal/mol) / Force (kcal/mol/Å)</td>
            <td nowrap="nowrap">0.22 / 0.38</td>
            <td nowrap="nowrap">1</td>
            <td nowrap="nowrap">~5 h</td>
            <td nowrap="nowrap"><a href="./spherenet_md17_salicylic.yaml">config</a></td>
            <td nowrap="nowrap"><a href="https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/interatomic_potentials/spherenet/spherenet_md17_salicylic.zip">checkpoint | log</a></td>
        </tr>
        <tr>
            <td nowrap="nowrap">spherenet_md17_toluene</td>
            <td nowrap="nowrap">MD17</td>
            <td nowrap="nowrap">Energy (kcal/mol) / Force (kcal/mol/Å)</td>
            <td nowrap="nowrap">0.12 / 0.21</td>
            <td nowrap="nowrap">1</td>
            <td nowrap="nowrap">~4 h</td>
            <td nowrap="nowrap"><a href="./spherenet_md17_toluene.yaml">config</a></td>
            <td nowrap="nowrap"><a href="https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/interatomic_potentials/spherenet/spherenet_md17_toluene.zip">checkpoint | log</a></td>
        </tr>
        <tr>
            <td nowrap="nowrap">spherenet_md17_uracil</td>
            <td nowrap="nowrap">MD17</td>
            <td nowrap="nowrap">Energy (kcal/mol) / Force (kcal/mol/Å)</td>
            <td nowrap="nowrap">0.12 / 0.30</td>
            <td nowrap="nowrap">1</td>
            <td nowrap="nowrap">~2 h</td>
            <td nowrap="nowrap"><a href="./spherenet_md17_uracil.yaml">config</a></td>
            <td nowrap="nowrap"><a href="https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/interatomic_potentials/spherenet/spherenet_md17_uracil.zip">checkpoint | log</a></td>
        </tr>
    </body>
</table>

On NVIDIA Ampere and newer GPUs, run SphereNet with `NVIDIA_TF32_OVERRIDE=0`
to keep matrix multiplication in full FP32 precision. This is required to match
the reference implementation when predicting large absolute MD17 energies.

### Training

```bash
# Single-GPU training — MD17 aspirin (energy + force)
NVIDIA_TF32_OVERRIDE=0 python interatomic_potentials/train.py \
  -c interatomic_potentials/configs/spherenet/spherenet_md17_aspirin.yaml
```

### Validation

```bash
NVIDIA_TF32_OVERRIDE=0 python interatomic_potentials/train.py \
  -c interatomic_potentials/configs/spherenet/spherenet_md17_aspirin.yaml \
  Global.do_eval=True Global.do_train=False Global.do_test=False \
  Trainer.pretrained_model_path='your_model.pdparams'
```

### Testing

```bash
NVIDIA_TF32_OVERRIDE=0 python interatomic_potentials/train.py \
  -c interatomic_potentials/configs/spherenet/spherenet_md17_aspirin.yaml \
  Global.do_test=True Global.do_train=False Global.do_eval=False \
  Trainer.pretrained_model_path='your_model.pdparams'
```

### Prediction

```bash
# Molecular prediction
NVIDIA_TF32_OVERRIDE=0 python interatomic_potentials/predict.py \
  --model_name spherenet_md17_aspirin \
  --xyz_file_path ./interatomic_potentials/example_data/xyz/md17_aspirin.xyz

# Using a local checkpoint
NVIDIA_TF32_OVERRIDE=0 python interatomic_potentials/predict.py \
  --config_path ./interatomic_potentials/configs/spherenet/spherenet_md17_aspirin.yaml \
  --checkpoint_path ./output/spherenet_aspirin/checkpoints/best.pdparams \
  --xyz_file_path ./interatomic_potentials/example_data/xyz/md17_aspirin.xyz
```

## Citation

```bibtex
@inproceedings{liu2021spherenet,
  title={Spherical Message Passing for 3D Molecular Graphs},
  author={Liu, Yi and Wang, Limei and Liu, Meng and Lin, Yuchao and Zhang, Xuan and
          Oztekin, Bora and Ji, Shuiwang},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2021}
}
```
