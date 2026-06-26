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
  <img src="figures/spherenet_architecture.svg" alt="SphereNet Architecture" width="80%"/>
  <br/>
  <em>Figure 1: SphereNet architecture.</em>
</p>

## Datasets

### QM9

The QM9 dataset contains 130,831 small organic molecules (up to 9 heavy
atoms: C, O, N, F) with 12 quantum-chemical properties computed at the
B3LYP/6-31G(2df,p) level of theory.

| Split   | Size   |
|---------|--------|
| Train   | 110,831 |
| Val     | 10,000  |
| Test    | 10,000  |
| **Total** | **130,831** |

**Data format**: Each molecule contains atomic numbers (`z`), 3D positions
(`pos`), and 12 property labels. The raw dataset is available at
[figshare](https://figshare.com/ndownloader/files/3195389).

**Reference**: [Quantum-chemical insights from deep learning](https://arxiv.org/abs/1708.04444) (Gaussian, 2017)

### MD17

The MD17 dataset contains DFT molecular dynamics trajectories for 8 small
organic molecules. Each configuration includes the total energy (kcal/mol) and
atomic forces (kcal/mol/Å).

| Molecule      | Train | Val  | Test | Atoms |
|---------------|------:|-----:|-----:|------:|
| Aspirin       | 1000  | 500  | 1000 | 21    |
| Benzene       | 1000  | 500  | 1000 | 12    |
| Ethanol       | 1000  | 500  | 1000 | 9     |
| Malonaldehyde | 1000  | 500  | 1000 | 9     |
| Naphthalene   | 1000  | 500  | 1000 | 18    |
| Salicylic     | 1000  | 500  | 1000 | 16    |
| Toluene       | 1000  | 500  | 1000 | 15    |
| Uracil        | 1000  | 500  | 1000 | 12    |

**Data format**: Each molecule is stored as a single `.npz` file with keys
`E` (energies), `F` (forces), `R` (positions), and `z` (atomic numbers).

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
            <th nowrap="nowrap">Checkpoint</th>
        </tr>
    </head>
    <body>
        <tr>
            <td nowrap="nowrap">spherenet_qm9_mu</td>
            <td nowrap="nowrap">QM9</td>
            <td nowrap="nowrap">$\mu$ (D)</td>
            <td nowrap="nowrap">0.032</td>
            <td nowrap="nowrap">1</td>
            <td nowrap="nowrap">~18 h</td>
            <td nowrap="nowrap"><a href="spherenet_qm9_mu.yaml">config</a></td>
            <td nowrap="nowrap">-</td>
        </tr>
        <tr>
            <td nowrap="nowrap">spherenet_qm9_alpha</td>
            <td nowrap="nowrap">QM9</td>
            <td nowrap="nowrap">$\alpha$ (Bohr³)</td>
            <td nowrap="nowrap">0.24</td>
            <td nowrap="nowrap">1</td>
            <td nowrap="nowrap">~24 h</td>
            <td nowrap="nowrap"><a href="spherenet_qm9_alpha.yaml">config</a></td>
            <td nowrap="nowrap">-</td>
        </tr>
        <tr>
            <td nowrap="nowrap">spherenet_qm9_homo</td>
            <td nowrap="nowrap">QM9</td>
            <td nowrap="nowrap">$\varepsilon_{\text{HOMO}}$ (meV)</td>
            <td nowrap="nowrap">42</td>
            <td nowrap="nowrap">1</td>
            <td nowrap="nowrap">~22 h</td>
            <td nowrap="nowrap"><a href="spherenet_qm9_homo.yaml">config</a></td>
            <td nowrap="nowrap">-</td>
        </tr>
        <tr>
            <td nowrap="nowrap">spherenet_qm9_lumo</td>
            <td nowrap="nowrap">QM9</td>
            <td nowrap="nowrap">$\varepsilon_{\text{LUMO}}$ (meV)</td>
            <td nowrap="nowrap">43</td>
            <td nowrap="nowrap">1</td>
            <td nowrap="nowrap">~22 h</td>
            <td nowrap="nowrap"><a href="spherenet_qm9_lumo.yaml">config</a></td>
            <td nowrap="nowrap">-</td>
        </tr>
        <tr>
            <td nowrap="nowrap">spherenet_qm9_gap</td>
            <td nowrap="nowrap">QM9</td>
            <td nowrap="nowrap">$\Delta\varepsilon$ (meV)</td>
            <td nowrap="nowrap">62</td>
            <td nowrap="nowrap">1</td>
            <td nowrap="nowrap">~22 h</td>
            <td nowrap="nowrap"><a href="spherenet_qm9_gap.yaml">config</a></td>
            <td nowrap="nowrap">-</td>
        </tr>
        <tr>
            <td nowrap="nowrap">spherenet_qm9_r2</td>
            <td nowrap="nowrap">QM9</td>
            <td nowrap="nowrap">$\langle R^2 \rangle$ (Bohr²)</td>
            <td nowrap="nowrap">0.30</td>
            <td nowrap="nowrap">1</td>
            <td nowrap="nowrap">~12 h</td>
            <td nowrap="nowrap"><a href="spherenet_qm9_r2.yaml">config</a></td>
            <td nowrap="nowrap">-</td>
        </tr>
        <tr>
            <td nowrap="nowrap">spherenet_qm9_zpve</td>
            <td nowrap="nowrap">QM9</td>
            <td nowrap="nowrap">ZPVE (meV)</td>
            <td nowrap="nowrap">1.4</td>
            <td nowrap="nowrap">1</td>
            <td nowrap="nowrap">~14 h</td>
            <td nowrap="nowrap"><a href="spherenet_qm9_zpve.yaml">config</a></td>
            <td nowrap="nowrap">-</td>
        </tr>
        <tr>
            <td nowrap="nowrap">spherenet_qm9_U0</td>
            <td nowrap="nowrap">QM9</td>
            <td nowrap="nowrap">$U_0$ (meV)</td>
            <td nowrap="nowrap">22</td>
            <td nowrap="nowrap">1</td>
            <td nowrap="nowrap">~20 h</td>
            <td nowrap="nowrap"><a href="spherenet_qm9_U0.yaml">config</a></td>
            <td nowrap="nowrap">-</td>
        </tr>
        <tr>
            <td nowrap="nowrap">spherenet_qm9_U</td>
            <td nowrap="nowrap">QM9</td>
            <td nowrap="nowrap">$U$ (meV)</td>
            <td nowrap="nowrap">22</td>
            <td nowrap="nowrap">1</td>
            <td nowrap="nowrap">~20 h</td>
            <td nowrap="nowrap"><a href="spherenet_qm9_U.yaml">config</a></td>
            <td nowrap="nowrap">-</td>
        </tr>
        <tr>
            <td nowrap="nowrap">spherenet_qm9_H</td>
            <td nowrap="nowrap">QM9</td>
            <td nowrap="nowrap">$H$ (meV)</td>
            <td nowrap="nowrap">22</td>
            <td nowrap="nowrap">1</td>
            <td nowrap="nowrap">~20 h</td>
            <td nowrap="nowrap"><a href="spherenet_qm9_H.yaml">config</a></td>
            <td nowrap="nowrap">-</td>
        </tr>
        <tr>
            <td nowrap="nowrap">spherenet_qm9_G</td>
            <td nowrap="nowrap">QM9</td>
            <td nowrap="nowrap">$G$ (meV)</td>
            <td nowrap="nowrap">22</td>
            <td nowrap="nowrap">1</td>
            <td nowrap="nowrap">~20 h</td>
            <td nowrap="nowrap"><a href="spherenet_qm9_G.yaml">config</a></td>
            <td nowrap="nowrap">-</td>
        </tr>
        <tr>
            <td nowrap="nowrap">spherenet_qm9_Cv</td>
            <td nowrap="nowrap">QM9</td>
            <td nowrap="nowrap">$C_v$ (cal/(mol·K))</td>
            <td nowrap="nowrap">0.052</td>
            <td nowrap="nowrap">1</td>
            <td nowrap="nowrap">~18 h</td>
            <td nowrap="nowrap"><a href="spherenet_qm9_Cv.yaml">config</a></td>
            <td nowrap="nowrap">-</td>
        </tr>
        <tr>
            <td nowrap="nowrap">spherenet_md17_aspirin</td>
            <td nowrap="nowrap">MD17</td>
            <td nowrap="nowrap">Energy (kcal/mol) / Force (kcal/mol/Å)</td>
            <td nowrap="nowrap">0.26 / 0.44</td>
            <td nowrap="nowrap">1</td>
            <td nowrap="nowrap">~6 h</td>
            <td nowrap="nowrap"><a href="spherenet_md17_aspirin.yaml">config</a></td>
            <td nowrap="nowrap">-</td>
        </tr>
        <tr>
            <td nowrap="nowrap">spherenet_md17_benzene_old</td>
            <td nowrap="nowrap">MD17</td>
            <td nowrap="nowrap">Energy (kcal/mol) / Force (kcal/mol/Å)</td>
            <td nowrap="nowrap">0.14 / 0.21</td>
            <td nowrap="nowrap">1</td>
            <td nowrap="nowrap">~3 h</td>
            <td nowrap="nowrap"><a href="spherenet_md17_benzene_old.yaml">config</a></td>
            <td nowrap="nowrap">-</td>
        </tr>
        <tr>
            <td nowrap="nowrap">spherenet_md17_ethanol</td>
            <td nowrap="nowrap">MD17</td>
            <td nowrap="nowrap">Energy (kcal/mol) / Force (kcal/mol/Å)</td>
            <td nowrap="nowrap">0.10 / 0.23</td>
            <td nowrap="nowrap">1</td>
            <td nowrap="nowrap">~2 h</td>
            <td nowrap="nowrap"><a href="spherenet_md17_ethanol.yaml">config</a></td>
            <td nowrap="nowrap">-</td>
        </tr>
        <tr>
            <td nowrap="nowrap">spherenet_md17_malonaldehyde</td>
            <td nowrap="nowrap">MD17</td>
            <td nowrap="nowrap">Energy (kcal/mol) / Force (kcal/mol/Å)</td>
            <td nowrap="nowrap">0.17 / 0.32</td>
            <td nowrap="nowrap">1</td>
            <td nowrap="nowrap">~2 h</td>
            <td nowrap="nowrap"><a href="spherenet_md17_malonaldehyde.yaml">config</a></td>
            <td nowrap="nowrap">-</td>
        </tr>
        <tr>
            <td nowrap="nowrap">spherenet_md17_naphthalene</td>
            <td nowrap="nowrap">MD17</td>
            <td nowrap="nowrap">Energy (kcal/mol) / Force (kcal/mol/Å)</td>
            <td nowrap="nowrap">0.16 / 0.26</td>
            <td nowrap="nowrap">1</td>
            <td nowrap="nowrap">~5 h</td>
            <td nowrap="nowrap"><a href="spherenet_md17_naphthalene.yaml">config</a></td>
            <td nowrap="nowrap">-</td>
        </tr>
        <tr>
            <td nowrap="nowrap">spherenet_md17_salicylic</td>
            <td nowrap="nowrap">MD17</td>
            <td nowrap="nowrap">Energy (kcal/mol) / Force (kcal/mol/Å)</td>
            <td nowrap="nowrap">0.22 / 0.38</td>
            <td nowrap="nowrap">1</td>
            <td nowrap="nowrap">~5 h</td>
            <td nowrap="nowrap"><a href="spherenet_md17_salicylic.yaml">config</a></td>
            <td nowrap="nowrap">-</td>
        </tr>
        <tr>
            <td nowrap="nowrap">spherenet_md17_toluene</td>
            <td nowrap="nowrap">MD17</td>
            <td nowrap="nowrap">Energy (kcal/mol) / Force (kcal/mol/Å)</td>
            <td nowrap="nowrap">0.12 / 0.21</td>
            <td nowrap="nowrap">1</td>
            <td nowrap="nowrap">~4 h</td>
            <td nowrap="nowrap"><a href="spherenet_md17_toluene.yaml">config</a></td>
            <td nowrap="nowrap">-</td>
        </tr>
        <tr>
            <td nowrap="nowrap">spherenet_md17_uracil</td>
            <td nowrap="nowrap">MD17</td>
            <td nowrap="nowrap">Energy (kcal/mol) / Force (kcal/mol/Å)</td>
            <td nowrap="nowrap">0.12 / 0.30</td>
            <td nowrap="nowrap">1</td>
            <td nowrap="nowrap">~2 h</td>
            <td nowrap="nowrap"><a href="spherenet_md17_uracil.yaml">config</a></td>
            <td nowrap="nowrap">-</td>
        </tr>
    </body>
</table>

### Training

```bash
# Single-GPU training — QM9 mu property
python property_prediction/train.py \
  -c property_prediction/configs/spherenet/spherenet_qm9_mu.yaml

# Single-GPU training — MD17 aspirin (energy + force)
python property_prediction/train.py \
  -c property_prediction/configs/spherenet/spherenet_md17_aspirin.yaml
```

### Validation / Testing

```bash
# Validation
python property_prediction/train.py \
  -c property_prediction/configs/spherenet/spherenet_qm9_mu.yaml \
  Global.do_eval=True Global.do_train=False Global.do_test=False \
  Trainer.pretrained_model_path='your_model.pdparams'

# Testing
python property_prediction/train.py \
  -c property_prediction/configs/spherenet/spherenet_qm9_mu.yaml \
  Global.do_test=True Global.do_train=False Global.do_eval=False \
  Trainer.pretrained_model_path='your_model.pdparams'
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
