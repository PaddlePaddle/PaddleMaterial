# InfGCN

[InfGCN: Equivariant Neural Operator Learning with Graphon Convolution](https://arxiv.org/abs/2311.10908)

## Abstract

We propose a general architecture that combines a coefficient-learning scheme with a residual operator layer for learning mappings between continuous functions in 3D Euclidean space. The model is SE(3)-equivariant by design. From a graph-spectrum view, the method can be interpreted as convolution on graphons (dense graphs with infinitely many nodes), which we term InfGCN. By leveraging both the continuous graphon structure and the discrete graph structure of the input data, the model effectively captures geometric information while preserving equivariance. On large-scale electron-density datasets, InfGCN outperforms current state-of-the-art architectures, and ablation studies confirm the effectiveness of the design.

![InfGCN Overview](../../docs/infgcn.png)

---

## Model Description

### Overview
InfGCN is an operator-learning model for **electron density prediction**. Given atom types $Z = (z_1,\ldots,z_N)$ and Cartesian coordinates $R = (r_1,\ldots,r_N) \in \mathbb{R}^{N \times 3}$, the model predicts a continuous electron-density field $\rho(x)$ (typically evaluated on a 3D grid). The core idea is:
- **Atom-centered basis expansion** to represent $\rho(x)$
- **SE(3)-equivariant graphon convolution** to learn expansion coefficients
- Optional **residual operator layer** to refine global details

### Method

#### 1) Atom-centered basis expansion
(1) Atom-centered basis expansion

The density field is expanded as a sum of atom-centered basis functions:

$$
\hat{\rho}(x) = \sum_{i=1}^{N} \sum_{n=1}^{N_r} \sum_{l=0}^{l_{\max}} \sum_{m=-l}^{l}
c_{i,nlm},\phi_{nlm}(x - r_i)
$$

A common choice for $\phi_{nlm}$ is a separable radial-angular basis:

$$
\phi_{nlm}(r) = g_n(|r|),Y_{lm}!\left(\widehat{r}\right),
\qquad r = x - r_i
$$

where $g_n(\cdot)$ is a radial basis and $Y_{lm}$ are spherical harmonics. All learnable information is in the coefficients $c_{i,nlm}$.

#### 2) SE(3)-equivariant coefficient learning
Coefficients are updated with equivariant message passing:

$$
C_i^{(s)} = \sum_{j \in \mathcal{N}(i)} W_{ij}^{(s)} \odot C_j^{(s-1)}, \quad s = 1,\ldots,S
$$

Edge weights $W_{ij}^{(s)}$ depend on distance and angle features (radial basis on $\lVert r_{ij}\rVert$, spherical harmonics on $\widehat{r_{ij}}$, and an MLP). This yields rotation equivariance, permutation invariance, and physically meaningful local-to-global aggregation.

#### 3) Residual operator layer (optional)

A lightweight refinement adds a learnable correction on top of the base expansion:

$$
\hat{\rho}(x) = \hat{\rho}{\text{base}}(x) + \Delta \rho{\theta}(x)
$$

where $\Delta \rho_{\theta}$ is produced by an extra operator acting on intermediate features (for example, grid features or learned coefficients).

#### 4) Training objective and metrics

A standard regression objective minimizes an $L_2$ error over the 3D domain:

$$
\mathcal{L} = \mathbb{E}!\left[\left|\hat{\rho} - \rho\right|_2^2\right]
$$

The density is discretized on an $n \times n \times n$ grid; grid points can be subsampled for memory efficiency. A common metric is **Normalized Mean Absolute Error (NMAE)**:

$$
\mathrm{NMAE} =
\frac{\sum_{i=1}^{n^3}\left|\hat{\rho}(x_i) - \rho(x_i)\right|}
{\sum_{i=1}^{n^3}\left|\rho(x_i)\right|}
$$

---

## Dataset Description

### Datasets
- **QM9_ES**: Electron densities stored as `*.CHGCAR.lz4` under the configured `./data/data_qm9` path (train 123,835 / val 50 / test 10,000).
- **MP_ES (cubic)**: Materials Project-style crystals serialized as `.json.xz` under the configured `./data/data_cubic` path (train 14,421 / val 1,000 / test 1,000).
- **OMol25_ES**: Metal-complex cubes stored under `data/dataset_OMol25_MC_5k` (full release: train 3,943 / val 493 / test 493). The InfGCN config uses the filtered split (train 3,933 / val 491 / test 491).
- **MD17_ES**: Small molecules (for example, ethanol, benzene, phenol, resorcinol) from the MD17 electron-density release under `./data/data_md`; each config points directly to its molecule/split directory. The default config trains on ethanol, and its validation split uses the published test samples.

---

## Results

<table>
    <thead>
        <tr>
            <th nowrap="nowrap">Model Name</th>
            <th nowrap="nowrap">Dataset</th>
            <th nowrap="nowrap">Normalized MAE of Density</th>
            <th nowrap="nowrap">GPUs</th>
            <th nowrap="nowrap">Training time</th>
            <th nowrap="nowrap">Config</th>
            <th nowrap="nowrap">Checkpoint | Log</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td nowrap="nowrap">infgcn_md17_benzene</td>
            <td nowrap="nowrap">MD17_EC_Benzene</td>
            <td nowrap="nowrap">21.2614%</td>
            <td nowrap="nowrap">1</td>
            <td nowrap="nowrap">59min</td>
            <td nowrap="nowrap"><a href="../../../electronic_structure/configs/infgcn/infgcn_md17_benzene.yaml">infgcn_md17_benzene</a></td>
            <td nowrap="nowrap"><a href="https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/electronic_structure/infgcn/infgcn_md17_benzene.zip">checkpoint</td>
        </tr>
        <tr>
            <td nowrap="nowrap">infgcn_md17_ethane</td>
            <td nowrap="nowrap">MD17_EC_Ethane</td>
            <td nowrap="nowrap">6.9443%</td>
            <td nowrap="nowrap">1</td>
            <td nowrap="nowrap">1hour17min</td>
            <td nowrap="nowrap"><a href="../../../electronic_structure/configs/infgcn/infgcn_md17_ethane.yaml">infgcn_md17_ethane</a></td>
            <td nowrap="nowrap"><a href="https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/electronic_structure/infgcn/infgcn_md17_ethane.zip">checkpoint</td>
        </tr>
        <tr>
            <td nowrap="nowrap">infgcn_md17_ethanol</td>
            <td nowrap="nowrap">MD17_EC_Ethanol</td>
            <td nowrap="nowrap">64.5951%</td>
            <td nowrap="nowrap">1</td>
            <td nowrap="nowrap">7min</td>
            <td nowrap="nowrap"><a href="../../../electronic_structure/configs/infgcn/infgcn_md17_ethanol.yaml">infgcn_md17_ethanol</a></td>
            <td nowrap="nowrap"><a href="https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/electronic_structure/infgcn/infgcn_md17_ethanol.zip">checkpoint</td>
        </tr>
        <tr>
            <td nowrap="nowrap">infgcn_md17_malonaldehyde</td>
            <td nowrap="nowrap">MD17_EC_Malonaldehyde</td>
            <td nowrap="nowrap">17.7947%</td>
            <td nowrap="nowrap">1</td>
            <td nowrap="nowrap">1hour29min</td>
            <td nowrap="nowrap"><a href="../../../electronic_structure/configs/infgcn/infgcn_md17_malonaldehyde.yaml">infgcn_md17_malonaldehyde</a></td>
            <td nowrap="nowrap"><a href="https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/electronic_structure/infgcn/infgcn_md17_malonaldehyde.zip">checkpoint</td>
        </tr>
        <tr>
            <td nowrap="nowrap">infgcn_md17_phenol</td>
            <td nowrap="nowrap">MD17_EC_Phenol</td>
            <td nowrap="nowrap">20.2144%</td>
            <td nowrap="nowrap">1</td>
            <td nowrap="nowrap">1hour17min</td>
            <td nowrap="nowrap"><a href="../../../electronic_structure/configs/infgcn/infgcn_md17_phenol.yaml">infgcn_md17_phenol</a></td>
            <td nowrap="nowrap"><a href="https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/electronic_structure/infgcn/infgcn_md17_phenol.zip">checkpoint</td>
        </tr>
        <tr>
            <td nowrap="nowrap">infgcn_md17_resorcinol</td>
            <td nowrap="nowrap">MD17_EC_Resorcinol</td>
            <td nowrap="nowrap">15.8850%</td>
            <td nowrap="nowrap">1</td>
            <td nowrap="nowrap">1hour23min</td>
            <td nowrap="nowrap"><a href="../../../electronic_structure/configs/infgcn/infgcn_md17_resorcinol.yaml">infgcn_md17_resorcinol</a></td>
            <td nowrap="nowrap"><a href="https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/electronic_structure/infgcn/infgcn_md17_resorcinol.zip">checkpoint</td>
        </tr>
        <tr>
            <td nowrap="nowrap">infgcn_qm9</td>
            <td nowrap="nowrap">QM9_EC</td>
            <td nowrap="nowrap">1.7542%</td>
            <td nowrap="nowrap">1</td>
            <td nowrap="nowrap">75hour41min</td>
            <td nowrap="nowrap"><a href="../../../electronic_structure/configs/infgcn/infgcn_qm9.yaml">infgcn_qm9</a></td>
            <td nowrap="nowrap"><a href="https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/electronic_structure/infgcn/infgcn_qm9.zip">checkpoint</td>
        </tr>
        <tr>
            <td nowrap="nowrap">infgcn_cubic</td>
            <td nowrap="nowrap">MP_EC (cubic)</td>
            <td nowrap="nowrap">47.3829%</td>
            <td nowrap="nowrap">1</td>
            <td nowrap="nowrap">12hour6min</td>
            <td nowrap="nowrap"><a href="../../../electronic_structure/configs/infgcn/infgcn_mp.yaml">infgcn_cubic</a></td>
            <td nowrap="nowrap"><a href="https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/electronic_structure/infgcn/infgcn_mp.zip">checkpoint</td>
        </tr>
        <tr>
            <td nowrap="nowrap">infgcn_omol25_mc_5k</td>
            <td nowrap="nowrap">OMol25_EC_5k</td>
            <td nowrap="nowrap">12.6260%</td>
            <td nowrap="nowrap">4</td>
            <td nowrap="nowrap">66hour28min</td>
            <td nowrap="nowrap"><a href="../../../electronic_structure/configs/infgcn/infgcn_omol25_MC_5k_trimmed.yaml">infgcn_omol25</a></td>
            <td nowrap="nowrap"><a href="https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/electronic_structure/infgcn/infgcn_omol25_mc_5k_trimmed.zip">checkpoint</td>
        </tr>
    </tbody>
</table>

---

## Command

### Training
```bash
# multi-gpu training (example with 4 GPUs)
python -m paddle.distributed.launch --gpus="0,1,2,3" electronic_structure/train.py -c electronic_structure/configs/infgcn/infgcn_qm9.yaml
# single-gpu training
python electronic_structure/train.py -c electronic_structure/configs/infgcn/infgcn_qm9.yaml
```

### Validation
```bash
# Enable eval-only mode with a saved checkpoint.
python electronic_structure/train.py -c electronic_structure/configs/infgcn/infgcn_qm9.yaml Global.do_eval=True Global.do_train=False Global.do_test=False Trainer.pretrained_model_path='path/to/model.pdparams'
```

### Testing
```bash
# Evaluate on the test split using a pretrained checkpoint.
python electronic_structure/train.py -c electronic_structure/configs/infgcn/infgcn_qm9.yaml Global.do_eval=False Global.do_train=False Global.do_test=True Trainer.pretrained_model_path='path/to/model.pdparams'
```

### Prediction
```bash
# 1) One-click inference with a registered pretrained model and bundled molecule.
python electronic_structure/predict.py \
  --model_name infgcn_qm9 \
  --weights_name best.pdparams \
  --mol_file_path electronic_structure/configs/infgcn/example/methane.mol \
  --grid_shape 8 \
  --grid_batch_size 4096 \
  --save_path output/infgcn_qm9/methane

# 2) Predict one sample (selected by --index) from the dataset split configured by the YAML.
python electronic_structure/predict.py \
  --config_path electronic_structure/configs/infgcn/infgcn_qm9.yaml \
  --checkpoint_path path/to/infgcn_qm9.pdparams

# 3) MOL-file inference (single file or directory).
# This mode predicts electron density from molecular structure files (*.mol),
# and can export predicted cube + html visualization.
python electronic_structure/predict.py \
  --config_path electronic_structure/configs/infgcn/infgcn_omol25_MC_5k_trimmed.yaml \
  --checkpoint_path path/to/infgcn_omol25.pdparams \
  --mol_file_path path/to/mols_or_mol_file \
  --save_path output/infgcn_mol \
  --save_html

# 4) MOL-file inference with reference (true) cube files.
# If --reference_cube_dir provides matching files, the script can additionally
# write the reference CUBE and true/difference visualizations.
python electronic_structure/predict.py \
  --config_path electronic_structure/configs/infgcn/infgcn_omol25_MC_5k_trimmed.yaml \
  --checkpoint_path path/to/infgcn_omol25.pdparams \
  --mol_file_path path/to/mols_or_mol_file \
  --reference_cube_dir path/to/true_cubes \
  --save_path output/infgcn_reference \
  --save_true_cube \
  --save_html
```

Notes:
- Replace `path/to/*.pdparams` with a downloaded pretrained checkpoint or a checkpoint produced by training.
- Model configs keep the recommended `grid_batch_size` and share the configured
  field builder between Dataset and Predict. Input selection, output
  paths, cube/html export, visualization, and MOL grid settings are runtime
  command-line options.
- `--mol_file_path` supports either one `.mol` file or a directory of `.mol`
  files. The atom vocabulary declared by `Vocabulary` is downloaded and used
  automatically.
- `--save_path` is an output directory and always receives the predicted CUBE.
- Optional MOL grid controls are `--grid_shape` (default `80,80,80`) and
  `--grid_padding` (default `6.0` Angstrom).
- MOL input coordinates and `--grid_padding` are interpreted directly in
  Angstrom. Reference CUBE geometry is read from the unit encoded by the file;
  its atom order is validated before its grid and atom coordinates are used.
- If true/reference cube is not provided, only predicted outputs are available (`*_pred.cube`, `*_pred_density.html`).
- If kaleido/Chrome is unavailable, the script writes interactive `.html` instead of `.png`.
- If a dataset already exists elsewhere, override its configured dataset path.

---

## Citation
```
@article{cheng2023infgcn,
  title={Equivariant neural operator learning with graphon convolution},
  author={Cheng, Chaoran and Peng, Jian},
  journal={arXiv preprint arXiv:2311.10908},
  year={2023}
}
```
