# InfGCN

[InfGCN: Equivariant Neural Operator Learning with Graphon Convolution](https://arxiv.org/abs/2311.10908)

## Abstract

We propose a general architecture that combines the coefficient learning scheme with a residual operator layer for learning mappings between continuous functions in the 3D Euclidean space. Our proposed model is guaranteed to achieve SE(3)-equivariance by design. From the graph spectrum view, our method can be interpreted as convolution on graphons (dense graphs with infinitely many nodes), which we term InfGCN. By leveraging both the continuous graphon structure and the discrete graph structure of the input data, our model can effectively capture the geometric information while preserving equivariance. Through extensive experiments on large-scale electron density datasets, we observed that our model significantly outperformed the current state-of-the-art architectures. Multiple ablation studies were also carried out to demonstrate the effectiveness of the proposed architecture.

![InfGCN Overview](../../docs/infgcn.png)

## Datasets

- **QM9_EC**: Electron densities stored as `*.CHGCAR.lz4` in `dataset_ES/data_qm9` (train 123,835 · val 50 · test 10,000). [Data](https://paddle-org.bj.bcebos.com/paddlematerials/datasets/QM9_ES/qm9_es.tar), [Atom dictionary](https://paddle-org.bj.bcebos.com/paddlematerials/datasets/QM9_ES/qm9.json), [Split file](https://paddle-org.bj.bcebos.com/paddlematerials/datasets/QM9_ES/qm9_data_split.json).
- **MP_EC (cubic)**: Materials Project-style crystals serialized as `.json.xz` under `dataset_ES/data_cubic` (train 14,421 · val 1,000 · test 1,000).[Data](https://paddle-org.bj.bcebos.com/paddlematerials/datasets/MP_ES/mp_es.tar), [Atom dictionary](https://paddle-org.bj.bcebos.com/paddlematerials/datasets/MP_ES/crystal.json), [Split file](https://paddle-org.bj.bcebos.com/paddlematerials/datasets/MP_ES/crystal_data_split.json). 
- **OMol25_EC**: Organic molecule cubes expected under `/home/liuxuwei01/processed_output` (train 16 · val 2 · test 2). [Data](https://paddle-org.bj.bcebos.com/paddlematerials/datasets/OMol25_ES/MC_5k/omol25_mc_5k.tar), [Atom dictionary](https://paddle-org.bj.bcebos.com/paddlematerials/datasets/OMol25_ES/MC_5k/omol25.json), [Split file](https://paddle-org.bj.bcebos.com/paddlematerials/datasets/OMol25_ES/MC_5k/omol25_data_split.json)
- **MD17_EC**: Small molecules (e.g., ethanol, benzene, phenol, resorcinol) from the MD17 electron-density release in `dataset_ES/data_md`. Dataset READMEs credit Bogojeski et al. (density release) and Chmiela et al. (MD17); default config trains on ethanol. [Data](https://paddle-org.bj.bcebos.com/paddlematerials/datasets/MD17_ES/md17_es.tar.gz), 

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
            <td nowrap="nowrap"><a href="https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/electronic_structure/infgcn/infgcn_md17_benzene_t_20260104_083255_s_42.zip">checkpoint | log</td>
        </tr>
        <tr>
            <td nowrap="nowrap">infgcn_md17_ethane</td>
            <td nowrap="nowrap">MD17_EC_Ethane</td>
            <td nowrap="nowrap">6.9443%</td>
            <td nowrap="nowrap">1</td>
            <td nowrap="nowrap">1hour17min</td>
            <td nowrap="nowrap"><a href="../../../electronic_structure/configs/infgcn/infgcn_md17_ethane.yaml">infgcn_md17_ethane</a></td>
            <td nowrap="nowrap"><a href="https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/electronic_structure/infgcn/infgcn_md17_ethane_t_20260105_035402_s_42.zip">checkpoint | log</td>
        </tr>
        <tr>
            <td nowrap="nowrap">infgcn_md17_ethanol</td>
            <td nowrap="nowrap">MD17_EC_Ethanol</td>
            <td nowrap="nowrap">64.5951%</td>
            <td nowrap="nowrap">1</td>
            <td nowrap="nowrap">7min</td>
            <td nowrap="nowrap"><a href="../../../electronic_structure/configs/infgcn/infgcn_md17_ethanol.yaml">infgcn_md17_ethanol</a></td>
            <td nowrap="nowrap"><a href="https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/electronic_structure/infgcn/infgcn_md17_ethanol_t_20260105_051323_s_42.zip">checkpoint | log</td>
        </tr>
        <tr>
            <td nowrap="nowrap">infgcn_md17_malonaldehyde</td>
            <td nowrap="nowrap">MD17_EC_Malonaldehyde</td>
            <td nowrap="nowrap">17.7947%</td>
            <td nowrap="nowrap">1</td>
            <td nowrap="nowrap">1hour29min</td>
            <td nowrap="nowrap"><a href="../../../electronic_structure/configs/infgcn/infgcn_md17_malonaldehyde.yaml">infgcn_md17_malonaldehyde</a></td>
            <td nowrap="nowrap"><a href="https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/electronic_structure/infgcn/infgcn_md17_malonaldehyde_t_20260105_052204_s_42.zip">checkpoint | log</td>
        </tr>
        <tr>
            <td nowrap="nowrap">infgcn_md17_phenol</td>
            <td nowrap="nowrap">MD17_EC_Phenol</td>
            <td nowrap="nowrap">20.2144%</td>
            <td nowrap="nowrap">1</td>
            <td nowrap="nowrap">1hour17min</td>
            <td nowrap="nowrap"><a href="../../../electronic_structure/configs/infgcn/infgcn_md17_phenol.yaml">infgcn_md17_phenol</a></td>
            <td nowrap="nowrap"><a href="https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/electronic_structure/infgcn/infgcn_md17_phenol_t_20260107_041611_s_42.zip">checkpoint | log</td>
        </tr>
        <tr>
            <td nowrap="nowrap">infgcn_md17_resorcinol</td>
            <td nowrap="nowrap">MD17_EC_Resorcinol</td>
            <td nowrap="nowrap">15.8850%</td>
            <td nowrap="nowrap">1</td>
            <td nowrap="nowrap">1hour23min</td>
            <td nowrap="nowrap"><a href="../../../electronic_structure/configs/infgcn/infgcn_md17_resorcinol.yaml">infgcn_md17_resorcinol</a></td>
            <td nowrap="nowrap"><a href="https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/electronic_structure/infgcn/infgcn_md17_resorcinol_t_20260104_083255_s_42.zip">checkpoint | log</td>
        </tr>
        <tr>
            <td nowrap="nowrap">infgcn_qm9</td>
            <td nowrap="nowrap">QM9_EC</td>
            <td nowrap="nowrap">TBD</td>
            <td nowrap="nowrap">~</td>
            <td nowrap="nowrap">~</td>
            <td nowrap="nowrap"><a href="../../../electronic_structure/configs/infgcn/infgcn_qm9.yaml">infgcn_qm9</a></td>
            <td nowrap="nowrap">TBD</td>
        </tr>
        <tr>
            <td nowrap="nowrap">infgcn_cubic</td>
            <td nowrap="nowrap">MP_EC (cubic)</td>
            <td nowrap="nowrap">TBD</td>
            <td nowrap="nowrap">~</td>
            <td nowrap="nowrap">~</td>
            <td nowrap="nowrap"><a href="../../../electronic_structure/configs/infgcn/infgcn_cubic.yaml">infgcn_cubic</a></td>
            <td nowrap="nowrap">TBD</td>
        </tr>
        <tr>
            <td nowrap="nowrap">infgcn_omol25</td>
            <td nowrap="nowrap">OMol25_EC</td>
            <td nowrap="nowrap">TBD</td>
            <td nowrap="nowrap">~</td>
            <td nowrap="nowrap">~</td>
            <td nowrap="nowrap"><a href="../../../electronic_structure/configs/infgcn/infgcn_omol25.yaml">infgcn_omol25</a></td>
            <td nowrap="nowrap">TBD</td>
        </tr>
    </tbody>
</table>

**Note**: Benchmarks are being regenerated in Paddle; metrics and downloadable checkpoints will be published once validation completes. Pretrained QM9 weights: [infgcn_qm9](https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/electronic_structure/infgcn/infgcn_qm9.pdparams)

### Training

```bash
# multi-gpu training
python -m paddle.distributed.launch --gpus="0,1,2,3" electronic_structure/train.py -c electronic_structure/configs/infgcn/infgcn_qm9.yaml
# single-gpu training
python electronic_structure/train.py -c electronic_structure/configs/infgcn/infgcn_qm9.yaml
```

### Validation

```bash
# Adjust runtime options via CLI without editing the YAML, e.g. enabling eval-only runs with a saved checkpoint.
python electronic_structure/train.py -c electronic_structure/configs/infgcn/infgcn_qm9.yaml Global.do_eval=True Global.do_train=False Global.do_test=False Trainer.pretrained_model_path='your checkpoint path (*.pdparams)'
```

### Testing

```bash
# Evaluate on the test split using a pretrained checkpoint.
python electronic_structure/train.py -c electronic_structure/configs/infgcn/infgcn_qm9.yaml Global.do_test=True Global.do_train=False Global.do_eval=False Trainer.pretrained_model_path='your checkpoint path (*.pdparams)'
```

### Prediction

```bash
# Run inference with the standalone predictor (uses the dataset paths from the YAML; override via flags if needed).
python electronic_structure/predict.py \
  --config electronic_structure/configs/infgcn/infgcn_qm9.yaml \
  --checkpoint output/infgcn_qm9_best/infgcn_qm9.pdparams \
  --split validation \
  --index 0 \
  --grid_batch_size 20000 \
  --output_dir output/infgcn_qm9_best/vis_val0
# Notes: create a symlink to your data root if it lives elsewhere, e.g. ln -s /path/to/dataset_ES dataset_ES.
# If kaleido is missing, the script writes interactive .html files instead of .png; install kaleido to export PNGs.
```

## Citation
```
@article{deng2023chgnet,
  title={Equivariant neural operator learning with graphon convolution},
  author={Chaoran Cheng and Jian Peng},
  booktitle={Advances in Neural Information Processing Systems 37: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, December 10-16, 2023},
  month={December},
  year={2023},
}
```
