# InfGCN

[InfGCN: Equivariant Neural Operator Learning with Graphon Convolution](https://arxiv.org/abs/2311.10908)

## Abstract

We propose a general architecture that combines the coefficient learning scheme with a residual operator layer for learning mappings between continuous functions in the 3D Euclidean space. Our proposed model is guaranteed to achieve SE(3)-equivariance by design. From the graph spectrum view, our method can be interpreted as convolution on graphons (dense graphs with infinitely many nodes), which we term InfGCN. By leveraging both the continuous graphon structure and the discrete graph structure of the input data, our model can effectively capture the geometric information while preserving equivariance. Through extensive experiments on large-scale electron density datasets, we observed that our model significantly outperformed the current state-of-the-art architectures. Multiple ablation studies were also carried out to demonstrate the effectiveness of the proposed architecture.

![InfGCN Overview](../../docs/infgcn.png)

## Datasets

- **QM9_EC**: Electron densities stored as `*.CHGCAR.lz4` in `dataset_ES/data_qm9` (train 123,835 · val 50 · test 10,000). [Atom dictionary](https://paddle-org.bj.bcebos.com/paddlematerials/datasets/QM9_ES/qm9.json), [Split file](https://paddle-org.bj.bcebos.com/paddlematerials/datasets/QM9_ES/qm9_data_split.json).
- **MP_EC (cubic)**: Materials Project-style crystals serialized as `.json.xz` under `dataset_ES/data_cubic` (train 14,421 · val 1,000 · test 1,000).[Atom dictionary](https://paddle-org.bj.bcebos.com/paddlematerials/datasets/MP_ES/crystal.json), [Split file](https://paddle-org.bj.bcebos.com/paddlematerials/datasets/MP_ES/crystal_data_split.json). 
- **OMol25_EC**: Organic molecule cubes expected under `/home/liuxuwei01/processed_output` (train 16 · val 2 · test 2). [Atom dictionary](https://paddle-org.bj.bcebos.com/paddlematerials/datasets/OMol25_ES/MC_5k/omol25.json), [Split file](https://paddle-org.bj.bcebos.com/paddlematerials/datasets/OMol25_ES/MC_5k/omol25_data_split.json)
- **MD17_EC**: Small molecules (e.g., ethanol, benzene, phenol, resorcinol) from the MD17 electron-density release in `dataset_ES/data_md`. Dataset READMEs credit Bogojeski et al. (density release) and Chmiela et al. (MD17); default config trains on ethanol.

## Results

<table>
    <thead>
        <tr>
            <th nowrap="nowrap">Model Name</th>
            <th nowrap="nowrap">Dataset</th>
            <th nowrap="nowrap">Density MAE</th>
            <th nowrap="nowrap">GPUs</th>
            <th nowrap="nowrap">Training time</th>
            <th nowrap="nowrap">Config</th>
            <th nowrap="nowrap">Checkpoint | Log</th>
        </tr>
    </thead>
    <tbody>
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
        <tr>
            <td nowrap="nowrap">infgcn_md</td>
            <td nowrap="nowrap">MD17_EC (ethanol)</td>
            <td nowrap="nowrap">TBD</td>
            <td nowrap="nowrap">~</td>
            <td nowrap="nowrap">~</td>
            <td nowrap="nowrap"><a href="../../../electronic_structure/configs/infgcn/infgcn_md.yaml">infgcn_md</a></td>
            <td nowrap="nowrap">TBD</td>
        </tr>
    </tbody>
</table>

**Note**: Benchmarks are being regenerated in Paddle; metrics and downloadable checkpoints will be published once validation completes.

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
# Run inference on custom data by pointing the test dataset fields to your electron-density files
# (e.g., updating Dataset.test.dataset.root / split_file / atom_file) and loading a trained model.
python electronic_structure/train.py -c electronic_structure/configs/infgcn/infgcn_qm9.yaml Global.do_train=False Global.do_eval=False Global.do_test=True Trainer.pretrained_model_path='your checkpoint path (*.pdparams)'
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
