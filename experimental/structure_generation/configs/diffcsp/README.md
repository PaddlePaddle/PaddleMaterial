# DiffCSP

[COMPLETE AND EFFICIENT GRAPH TRANSFORMERS FOR CRYSTAL MATERIAL PROPERTY PREDICTION](https://arxiv.org/abs/2309.04475)

## Abstract

Crystal structures are characterized by atomic bases within a primitive unit cell that repeats along a regular lattice throughout 3D space. The periodic and infinite nature of crystals poses unique challenges for geometric graph representation learning. Specifically, constructing graphs that effectively capture the complete geometric information of crystals and handle chiral crystals remains an unsolved and challenging problem. In this paper, we introduce a novel approach that utilizes the periodic patterns of unit cells to establish the lattice-based representation for each atom, enabling efficient and expressive graph representations of crystals. Furthermore, we propose ComFormer, a SE(3) transformer designed specifically for crystalline materials. ComFormer includes two variants; namely, iComFormer that employs invariant geometric descriptors of Euclidean distances and angles, and eComFormer that utilizes equivariant vector representations. Experimental results demonstrate the state-of-the-art predictive accuracy of ComFormer variants on various tasks across three widely-used crystal benchmarks.

## Datasets:

- MP20:

    MP-20 selects 45,231 stable inorganic materials from Material Projects, which includes the majority of experimentally-generated materials with at most 20 atoms in a unit cell.

    |                                     Dataset                                      | train |  val  | test  |
    | :------------------------------------------------------------------------------: | :---: | :---: | :---: |
    | [MP20](https://paddle-org.bj.bcebos.com/paddlematerial/datasets/mp_20/mp_20.zip) | 27136 | 9047  | 9046  |


## Results

<table>
    <head>
        <tr>
            <th  nowrap="nowrap">Model</th>
            <th  nowrap="nowrap">Dataset</th>
            <th  nowrap="nowrap">Match Rate</th>
            <th  nowrap="nowrap">RMS Dist</th>
            <th  nowrap="nowrap">GPUs</th>
            <th  nowrap="nowrap">Training time</th>
            <th  nowrap="nowrap">Config</th>
            <th  nowrap="nowrap">Checkpoint | Log</th>
        </tr>
    </head>
    <body>
        <tr>
            <td  nowrap="nowrap">DiffCSP</td>
            <td  nowrap="nowrap">mp20</td>
            <th  nowrap="nowrap">51.72</th>
            <td  nowrap="nowrap">0.0591</td>
            <td  nowrap="nowrap">1</td>
            <td  nowrap="nowrap">~13.5 hours</td>
            <td  nowrap="nowrap"><a href="diffcsp_mp20.yaml">diffcsp_mp20</a></td>
            <td  nowrap="nowrap"><a href="https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/structure_generation/diffcsp/diffcsp_mp20.zip">checkpoint | log</a></td>
        </tr>  
    </body>
</table>

### Training
```bash
# multi-gpu training, we use 4 gpus here
python -m paddle.distributed.launch --gpus="0,1,2,3" structure_generation/train.py -c structure_generation/configs/diffcsp/diffcsp_mp20.yaml
# single-gpu training
python structure_generation/train.py -c structure_generation/configs/diffcsp/diffcsp_mp20.yaml
```

### Validation
```bash
# Adjust program behavior on-the-fly using command-line parameters – this provides a convenient way to customize settings without modifying the configuration file directly.
# such as: --Global.do_eval=True
python structure_generation/train.py -c structure_generation/configs/diffcsp/diffcsp_mp20.yaml Global.do_eval=True Global.do_train=False Global.do_test=False
```

### Testing
```bash
# This command is used to evaluate the model's performance on the test dataset.
python structure_generation/train.py -c structure_generation/configs/diffcsp/diffcsp_mp20.yaml Global.do_eval=False Global.do_train=False Global.do_test=True
```

### Sample
```bash
# This command is used to predict the  crystal structure using a trained model.
# Note: The model_name and weights_name parameters are used to specify the pre-trained model and its corresponding weights. The chemical_formula parameter is used to specify the chemical formula of the crystal structure to be predicted.
# The prediction results will be saved in the folder specified by the `save_path` parameter, with the default set to `result`.

# Mode 1: Leverage a pre-trained machine learning model for crystal structure prediction. The implementation includes automated model download functionality, eliminating the need for manual configuration.
python structure_generation/sample.py --model_name='diffcsp_mp20' --weights_name='latest.pdparams' --save_path='result_diffcsp_mp20/' ----chemical_formula="LiMnO2"

# Mode2: Use a custom configuration file and checkpoint for crystal structure prediction. This approach allows for more flexibility and customization.
python structure_generation/sample.py --config_path='structure_generation/configs/diffcsp/diffcsp_mp20.yaml' --checkpoint_path='./output/diffcsp_mp20/checkpoints/latest.pdparams' --save_path='result_diffcsp_mp20/' --chemical_formula="LiMnO2"
```

## Citation
```
@article{jiao2023crystal,
  title={Crystal structure prediction by joint equivariant diffusion},
  author={Jiao, Rui and Huang, Wenbing and Lin, Peijia and Han, Jiaqi and Chen, Pin and Lu, Yutong and Liu, Yang},
  journal={arXiv preprint arXiv:2309.04475},
  year={2023}
}
```
