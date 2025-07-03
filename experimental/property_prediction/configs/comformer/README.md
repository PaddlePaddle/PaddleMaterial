# ComFormer

[COMPLETE AND EFFICIENT GRAPH TRANSFORMERS FOR CRYSTAL MATERIAL PROPERTY PREDICTION](https://arxiv.org/pdf/2403.11857)

## Abstract

Crystal structures are characterized by atomic bases within a primitive unit cell that repeats along a regular lattice throughout 3D space. The periodic and infinite nature of crystals poses unique challenges for geometric graph representation learning. Specifically, constructing graphs that effectively capture the complete geometric information of crystals and handle chiral crystals remains an unsolved and challenging problem. In this paper, we introduce a novel approach that utilizes the periodic patterns of unit cells to establish the lattice-based representation for each atom, enabling efficient and expressive graph representations of crystals. Furthermore, we propose ComFormer, a SE(3) transformer designed specifically for crystalline materials. ComFormer includes two variants; namely, iComFormer that employs invariant geometric descriptors of Euclidean distances and angles, and eComFormer that utilizes equivariant vector representations. Experimental results demonstrate the state-of-the-art predictive accuracy of ComFormer variants on various tasks across three widely-used crystal benchmarks.


![ComFormer pipeline](../../docs/ComFormer_pipline.png)

## Datasets:

- MP2018.6.1:

    The original dataset can download from [here](https://figshare.com/ndownloader/files/15087992). Following the methodology outlined in the Comformer paper, we randomly partitioned the dataset into subsets, with the specific sample sizes for each subset detailed in the table below.

    |                                   Dataset                                    | Train |  Val  | Test  |
    | :--------------------------------------------------------------------------: | :---: | :---: | :---: |
    | [mp2018_train_60k](https://paddle-org.bj.bcebos.com/paddlematerial/datasets/mp2018/mp2018_train_60k.zip) | 60000 | 5000  | 4239  |

- MP2024

    |                                   Dataset                                    | Train |  Val  | Test  |
    | :--------------------------------------------------------------------------: | :---: | :---: | :---: |
    | [mp2024_train_130k](https://paddle-org.bj.bcebos.com/paddlematerial/datasets/mp2024/mp2024_train_130k.zip) | 130000 | 10000  | 15361  |

- Jarvis

    The original dataset can download from [here](https://github.com/usnistgov/jarvis).
    | Dataset | Count |
    | :----: | :---: |
    | dft_3d | 1109 |
    | dft_2d | 75993|

- Alexandria Material Project

    | Dataset | Count |
    | :---: | :---: |
    | pbe_2d | 100000 |

## Results

<table>
    <head>
        <tr>
            <th  nowrap="nowrap">Model</th>
            <th  nowrap="nowrap">Dataset</th>
            <th  nowrap="nowrap">Property</th>
            <th  nowrap="nowrap">MAE(Val / Test dataset)</th>
            <th  nowrap="nowrap">GPUs</th>
            <th  nowrap="nowrap">Training time</th>
            <th  nowrap="nowrap">Config</th>
            <th  nowrap="nowrap">Checkpoint | Log</th>
        </tr>
    </head>
    <body>
        <tr>
            <td  nowrap="nowrap">Comformer</td>
            <td  nowrap="nowrap">mp2018_train_60k</td>
            <td  nowrap="nowrap">Form. Energy(meV/atom)</td>
            <td  nowrap="nowrap">16.4 / 18.1</td>
            <td  nowrap="nowrap">4</td>
            <td  nowrap="nowrap">~12 hours</td>
            <td  nowrap="nowrap"><a href="comformer_mp2018_train_60k_e_form.yaml">comformer_mp2018_train_60k_e_form</a></td>
            <td  nowrap="nowrap"><a href="https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/property_prediction/comformer/comformer_mp2018_train_60k_e_form.zip">checkpoint | log</a></td>
        </tr>  
        <tr>
            <td  nowrap="nowrap">Comformer</td>
            <td  nowrap="nowrap">mp2018_train_60k</td>
            <td  nowrap="nowrap">Band Gap(eV)</td>
            <td  nowrap="nowrap">0.223 / 0.209</td>
            <td  nowrap="nowrap">4</td>
            <td  nowrap="nowrap">~12 hours</td>
            <td  nowrap="nowrap"><a href="comformer_mp2018_train_60k_band_gap.yaml">comformer_mp2018_train_60k_band_gap</a></td>
            <td  nowrap="nowrap"><a href="https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/property_prediction/comformer/comformer_mp2018_train_60k_band_gap.zip">checkpoint | log</a></td>
        </tr>  
        <tr>
            <td  nowrap="nowrap">Comformer</td>
            <td  nowrap="nowrap">mp2018_train_60k</td>
            <td  nowrap="nowrap">Bulk Modulus( log(GPa) )</td>
            <td  nowrap="nowrap">0.0346 / 0.0416</td>
            <td  nowrap="nowrap">4</td>
            <td  nowrap="nowrap">~0.5 hours</td>
            <td  nowrap="nowrap"><a href="comformer_mp2018_train_60k_K.yaml">comformer_mp2018_train_60k_k</a></td>
            <td  nowrap="nowrap"><a href="https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/property_prediction/comformer/comformer_mp2018_train_60k_K.zip">checkpoint | log</a></td>
        </tr>
        <tr>
            <td  nowrap="nowrap">Comformer</td>
            <td  nowrap="nowrap">mp2018_train_60k</td>
            <td  nowrap="nowrap">Shear Modulus( log(GPa) )</td>
            <td  nowrap="nowrap">0.0615 / 0.0651</td>
            <td  nowrap="nowrap">4</td>
            <td  nowrap="nowrap">~0.5 hours</td>
            <td  nowrap="nowrap"><a href="comformer_mp2018_train_60k_G.yaml">comformer_mp2018_train_60k_G</a></td>
            <td  nowrap="nowrap"><a href="https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/property_prediction/comformer/comformer_mp2018_train_60k_G.zip">checkpoint | log</a></td>
        </tr>
        <tr>
            <td  nowrap="nowrap">Comformer</td>
            <td  nowrap="nowrap">mp2024_train_130k</td>
            <td  nowrap="nowrap">Form. Energy(meV/atom)</td>
            <td  nowrap="nowrap"> 28.475 / 28.331</td>
            <td  nowrap="nowrap">1</td>
            <td  nowrap="nowrap">~88 hours</td>
            <td  nowrap="nowrap"><a href="comformer_mp2024_train_130k_e_form.yaml">comformer_mp2024_train_130k_e_form</a></td>
            <td  nowrap="nowrap"><a href="https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/property_prediction/comformer/comformer_mp2024_train_130k_e_form.zip">checkpoint | log</a></td>
        </tr>
        <tr>
            <td  nowrap="nowrap">Comformer</td>
            <td  nowrap="nowrap">Jarvis_dft_2d</td>
            <td  nowrap="nowrap">Form. Energy(meV/atom)</td>
            <td  nowrap="nowrap"> 225.444 / 191.876</td>
            <td  nowrap="nowrap">1</td>
            <td  nowrap="nowrap">~0.2 hours</td>
            <td  nowrap="nowrap"><a href="comformer_jarvis_dft_2d_train_e_form.yaml">comformer_jarvis_dft_2d_train_e_form</a></td>
            <td  nowrap="nowrap"><a href="https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/property_prediction/comformer/comformer_jarvis_dft_2d_e_form.zip">checkpoint | log</a></td>
        </tr>
        <tr>
            <td  nowrap="nowrap">Comformer</td>
            <td  nowrap="nowrap">Jarvis_dft_3d</td>
            <td  nowrap="nowrap">Form. Energy(meV/atom)</td>
            <td  nowrap="nowrap"> 35.101 / 35.496 </td>
            <td  nowrap="nowrap">1</td>
            <td  nowrap="nowrap">~12 hours</td>
            <td  nowrap="nowrap"><a href="comformer_jarvis_dft_3d_train_e_form.yaml">comformer_jarvis_dft_3d_train_e_form</a></td>
            <td  nowrap="nowrap"><a href="https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/property_prediction/comformer/comformer_jarvis_dft_3d_e_form.zip">checkpoint | log</a></td>
        </tr>
        <tr>
            <td  nowrap="nowrap">Comformer</td>
            <td  nowrap="nowrap">Alex_pbe_2d_all</td>
            <td  nowrap="nowrap">Form. Energy(meV/atom)</td>
            <td  nowrap="nowrap"> 29.164 / 28.991  </td>
            <td  nowrap="nowrap">1</td>
            <td  nowrap="nowrap">~20 hours</td>
            <td  nowrap="nowrap"><a href="comformer_jarvis_alex_pbe_2d_train_e_form.yaml">comformer_jarvis_alex_pbe_2d_train_e_form</a></td>
            <td  nowrap="nowrap"><a href="https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/property_prediction/comformer/comformer_jarvis_alex_pbe_2d_all_e_form.zip">checkpoint | log</a></td>
        </tr>
    </body>
</table>

### Training
```bash
# formation energy per atom
# multi-gpu training, we use 4 gpus here
python -m paddle.distributed.launch --gpus="0,1,2,3" property_prediction/train.py -c property_prediction/configs/comformer/comformer_mp2018_train_60k_e_form.yaml
# single-gpu training
python property_prediction/train.py -c property_prediction/configs/comformer/comformer_mp2018_train_60k_e_form.yaml

# band gap
# multi-gpu training, we use 4 gpus here
python -m paddle.distributed.launch --gpus="0,1,2,3" property_prediction/train.py -c property_prediction/configs/comformer/comformer_mp2018_train_60k_band_gap.yaml
# single-gpu training
python property_prediction/train.py -c property_prediction/configs/comformer/comformer_mp2018_train_60k_band_gap.yaml

# bulk modulus
# multi-gpu training, we use 4 gpus here
python -m paddle.distributed.launch --gpus="0,1,2,3" property_prediction/train.py -c property_prediction/configs/comformer/comformer_mp2018_train_60k_K.yaml
# single-gpu training
python property_prediction/train.py -c property_prediction/configs/comformer/comformer_mp2018_train_60k_K.yaml

# shear modulus
# multi-gpu training, we use 4 gpus here
python -m paddle.distributed.launch --gpus="0,1,2,3" property_prediction/train.py -c property_prediction/configs/comformer/comformer_mp2018_train_60k_G.yaml
# single-gpu training
python property_prediction/train.py -c property_prediction/configs/comformer/comformer_mp2018_train_60k_G.yaml
```

### Validation
```bash
# Adjust program behavior on-the-fly using command-line parameters – this provides a convenient way to customize settings without modifying the configuration file directly.
# such as: --Global.do_eval=True

# formation energy per atom
python property_prediction/train.py -c property_prediction/configs/comformer/comformer_mp2018_train_60k_e_form.yaml Global.do_eval=True Global.do_train=False Global.do_test=False

# band gap
python property_prediction/train.py -c property_prediction/configs/comformer/comformer_mp2018_train_60k_band_gap.yaml Global.do_eval=True Global.do_train=False Global.do_test=False

# bulk modulus
python property_prediction/train.py -c property_prediction/configs/comformer/comformer_mp2018_train_60k_K.yaml Global.do_eval=True Global.do_train=False Global.do_test=False

# shear modulus
python property_prediction/train.py -c property_prediction/configs/comformer/comformer_mp2018_train_60k_G.yaml Global.do_eval=True Global.do_train=False Global.do_test=False
```

### Testing
```bash
# This command is used to evaluate the model's performance on the test dataset.

# formation energy per atom
python property_prediction/train.py -c property_prediction/configs/comformer/comformer_mp2018_train_60k_e_form.yaml Global.do_test=True Global.do_train=False Global.do_eval=False

# band gap
python property_prediction/train.py -c property_prediction/configs/comformer/comformer_mp2018_train_60k_band_gap.yaml Global.do_test=True Global.do_train=False Global.do_eval=False

# bulk modulus
python property_prediction/train.py -c property_prediction/configs/comformer/comformer_mp2018_train_60k_K.yaml Global.do_test=True Global.do_train=False Global.do_eval=False

# shear modulus
python property_prediction/train.py -c property_prediction/configs/comformer/comformer_mp2018_train_60k_G.yaml Global.do_test=True Global.do_train=False Global.do_eval=False
```

### Prediction

```bash
# This command is used to predict the properties of new crystal structures using a trained model.
# Note: The model_name and weights_name parameters are used to specify the pre-trained model and its corresponding weights. The cif_file_path parameter is used to specify the path to the CIF files for which properties need to be predicted.
# The prediction results will be saved in a CSV file specified by the save_path parameter. Default save_path is 'result.csv'.

# formation energy per atom

# Mode 1: Leverage a pre-trained machine learning model for crystal formation energy prediction. The implementation includes automated model download functionality, eliminating the need for manual configuration.
python property_prediction/predict.py --model_name='comformer_mp2018_train_60k_e_form' --weights_name='best.pdparams' --cif_file_path='./property_prediction/example_data/cifs/'

# Mode2: Use a custom configuration file and checkpoint for crystal formation energy prediction. This approach allows for more flexibility and customization.
python property_prediction/predict.py --config_path='property_prediction/configs/comformer/comformer_mp2018_train_60k_e_form.yaml' --checkpoint_path='you_checkpoint_path.pdparams' --cif_file_path='./property_prediction/example_data/cifs/'


# band gap

# Mode 1: Leverage a pre-trained machine learning model for crystal band gap prediction. The implementation includes automated model download functionality, eliminating the need for manual configuration.
python property_prediction/predict.py --model_name='comformer_mp2018_train_60k_band_gap' --weights_name='best.pdparams' --cif_file_path='./property_prediction/example_data/cifs/'

# Mode2: Use a custom configuration file and checkpoint for crystal band gap prediction. This approach allows for more flexibility and customization.
python property_prediction/predict.py --config_path='property_prediction/configs/comformer/comformer_mp2018_train_60k_band_gap.yaml' --checkpoint_path='you_checkpoint_path.pdparams' --cif_file_path='./property_prediction/example_data/cifs/'

# bulk modulus

# Mode 1: Leverage a pre-trained machine learning model for crystal bulk modulus prediction. The implementation includes automated model download functionality, eliminating the need for manual configuration.
python property_prediction/predict.py --model_name='comformer_mp2018_train_60k_K' --weights_name='best.pdparams' --cif_file_path='./property_prediction/example_data/cifs/'

# Mode2: Use a custom configuration file and checkpoint for crystal bulk modulus prediction. This approach allows for more flexibility and customization.
python property_prediction/predict.py --config_path='property_prediction/configs/comformer/comformer_mp2018_train_60k_K.yaml' --checkpoint_path='you_checkpoint_path.pdparams' --cif_file_path='./property_prediction/example_data/cifs/'


# shear modulus

# Mode 1: Leverage a pre-trained machine learning model for crystal shear modulus prediction. The implementation includes automated model download functionality, eliminating the need for manual configuration.
python property_prediction/predict.py --model_name='comformer_mp2018_train_60k_G' --weights_name='best.pdparams' --cif_file_path='./property_prediction/example_data/cifs/'

# Mode2: Use a custom configuration file and checkpoint for crystal shear modulus prediction. This approach allows for more flexibility and customization.
python property_prediction/predict.py --config_path='property_prediction/configs/comformer/comformer_mp2018_train_60k_G.yaml' --checkpoint_path='you_checkpoint_path.pdparams' --cif_file_path='./property_prediction/example_data/cifs/'
```


## Citation
```
@inproceedings{yan2024complete,
  title={Complete and Efficient Graph Transformers for Crystal Material Property Prediction},
  author={Yan, Keqiang and Fu, Cong and Qian, Xiaofeng and Qian, Xiaoning and Ji, Shuiwang},
  booktitle={International Conference on Learning Representations},
  year={2024}
}
```
