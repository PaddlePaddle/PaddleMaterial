# MatterGen

[A generative model for inorganic materials design](https://arxiv.org/abs/2309.04475)

## Abstract

The design of functional materials with desired properties is essential in driving technological advances in areas like energy storage, catalysis, and carbon capture. Generative models provide a new paradigm for materials design by directly generating novel materials given desired property constraints, but current methods have low success rate in proposing stable crystals or can only satisfy a limited set of property constraints. Here, we present MatterGen, a model that generates stable, diverse inorganic materials across the periodic table and can further be fine-tuned to steer the generation towards a broad range of property constraints. Compared to prior generative models, structures produced by MatterGen are more than twice as likely to be novel and stable, and more than 10 times closer to the local energy minimum. After fine-tuning, MatterGen successfully generates stable, novel materials with desired chemistry, symmetry, as well as mechanical, electronic and magnetic properties. As a proof of concept, we synthesize one of the generated structures and measure its property value to be within 20 % of our target. We believe that the quality of generated materials and the breadth of MatterGen’s capabilities represent a major advancement towards creating a foundational generative model for materials design.


## Results

<table>
    <head>
        <tr>
            <th  nowrap="nowrap">Model</th>
            <th  nowrap="nowrap">Dataset</th>
            <th  nowrap="nowrap">Val(loss)</th>
            <th  nowrap="nowrap">Config</th>
            <th  nowrap="nowrap">Checkpoint | Log</th>
        </tr>
    </head>
    <body>
        <tr>
            <td  nowrap="nowrap">MatterGen</td>
            <td  nowrap="nowrap">mp20</td>
            <td  nowrap="nowrap">0.3721</td>
            <td  nowrap="nowrap"><a href="mattergen_mp20.yaml">mattergen_mp20</a></td>
            <td  nowrap="nowrap"><a href="https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/structure_generation/mattergen/mattergen_mp20.zip">checkpoint | log</a></td>
        </tr>  
        <tr>
            <td  nowrap="nowrap">MatterGen</td>
            <td  nowrap="nowrap">mp20</td>
            <td  nowrap="nowrap">0.3121</td>
            <td  nowrap="nowrap"><a href="mattergen_mp20_chemical_system.yaml">mattergen_mp20_chemical_system</a></td>
            <td  nowrap="nowrap"><a href="https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/structure_generation/mattergen/mattergen_mp20_chemical_system.zip">checkpoint | log</a></td>
        </tr>  
        <tr>
            <td  nowrap="nowrap">MatterGen</td>
            <td  nowrap="nowrap">mp20</td>
            <td  nowrap="nowrap">0.3575</td>
            <td  nowrap="nowrap"><a href="mattergen_mp20_dft_band_gap.yaml">mattergen_mp20_dft_band_gap</a></td>
            <td  nowrap="nowrap"><a href="https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/structure_generation/mattergen/mattergen_mp20_dft_band_gap.zip">checkpoint | log</a></td>
        </tr>  
        <tr>
            <td  nowrap="nowrap">MatterGen</td>
            <td  nowrap="nowrap">mp20</td>
            <td  nowrap="nowrap">0.2942</td>
            <td  nowrap="nowrap"><a href="mattergen_mp20_dft_bulk_modulus.yaml">mattergen_mp20_dft_bulk_modulus</a></td>
            <td  nowrap="nowrap"><a href="https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/structure_generation/mattergen/mattergen_mp20_dft_bulk_modulus.zip">checkpoint | log</a></td>
        </tr>  
        <tr>
            <td  nowrap="nowrap">MatterGen</td>
            <td  nowrap="nowrap">mp20</td>
            <td  nowrap="nowrap">0.3620</td>
            <td  nowrap="nowrap"><a href="mattergen_mp20_dft_mag_density.yaml">mattergen_mp20_dft_mag_density</a></td>
            <td  nowrap="nowrap"><a href="https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/structure_generation/mattergen/mattergen_mp20_dft_mag_density.zip">checkpoint | log</a></td>
        </tr>  
        <tr>
            <td  nowrap="nowrap">MatterGen</td>
            <td  nowrap="nowrap">alex_mp20</td>
            <td  nowrap="nowrap">0.2960</td>
            <td  nowrap="nowrap"><a href="mattergen_alex_mp20.yaml">mattergen_alex_mp20</a></td>
            <td  nowrap="nowrap"><a href="https://paddle-org.bj.bcebos.com/paddlematerial/checkpoints/structure_generation/mattergen/mattergen_alex_mp20.zip">checkpoint | log</a></td>
        </tr>  
    </body>
</table>

### Training
```bash
# mp20 dataset, without conditional constraints
# multi-gpu training, we use 8 gpus here
python -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" structure_generation/train.py -c structure_generation/configs/mattergen/mattergen_mp20.yaml
# single-gpu training
python structure_generation/train.py -c structure_generation/configs/mattergen/mattergen_mp20.yaml

# mp20 dataset, with chemical system constraints, pre-trained model is mattergen_mp20, will be downloaded automatically
# multi-gpu training, we use 8 gpus here
python -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" structure_generation/train.py -c structure_generation/configs/mattergen/mattergen_mp20_chemical_system.yaml
# single-gpu training
python structure_generation/train.py -c structure_generation/configs/mattergen/mattergen_mp20_chemical_system.yaml

# mp20 dataset, with dft_band_gap constraints, pre-trained model is mattergen_mp20, will be downloaded automatically
# multi-gpu training, we use 8 gpus here
python -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" structure_generation/train.py -c structure_generation/configs/mattergen/mattergen_mp20_dft_band_gap.yaml
# single-gpu training
python structure_generation/train.py -c structure_generation/configs/mattergen/mattergen_mp20_dft_band_gap.yaml

# mp20 dataset, with dft_bulk_modulus constraints, pre-trained model is mattergen_mp20, will be downloaded automatically
# multi-gpu training, we use 8 gpus here
python -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" structure_generation/train.py -c structure_generation/configs/mattergen/mattergen_mp20_dft_bulk_modulus.yaml
# single-gpu training
python structure_generation/train.py -c structure_generation/configs/mattergen/mattergen_mp20_dft_bulk_modulus.yaml

# mp20 dataset, with dft_mag_density constraints, pre-trained model is mattergen_mp20, will be downloaded automatically
python -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" structure_generation/train.py -c structure_generation/configs/mattergen/mattergen_mp20_dft_mag_density.yaml
# single-gpu training
python structure_generation/train.py -c structure_generation/configs/mattergen/mattergen_mp20_dft_mag_density.yaml


# alex_mp20 dataset, without conditional constraints
# multi-gpu training, we use 8 gpus here
python -m paddle.distributed.launch --gpus="0,1,2,3,4,5,6,7" structure_generation/train.py -c structure_generation/configs/mattergen/mattergen_alex_mp20.yaml
# single-gpu training
python structure_generation/train.py -c structure_generation/configs/mattergen/mattergen_alex_mp20.yaml
```

### Validation
```bash
# Adjust program behavior on-the-fly using command-line parameters – this provides a convenient way to customize settings without modifying the configuration file directly.
# such as: --Global.do_eval=True

# mp20 dataset, without conditional constraints
python structure_generation/train.py -c structure_generation/configs/mattergen/mattergen_mp20.yaml Global.do_eval=True Global.do_train=False Global.do_test=False

# mp20 dataset, with chemical system constraints
python structure_generation/train.py -c structure_generation/configs/mattergen/mattergen_mp20_chemical_system.yaml Global.do_eval=True Global.do_train=False Global.do_test=False

# mp20 dataset, with dft_band_gap constraints
python structure_generation/train.py -c structure_generation/configs/mattergen/mattergen_mp20_dft_band_gap.yaml Global.do_eval=True Global.do_train=False Global.do_test=False

# mp20 dataset, with dft_bulk_modulus constraints
python structure_generation/train.py -c structure_generation/configs/mattergen/mattergen_mp20_dft_bulk_modulus.yaml Global.do_eval=True Global.do_train=False Global.do_test=False

# mp20 dataset, with dft_mag_density constraints
python structure_generation/train.py -c structure_generation/configs/mattergen/mattergen_mp20_dft_mag_density.yaml Global.do_eval=True Global.do_train=False Global.do_test=False

# alex_mp20 dataset, without conditional constraints
python structure_generation/train.py -c structure_generation/configs/mattergen/mattergen_alex_mp20.yaml Global.do_eval=True Global.do_train=False Global.do_test=False
```

### Testing
```bash
# This command is used to evaluate the model's performance on the test dataset.

# mp20 dataset, without conditional constraints
python structure_generation/train.py -c structure_generation/configs/mattergen/mattergen_mp20.yaml Global.do_eval=False Global.do_train=False Global.do_test=True

# mp20 dataset, with chemical system constraints
python structure_generation/train.py -c structure_generation/configs/mattergen/mattergen_mp20_chemical_system.yaml Global.do_eval=False Global.do_train=False Global.do_test=True

# mp20 dataset, with dft_band_gap constraints
python structure_generation/train.py -c structure_generation/configs/mattergen/mattergen_mp20_dft_band_gap.yaml Global.do_eval=False Global.do_train=False Global.do_test=True

# mp20 dataset, with dft_bulk_modulus constraints
python structure_generation/train.py -c structure_generation/configs/mattergen/mattergen_mp20_dft_bulk_modulus.yaml Global.do_eval=False Global.do_train=False Global.do_test=True

# mp20 dataset, with dft_mag_density constraints
python structure_generation/train.py -c structure_generation/configs/mattergen/mattergen_mp20_dft_mag_density.yaml Global.do_eval=False Global.do_train=False Global.do_test=True

# alex_mp20 dataset, without conditional constraints
python structure_generation/train.py -c structure_generation/configs/mattergen/mattergen_alex_mp20.yaml Global.do_eval=False Global.do_train=False Global.do_test=True
```

### Sample
```bash
# This command is used to predict the  crystal structure using a trained model.
# Note: The model_name and weights_name parameters are used to specify the pre-trained model and its corresponding weights. The chemical_formula parameter is used to specify the chemical formula of the crystal structure to be predicted.
# The prediction results will be saved in the folder specified by the `save_path` parameter, with the default set to `result`.

# mp20 dataset, without conditional constraints

# Mode 1: Leverage a pre-trained machine learning model for crystal structure prediction. The implementation includes automated model download functionality, eliminating the need for manual configuration.
python structure_generation/sample.py --model_name='mattergen_mp20' --weights_name='latest.pdparams' --save_path='result_mattergen_mp20/' --mode='by_num_atoms' --num_atoms=4

# or
python structure_generation/sample.py --model_name='mattergen_mp20' --weights_name='latest.pdparams' --save_path='result_mattergen_mp20/' --mode='by_dataloader'

# Mode2: Use a custom configuration file and checkpoint for crystal structure prediction. This approach allows for more flexibility and customization.

python structure_generation/sample.py --config_path='structure_generation/configs/mattergen/mattergen_mp20.yaml' --checkpoint_path='./output/mattergen_mp20/checkpoints/latest.pdparams' --save_path='result_mattergen_mp20/' --mode='by_num_atoms' --num_atoms=4

# or
python structure_generation/sample.py --config_path='structure_generation/configs/mattergen/mattergen_mp20.yaml' --checkpoint_path='./output/mattergen_mp20/checkpoints/latest.pdparams' --save_path='result_mattergen_mp20/' --mode='by_dataloader'


# mp20 dataset, with chemical system constraints

# Mode 1: Leverage a pre-trained machine learning model for crystal structure prediction. The implementation includes automated model download functionality, eliminating the need for manual configuration.
python structure_generation/sample.py --model_name='mattergen_mp20_chemical_system' --weights_name='latest.pdparams' --save_path='result_mattergen_mp20_chemical_system/' --mode='by_dataloader'

# Mode2: Use a custom configuration file and checkpoint for crystal structure prediction. This approach allows for more flexibility and customization.
python structure_generation/sample.py --config_path='structure_generation/configs/mattergen/mattergen_mp20_chemical_system.yaml' --checkpoint_path='./outpout/mattergen_mp20_chemical_system/checkpoints/latest.pdparams' --save_path='result_mattergen_mp20_chemical_system/' --mode='by_dataloader'

# mp20 dataset, with dft_band_gap constraints

# Mode 1: Leverage a pre-trained machine learning model for crystal structure prediction. The implementation includes automated model download functionality, eliminating the need for manual configuration.
python structure_generation/sample.py --model_name='mattergen_mp20_dft_band_gap' --weights_name='latest.pdparams' --save_path='result_mattergen_mp20_dft_band_gap/' --mode='by_dataloader'

# Mode2: Use a custom configuration file and checkpoint for crystal structure prediction. This approach allows for more flexibility and customization.
python structure_generation/sample.py --config_path='structure_generation/configs/mattergen/mattergen_mp20_dft_band_gap.yaml' --checkpoint_path='./outpout/mattergen_mp20_dft_band_gap/checkpoints/latest.pdparams' --save_path='result_mattergen_mp20_dft_band_gap/' --mode='by_dataloader'

# mp20 dataset, with dft_bulk_modulus constraints

# Mode 1: Leverage a pre-trained machine learning model for crystal structure prediction. The implementation includes automated model download functionality, eliminating the need for manual configuration.
python structure_generation/sample.py --model_name='mattergen_mp20_dft_bulk_modulus' --weights_name='latest.pdparams' --save_path='result_mattergen_mp20_dft_bulk_modulus/' --mode='by_dataloader'

# Mode2: Use a custom configuration file and checkpoint for crystal structure prediction. This approach allows for more flexibility and customization.
python structure_generation/sample.py --config_path='structure_generation/configs/mattergen/mattergen_mp20_dft_bulk_modulus.yaml' --checkpoint_path='./outpout/mattergen_mp20_dft_bulk_modulus/checkpoints/latest.pdparams' --save_path='result_mattergen_mp20_dft_bulk_modulus/' --mode='by_dataloader'

# mp20 dataset, with dft_mag_density constraints

# Mode 1: Leverage a pre-trained machine learning model for crystal structure prediction. The implementation includes automated model download functionality, eliminating the need for manual configuration.
python structure_generation/sample.py --model_name='mattergen_mp20_dft_mag_density' --weights_name='latest.pdparams' --save_path='result_mattergen_mp20_dft_mag_density/' --mode='by_dataloader'

# Mode2: Use a custom configuration file and checkpoint for crystal structure prediction. This approach allows for more flexibility and customization.
python structure_generation/sample.py --config_path='structure_generation/configs/mattergen/mattergen_mp20_dft_mag_density.yaml' --checkpoint_path='./outpout/mattergen_mp20_dft_mag_density/checkpoints/latest.pdparams' --save_path='result_mattergen_mp20_dft_mag_density/' --mode='by_dataloader'

# alex_mp20 dataset, without conditional constraints

# Mode 1: Leverage a pre-trained machine learning model for crystal structure prediction. The implementation includes automated model download functionality, eliminating the need for manual configuration.
python structure_generation/sample.py --model_name='mattergen_alex_mp20' --weights_name='latest.pdparams' --save_path='result_mattergen_alex_mp20/' --mode='by_dataloader'

# Mode2: Use a custom configuration file and checkpoint for crystal structure prediction. This approach allows for more flexibility and customization.
python structure_generation/sample.py --config_path='structure_generation/configs/mattergen/mattergen_alex_mp20.yaml' --checkpoint_path='./outpout/mattergen_alex_mp20/checkpoints/latest.pdparams' --save_path='result_mattergen_alex_mp20/' --mode='by_dataloader'
```

## Citation
```
@article{zeni2025generative,
  title={A generative model for inorganic materials design},
  author={Zeni, Claudio and Pinsler, Robert and Z{\"u}gner, Daniel and Fowler, Andrew and Horton, Matthew and Fu, Xiang and Wang, Zilong and Shysheya, Aliaksandra and Crabb{\'e}, Jonathan and Ueda, Shoko and others},
  journal={Nature},
  pages={1--3},
  year={2025},
  publisher={Nature Publishing Group UK London}
}
```
