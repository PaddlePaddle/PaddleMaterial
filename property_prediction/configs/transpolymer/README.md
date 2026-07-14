# TransPolymer

[TransPolymer: a Transformer-based language model for polymer property predictions](https://arxiv.org/abs/2209.01307)

## Abstract

TransPolymer is a RoBERTa-style language model designed for polymer sequence
representation learning and downstream polymer property prediction. It tokenizes
polymer SMILES strings with a chemistry-aware tokenizer, encodes them with a
pretrained Transformer backbone, and finetunes a regression head for target
polymer properties.

![TransPolymer Pipeline](../../../pipeline.png)

Reference implementation: https://github.com/ChangwenXu98/TransPolymer

## Datasets:

The TransPolymer downstream benchmarks cover polymer electronic, optical,
photovoltaic, and polymer electrolyte properties. The PE-I task uses the
provided train/test split, while the other downstream tasks are reported with
5-fold cross validation.

| Dataset | Samples | SMILES column | Target column | Config |
| :---: | :---: | :---: | :---: | :---: |
| Eea | 368 | smiles | value | [transpolymer_eea_finetune.yaml](transpolymer_eea_finetune.yaml) |
| Egb | 561 | smiles | value | [transpolymer_egb_finetune.yaml](transpolymer_egb_finetune.yaml) |
| Egc | 3380 | smiles | value | [transpolymer_egc_finetune.yaml](transpolymer_egc_finetune.yaml) |
| Ei | 370 | smiles | value | [transpolymer_ei_finetune.yaml](transpolymer_ei_finetune.yaml) |
| EPS | 382 | smiles | value | [transpolymer_eps_finetune.yaml](transpolymer_eps_finetune.yaml) |
| Nc | 382 | smiles | value | [transpolymer_nc_finetune.yaml](transpolymer_nc_finetune.yaml) |
| Xc | 432 | smiles | value | [transpolymer_xc_finetune.yaml](transpolymer_xc_finetune.yaml) |
| OPV | 1203 | CSMILES | PCE_ave | [transpolymer_opv_finetune.yaml](transpolymer_opv_finetune.yaml) |
| PE-I | 34803 / 146 | smiles | Conductivity [S/cm] | [transpolymer_pe_i_finetune.yaml](transpolymer_pe_i_finetune.yaml) |
| PE-II | 271 | SMILES descriptor 1 | logCond60 | [transpolymer_pe_ii_finetune.yaml](transpolymer_pe_ii_finetune.yaml) |

The downstream datasets are provided in one artifact:

| Artifact | Link | Extraction code |
| :---: | :---: | :---: |
| Datasets | [download](https://pan.baidu.com/s/12Y6iJXOXzff5AQzJhVlzOg) | 1227 |

Download `transpolymer_pretrain_and_data.zip` and extract the `data/` directory
to the PaddleMaterials root directory. After extraction, the dataset directory
structure should be:

```text
PaddleMaterials/
|-- data/
|   |-- Eea.csv
|   |-- Egb.csv
|   |-- Egc.csv
|   |-- Ei.csv
|   |-- EPS.csv
|   |-- Nc.csv
|   |-- Xc.csv
|   |-- OPV.csv
|   |-- PE_II.csv
|   |-- train_PE_I.csv
|   |-- test_PE_I.csv
|   `-- vocab/
|       |-- vocab_sup_OPV.csv
|       |-- vocab_sup_PE_I.csv
|       `-- vocab_sup_PE_II.csv
```

The converted Paddle pretrained checkpoint is hosted in the model repository:

```text
https://git.aistudio.baidu.com/TransPolymer/TransPolymer123
```

The training configs set `pretrained_model_path=./ckpt/pretrain.pt` and
`pretrained_model_url` to the Git LFS clone URL. If
`./ckpt/pretrain.pt/config.json` and
`./ckpt/pretrain.pt/model_state.pdparams` are missing, the model will clone the
repository and prepare the checkpoint automatically before training. To download
it manually:

```bash
git lfs install
git clone https://28bf65435bc4c13f5b89a153488f09972c18f7f4@git.aistudio.baidu.com/TransPolymer/TransPolymer123.git /tmp/TransPolymer123
mkdir -p ./ckpt/pretrain.pt
cp /tmp/TransPolymer123/config.json ./ckpt/pretrain.pt/
cp /tmp/TransPolymer123/model_state.pdparams ./ckpt/pretrain.pt/
```

The expected local checkpoint layout is:

```text
PaddleMaterials/
`-- ckpt/
    `-- pretrain.pt/
        |-- config.json
        `-- model_state.pdparams
```

The training configs use relative paths under `./data`, `./data/vocab`, and
`./ckpt/pretrain.pt`.

The complete downstream finetuned weights, training logs, finetune configs, and
summary file are provided separately:

| Artifact | Datasets | Link | Extraction code |
| :---: | :---: | :---: | :---: |
| Downstream outputs part 1 | Eea, Egb, Egc | [download](https://pan.baidu.com/s/1ogiNbW7KBamiok59fLQTkA) | 1227 |
| Downstream outputs part 2a | Ei, EPS | [download](https://pan.baidu.com/s/16AyhP-WmQYGRRhALNvpFPA) | 1227 |
| Downstream outputs part 2b | Nc, Xc | [download](https://pan.baidu.com/s/1dq7zbf54NpM74CAKuCc2Aw) | 1227 |
| Downstream outputs part 3 | OPV, PE-II | [download](https://pan.baidu.com/s/1zLA5xbm6xoEFMOyN679dxQ) | 1227 |
| Downstream outputs part 4 | PE-I, summary | [download](https://pan.baidu.com/s/14_2lIVvU5v2HlYOJZx9M9w) | 1227 |

Each downstream output archive keeps the `output/` prefix. Extract the required
parts to the PaddleMaterials root directory to obtain:

```text
PaddleMaterials/
`-- output/
    |-- Eea/
    |   |-- Eea_best_model.pdparams
    |   |-- Eea_train.pdparams
    |   |-- Eea.log
    |   `-- config_finetune.Eea.yaml
    |-- Egb/
    |-- Egc/
    |-- Ei/
    |-- EPS/
    |-- Nc/
    |-- Xc/
    |-- OPV/
    |-- PE_I/
    |-- PE_II/
    `-- summary_all_9.tsv
```

## Model

TransPolymer uses a polymer-specific SMILES tokenizer and a RoBERTa-style
Transformer encoder. The converted Paddle pretraining checkpoint is loaded as the
backbone, and a lightweight regression head is finetuned for each downstream
property prediction task. Tokenization and model input construction are handled
inside the model, while the dataset only loads raw SMILES strings and property
labels.

## Configuration

Key options are defined in the `transpolymer_*_finetune.yaml` files:

- `Dataset.*.dataset.__init_params__.path`: downstream CSV file path.
- `Dataset.*.dataset.__init_params__.smiles_key`: SMILES column name.
- `Model.__init_params__.blocksize`: maximum sequence length after tokenization.
- `Model.__init_params__.vocab_sup_file`: supplementary vocabulary file, used by OPV, PE-I, and PE-II.
- `Model.__init_params__.pretrained_model_path`: converted Paddle checkpoint directory.
- `Model.__init_params__.pretrained_model_url`: Git LFS model repository used to automatically prepare the pretrained checkpoint when missing.
- `Optimizer.lr.__init_params__.learning_rate`: finetuning learning rate.
- `Trainer.max_epochs`: maximum finetuning epochs.

## Results

<table>
    <head>
        <tr>
            <th nowrap="nowrap">Model Name</th>
            <th nowrap="nowrap">Dataset</th>
            <th nowrap="nowrap">Property</th>
            <th nowrap="nowrap">RMSE / R2</th>
            <th nowrap="nowrap">Std RMSE / Std R2</th>
            <th nowrap="nowrap">Setting</th>
            <th nowrap="nowrap">GPUs</th>
            <th nowrap="nowrap">Training time</th>
            <th nowrap="nowrap">Config</th>
            <th nowrap="nowrap">Checkpoint | Log</th>
        </tr>
    </head>
    <body>
        <tr>
            <td nowrap="nowrap">transpolymer_pe_i_finetune</td>
            <td nowrap="nowrap">PE-I</td>
            <td nowrap="nowrap">Conductivity [S/cm]</td>
            <td nowrap="nowrap">0.8993 / 0.4508</td>
            <td nowrap="nowrap">- / -</td>
            <td nowrap="nowrap">Fixed split</td>
            <td nowrap="nowrap">1</td>
            <td nowrap="nowrap">~3 hours</td>
            <td nowrap="nowrap"><a href="transpolymer_pe_i_finetune.yaml">transpolymer_pe_i_finetune</a></td>
            <td nowrap="nowrap"><a href="https://pan.baidu.com/s/14_2lIVvU5v2HlYOJZx9M9w">checkpoint | log</a> (code: 1227)</td>
        </tr>
        <tr>
            <td nowrap="nowrap">transpolymer_eea_finetune</td>
            <td nowrap="nowrap">Eea</td>
            <td nowrap="nowrap">value</td>
            <td nowrap="nowrap">0.3307 / 0.9000</td>
            <td nowrap="nowrap">0.0511 / 0.0351</td>
            <td nowrap="nowrap">5-fold CV</td>
            <td nowrap="nowrap">1</td>
            <td nowrap="nowrap">see log</td>
            <td nowrap="nowrap"><a href="transpolymer_eea_finetune.yaml">transpolymer_eea_finetune</a></td>
            <td nowrap="nowrap"><a href="https://pan.baidu.com/s/1ogiNbW7KBamiok59fLQTkA">checkpoint | log</a> (code: 1227)</td>
        </tr>
        <tr>
            <td nowrap="nowrap">transpolymer_egb_finetune</td>
            <td nowrap="nowrap">Egb</td>
            <td nowrap="nowrap">value</td>
            <td nowrap="nowrap">0.6188 / 0.8978</td>
            <td nowrap="nowrap">0.0582 / 0.0158</td>
            <td nowrap="nowrap">5-fold CV</td>
            <td nowrap="nowrap">1</td>
            <td nowrap="nowrap">see log</td>
            <td nowrap="nowrap"><a href="transpolymer_egb_finetune.yaml">transpolymer_egb_finetune</a></td>
            <td nowrap="nowrap"><a href="https://pan.baidu.com/s/1ogiNbW7KBamiok59fLQTkA">checkpoint | log</a> (code: 1227)</td>
        </tr>
        <tr>
            <td nowrap="nowrap">transpolymer_egc_finetune</td>
            <td nowrap="nowrap">Egc</td>
            <td nowrap="nowrap">value</td>
            <td nowrap="nowrap">0.4746 / 0.9074</td>
            <td nowrap="nowrap">0.0258 / 0.0105</td>
            <td nowrap="nowrap">5-fold CV</td>
            <td nowrap="nowrap">1</td>
            <td nowrap="nowrap">see log</td>
            <td nowrap="nowrap"><a href="transpolymer_egc_finetune.yaml">transpolymer_egc_finetune</a></td>
            <td nowrap="nowrap"><a href="https://pan.baidu.com/s/1ogiNbW7KBamiok59fLQTkA">checkpoint | log</a> (code: 1227)</td>
        </tr>
        <tr>
            <td nowrap="nowrap">transpolymer_ei_finetune</td>
            <td nowrap="nowrap">Ei</td>
            <td nowrap="nowrap">value</td>
            <td nowrap="nowrap">0.4298 / 0.8015</td>
            <td nowrap="nowrap">0.0564 / 0.0609</td>
            <td nowrap="nowrap">5-fold CV</td>
            <td nowrap="nowrap">1</td>
            <td nowrap="nowrap">see log</td>
            <td nowrap="nowrap"><a href="transpolymer_ei_finetune.yaml">transpolymer_ei_finetune</a></td>
            <td nowrap="nowrap"><a href="https://pan.baidu.com/s/16AyhP-WmQYGRRhALNvpFPA">checkpoint | log</a> (code: 1227)</td>
        </tr>
        <tr>
            <td nowrap="nowrap">transpolymer_eps_finetune</td>
            <td nowrap="nowrap">EPS</td>
            <td nowrap="nowrap">value</td>
            <td nowrap="nowrap">0.5714 / 0.7153</td>
            <td nowrap="nowrap">0.0798 / 0.1114</td>
            <td nowrap="nowrap">5-fold CV</td>
            <td nowrap="nowrap">1</td>
            <td nowrap="nowrap">see log</td>
            <td nowrap="nowrap"><a href="transpolymer_eps_finetune.yaml">transpolymer_eps_finetune</a></td>
            <td nowrap="nowrap"><a href="https://pan.baidu.com/s/16AyhP-WmQYGRRhALNvpFPA">checkpoint | log</a> (code: 1227)</td>
        </tr>
        <tr>
            <td nowrap="nowrap">transpolymer_nc_finetune</td>
            <td nowrap="nowrap">Nc</td>
            <td nowrap="nowrap">value</td>
            <td nowrap="nowrap">0.1047 / 0.7951</td>
            <td nowrap="nowrap">0.0197 / 0.0824</td>
            <td nowrap="nowrap">5-fold CV</td>
            <td nowrap="nowrap">1</td>
            <td nowrap="nowrap">see log</td>
            <td nowrap="nowrap"><a href="transpolymer_nc_finetune.yaml">transpolymer_nc_finetune</a></td>
            <td nowrap="nowrap"><a href="https://pan.baidu.com/s/1dq7zbf54NpM74CAKuCc2Aw">checkpoint | log</a> (code: 1227)</td>
        </tr>
        <tr>
            <td nowrap="nowrap">transpolymer_xc_finetune</td>
            <td nowrap="nowrap">Xc</td>
            <td nowrap="nowrap">value</td>
            <td nowrap="nowrap">18.0286 / 0.4124</td>
            <td nowrap="nowrap">0.8551 / 0.0556</td>
            <td nowrap="nowrap">5-fold CV</td>
            <td nowrap="nowrap">1</td>
            <td nowrap="nowrap">see log</td>
            <td nowrap="nowrap"><a href="transpolymer_xc_finetune.yaml">transpolymer_xc_finetune</a></td>
            <td nowrap="nowrap"><a href="https://pan.baidu.com/s/1dq7zbf54NpM74CAKuCc2Aw">checkpoint | log</a> (code: 1227)</td>
        </tr>
        <tr>
            <td nowrap="nowrap">transpolymer_opv_finetune</td>
            <td nowrap="nowrap">OPV</td>
            <td nowrap="nowrap">PCE_ave</td>
            <td nowrap="nowrap">1.9285 / 0.3176</td>
            <td nowrap="nowrap">0.0759 / 0.0568</td>
            <td nowrap="nowrap">5-fold CV</td>
            <td nowrap="nowrap">1</td>
            <td nowrap="nowrap">see log</td>
            <td nowrap="nowrap"><a href="transpolymer_opv_finetune.yaml">transpolymer_opv_finetune</a></td>
            <td nowrap="nowrap"><a href="https://pan.baidu.com/s/1zLA5xbm6xoEFMOyN679dxQ">checkpoint | log</a> (code: 1227)</td>
        </tr>
        <tr>
            <td nowrap="nowrap">transpolymer_pe_ii_finetune</td>
            <td nowrap="nowrap">PE-II</td>
            <td nowrap="nowrap">logCond60</td>
            <td nowrap="nowrap">0.7478 / 0.5961</td>
            <td nowrap="nowrap">0.0731 / 0.0665</td>
            <td nowrap="nowrap">5-fold CV</td>
            <td nowrap="nowrap">1</td>
            <td nowrap="nowrap">see log</td>
            <td nowrap="nowrap"><a href="transpolymer_pe_ii_finetune.yaml">transpolymer_pe_ii_finetune</a></td>
            <td nowrap="nowrap"><a href="https://pan.baidu.com/s/1zLA5xbm6xoEFMOyN679dxQ">checkpoint | log</a> (code: 1227)</td>
        </tr>
    </body>
</table>

### Training

Run from PaddleMaterials root:

```bash
python property_prediction/train.py \
  -c property_prediction/configs/transpolymer/transpolymer_pe_i_finetune.yaml
```

Replace `transpolymer_pe_i_finetune.yaml` with any other
`transpolymer_*_finetune.yaml` file to finetune a different downstream task.

### Validation

Evaluate a finetuned checkpoint on the validation split:

```bash
python property_prediction/train.py \
  -c property_prediction/configs/transpolymer/transpolymer_pe_i_finetune.yaml \
  Global.do_train=False \
  Global.do_eval=True \
  Trainer.pretrained_model_path=./output/PE_I \
  Trainer.pretrained_weight_name=PE_I_best_model.pdparams
```

### Testing

Evaluate a finetuned checkpoint on the test split:

```bash
python property_prediction/train.py \
  -c property_prediction/configs/transpolymer/transpolymer_pe_i_finetune.yaml \
  Global.do_train=False \
  Global.do_test=True \
  Trainer.pretrained_model_path=./output/PE_I \
  Trainer.pretrained_weight_name=PE_I_best_model.pdparams
```

For other tasks, replace the config path and checkpoint name with the
corresponding dataset checkpoint, for example
`Trainer.pretrained_model_path=./output/Eea` and
`Trainer.pretrained_weight_name=Eea_best_model.pdparams`.

### Prediction

Run from PaddleMaterials root:

```bash
python property_prediction/predict.py \
  --config_path property_prediction/configs/transpolymer/transpolymer_pe_i_finetune.yaml \
  --checkpoint_path ./output/PE_I/PE_I_best_model.pdparams \
  --csv_file_path ./property_prediction/example_data/transpolymer_predict.csv \
  --save_path ./output/transpolymer_prediction.csv
```

The input CSV should contain the SMILES column required by the selected config,
for example `smiles`, `CSMILES`, or `SMILES descriptor 1`. The prediction result
is saved to the path specified by `--save_path`.

## Citation

```bibtex
@article{xu2022transpolymer,
  title={TransPolymer: a Transformer-based language model for polymer property predictions},
  author={Xu, Changwen and Wang, Yuyang and Farimani, Amir Barati},
  journal={arXiv preprint arXiv:2209.01307},
  year={2022}
}
```
