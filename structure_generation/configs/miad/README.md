# MiAD

MiAD (Mirage Atom Diffusion) is a diffusion-based framework for de novo crystal generation. It introduces the concept of Mirage Infusion, a mechanism that allows diffusion models to dynamically adjust the number of atoms in a crystal structure during the generation trajectory  By treating a variable number of atoms as "mirage" atoms (sentinel states), MiAD achieves state-of-the-art performance in generating stable, unique, and novel (S.U.N.) materials.

## How It Works

1. Mirage Atoms: The model uses sentinel nodes called "mirage atoms" (assigned atom_type = 0) that can be transformed into real chemical elements or discarded as empty space during the diffusion trajectory .

2. Training Phase: Real crystals are padded with mirage atoms up to a maximum limit (e.g., 25 atoms), where mirage nodes have random fractional coordinates and type 0 .

3. Generation Phase: The model initializes all crystals with the maximum atom count and predicts which nodes should materialize into real atoms versus remaining as mirage atoms .

4. Post-Processing: Remaining mirage atoms are stripped before exporting to final CIF files .

## Architecture

MiAD uses DiffCSP as its backbone architecture, with the CrystalGen orchestrator handling the diffusion process for lattice parameters, fractional coordinates, and atom types.

## Dataset

[Original dataset address](https://drive.google.com/file/d/1BLI3VtvzfIIXlH6UHQ4o-gQaCIOZ1UR7/view?usp=sharing)

[Mirror AIStudio address](https://aistudio.baidu.com/modelsdetail/48578/intro)

Extract the data to `./data/mp_20/` so that the CSV files are at `./data/mp_20/train.csv`, `./data/mp_20/val.csv`, and `./data/mp_20/test.csv`.

## Environment Dependencies

- Python >= 3.10
- PaddlePaddle >= 3.3.0 (official release)
- paddle_scatter >= 2.1.2
- numpy, pymatgen, omegaconf

## Checkpoints

[Original checkpoints address](https://drive.google.com/file/d/1KyD6KzvjYFPfU8lutFyO_0b8EbeHSGqf/view?usp=sharing)

[Mirror AIStudio address](https://aistudio.baidu.com/modelsdetail/48578/intro)

[Paddle checkpoints address](https://aistudio.baidu.com/modelsdetail/48638/intro)



## Commands

### Training

```bash
# single GPU
python structure_generation/train.py -c structure_generation/configs/miad/miad_mp20.yaml

# multi-GPU (example: 4 GPUs)
python -m paddle.distributed.launch --gpus="0,1,2,3" structure_generation/train.py -c structure_generation/configs/miad/miad_mp20.yaml
```

### Validation

```bash
python structure_generation/train.py -c structure_generation/configs/miad/miad_mp20.yaml Global.do_eval=True Global.do_train=False Global.do_test=False Trainer.pretrained_model_path='path/to/model.pdparams'
```

### Sampling (Structure Generation)

```bash
# Option 1: Use pre-trained model (auto-download)
python structure_generation/sample.py --model_name='miad_mp20' --weights_name='miad_mp20.pdparams' --save_path='result_miad/' --mode='by_num_atoms' --num_atoms=20

# Option 2: Custom checkpoint
python structure_generation/sample.py --config_path='structure_generation/configs/miad/miad_mp20.yaml' --checkpoint_path='./output/miad_mp20/checkpoints/latest.pdparams' --save_path='result_miad/' --mode='by_dataloader'
```

### Evaluation (Compute Metrics)

```bash
# Evaluate generated structures against ground truth using CSPMetric
python structure_generation/sample.py --model_name='miad_mp20' --weights_name='latest.pdparams' --save_path='result_eval/' --mode='compute_metric'
```