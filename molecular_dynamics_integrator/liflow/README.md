# LiFlow

[Flow Matching for Accelerated Simulation of Atomic Transport in Crystalline Materials](https://www.nature.com/articles/s42256-025-01125-4) (Nature Machine Intelligence, 2025)

## Abstract

LiFlow is a flow-matching framework for accelerating molecular dynamics simulations of crystalline materials. It learns atomic displacement fields from reference trajectories and uses two complementary components:

- The **propagator** advances atomic configurations over a coarse simulation interval.
- The **corrector** refines the propagated configuration with a noise-conditioned flow.

The model conditions the atomic velocity field on element types, periodic positions, simulation time, and temperature. This design targets long-timescale atomic transport while preserving the structural statistics of molecular dynamics trajectories.

In PaddleMaterials, the LiFlow implementation is organized as follows:

- `ppmat/datasets/liflow_dataset.py`: trajectory-pair dataset and structure cache.
- `ppmat/models/liflow/liflow.py`: Paddle LiFlow model with `_forward`, `forward`, and `predict` interfaces.
- `ppmat/predictor/integrator_predictor.py`: `IntegratorPredictor` based on `BasePredictor`.
- `molecular_dynamics_integrator/train.py`: shared PaddleMaterials training entry.
- `molecular_dynamics_integrator/predict.py`: shared integrator prediction entry.

## Dataset

LiFlow consumes preprocessed molecular dynamics trajectory data. Each dataset directory contains atomic species, lattice, position trajectories, and train/test trajectory-index CSV files. The dataset loader samples two frames from each trajectory and constructs the model inputs, prior noise, and displacement target. Dataset files are distributed separately from the source code.

## Training

```bash
python -m paddle.distributed.launch --gpus="0,1,2,3" \
  molecular_dynamics_integrator/train.py \
  -c molecular_dynamics_integrator/liflow/configs/liflow_universal.yaml
```

For a single device:

```bash
python molecular_dynamics_integrator/train.py \
  -c molecular_dynamics_integrator/liflow/configs/liflow_universal.yaml
```

## Evaluation

```bash
python molecular_dynamics_integrator/train.py \
  -c molecular_dynamics_integrator/liflow/configs/liflow_universal.yaml \
  Global.do_eval=True Global.do_train=False Global.do_test=False \
  Trainer.pretrained_model_path=path/to/model.pdparams
```

The configured evaluation metric is the mean squared error between the predicted and reference displacement fields. The command above reports the evaluation value when run with the released checkpoint and dataset.

The released checkpoint and logs are available at:

`https://paddle-org.bj.bcebos.com/paddlematerials/checkpoints/molecular_dynamics_integrator/liflow/liflow_universal.zip`

## Prediction

Use a local configuration and checkpoint:

```bash
python molecular_dynamics_integrator/predict.py \
  --config_path molecular_dynamics_integrator/liflow/configs/liflow_universal.yaml \
  --checkpoint_path path/to/model.pdparams \
  --data_path path/to/liflow \
  --index_file test_800K.csv
```

A registered checkpoint can be used after `liflow_universal` is added to the model registry:

```bash
python molecular_dynamics_integrator/predict.py \
  --model_name liflow_universal \
  --data_path path/to/liflow \
  --index_file test_800K.csv
```

## Citation

```bibtex
@article{nam2025flow,
  title={Flow Matching for Accelerated Simulation of Atomic Transport in Crystalline Materials},
  author={Juno Nam and Sulin Liu and Gavin Winter and KyuJung Jun and Soojung Yang and Rafael G{\'o}mez-Bombarelli},
  journal={Nature Machine Intelligence},
  year={2025},
  doi={10.1038/s42256-025-01125-4},
}
```
