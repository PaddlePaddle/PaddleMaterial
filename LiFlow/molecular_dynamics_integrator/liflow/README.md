# LiFlow

LiFlow is a flow-matching model for accelerated atomic transport simulation. This integration keeps the PaddleMaterials model, dataset, trainer, predictor, and registry interfaces.

## Layout

- `ppmat/models/liflow/layers.py`: basis, cutoff, message, update, and equivariant output layers.
- `ppmat/models/liflow/dual_painn.py`: DualPaiNN network.
- `ppmat/models/liflow/flow_module.py`: propagator/corrector flow targets and trainer-compatible loss.
- `ppmat/datasets/liflow_dataset.py`: trajectory-pair dataset and deterministic validation sampling.
- `ppmat/predictor/integrator_predictor.py`: local/registered model loading and trajectory API.
- `ppmat/metrics/liflow_metrics.py`: MSD, RDF, final-step, and relative-error helpers.

## Training

Propagator:

```bash
python molecular_dynamics_integrator/train.py -c molecular_dynamics_integrator/liflow/configs/liflow_universal_propagator.yaml
```

Corrector:

```bash
python molecular_dynamics_integrator/train.py -c molecular_dynamics_integrator/liflow/configs/liflow_universal_corrector.yaml
```

Both configurations use the shared `BaseTrainer`. The propagator predicts the source-to-target velocity field; the corrector uses the data displacement target.

## Local inference

```bash
python molecular_dynamics_integrator/predict.py \
  --config_path molecular_dynamics_integrator/liflow/configs/liflow_universal_inference.yaml \
  --checkpoint_path path/to/propagator.pdparams \
  --data_path path/to/liflow \
  --index_file test_800K.csv \
  --steps 25 --flow_steps 10 --solver euler --output trajectory.npy
```

The trajectory output has shape `[steps + 1, num_atoms, 3]`. `--solver heun` is also supported.

## Registered inference

```bash
python molecular_dynamics_integrator/predict.py --model_name liflow_universal
```

The registry package must contain a model YAML and the corresponding Paddle checkpoint. The registry loader resolves the downloaded package directly and does not append a model-name directory unconditionally.

## Reproducible assets and data

The local model package is built and verified by:

```bash
python molecular_dynamics_integrator/liflow/tools/package_artifacts.py \
  --propagator path/to/propagator.pdparams \
  --corrector path/to/corrector.pdparams \
  --out-dir path/to/artifacts
```

The archive layout is `liflow_universal/` with the inference YAML and the
Propagator/Corrector checkpoints under `checkpoints/`. The current local
archive SHA256 is
`d2bc3fded05bc2aa63b06240c107e4b0601a87b1a83fc0b3b20a216eadda00f7`.
A stable public download URL has not yet been assigned; the manifest records
per-entry SHA256 values and the package script performs a fresh-extract
round-trip check.

The trajectory dataset is published at
<https://zenodo.org/records/14889658> (DOI:
<https://doi.org/10.5281/zenodo.14889658>). Extract it so that `path` contains
`element_index.npy`, `atomic_numbers.npy`, `lattice.npy`,
`positions_{temp}K.npz`, and the train/test CSV files. The checked-in mini
fixture is only for schema, graph, determinism, and pipeline tests; it must not
be used for scientific checkpoint or sampling validation.

## Evaluation and checks

```bash
python molecular_dynamics_integrator/evaluate.py --reference reference.npy --prediction trajectory.npy
bash molecular_dynamics_integrator/liflow/ci_smoke.sh
```

`evaluate.py` returns a non-zero status when a reported sampling error exceeds 5%. Weight conversion is explicit and records a SHA256 digest:

```bash
python molecular_dynamics_integrator/liflow/convert_weights.py source.pdparams destination.pdparams
```

## Known limitations

- CINN performance validation is not available on the current Windows installation. The official Windows CUDA-enabled Paddle wheel reports `paddle.base.is_compiled_with_cinn() == False`, so this package does not claim CINN acceleration or a compiler speedup.
- The official LPS trajectory dataset was not available from a public download address during this release. LPS checkpoint conversion and audit artifacts are included under `test/fixtures/liflow/reference_outputs/special_ckpt`, but LPS end-to-end data validation is intentionally omitted. Universal and LGPS validation use their real datasets and checkpoints; they are not substitutes for LPS data.
- The archive contains source code, configuration, deterministic test fixtures, converted checkpoint audit artifacts, and reproducibility reports. Large external trajectory data and the original PyTorch checkpoints are not duplicated in this source archive. Follow the dataset and checkpoint instructions above to obtain them from their original locations.

The reference implementation is described in the LiFlow paper: <https://www.nature.com/articles/s42256-025-01125-4>.
