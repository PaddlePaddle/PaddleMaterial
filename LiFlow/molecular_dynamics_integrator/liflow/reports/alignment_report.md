# LiFlow Weight Conversion & Audit Report

Reference frozen commit: `learningmatter-mit/liflow` @ `e6fc475361d046865f12cae1aee11c4f56c48d87`
（PaddleMaterials develop `a9a689cf64d98a2b415a15f794fe88598b5943ef`）

## Converter

`molecular_dynamics_integrator/liflow/convert_weights.py`

```bash
# torch environment (liflow-torch)
python convert_weights.py dump-torch --source ckpt/P_universal.ckpt --out-dir ckpt/converted/P_universal
python convert_weights.py dump-torch --source ckpt/C_universal.ckpt --out-dir ckpt/converted/C_universal

# paddle environment (ppmat-liflow)
python convert_weights.py build-paddle --meta-dir ckpt/converted/P_universal \
    --output ckpt/converted/P_universal/propagator.pdparams --audit ckpt/converted/P_universal/audit.json
python convert_weights.py build-paddle --meta-dir ckpt/converted/C_universal \
    --output ckpt/converted/C_universal/corrector.pdparams --audit ckpt/converted/C_universal/audit.json
```

Conversion rules:
- strip the Lightning `model.` prefix from every state key;
- Paddle Linear weights are stored `[in, out]` while torch stores `[out, in]`, so
  every 2-D weight is transposed **except** `atom_embedding.weight` (shared layout);
- 1-D biases and fixed buffers (`*freqs`, `prefactor`, `r_max`) pass through;
- any missing / unexpected / shape-mismatched key aborts the build.

## Audit results

| item | Propagator (P_universal) | Corrector (C_universal) |
|---|---|---|
| source file | `ckpt/P_universal.ckpt` | `ckpt/C_universal.ckpt` |
| source SHA256 | `41c1763f56ec1666bc7bbf2a9a17971633bb1e59d4ea4a75cbf0cf999dd50f2d` | `a38418094109e250f31cfd4860da485403c0a39b5968a7733f94ab925f557088` |
| torch top-level keys | `epoch, global_step, pytorch-lightning_version, state_dict, loops, callbacks, optimizer_states, lr_schedulers, hparams_name, hyper_parameters` | same |
| state-dict entries | 45 | 45 |
| model hyper-params | F=64, R=20, L=3, elements=77, r_max=5.0, r_offset=0.5, ref_temp=1000.0, velocity | same |
| output file | `ckpt/converted/P_universal/propagator.pdparams` | `ckpt/converted/C_universal/corrector.pdparams` |
| output SHA256 | `3373e5670965da27bf521b0071d56ac1eb402fdbb6e338b9749eaa0e1106a953` | `6354d99cab9f8b128751d5276aaf0777f5a75ae2413ef757e040244499a2c25b` |
| parameters loaded | 45 | 45 |
| fixed buffers | 5 | 5 |
| transposed Linear weights | 22 | 22 |
| **missing_keys** | `[]` | `[]` |
| **unexpected_keys** | `[]` | `[]` |
| **shape_mismatch** | `[]` | `[]` |
| reload-back check | clean | clean |

Full key mapping (`model.* -> *`) and the transposed-key list are in
`ckpt/converted/{P,C}_universal/audit.json`.

## Task 14: two-epoch training alignment

Reference goldens were generated in `liflow-torch` from the frozen reference commit:

```powershell
& C:\Users\YU\miniconda3\envs\liflow-torch\python.exe scripts/golden_export.py --mode train --prediction-mode velocity --out-dir D:\实验\PaddleMaterials_upstream\test\fixtures\liflow\reference_outputs
& C:\Users\YU\miniconda3\envs\liflow-torch\python.exe scripts/golden_export.py --mode train --prediction-mode data --out-dir D:\实验\PaddleMaterials_upstream\test\fixtures\liflow\reference_outputs
```

The test uses the frozen mini weights, the three fixed fixtures in the order
`14atoms`, `nonperiodic`, `cutoff_edge`, the six frozen flow times, seed 0, and
Adam with `lr=1e-3`. It checks all six per-step losses, epoch-2 parameters, and
checkpoint resume after step 3 against continuous six-step training for both
`velocity` and `data`.

Measured reference loss curves:

- velocity: `[0.0600811504, 0.2549899518, 0.2740780115, 0.0390504524, 0.2112659216, 0.0692918524]`
- data: `[1.4627078772, 0.5062361956, 0.9637148380, 1.3650277853, 0.4266535044, 1.7089003325]`

Paddle verification command (environment `ppmat-liflow`):

```powershell
python -m pytest test/test_liflow_align.py test/test_liflow_integration.py -q
```

Result: `13 passed`. Forward max absolute errors remain `6.659e-08`,
`2.980e-08`, `1.239e-07`; backward max absolute errors are `3.725e-08`,
`1.192e-07`, `1.192e-07` for the three fixtures. Training loss and all
parameter/checkpoint comparisons pass at `rtol=2e-5, atol=2e-5`; no observed
failure or drift exceeded that tolerance.

## Still open

- Real-forward numerical verification of `propagator.pdparams`/`corrector.pdparams`
  against torch requires the official `element_index.npy` (atomic-number -> embedding-index map) shipped with the LGPS/MACE dataset.
- Sampling metric comparison (MSD/RDF/final-step) remains outside Task 14 and is
  not covered by the mini training fixture.
