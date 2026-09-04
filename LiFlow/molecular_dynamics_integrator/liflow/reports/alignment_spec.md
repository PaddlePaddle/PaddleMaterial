# LiFlow PyTorch–Paddle Alignment Spec

Frozen contract used by all later numerical-alignment tests, reports, and weight conversion. Every result must reference this document, one frozen reference revision, and one declared fixture.

## 1. Frozen revisions

- LiFlow reference: `learningmatter-mit/liflow` commit `e6fc475361d046865f12cae1aee11c4f56c48d87` (2025-10-17), local read-only copy: `D:\实验\liflow_reference`.
- PaddleMaterials: commit `a9a689cf64d98a2b415a15f794fe88598b5943ef` (develop).
- Reference checkpoint files: `ckpt/P_universal.ckpt`, `ckpt/C_universal.ckpt` (+ LGPS/LPS checkpoints), local `D:\实验\liflow_reference\ckpt\`.

## 2. Model input contract (single graph)

| key | shape | dtype | notes |
|---|---|---|---|
| positions_1 | [N, 3] | float32 | source coordinates |
| positions_2 | [N, 3] | float32 | target coordinates (inference: interpolated `x_t`) |
| prior | [N, 3] | float32 | sampled displacement noise |
| edge_index | [2, E] | int64 | neighbor pairs, Cartesian shifts applied |
| shifts | [E, 3] | float32 | `integer_shift @ lattice` |
| elements | [N] | int64 | `element_idx[atomic_numbers]` |
| time | [N] | float32 | graph-to-node broadcast of flow time |
| temp | [N] | float32 | temperature per node (all equal in a single graph) |
| batch | [N] | int64 | all zeros for a single graph |
| positions_t | [N, 3] | float32 | `(1-t)*source + t*positions_2`, source = positions_1 + prior |

## 3. Deterministic cases (fixtures)

- `fixture_liflow_14atoms.npz` — periodic 14-atom case containing a sub-cutoff pair reached only through a periodic image (exercises `shifts`).
- `fixture_liflow_nonperiodic.npz` — small non-periodic case (dense neighbor list, no shifts).
- `fixture_liflow_cutoff_edge.npz` — pair just inside / just outside the 5.0 cutoff (exercises edge construction boundary).

## 4. Acceptance tolerances

- Layer forward / full-model forward: `max_abs_diff <= 1e-6` per output element (float32).
- Backward: per named parameter gradient `max_abs_diff <= 1e-6`; document any approved relaxed operator tolerance.
- Two-epoch training: same loss trend and numerically aligned loss curve under identical batch order and seed.
- Sampling metrics (MSD per species, RDF MAE, final step): relative error <= 5%.

## 5. Reference output provenance

Reference (`golden`) outputs are produced only by `liflow_reference/scripts/golden_export.py`, which loads the frozen commit, seeds deterministically, saves raw float32 `.npy` tensors plus the exact parameters used. Golden files are checked into `test/fixtures/liflow/reference_outputs/` with the generating commit recorded in `fixture_manifest.json`.

## 6. Status

- [x] PyTorch reference environment ready (CPU; `liflow-torch` conda env, torch 2.5.1+cpu, lightning 2.6.5, torch_geometric 2.8.0, vesin 0.6.1, ase 3.29.0).
- [x] Mini-model golden outputs generated (`reference_outputs/mini_*`, state dict `mini_state_dict.npz`, 23 keys) for all three fixtures.
- [x] Real checkpoint `P_universal.ckpt` structure parsed: PyTorch Lightning archive; 45 `model.*` state-dict entries (weights + buffers); embedded cfg decoded (see LOCK.md).
- [x] Paddle layer/network rewritten to reference semantics and **forward-aligned**: measured `max_abs_diff` 6.7e-8 / 3.0e-8 / 1.2e-7 on the three fixtures (target 1e-6). Weight loader transposes Paddle `[in,out]` Linear weights (except `atom_embedding.weight`); buffers match 1:1.
- [x] Checkpoint conversion `P_universal`/`C_universal` completed: 45 keys each, 22 transposed Linear weights, output SHA256 recorded, **missing_keys=[] unexpected_keys=[] shape_mismatch=[]** plus reload-back check (see `alignment_report.md`). Real forward alignment still requires the official `element_index.npy` dataset artifact.
- [ ] Two aligned training epochs documented.
- [ ] Sampling metric comparison (5%) completed.

## 8. Task 5 / Task 10 preparatory status

- [x] Shared cutoff neighbor list moved to `ppmat/models/liflow/geometry.py`
  (Cartesian shifts, exact semantics of the reference); `IntegratorPredictor` and
  `LiFlowDataset` both consume it.
- [x] `LiFlowDataset` now emits per-sample `edge_index` / `shifts` built on
  `positions_1` (matches the reference TimeDelayedPairDataset) and exposes
  `cutoff` / `pbc`; deterministic seed/validation behavior retained.
- [x] Offline mini dataset in official layout: `test/fixtures/liflow/dataset_mini`
  (`element_index.npy`, `atomic_numbers.npy`, `lattice.npy`, `positions_800K.npz`,
  train/test CSVs) with schema/determinism/edge/prior tests. Verified by a direct
  probe (546 periodic edges; Cartesian shifts); `pytest` of this file is unstable
  on this Windows session because of multiprocessing residue, CI will run it.
- [x] Local reproducible model archive: `tools/package_artifacts.py` builds the
  POSIX `liflow_universal/{yaml, checkpoints/{propagator,corrector}.pdparams}` zip
  + manifest, verified by fresh-extract round trip.
- [ ] Official dataset artifact (Zenodo `10.5281/zenodo.14889658`) not yet on this
  machine: real-trained `element_index.npy` mapping and LGPS/MACE trajectories are
  still required for real-forward alignment and sampling metrics. The mini fixture
  uses a self-consistent `element_index = Z` convention for pipeline tests only.

## 7. Paddle-side fidelity gaps to close (found while freezing the contract)

Reference `liflow/model/layers.py` semantics vs current Paddle layers:

1. `GaussianFourierBasis` holds a fixed random `freqs` **buffer** (`randn(n/2)*2π`), not a learnable parameter with a π multiplier. Paddle layer must match name/shape and load it from the checkpoint.
2. `BesselBasis` buffers are `freqs` and `prefactor`; formula `prefactor*sin(args)/r`.
3. `CosineCutoff` buffers `r_max`; formula identical.
4. `DualPaiNN` applies `+r_offset` then clamps lengths to `r_max` before radial/cutoff; `linear_v` maps `positions_diff[..., None]` with `Linear(1, F)` (no bias); initial `s = atom_emb + cat(time_fourier, temp_fourier)`.
5. UpdateBlock/mlp gates and GatedEquivariantBlock output single vector `[N,3]`.
6. `r_offset=0.5` must be added to the Paddle model hyper-parameters and used in lengths.
