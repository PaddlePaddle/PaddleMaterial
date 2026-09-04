# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0

"""Deterministic PyTorch -> Paddle weight conversion with a full key audit.

Two phases so each runs in one framework environment:

1. `dump-torch`  (requires torch): unpack a Lightning/PyTorch checkpoint into a
   portable NumPy state dict (`state_dict.npz`) plus `meta.json` describing the
   embedded model config, original SHA256, and parameter inventory.

2. `build-paddle` (requires paddle): build the same-hyperparameter Paddle model,
   map every source key onto it, transpose Linear 2-D weights to the Paddle
   `[in, out]` layout (except `atom_embedding.weight`), fail on any
   missing/unexpected/shape-mismatched key, save `.pdparams`, and write an audit
   JSON. Loading the produced file back into the model must report empty
   missing/unexpected/shape lists.

Reference: liflow_reference/LOCK.md (frozen commit e6fc475361d046865f12cae1aee11c4f56c48d87).
"""

import argparse
import ast
import hashlib
import json
import os
import sys

import numpy as np

# Allow running from the task directory while still importing the PaddleMaterials
# package living at the repository root (…/molecular_dynamics_integrator/liflow).
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Model hyper-parameters observed in P_universal.ckpt / C_universal.ckpt
# (also the default in liflow/config/train.yaml).
DEFAULT_MODEL_CFG = {
    "num_features": 64,
    "num_radial_basis": 20,
    "num_layers": 3,
    "num_elements": 77,
    "r_max": 5.0,
    "r_offset": 0.5,
    "ref_temp": 1000.0,
}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dump_torch(source, out_dir):
    import torch  # deferred: only required in this phase

    os.makedirs(out_dir, exist_ok=True)
    ckpt = torch.load(source, map_location="cpu", weights_only=False)
    if "state_dict" not in ckpt:
        raise ValueError("checkpoint has no top-level 'state_dict' key")
    state = ckpt["state_dict"]
    model_keys = sorted(k for k in state if k.startswith("model."))
    non_model = sorted(k for k in state if not k.startswith("model."))
    if non_model:
        raise ValueError(f"unexpected non-model state keys: {non_model}")

    hyper = ckpt.get("hyper_parameters", {})
    cfg_text = hyper.get("cfg", "{}")
    if isinstance(cfg_text, str):
        cfg = ast.literal_eval(cfg_text)
    elif isinstance(cfg_text, dict):
        cfg = cfg_text
    else:
        cfg = {}
    model_cfg = dict(DEFAULT_MODEL_CFG)
    model_cfg.update({k: cfg.get("model", {}).get(k, v) for k, v in model_cfg.items()})
    model_cfg["prediction_mode"] = cfg.get("model", {}).get("prediction_mode", "velocity")

    npz_path = os.path.join(out_dir, "state_dict.npz")
    arrays = {k: np.asarray(state[k].detach().cpu().numpy()) for k in model_keys}
    np.savez(npz_path, **arrays)

    meta = {
        "source": os.path.abspath(source),
        "source_sha256": sha256_file(source),
        "source_top_keys": list(ckpt.keys()),
        "num_model_params": len(model_keys),
        "model_cfg": model_cfg,
        "propagate_prior": cfg.get("propagate_prior"),
        "correct_prior": cfg.get("correct_prior"),
        "data_cutoff": (cfg.get("data") or {}).get("cutoff"),
    }
    meta_path = os.path.join(out_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"dumped {len(model_keys)} keys -> {npz_path}")
    print(f"model_cfg={model_cfg}")
    return npz_path, meta_path


def build_paddle(meta_dir, output, audit):
    import paddle  # deferred: only required in this phase
    from ppmat.models.liflow.dual_painn import DualPaiNN

    with open(os.path.join(meta_dir, "meta.json")) as f:
        meta = json.load(f)
    src = np.load(os.path.join(meta_dir, "state_dict.npz"))

    cfg = meta["model_cfg"]
    model = DualPaiNN(
        num_features=cfg["num_features"],
        num_radial_basis=cfg["num_radial_basis"],
        num_layers=cfg["num_layers"],
        num_elements=cfg["num_elements"],
        r_max=cfg["r_max"],
        r_offset=cfg["r_offset"],
        ref_temp=cfg["ref_temp"],
    )
    target = dict(model.state_dict())

    missing = []       # paddle parameter/buffer never produced by the source
    unexpected = []    # source keys not present in the Paddle model
    shape_mismatch = []
    transposed = []
    mapping = {}
    payload = {}

    for src_key in src.files:
        # torch key "model.<name>" -> Paddle key "<name>"
        dst_key = src_key[len("model."):]
        mapping[src_key] = dst_key
        if dst_key not in target:
            unexpected.append(dst_key)
            continue
        arr = np.asarray(src[src_key])
        tshape = tuple(target[dst_key].shape)
        if dst_key == "atom_embedding.weight":
            value = arr  # Embedding layout identical in both frameworks
        elif arr.ndim == 2:
            # Paddle Linear weight [in, out] vs torch [out, in]
            value = arr.T.copy()
            transposed.append(dst_key)
        else:
            value = arr  # biases / fixed buffers
        if tuple(value.shape) != tshape:
            shape_mismatch.append(
                {"key": dst_key, "source": tuple(value.shape), "paddle": tshape}
            )
            continue
        payload[dst_key] = paddle.to_tensor(value)

    for dst_key in target:
        if dst_key not in payload:
            missing.append(dst_key)

    if missing or unexpected or shape_mismatch:
        raise ValueError(
            "conversion audit failed: "
            f"missing={missing} unexpected={unexpected} shape_mismatch={shape_mismatch}"
        )

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    paddle.save(payload, output)

    # verify: loading the produced file back must also be clean
    reload = paddle.load(output)
    result = model.set_state_dict(reload)
    if result is not None and len(result) == 2 and (result[0] or result[1]):
        raise ValueError(f"reload audit failed: missing={result[0]} unexpected={result[1]}")

    report = {
        "meta": meta,
        "output": os.path.abspath(output),
        "output_sha256": sha256_file(output),
        "num_parameters": len(payload),
        "num_buffers": sum(1 for k in payload if not k.endswith((".weight", ".bias"))),
        "num_transposed": len(transposed),
        "transposed_keys": transposed,
        "mapping": mapping,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "shape_mismatch": shape_mismatch,
    }
    with open(audit, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="phase", required=True)

    p1 = sub.add_parser("dump-torch")
    p1.add_argument("--source", required=True)
    p1.add_argument("--out-dir", required=True)

    p2 = sub.add_parser("build-paddle")
    p2.add_argument("--meta-dir", required=True)
    p2.add_argument("--output", required=True)
    p2.add_argument("--audit", required=True)

    args = ap.parse_args()
    if args.phase == "dump-torch":
        dump_torch(args.source, args.out_dir)
    else:
        build_paddle(args.meta_dir, args.output, args.audit)


if __name__ == "__main__":
    main()
