# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0

import argparse
import csv
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import paddle

from ppmat.datasets.collate_fn import DefaultCollator
from ppmat.datasets.liflow_dataset import LiFlowDataset
from ppmat.models.liflow import LiFlow


def _model(backend):
    return LiFlow(
        num_features=64,
        num_radial_basis=20,
        num_layers=3,
        num_elements=77,
        r_max=5.0,
        r_offset=0.5,
        ref_temp=1000.0,
        prediction_mode="velocity",
        execution_backend=backend,
    )


def _batch_from_first_sample(data_root):
    index_path = os.path.join(data_root, "test_800K.csv")
    if not os.path.isfile(index_path):
        raise FileNotFoundError(f"LiFlow test index not found: {index_path}")
    with open(index_path, newline="", encoding="utf-8") as file:
        row = next(csv.DictReader(file))
    with tempfile.TemporaryDirectory(prefix="liflow_benchmark_") as cache_dir:
        dataset = LiFlowDataset(
            path=data_root,
            index_file=index_path,
            time_delay_steps=100,
            seed=42,
            random_time=False,
            cache_path=os.path.join(cache_dir, "dataset_cache"),
        )
        sample = dataset[0]
        collated = DefaultCollator()([sample])
        batch = {key: getattr(collated, key) for key in collated.keys}
    for key, value in batch.items():
        if isinstance(value, np.ndarray):
            dtype = "int64" if np.issubdtype(value.dtype, np.integer) else "float32"
            batch[key] = paddle.to_tensor(value, dtype=dtype)
    return batch, sample


def _synchronize():
    if paddle.get_device().startswith("gpu"):
        paddle.device.cuda.synchronize()


def benchmark(fn, warmup=20, iterations=100):
    _synchronize()
    first_start = time.perf_counter()
    fn()
    _synchronize()
    first_seconds = time.perf_counter() - first_start

    warmup_samples = []
    for _ in range(warmup):
        start = time.perf_counter()
        fn()
        _synchronize()
        warmup_samples.append(time.perf_counter() - start)

    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        _synchronize()
        samples.append(time.perf_counter() - start)
    return {
        "first_seconds": first_seconds,
        "warmup_mean_seconds": float(np.mean(warmup_samples)) if warmup_samples else None,
        "warmup_std_seconds": float(np.std(warmup_samples)) if warmup_samples else None,
        "mean_seconds": float(np.mean(samples)),
        "std_seconds": float(np.std(samples)),
    }


def _run_backend(backend, batch, checkpoint, warmup, iterations):
    model = _model(backend)
    model.eval()
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    state_dict = paddle.load(checkpoint)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    target_keys = set(model.state_dict().keys())
    if state_dict and not any(key in target_keys for key in state_dict):
        prefix = "network."
        if all(prefix + key in target_keys for key in state_dict):
            state_dict = {prefix + key: value for key, value in state_dict.items()}
    model.set_state_dict(state_dict)
    return benchmark(
        lambda: model.predict(batch), warmup=warmup, iterations=iterations
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="Benchmark one real LiFlow Universal forward.")
    parser.add_argument("--backend", choices=("eager", "cinn", "both"), default="both")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    if args.warmup < 0 or args.iterations <= 0:
        parser.error("--warmup must be >= 0 and --iterations must be > 0")

    batch, sample = _batch_from_first_sample(args.data_root)
    backends = ("eager", "cinn") if args.backend == "both" else (args.backend,)
    result = {
        "paddle_version": paddle.__version__,
        "device": paddle.get_device(),
        "cuda_compiled": bool(paddle.is_compiled_with_cuda()),
        "cinn_compiled": bool(paddle.base.is_compiled_with_cinn()),
        "data_root": os.path.abspath(args.data_root),
        "checkpoint": os.path.abspath(args.checkpoint),
        "sample_name": sample["name"],
        "num_atoms": int(sample["num_atoms"]),
        "warmup": args.warmup,
        "iterations": args.iterations,
        "backends": {},
    }
    for backend in backends:
        try:
            metrics = _run_backend(
                backend, batch, args.checkpoint, args.warmup, args.iterations
            )
            result["backends"][backend] = {"status": "ok", **metrics}
            print(
                f"{backend}: first_seconds={metrics['first_seconds']:.8f} "
                f"warmup_mean_seconds={metrics['warmup_mean_seconds']!s} "
                f"mean_seconds={metrics['mean_seconds']:.8f} "
                f"std_seconds={metrics['std_seconds']:.8f}"
            )
        except Exception as exc:
            if backend == "cinn":
                reason = f"{type(exc).__name__}: {exc}"
                result["backends"][backend] = {"status": "skipped", "reason": reason}
                print(f"cinn: skipped ({reason})")
            else:
                raise

    eager = result["backends"].get("eager")
    cinn = result["backends"].get("cinn")
    if eager and eager.get("status") == "ok" and cinn and cinn.get("status") == "ok":
        result["speedup_eager_over_cinn"] = eager["mean_seconds"] / cinn["mean_seconds"]
        print(f"speedup_eager_over_cinn={result['speedup_eager_over_cinn']:.6f}")
    if args.output:
        with open(args.output, "w", encoding="utf-8") as file:
            json.dump(result, file, indent=2, ensure_ascii=False)
    return result


if __name__ == "__main__":
    main()
