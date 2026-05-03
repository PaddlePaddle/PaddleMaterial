#!/usr/bin/env python3
# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Multi-dataset CrystalLLM evaluation across all 4 paper datasets.

Evaluates the Paddle CrystalLLM port on each dataset checkpoint from Zenodo:
  - Perov-5 (11,140 perovskites)
  - Carbon-24 (10,153 carbon structures)
  - MP-20 (45,231 general inorganic)
  - MPTS-52 (40,476 ternary+)

For each dataset, downloads the small model checkpoint, converts to Paddle,
generates N unprompted samples, and reports validity metrics.

Usage:
    # Quick smoke test (50 samples per dataset):
    python eval_multi_dataset.py --num-samples 50 --device gpu

    # Full evaluation (10K per dataset, ~14h on GTX 1060):
    python eval_multi_dataset.py --num-samples 10000 --device gpu

    # Single dataset only:
    python eval_multi_dataset.py --datasets perov_5 --num-samples 500 --device gpu
"""

import argparse
import importlib.util
import json
import os
import sys
import tarfile
import time
import urllib.request

import numpy as np
import paddle

# ---------------------------------------------------------------------------
# Module loading (bypass ppmat's pgl-eager imports)
# ---------------------------------------------------------------------------
_repo_root = os.path.dirname(os.path.abspath(__file__))


def _load_module(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_tok_mod = _load_module(
    "cif_tokenizer",
    os.path.join(_repo_root, "ppmat", "models", "crystalllm", "cif_tokenizer.py"),
)
CIFTokenizer = _tok_mod.CIFTokenizer

_model_mod = _load_module(
    "crystalllm",
    os.path.join(_repo_root, "ppmat", "models", "crystalllm", "crystalllm.py"),
)
CrystalLLM = _model_mod.CrystalLLM
GPTConfig = _model_mod.GPTConfig

_metrics_mod = _load_module(
    "crystal_metrics",
    os.path.join(_repo_root, "ppmat", "metrics", "crystal_metrics.py"),
)
CrystalMetrics = _metrics_mod.CrystalMetrics


# ---------------------------------------------------------------------------
# Dataset / checkpoint definitions
# ---------------------------------------------------------------------------

ZENODO_BASE = "https://zenodo.org/api/records/10642388/files"

# Paper Table 1: Ab-initio generation validity rates (small models)
PAPER_TARGETS = {
    "perov_5": {
        "description": "Perov-5: 11,140 perovskite structures",
        "ckpt_small": "crystallm_perov_5_small",
        "ckpt_large": "crystallm_perov_5_large",
        # Paper doesn't report per-dataset ab-initio validity for individual
        # datasets separately. Using v1 as reference baseline.
        "paper_validity": None,
        "paper_note": "No separate ab-initio target in paper for this dataset",
    },
    "carbon_24": {
        "description": "Carbon-24: 10,153 carbon allotrope structures",
        "ckpt_small": "crystallm_carbon_24_small",
        "ckpt_large": "crystallm_carbon_24_large",
        "paper_validity": None,
        "paper_note": "No separate ab-initio target in paper for this dataset",
    },
    "mp_20": {
        "description": "MP-20: 45,231 general inorganic structures from Materials Project",
        "ckpt_small": "crystallm_mp_20_small",
        "ckpt_large": "crystallm_mp_20_large",
        "paper_validity": None,
        "paper_note": "No separate ab-initio target in paper for this dataset",
    },
    "mpts_52": {
        "description": "MPTS-52: 40,476 ternary+ structures from Materials Project",
        "ckpt_small": "crystallm_mpts_52_small",
        "ckpt_large": "crystallm_mpts_52_large",
        "paper_validity": None,
        "paper_note": "No separate ab-initio target in paper for this dataset",
    },
}


def _download(url, dest_path, label=""):
    """Download file if not already present."""
    if os.path.exists(dest_path):
        return
    print(f"  Downloading {label or url}...")
    start = time.time()
    urllib.request.urlretrieve(url, dest_path)
    elapsed = time.time() - start
    size_mb = os.path.getsize(dest_path) / 1e6
    print(f"  Downloaded {size_mb:.1f} MB in {elapsed:.1f}s")


def download_and_convert(dataset_key, data_dir, model_size="small"):
    """Download checkpoint and convert to Paddle. Returns (pd_path, model_args)."""
    info = PAPER_TARGETS[dataset_key]
    ckpt_name = info[f"ckpt_{model_size}"]
    ckpt_dir = os.path.join(data_dir, ckpt_name)
    pd_path = os.path.join(ckpt_dir, "ckpt.pdparams")
    args_path = os.path.join(ckpt_dir, "model_args.json")

    if os.path.exists(pd_path) and os.path.exists(args_path):
        print(f"  Cached: {pd_path}")
        with open(args_path) as f:
            return pd_path, json.load(f)

    # Download tarball
    tar_path = os.path.join(data_dir, f"{ckpt_name}.tar.gz")
    url = f"{ZENODO_BASE}/{ckpt_name}.tar.gz/content"
    _download(url, tar_path, f"{ckpt_name}.tar.gz")

    # Extract
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir, exist_ok=True)
    print(f"  Extracting {ckpt_name}...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=data_dir)

    # Find .pt file
    pt_path = None
    for root, dirs, files in os.walk(ckpt_dir):
        for f in files:
            if f.endswith(".pt"):
                pt_path = os.path.join(root, f)
                break
    # Also check data_dir level (some tarballs extract directly)
    if pt_path is None:
        for root, dirs, files in os.walk(data_dir):
            if ckpt_name in root:
                for f in files:
                    if f.endswith(".pt"):
                        pt_path = os.path.join(root, f)
                        break

    if pt_path is None:
        raise FileNotFoundError(f"No .pt file found for {ckpt_name} in {data_dir}")

    # Convert
    import torch
    print(f"  Converting {pt_path} -> Paddle...")
    checkpoint = torch.load(pt_path, map_location="cpu")
    raw_sd = checkpoint["model"]
    model_args = checkpoint.get("model_args", {})

    with open(args_path, "w") as f:
        json.dump(model_args, f, indent=2)

    paddle_sd = {}
    for key, tensor in raw_sd.items():
        clean_key = key
        if clean_key.startswith("_orig_mod.transformer."):
            clean_key = clean_key[len("_orig_mod.transformer."):]
        elif clean_key.startswith("_orig_mod."):
            clean_key = clean_key[len("_orig_mod."):]
        if clean_key == "lm_head.weight":
            continue
        arr = tensor.numpy()
        if "weight" in clean_key and arr.ndim == 2 and "wte" not in clean_key and "wpe" not in clean_key:
            arr = arr.T
        paddle_sd[clean_key] = arr

    paddle.save(paddle_sd, pd_path)
    print(f"  Saved {len(paddle_sd)} params to {pd_path}")
    del checkpoint
    return pd_path, model_args


def _extract_first_cif(raw_text):
    idx = raw_text.find("data_")
    if idx < 0:
        return raw_text.strip()
    text = raw_text[idx:]
    next_data = text.find("\n\ndata_", 1)
    if next_data > 0:
        text = text[:next_data]
    return text.strip()


def evaluate_dataset(
    dataset_key,
    num_samples=500,
    max_tokens=1023,
    temperature=1.0,
    top_k=10,
    device="gpu",
    data_dir="./data/crystalllm_checkpoints",
    model_size="small",
    log_interval=50,
):
    """Evaluate a single dataset checkpoint. Returns results dict."""
    info = PAPER_TARGETS[dataset_key]
    print(f"\n{'='*70}")
    print(f"  Dataset: {dataset_key} ({info['description']})")
    print(f"  Model size: {model_size}")
    print(f"{'='*70}")

    # Download & convert
    pd_path, model_args = download_and_convert(dataset_key, data_dir, model_size)

    # Load model
    config = GPTConfig(
        block_size=model_args.get("block_size", 1024),
        vocab_size=model_args.get("vocab_size", 371),
        n_layer=model_args.get("n_layer", 8),
        n_head=model_args.get("n_head", 8),
        n_embd=model_args.get("n_embd", 512),
        dropout=0.0,
        bias=model_args.get("bias", True),
    )
    model = CrystalLLM(
        block_size=config.block_size,
        vocab_size=config.vocab_size,
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_embd=config.n_embd,
        dropout=0.0,
        bias=config.bias,
    )
    state = paddle.load(pd_path)
    model.set_state_dict(state)
    model.eval()
    print(f"  Model: {model.get_num_params():,} params ({config.n_layer}L/{config.n_head}H/{config.n_embd}D)")

    # Generate
    tok = CIFTokenizer()
    data_id = tok.token_to_id["data_"]
    raw_texts = []
    start_time = time.time()

    for i in range(num_samples):
        seed_ids = paddle.to_tensor([[data_id]], dtype="int64")
        with paddle.no_grad():
            generated = model.generate(
                seed_ids,
                max_new_tokens=min(max_tokens, config.block_size - 1),
                temperature=temperature,
                top_k=top_k,
            )
        raw_texts.append(tok.decode(generated[0].numpy().tolist()))

        if (i + 1) % log_interval == 0 or (i + 1) == num_samples:
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            eta = (num_samples - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1:>6}/{num_samples}] {rate:.2f} s/s | ETA {eta:.0f}s", flush=True)

    total_time = time.time() - start_time

    # Evaluate
    generated_cifs = [_extract_first_cif(raw) for raw in raw_texts]
    metrics = CrystalMetrics()
    results = metrics(generated_cifs)

    print(f"\n  Results for {dataset_key} ({model_size}):")
    print(f"    Validity:         {results['validity_rate']:.2%}")
    print(f"    Bond score:       {results['avg_bond_score']:.4f}")
    print(f"    SG consistency:   {results['sg_consistency_rate']:.2%}")
    print(f"    Sensible:         {results['sensible_rate']:.2%}")
    print(f"    Time:             {total_time:.1f}s ({total_time/num_samples:.2f}s/sample)")

    # Free model memory
    del model, state
    if device == "gpu":
        paddle.device.cuda.empty_cache()

    return {
        "dataset": dataset_key,
        "description": info["description"],
        "model_size": model_size,
        "num_samples": num_samples,
        "results": {k: round(v, 4) for k, v in results.items()},
        "timing": {
            "total_seconds": round(total_time, 1),
            "per_sample": round(total_time / num_samples, 3),
        },
        "config": {
            "n_layer": config.n_layer,
            "n_head": config.n_head,
            "n_embd": config.n_embd,
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Multi-dataset CrystalLLM evaluation (all 4 paper datasets)"
    )
    parser.add_argument("--datasets", nargs="+",
                        choices=list(PAPER_TARGETS.keys()) + ["all"],
                        default=["all"],
                        help="Datasets to evaluate (default: all)")
    parser.add_argument("--num-samples", type=int, default=500,
                        help="Samples per dataset (default: 500)")
    parser.add_argument("--model-size", choices=["small", "large"], default="small")
    parser.add_argument("--max-tokens", type=int, default=1023)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    parser.add_argument("--data-dir", default="./data/crystalllm_checkpoints")
    parser.add_argument("--output", default=None)
    parser.add_argument("--log-interval", type=int, default=50)
    args = parser.parse_args()

    paddle.set_device(args.device)
    datasets = list(PAPER_TARGETS.keys()) if "all" in args.datasets else args.datasets

    print("=" * 70)
    print(f"CrystalLLM Multi-Dataset Evaluation")
    print(f"  Framework: PaddlePaddle {paddle.__version__}")
    print(f"  Datasets:  {', '.join(datasets)}")
    print(f"  Samples:   {args.num_samples} per dataset")
    print(f"  Model:     {args.model_size}")
    print(f"  Device:    {args.device}")
    if args.device == "gpu":
        print(f"  GPU:       {paddle.device.cuda.get_device_name()}")
    print("=" * 70)

    all_results = {}
    for ds in datasets:
        try:
            result = evaluate_dataset(
                ds,
                num_samples=args.num_samples,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                device=args.device,
                data_dir=args.data_dir,
                model_size=args.model_size,
                log_interval=args.log_interval,
            )
            all_results[ds] = result
        except Exception as e:
            print(f"\n  ERROR evaluating {ds}: {e}")
            all_results[ds] = {"dataset": ds, "error": str(e)}

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY — Multi-Dataset Results")
    print("=" * 70)
    print(f"{'Dataset':<12} {'Validity':>10} {'Bond':>8} {'SG':>10} {'Sensible':>10} {'Time':>8}")
    print("-" * 70)
    for ds, res in all_results.items():
        if "error" in res:
            print(f"{ds:<12} {'ERROR':>10}")
            continue
        r = res["results"]
        t = res["timing"]["total_seconds"]
        print(f"{ds:<12} {r['validity_rate']:>9.1%} {r['avg_bond_score']:>8.4f} "
              f"{r['sg_consistency_rate']:>9.1%} {r['sensible_rate']:>9.1%} {t:>7.0f}s")
    print("=" * 70)

    # Save
    output_path = args.output or f"eval_multi_dataset_{args.num_samples}samples.json"
    output = {
        "framework": f"PaddlePaddle {paddle.__version__}",
        "device": args.device,
        "model_size": args.model_size,
        "num_samples_per_dataset": args.num_samples,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "datasets": all_results,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
