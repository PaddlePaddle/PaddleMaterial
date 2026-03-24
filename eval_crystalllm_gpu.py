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
CrystalLLM GPU Evaluation — 10,000 sample generation for production metrics.

This script:
1. Downloads the perov-5-small checkpoint from Zenodo
2. Converts PyTorch weights to Paddle
3. Generates N samples on GPU (default 10,000)
4. Evaluates with CrystalMetrics (validity, bond score, SG consistency)
5. Saves results to JSON

Usage:
    # Full 10K eval on GPU:
    python eval_crystalllm_gpu.py --device gpu --num-samples 10000

    # Quick smoke test (100 samples):
    python eval_crystalllm_gpu.py --device gpu --num-samples 100

    # CPU fallback (slow):
    python eval_crystalllm_gpu.py --device cpu --num-samples 10
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
# Checkpoint download
# ---------------------------------------------------------------------------
ZENODO_BASE = "https://zenodo.org/api/records/10642388/files"
CHECKPOINT_NAME = "crystallm_perov_5_small"
CHECKPOINT_URL = f"{ZENODO_BASE}/{CHECKPOINT_NAME}.tar.gz/content"


def _download_and_extract(url, dest_dir):
    """Download a tar.gz from URL and extract to dest_dir."""
    # Check if already extracted
    for root, dirs, files in os.walk(dest_dir):
        for f in files:
            if f.endswith(".pt"):
                pt_path = os.path.join(root, f)
                print(f"  Using cached {pt_path}")
                return pt_path

    tar_path = os.path.join(dest_dir, f"{CHECKPOINT_NAME}.tar.gz")
    if not os.path.exists(tar_path):
        print(f"  Downloading {url}...")
        start = time.time()
        urllib.request.urlretrieve(url, tar_path)
        elapsed = time.time() - start
        size_mb = os.path.getsize(tar_path) / 1e6
        print(f"  Downloaded {size_mb:.1f} MB in {elapsed:.1f}s")
    else:
        print(f"  Using cached tarball {tar_path}")

    print("  Extracting...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=dest_dir)

    for root, dirs, files in os.walk(dest_dir):
        for f in files:
            if f.endswith(".pt"):
                return os.path.join(root, f)
    raise FileNotFoundError(f"No .pt file found after extracting {tar_path}")


def _extract_first_cif(raw_text: str) -> str:
    """Extract the first complete CIF block from generated text."""
    idx = raw_text.find("data_")
    if idx < 0:
        return raw_text.strip()
    text = raw_text[idx:]
    next_data = text.find("\n\ndata_", 1)
    if next_data > 0:
        text = text[:next_data]
    return text.strip()


# ---------------------------------------------------------------------------
# Weight conversion (inline to avoid import issues on AI Studio)
# ---------------------------------------------------------------------------
def _convert_weights(pt_path, pd_path):
    """Convert PyTorch checkpoint to Paddle format."""
    import torch

    checkpoint = torch.load(pt_path, map_location="cpu")
    raw_sd = checkpoint["model"]

    paddle_sd = {}
    skip_keys = {"lm_head.weight"}

    for key, tensor in raw_sd.items():
        clean_key = key
        if clean_key.startswith("_orig_mod.transformer."):
            clean_key = clean_key[len("_orig_mod.transformer."):]
        elif clean_key.startswith("_orig_mod."):
            clean_key = clean_key[len("_orig_mod."):]

        if clean_key in skip_keys:
            continue

        arr = tensor.numpy()
        # Transpose Linear weight matrices (PyTorch [out, in] -> Paddle [in, out])
        if "weight" in clean_key and arr.ndim == 2 and "wte" not in clean_key and "wpe" not in clean_key:
            arr = arr.T

        paddle_sd[clean_key] = arr

    paddle.save(paddle_sd, pd_path)
    return checkpoint.get("model_args", {})


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------
def evaluate(
    num_samples: int = 10000,
    max_tokens: int = 1023,
    temperature: float = 1.0,
    top_k: int = 10,
    device: str = "gpu",
    data_dir: str = "./data/crystalllm_checkpoints",
    output_json: str = None,
    batch_log_interval: int = 100,
):
    """Run full CrystalLLM evaluation pipeline."""
    paddle.set_device(device)
    print(f"Device: {device}")
    if device == "gpu":
        print(f"GPU: {paddle.device.cuda.get_device_name()}")

    os.makedirs(data_dir, exist_ok=True)

    # Step 1: Download checkpoint
    print("\n[1/5] Download checkpoint...")
    pt_path = _download_and_extract(CHECKPOINT_URL, data_dir)
    print(f"  Checkpoint: {pt_path}")

    # Step 2: Convert to Paddle
    print("\n[2/5] Convert weights...")
    pd_path = pt_path.replace(".pt", ".pdparams")
    if not os.path.exists(pd_path):
        model_args = _convert_weights(pt_path, pd_path)
        print(f"  Converted to {pd_path}")
    else:
        import torch
        checkpoint = torch.load(pt_path, map_location="cpu")
        model_args = checkpoint.get("model_args", {})
        del checkpoint
        print(f"  Using cached {pd_path}")

    # Step 3: Load model
    print("\n[3/5] Load model...")
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
    num_params = model.get_num_params()
    print(f"  Model loaded: {num_params:,} params")
    print(f"  Config: {config.n_layer}L / {config.n_head}H / {config.n_embd}D / block_size={config.block_size}")

    # Step 4: Generate samples
    print(f"\n[4/5] Generate {num_samples} samples (max_tokens={max_tokens}, T={temperature}, top_k={top_k})...")
    tok = CIFTokenizer()
    # Use "data_" as start prompt — matches upstream generate_cifs.py ab initio generation
    data_id = tok.token_to_id["data_"]

    raw_texts = []
    start_time = time.time()
    last_log = start_time

    for i in range(num_samples):
        seed_ids = paddle.to_tensor([[data_id]], dtype="int64")
        max_gen = min(max_tokens, config.block_size - 1)

        with paddle.no_grad():
            generated = model.generate(
                seed_ids,
                max_new_tokens=max_gen,
                temperature=temperature,
                top_k=top_k,
            )

        gen_text = tok.decode(generated[0].numpy().tolist())
        raw_texts.append(gen_text)

        # Progress logging
        if (i + 1) % batch_log_interval == 0 or (i + 1) == num_samples:
            now = time.time()
            elapsed = now - start_time
            rate = (i + 1) / elapsed
            eta = (num_samples - i - 1) / rate if rate > 0 else 0
            batch_time = now - last_log
            print(f"  [{i+1:>6}/{num_samples}] {rate:.1f} samples/s | "
                  f"elapsed {elapsed:.0f}s | ETA {eta:.0f}s | "
                  f"last {batch_log_interval} in {batch_time:.1f}s",
                  flush=True)
            last_log = now

    total_time = time.time() - start_time
    print(f"  Total generation: {total_time:.1f}s ({total_time/num_samples:.2f}s/sample)")

    # Extract CIFs
    generated_cifs = [_extract_first_cif(raw) for raw in raw_texts]
    valid_header_count = sum(1 for c in generated_cifs if c.startswith("data_"))
    print(f"  CIFs with 'data_' header: {valid_header_count}/{num_samples}")

    # Step 5: Evaluate
    print(f"\n[5/5] Evaluate with CrystalMetrics...")
    eval_start = time.time()
    metrics = CrystalMetrics()
    results = metrics(generated_cifs)
    eval_time = time.time() - eval_start
    print(f"  Evaluation time: {eval_time:.1f}s")

    # Print results
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    print(f"  Samples generated:    {num_samples}")
    print(f"  Sensible rate:        {results['sensible_rate']:.2%}")
    print(f"  Formula consistency:  {results['formula_consistency_rate']:.2%}")
    print(f"  Validity rate:        {results['validity_rate']:.2%}")
    print(f"  Avg bond score:       {results['avg_bond_score']:.3f}")
    print(f"  SG consistency:       {results['sg_consistency_rate']:.2%}")
    print(f"  Generation time:      {total_time:.1f}s ({total_time/num_samples:.2f}s/sample)")
    print(f"  Evaluation time:      {eval_time:.1f}s")
    print()
    print("  Paper targets (v1_small model, trained on 2.3M structures):")
    print(f"    Validity:       94.0%  (ours: {results['validity_rate']:.1%})")
    print(f"    Bond score:     0.988  (ours: {results['avg_bond_score']:.3f})")
    print(f"    SG consistency: 98.9%  (ours: {results['sg_consistency_rate']:.1%})")
    print("  NOTE: perov-5-small (11K structures) has no published ab initio validity")
    print("  targets. The above are from v1_small as reference only.")
    print("=" * 70)

    # Save results to JSON
    output = {
        "model": "CrystalLLM (perov-5-small)",
        "framework": "PaddlePaddle",
        "device": device,
        "num_samples": num_samples,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_k": top_k,
        "config": {
            "n_layer": config.n_layer,
            "n_head": config.n_head,
            "n_embd": config.n_embd,
            "block_size": config.block_size,
            "vocab_size": config.vocab_size,
        },
        "results": {
            "validity_rate": round(results["validity_rate"], 4),
            "avg_bond_score": round(results["avg_bond_score"], 4),
            "sg_consistency_rate": round(results["sg_consistency_rate"], 4),
            "sensible_rate": round(results["sensible_rate"], 4),
            "formula_consistency_rate": round(results["formula_consistency_rate"], 4),
        },
        "paper_targets": {
            "note": "v1_small model (2.3M structures), NOT perov-5-small (11K). Reference only.",
            "validity_rate": 0.94,
            "avg_bond_score": 0.988,
            "sg_consistency_rate": 0.989,
        },
        "timing": {
            "generation_seconds": round(total_time, 1),
            "seconds_per_sample": round(total_time / num_samples, 3),
            "evaluation_seconds": round(eval_time, 1),
        },
    }

    if output_json is None:
        output_json = f"crystalllm_eval_{num_samples}samples.json"
    with open(output_json, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_json}")

    # Save raw CIFs for diagnostic analysis
    cif_path = output_json.replace(".json", "_cifs.txt")
    with open(cif_path, "w") as f:
        for i, cif in enumerate(generated_cifs):
            f.write(f"# === SAMPLE {i+1} ===\n")
            f.write(cif)
            f.write("\n\n")
    print(f"Raw CIFs saved to {cif_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CrystalLLM GPU Evaluation")
    parser.add_argument("--num-samples", type=int, default=10000,
                        help="Number of samples to generate (default: 10000)")
    parser.add_argument("--max-tokens", type=int, default=1023,
                        help="Max tokens per sample (default: 1023 = block_size-1)")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=10,
                        help="Top-k sampling (default: 10, matches paper)")
    parser.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    parser.add_argument("--data-dir", default="./data/crystalllm_checkpoints")
    parser.add_argument("--output", default=None, help="Output JSON path")
    parser.add_argument("--log-interval", type=int, default=100,
                        help="Log progress every N samples")
    args = parser.parse_args()

    evaluate(
        num_samples=args.num_samples,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=args.device,
        data_dir=args.data_dir,
        output_json=args.output,
        batch_log_interval=args.log_interval,
    )
