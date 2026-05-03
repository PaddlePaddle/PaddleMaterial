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
CrystalLLM v1_small Prompted Evaluation — reproduces paper's 94% validity.

Paper protocol: prompted generation from v1 test set (10,286 cell compositions).
Each sample starts with `data_<composition>\n` as prompt, model generates rest.

Usage:
    # Run 500 prompted samples (subset of test set):
    python eval_v1_small.py --num-samples 500 --device gpu

    # Full 10K test set (paper's exact protocol):
    python eval_v1_small.py --num-samples 10286 --device gpu
"""

import argparse
import gzip
import importlib.util
import json
import os
import pickle
import re
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
# Zenodo URLs
# ---------------------------------------------------------------------------
ZENODO_BASE = "https://zenodo.org/api/records/10642388/files"


def _zenodo_url(filename):
    return f"{ZENODO_BASE}/{filename}/content"


def _download(url, dest_path, label=""):
    """Download file if not already present."""
    if os.path.exists(dest_path):
        print(f"  Cached: {dest_path}")
        return
    print(f"  Downloading {label or url}...")
    start = time.time()
    urllib.request.urlretrieve(url, dest_path)
    elapsed = time.time() - start
    size_mb = os.path.getsize(dest_path) / 1e6
    print(f"  Downloaded {size_mb:.1f} MB in {elapsed:.1f}s")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_test_prompts(data_dir):
    """Download and load test set CIFs, extract cell composition prompts.

    Returns list of prompt strings like 'data_Na2Cl2\\n'.
    """
    pkl_path = os.path.join(data_dir, "cifs_v1_test.pkl.gz")
    _download(_zenodo_url("cifs_v1_test.pkl.gz"), pkl_path, "cifs_v1_test.pkl.gz (1.4 MB)")

    print("  Loading test CIFs...")
    with gzip.open(pkl_path, "rb") as f:
        test_cifs = pickle.load(f)

    print(f"  Loaded {len(test_cifs)} test CIFs")

    # Extract prompt from each CIF: find the "data_<composition>" line
    prompts = []
    for cif_id, cif_str in test_cifs:
        # Find the data_ line (skip comment lines starting with #)
        for line in cif_str.split("\n"):
            if line.startswith("data_"):
                prompts.append(line + "\n")
                break
        else:
            # Fallback: first non-comment line
            for line in cif_str.split("\n"):
                if not line.startswith("#") and line.strip():
                    prompts.append(line + "\n")
                    break

    print(f"  Extracted {len(prompts)} prompts")
    return prompts


def download_and_convert_v1_small(data_dir):
    """Download v1_small checkpoint and convert to Paddle."""
    ckpt_dir = os.path.join(data_dir, "crystallm_v1_small")
    tar_path = os.path.join(data_dir, "crystallm_v1_small.tar.gz")
    pd_path = os.path.join(ckpt_dir, "ckpt.pdparams")
    args_path = os.path.join(ckpt_dir, "model_args.json")

    # Check if already converted
    if os.path.exists(pd_path) and os.path.exists(args_path):
        print(f"  Cached: {pd_path}")
        with open(args_path) as f:
            model_args = json.load(f)
        return pd_path, model_args

    # Download
    _download(_zenodo_url("crystallm_v1_small.tar.gz"), tar_path,
              "crystallm_v1_small.tar.gz (285 MB)")

    # Extract
    if not os.path.isdir(ckpt_dir):
        print("  Extracting...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=data_dir)

    # Find .pt file
    pt_path = None
    for root, dirs, files in os.walk(ckpt_dir):
        for f in files:
            if f.endswith(".pt"):
                pt_path = os.path.join(root, f)
                break
    if pt_path is None:
        raise FileNotFoundError(f"No .pt file found in {ckpt_dir}")

    # Convert
    import torch

    print(f"  Converting {pt_path} -> Paddle...")
    checkpoint = torch.load(pt_path, map_location="cpu")
    raw_sd = checkpoint["model"]
    model_args = checkpoint.get("model_args", {})

    # Save model_args for future loads without torch
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
# Main evaluation
# ---------------------------------------------------------------------------
def evaluate(
    num_samples=500,
    max_tokens=1023,
    temperature=1.0,
    top_k=10,
    device="gpu",
    data_dir="./data/crystalllm_checkpoints",
    output_json=None,
    log_interval=50,
    seed=42,
):
    """Run prompted CrystalLLM v1_small evaluation (paper protocol)."""
    paddle.set_device(device)
    print(f"Device: {device}")
    if device == "gpu":
        print(f"GPU: {paddle.device.cuda.get_device_name()}")

    os.makedirs(data_dir, exist_ok=True)

    # Step 1: Download checkpoint
    print("\n[1/5] Download v1_small checkpoint...")
    pd_path, model_args = download_and_convert_v1_small(data_dir)

    # Step 2: Load test prompts
    print("\n[2/5] Load test set prompts...")
    all_prompts = load_test_prompts(data_dir)

    # Subsample if needed
    rng = np.random.RandomState(seed)
    if num_samples < len(all_prompts):
        indices = rng.choice(len(all_prompts), size=num_samples, replace=False)
        indices.sort()
        prompts = [all_prompts[i] for i in indices]
    else:
        prompts = all_prompts
        num_samples = len(prompts)

    print(f"  Using {num_samples} prompts (of {len(all_prompts)} total)")
    print(f"  Example prompt: {prompts[0]!r}")

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

    # Step 4: Prompted generation
    print(f"\n[4/5] Generate {num_samples} prompted samples (max_tokens={max_tokens}, T={temperature}, top_k={top_k})...")
    tok = CIFTokenizer()

    raw_texts = []
    prompt_lengths = []
    start_time = time.time()
    last_log = start_time

    for i, prompt in enumerate(prompts):
        # Tokenize prompt: tokenize_cif → list of string tokens → encode to IDs
        prompt_tokens = tok.tokenize_cif(prompt)
        prompt_ids = tok.encode(prompt_tokens)
        prompt_lengths.append(len(prompt_ids))

        seed_ids = paddle.to_tensor([prompt_ids], dtype="int64")
        max_gen = min(max_tokens, config.block_size - len(prompt_ids))

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
        if (i + 1) % log_interval == 0 or (i + 1) == num_samples:
            now = time.time()
            elapsed = now - start_time
            rate = (i + 1) / elapsed
            eta = (num_samples - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1:>6}/{num_samples}] {rate:.2f} samples/s | "
                  f"elapsed {elapsed:.0f}s | ETA {eta:.0f}s",
                  flush=True)
            last_log = now

    total_time = time.time() - start_time
    avg_prompt_len = np.mean(prompt_lengths)
    print(f"  Total generation: {total_time:.1f}s ({total_time/num_samples:.2f}s/sample)")
    print(f"  Avg prompt length: {avg_prompt_len:.1f} tokens")

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
    print("EVALUATION RESULTS — CrystalLLM v1_small (prompted, paper protocol)")
    print("=" * 70)
    print(f"  Model:                v1_small (trained on 2.3M structures)")
    print(f"  Framework:            PaddlePaddle {paddle.__version__}")
    print(f"  Samples generated:    {num_samples} / {len(all_prompts)} test set")
    print(f"  Sensible rate:        {results['sensible_rate']:.2%}")
    print(f"  Formula consistency:  {results['formula_consistency_rate']:.2%}")
    print(f"  Validity rate:        {results['validity_rate']:.2%}")
    print(f"  Avg bond score:       {results['avg_bond_score']:.4f}")
    print(f"  SG consistency:       {results['sg_consistency_rate']:.2%}")
    print(f"  Generation time:      {total_time:.1f}s ({total_time/num_samples:.2f}s/sample)")
    print(f"  Evaluation time:      {eval_time:.1f}s")
    print()
    print("  Paper targets (v1_small, 10,286 test prompts):")
    print(f"    Validity:        94.0%  (ours: {results['validity_rate']:.1%})")
    print(f"    Bond score:      0.988  (ours: {results['avg_bond_score']:.4f})")
    print(f"    SG consistency:  98.9%  (ours: {results['sg_consistency_rate']:.1%})")
    print("=" * 70)

    # Save results to JSON
    output = {
        "model": "CrystalLLM v1_small",
        "training_data": "2.3M structures (MP + OQMD + NOMAD)",
        "evaluation": "prompted (paper protocol)",
        "framework": f"PaddlePaddle {paddle.__version__}",
        "device": device,
        "num_samples": num_samples,
        "total_test_set": len(all_prompts),
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_k": top_k,
        "seed": seed,
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
            "validity_rate": 0.941,
            "avg_bond_score": 0.988,
            "sg_consistency_rate": 0.989,
            "note": "From v1_small on full 10,286 test set (Nature Comms 2024)",
        },
        "timing": {
            "generation_seconds": round(total_time, 1),
            "seconds_per_sample": round(total_time / num_samples, 3),
            "evaluation_seconds": round(eval_time, 1),
            "avg_prompt_tokens": round(float(avg_prompt_len), 1),
        },
    }

    if output_json is None:
        output_json = f"eval_v1_small_{num_samples}samples.json"
    with open(output_json, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_json}")

    # Save raw CIFs
    cif_path = output_json.replace(".json", "_cifs.txt")
    with open(cif_path, "w") as f:
        for i, (prompt, cif) in enumerate(zip(prompts, generated_cifs)):
            f.write(f"# === SAMPLE {i+1} (prompt: {prompt.strip()}) ===\n")
            f.write(cif)
            f.write("\n\n")
    print(f"Raw CIFs saved to {cif_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CrystalLLM v1_small Prompted Evaluation (paper protocol)"
    )
    parser.add_argument("--num-samples", type=int, default=500,
                        help="Number of test prompts to evaluate (default: 500)")
    parser.add_argument("--max-tokens", type=int, default=1023,
                        help="Max tokens per sample (default: 1023)")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=10,
                        help="Top-k sampling (default: 10, matches paper)")
    parser.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    parser.add_argument("--data-dir", default="./data/crystalllm_checkpoints")
    parser.add_argument("--output", default=None, help="Output JSON path")
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    evaluate(
        num_samples=args.num_samples,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=args.device,
        data_dir=args.data_dir,
        output_json=args.output,
        log_interval=args.log_interval,
        seed=args.seed,
    )
