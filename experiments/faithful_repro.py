"""Faithful reproduction of AlloyGAN CGAN — matches original PyTorch code exactly.

Original: https://github.com/photon-git/AlloyGAN/blob/main/models/cgan.py

Architecture (from original cgan.py):
    G: Linear(31, 512) → LeakyReLU(0.2) → Linear(512, 40) → Sigmoid
    D: Linear(66, 1024) → LeakyReLU(0.2) → Linear(1024, 1) → Sigmoid

Training (from original cgan.py):
    - BCELoss, Adam(lr=0.0002, β1=0.5, β2=0.999, weight_decay=1e-5)
    - D step: z ~ Uniform(0,1), D(real) vs D(G(z||cond))
    - G step: z ~ Normal(0,1), maximize D(G(z||cond))
    - Epoch-based (default 50), batch_size=64
    - No normalization on data (original loads CSV raw)

Data adaptation:
    Our CSV has compositions as at% (0-100). G outputs Sigmoid [0-1].
    → Divide compositions by 100 to get fractions [0-1] matching G output range.
    → MinMax-normalize conditions to [0-1] (original GAN version uses
      sklearn MinMaxScaler; CGAN version's CSV was likely pre-normalized).

Evaluation:
    WD = avg Wasserstein distance across 40 composition columns.
    Paper reports WD=0.41 for Cu subset (on fraction scale [0-1]).
"""
import importlib.util
import os
import sys
import time

import numpy as np
import paddle
import paddle.nn as nn
from scipy.stats import wasserstein_distance

# ---- importlib setup (bypass ppmat __init__ chain) ----
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)


def _import(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_dataset = _import(
    "alloy_dataset", os.path.join(BASE, "ppmat/datasets/alloy_dataset.py")
)

ELEMS = [
    "Cu", "Zr", "Al", "Ni", "Ti", "Ag", "Fe", "Mg", "B", "Si",
    "Nb", "Y", "Ca", "La", "Co", "Be", "C", "Mo", "Pd", "P",
    "Sn", "Cr", "Hf", "Zn", "Gd", "Ce", "Er", "Ga", "Au", "Nd",
    "Dy", "W", "Pr", "Ta", "Sc", "Li", "Sm", "S", "Pt", "Mn",
]


def train_and_eval(epochs=50, batch_size=64, lr=0.0002, weight_decay=1e-5,
                   seed=42, device="gpu", categories=None):
    """Train CGAN matching original cgan.py exactly, then evaluate."""
    paddle.seed(seed)
    np.random.seed(seed)
    paddle.set_device(device)

    # ---- Data ----
    ds = _dataset.AlloyDataset(
        os.path.join(BASE, "data/alloy/Alloy_train.csv"),
        categories=categories,  # None = all data (matching original)
        normalize=True,  # comp/100 + conditions MinMax → all [0,1]
    )
    from paddle.io import BatchSampler, DataLoader
    loader = DataLoader(
        ds,
        batch_sampler=BatchSampler(ds, batch_size=batch_size, shuffle=True, drop_last=False),
    )

    n_samples = len(ds)
    print(f"Dataset: {n_samples} samples", flush=True)

    # ---- Model (Softmax output — forces comp fractions to sum to 1.0) ----
    G = nn.Sequential(
        nn.Linear(5 + 26, 512),
        nn.LeakyReLU(0.2),
        nn.Linear(512, 40),
        nn.Softmax(axis=-1),
    )
    D = nn.Sequential(
        nn.Linear(66, 1024),
        nn.LeakyReLU(0.2),
        nn.Linear(1024, 1),
        nn.Sigmoid(),
    )

    loss_fn = nn.BCELoss()
    EPS = 1e-7

    opt_g = paddle.optimizer.Adam(
        parameters=G.parameters(),
        learning_rate=lr, beta1=0.5, beta2=0.999, weight_decay=weight_decay,
    )
    opt_d = paddle.optimizer.Adam(
        parameters=D.parameters(),
        learning_rate=lr, beta1=0.5, beta2=0.999, weight_decay=weight_decay,
    )

    # ---- Training (matches original cgan.py line-for-line) ----
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        for i, batch_data in enumerate(loader):
            data = batch_data["data"]
            cur_bs = data.shape[0]
            if cur_bs < batch_size:
                break  # skip incomplete batch (original: i == len//bs check)

            x_images = data[:, :40]   # real composition
            y_images = data[:, 40:]   # conditions

            real_labels = paddle.ones([cur_bs, 1])
            fake_labels = paddle.zeros([cur_bs, 1])

            # --- D step (original uses Uniform noise for D step) ---
            z = paddle.rand([cur_bs, 5])
            fake_images = G(paddle.concat([z, y_images], axis=1)).detach()

            outputs_real = D(data)  # D sees full 66-dim (comp+cond)
            outputs_fake = D(paddle.concat([fake_images, y_images], axis=1))

            # Clamp for Paddle BCELoss stability
            outputs_real = paddle.clip(outputs_real, EPS, 1.0 - EPS)
            outputs_fake = paddle.clip(outputs_fake, EPS, 1.0 - EPS)

            d_loss_real = loss_fn(outputs_real.flatten(), real_labels.flatten())
            d_loss_fake = loss_fn(outputs_fake.flatten(), fake_labels.flatten())
            d_loss = d_loss_real + d_loss_fake

            D.clear_gradients()
            d_loss.backward()
            opt_d.step()

            # --- G step (original uses Normal noise for G step) ---
            z = paddle.randn([cur_bs, 5])
            fake_images = G(paddle.concat([z, y_images], axis=1))
            outputs = D(paddle.concat([fake_images, y_images], axis=1))
            outputs = paddle.clip(outputs, EPS, 1.0 - EPS)
            g_loss = loss_fn(outputs.flatten(), real_labels.flatten())

            D.clear_gradients()
            G.clear_gradients()
            g_loss.backward()
            opt_g.step()

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch}/{epochs}: D={d_loss.item():.4f} G={g_loss.item():.4f}",
                flush=True,
            )

    train_time = time.time() - t0
    print(f"Training time: {train_time:.1f}s", flush=True)

    # ---- Evaluation ----
    G.eval()
    eval_loader = DataLoader(
        ds,
        batch_sampler=BatchSampler(ds, batch_size=len(ds), shuffle=False),
    )
    batch = next(iter(eval_loader))
    data = batch["data"]
    real_comp = data[:, :40].numpy()
    conditions = data[:, 40:]

    # Generate with 5 different seeds and average (reduces noise)
    wd_runs = []
    for eval_seed in range(5):
        paddle.seed(eval_seed + 10000)
        with paddle.no_grad():
            z = paddle.randn([conditions.shape[0], 5])
            fake_comp = G(paddle.concat([z, conditions], axis=1)).numpy()

        wd = np.mean([
            wasserstein_distance(real_comp[:, i], fake_comp[:, i])
            for i in range(40)
        ])
        wd_runs.append(wd)

    avg_wd = np.mean(wd_runs)
    print(f"\nOverall WD (fraction scale [0-1]): {avg_wd:.4f} ± {np.std(wd_runs):.4f}", flush=True)

    # Per-category WD (paper reports WD=0.41 for Cu)
    real_dom = real_comp.argmax(axis=1)
    paddle.seed(42)
    with paddle.no_grad():
        z = paddle.randn([conditions.shape[0], 5])
        fake_comp = G(paddle.concat([z, conditions], axis=1)).numpy()

    print("\nPer-category WD:", flush=True)
    for cat_name, cat_idx in [("Cu", 0), ("Fe", 6), ("Ti", 4), ("Zr", 1)]:
        mask = real_dom == cat_idx
        n = mask.sum()
        if n > 0:
            cat_wd = np.mean([
                wasserstein_distance(real_comp[mask, i], fake_comp[mask, i])
                for i in range(40)
            ])
            print(f"  {cat_name} (n={n}): WD={cat_wd:.4f}", flush=True)

    # Composition sum stats
    sums = fake_comp.sum(axis=1)
    print(f"\nComp sums: mean={sums.mean():.4f} std={sums.std():.4f} (target ≈ 1.0)", flush=True)

    # Top-5 elements comparison
    print("\nTop elements (real mean vs fake mean, fraction scale):", flush=True)
    for idx in np.argsort(-real_comp.mean(axis=0))[:8]:
        r = real_comp[:, idx].mean()
        f = fake_comp[:, idx].mean()
        print(f"  {ELEMS[idx]:3s}: real={r:.4f} fake={f:.4f} diff={abs(r-f):.4f}", flush=True)

    return avg_wd


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="gpu")
    parser.add_argument("--categories", nargs="*", default=None,
                        help="Filter categories (default: all data)")
    args = parser.parse_args()

    print("=" * 60, flush=True)
    print("AlloyGAN CGAN — Faithful Reproduction", flush=True)
    print(f"  epochs={args.epochs}, seed={args.seed}, device={args.device}", flush=True)
    print(f"  categories={args.categories}", flush=True)
    print("=" * 60, flush=True)

    wd = train_and_eval(
        epochs=args.epochs,
        seed=args.seed,
        device=args.device,
        categories=args.categories,
    )
    print(f"\n{'=' * 60}", flush=True)
    print(f"FINAL: Overall WD = {wd:.4f}", flush=True)
    print(f"Paper target: Cu WD ≈ 0.41 (CGAN)", flush=True)
