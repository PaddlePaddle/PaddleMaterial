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
AlloyGAN training script with adversarial training loop.

Implements alternating discriminator/generator updates:
    - D is trained `d_steps` times per G step
    - G is trained once per outer iteration
    - Total iterations controlled by `generator_iters`

Usage:
    python inverse_design/train.py -c inverse_design/configs/alloygan/alloygan_cgan.yaml

    # Override config values from CLI:
    python inverse_design/train.py -c <config>.yaml Trainer.generator_iters=5000
"""

import argparse
import importlib.util
import os
import os.path as osp
import sys
import time

import numpy as np
import paddle
import paddle.nn as nn
from omegaconf import OmegaConf
from paddle.io import BatchSampler
from paddle.io import DataLoader

# Add project root to path
_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))

# Import AlloyGAN components directly via importlib to bypass ppmat's eager
# __init__.py import chain (which requires pgl and other heavy graph deps).


def _import_module_from_file(name, filepath):
    """Import a single .py file as a module, bypassing package __init__.py."""
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_alloygan_mod = _import_module_from_file(
    "alloygan", osp.join(_ROOT, "ppmat", "models", "alloygan", "alloygan.py")
)
_dataset_mod = _import_module_from_file(
    "alloy_dataset", osp.join(_ROOT, "ppmat", "datasets", "alloy_dataset.py")
)

AlloyGAN = _alloygan_mod.AlloyGAN
AlloyCGAN = _alloygan_mod.AlloyCGAN
AlloyDataset = _dataset_mod.AlloyDataset

_MODEL_REGISTRY = {
    "AlloyGAN": AlloyGAN,
    "AlloyCGAN": AlloyCGAN,
}

_DATASET_REGISTRY = {
    "AlloyDataset": AlloyDataset,
}

# ---- Lightweight logger (avoids ppmat.utils.logger's heavy deps) ----
import logging as _logging
import random as _random

_log = _logging.getLogger("alloygan")


class _Logger:
    """Minimal logger matching ppmat.utils.logger API."""

    def init_logger(self, log_file=None, level=_logging.INFO):
        handler = _logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            _logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
        )
        _log.addHandler(handler)
        _log.setLevel(level)
        if log_file:
            fh = _logging.FileHandler(log_file)
            fh.setFormatter(
                _logging.Formatter("[%(asctime)s] %(message)s", datefmt="%H:%M:%S")
            )
            _log.addHandler(fh)

    def info(self, msg):
        _log.info(msg)


logger = _Logger()


class _Misc:
    @staticmethod
    def set_random_seed(seed):
        _random.seed(seed)
        np.random.seed(seed)
        paddle.seed(seed)


misc = _Misc()


def _build_model(cfg):
    cls_name = cfg["__class_name__"]
    params = cfg.get("__init_params__", {})
    return _MODEL_REGISTRY[cls_name](**params)


def _build_dataloader(cfg):
    ds_cfg = cfg["dataset"]
    cls_name = ds_cfg["__class_name__"]
    params = ds_cfg.get("__init_params__", {})
    dataset = _DATASET_REGISTRY[cls_name](**params)

    sampler_cfg = cfg.get("sampler", {})
    sampler_params = sampler_cfg.get("__init_params__", {})
    sampler = BatchSampler(
        dataset=dataset, **sampler_params
    )

    loader_cfg = cfg.get("loader", {})
    return DataLoader(
        dataset=dataset,
        batch_sampler=sampler,
        num_workers=loader_cfg.get("num_workers", 0),
        use_shared_memory=loader_cfg.get("use_shared_memory", False),
    )


def infinite_dataloader(dataloader):
    """Wrap a DataLoader to yield batches infinitely."""
    while True:
        for batch in dataloader:
            yield batch


def train_gan(config, model, train_loader):
    """Adversarial training loop — faithful port of original PyTorch AlloyGAN.

    For CGAN: epoch-based, D and G alternate 1:1 every batch (matches cgan.py).
    For GAN:  iterator-based, D trains d_steps per G step (matches gan.py).
    """
    trainer_cfg = config["Trainer"]
    epochs = trainer_cfg.get("epochs", 50)
    output_dir = trainer_cfg.get("output_dir", "./output/alloygan")
    save_freq = trainer_cfg.get("save_freq", 1000)
    log_freq = trainer_cfg.get("log_freq", 100)

    os.makedirs(output_dir, exist_ok=True)

    # Optimizers — match original: lr=0.0002, weight_decay=1e-5
    optim_cfg = config.get("Optimizer", {})
    lr = optim_cfg.get("lr", 0.0002)
    beta1 = optim_cfg.get("beta1", 0.5)
    beta2 = optim_cfg.get("beta2", 0.999)
    weight_decay = optim_cfg.get("weight_decay", 0.00001)

    opt_g = paddle.optimizer.Adam(
        parameters=model.generator.parameters(),
        learning_rate=lr,
        beta1=beta1,
        beta2=beta2,
        weight_decay=weight_decay,
    )
    opt_d = paddle.optimizer.Adam(
        parameters=model.discriminator.parameters(),
        learning_rate=lr,
        beta1=beta1,
        beta2=beta2,
        weight_decay=weight_decay,
    )

    # BCELoss with Sigmoid in D (matching original PyTorch AlloyGAN)
    # Paddle's BCELoss lacks PyTorch's internal eps clamping, so we
    # clamp D outputs to [eps, 1-eps] before loss computation.
    loss_fn = nn.BCELoss()
    _EPS = 1e-7  # clamp boundary for numerical stability

    comp_dim = model.comp_dim
    noise_dim = model.noise_dim
    is_cgan = hasattr(model, "cond_dim")
    batch_size = trainer_cfg.get("batch_size", 64)

    g_losses = []
    d_losses = []
    g_iter = 0

    logger.info(
        f"Starting training: {epochs} epochs, lr={lr}, "
        f"weight_decay={weight_decay}, batch_size={batch_size}"
    )
    start_time = time.perf_counter()

    for epoch in range(1, epochs + 1):
        for i, batch_data in enumerate(train_loader):
            data = batch_data["data"]  # (B, 66)
            cur_bs = data.shape[0]
            if cur_bs < batch_size:
                break  # skip incomplete last batch (matches original)

            real_comp = data[:, :comp_dim]
            ones = paddle.ones([cur_bs, 1])
            zeros = paddle.zeros([cur_bs, 1])

            if is_cgan:
                conditions = data[:, comp_dim:]

                # --- Train D (once per batch, matching original cgan.py) ---
                z = paddle.rand([cur_bs, noise_dim])
                g_input = paddle.concat([z, conditions], axis=1)
                fake_comp = model.generator(g_input).detach()

                d_real = model.discriminator(
                    paddle.concat([real_comp, conditions], axis=1)
                )
                d_fake = model.discriminator(
                    paddle.concat([fake_comp, conditions], axis=1)
                )
                # Clamp for Paddle BCELoss (no internal eps like PyTorch)
                d_real_c = paddle.clip(d_real, _EPS, 1.0 - _EPS)
                d_fake_c = paddle.clip(d_fake, _EPS, 1.0 - _EPS)
                d_loss = loss_fn(d_real_c, ones) + loss_fn(d_fake_c, zeros)

                opt_d.clear_grad()
                d_loss.backward()
                opt_d.step()

                # --- Train G (once per batch) ---
                # Original uses randn (normal) for G step
                z = paddle.randn([cur_bs, noise_dim])
                g_input = paddle.concat([z, conditions], axis=1)
                fake_comp = model.generator(g_input)
                d_output = model.discriminator(
                    paddle.concat([fake_comp, conditions], axis=1)
                )
                d_output_c = paddle.clip(d_output, _EPS, 1.0 - _EPS)
                g_loss = loss_fn(d_output_c, ones)

                opt_d.clear_grad()
                opt_g.clear_grad()
                g_loss.backward()
                opt_g.step()

            else:
                # Standard GAN — matches original gan.py (d_steps per G step)
                d_steps = trainer_cfg.get("d_steps", 5)
                z = paddle.rand([cur_bs, noise_dim])
                fake_comp = model.generator(z).detach()
                d_real = model.discriminator(real_comp)
                d_fake = model.discriminator(fake_comp)
                d_loss = loss_fn(d_real, ones) + loss_fn(d_fake, zeros)
                opt_d.clear_grad()
                d_loss.backward()
                opt_d.step()

                # G step every d_steps batches
                if (i + 1) % d_steps == 0:
                    z = paddle.randn([cur_bs, noise_dim])
                    fake_comp = model.generator(z)
                    d_output = model.discriminator(fake_comp)
                    g_loss = loss_fn(d_output, ones)
                    opt_d.clear_grad()
                    opt_g.clear_grad()
                    g_loss.backward()
                    opt_g.step()

            g_iter += 1
            g_losses.append(g_loss.item())
            d_losses.append(d_loss.item())

            # --- Logging ---
            if g_iter % log_freq == 0 or g_iter == 1:
                elapsed = time.perf_counter() - start_time
                logger.info(
                    f"[Epoch {epoch}/{epochs}][Batch {i+1}] "
                    f"D_loss: {d_loss.item():.4f}, G_loss: {g_loss.item():.4f}, "
                    f"g_iter: {g_iter}, Time: {elapsed:.1f}s"
                )

            # --- Save checkpoint ---
            if save_freq > 0 and g_iter % save_freq == 0:
                ckpt_path = osp.join(output_dir, f"checkpoint_iter_{g_iter}")
                os.makedirs(ckpt_path, exist_ok=True)
                paddle.save(
                    model.generator.state_dict(),
                    osp.join(ckpt_path, "generator.pdparams"),
                )
                paddle.save(
                    model.discriminator.state_dict(),
                    osp.join(ckpt_path, "discriminator.pdparams"),
                )
                np.savez(
                    osp.join(ckpt_path, "losses.npz"),
                    g_losses=np.array(g_losses),
                    d_losses=np.array(d_losses),
                )
                logger.info(f"Saved checkpoint at iter {g_iter} → {ckpt_path}")

    # Final save
    ckpt_path = osp.join(output_dir, f"checkpoint_final")
    os.makedirs(ckpt_path, exist_ok=True)
    paddle.save(
        model.generator.state_dict(),
        osp.join(ckpt_path, "generator.pdparams"),
    )
    paddle.save(
        model.discriminator.state_dict(),
        osp.join(ckpt_path, "discriminator.pdparams"),
    )
    np.savez(
        osp.join(ckpt_path, "losses.npz"),
        g_losses=np.array(g_losses),
        d_losses=np.array(d_losses),
    )

    total_time = time.perf_counter() - start_time
    logger.info(
        f"Training complete: {epochs} epochs, {g_iter} iters in {total_time:.1f}s "
        f"({total_time / max(g_iter, 1) * 1000:.1f}ms/iter)"
    )

    return g_losses, d_losses


def evaluate_cgan(config, model, test_loader, output_dir):
    """Generate compositions using trained CGAN and compute Wasserstein distance.

    Reports WD on the original paper's scale (compositions as fractions [0,1])
    and also per-category WD for comparison with Table 1 (WD=0.41 for Cu).
    """
    model.eval()
    comp_dim = model.comp_dim

    ELEMENTS = [
        "Cu", "Zr", "Al", "Ni", "Ti", "Ag", "Fe", "Mg", "B", "Si",
        "Nb", "Y", "Ca", "La", "Co", "Be", "C", "Mo", "Pd", "P",
        "Sn", "Cr", "Hf", "Zn", "Gd", "Ce", "Er", "Ga", "Au", "Nd",
        "Dy", "W", "Pr", "Ta", "Sc", "Li", "Sm", "S", "Pt", "Mn",
    ]

    all_real = []
    all_fake = []
    all_cond = []

    for batch in test_loader:
        data = batch["data"]
        real_comp = data[:, :comp_dim]
        conditions = data[:, comp_dim:]

        fake_comp = model.generate(conditions)

        all_real.append(real_comp.numpy())
        all_fake.append(fake_comp.numpy())
        all_cond.append(conditions.numpy())

    all_real = np.concatenate(all_real, axis=0)
    all_fake = np.concatenate(all_fake, axis=0)
    all_cond = np.concatenate(all_cond, axis=0)

    # Save generated data
    os.makedirs(output_dir, exist_ok=True)
    np.savez(
        osp.join(output_dir, "generated_data.npz"),
        real_comp=all_real,
        fake_comp=all_fake,
        conditions=all_cond,
    )
    logger.info(
        f"Generated {len(all_fake)} compositions → "
        f"{osp.join(output_dir, 'generated_data.npz')}"
    )

    # Compute Wasserstein distance (per-column earth mover's distance)
    from scipy.stats import wasserstein_distance

    # Overall WD
    wd_per_col = []
    for i in range(comp_dim):
        wd = wasserstein_distance(all_real[:, i], all_fake[:, i])
        wd_per_col.append(wd)
    avg_wd = np.mean(wd_per_col)
    logger.info(f"Overall WD (fraction scale): {avg_wd:.4f}")

    # Per-category WD (paper reports WD=0.41 for Cu on fraction scale)
    real_dom = all_real.argmax(axis=1)
    for cat_name, cat_idx in [("Cu", 0), ("Fe", 6), ("Ti", 4), ("Zr", 1)]:
        mask = real_dom == cat_idx
        n = mask.sum()
        if n > 0:
            cat_wd = np.mean([
                wasserstein_distance(all_real[mask, i], all_fake[mask, i])
                for i in range(comp_dim)
            ])
            logger.info(f"  {cat_name} (n={n}): WD={cat_wd:.4f}")

    # Composition sum stats
    sums = all_fake.sum(axis=1)
    logger.info(
        f"Composition sums: mean={sums.mean():.4f} std={sums.std():.4f} "
        f"(target: 1.0 on fraction scale)"
    )

    return avg_wd


def main():
    parser = argparse.ArgumentParser(description="AlloyGAN Training")
    parser.add_argument(
        "-c", "--config", type=str, required=True,
        help="Path to YAML config file",
    )
    args, dynamic_args = parser.parse_known_args()

    config = OmegaConf.load(args.config)
    cli_config = OmegaConf.from_dotlist(dynamic_args)
    config = OmegaConf.merge(config, cli_config)
    config = OmegaConf.to_container(config, resolve=True)

    output_dir = config["Trainer"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # Save config
    config_name = os.path.basename(args.config)
    OmegaConf.save(
        OmegaConf.create(config),
        osp.join(output_dir, config_name),
    )

    # Logger
    logger_path = osp.join(output_dir, "run.log")
    logger.init_logger(log_file=logger_path)
    logger.info(f"Config: {args.config}")
    logger.info(f"Output: {output_dir}")

    # Seed
    seed = config["Trainer"].get("seed", 42)
    misc.set_random_seed(seed)
    logger.info(f"Seed: {seed}")

    # Build model
    model = _build_model(config["Model"])
    total_params = sum(p.numel().item() for p in model.parameters())
    logger.info(f"Model: {config['Model']['__class_name__']} ({total_params} params)")

    # Build dataloader
    train_loader = _build_dataloader(config["Dataset"]["train"])
    logger.info(f"Training data: {len(train_loader.dataset)} samples")

    do_train = config["Global"].get("do_train", True)
    do_eval = config["Global"].get("do_eval", False)

    if do_train:
        train_gan(config, model, train_loader)

    if do_eval:
        # Load best generator if available
        ckpt_dir = config["Trainer"].get("pretrained_model_path")
        if ckpt_dir:
            g_path = osp.join(ckpt_dir, "generator.pdparams")
            if os.path.exists(g_path):
                model.generator.set_state_dict(paddle.load(g_path))
                logger.info(f"Loaded generator from {g_path}")

        test_loader = _build_dataloader(config["Dataset"].get("test"))
        if test_loader:
            evaluate_cgan(config, model, test_loader, output_dir)
        else:
            logger.info("No test set configured, skipping evaluation")


if __name__ == "__main__":
    main()
