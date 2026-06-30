# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

import os
import tempfile

import paddle

from omegaconf import OmegaConf

from ppmat.models import build_model


def _load_cfg(name):
    path = f"structure_generation/configs/chemeleon2/chemeleon2_mp20_{name}.yaml"
    return OmegaConf.to_container(OmegaConf.load(path).Model, resolve=True)


def _build_vae():
    return build_model(_load_cfg("vae"))


def _build_ldm():
    return build_model(_load_cfg("ldm"))


def _batch(num_atoms_list, seed=42):
    from ppmat.models.chemeleon2.common.schema import CrystalBatch

    paddle.seed(seed)
    total = sum(num_atoms_list)
    b = CrystalBatch()
    b.atom_types = paddle.randint(1, 90, [total])
    b.frac_coords = paddle.rand([total, 3])
    b.cart_coords = paddle.rand([total, 3])
    b.lattices = paddle.stack([paddle.eye(3) * 5 for _ in num_atoms_list])
    b.lengths = paddle.to_tensor([[5.0, 5.0, 5.0] for _ in num_atoms_list])
    b.angles = paddle.to_tensor([[90.0, 90.0, 90.0] for _ in num_atoms_list])
    b.lengths_scaled = b.lengths / paddle.to_tensor(num_atoms_list, dtype="float32").unsqueeze(-1) ** (1 / 3)
    b.angles_radians = paddle.deg2rad(b.angles)
    b.num_atoms = paddle.to_tensor(num_atoms_list)
    b.batch = paddle.repeat_interleave(paddle.arange(len(num_atoms_list)), paddle.to_tensor(num_atoms_list))
    b.token_idx = paddle.concat([paddle.arange(n) for n in num_atoms_list])
    b.num_graphs = len(num_atoms_list)
    b.mask = paddle.ones([len(num_atoms_list), max(num_atoms_list)], dtype="bool")
    return b


def test_vae_pipeline():
    vae = _build_vae()
    b = _batch([8, 12])

    paddle.seed(42)
    loss = vae.calculate_loss(b, training=False)
    assert loss["total_loss"].item() > 0
    assert not paddle.isnan(loss["total_loss"]).item()

    encoded = vae.encode(b)
    assert "x" in encoded and "moments" in encoded and "posterior" in encoded

    encoded["x"] = encoded["posterior"].sample()
    decoded = vae.decode(encoded)
    assert decoded["atom_types"].shape[0] == 20

    rec = vae.reconstruct(decoded, b)
    assert rec.lattices is not None and rec.lengths is not None

    cfg = vae.get_config()
    assert cfg is not None and "latent_dim" in cfg

    rec = vae.reconstruct(decoded, b)
    assert rec.lattices is not None


def test_ldm_pipeline():
    ldm = _build_ldm()
    b = _batch([8, 12])

    paddle.seed(42)
    loss = ldm.calculate_loss(b, training=False)
    assert loss["total_loss"].item() > 0

    with paddle.no_grad():
        for sampler in ("ddpm", "ddim"):
            r = ldm.sample(b, sampler=sampler, sampling_steps=5, progress=False)
            assert "result" in r and len(r["result"]) == 2

    with paddle.no_grad():
        r = ldm.predict({"num_samples": 2, "batch_size": 2, "num_atoms": 8}, sampling_steps=5, sampler="ddim")
    assert "result" in r

    cfg = ldm.get_config()
    assert cfg is not None and "use_cfg" in cfg

    ldm.merge_lora()
    assert ldm.lora_configs is None


def test_rl_module_build():
    from ppmat.models.chemeleon2.rl_module.rl import RLModule
    from ppmat.models.chemeleon2.rl_module.components import CustomReward

    ldm = _build_ldm()
    state_dict = ldm.state_dict()
    init_params = _load_cfg("ldm")["__init_params__"]

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "ldm_ckpt.pdparams")
        paddle.save({"model_config": init_params, "model_state_dict": state_dict}, ckpt_path)

        reward_fn = CustomReward()
        rl = RLModule(
            ldm_ckpt_path=ckpt_path,
            rl_configs={"clip_ratio": 0.2, "kl_weight": 0.1},
            reward_fn=reward_fn,
            sampling_configs={"sampler": "ddim", "sampling_steps": 5},
        )
        assert rl.ldm is not None
        assert rl.reward_fn is reward_fn


if __name__ == "__main__":
    tests = [
        test_vae_pipeline,
        test_ldm_pipeline,
        test_rl_module_build,
    ]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"[PASS] {t.__name__}")
        except Exception as e:
            print(f"[FAIL] {t.__name__}: {e}")
            import traceback

            traceback.print_exc()
            failed += 1
    print(f"\nTotal: {len(tests)}, Passed: {len(tests) - failed}, Failed: {failed}")
    exit(1 if failed > 0 else 0)
