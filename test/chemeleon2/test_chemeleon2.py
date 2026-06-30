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

import paddle

from ppmat.models.chemeleon2.vae_module.vae import VAEModule
from ppmat.models.chemeleon2.vae_module.encoder import TransformerEncoder
from ppmat.models.chemeleon2.vae_module.decoder import TransformerDecoder
from ppmat.models.chemeleon2.ldm_module.ldm import LDMModule
from ppmat.models.chemeleon2.ldm_module.dit import DiT
from ppmat.models.chemeleon2.ldm_module.condition import ConditionModule, ConditionType
from ppmat.models.chemeleon2.common.schema import CrystalBatch, create_empty_batch
from ppmat.utils.scatter import scatter_mean
from ppmat.utils.crystal import lattice_params_to_matrix_paddle as lattice_params_to_matrix
from ppmat.models.chemeleon2.common.utils import lattice_vector_to_volume
from ppmat.models.chemeleon2.common import (
    DiagonalGaussianDistribution, to_dense_batch,
    get_index_embedding, apply_augmentation, apply_noise,
    LoRALayer, apply_lora_to_linear, merge_lora_weights,
)

SMALL = dict(d_model=64, nhead=2, dim_feedforward=256, num_layers=2)


def _batch(num_atoms_list, seed=42):
    paddle.seed(seed)
    total = sum(num_atoms_list)
    b = CrystalBatch()
    b.atom_types = paddle.randint(1, 90, [total])
    b.frac_coords = paddle.rand([total, 3])
    b.cart_coords = paddle.rand([total, 3])
    b.lattices = paddle.stack([paddle.eye(3) * 5 for _ in num_atoms_list])
    b.lengths = paddle.to_tensor([[5.0, 5.0, 5.0] for _ in num_atoms_list])
    b.angles = paddle.to_tensor([[90.0, 90.0, 90.0] for _ in num_atoms_list])
    b.lengths_scaled = b.lengths / paddle.to_tensor(num_atoms_list, dtype='float32').unsqueeze(-1) ** (1/3)
    b.angles_radians = paddle.deg2rad(b.angles)
    b.num_atoms = paddle.to_tensor(num_atoms_list)
    b.batch = paddle.repeat_interleave(paddle.arange(len(num_atoms_list)), paddle.to_tensor(num_atoms_list))
    b.token_idx = paddle.concat([paddle.arange(n) for n in num_atoms_list])
    b.num_graphs = len(num_atoms_list)
    b.mask = paddle.ones([len(num_atoms_list), max(num_atoms_list)], dtype='bool')
    return b


# 1. common utils

def test_diagonal_gaussian():
    params = paddle.randn([4, 8])
    dist = DiagonalGaussianDistribution(params)
    assert dist.mean.shape == [4, 4]
    z = dist.sample()
    assert z.shape == [4, 4]
    assert dist.kl().shape == [4]
    assert dist.mode().shape == [4, 4]
    print("test_diagonal_gaussian: PASS")


def test_scatter_mean():
    src = paddle.to_tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    idx = paddle.to_tensor([0, 0, 1, 1])
    out = scatter_mean(src, idx, dim=0)
    assert out.shape == [2, 2]
    assert float(out[0, 0]) == 2.0
    print("test_scatter_mean: PASS")


def test_to_dense_batch():
    x = paddle.randn([10, 4])
    xd, mask = to_dense_batch(x, paddle.to_tensor([0]*5 + [1]*5))
    assert xd.shape == [2, 5, 4]
    assert mask.all()
    print("test_to_dense_batch: PASS")


def test_get_index_embedding():
    emb = get_index_embedding(paddle.arange(5), 16)
    assert emb.shape == [5, 16]
    print("test_get_index_embedding: PASS")


def test_lattice_utils():
    lat = lattice_params_to_matrix(paddle.to_tensor([[5.0, 5.0, 5.0]]), paddle.to_tensor([[90.0, 90.0, 90.0]]))
    assert lat.shape == [1, 3, 3]
    vol = lattice_vector_to_volume(lat)
    assert abs(float(vol[0]) - 125.0) < 0.5
    print("test_lattice_utils: PASS")


def test_data_augmentation():
    b = _batch([8, 12])
    aug = apply_augmentation(b, translate=True, rotate=False)
    assert aug.frac_coords is not None
    noisy = apply_noise(b, ratio=0.5, corruption_scale=0.1)
    assert noisy.atom_types is not None
    print("test_data_augmentation: PASS")


def test_lora():
    lora = LoRALayer(16, 32, rank=4)
    out = lora(paddle.randn([2, 16]))
    assert out.shape == [2, 32]
    seq = paddle.nn.Sequential(paddle.nn.Linear(16, 32))
    seq = apply_lora_to_linear(seq, rank=4)
    seq = merge_lora_weights(seq)
    assert len(seq.sublayers()) >= 1
    print("test_lora: PASS")


# 2. VAE

def _make_vae():
    return VAEModule(
        TransformerEncoder(**SMALL), TransformerDecoder(**SMALL),
        latent_dim=4, loss_weights={"atom_types": 1.0, "lengths": 1.0, "angles": 1.0, "frac_coords": 1.0, "kl": 1e-5},
    )


def test_vae_encode():
    encoded = _make_vae().encode(_batch([8, 12]))
    assert "x" in encoded and "moments" in encoded and "posterior" in encoded
    assert encoded["x"].shape[0] == 20
    print("test_vae_encode: PASS")


def test_vae_decode():
    vae = _make_vae()
    encoded = vae.encode(_batch([8, 12]))
    encoded["x"] = encoded["posterior"].sample()
    decoded = vae.decode(encoded)
    assert decoded["atom_types"].shape[0] == 20
    print("test_vae_decode: PASS")


def test_vae_reconstruct():
    vae = _make_vae()
    batch = _batch([8, 12])
    encoded = vae.encode(batch)
    encoded["x"] = encoded["posterior"].sample()
    rec = vae.reconstruct(vae.decode(encoded), batch)
    assert rec.lattices is not None and rec.lengths is not None
    print("test_vae_reconstruct: PASS")


def test_vae_training_loss():
    vae = _make_vae()
    paddle.seed(42)
    loss = vae.calculate_loss(_batch([8, 12]), training=False)
    assert loss["total_loss"].item() > 0
    for k in ("loss_atom_types", "loss_lengths", "loss_angles", "loss_frac_coords", "loss_kl"):
        assert k in loss
    print("test_vae_training_loss: PASS")


def test_vae_forward_api():
    batch = _batch([8, 12])
    sa = {"atom_types": batch.atom_types, "frac_coords": batch.frac_coords,
          "num_atoms": batch.num_atoms, "lengths": batch.lengths, "angles": batch.angles, "lattice": batch.lattices}
    cb = _make_vae()._convert_train_batch({"structure_array": sa})
    assert cb.lengths_scaled is not None and cb.angles_radians is not None and cb.cart_coords is not None
    print("test_vae_forward_api: PASS")


# 3. DiT

def test_dit_forward():
    dit = DiT(input_dim=4, hidden_dim=64, num_heads=2, num_layers=2, learn_sigma=False)
    out = dit(paddle.randn([2, 8, 4]), paddle.randint(0, 100, [2]), mask=paddle.ones([2, 8], dtype='bool'))
    assert out.shape == [2, 8, 4]
    print("test_dit_forward: PASS")


def test_dit_learn_sigma():
    dit = DiT(input_dim=4, hidden_dim=64, num_heads=2, num_layers=2, learn_sigma=True)
    out = dit(paddle.randn([2, 8, 4]), paddle.randint(0, 100, [2]), mask=paddle.ones([2, 8], dtype='bool'))
    assert out.shape == [2, 16, 4]
    print("test_dit_learn_sigma: PASS")


def test_dit_cfg():
    dit = DiT(input_dim=4, hidden_dim=64, num_heads=2, num_layers=2, learn_sigma=False, condition_dim=16)
    out = dit.forward_with_cfg(paddle.randn([4, 8, 4]), paddle.randint(0, 100, [4]),
                                paddle.ones([4, 8], dtype='bool'), paddle.randn([4, 16]), cfg_scale=2.0)
    assert out.shape == [4, 8, 4]
    print("test_dit_cfg: PASS")


# 4. LDM

def _make_ldm(vae=None):
    if vae is None:
        vae = _make_vae()
    return LDMModule(
        denoiser=DiT(input_dim=4, hidden_dim=64, num_heads=2, num_layers=2, learn_sigma=True),
        normalize_latent=True,
        diffusion_configs={"timestep_respacing": "", "noise_schedule": "linear", "diffusion_steps": 20, "learn_sigma": True},
        vae=vae,
    )


def test_ldm_training_loss():
    ldm = _make_ldm()
    paddle.seed(42)
    loss = ldm.calculate_loss(_batch([8, 12]), training=False)
    assert loss["total_loss"].item() > 0
    print("test_ldm_training_loss: PASS")


def _sample(b, ldm, sampler):
    b.mask = paddle.ones([2, 12], dtype='bool')
    paddle.seed(42)
    with paddle.no_grad():
        return ldm.sample(b, sampler=sampler, sampling_steps=5, progress=False)


def test_ldm_ddpm_sample():
    result = _sample(_batch([8, 12]), _make_ldm(), "ddpm")
    assert len(result["result"]) == 2
    print("test_ldm_ddpm_sample: PASS")


def test_ldm_ddim_sample():
    result = _sample(_batch([8, 12]), _make_ldm(), "ddim")
    assert len(result["result"]) == 2
    print("test_ldm_ddim_sample: PASS")


def test_ldm_predict():
    paddle.seed(42)
    with paddle.no_grad():
        result = _make_ldm().predict({"num_samples": 2, "batch_size": 2, "num_atoms": 8}, sampling_steps=5, sampler="ddim")
    assert "result" in result
    print("test_ldm_predict: PASS")


def test_ldm_cond_sample():
    cond = ConditionModule(condition_type={"target": ConditionType.VALUE.value}, hidden_dim=16, drop_prob=0.1,
                            stats={"target": {"mean": 0.0, "std": 1.0}})
    ldm = LDMModule(
        denoiser=DiT(input_dim=4, hidden_dim=64, num_heads=2, num_layers=1, learn_sigma=False, condition_dim=16),
        normalize_latent=True, condition_module=cond,
        diffusion_configs={"timestep_respacing": "", "noise_schedule": "linear", "diffusion_steps": 10},
        vae=_make_vae(),
    )
    b = _batch([8, 12])
    b.y = {"target": [0.5, 0.8]}
    b.mask = paddle.ones([2, 12], dtype='bool')
    paddle.seed(42)
    with paddle.no_grad():
        result = ldm.sample(b, sampler="ddim", sampling_steps=5, progress=False, cfg_scale=2.0)
    assert "result" in result
    print("test_ldm_cond_sample: PASS")


# 5. condition

def test_condition_module():
    cond = ConditionModule(condition_type={"target": ConditionType.VALUE.value}, hidden_dim=16, drop_prob=0.1,
                            stats={"target": {"mean": 0.0, "std": 1.0}})
    out = cond({"target": [0.5, 0.8]}, training=False)
    assert out.shape == [4, 16]
    print("test_condition_module: PASS")


# 6. schema

def test_crystal_batch_schema():
    b = _batch([5, 7])
    assert b.num_graphs == 2
    assert b.atom_types.shape[0] == 12
    assert len(b._split_by_batch_index()) == 2
    print("test_crystal_batch_schema: PASS")


def test_create_empty_batch():
    b = create_empty_batch([6, 10])
    assert hasattr(b, 'atom_types') and hasattr(b, 'batch')
    print("test_create_empty_batch: PASS")

# 7. full pipeline (build -> train -> sample)

def test_e2e_full_pipeline():
    from omegaconf import OmegaConf
    from ppmat.models import build_model

    vae_cfg = OmegaConf.to_container(OmegaConf.load("structure_generation/configs/chemeleon2/chemeleon2_mp20_vae.yaml").Model, resolve=True)
    vae = build_model(vae_cfg)
    b = _batch([8, 12])
    paddle.seed(42)
    vae_loss = vae.calculate_loss(b, training=False)
    assert vae_loss["total_loss"].item() > 0

    encoded = vae.encode(b)
    z = encoded["posterior"].sample()
    encoded["x"] = z
    decoded = vae.decode(encoded)
    rec = vae.reconstruct(decoded, b)
    assert rec.lattices is not None

    ldm_cfg = OmegaConf.to_container(OmegaConf.load("structure_generation/configs/chemeleon2/chemeleon2_mp20_ldm.yaml").Model, resolve=True)
    ldm = build_model(ldm_cfg)
    b.mask = paddle.ones([2, 12], dtype='bool')
    paddle.seed(42)
    ldm_loss = ldm.calculate_loss(b, training=False)
    assert ldm_loss["total_loss"].item() > 0

    with paddle.no_grad():
        result = ldm.predict({"num_samples": 2, "batch_size": 2, "num_atoms": 8}, sampling_steps=5, sampler="ddim")
    assert "result" in result

    with paddle.no_grad():
        result = ldm.sample(b, sampler="ddpm", sampling_steps=5, progress=False)
    assert len(result["result"]) == 2

    print("test_e2e_full_pipeline: PASS")

def test_ldm_lora():
    ldm = LDMModule(
        denoiser=DiT(input_dim=4, hidden_dim=64, num_heads=2, num_layers=1),
        normalize_latent=True,
        diffusion_configs={"timestep_respacing": "", "noise_schedule": "linear", "diffusion_steps": 10},
        vae=_make_vae(),
        lora_configs={"r": 4, "lora_alpha": 8, "lora_dropout": 0.0},
    )
    assert ldm.lora_configs is not None
    paddle.seed(42)
    loss = ldm.calculate_loss(_batch([8, 12]), training=False)
    assert loss["total_loss"].item() > 0
    ldm.merge_lora()
    assert ldm.lora_configs is None
    print("test_ldm_lora: PASS")


if __name__ == "__main__":
    groups = [
        ("common", [test_diagonal_gaussian, test_scatter_mean, test_to_dense_batch,
                     test_get_index_embedding, test_lattice_utils, test_data_augmentation, test_lora]),
        ("vae", [test_vae_encode, test_vae_decode, test_vae_reconstruct,
                 test_vae_training_loss, test_vae_forward_api]),
        ("dit", [test_dit_forward, test_dit_learn_sigma, test_dit_cfg]),
        ("ldm", [test_ldm_training_loss, test_ldm_ddpm_sample, test_ldm_ddim_sample,
                 test_ldm_predict, test_ldm_cond_sample]),
        ("condition", [test_condition_module]),
        ("schema", [test_crystal_batch_schema, test_create_empty_batch]),
        ("lora", [test_ldm_lora]),
        ("e2e", [test_e2e_full_pipeline]),
    ]
    failed = 0
    total = 0
    for name, tests in groups:
        print(f"\n[{name}]")
        for t in tests:
            total += 1
            try:
                t()
            except Exception as e:
                print(f"  {t.__name__}: FAIL ({e})")
                failed += 1
    print(f"\nTotal: {total}, Passed: {total - failed}, Failed: {failed}")
    exit(1 if failed > 0 else 0)
