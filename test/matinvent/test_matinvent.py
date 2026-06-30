# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import tempfile

import numpy as np
import pytest
from omegaconf import OmegaConf
from pymatgen.core.structure import Lattice
from pymatgen.core.structure import Structure


@pytest.fixture(autouse=True)
def seed_random_state(seed: int = 42):
    np.random.seed(seed)
    yield


def _make_structure(a=5.0, b=5.0, c=5.0, species=("Si",), frac_coords=((0.0, 0.0, 0.0),)):
    lattice = Lattice.from_parameters(a, b, c, 90, 90, 90)
    return Structure(lattice=lattice, species=species, coords=frac_coords)


def _make_si_structure(a=5.43, b=None, c=None):
    _b = b if b is not None else a
    _c = c if c is not None else a
    return _make_structure(a, _b, _c, species=("Si", "Si"), frac_coords=((0.0, 0.0, 0.0), (0.25, 0.25, 0.25)))


class TestImports:
    def test_import_matinvent(self):
        from ppmat.models.matinvent import MatInvent, MatinventRL
        assert MatInvent is not None
        assert MatinventRL is not None

    def test_import_memory(self):
        from ppmat.models.matinvent.memory import LongTimeMem, ReplayBuffer
        assert LongTimeMem is not None
        assert ReplayBuffer is not None

    def test_import_reward(self):
        from ppmat.models.matinvent.rewards import Reward, Calculator
        assert Reward is not None
        assert Calculator is not None

    def test_import_dataset(self):
        from ppmat.models.matinvent.dataset import RLDataset, create_rl_dataloader, is_valid_structure, save_structures
        assert RLDataset is not None
        assert create_rl_dataloader is not None
        assert is_valid_structure is not None
        assert save_structures is not None

    def test_import_suites(self):
        from ppmat.models.matinvent.suites import ModelSuite, MatterGenAdapter
        assert ModelSuite is not None
        assert MatterGenAdapter is not None

    def test_import_sampler(self):
        from ppmat.sampler.matinvent import BaseSampler, MatterGenSampler, DiffCSPSampler
        assert BaseSampler is not None
        assert MatterGenSampler is not None
        assert DiffCSPSampler is not None


class TestDataset:
    def test_is_valid_structure_none(self):
        from ppmat.models.matinvent.dataset import is_valid_structure
        assert not is_valid_structure(None)

    def test_is_valid_structure_empty(self):
        from ppmat.models.matinvent.dataset import is_valid_structure
        lattice = Lattice.from_parameters(3.0, 3.0, 3.0, 90, 90, 90)
        empty_s = Structure(lattice=lattice, species=[], coords=[])
        assert not is_valid_structure(empty_s)

    def test_is_valid_structure_normal(self):
        from ppmat.models.matinvent.dataset import is_valid_structure
        s = _make_si_structure()
        assert is_valid_structure(s)

    def test_is_valid_structure_small_volume(self):
        from ppmat.models.matinvent.dataset import is_valid_structure
        s = _make_structure(a=0.1, b=0.1, c=0.1, species=("Si",), frac_coords=((0.0, 0.0, 0.0),))
        assert not is_valid_structure(s, min_volume=1.0)

    def test_is_valid_structure_large_lattice(self):
        from ppmat.models.matinvent.dataset import is_valid_structure
        s = _make_structure(a=30.0, b=5.0, c=5.0, species=("Si",), frac_coords=((0.0, 0.0, 0.0),))
        assert not is_valid_structure(s, max_lattice_param=20.0)

    def test_rl_dataset_construction(self):
        from ppmat.models.matinvent.dataset import RLDataset
        structures = [_make_si_structure(), _make_structure(a=4.0, b=4.0, c=4.0, species=("C",),
                       frac_coords=((0.0, 0.0, 0.0),))]
        rewards = np.array([0.8, 0.3], dtype=np.float32)
        ds = RLDataset(structures, rewards)
        assert len(ds) == 2

    def test_rl_dataset_length_mismatch(self):
        from ppmat.models.matinvent.dataset import RLDataset
        structures = [_make_si_structure()]
        rewards = np.array([0.8, 0.3], dtype=np.float32)
        with pytest.raises(AssertionError):
            RLDataset(structures, rewards)

    def test_rl_dataset_getitem(self):
        from ppmat.models.matinvent.dataset import RLDataset
        s = _make_si_structure()
        ds = RLDataset([s], np.array([0.5], dtype=np.float32))
        item = ds[0]
        assert "structure_array" in item
        sa = item["structure_array"]
        assert "frac_coords" in sa
        assert "atom_types" in sa
        assert "lattice" in sa
        assert "num_atoms" in sa
        assert "reward" in sa
        assert sa["num_atoms"] == 2

    def test_save_structures(self):
        from ppmat.models.matinvent.dataset import save_structures
        s = _make_si_structure()
        with tempfile.TemporaryDirectory() as tmp:
            path = save_structures([s], tmp, "test.extxyz")
            assert os.path.exists(path)

    def test_create_rl_dataloader(self):
        from ppmat.models.matinvent.dataset import create_rl_dataloader
        structures = [_make_si_structure() for _ in range(4)]
        rewards = np.array([0.5, 0.6, 0.7, 0.8], dtype=np.float32)
        dl = create_rl_dataloader(structures, rewards, batch_size=2)
        assert len(dl.dataset) == 4

    def test_rl_dataloader_collate(self):
        from ppmat.models.matinvent.dataset import RLDataset
        s = _make_si_structure()
        ds = RLDataset([s, s], np.array([0.5, 0.6], dtype=np.float32))
        batch = [ds[0], ds[1]]
        collated = RLDataset.collate_fn(batch)
        assert "structure_array" in collated
        sa = collated["structure_array"]
        assert "batch" in sa


class TestMemory:
    def test_long_term_mem_init(self):
        from ppmat.models.matinvent.memory import LongTimeMem
        ltm = LongTimeMem()
        assert len(ltm) == 0
        assert ltm.unique_comps == []

    def test_long_term_mem_extend(self):
        from ppmat.models.matinvent.memory import LongTimeMem
        ltm = LongTimeMem()
        s1 = _make_si_structure()
        s2 = _make_structure(a=4.0, b=4.0, c=4.0, species=("C",), frac_coords=((0.0, 0.0, 0.0),))
        ltm.extend([s1, s2], np.array([0.8, 0.3]), step=0)
        assert len(ltm) == 2
        assert len(ltm.unique_comps) == 2

    def test_long_term_mem_accumulate(self):
        from ppmat.models.matinvent.memory import LongTimeMem
        ltm = LongTimeMem()
        s1 = _make_si_structure()
        ltm.extend([s1], np.array([0.8]), step=0)
        ltm.extend([s1], np.array([0.9]), step=1)
        assert len(ltm) == 2

    def test_long_term_mem_calc_metrics(self):
        from ppmat.models.matinvent.memory import LongTimeMem
        ltm = LongTimeMem()
        structures = []
        rewards = []
        for i in range(5):
            species = (f"X{i}",)
            s = _make_structure(a=3.0 + i * 0.1, b=3.0, c=3.0, species=species, frac_coords=((0.0, 0.0, 0.0),))
            structures.append(s)
            rewards.append(0.5 + i * 0.1)
        ltm.extend(structures, np.array(rewards), step=0)
        burden, div_ratio = ltm.calc_metrics(threshold=0.5, budget=100, num_candidate=3)
        assert isinstance(burden, (float, type(None)))
        assert isinstance(div_ratio, (float, type(None)))

    def test_long_term_mem_get_baseline(self):
        from ppmat.models.matinvent.memory import LongTimeMem
        ltm = LongTimeMem()
        s1 = _make_si_structure()
        ltm.extend([s1], np.array([0.8]), step=0)
        ltm.extend([s1], np.array([0.9]), step=1)
        ltm.extend([s1], np.array([0.7]), step=2)
        baseline = ltm.get_baseline(step=3, prev=3)
        assert baseline is not None and not np.isnan(baseline)

    def test_long_term_mem_save(self):
        from ppmat.models.matinvent.memory import LongTimeMem
        ltm = LongTimeMem()
        s1 = _make_si_structure()
        ltm.extend([s1], np.array([0.8]), step=0)
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "ltm.csv")
            ltm.save(csv_path)
            assert os.path.exists(csv_path)

    def test_div_filter(self):
        from ppmat.models.matinvent.memory import LongTimeMem
        ltm = LongTimeMem()
        s_si = _make_si_structure()
        ltm.extend([s_si, s_si], np.array([0.8, 0.9]), step=0)
        s_new = _make_structure(a=4.0, b=4.0, c=4.0, species=("C",), frac_coords=((0.0, 0.0, 0.0),))
        new_rewards = np.array([0.7, 0.6])
        result, penalty_idx, tol_n, buff_n = ltm.div_filter(
            [s_si, s_new], new_rewards, tol=3, buff=10, method="composition")
        assert isinstance(result, np.ndarray)
        assert isinstance(penalty_idx, list)
        assert isinstance(tol_n, int)
        assert isinstance(buff_n, int)

    def test_replay_buffer_init(self):
        from ppmat.models.matinvent.memory import ReplayBuffer
        rb = ReplayBuffer(buffer_size=50, sample_size=4, reward_cutoff=0.0)
        assert len(rb) == 0

    def test_replay_buffer_extend_and_sample(self):
        from ppmat.models.matinvent.memory import ReplayBuffer
        rb = ReplayBuffer(buffer_size=20, sample_size=4)
        s1 = _make_si_structure()
        s2 = _make_structure(a=4.0, b=4.0, c=4.0, species=("C",), frac_coords=((0.0, 0.0, 0.0),))
        rb.extend([{"id": 0}, {"id": 1}], [s1, s2], np.array([0.8, 0.3]))
        assert len(rb) == 2
        data, rewards = rb.sample()
        assert isinstance(data, list)
        assert isinstance(rewards, np.ndarray)
        assert len(data) <= 4
        assert len(rewards) == len(data)

    def test_replay_buffer_memory_purge(self):
        from ppmat.models.matinvent.memory import ReplayBuffer
        rb = ReplayBuffer(buffer_size=20, sample_size=4)
        s1 = _make_si_structure()
        s2 = _make_structure(a=4.0, b=4.0, c=4.0, species=("C",), frac_coords=((0.0, 0.0, 0.0),))
        rb.extend([{"id": 0}, {"id": 1}], [s1, s2], np.array([0.8, 0.3]))
        rb.memory_purge([s1])
        assert len(rb) == 1

    def test_replay_buffer_empty_sample(self):
        from ppmat.models.matinvent.memory import ReplayBuffer
        rb = ReplayBuffer(buffer_size=20, sample_size=4)
        data, rewards = rb.sample()
        assert data == []
        assert len(rewards) == 0




def _cfg_density(tmp_dir=None):
    d = OmegaConf.create({
        "name": "density",
        "calculator": {"__class_name__": "PyMatGen", "task": "density"},
        "target": "ascending",
        "minv": 1.0,
        "maxv": 10.0,
    })
    if tmp_dir:
        d.calculator["root_dir"] = tmp_dir
    return d


def _cfg_num_atoms(tmp_dir=None):
    d = OmegaConf.create({
        "name": "num_atoms",
        "calculator": {"__class_name__": "PyMatGen", "task": "num_atoms"},
        "target": "ascending",
        "minv": 0.0,
        "maxv": 20.0,
        "weight": 0.4,
    })
    if tmp_dir:
        d.calculator["root_dir"] = tmp_dir
    return d


def _cfg_hhi(tmp_dir=None):
    d = OmegaConf.create({
        "name": "hhi",
        "calculator": {"__class_name__": "PyMatGen", "task": "hhi"},
        "target": "descending",
        "minv": 0.0,
        "maxv": 100.0,
    })
    if tmp_dir:
        d.calculator["root_dir"] = tmp_dir
    return d


def _cfg_nonexistent(tmp_dir=None):
    d = OmegaConf.create({
        "name": "test",
        "calculator": {"__class_name__": "NonExistent"},
        "target": "ascending",
        "minv": 0.0,
        "maxv": 1.0,
    })
    if tmp_dir:
        d.calculator["root_dir"] = tmp_dir
    return d


class TestReward:
    def test_linear_scale(self):
        from ppmat.models.matinvent.rewards.reward import _linear_scale
        values = np.array([0.0, 3.0, 6.0, -1.0, 7.0])
        result = _linear_scale(values, minv=0.0, maxv=6.0)
        expected = np.array([0.0, 0.5, 1.0, 0.0, 1.0])
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_reward_init(self):
        from ppmat.models.matinvent.rewards.reward import Reward
        with tempfile.TemporaryDirectory() as tmp:
            prop_cfg = [_cfg_density(tmp)]
            r = Reward(root_dir=tmp, prop_cfg=prop_cfg, reward_threshold=0.5, reduce="mean")
            assert r.threshold == 0.5
            assert r.reduce == "mean"

    def test_reward_scoring(self):
        from ppmat.models.matinvent.rewards.reward import Reward
        s1 = _make_si_structure()
        s2 = _make_structure(a=4.0, b=4.0, c=4.0, species=("C",), frac_coords=((0.0, 0.0, 0.0),))
        with tempfile.TemporaryDirectory() as tmp:
            calc_dir = os.path.join(tmp, "calcs")
            prop_cfg = [_cfg_density(calc_dir)]
            r = Reward(root_dir=calc_dir, prop_cfg=prop_cfg, reward_threshold=0.5, reduce="mean")
            rewards, prop_dict, failed_mask = r.scoring(([s1, s2], tmp), "test")
            assert isinstance(rewards, np.ndarray)
            assert len(rewards) == 2
            assert "density" in prop_dict
            assert isinstance(failed_mask, np.ndarray)

    def test_reward_scoring_weight_reduce(self):
        from ppmat.models.matinvent.rewards.reward import Reward
        s1 = _make_si_structure()
        with tempfile.TemporaryDirectory() as tmp:
            calc_dir = os.path.join(tmp, "calcs")
            density_cfg = _cfg_density(calc_dir)
            density_cfg.weight = 0.6
            num_atoms_cfg = _cfg_num_atoms(calc_dir)
            prop_cfg = [density_cfg, num_atoms_cfg]
            r = Reward(root_dir=calc_dir, prop_cfg=prop_cfg, reward_threshold=0.5, reduce="weight")
            rewards, prop_dict, failed_mask = r.scoring(([s1], tmp), "test")
            assert len(rewards) == 1

    def test_reward_scoring_descending(self):
        from ppmat.models.matinvent.rewards.reward import Reward
        s1 = _make_si_structure()
        with tempfile.TemporaryDirectory() as tmp:
            calc_dir = os.path.join(tmp, "calcs")
            prop_cfg = [_cfg_hhi(calc_dir)]
            r = Reward(root_dir=calc_dir, prop_cfg=prop_cfg, reward_threshold=0.0, reduce="mean")
            rewards, _, _ = r.scoring(([s1], tmp), "test")
            assert len(rewards) == 1

    def test_unknown_calculator_raises(self):
        from ppmat.models.matinvent.rewards.reward import Reward
        s1 = _make_si_structure()
        with tempfile.TemporaryDirectory() as tmp:
            calc_dir = os.path.join(tmp, "calcs")
            prop_cfg = [_cfg_nonexistent(calc_dir)]
            r = Reward(root_dir=calc_dir, prop_cfg=prop_cfg, reward_threshold=0.5)
            with pytest.raises(ValueError, match="Unknown calculator"):
                r.scoring(([s1], tmp), "test")


class TestSampler:
    def test_base_sampler_abstract(self):
        from ppmat.sampler.matinvent import BaseSampler
        sampler = BaseSampler(batch_size=4, num_batches=2, num_inference_steps=100)
        assert sampler.batch_size == 4
        assert sampler.num_batches == 2
        assert sampler.num_inference_steps == 100
        with pytest.raises(NotImplementedError):
            sampler.generate(None)

    def test_mattergen_sampler_init(self):
        from ppmat.sampler.matinvent import MatterGenSampler
        sampler = MatterGenSampler(batch_size=8, num_batches=2, num_inference_steps=50)
        assert sampler.batch_size == 8
        assert sampler.num_batches == 2
        assert sampler.num_inference_steps == 50

    def test_diffcsp_sampler_init(self):
        from ppmat.sampler.matinvent import DiffCSPSampler
        sampler = DiffCSPSampler(batch_size=8, num_batches=2, num_inference_steps=500)
        assert sampler.batch_size == 8
        assert sampler.num_batches == 2
        assert sampler.num_inference_steps == 500

    def test_process_result_dict(self):
        from ppmat.sampler.matinvent import _process_result_dict
        r = {
            "num_atoms": 2,
            "frac_coords": np.array([[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]], dtype=np.float32),
            "atom_types": np.array([14, 14], dtype=np.int32),
            "lattice": np.eye(3, dtype=np.float32) * 5.43,
        }
        sd, pmg = _process_result_dict(r)
        assert isinstance(sd, dict)
        assert pmg is not None

    def test_process_result_dict_invalid(self):
        from ppmat.sampler.matinvent import _process_result_dict
        r = {
            "num_atoms": 2,
            "frac_coords": np.array([[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]], dtype=np.float32),
            "atom_types": np.array([0, 0], dtype=np.int32),
            "lattice": np.eye(3, dtype=np.float32) * 5.43,
        }
        sd, pmg = _process_result_dict(r)
        assert isinstance(sd, dict)
        assert pmg is None


class TestPyMatGenCalculator:
    def test_pymatgen_calc_density(self):
        from ppmat.models.matinvent.rewards.calculators.pymatgen import PyMatGen
        s1 = _make_si_structure()
        s2 = _make_structure(a=4.0, b=4.0, c=4.0, species=("C",), frac_coords=((0.0, 0.0, 0.0),))
        with tempfile.TemporaryDirectory() as tmp:
            c = PyMatGen(root_dir=tmp, task="density")
            results = c.calc(([s1, s2], tmp), label="test")
            assert len(results) == 2
            assert results[0] > 0

    def test_pymatgen_calc_num_atoms(self):
        from ppmat.models.matinvent.rewards.calculators.pymatgen import PyMatGen
        s1 = _make_si_structure()
        with tempfile.TemporaryDirectory() as tmp:
            c = PyMatGen(root_dir=tmp, task="num_atoms")
            results = c.calc(([s1], tmp), label="test")
            assert len(results) == 1
            assert results[0] == 2

    def test_pymatgen_calc_num_elements(self):
        from ppmat.models.matinvent.rewards.calculators.pymatgen import PyMatGen
        s1 = _make_si_structure()
        with tempfile.TemporaryDirectory() as tmp:
            c = PyMatGen(root_dir=tmp, task="num_elements")
            results = c.calc(([s1], tmp), label="test")
            assert len(results) == 1
            assert results[0] == 1

    def test_pymatgen_calc_volume(self):
        from ppmat.models.matinvent.rewards.calculators.pymatgen import PyMatGen
        s1 = _make_si_structure(a=5.0, b=5.0, c=5.0)
        with tempfile.TemporaryDirectory() as tmp:
            c = PyMatGen(root_dir=tmp, task="volume")
            results = c.calc(([s1], tmp), label="test")
            assert len(results) == 1
            assert results[0] > 0

    def test_pymatgen_calc_unknown_task(self):
        from ppmat.models.matinvent.rewards.calculators.pymatgen import PyMatGen
        s1 = _make_si_structure()
        with tempfile.TemporaryDirectory() as tmp:
            c = PyMatGen(root_dir=tmp, task="unknown_task")
            with pytest.raises(ValueError, match="Unknown task"):
                c.calc(([s1], tmp), label="test")


class TestMatterGenAdapter:
    def test_adapter_passthrough(self):
        from ppmat.models.matinvent.suites import MatterGenAdapter
        import paddle.nn as nn

        class DummyModel(nn.Layer):
            def __init__(self):
                super().__init__()
                self.time_dim = 256
                self.lattice_loss_weight = 0.5
                self.coord_loss_weight = 0.1
                self.atom_loss_weight = 1.0
                self.param0 = self.create_parameter(shape=[1], default_initializer=nn.initializer.Constant(1.0))

            def forward(self, x):
                return x

        model = DummyModel()
        adapted = MatterGenAdapter(model)
        assert adapted.time_dim == 256
        assert adapted.lattice_loss_weight == 0.5
        assert adapted.coord_loss_weight == 0.1
        assert adapted.atom_loss_weight == 1.0
        assert adapted.d3pm_hybrid_lambda is None

    @pytest.mark.skip(reason="requires paddle compiled with cuda or pretrained weight files")
    def test_model_suite_load_model(self):
        from ppmat.models.matinvent.suites import ModelSuite
        with tempfile.TemporaryDirectory() as tmp:
            from ppmat.models.matinvent.suites import _DIFFCSP_DEFAULT
            from ppmat.models.diffcsp.diffcsp import DiffCSP
            import paddle
            model = DiffCSP(**_DIFFCSP_DEFAULT)
            model.eval()
            weight_path = os.path.join(tmp, "dummy.pdparams")
            paddle.save(model.state_dict(), weight_path)
            suite = ModelSuite(
                model_name="diffcsp",
                sample_cfg={"batch_size": 2, "num_batches": 1, "num_inference_steps": 10},
                finetune_cfg={"batch_size": 2, "epochs": 1, "lr": 1e-4, "accum_steps": 1, "timesteps": 1},
                pretrained_model_path=weight_path,
                device="cpu",
            )
            loaded = suite.load_model()
            assert loaded is not None

    @pytest.mark.skip(reason="requires pretrained weight files")
    def test_model_suite_get_sampler_diffcsp(self):
        from ppmat.models.matinvent.suites import ModelSuite
        suite = ModelSuite(
            model_name="diffcsp",
            sample_cfg={"batch_size": 4, "num_batches": 2, "num_inference_steps": 100},
            finetune_cfg={"batch_size": 4, "epochs": 1, "lr": 1e-4, "accum_steps": 1, "timesteps": 1},
            pretrained_model_path="dummy",
        )
        sampler = suite.get_sampler()
        assert sampler.batch_size == 4
        assert sampler.num_batches == 2

    @pytest.mark.skip(reason="requires pretrained weight files")
    def test_model_suite_get_sampler_mattergen(self):
        from ppmat.models.matinvent.suites import ModelSuite
        suite = ModelSuite(
            model_name="mattergen",
            sample_cfg={"batch_size": 4, "num_batches": 2, "num_inference_steps": 100},
            finetune_cfg={"batch_size": 4, "epochs": 1, "lr": 1e-4, "accum_steps": 1, "timesteps": 1},
            pretrained_model_path="dummy",
        )
        sampler = suite.get_sampler()
        assert sampler.batch_size == 4
        assert sampler.num_batches == 2

    def test_model_suite_get_dataloader(self):
        from ppmat.models.matinvent.suites import ModelSuite
        structures = [_make_si_structure() for _ in range(4)]
        rewards = np.array([0.5, 0.6, 0.7, 0.8], dtype=np.float32)
        dl = ModelSuite.get_dataloader(structures, rewards, batch_size=2)
        assert len(dl.dataset) == 4

    def test_model_suite_save_model(self):
        from ppmat.models.matinvent.suites import ModelSuite
        import paddle.nn as nn

        class DummyModel(nn.Layer):
            def __init__(self):
                super().__init__()
                self.param = self.create_parameter(shape=[1], default_initializer=nn.initializer.Constant(1.0))

        with tempfile.TemporaryDirectory() as tmp:
            ckpt = os.path.join(tmp, "checkpoint")
            model = DummyModel()
            ModelSuite.save_model(model, ckpt)
            assert os.path.exists(os.path.join(ckpt, "model.pdparams"))


class TestRLDatasetCollate:
    def test_to_tensor(self):
        from ppmat.models.matinvent.dataset import _to_tensor
        import paddle
        result = _to_tensor(np.array([1.0, 2.0]))
        assert isinstance(result, paddle.Tensor)
        result_dict = _to_tensor({"a": np.array([1.0])})
        assert isinstance(result_dict["a"], paddle.Tensor)


class TestMatInventInit:
    def test_matinvent_init_without_pretrained(self):
        from ppmat.models.matinvent import MatInvent
        from ppmat.models.matinvent.suites import ModelSuite
        from ppmat.models.matinvent.rewards.reward import Reward
        from ppmat.models.matinvent.suites import _DIFFCSP_DEFAULT
        from ppmat.models.diffcsp.diffcsp import DiffCSP
        import paddle
        with tempfile.TemporaryDirectory() as tmp:
            model = DiffCSP(**_DIFFCSP_DEFAULT)
            model.eval()
            weight_path = os.path.join(tmp, "dummy.pdparams")
            paddle.save(model.state_dict(), weight_path)
            suite = ModelSuite(
                model_name="diffcsp",
                sample_cfg={"batch_size": 2, "num_batches": 1, "num_inference_steps": 10},
                finetune_cfg={"batch_size": 2, "epochs": 1, "lr": 1e-4, "accum_steps": 1, "timesteps": 1},
                pretrained_model_path=weight_path,
                device="cpu",
            )
            reward_dir = os.path.join(tmp, "rewards")
            prop_cfg = [_cfg_num_atoms(reward_dir)]
            prop_cfg[0].target = "ascending"
            prop_cfg[0].minv = 1.0
            prop_cfg[0].maxv = 20.0
            reward = Reward(root_dir=reward_dir, prop_cfg=prop_cfg, reward_threshold=0.5, reduce="mean")
            mat_invent = MatInvent(
                rl_epoch=1,
                model_suite=suite,
                reward=reward,
                sample_cfg={},
                finetune_cfg={},
                topk_ratio=0.5,
                save_dir=tmp,
                save_freq=50,
                device="cpu",
            )
            assert mat_invent.rl_epoch == 1
            assert mat_invent.step == 0
            assert mat_invent.agent is not None
            assert mat_invent.prior is not None
