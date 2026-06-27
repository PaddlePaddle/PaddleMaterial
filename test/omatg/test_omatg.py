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

import unittest

import paddle

from ppmat.models.omatg.model import OMATGCSPNet as CSPNet, OMATGCSPNetFull
from ppmat.utils.crystal import (
    cart_to_frac_coords,
    frac_to_cart_coords,
    lattice_params_to_matrix_paddle,
)
from ppmat.datasets.omatg_dataset import Structure
from ppmat.datasets.omatg_dataset import OMATGData


def _make_batch(batch_size=2, atoms_per_struct=(3, 4), max_z=10):
    """Build a minimal OMATGData batch for forward/loss tests."""
    num_list = list(atoms_per_struct[:batch_size])
    total = sum(num_list)
    atom_types = paddle.randint(1, max_z, [total], dtype="int64")
    frac_coords = paddle.rand([total, 3])
    lattices = paddle.rand([batch_size, 3, 3]) * 2.0 + paddle.eye(3).unsqueeze(0)
    num_atoms = paddle.to_tensor(num_list, dtype="int64")
    node2graph = paddle.repeat_interleave(
        paddle.arange(batch_size, dtype="int64"), num_atoms
    )
    d = {
        "atom_types": atom_types,
        "frac_coords": frac_coords,
        "lattices": lattices,
        "num_atoms": num_atoms,
        "node2graph": node2graph,
    }
    return OMATGData.from_collate_dict(d)


class TestCSPNetForward(unittest.TestCase):
    """CSPNet and OMATGCSPNetFull minimal forward / loss smoke tests."""

    def test_cspnet_forward_returns_b_eta(self):
        """CSP mode forward returns 4-tuple (cell_b, pos_b, cell_eta, pos_eta)."""
        batch_size = 2
        num_atoms = paddle.to_tensor([3, 4], dtype="int64")
        total_atoms = int(num_atoms.sum())
        node2graph = paddle.to_tensor([0, 0, 0, 1, 1, 1, 1], dtype="int64")

        model = CSPNet(
            hidden_dim=64, num_layers=2, max_atoms=100,
            time_embed_dim=32, edge_style="fc",
        )
        t = paddle.rand([batch_size])
        atom_types = paddle.randint(1, 10, [total_atoms])
        frac_coords = paddle.rand([total_atoms, 3])
        lattices = paddle.rand([batch_size, 3, 3]) * 2.0

        output = model(t, atom_types, frac_coords, lattices, num_atoms, node2graph)
        self.assertEqual(len(output), 4)
        self.assertEqual(output[0].shape, [batch_size, 3, 3])
        self.assertEqual(output[1].shape, [total_atoms, 3])
        self.assertEqual(output[2].shape, [batch_size, 3, 3])
        self.assertEqual(output[3].shape, [total_atoms, 3])

    def test_cspnet_forward_dict_keys(self):
        """forward_dict returns b/eta dict with correct keys."""
        model = CSPNet(
            hidden_dim=64, num_layers=2, max_atoms=100,
            time_embed_dim=32, edge_style="fc",
        )
        data = _make_batch()
        t = paddle.rand([2])
        out = model.forward_dict(
            t, data.species, data.pos, data.cell,
            data.n_atoms, data.batch,
        )
        self.assertIn("pos_b", out)
        self.assertIn("pos_eta", out)
        self.assertIn("cell_b", out)
        self.assertIn("cell_eta", out)

    def test_omgcspnetfull_forward_csp_loss(self):
        """CSP mode forward returns loss_dict with loss/loss_lattice/loss_coord."""
        model = OMATGCSPNetFull(
            hidden_dim=64, num_layers=2, max_atoms=100,
            time_embed_dim=32, edge_style="fc", pred_type=False,
        )
        data = _make_batch()
        output = model(data)
        self.assertIn("loss_dict", output)
        self.assertIn("loss", output["loss_dict"])
        self.assertIn("loss_lattice", output["loss_dict"])
        self.assertIn("loss_coord", output["loss_dict"])
        self.assertNotIn("loss_type", output["loss_dict"])
        loss_val = float(output["loss_dict"]["loss"])
        self.assertFalse(loss_val != loss_val, "loss is NaN")

    def test_omgcspnetfull_forward_dng_loss(self):
        """DNG mode forward returns loss_dict including loss_type (cross-entropy)."""
        model = OMATGCSPNetFull(
            hidden_dim=64, num_layers=2, max_atoms=100,
            time_embed_dim=32, edge_style="fc", pred_type=True,
        )
        model.enable_masked_species()
        data = _make_batch()
        output = model(data)
        self.assertIn("loss_type", output["loss_dict"])
        self.assertIn("loss", output["loss_dict"])
        loss_val = float(output["loss_dict"]["loss"])
        self.assertFalse(loss_val != loss_val, "DNG loss is NaN")

    def test_cspnet_backward_gradient(self):
        """Loss backward produces non-None gradients on model parameters."""
        model = OMATGCSPNetFull(
            hidden_dim=64, num_layers=2, max_atoms=100,
            time_embed_dim=32, edge_style="fc",
        )
        data = _make_batch()
        output = model(data)
        loss = output["loss_dict"]["loss"]
        loss.backward()
        has_grad = any(
            p.grad is not None and float(p.grad.abs().sum()) > 0
            for p in model.parameters()
        )
        self.assertTrue(has_grad, "No non-zero gradients after backward")


class TestStructureData(unittest.TestCase):
    """Structure and OMATGData smoke tests."""

    def test_structure_creation(self):
        cell = paddle.eye(3) * 5.0
        atomic_numbers = paddle.to_tensor([1, 1, 8], dtype="int64")
        pos = paddle.rand([3, 3])
        struct = Structure(cell, atomic_numbers, pos, pos_is_fractional=True)
        self.assertEqual(struct.cell.shape, [3, 3])
        self.assertTrue(struct.pos_is_fractional)
        self.assertEqual(len(struct.atomic_numbers), 3)

    def test_structure_convert_coords(self):
        cell = paddle.to_tensor([[5.0, 0, 0], [0, 5.0, 0], [0, 0, 5.0]])
        atomic_numbers = paddle.to_tensor([1], dtype="int64")
        pos = paddle.to_tensor([[2.5, 2.5, 2.5]])
        struct = Structure(cell, atomic_numbers, pos, pos_is_fractional=False)
        struct.convert_to_fractional()
        self.assertTrue(struct.pos_is_fractional)
        self.assertAlmostEqual(float(struct.pos[0, 0]), 0.5, places=4)
        struct.convert_to_cartesian()
        self.assertFalse(struct.pos_is_fractional)
        self.assertAlmostEqual(float(struct.pos[0, 0]), 2.5, places=4)

    def test_structure_get_ase_atoms(self):
        cell = paddle.eye(3) * 5.0
        atomic_numbers = paddle.to_tensor([1, 8], dtype="int64")
        pos = paddle.to_tensor([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
        struct = Structure(cell, atomic_numbers, pos, pos_is_fractional=True)
        atoms = struct.get_ase_atoms()
        self.assertEqual(len(atoms), 2)

    def test_omgdata_from_structure(self):
        cell = paddle.eye(3) * 5.0
        atomic_numbers = paddle.to_tensor([1, 1, 8], dtype="int64")
        pos = paddle.rand([3, 3])
        struct = Structure(cell, atomic_numbers, pos, pos_is_fractional=True)
        data = OMATGData(struct)
        self.assertEqual(data.num_graphs, 1)
        self.assertEqual(data.num_atoms, 3)
        self.assertIsNotNone(data.species)
        self.assertIsInstance(data.property_dict, list)

    def test_omgdata_batch(self):
        structs = []
        for _ in range(3):
            cell = paddle.eye(3) * 5.0
            atomic_numbers = paddle.to_tensor([1, 8], dtype="int64")
            pos = paddle.rand([2, 3])
            structs.append(Structure(cell, atomic_numbers, pos, pos_is_fractional=True))
        batch = OMATGData.from_batch(structs, concatenate=True)
        self.assertEqual(batch.num_graphs, 3)
        self.assertEqual(batch.num_atoms, 6)
        self.assertEqual(batch.batch.shape, [6])
        self.assertEqual(batch.ptr.shape, [4])

    def test_omgdata_get_graph(self):
        structs = []
        for _ in range(2):
            cell = paddle.eye(3) * 5.0
            atomic_numbers = paddle.to_tensor([1, 8], dtype="int64")
            pos = paddle.rand([2, 3])
            structs.append(Structure(cell, atomic_numbers, pos, pos_is_fractional=True))
        batch = OMATGData.from_batch(structs, concatenate=True)
        g0 = batch.get_graph(0)
        self.assertEqual(len(g0.atomic_numbers), 2)
        g1 = batch.get_graph(1)
        self.assertEqual(len(g1.atomic_numbers), 2)

    def test_omgdata_clone(self):
        cell = paddle.eye(3) * 5.0
        atomic_numbers = paddle.to_tensor([1, 8], dtype="int64")
        pos = paddle.rand([2, 3])
        struct = Structure(cell, atomic_numbers, pos, pos_is_fractional=True)
        data = OMATGData(struct)
        cloned = data.clone()
        self.assertEqual(cloned.num_atoms, data.num_atoms)
        self.assertTrue(paddle.equal_all(cloned.species, data.species))

    def test_omgdata_set_get_field(self):
        cell = paddle.eye(3) * 5.0
        atomic_numbers = paddle.to_tensor([1, 8], dtype="int64")
        pos = paddle.rand([2, 3])
        struct = Structure(cell, atomic_numbers, pos, pos_is_fractional=True)
        data = OMATGData(struct)
        self.assertTrue(paddle.equal_all(data.get_field("species"), data.species))
        new_pos = paddle.rand([2, 3])
        data.set_field("pos", new_pos)
        self.assertTrue(paddle.equal_all(data.get_field("pos"), new_pos))


class TestLatticeUtils(unittest.TestCase):
    """Lattice utility smoke tests."""

    def test_lattice_params_to_matrix(self):
        lengths = paddle.to_tensor([[3.0, 3.0, 5.0], [4.0, 4.0, 4.0]])
        angles = paddle.to_tensor([[90.0, 90.0, 90.0], [90.0, 90.0, 120.0]])
        matrices = lattice_params_to_matrix_paddle(lengths, angles)
        self.assertEqual(matrices.shape, [2, 3, 3])

    def test_frac_cart_conversion(self):
        lengths = paddle.to_tensor([[3.0, 3.0, 5.0]])
        angles = paddle.to_tensor([[90.0, 90.0, 90.0]])
        num_atoms = paddle.to_tensor([3], dtype="int64")
        frac_coords = paddle.rand([3, 3])
        cart = frac_to_cart_coords(frac_coords, num_atoms, lengths=lengths, angles=angles)
        self.assertEqual(cart.shape, [3, 3])
        frac_back = cart_to_frac_coords(cart, num_atoms, lengths=lengths, angles=angles)
        self.assertEqual(frac_back.shape, [3, 3])


class TestSIComponents(unittest.TestCase):
    """SI framework component smoke tests."""

    def test_gamma_sqrt(self):
        from ppmat.models.omatg.si.interpolants import LatentGammaSqrt
        t = paddle.to_tensor([0.3, 0.5, 0.7])
        g = LatentGammaSqrt(a=1.0)
        self.assertEqual(g.gamma(t).shape, [3])
        self.assertTrue(g.requires_antithetic())
        self.assertEqual(g.gamma_derivative(t).shape, [3])

    def test_gamma_encdec(self):
        from ppmat.models.omatg.si.interpolants import LatentGammaEncoderDecoder
        t = paddle.to_tensor([0.3, 0.5, 0.7])
        g = LatentGammaEncoderDecoder()
        self.assertEqual(g.gamma(t).shape, [3])
        self.assertFalse(g.requires_antithetic())

    def test_epsilon_vanishing(self):
        from ppmat.models.omatg.si.interpolants import VanishingEpsilon
        t = paddle.to_tensor([0.3, 0.5, 0.7])
        eps = VanishingEpsilon(c=1.0)
        self.assertEqual(eps.epsilon(t).shape, [3])

    def test_sigma_geometric(self):
        from ppmat.models.omatg.si.interpolants import GeometricSigma
        s = paddle.to_tensor([0.0, 0.5, 1.0])
        sig = GeometricSigma(sigma_min=0.1, sigma_max=10.0)
        self.assertEqual(sig.sigma(s).shape, [3])
        self.assertEqual(sig.sigma_dot(s).shape, [3])

    def test_tau_constant(self):
        from ppmat.models.omatg.si.interpolants import TauConstantSchedule
        t = paddle.to_tensor([0.3, 0.7])
        tau = TauConstantSchedule()
        self.assertEqual(tau.tau(t).shape, [2])
        self.assertEqual(tau.tau_dot(t).shape, [2])

    def test_interpolant_linear(self):
        from ppmat.models.omatg.si.interpolants import LinearInterpolant
        t = paddle.to_tensor([0.3])
        interp = LinearInterpolant()
        self.assertEqual(interp.alpha(t).shape, [1])
        self.assertEqual(interp.beta(t).shape, [1])

    def test_interpolant_periodic_linear(self):
        from ppmat.models.omatg.si.interpolants import PeriodicLinearInterpolant
        interp = PeriodicLinearInterpolant()
        corr = interp.get_corrector()
        self.assertIsNotNone(corr)

    def test_interpolant_vp(self):
        from ppmat.models.omatg.si.interpolants import (
            ScoreBasedDiffusionModelInterpolantVP,
        )
        from ppmat.models.omatg.si.interpolants import TauConstantSchedule
        t = paddle.to_tensor([0.3])
        interp = ScoreBasedDiffusionModelInterpolantVP(TauConstantSchedule())
        self.assertEqual(interp.alpha(t).shape, [1])

    def test_interpolant_ve(self):
        from ppmat.models.omatg.si.interpolants import (
            ScoreBasedDiffusionModelInterpolantVE,
            GeometricSigma,
        )
        t = paddle.to_tensor([0.3])
        interp = ScoreBasedDiffusionModelInterpolantVE(GeometricSigma(0.1, 10.0))
        self.assertEqual(interp.alpha(t).shape, [1])

    def test_dfm_mask_loss(self):
        """DiscreteFlowMatchingMask cross-entropy loss."""
        from ppmat.models.omatg.si.core import (
            DiscreteFlowMatchingMask,
        )
        dfm = DiscreteFlowMatchingMask(noise=0.1)
        x_0 = paddle.zeros([5], dtype="int64")
        x_1 = paddle.randint(1, 10, [5], dtype="int64")
        t = paddle.to_tensor([0.5])
        x_t, z = dfm.interpolate(t, x_0, x_1, paddle.to_tensor([0, 0, 0, 0, 0]))
        self.assertEqual(x_t.shape, [5])

        def model_fn(x_t):
            pred = paddle.rand([5, 100])
            return pred, paddle.zeros_like(pred)
        losses = dfm.loss(model_fn, t, x_0, x_1, x_t, z,
                          paddle.to_tensor([0, 0, 0, 0, 0]))
        self.assertIn("loss", losses)
        self.assertTrue(dfm.uses_masked_species())

    def test_identity_interpolant(self):
        from ppmat.models.omatg.si.core import (
            SingleStochasticInterpolantIdentity,
        )
        ident = SingleStochasticInterpolantIdentity()
        x = paddle.to_tensor([1, 2, 3], dtype="int64")
        t = paddle.to_tensor([0.5])
        x_t, z = ident.interpolate(t, x, x, paddle.to_tensor([0, 0, 0]))
        self.assertTrue(paddle.equal_all(x_t, x))
        self.assertFalse(ident.uses_masked_species())


class TestSITrainingPath(unittest.TestCase):
    """SI velocity-matching training path smoke tests."""

    def _build_csp_si_model(self):
        """Build a small CSP model with SI (Linear-ODE)."""
        from ppmat.models.omatg.si import (
            StochasticInterpolants, SingleStochasticInterpolant,
            SingleStochasticInterpolantIdentity,
            PeriodicLinearInterpolant, LinearInterpolant,
        )
        from ppmat.models.omatg.model import IndependentSampler
        si = StochasticInterpolants(
            stochastic_interpolants=[
                SingleStochasticInterpolantIdentity(),
                SingleStochasticInterpolant(
                    interpolant=PeriodicLinearInterpolant(), gamma=None,
                    epsilon=None, differential_equation_type="ODE",
                    velocity_annealing_factor=10.18,
                    correct_center_of_mass_motion=True,
                ),
                SingleStochasticInterpolant(
                    interpolant=LinearInterpolant(), gamma=None, epsilon=None,
                    differential_equation_type="ODE",
                    velocity_annealing_factor=1.82,
                ),
            ],
            data_fields=["species", "pos", "cell"],
            integration_time_steps=210,
        )
        sampler = IndependentSampler(dataset_name="mp_20", mirror_species=True)
        model = OMATGCSPNetFull(
            hidden_dim=32, num_layers=1, max_atoms=100,
            time_embed_dim=16, pred_type=False, use_si=False,
        )
        model._si = si
        model._sampler = sampler
        model._relative_si_costs = {
            "species_loss": 0.0, "pos_loss_b": 0.9994, "cell_loss_b": 0.0006,
        }
        model.use_si = True
        return model

    def test_csp_si_forward_loss(self):
        """CSP SI training path produces velocity-matching loss."""
        model = self._build_csp_si_model()
        data = _make_batch()
        out = model(data)
        self.assertIn("loss_dict", out)
        self.assertIn("loss", out["loss_dict"])
        self.assertIn("pos_loss_b", out["loss_dict"])
        self.assertIn("cell_loss_b", out["loss_dict"])
        self.assertIn("species_loss", out["loss_dict"])
        loss_val = float(out["loss_dict"]["loss"])
        self.assertFalse(loss_val != loss_val, "SI loss is NaN")

    def test_csp_si_backward(self):
        """SI loss backward produces gradients."""
        model = self._build_csp_si_model()
        data = _make_batch()
        out = model(data)
        loss = out["loss_dict"]["loss"]
        loss.backward()
        has_grad = any(
            p.grad is not None and float(p.grad.abs().sum()) > 0
            for p in model.parameters()
        )
        self.assertTrue(has_grad, "No gradients after SI backward")

    def test_dng_si_forward_loss(self):
        """DNG SI training path (SDE + gamma + DFM mask)."""
        from ppmat.models.omatg.si import (
            StochasticInterpolants, SingleStochasticInterpolant,
            PeriodicLinearInterpolant, LinearInterpolant,
        )
        from ppmat.models.omatg.si.core import DiscreteFlowMatchingMask
        from ppmat.models.omatg.si.interpolants import LatentGammaSqrt, VanishingEpsilon
        from ppmat.models.omatg.model import IndependentSampler
        si = StochasticInterpolants(
            stochastic_interpolants=[
                DiscreteFlowMatchingMask(noise=0.189),
                SingleStochasticInterpolant(
                    interpolant=PeriodicLinearInterpolant(),
                    gamma=LatentGammaSqrt(a=0.018),
                    epsilon=VanishingEpsilon(c=9.7, mu=0.17, sigma=0.029),
                    differential_equation_type="SDE",
                    velocity_annealing_factor=6.33,
                    correct_center_of_mass_motion=True,
                ),
                SingleStochasticInterpolant(
                    interpolant=LinearInterpolant(), gamma=None, epsilon=None,
                    differential_equation_type="ODE",
                    velocity_annealing_factor=1.07,
                ),
            ],
            data_fields=["species", "pos", "cell"],
            integration_time_steps=710,
        )
        sampler = IndependentSampler(dataset_name="mp_20", mask_species=True, mirror_species=False)
        model = OMATGCSPNetFull(
            hidden_dim=32, num_layers=1, max_atoms=100,
            time_embed_dim=16, pred_type=True, use_si=False,
        )
        model.enable_masked_species()
        model._si = si
        model._sampler = sampler
        model._relative_si_costs = {
            "species_loss": 0.5918, "pos_loss_b": 0.1309,
            "pos_loss_z": 0.2708, "cell_loss_b": 0.0065,
        }
        model.use_si = True
        data = _make_batch()
        out = model(data)
        self.assertIn("loss", out["loss_dict"])
        self.assertIn("pos_loss_z", out["loss_dict"])
        self.assertIn("species_loss", out["loss_dict"])


class TestOSInterpolants(unittest.TestCase):
    """One-sided interpolant smoke tests for VESBD/VPSBD variants."""

    def test_vesbd_ode_forward(self):
        """VESBD-ODE SI training via SingleStochasticInterpolantOS."""
        from ppmat.models.omatg.si import (
            StochasticInterpolants, SingleStochasticInterpolant,
            SingleStochasticInterpolantIdentity,
            SingleStochasticInterpolantOS,
            LinearInterpolant,
        )
        from ppmat.models.omatg.si.interpolants import ScoreBasedDiffusionModelInterpolantVE
        from ppmat.models.omatg.si.interpolants import GeometricSigma
        from ppmat.models.omatg.model import IndependentSampler
        os_interp = SingleStochasticInterpolantOS(
            interpolant=ScoreBasedDiffusionModelInterpolantVE(
                GeometricSigma(0.1, 10.0)
            ),
            epsilon=None, differential_equation_type="ODE",
            velocity_annealing_factor=1.0,
        )
        si = StochasticInterpolants(
            stochastic_interpolants=[
                SingleStochasticInterpolantIdentity(),
                os_interp,
                SingleStochasticInterpolant(
                    interpolant=LinearInterpolant(), gamma=None, epsilon=None,
                    differential_equation_type="ODE",
                    velocity_annealing_factor=1.82,
                ),
            ],
            data_fields=["species", "pos", "cell"],
            integration_time_steps=210,
        )
        sampler = IndependentSampler(dataset_name="mp_20", mirror_species=True)
        model = OMATGCSPNetFull(
            hidden_dim=32, num_layers=1, max_atoms=100,
            time_embed_dim=16, pred_type=False, use_si=False,
        )
        model._si = si
        model._sampler = sampler
        model._relative_si_costs = {
            "species_loss": 0.0, "pos_loss_b": 0.9994, "cell_loss_b": 0.0006,
        }
        model.use_si = True
        data = _make_batch()
        out = model(data)
        self.assertIn("loss", out["loss_dict"])

    def test_vpsbd_sde_forward(self):
        """VPSBD-SDE SI training via SingleStochasticInterpolantOS."""
        from ppmat.models.omatg.si import (
            StochasticInterpolants, SingleStochasticInterpolant,
            SingleStochasticInterpolantIdentity,
            SingleStochasticInterpolantOS,
            LinearInterpolant,
        )
        from ppmat.models.omatg.si.interpolants import ScoreBasedDiffusionModelInterpolantVP
        from ppmat.models.omatg.si.interpolants import TauConstantSchedule, ConstantEpsilon
        from ppmat.models.omatg.model import IndependentSampler
        os_interp = SingleStochasticInterpolantOS(
            interpolant=ScoreBasedDiffusionModelInterpolantVP(
                TauConstantSchedule()
            ),
            epsilon=ConstantEpsilon(1.0),
            differential_equation_type="SDE",
            velocity_annealing_factor=1.0,
        )
        si = StochasticInterpolants(
            stochastic_interpolants=[
                SingleStochasticInterpolantIdentity(),
                os_interp,
                SingleStochasticInterpolant(
                    interpolant=LinearInterpolant(), gamma=None, epsilon=None,
                    differential_equation_type="ODE",
                    velocity_annealing_factor=1.07,
                ),
            ],
            data_fields=["species", "pos", "cell"],
            integration_time_steps=710,
        )
        sampler = IndependentSampler(dataset_name="mp_20", mirror_species=True)
        model = OMATGCSPNetFull(
            hidden_dim=32, num_layers=1, max_atoms=100,
            time_embed_dim=16, pred_type=False, use_si=False,
        )
        model._si = si
        model._sampler = sampler
        model._relative_si_costs = {
            "species_loss": 0.0, "pos_loss_b": 0.5,
            "pos_loss_z": 0.3, "cell_loss_b": 0.2,
        }
        model.use_si = True
        data = _make_batch()
        out = model(data)
        self.assertIn("loss", out["loss_dict"])
        self.assertIn("pos_loss_z", out["loss_dict"])


class TestSIFactory(unittest.TestCase):
    """Config-driven SI/sampler factory smoke tests."""

    def test_build_si_from_cfg_csp(self):
        """build_si_from_cfg builds CSP StochasticInterpolants from config."""
        from ppmat.models.omatg.si import build_si_from_cfg, build_sampler_from_cfg
        si_cfg = {
            "stochastic_interpolants": [
                {"__class_name__": "SingleStochasticInterpolantIdentity"},
                {"__class_name__": "SingleStochasticInterpolant",
                 "__init_params__": {
                     "interpolant": {"__class_name__": "PeriodicLinearInterpolant"},
                     "gamma": None, "epsilon": None,
                     "differential_equation_type": "ODE",
                     "velocity_annealing_factor": 10.18,
                     "correct_center_of_mass_motion": True,
                 }},
                {"__class_name__": "SingleStochasticInterpolant",
                 "__init_params__": {
                     "interpolant": {"__class_name__": "LinearInterpolant"},
                     "gamma": None, "epsilon": None,
                     "differential_equation_type": "ODE",
                     "velocity_annealing_factor": 1.82,
                 }},
            ],
            "data_fields": ["species", "pos", "cell"],
            "integration_time_steps": 210,
        }
        sampler_cfg = {
            "position_distribution": {"__class_name__": "UniformPositionDistribution"},
            "cell_distribution": {
                "__class_name__": "InformedLatticeDistribution",
                "__init_params__": {"dataset_name": "mp_20"},
            },
            "species_distribution": {"__class_name__": "MirrorSpecies"},
        }
        si = build_si_from_cfg(si_cfg)
        self.assertEqual(len(si), 3)
        sampler = build_sampler_from_cfg(sampler_cfg)
        self.assertIsNotNone(sampler)

    def test_build_si_from_cfg_dng(self):
        """build_si_from_cfg builds DNG config with nested gamma/epsilon/DFM."""
        from ppmat.models.omatg.si import build_si_from_cfg
        si_cfg = {
            "stochastic_interpolants": [
                {"__class_name__": "DiscreteFlowMatchingMask",
                 "__init_params__": {"noise": 0.189}},
                {"__class_name__": "SingleStochasticInterpolant",
                 "__init_params__": {
                     "interpolant": {"__class_name__": "PeriodicLinearInterpolant"},
                     "gamma": {"__class_name__": "LatentGammaSqrt",
                               "__init_params__": {"a": 0.018}},
                     "epsilon": {"__class_name__": "VanishingEpsilon",
                                 "__init_params__": {"c": 9.7}},
                     "differential_equation_type": "SDE",
                     "velocity_annealing_factor": 6.33,
                 }},
                {"__class_name__": "SingleStochasticInterpolant",
                 "__init_params__": {
                     "interpolant": {"__class_name__": "LinearInterpolant"},
                     "differential_equation_type": "ODE",
                 }},
            ],
            "data_fields": ["species", "pos", "cell"],
            "integration_time_steps": 710,
        }
        si = build_si_from_cfg(si_cfg)
        self.assertEqual(len(si), 3)

    def test_config_driven_si_forward(self):
        """Full config-driven SI forward via OMATGCSPNetFull constructor."""
        si_cfg = {
            "stochastic_interpolants": [
                {"__class_name__": "SingleStochasticInterpolantIdentity"},
                {"__class_name__": "SingleStochasticInterpolant",
                 "__init_params__": {
                     "interpolant": {"__class_name__": "PeriodicLinearInterpolant"},
                     "gamma": None, "epsilon": None,
                     "differential_equation_type": "ODE",
                     "velocity_annealing_factor": 10.18,
                 }},
                {"__class_name__": "SingleStochasticInterpolant",
                 "__init_params__": {
                     "interpolant": {"__class_name__": "LinearInterpolant"},
                     "differential_equation_type": "ODE",
                     "velocity_annealing_factor": 1.82,
                 }},
            ],
            "data_fields": ["species", "pos", "cell"],
            "integration_time_steps": 210,
            "relative_si_costs": {
                "species_loss": 0.0, "pos_loss_b": 0.9994, "cell_loss_b": 0.0006,
            },
        }
        sampler_cfg = {
            "dataset_name": "mp_20",
            "mirror_species": True,
            "mask_species": False,
        }
        model = OMATGCSPNetFull(
            hidden_dim=32, num_layers=1, max_atoms=100,
            time_embed_dim=16, pred_type=False,
            use_si=True, si_cfg=si_cfg, sampler_cfg=sampler_cfg,
        )
        data = _make_batch()
        out = model(data)
        self.assertIn("loss", out["loss_dict"])


class TestSISampling(unittest.TestCase):
    """SI integrate-based sampling smoke tests."""

    def test_csp_si_sample(self):
        """CSP SI sampling produces structures via si.integrate."""
        from ppmat.models.omatg.si import (
            StochasticInterpolants, SingleStochasticInterpolant,
            SingleStochasticInterpolantIdentity,
            PeriodicLinearInterpolant, LinearInterpolant,
        )
        from ppmat.models.omatg.model import IndependentSampler
        si = StochasticInterpolants(
            stochastic_interpolants=[
                SingleStochasticInterpolantIdentity(),
                SingleStochasticInterpolant(
                    interpolant=PeriodicLinearInterpolant(), gamma=None,
                    epsilon=None, differential_equation_type="ODE",
                    velocity_annealing_factor=10.18,
                ),
                SingleStochasticInterpolant(
                    interpolant=LinearInterpolant(), gamma=None, epsilon=None,
                    differential_equation_type="ODE",
                    velocity_annealing_factor=1.82,
                ),
            ],
            data_fields=["species", "pos", "cell"],
            integration_time_steps=10,
        )
        sampler = IndependentSampler(dataset_name="mp_20", mirror_species=True)
        model = OMATGCSPNetFull(
            hidden_dim=32, num_layers=1, max_atoms=100,
            time_embed_dim=16, pred_type=False, use_si=False,
        )
        model._si = si
        model._sampler = sampler
        model._relative_si_costs = {
            "species_loss": 0.0, "pos_loss_b": 0.9994, "cell_loss_b": 0.0006,
        }
        model.use_si = True
        data = _make_batch()
        result = model.sample(data, num_inference_steps=10)
        self.assertIn("result", result)
        self.assertEqual(len(result["result"]), 2)
        self.assertIn("num_atoms", result["result"][0])
        self.assertIn("frac_coords", result["result"][0])


class TestSamplers(unittest.TestCase):
    """Sampler smoke tests using Paddle native APIs."""

    def test_independent_sampler_sample_p_0(self):
        """IndependentSampler.sample_p_0 returns OMATGData with correct batch."""
        from ppmat.models.omatg.model import IndependentSampler
        sampler = IndependentSampler(dataset_name="mp_20", mirror_species=True)
        data = _make_batch()
        x_1 = OMATGData()
        x_1.n_atoms = data.n_atoms
        x_1.species = data.species
        x_1.cell = data.cell
        x_1.pos = data.pos
        x_1.pos_is_fractional = paddle.ones_like(data.n_atoms, dtype="bool")
        x_1.batch = data.batch
        x_1.ptr = paddle.concat([
            paddle.to_tensor([0], dtype="int64"),
            paddle.cumsum(data.n_atoms, axis=0).cast("int64"),
        ])
        x_1.property_dict = [{} for _ in range(len(data.n_atoms))]
        x_0 = sampler.sample_p_0(x_1)
        self.assertEqual(x_0.num_graphs, 2)
        self.assertEqual(x_0.num_atoms, x_1.num_atoms)


if __name__ == "__main__":
    unittest.main()
