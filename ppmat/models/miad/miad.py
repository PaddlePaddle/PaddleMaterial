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

import numpy as np
import paddle
import paddle.nn as nn

from ppmat.models.common.runtime import RuntimeMixin
from ppmat.models.common.runtime import runtime_boundary
from ppmat.models.diffcsp.diffcsp import CSPNet
from ppmat.models.miad.crystal_diffusion import CrystalGen
from ppmat.models.miad.crystal_diffusion import parse_num_atoms_to_per_crystal
from ppmat.utils import logger
from ppmat.utils.crystal import lattices_to_params_shape_numpy


class MiADCSPNet(CSPNet):
    """CSPNet with block-diagonal edge generation (avoids GPU crash on certain
    batch sizes), stripping the ``prop_mlp`` sub-layers the shared Paddle
    CSPNet adds for property-guided models: the MiAD checkpoint has no such
    parameters, so keeping them would silently leave 24 weights unloaded.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for i in range(self.num_layers):
            delattr(getattr(self, "csp_layer_%d" % i), "prop_mlp")

    def gen_edges(self, num_atoms, frac_coords):
        lis = [paddle.ones([int(n), int(n)], dtype="int64") for n in num_atoms]
        fc_graph = paddle.block_diag(lis)
        fc_edges = paddle.nonzero(fc_graph).t()
        return fc_edges, frac_coords[fc_edges[1]] - frac_coords[fc_edges[0]]


def _to_numpy(x):
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)


def _build_batch_idx(num_atoms_np):
    """Per-atom batch indices from per-crystal atom counts."""
    num_atoms_np = np.asarray(num_atoms_np).flatten().astype("int64")
    batch_idx_np = np.concatenate(
        [np.full(int(n), i) for i, n in enumerate(num_atoms_np)]
    ).astype("int64")
    return batch_idx_np, len(num_atoms_np)


def _extract_x0(batch, mirage_num_atoms=None):
    if "x0" in batch:
        return batch
    sa = batch.get("structure_array", batch)
    num_atoms_np = _to_numpy(sa["num_atoms"]).flatten().astype("int64")
    frac_coords_np = _to_numpy(sa["frac_coords"]).astype("float32")
    atom_types_np = _to_numpy(sa["atom_types"]).flatten().astype("int64")
    lattice_np = _to_numpy(sa["lattice"]).astype("float32")
    if lattice_np.ndim == 3:
        if lattice_np.shape[0] == 1:
            lattice_np = lattice_np.reshape(3, 3)
        else:
            lattice_np = lattice_np.reshape(-1, 3)
    batch_size = len(num_atoms_np)
    total_atoms = int(num_atoms_np.sum())
    if frac_coords_np.shape[0] != total_atoms or atom_types_np.shape[0] != total_atoms:
        raise ValueError(
            f"frac_coords/atom_types length ({frac_coords_np.shape[0]}/"
            f"{atom_types_np.shape[0]}) does not match total atoms ({total_atoms})"
        )
    if lattice_np.ndim != 2 or lattice_np.shape != (batch_size * 3, 3):
        raise ValueError(
            f"lattice must be ({batch_size * 3}, 3) or ({batch_size}, 3, 3), "
            f"got {lattice_np.shape}"
        )
    if mirage_num_atoms is not None:
        # Pad every crystal to mirage_num_atoms atoms with type-0 mirage
        # atoms at random fractional coordinates.
        padded_frac = []
        padded_types = []
        padded_num_atoms = []
        offset = 0
        for i, n in enumerate(num_atoms_np):
            n = int(n)
            n_m = max(int(mirage_num_atoms), n)
            padded_frac.append(frac_coords_np[offset : offset + n])
            padded_types.append(atom_types_np[offset : offset + n])
            if n_m > n:
                padded_frac.append(np.random.rand(n_m - n, 3).astype("float32"))
                padded_types.append(np.zeros(n_m - n, dtype="int64"))
            padded_num_atoms.append(n_m)
            offset += n
        frac_coords_np = np.concatenate(padded_frac)
        atom_types_np = np.concatenate(padded_types)
        num_atoms_np = np.array(padded_num_atoms, dtype="int64")
        total_atoms = int(num_atoms_np.sum())
        batch_size = len(num_atoms_np)
    batch_idx_np, _ = _build_batch_idx(num_atoms_np)
    batch["x0"] = [
        paddle.to_tensor(lattice_np.reshape(batch_size, 3, 3)),
        paddle.to_tensor(frac_coords_np.reshape(total_atoms, 3)),
        paddle.to_tensor(atom_types_np.reshape(total_atoms)),
    ]
    batch["num_atoms"] = paddle.to_tensor(num_atoms_np)
    batch["batch_idx"] = paddle.to_tensor(batch_idx_np)
    batch["atom_types"] = paddle.to_tensor(atom_types_np)
    batch["batch_size"] = batch_size
    return batch


class MiAD(RuntimeMixin, paddle.nn.Layer):
    """Mirage Atom Diffusion model.

    Supports num-atoms-based sampling: ``sample_by_num_atoms`` provides only
    ``num_atoms``; all atoms start as mirage (type 0) and the trained
    mirage-infusion prior determines which become real elements, then the
    type-0 atoms are filtered out.
    """

    supports_num_atoms_sampling = True

    # Official MAX_ATOMIC_NUM; class index 0 is the mirage type.
    _MAX_ATOMIC_NUM = 100

    # Sentinel class index of mirage atoms, filtered from sampled output.
    _MIRAGE_TYPE = 0

    # Atomic number used when every sampled atom is mirage, so the output
    # structure stays non-empty (hydrogen).
    _FALLBACK_ATOMIC_NUM = 1

    def __init__(
        self,
        model_cfg=None,
        diffusion_cfg=None,
        execution_backend="eager",
        runtime_options=None,
    ):
        super().__init__()
        self._init_runtime(execution_backend, runtime_options)

        model_cfg = model_cfg or {}
        diffusion_cfg = diffusion_cfg or {}

        self.mirage_num_atoms = model_cfg.get("mirage_num_atoms", None)
        model_cfg = dict(model_cfg)
        # max_atoms (MiAD naming) maps to num_classes (CSPNet naming)
        model_cfg.setdefault(
            "num_classes", model_cfg.pop("max_atoms", self._MAX_ATOMIC_NUM)
        )
        # prop_dim/pred_scalar excluded: MiAD has no property-guided branches
        _cspnet_keys = {
            "hidden_dim",
            "latent_dim",
            "num_layers",
            "act_fn",
            "dis_emb",
            "num_freqs",
            "edge_style",
            "ln",
            "ip",
            "smooth",
            "pred_type",
            "num_classes",
        }
        cspnet_kwargs = {k: v for k, v in model_cfg.items() if k in _cspnet_keys}
        self.decoder = MiADCSPNet(**cspnet_kwargs)
        self.diffusion = CrystalGen(diffusion_cfg)

    def set_state_dict(self, state_dict, use_structured_name=True):
        # Official checkpoints: no "decoder." prefix, PyTorch (out, in) layout
        # for Linear weights (Embedding keeps the same layout). Adapt per key:
        # unprefixed keys get the prefix and, if Linear, a transposed weight.
        linear_param_names = set()
        for module_name, module in self.decoder.named_modules():
            if isinstance(module, nn.Linear):
                for pname in ("weight",):
                    if module_name:
                        full = f"decoder.{module_name}.{pname}"
                    else:
                        full = f"decoder.{pname}"
                    linear_param_names.add(full)
        adapted = {}
        for k, v in state_dict.items():
            if k.startswith("decoder."):
                new_key = k
            else:
                new_key = f"decoder.{k}"
                if (
                    new_key in linear_param_names
                    and hasattr(v, "shape")
                    and len(v.shape) == 2
                ):
                    v = v.T
            adapted[new_key] = v
        state_dict = adapted
        missing_keys = []
        shape_mismatch_keys = []
        param_state = {}
        for name, param in self.named_parameters():
            if name not in state_dict:
                missing_keys.append(name)
                continue
            v = state_dict[name]
            if hasattr(v, "numpy"):
                v = v.numpy()
            elif not isinstance(v, np.ndarray):
                v = np.asarray(v)
            if v.shape == param.shape:
                param_state[name] = v.astype(param.numpy().dtype)
            else:
                shape_mismatch_keys.append(name)
        loaded = set(param_state.keys())
        unexpected_keys = [k for k in state_dict.keys() if k not in loaded]
        if shape_mismatch_keys:
            logger.warning(
                "Shape mismatch, skipped: %s", ", ".join(shape_mismatch_keys)
            )
        for name in param_state:
            self.get_parameter(name).set_value(param_state[name])
        return missing_keys, unexpected_keys

    def _decode(
        self,
        time_emb,
        atom_types,
        frac_coords,
        lattices,
        num_atoms,
        node2graph,
    ):
        edges, frac_diff = self.decoder.gen_edges(num_atoms, frac_coords)
        return self._runtime_decode(
            time_emb,
            atom_types,
            frac_coords,
            lattices,
            num_atoms,
            node2graph,
            edges,
            frac_diff,
        )

    @runtime_boundary("denoise_step")
    def _runtime_decode(
        self,
        time_emb,
        atom_types,
        frac_coords,
        lattices,
        num_atoms,
        node2graph,
        edges,
        frac_diff,
    ):
        return self.decoder.forward_with_edges(
            time_emb,
            atom_types,
            frac_coords,
            lattices,
            num_atoms,
            node2graph,
            edges,
            frac_diff,
        )

    def forward(self, batch, **kwargs):
        batch = _extract_x0(batch, mirage_num_atoms=self.mirage_num_atoms)
        batch = self.diffusion.train_step(
            batch=batch,
            model=self._decode,
        )
        loss = batch["loss"]
        loss_dict = {
            "loss": loss,
        }
        return {"loss_dict": loss_dict}

    @paddle.no_grad()
    def sample(self, batch_data, num_inference_steps=None):
        if "structure_array" in batch_data:
            num_atoms_data = batch_data["structure_array"].get("num_atoms", None)
        else:
            num_atoms_data = batch_data.get("num_atoms", None)

        if "batch_idx" in batch_data:
            for key in ("num_atoms", "atom_types", "batch_size"):
                if key not in batch_data:
                    raise ValueError(
                        f"batch_idx provided but missing required key '{key}'"
                    )
        elif num_atoms_data is not None:
            parsed = parse_num_atoms_to_per_crystal(num_atoms_data)
            if parsed is not None:
                num_atoms, num_atoms_np = parsed
                batch_idx_np, batch_size = _build_batch_idx(num_atoms_np)
                batch_idx = paddle.to_tensor(batch_idx_np)
                atom_types = paddle.zeros([int(num_atoms_np.sum())], dtype="int64")
                batch_data = {
                    **batch_data,
                    "num_atoms": num_atoms,
                    "batch_idx": batch_idx,
                    "atom_types": atom_types,
                    "batch_size": batch_size,
                }

        if num_inference_steps is not None:
            original_steps = self.diffusion.num_steps
            self.diffusion.num_steps = num_inference_steps
        else:
            original_steps = None

        try:
            batch = self.diffusion.sampling_procedure(
                model=self._decode,
                batch=batch_data,
            )
        finally:
            if original_steps is not None:
                self.diffusion.num_steps = original_steps

        x0_pred = batch["x0_prediction"]
        lattices = x0_pred[0]
        frac_coords = x0_pred[1]
        atom_types = x0_pred[2]

        result = []
        num_atoms = batch_data.get("num_atoms", None)
        batch_size = batch_data.get("batch_size", lattices.shape[0])

        start_idx = 0
        for i in range(batch_size):
            if num_atoms is not None:
                n = int(num_atoms[i])
            else:
                n = frac_coords.shape[0]
                if i > 0:
                    break
            # Convert to numpy for downstream consumers (BuildStructure, CSPMetric)
            lat_i = lattices[i]
            fc_i = frac_coords[start_idx : start_idx + n]
            at_i = atom_types[start_idx : start_idx + n]
            lat_np = _to_numpy(lat_i)
            fc_np = _to_numpy(fc_i)
            at_np = _to_numpy(at_i)
            start_idx += n
            # Filter out mirage atoms (type 0)
            valid_mask = at_np != self._MIRAGE_TYPE
            if valid_mask.any():
                at_np = at_np[valid_mask]
                fc_np = fc_np[valid_mask]
                n = int(valid_mask.sum())
            else:
                # All-mirage fallback: keep the count, replace the mirage
                # type with hydrogen so the structure stays non-empty.
                at_np = np.where(
                    at_np == self._MIRAGE_TYPE, self._FALLBACK_ATOMIC_NUM, at_np
                )
            lat_for_params = lat_np.reshape(1, 3, 3) if lat_np.ndim == 2 else lat_np
            lengths, angles = lattices_to_params_shape_numpy(lat_for_params)
            result.append(
                {
                    "num_atoms": n,
                    "atom_types": at_np,
                    "frac_coords": fc_np,
                    "lattice": lat_np,
                    "lengths": lengths.flatten(),
                    "angles": angles.flatten(),
                }
            )

        return {"result": result}
