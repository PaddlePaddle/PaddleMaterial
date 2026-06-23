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

from types import SimpleNamespace

import numpy as np
import paddle
import paddle.nn as nn

from ppmat.models.miad.crystal_diffusion import CrystalGen
from ppmat.models.miad.crystal_diffusion import parse_num_atoms_to_per_crystal
from ppmat.models.diffcsp.diffcsp import CSPNet
from ppmat.utils.crystal import lattices_to_params_shape_numpy


def _to_numpy(x):
    if hasattr(x, "numpy"):
        return x.numpy()
    return np.asarray(x)


def _extract_x0(batch):
    if "x0" in batch:
        return batch
    sa = batch.get("structure_array", batch)
    num_atoms_np = _to_numpy(sa["num_atoms"]).flatten().astype("int64")
    frac_coords_np = _to_numpy(sa["frac_coords"]).astype("float32")
    atom_types_np = _to_numpy(sa["atom_types"]).flatten().astype("int64")
    lattice_np = _to_numpy(sa["lattice"]).astype("float32")
    if lattice_np.ndim == 3 and lattice_np.shape[0] == 1:
        lattice_np = lattice_np.reshape(3, 3)
    batch_size = len(num_atoms_np)
    total_atoms = int(num_atoms_np.sum())
    batch_idx_np = np.concatenate(
        [np.full(int(n), i) for i, n in enumerate(num_atoms_np)]
    ).astype("int64")
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


def _dict_to_sns(d):
    """Recursively convert dict to SimpleNamespace for attribute access."""
    if not isinstance(d, dict):
        return d
    return SimpleNamespace(**{k: _dict_to_sns(v) for k, v in d.items()})


class MiAD(nn.Layer):
    """Mirage Atom Diffusion model."""

    def __init__(self, model_cfg=None, diffusion_cfg=None, **kwargs):
        super().__init__()

        model_cfg = model_cfg or {}
        diffusion_cfg = diffusion_cfg or {}

        model_cfg = dict(model_cfg)
        model_cfg.setdefault("num_classes", model_cfg.pop("max_atoms", 100))
        _cspnet_keys = {
            "hidden_dim", "latent_dim", "num_layers", "act_fn", "dis_emb",
            "num_freqs", "edge_style", "ln", "ip", "smooth", "pred_type",
            "prop_dim", "pred_scalar", "num_classes",
        }
        cspnet_kwargs = {k: v for k, v in model_cfg.items() if k in _cspnet_keys}
        self.decoder = CSPNet(**cspnet_kwargs)
        # Remove unused prop_mlp (parent creates it; checkpoint lacks these weights)
        for i in range(self.decoder.num_layers):
            layer = getattr(self.decoder, f"csp_layer_{i}", None)
            if layer and hasattr(layer, "prop_mlp"):
                del layer.prop_mlp
        # Replace gen_edges with block_diag (parent meshgrid causes GPU crash on certain batch sizes)
        def _gen_edges(num_atoms, frac_coords):
            lis = [paddle.ones([int(n), int(n)], dtype="int64") for n in num_atoms]
            fc_graph = paddle.block_diag(lis)
            fc_edges = paddle.nonzero(fc_graph).t()
            return fc_edges, (frac_coords[fc_edges[1]] - frac_coords[fc_edges[0]])
        self.decoder.gen_edges = _gen_edges
        if isinstance(diffusion_cfg, dict):
            diffusion_cfg = _dict_to_sns(diffusion_cfg)
        self.diffusion = CrystalGen(diffusion_cfg, logger=None)

    def set_state_dict(self, state_dict, use_structured_name=True):
        if not any(k.startswith("decoder.") for k in state_dict.keys()):
            state_dict = {f"decoder.{k}": v for k, v in state_dict.items()}
        processed = {}
        for k, v in state_dict.items():
            if k.endswith(".weight") and len(v.shape) == 2:
                v = v.T
            processed[k] = v
        return super().set_state_dict(processed, use_structured_name)

    def forward(self, batch, **kwargs):
        mode = "train" if self.training else "val"
        batch = _extract_x0(batch)
        batch = self.diffusion.train_step(
            batch=batch,
            model=self.decoder,
            mode=mode,
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
            pass
        elif num_atoms_data is not None:
            parsed = parse_num_atoms_to_per_crystal(num_atoms_data)
            if parsed is not None:
                num_atoms, num_atoms_np = parsed
                batch_size = len(num_atoms_np)
                batch_idx_np = np.concatenate(
                    [np.full(int(n), i) for i, n in enumerate(num_atoms_np)]
                )
                batch_idx = paddle.to_tensor(batch_idx_np.astype("int64"))
                atom_types = paddle.zeros([int(num_atoms_np.sum())], dtype="int64")
                batch_data = {
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

        def progress_printer(t):
            return None

        batch = self.diffusion.sampling_procedure(
            model=self.decoder,
            batch=batch_data,
            progress_printer=progress_printer,
        )

        if original_steps is not None:
            self.diffusion.num_steps = original_steps

        # Extract results
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
            # Convert tensors to numpy so downstream consumers (BuildStructure,
            # CSPMetric) can handle them in multi-process p_map without
            # serializing paddle tensors.
            lat_i = lattices[i]
            fc_i = frac_coords[start_idx : start_idx + n]
            at_i = atom_types[start_idx : start_idx + n]
            lat_np = lat_i.numpy() if hasattr(lat_i, "numpy") else np.asarray(lat_i)
            fc_np = fc_i.numpy() if hasattr(fc_i, "numpy") else np.asarray(fc_i)
            at_np = at_i.numpy() if hasattr(at_i, "numpy") else np.asarray(at_i)
            start_idx += n
            # Filter out mirage atoms (atom_types == 0) produced by Mirage
            # Infusion, mirroring lib/data/crystal_data_storage.py save_batch.
            valid_mask = at_np != 0
            if valid_mask.any():
                at_np = at_np[valid_mask]
                fc_np = fc_np[valid_mask]
                n = int(valid_mask.sum())
            else:
                # All atoms are mirage atoms; keep original count but replace
                # invalid type 0 with 1 (H) to avoid empty structure errors.
                at_np = np.where(at_np == 0, 1, at_np)
            # Pre-compute lattice params so BuildStructure does not need to
            # derive them from a (possibly non-list) lattice matrix.
            lat_for_params = (
                lat_np.reshape(1, 3, 3) if lat_np.ndim == 2 else lat_np
            )
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
