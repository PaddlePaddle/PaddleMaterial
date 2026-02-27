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


import argparse
import copy
import gzip
import json
import lzma
import math
from pathlib import Path
import numpy as np
import time

import paddle
import plotly.graph_objects as go
from omegaconf import OmegaConf
from tqdm import tqdm

try:
    from IPython.display import Image, display
except ImportError:  # Optional dependency; visualization still works for files
    Image, display = None, None

from ppmat.datasets import DensityDataset
from ppmat.datasets import SmallDensityDataset
from ppmat.datasets.geometric_data_type.data import Data
from ppmat.models import build_model
from ppmat.utils import logger
from ppmat.utils.misc import set_random_seed

BOHR2ANG = 0.529177
ANG2BOHR = 1.0 / BOHR2ANG


def get_pretrained_model(cfg_path, model_path):
    logger.info(f"from {cfg_path} loading config")
    cfg = OmegaConf.load(cfg_path)
    cfg = OmegaConf.to_container(cfg, resolve=True)

    model = build_model(cfg["Model"])
    logger.info(f"from {model_path}loading model")
    # if a directory is given, pick best > latest > highest epoch > any pdparams
    mpath = Path(model_path)
    if mpath.is_dir():
        candidates = list(mpath.glob("**/*.pdparams"))
        chosen = None
        for name in ["best.pdparams", "latest.pdparams"]:
            hits = [c for c in candidates if c.name == name]
            if hits:
                chosen = hits[0]
                break
        if chosen is None:
            epochs = []
            for c in candidates:
                stem = c.stem
                if stem.startswith("epoch_"):
                    try:
                        ep = int(stem.split("_")[1])
                        epochs.append((ep, c))
                    except Exception:
                        pass
            if epochs:
                epochs.sort(key=lambda x: -x[0])
                chosen = epochs[0][1]
        if chosen is None and candidates:
            chosen = candidates[0]
        if chosen is None:
            raise FileNotFoundError(f"No .pdparams found under {model_path}")
        model_path = str(chosen)
        logger.info(f"Resolved checkpoint path: {model_path}")

    state_dict = paddle.load(model_path)
    if isinstance(state_dict, dict) and "model" in state_dict:
        model.set_state_dict(state_dict["model"])
    else:
        model.set_state_dict(state_dict)
    return model


def inference_model(model, g, density, grid_coord, infos, grid_batch_size=8196):
    with paddle.no_grad():
        model.eval()
        device = paddle.get_device()
        prepared_infos = (
            model._prepare_infos(infos, device) if hasattr(model, "_prepare_infos") else infos
        )
        if grid_batch_size is None:
            if hasattr(model, "_forward_density"):
                preds = model._forward_density(
                    g.x, g.pos, grid_coord, g.batch, prepared_infos
                ).squeeze(0)
            else:
                # Fallback for legacy models expecting raw tensors
                preds = model(g.x, g.pos, grid_coord, g.batch, prepared_infos).squeeze(0)
        else:
            preds = []
            total = grid_coord.shape[1]
            step = grid_batch_size
            num_iter = (total + step - 1) // step
            for start in tqdm(range(0, total, step), total=num_iter):
                end = min(start + step, total)
                grid = grid_coord[:, start:end]
                if hasattr(model, "_forward_density"):
                    preds.append(
                        model._forward_density(
                            g.x, g.pos, grid, g.batch, prepared_infos
                        ).squeeze(0)
                    )
                else:
                    preds.append(
                        model(g.x, g.pos, grid, g.batch, prepared_infos).squeeze(0)
                    )
            preds = paddle.concat(preds, axis=0)

        if density is None:
            return preds, None, None

        mask = (density > 0).astype(dtype="float32")
        preds = preds * mask
        density = density * mask
        diff = paddle.abs(preds - density)
        loss = diff.pow(2).sum()
        denom = paddle.clip(density.sum(), min=1e-12)
        mae = diff.sum() / denom
    return preds, loss, mae


def draw_volume(
    grid,
    density,
    atom_type,
    atom_coord,
    isomin=0.05,
    isomax=None,
    surface_count=5,
    title=None,
):
    atom_colorscale = ["grey", "white", "red", "blue", "green"]
    fig = go.Figure()
    fig.add_trace(
        go.Volume(
            x=grid[..., 0],
            y=grid[..., 1],
            z=grid[..., 2],
            value=density,
            isomin=isomin,
            isomax=isomax,
            opacity=0.1,
            surface_count=surface_count,
            caps=dict(x_show=False, y_show=False, z_show=False),
        )
    )

    axis_dict = dict(
        showgrid=False,
        showbackground=False,
        zeroline=False,
        visible=False,
    )

    fig.add_trace(
        go.Scatter3d(
            x=atom_coord[:, 0],
            y=atom_coord[:, 1],
            z=atom_coord[:, 2],
            mode="markers",
            marker=dict(
                size=10,
                color=atom_type,
                cmin=0,
                cmax=4,
                colorscale=atom_colorscale,
                opacity=0.6,
            ),
        )
    )

    if title is not None:
        title = dict(
            text=title,
            x=0.5,
            y=0.3,
            xanchor="center",
            yanchor="bottom",
        )

    fig.update_layout(
        autosize=False,
        width=800,
        height=800,
        showlegend=False,
        scene=dict(xaxis=axis_dict, yaxis=axis_dict, zaxis=axis_dict),
        title=title,
        title_font_family="Times New Roman",
    )

    return fig


def safe_write_image(fig, path, show_plot=False):
    try:
        fig.write_image(path)
        logger.info(f"Image saved to: {path}")
    except Exception as e:
        logger.warning(f"Failed to save image {path}: {e}")
        try:
            html_path = path.with_suffix(".html")
            fig.write_html(html_path)
            logger.info(f"Saved interactive HTML instead: {html_path}")
        except Exception as html_e:
            logger.warning(f"Failed to save HTML fallback for {path}: {html_e}")

    if show_plot:
        try:
            if Image is None or display is None:
                raise ImportError("IPython not installed")
            img_bytes = fig.to_image(format="png", scale=2)
            display(Image(img_bytes))
        except Exception as e:
            logger.warning(f"Failed to display image: {e}")


def maybe_downsample_volume(grid, values, shape, max_points=250_000):
    """
    Downsample a regular 3D grid for visualization to keep Plotly volume traces responsive.
    grid: numpy array of shape (n_points, 3)
    values: list of numpy arrays aligned with grid, each of shape (n_points,)
    shape: original lattice shape [nx, ny, nz]
    """
    if shape is None or len(shape) != 3:
        return grid, values, False, 1

    try:
        shape = [int(s) for s in shape]
        total = shape[0] * shape[1] * shape[2]
    except Exception:
        return grid, values, False, 1

    if total != grid.shape[0] or any(val.shape[0] != grid.shape[0] for val in values):
        return grid, values, False, 1
    if total <= max_points:
        return grid, values, False, 1

    stride = max(1, math.ceil((total / max_points) ** (1 / 3)))
    try:
        grid_view = grid.reshape(shape[0], shape[1], shape[2], 3)
        grid_ds = grid_view[::stride, ::stride, ::stride, :].reshape(-1, 3)
        values_ds = [
            val.reshape(shape[0], shape[1], shape[2])[::stride, ::stride, ::stride].reshape(-1)
            for val in values
        ]
    except Exception as e:
        logger.warning(f"Failed to downsample grid for visualization: {e}")
        return grid, values, False, 1

    return grid_ds, values_ds, True, stride


def write_cube_generic(fileobj, atom_type, atom_coord, density, info, idx2atom_num=None):
    """
    Minimal cube writer for datasets without a built-in write_cube method.
    idx2atom_num maps dataset atom indices to atomic numbers (e.g., [6,1,8] for C/H/O).
    """
    fileobj.write("Cube file written on " + time.strftime("%c"))
    fileobj.write("\nOUTER LOOP: X, MIDDLE LOOP: Y, INNER LOOP: Z\n")
    cell = info["cell"]
    shape = info["shape"]
    origin = info.get("origin", np.zeros(3, dtype=np.float32))
    fileobj.write("{0:5}{1:12.6f}{2:12.6f}{3:12.6f}\n".format(len(atom_type), *origin))
    for s, c in zip(shape, cell):
        d = c / s
        fileobj.write("{0:5}{1:12.6f}{2:12.6f}{3:12.6f}\n".format(s, *d))
    for Z, (x, y, z) in zip(atom_type, atom_coord):
        atomic_num = int(idx2atom_num[int(Z)]) if idx2atom_num is not None else int(Z)
        fileobj.write(
            "{0:5}{1:12.6f}{2:12.6f}{3:12.6f}{4:12.6f}\n".format(
                atomic_num, float(atomic_num), x, y, z
            )
        )
    density.tofile(fileobj, sep="\n", format="%e")


def parse_grid_shape(shape_str):
    parts = [p.strip() for p in str(shape_str).split(",") if p.strip()]
    if len(parts) == 1:
        n = int(parts[0])
        if n <= 1:
            raise ValueError(f"Invalid mol_grid_shape {shape_str}, each dimension must be > 1")
        return [n, n, n]
    if len(parts) == 3:
        shape = [int(p) for p in parts]
        if any(s <= 1 for s in shape):
            raise ValueError(f"Invalid mol_grid_shape {shape_str}, each dimension must be > 1")
        return shape
    raise ValueError(f"Invalid mol_grid_shape {shape_str}, expected 'N' or 'Nx,Ny,Nz'")


def normalize_element_symbol(symbol):
    sym = str(symbol).strip()
    if len(sym) == 0:
        return sym
    if len(sym) == 1:
        return sym.upper()
    return sym[0].upper() + sym[1:].lower()


def load_atom_mapping(atom_file):
    with Path(atom_file).open() as f:
        atom_info = json.load(f)

    atom_name2idx = {}
    idx2atom_num = {}
    for idx, item in enumerate(atom_info):
        sym = normalize_element_symbol(item["name"])
        atom_name2idx[sym] = idx
        idx2atom_num[idx] = int(item["atom_num"])
    return atom_name2idx, idx2atom_num


def resolve_atom_file_for_mol(args_atom_file, dataset_atom_file):
    candidates = []
    if args_atom_file is not None:
        candidates.append(Path(args_atom_file).expanduser())
    if dataset_atom_file is not None:
        candidates.append(Path(dataset_atom_file).expanduser())

    for cand in candidates:
        if cand.exists():
            return cand

    fallback = Path("/home/liuxuwei01/processed_output/omol25.json")
    if fallback.exists():
        logger.warning(
            f"Configured atom_file not found ({candidates}); falling back to {fallback}"
        )
        return fallback

    raise FileNotFoundError(
        "Could not resolve atom_file for MOL inference. "
        f"Checked: {[str(c) for c in candidates]} and fallback {fallback}"
    )


def collect_mol_files(mol_input, mol_pattern):
    mol_path = Path(mol_input).expanduser()
    if mol_path.is_file():
        return [mol_path]
    if not mol_path.is_dir():
        raise FileNotFoundError(f"mol_input path not found: {mol_path}")

    files = sorted([p for p in mol_path.glob(mol_pattern) if p.is_file()])
    if not files:
        files = sorted([p for p in mol_path.iterdir() if p.is_file() and p.suffix.lower() == ".mol"])
    if not files:
        raise FileNotFoundError(f"No .mol files found in directory: {mol_path}")
    return files


def open_text_maybe_compressed(path):
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".lz4"):
        import lz4.frame

        return lz4.frame.open(path, mode="rt")
    if suffixes.endswith(".xz"):
        return lzma.open(path, mode="rt")
    if suffixes.endswith(".gz"):
        return gzip.open(path, mode="rt")
    return path.open(mode="rt")


def read_cube_density(path):
    with open_text_maybe_compressed(path) as f:
        f.readline()
        f.readline()
        line = f.readline().split()
        if len(line) < 4:
            raise ValueError(f"Invalid CUBE header (line 3) in {path}")
        n_atom = int(line[0])
        origin = np.array([float(x) for x in line[1:4]], dtype=np.float32)

        shape = []
        cell = np.zeros((3, 3), dtype=np.float32)
        for i in range(3):
            row = f.readline().split()
            if len(row) < 4:
                raise ValueError(f"Invalid CUBE axis line in {path}")
            n, x, y, z = [float(s) for s in row[:4]]
            shape.append(int(n))
            cell[i] = np.array([x, y, z], dtype=np.float32)

        x_coord = np.arange(shape[0], dtype=np.float32)[:, None] * cell[0][None, :]
        y_coord = np.arange(shape[1], dtype=np.float32)[:, None] * cell[1][None, :]
        z_coord = np.arange(shape[2], dtype=np.float32)[:, None] * cell[2][None, :]
        grid_coord = (
            x_coord.reshape(-1, 1, 1, 3)
            + y_coord.reshape(1, -1, 1, 3)
            + z_coord.reshape(1, 1, -1, 3)
        ).reshape(-1, 3)
        grid_coord = grid_coord + origin

        atom_coord_ref = []
        for _ in range(n_atom):
            row = f.readline().split()
            if len(row) < 5:
                raise ValueError(f"Invalid CUBE atom line in {path}")
            atom_coord_ref.append([float(row[2]), float(row[3]), float(row[4])])

        n_grid = shape[0] * shape[1] * shape[2]
        vals = []
        for line in f:
            parts = line.split()
            if parts:
                vals.extend(parts)
        if len(vals) < n_grid:
            raise ValueError(f"CUBE data too short in {path}: expect {n_grid}, got {len(vals)}")
        density = np.array(vals[:n_grid], dtype=np.float32)

    return (
        paddle.to_tensor(density, dtype="float32"),
        paddle.to_tensor(grid_coord, dtype="float32"),
        {
            "shape": shape,
            "cell": paddle.to_tensor(cell, dtype="float32"),
            "origin": paddle.to_tensor(origin, dtype="float32"),
            "atom_coord_ref": np.asarray(atom_coord_ref, dtype=np.float32),
        },
    )


def align_mol_atoms_to_cube(g, atom_coord_ref, sample_name, tol=0.05):
    if atom_coord_ref is None:
        return g
    ref = np.asarray(atom_coord_ref, dtype=np.float32)
    mol = g.pos.numpy().astype(np.float32)
    if ref.ndim != 2 or ref.shape[1] != 3:
        logger.warning(f"Invalid reference atom coordinates for {sample_name}, skip alignment")
        return g
    if mol.shape != ref.shape:
        logger.warning(
            f"Atom count mismatch for {sample_name} (mol={mol.shape[0]}, cube={ref.shape[0]}), "
            "skip alignment"
        )
        return g

    mol_center = mol.mean(axis=0)
    ref_center = ref.mean(axis=0)
    mol_c = mol - mol_center
    ref_c = ref - ref_center
    denom = float(np.sqrt((mol_c * mol_c).sum()))
    numer = float(np.sqrt((ref_c * ref_c).sum()))
    if denom < 1e-12 or numer < 1e-12:
        return g

    scale = numer / denom
    aligned = mol_c * scale + ref_center
    rms = float(np.sqrt(np.mean((aligned - ref) ** 2)))

    # Typical unit mismatch is Angstrom->Bohr (about 1.8897).
    # Apply alignment when scale obviously differs from 1.0 or residual is tiny after scaling.
    if abs(scale - 1.0) > tol or rms < 1e-3:
        g.pos = paddle.to_tensor(aligned, dtype="float32")
        logger.info(
            f"Aligned MOL coordinates to CUBE frame for {sample_name}: "
            f"scale={scale:.6f} (A->Bohr~{ANG2BOHR:.6f}), rms={rms:.6e}"
        )
    else:
        logger.info(
            f"No coordinate rescale needed for {sample_name}: scale={scale:.6f}, rms={rms:.6e}"
        )
    return g


def resolve_true_cube_for_mol(mol_path, true_cube_dir=None):
    base = sanitize_base_name(mol_path.name)
    base_density = f"{base[:-3]}Density" if base.endswith("Opt") else f"{base}Density"
    roots = []
    if true_cube_dir is not None:
        roots.append(Path(true_cube_dir).expanduser())
    roots.append(mol_path.parent)

    stems = [base, f"{base}_true", base_density]
    exts = [".cube", ".cub", ".cube.lz4", ".cube.gz", ".cube.xz", ".cub.lz4", ".cub.gz", ".cub.xz"]
    name_candidates = []
    for s in stems:
        for ext in exts:
            name_candidates.append(f"{s}{ext}")

    seen = set()
    uniq_candidates = []
    for name in name_candidates:
        if name not in seen:
            uniq_candidates.append(name)
            seen.add(name)

    for root in roots:
        if not root.exists():
            continue
        for name in uniq_candidates:
            p = root / name
            if p.is_file():
                return p
    return None


def parse_mol_v2000(mol_path):
    lines = mol_path.read_text(errors="replace").splitlines()
    if len(lines) < 4:
        raise ValueError(f"MOL file too short: {mol_path}")

    counts = lines[3]
    if "V3000" in counts.upper():
        raise NotImplementedError(f"V3000 MOL is not supported yet: {mol_path}")

    try:
        n_atom = int(counts[:3])
    except Exception:
        parts = counts.split()
        if len(parts) < 2:
            raise ValueError(f"Failed to parse counts line in MOL file: {mol_path}")
        n_atom = int(parts[0])

    atom_start = 4
    atom_end = atom_start + n_atom
    if len(lines) < atom_end:
        raise ValueError(f"Atom block incomplete in MOL file: {mol_path}")

    coords = []
    symbols = []
    for line in lines[atom_start:atom_end]:
        parts = line.split()
        x = y = z = None
        sym = None
        if len(parts) >= 4:
            try:
                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
                sym = parts[3]
            except Exception:
                x = y = z = None
                sym = None
        if x is None:
            try:
                x = float(line[0:10])
                y = float(line[10:20])
                z = float(line[20:30])
                sym = line[31:34].strip()
            except Exception as e:
                raise ValueError(f"Failed to parse atom line in {mol_path}: {line}") from e

        coords.append([x, y, z])
        symbols.append(normalize_element_symbol(sym))

    return np.asarray(coords, dtype=np.float32), symbols


def build_mol_sample(mol_path, atom_name2idx, mol_grid_shape, mol_grid_padding):
    atom_coord_np, atom_symbols = parse_mol_v2000(mol_path)

    atom_type_idx = []
    missing = set()
    for sym in atom_symbols:
        idx = atom_name2idx.get(sym)
        if idx is None:
            missing.add(sym)
        else:
            atom_type_idx.append(idx)
    if missing:
        raise ValueError(
            f"Found atoms not covered by atom_file mapping in {mol_path}: {sorted(missing)}"
        )

    atom_type = paddle.to_tensor(atom_type_idx, dtype="int64")
    atom_coord = paddle.to_tensor(atom_coord_np, dtype="float32")
    g = Data(x=atom_type, pos=atom_coord)

    shape = [int(s) for s in mol_grid_shape]
    min_coord = atom_coord_np.min(axis=0)
    max_coord = atom_coord_np.max(axis=0)
    span = np.maximum(max_coord - min_coord, np.array([1e-3, 1e-3, 1e-3], dtype=np.float32))
    axis_len = span + 2.0 * float(mol_grid_padding)
    center = 0.5 * (min_coord + max_coord)
    origin = center - 0.5 * axis_len

    x = np.linspace(origin[0], origin[0] + axis_len[0], num=shape[0], endpoint=False, dtype=np.float32)
    y = np.linspace(origin[1], origin[1] + axis_len[1], num=shape[1], endpoint=False, dtype=np.float32)
    z = np.linspace(origin[2], origin[2] + axis_len[2], num=shape[2], endpoint=False, dtype=np.float32)
    grid = np.stack(np.meshgrid(x, y, z, indexing="ij"), axis=-1).reshape(-1, 3).astype(np.float32)
    grid_coord = paddle.to_tensor(grid, dtype="float32")

    cell = np.diag(axis_len.astype(np.float32))
    info = {
        "shape": shape,
        "cell": paddle.to_tensor(cell, dtype="float32"),
        "origin": paddle.to_tensor(origin.astype(np.float32), dtype="float32"),
        "file_name": mol_path.name,
    }

    return g, None, grid_coord, info


def sanitize_base_name(sample_name):
    base_name = Path(sample_name).name
    for suf in [".lz4", ".zst", ".gz"]:
        if base_name.endswith(suf):
            base_name = base_name[: -len(suf)]
    for suf in [".cube", ".CHGCAR", ".json", ".mol"]:
        if base_name.endswith(suf):
            base_name = base_name[: -len(suf)]
    return base_name


def prepare_info_cube(info, grid_coord):
    info_cube = {}
    shape = info.get("shape")
    cell = info.get("cell")
    origin = info.get("origin", None)
    grid_np_full = grid_coord.detach().cpu().numpy()

    if shape is not None and len(shape) == 3:
        try:
            shape_i = [int(s) for s in shape]
            grid_view = grid_np_full.reshape(shape_i[0], shape_i[1], shape_i[2], 3)
            origin_np = grid_view[0, 0, 0]
            step_x = (
                grid_view[1, 0, 0] - grid_view[0, 0, 0]
                if shape_i[0] > 1
                else np.zeros(3, dtype=np.float32)
            )
            step_y = (
                grid_view[0, 1, 0] - grid_view[0, 0, 0]
                if shape_i[1] > 1
                else np.zeros(3, dtype=np.float32)
            )
            step_z = (
                grid_view[0, 0, 1] - grid_view[0, 0, 0]
                if shape_i[2] > 1
                else np.zeros(3, dtype=np.float32)
            )
            cell_from_grid = np.stack(
                [step_x * shape_i[0], step_y * shape_i[1], step_z * shape_i[2]], axis=0
            )
        except Exception:
            origin_np = None
            cell_from_grid = None
    else:
        origin_np = None
        cell_from_grid = None

    if shape is not None:
        info_cube["shape"] = [int(s) for s in shape]
    if cell is not None:
        if hasattr(cell, "numpy"):
            info_cube["cell"] = cell.numpy()
        else:
            info_cube["cell"] = np.array(cell, dtype=np.float32)
    if cell_from_grid is not None:
        info_cube["cell"] = cell_from_grid
    if origin is not None:
        if hasattr(origin, "numpy"):
            info_cube["origin"] = origin.numpy()
        else:
            info_cube["origin"] = np.array(origin, dtype=np.float32)
    if origin_np is not None:
        info_cube["origin"] = origin_np
    return info_cube


def main():
    parser = argparse.ArgumentParser(description="InfGCN electron density inference")
    parser.add_argument(
        "--config",
        default="electronic_structure/configs/infgcn/infgcn_qm9.yaml",
        help="Path to config yaml",
    )
    parser.add_argument(
        "--checkpoint",
        default="output/infgcn_qm9_best/infgcn_qm9.pdparams",
        help="Checkpoint (.pdparams) to load",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "validation", "test"],
        help="Dataset split to sample from",
    )
    parser.add_argument(
        "--index",
        default=0,
        type=int,
        help="Index within the chosen split",
    )
    parser.add_argument(
        "--data_root",
        default=None,
        help="Override dataset root; defaults to value in config",
    )
    parser.add_argument(
        "--split_file",
        default=None,
        help="Override split file path; defaults to value in config",
    )
    parser.add_argument(
        "--atom_file",
        default=None,
        help="Override atom info file; defaults to value in config",
    )
    parser.add_argument(
        "--output_dir",
        default="./results",
        help="Directory to store predictions/visualizations",
    )
    parser.add_argument(
        "--grid_batch_size",
        default=4096,
        type=int,
        help="Number of grid points per forward pass",
    )
    parser.add_argument(
        "--skip_vis",
        action="store_true",
        help="Skip writing/visualizing density plots",
    )
    parser.add_argument(
        "--save_true_cube",
        action="store_true",
        help="Save reference (DFT) electron density as a cube file",
    )
    parser.add_argument(
        "--save_pred_cube",
        action="store_true",
        help="Save predicted electron density as a cube file",
    )
    parser.add_argument(
        "--save_html",
        action="store_true",
        help="Save Plotly figures as interactive HTML (in addition to PNG)",
    )
    parser.add_argument(
        "--cube_dir",
        default=None,
        help="Directory to store cube files (defaults to output_dir)",
    )
    parser.add_argument(
        "--show_plot",
        action="store_true",
        help="Display plotly figures inline (requires kaleido)",
    )
    parser.add_argument(
        "--mol_input",
        default=None,
        help="Path to a .mol file or a directory of .mol files for direct structure inference",
    )
    parser.add_argument(
        "--mol_pattern",
        default="*.mol",
        help="Glob pattern when --mol_input is a directory",
    )
    parser.add_argument(
        "--mol_grid_shape",
        default="80,80,80",
        help="Grid shape for MOL inference, e.g. '80' or '80,80,80'",
    )
    parser.add_argument(
        "--mol_grid_padding",
        default=6.0,
        type=float,
        help="Padding (Angstrom) around molecular coordinates for MOL grid generation",
    )
    parser.add_argument(
        "--mol_true_cube_dir",
        default=None,
        help=(
            "Optional directory containing reference/true CUBE files for MOL inputs. "
            "Expected names: <mol_basename>.cube or <mol_basename>_true.cube"
        ),
    )
    args = parser.parse_args()

    set_random_seed(42)

    cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.to_container(cfg, resolve=True)

    split_key = "val" if args.split == "validation" else args.split
    ds_cfg_full = cfg["Dataset"][split_key]["dataset"]
    dataset_cfg = ds_cfg_full.get("__init_params__", {})
    dataset_params = copy.deepcopy(dataset_cfg)
    dataset_params["split"] = args.split
    if args.data_root is not None:
        dataset_params["root"] = args.data_root
    if args.split_file is not None:
        dataset_params["split_file"] = args.split_file
    if args.atom_file is not None:
        dataset_params["atom_file"] = args.atom_file

    use_mol_mode = args.mol_input is not None

    dataset = None
    cube_writer = None
    idx2atom_num = None
    atom_name2idx = None
    mol_files = []
    mol_grid_shape = None

    if use_mol_mode:
        atom_file_path = resolve_atom_file_for_mol(
            args.atom_file,
            dataset_params.get("atom_file"),
        )
        atom_name2idx, idx2atom_num = load_atom_mapping(atom_file_path)
        mol_files = collect_mol_files(args.mol_input, args.mol_pattern)
        mol_grid_shape = parse_grid_shape(args.mol_grid_shape)
        cube_writer = lambda f, a, c, d, i: write_cube_generic(
            f, a, c, d, i, idx2atom_num
        )
        logger.info(
            f"MOL mode enabled: {len(mol_files)} file(s), atom_file={atom_file_path}, "
            f"grid_shape={mol_grid_shape}, padding={args.mol_grid_padding}, "
            f"true_cube_dir={args.mol_true_cube_dir}"
        )
    else:
        dataset_cls_name = ds_cfg_full.get("__class_name__", "DensityDataset")
        dataset_cls_map = {
            "DensityDataset": DensityDataset,
            "SmallDensityDataset": SmallDensityDataset,
        }
        if dataset_cls_name not in dataset_cls_map:
            raise ValueError(f"Unsupported dataset class {dataset_cls_name}")
        dataset = dataset_cls_map[dataset_cls_name](**dataset_params)
        cube_writer = getattr(dataset, "write_cube", None)
        idx2atom_num = getattr(dataset, "idx2atom_num", None)
        if cube_writer is None:
            if isinstance(dataset, SmallDensityDataset):
                # Atom order in SmallDensityDataset: C=0, H=1, O=2
                idx2atom_num = np.array([6, 1, 8], dtype=np.int64)
                cube_writer = lambda f, a, c, d, i: write_cube_generic(
                    f, a, c, d, i, idx2atom_num
                )
            else:
                cube_writer = lambda *args, **kwargs: (_ for _ in ()).throw(
                    AttributeError("Cube writer not available for this dataset")
                )
        if args.index >= len(dataset):
            raise IndexError(
                f"Index {args.index} exceeds dataset size {len(dataset)} for split {args.split}"
            )

    device = "gpu" if paddle.is_compiled_with_cuda() else "cpu"
    paddle.set_device(device)
    logger.info(f"Running inference on device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cube_dir = Path(args.cube_dir) if args.cube_dir is not None else output_dir
    cube_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading the pretrained model from {args.checkpoint}")
    model = get_pretrained_model(args.config, args.checkpoint)
    logger.info("Model loaded successfully.")

    if use_mol_mode:
        sample_iter = tqdm(mol_files, desc="MOL inference")
    else:
        sample_iter = [args.index]

    for sample_item in sample_iter:
        if use_mol_mode:
            mol_path = sample_item
            g, density, grid_coord, info = build_mol_sample(
                mol_path,
                atom_name2idx,
                mol_grid_shape,
                args.mol_grid_padding,
            )
            true_cube_path = resolve_true_cube_for_mol(mol_path, args.mol_true_cube_dir)
            if true_cube_path is not None:
                try:
                    density, grid_coord, info_ref = read_cube_density(true_cube_path)
                    g = align_mol_atoms_to_cube(g, info_ref.get("atom_coord_ref"), mol_path.name)
                    info = dict(info_ref)
                    info["file_name"] = mol_path.name
                    info["true_cube_file"] = str(true_cube_path)
                    logger.info(f"Using reference cube for {mol_path.name}: {true_cube_path}")
                except Exception as e:
                    logger.warning(
                        f"Failed to read reference cube for {mol_path.name} at {true_cube_path}: {e}"
                    )
            sample_name = info.get("file_name", mol_path.name)
        else:
            sample_name = f"{args.split}_{args.index}"
            g, density, grid_coord, info = dataset[args.index]
            sample_name = info.get("file_name", sample_name)

        g.batch = paddle.zeros_like(g.x)
        g = g.to(device)
        if density is not None:
            density = density.to(device)
        grid_coord = grid_coord.to(device)

        logger.info(f"Starting prediction for sample: {sample_name}")
        preds, loss, mae = inference_model(
            model,
            g,
            density,
            grid_coord[None],
            [info],
            grid_batch_size=args.grid_batch_size,
        )
        if loss is not None and mae is not None:
            logger.info(
                f"Prediction completed for {sample_name}, "
                f"Loss: {float(loss):.6f}, MAE: {float(mae):.6f}"
            )
        else:
            logger.info(f"Prediction completed for {sample_name} (no reference density)")

        sample_tag = sanitize_base_name(sample_name)

        if args.save_true_cube or args.save_pred_cube:
            atom_type_np = g.x.detach().cpu().numpy()
            atom_coord_np = g.pos.detach().cpu().numpy()
            info_cube = prepare_info_cube(info, grid_coord)

            if args.save_true_cube:
                if density is None:
                    logger.warning(
                        f"Skipping true cube for {sample_name}: no reference density available"
                    )
                else:
                    true_cube_path = cube_dir / f"{sample_tag}_true.cube"
                    with true_cube_path.open("w") as f:
                        cube_writer(
                            f,
                            atom_type_np,
                            atom_coord_np,
                            density.detach().cpu().numpy(),
                            info_cube,
                        )
                    logger.info(f"Saved reference density cube to: {true_cube_path}")

            if args.save_pred_cube:
                pred_cube_path = cube_dir / f"{sample_tag}_pred.cube"
                with pred_cube_path.open("w") as f:
                    cube_writer(
                        f,
                        atom_type_np,
                        atom_coord_np,
                        preds.detach().cpu().numpy(),
                        info_cube,
                    )
                logger.info(f"Saved predicted density cube to: {pred_cube_path}")

        if not args.skip_vis:
            grid_np = grid_coord.detach().cpu().numpy()
            preds_np = preds.detach().cpu().numpy()
            shape = info.get("shape")
            atom_type = g.x.detach().cpu().numpy()
            atom_coord = g.pos.detach().cpu().numpy()

            if density is not None:
                density_np = density.detach().cpu().numpy()
                diff_np = density_np - preds_np
                grid_vis, (density_vis, diff_vis, preds_vis), did_downsample, stride = (
                    maybe_downsample_volume(
                        grid_np,
                        [density_np, diff_np, preds_np],
                        shape if shape is None else [int(s) for s in shape],
                    )
                )
                if did_downsample:
                    logger.warning(
                        f"Downsampled volume grid from {grid_np.shape[0]} to {grid_vis.shape[0]} "
                        f"points for visualization (stride={stride}) to keep HTML output responsive."
                    )

                logger.info("Visualizing the DFT electron density")
                fig = draw_volume(
                    grid_vis,
                    density_vis,
                    atom_type,
                    atom_coord,
                    isomin=0.05,
                    isomax=3.5,
                    surface_count=5,
                    title="DFT electron density",
                )
                true_density_path = output_dir / f"{sample_tag}_true_density.png"
                safe_write_image(fig, true_density_path, show_plot=args.show_plot)
                if args.save_html:
                    fig.write_html(output_dir / f"{sample_tag}_true_density.html")

                logger.info("Visualizing electron density difference")
                fig = draw_volume(
                    grid_vis,
                    diff_vis,
                    atom_type,
                    atom_coord,
                    isomin=-0.06,
                    isomax=0.06,
                    surface_count=4,
                    title="Electron Density Difference",
                )
                diff_density_path = output_dir / f"{sample_tag}_diff_density.png"
                safe_write_image(fig, diff_density_path, show_plot=args.show_plot)
                if args.save_html:
                    fig.write_html(output_dir / f"{sample_tag}_diff_density.html")

                logger.info("Visualizing predicted electron density")
                fig = draw_volume(
                    grid_vis,
                    preds_vis,
                    atom_type,
                    atom_coord,
                    isomin=0.05,
                    isomax=3.5,
                    surface_count=5,
                    title="Predicted Electron Density",
                )
                pred_density_path = output_dir / f"{sample_tag}_pred_density.png"
                safe_write_image(fig, pred_density_path, show_plot=args.show_plot)
                if args.save_html:
                    fig.write_html(output_dir / f"{sample_tag}_pred_density.html")
            else:
                grid_vis, (preds_vis,), did_downsample, stride = maybe_downsample_volume(
                    grid_np,
                    [preds_np],
                    shape if shape is None else [int(s) for s in shape],
                )
                if did_downsample:
                    logger.warning(
                        f"Downsampled volume grid from {grid_np.shape[0]} to {grid_vis.shape[0]} "
                        f"points for visualization (stride={stride}) to keep HTML output responsive."
                    )

                logger.info("Visualizing predicted electron density")
                fig = draw_volume(
                    grid_vis,
                    preds_vis,
                    atom_type,
                    atom_coord,
                    isomin=0.05,
                    isomax=3.5,
                    surface_count=5,
                    title="Predicted Electron Density",
                )
                pred_density_path = output_dir / f"{sample_tag}_pred_density.png"
                safe_write_image(fig, pred_density_path, show_plot=args.show_plot)
                if args.save_html:
                    fig.write_html(output_dir / f"{sample_tag}_pred_density.html")


if __name__ == "__main__":
    main()
