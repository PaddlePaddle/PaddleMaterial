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


import copy
import json
from functools import partial
from pathlib import Path

import numpy as np
import paddle
from omegaconf import OmegaConf
from tqdm import tqdm

from ppmat.datasets import DensityDataset
from ppmat.datasets import SmallDensityDataset
from ppmat.datasets.geometric_data_type.data import Data
from ppmat.models import build_model
from ppmat.models import build_model_from_name
from ppmat.predictor.base import BasePredictor
from ppmat.utils import logger
from ppmat.utils.io import prepare_info_cube
from ppmat.utils.io import read_cube_density
from ppmat.utils.io import unavailable_cube_writer
from ppmat.utils.io import write_cube_generic
from ppmat.utils.misc import set_random_seed
from ppmat.utils.visualization import draw_volume
from ppmat.utils.visualization import maybe_downsample_volume
from ppmat.utils.visualization import safe_write_image

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


def apply_predict_config(args, cfg):
    predict_cfg = cfg.get("Predict", {}) or {}
    defaults = {
        "split": "test",
        "index": 0,
        "data_root": None,
        "split_file": None,
        "atom_file": None,
        "output_dir": "./results",
        "grid_batch_size": 4096,
        "skip_vis": False,
        "save_true_cube": False,
        "save_pred_cube": False,
        "save_html": False,
        "cube_dir": None,
        "show_plot": False,
        "mol_pattern": "*.mol",
        "mol_grid_shape": "80,80,80",
        "mol_grid_padding": 6.0,
        "mol_true_cube_dir": None,
    }

    for name, default in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, predict_cfg.get(name, default))

    return args


def inference_model(model, g, density, grid_coord, infos, grid_batch_size=8196):
    with paddle.no_grad():
        model.eval()
        device = paddle.get_device()
        prepared_infos = (
            model._prepare_infos(infos, device)
            if hasattr(model, "_prepare_infos")
            else infos
        )
        if grid_batch_size is None:
            if hasattr(model, "_forward_density"):
                preds = model._forward_density(
                    g.x, g.pos, grid_coord, g.batch, prepared_infos
                ).squeeze(0)
            else:
                # Fallback for legacy models expecting raw tensors
                preds = model(g.x, g.pos, grid_coord, g.batch, prepared_infos).squeeze(
                    0
                )
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


def parse_grid_shape(shape_str):
    parts = [p.strip() for p in str(shape_str).split(",") if p.strip()]
    if len(parts) == 1:
        n = int(parts[0])
        if n <= 1:
            raise ValueError(
                f"Invalid mol_grid_shape {shape_str}, each dimension must be > 1"
            )
        return [n, n, n]
    if len(parts) == 3:
        shape = [int(p) for p in parts]
        if any(s <= 1 for s in shape):
            raise ValueError(
                f"Invalid mol_grid_shape {shape_str}, each dimension must be > 1"
            )
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
        files = sorted(
            [
                p
                for p in mol_path.iterdir()
                if p.is_file() and p.suffix.lower() == ".mol"
            ]
        )
    if not files:
        raise FileNotFoundError(f"No .mol files found in directory: {mol_path}")
    return files


def align_mol_atoms_to_cube(g, atom_coord_ref, sample_name, tol=0.05):
    if atom_coord_ref is None:
        return g
    ref = np.asarray(atom_coord_ref, dtype=np.float32)
    mol = g.pos.numpy().astype(np.float32)
    if ref.ndim != 2 or ref.shape[1] != 3:
        logger.warning(
            f"Invalid reference atom coordinates for {sample_name}, skip alignment"
        )
        return g
    if mol.shape != ref.shape:
        logger.warning(
            f"Atom count mismatch for {sample_name} "
            f"(mol={mol.shape[0]}, cube={ref.shape[0]}), "
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
    # Apply alignment when scale differs from 1.0 or residual is tiny after scaling.
    if abs(scale - 1.0) > tol or rms < 1e-3:
        g.pos = paddle.to_tensor(aligned, dtype="float32")
        logger.info(
            f"Aligned MOL coordinates to CUBE frame for {sample_name}: "
            f"scale={scale:.6f} (A->Bohr~{ANG2BOHR:.6f}), rms={rms:.6e}"
        )
    else:
        logger.info(
            f"No coordinate rescale needed for {sample_name}: "
            f"scale={scale:.6f}, rms={rms:.6e}"
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
    exts = [
        ".cube",
        ".cub",
        ".cube.lz4",
        ".cube.gz",
        ".cube.xz",
        ".cub.lz4",
        ".cub.gz",
        ".cub.xz",
    ]
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
                raise ValueError(
                    f"Failed to parse atom line in {mol_path}: {line}"
                ) from e

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
            "Found atoms not covered by atom_file mapping in "
            f"{mol_path}: {sorted(missing)}"
        )

    atom_type = paddle.to_tensor(atom_type_idx, dtype="int64")
    atom_coord = paddle.to_tensor(atom_coord_np, dtype="float32")
    g = Data(x=atom_type, pos=atom_coord)

    shape = [int(s) for s in mol_grid_shape]
    min_coord = atom_coord_np.min(axis=0)
    max_coord = atom_coord_np.max(axis=0)
    span = np.maximum(
        max_coord - min_coord, np.array([1e-3, 1e-3, 1e-3], dtype=np.float32)
    )
    axis_len = span + 2.0 * float(mol_grid_padding)
    center = 0.5 * (min_coord + max_coord)
    origin = center - 0.5 * axis_len

    x = np.linspace(
        origin[0],
        origin[0] + axis_len[0],
        num=shape[0],
        endpoint=False,
        dtype=np.float32,
    )
    y = np.linspace(
        origin[1],
        origin[1] + axis_len[1],
        num=shape[1],
        endpoint=False,
        dtype=np.float32,
    )
    z = np.linspace(
        origin[2],
        origin[2] + axis_len[2],
        num=shape[2],
        endpoint=False,
        dtype=np.float32,
    )
    grid = (
        np.stack(np.meshgrid(x, y, z, indexing="ij"), axis=-1)
        .reshape(-1, 3)
        .astype(np.float32)
    )
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


class FieldPredictor(BasePredictor):
    """Electron-density field predictor."""

    def __init__(
        self,
        model_name=None,
        weights_name=None,
        config_path=None,
        checkpoint_path=None,
        seed=42,
    ):
        super().__init__(
            model_name=model_name,
            weights_name=weights_name,
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            work_dir="",
            device=None,
        )
        set_random_seed(seed)
        self._load_model()
        self.model.eval()
        logger.info("Model loaded successfully.")

    def _load_model(self):
        if self.model_name is not None:
            logger.info(f"Loading registered model: {self.model_name}")
            self.model, self.config = build_model_from_name(
                self.model_name, self.weights_name
            )
        else:
            assert self.config_path is not None and self.checkpoint_path is not None, (
                "config_path and checkpoint_path must be provided when model_name "
                "is None."
            )
            logger.info(f"Loading the pretrained model from {self.checkpoint_path}")
            cfg = OmegaConf.load(self.config_path)
            self.config = OmegaConf.to_container(cfg, resolve=True)
            self.model = get_pretrained_model(self.config_path, self.checkpoint_path)

    def predict(self, args):
        return run_prediction(args, self.model, self.config)

    @staticmethod
    def _save_cubes(
        args,
        cube_dir,
        cube_writer,
        sample_name,
        sample_tag,
        g,
        density,
        preds,
        info,
        grid_coord,
    ):
        if not (args.save_true_cube or args.save_pred_cube):
            return

        atom_type_np = g.x.detach().cpu().numpy()
        atom_coord_np = g.pos.detach().cpu().numpy()
        info_cube = prepare_info_cube(info, grid_coord)

        if args.save_true_cube:
            if density is None:
                logger.warning(
                    f"Skipping true cube for {sample_name}: "
                    "no reference density available"
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

    @staticmethod
    def _save_visualizations(
        args,
        output_dir,
        sample_tag,
        g,
        density,
        preds,
        info,
        grid_coord,
    ):
        if args.skip_vis:
            return

        grid_np = grid_coord.detach().cpu().numpy()
        preds_np = preds.detach().cpu().numpy()
        shape = info.get("shape")
        atom_type = g.x.detach().cpu().numpy()
        atom_coord = g.pos.detach().cpu().numpy()
        shape = shape if shape is None else [int(s) for s in shape]

        if density is not None:
            density_np = density.detach().cpu().numpy()
            diff_np = density_np - preds_np
            (
                grid_vis,
                (density_vis, diff_vis, preds_vis),
                did_downsample,
                stride,
            ) = maybe_downsample_volume(
                grid_np,
                [density_np, diff_np, preds_np],
                shape,
            )
            FieldPredictor._log_downsample(grid_np, grid_vis, did_downsample, stride)

            FieldPredictor._write_volume_plot(
                args,
                output_dir,
                sample_tag,
                "true_density",
                "DFT electron density",
                grid_vis,
                density_vis,
                atom_type,
                atom_coord,
                isomin=0.05,
                isomax=3.5,
                surface_count=5,
            )
            FieldPredictor._write_volume_plot(
                args,
                output_dir,
                sample_tag,
                "diff_density",
                "Electron Density Difference",
                grid_vis,
                diff_vis,
                atom_type,
                atom_coord,
                isomin=-0.06,
                isomax=0.06,
                surface_count=4,
            )
            FieldPredictor._write_volume_plot(
                args,
                output_dir,
                sample_tag,
                "pred_density",
                "Predicted Electron Density",
                grid_vis,
                preds_vis,
                atom_type,
                atom_coord,
                isomin=0.05,
                isomax=3.5,
                surface_count=5,
            )
            return

        (
            grid_vis,
            (preds_vis,),
            did_downsample,
            stride,
        ) = maybe_downsample_volume(grid_np, [preds_np], shape)
        FieldPredictor._log_downsample(grid_np, grid_vis, did_downsample, stride)
        FieldPredictor._write_volume_plot(
            args,
            output_dir,
            sample_tag,
            "pred_density",
            "Predicted Electron Density",
            grid_vis,
            preds_vis,
            atom_type,
            atom_coord,
            isomin=0.05,
            isomax=3.5,
            surface_count=5,
        )

    @staticmethod
    def _log_downsample(grid_np, grid_vis, did_downsample, stride):
        if did_downsample:
            logger.warning(
                f"Downsampled volume grid from {grid_np.shape[0]} "
                f"to {grid_vis.shape[0]} points for visualization "
                f"(stride={stride}) to keep HTML output responsive."
            )

    @staticmethod
    def _write_volume_plot(
        args,
        output_dir,
        sample_tag,
        suffix,
        title,
        grid,
        values,
        atom_type,
        atom_coord,
        isomin,
        isomax,
        surface_count,
    ):
        logger.info(f"Visualizing {title}")
        fig = draw_volume(
            grid,
            values,
            atom_type,
            atom_coord,
            isomin=isomin,
            isomax=isomax,
            surface_count=surface_count,
            title=title,
        )
        image_path = output_dir / f"{sample_tag}_{suffix}.png"
        safe_write_image(fig, image_path, show_plot=args.show_plot)
        if args.save_html:
            fig.write_html(output_dir / f"{sample_tag}_{suffix}.html")


def run_prediction(args, model, cfg):
    apply_predict_config(args, cfg)

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
        cube_writer = partial(write_cube_generic, idx2atom_num=idx2atom_num)
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
                cube_writer = partial(
                    write_cube_generic,
                    idx2atom_num=idx2atom_num,
                )
            else:
                cube_writer = unavailable_cube_writer
        if args.index >= len(dataset):
            raise IndexError(
                f"Index {args.index} exceeds dataset size {len(dataset)} "
                f"for split {args.split}"
            )

    device = "gpu" if paddle.is_compiled_with_cuda() else "cpu"
    paddle.set_device(device)
    logger.info(f"Running inference on device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cube_dir = Path(args.cube_dir) if args.cube_dir is not None else output_dir
    cube_dir.mkdir(parents=True, exist_ok=True)

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
                    g = align_mol_atoms_to_cube(
                        g, info_ref.get("atom_coord_ref"), mol_path.name
                    )
                    info = dict(info_ref)
                    info["file_name"] = mol_path.name
                    info["true_cube_file"] = str(true_cube_path)
                    logger.info(
                        f"Using reference cube for {mol_path.name}: {true_cube_path}"
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to read reference cube for {mol_path.name} "
                        f"at {true_cube_path}: {e}"
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
            logger.info(
                f"Prediction completed for {sample_name} (no reference density)"
            )

        sample_tag = sanitize_base_name(sample_name)

        FieldPredictor._save_cubes(
            args,
            cube_dir,
            cube_writer,
            sample_name,
            sample_tag,
            g,
            density,
            preds,
            info,
            grid_coord,
        )
        FieldPredictor._save_visualizations(
            args,
            output_dir,
            sample_tag,
            g,
            density,
            preds,
            info,
            grid_coord,
        )
