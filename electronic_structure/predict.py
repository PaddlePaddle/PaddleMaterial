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
from ppmat.models import build_model
from ppmat.utils import logger
from ppmat.utils.misc import set_random_seed


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

        mask = (density > 0).astype(dtype="float32")
        preds = preds * mask
        density = density * mask
        diff = paddle.abs(preds - density)
        loss = diff.pow(2).sum()
        mae = diff.sum() / density.sum()
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
    args = parser.parse_args()

    set_random_seed(42)

    cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.to_container(cfg, resolve=True)

    split_key = "val" if args.split == "validation" else args.split
    ds_cfg_full = cfg["Dataset"][split_key]["dataset"]
    dataset_cls_name = ds_cfg_full.get("__class_name__", "DensityDataset")
    dataset_cfg = ds_cfg_full.get("__init_params__", {})
    dataset_params = copy.deepcopy(dataset_cfg)
    dataset_params["split"] = args.split
    if args.data_root is not None:
        dataset_params["root"] = args.data_root
    if args.split_file is not None:
        dataset_params["split_file"] = args.split_file
    if args.atom_file is not None:
        dataset_params["atom_file"] = args.atom_file
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

    sample_name = f"{args.split}_{args.index}"
    g, density, grid_coord, info = dataset[args.index]
    sample_name = info.get("file_name", sample_name)

    device = "gpu" if paddle.is_compiled_with_cuda() else "cpu"
    paddle.set_device(device)
    logger.info(f"Running inference on device: {device}")

    g.batch = paddle.zeros_like(g.x)
    g = g.to(device)
    density = density.to(device)
    grid_coord = grid_coord.to(device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading the pretrained model from {args.checkpoint}")
    model = get_pretrained_model(args.config, args.checkpoint)
    logger.info("Model loaded successfully.")

    logger.info("Starting prediction")
    preds, loss, mae = inference_model(
        model,
        g,
        density,
        grid_coord[None],
        [info],
        grid_batch_size=args.grid_batch_size,
    )
    logger.info(f"Prediction completed, Loss: {float(loss):.6f}, MAE: {float(mae):.6f}")

    cube_dir = Path(args.cube_dir) if args.cube_dir is not None else output_dir
    cube_dir.mkdir(parents=True, exist_ok=True)

    if args.save_true_cube or args.save_pred_cube:
        atom_type_np = g.x.detach().cpu().numpy()
        atom_coord_np = g.pos.detach().cpu().numpy()

        info_cube = {}
        shape = info.get("shape")
        cell = info.get("cell")
        origin = info.get("origin", None)
        grid_np_full = grid_coord.detach().cpu().numpy()
        if shape is not None and len(shape) == 3:
            try:
                grid_view = grid_np_full.reshape(shape[0], shape[1], shape[2], 3)
                origin_np = grid_view[0, 0, 0]
                step_x = (
                    grid_view[1, 0, 0] - grid_view[0, 0, 0]
                    if shape[0] > 1
                    else np.zeros(3, dtype=np.float32)
                )
                step_y = (
                    grid_view[0, 1, 0] - grid_view[0, 0, 0]
                    if shape[1] > 1
                    else np.zeros(3, dtype=np.float32)
                )
                step_z = (
                    grid_view[0, 0, 1] - grid_view[0, 0, 0]
                    if shape[2] > 1
                    else np.zeros(3, dtype=np.float32)
                )
                # write_cube expects lattice vectors; multiply voxel steps by shape
                # so that write_cube's division recovers the original voxel step.
                cell_from_grid = np.stack(
                    [step_x * shape[0], step_y * shape[1], step_z * shape[2]], axis=0
                )
            except Exception:
                grid_view = None
                origin_np = None
                cell_from_grid = None
        else:
            grid_view = None
            origin_np = None
            cell_from_grid = None

        if shape is not None:
            info_cube["shape"] = [int(s) for s in shape]
        if cell is not None:
            if hasattr(cell, "numpy"):
                cell_np = cell.numpy()
            else:
                cell_np = np.array(cell, dtype=np.float32)
            info_cube["cell"] = cell_np
        if cell_from_grid is not None:
            info_cube["cell"] = cell_from_grid
        if origin is not None:
            if hasattr(origin, "numpy"):
                origin_np = origin.numpy()
            else:
                origin_np = np.array(origin, dtype=np.float32)
            info_cube["origin"] = origin_np
        if origin_np is not None:
            info_cube["origin"] = origin_np

        # Strip compression / double extensions for cleaner filenames
        base_name = Path(sample_name).name
        for suf in [".lz4", ".zst", ".gz"]:
            if base_name.endswith(suf):
                base_name = base_name[: -len(suf)]
        for suf in [".cube", ".CHGCAR", ".json"]:
            if base_name.endswith(suf):
                base_name = base_name[: -len(suf)]

        if args.save_true_cube:
            true_cube_path = cube_dir / f"{base_name}_true.cube"
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
            pred_cube_path = cube_dir / f"{base_name}_pred.cube"
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
        # Move tensors to CPU and optionally downsample for lightweight rendering
        grid_np = grid_coord.detach().cpu().numpy()
        density_np = density.detach().cpu().numpy()
        preds_np = preds.detach().cpu().numpy()
        diff_np = density_np - preds_np

        shape = info.get("shape")
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

        atom_type = g.x.detach().cpu().numpy()
        atom_coord = g.pos.detach().cpu().numpy()

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
        true_density_path = output_dir / f"{sample_name}_true_density.png"
        safe_write_image(fig, true_density_path, show_plot=args.show_plot)
        if args.save_html:
            fig.write_html(output_dir / f"{sample_name}_true_density.html")

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
        diff_density_path = output_dir / f"{sample_name}_diff_density.png"
        safe_write_image(fig, diff_density_path, show_plot=args.show_plot)
        if args.save_html:
            fig.write_html(output_dir / f"{sample_name}_diff_density.html")

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
        pred_density_path = output_dir / f"{sample_name}_pred_density.png"
        safe_write_image(fig, pred_density_path, show_plot=args.show_plot)
        if args.save_html:
            fig.write_html(output_dir / f"{sample_name}_pred_density.html")


if __name__ == "__main__":
    main()
