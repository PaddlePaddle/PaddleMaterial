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
from pathlib import Path

import paddle
import plotly.graph_objects as go
from omegaconf import OmegaConf
from tqdm import tqdm

try:
    from IPython.display import Image, display
except ImportError:  # Optional dependency; visualization still works for files
    Image, display = None, None

from ppmat.datasets import DensityDataset
from ppmat.models import build_model
from ppmat.utils import logger
from ppmat.utils.misc import set_random_seed


def get_pretrained_model(cfg_path, model_path):
    logger.info(f"from {cfg_path} loading config")
    cfg = OmegaConf.load(cfg_path)
    cfg = OmegaConf.to_container(cfg, resolve=True)

    model = build_model(cfg["Model"])
    logger.info(f"from {model_path}loading model")
    state_dict = paddle.load(model_path)
    if "model" in state_dict:
        model.set_state_dict(state_dict["model"])
    else:
        model.set_state_dict(state_dict)
    return model


def inference_model(model, g, density, grid_coord, infos, grid_batch_size=8196):
    with paddle.no_grad():
        model.eval()
        if grid_batch_size is None:
            preds = model(g.x, g.pos, grid_coord, g.batch, infos).squeeze(0)
        else:
            preds = []
            total = grid_coord.shape[1]
            step = grid_batch_size
            num_iter = (total + step - 1) // step
            for start in tqdm(range(0, total, step), total=num_iter):
                end = min(start + step, total)
                grid = grid_coord[:, start:end]
                preds.append(model(g.x, g.pos, grid, g.batch, infos).squeeze(0))
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
        "--show_plot",
        action="store_true",
        help="Display plotly figures inline (requires kaleido)",
    )
    args = parser.parse_args()

    set_random_seed(42)

    cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.to_container(cfg, resolve=True)

    split_key = "val" if args.split == "validation" else args.split
    dataset_cfg = cfg["Dataset"][split_key]["dataset"]["__init_params__"]
    dataset_params = copy.deepcopy(dataset_cfg)
    dataset_params["split"] = args.split
    if args.data_root is not None:
        dataset_params["root"] = args.data_root
    if args.split_file is not None:
        dataset_params["split_file"] = args.split_file
    if args.atom_file is not None:
        dataset_params["atom_file"] = args.atom_file
    dataset = DensityDataset(**dataset_params)
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

    if not args.skip_vis:
        logger.info("Visualizing the DFT electron density")
        fig = draw_volume(
            grid_coord.detach().cpu().numpy(),
            density.detach().cpu().numpy(),
            g.x.detach().cpu().numpy(),
            g.pos.detach().cpu().numpy(),
            isomin=0.05,
            isomax=3.5,
            surface_count=5,
            title="DFT electron density",
        )
        true_density_path = output_dir / f"{sample_name}_true_density.png"
        safe_write_image(fig, true_density_path, show_plot=args.show_plot)

        logger.info("Visualizing electron density difference")
        fig = draw_volume(
            grid_coord.detach().cpu().numpy(),
            (density - preds).detach().cpu().numpy(),
            g.x.detach().cpu().numpy(),
            g.pos.detach().cpu().numpy(),
            isomin=-0.06,
            isomax=0.06,
            surface_count=4,
            title="Electron Density Difference",
        )
        diff_density_path = output_dir / f"{sample_name}_diff_density.png"
        safe_write_image(fig, diff_density_path, show_plot=args.show_plot)

        logger.info("Visualizing predicted electron density")
        fig = draw_volume(
            grid_coord.detach().cpu().numpy(),
            preds.detach().cpu().numpy(),
            g.x.detach().cpu().numpy(),
            g.pos.detach().cpu().numpy(),
            isomin=0.05,
            isomax=3.5,
            surface_count=5,
            title="Predicted Electron Density",
        )
        pred_density_path = output_dir / f"{sample_name}_pred_density.png"
        safe_write_image(fig, pred_density_path, show_plot=args.show_plot)


if __name__ == "__main__":
    main()
