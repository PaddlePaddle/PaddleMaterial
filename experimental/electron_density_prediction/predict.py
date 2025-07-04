# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

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

import paddle
import plotly.graph_objects as go
from IPython.display import Image
from IPython.display import display
from tqdm import tqdm

# from ppmat.datasets import DensityDataset
from ppmat.datasets import SmallDensityDataset
from ppmat.models import build_model
from ppmat.utils import logger
from ppmat.utils.misc import set_random_seed


def split_tensor_func(self, split_size, dim=0):
    total_size = self.shape[dim]

    if isinstance(split_size, int):
        sections = []
        for i in range(0, total_size, split_size):
            sections.append(min(split_size, total_size - i))
        return paddle.split(self, sections, dim)
    else:
        return paddle.split(self, split_size, dim)


setattr(paddle.Tensor, "split", split_tensor_func)


device = "gpu:1"
paddle.set_device(device)
static_fig = True


def get_pretrained_model(cfg_path, model_path):
    logger.info(f"from {cfg_path} loading config")
    from omegaconf import OmegaConf

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
            for grid in tqdm(grid_coord.split(grid_batch_size, dim=1)):
                preds.append(model(g.x, g.pos, grid, g.batch, infos).squeeze(0))
            preds = paddle.concat(preds, axis=0)

        # 计算损失和MAE
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


if __name__ == "__main__":
    set_random_seed(42)

    output_dir = "./results"
    os.makedirs(output_dir, exist_ok=True)

    mol_name = (
        "ethanol"  # : benzene, ethanol, phenol, resorcinol, ethane, malonaldehyde
    )
    file_id = 1

    logger.info(f"Loading {mol_name} dataset")
    dataset = SmallDensityDataset(
        root="/home/zhoujingwen03/InfGCN-pytorch/data/md",
        mol_name=mol_name,
        split="test",
    )
    # dataset = DensityDataset(
    #    "data/QM9", "test", "data_split.json", "./atom_info/qm9.json", "CHGCAR", "lz4"
    # )        file_id = 24492  # indole
    # file_id = 114514  # nonane
    # file_id = 214  # benzene
    # file_id = 2  # ammonia
    # with lz4.frame.open(f"data/QM9/{file_id:06d}.CHGCAR.lz4") as f:
    #    g, density, grid_coord, info = dataset.read_chgcar(f)
    """
    CUBIC_ROOT = "./data/cubic"
    SPLIT_FILE = "./configs/data_split.json"
    ATOM_FILE = "./atom_info/crystal.json"

    dataset = DensityDataset(
        root=CUBIC_ROOT,
        split="test",
        split_file=SPLIT_FILE,
        atom_file=ATOM_FILE,
        extension="json",
        compression="xz"
    )

    material_id = "mp-18062"
    file_path = f"{CUBIC_ROOT}/{material_id}.json.xz"
    import lzma
    with lzma.open(file_path) as f:
        g, density, grid_coord, info = dataset.read_json(f)
    """
    g, density, grid_coord, info = dataset[file_id]

    g.batch = paddle.zeros_like(x=g.x)

    g = g.to(device)
    density = density.to(device)
    grid_coord = grid_coord.to(device)

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

    true_density_path = os.path.join(output_dir, f"{mol_name}_true_density.png")
    fig.write_image(true_density_path)
    logger.info(f"DFT electron density image saved to: {true_density_path}")

    if static_fig:
        img_bytes = fig.to_image(format="png", scale=2)
        display(Image(img_bytes))
    else:
        fig.show()

    logger.info("Loading the pretrained model")
    model = get_pretrained_model(
        "./electron_density_prediction/configs/md/infgcn_md.yaml",
        "./output/infgcn_md/step_4000.pdparams",
    )
    logger.info("Model loaded successfully!")

    logger.info("Starting prediction")
    grid_batch_size = 4096
    preds, loss, mae = inference_model(
        model, g, density, grid_coord[None], [info], grid_batch_size=grid_batch_size
    )
    logger.info(f"Predictiom completed, Loss: {float(loss):.6f}, MAE: {float(mae):.6f}")

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

    diff_density_path = os.path.join(output_dir, f"{mol_name}_diff_density.png")
    fig.write_image(diff_density_path)
    logger.info(f"Density difference image saved to: {diff_density_path}")

    if static_fig:
        img_bytes = fig.to_image(format="png", scale=2)
        display(Image(img_bytes))
    else:
        fig.show()

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

    pred_density_path = os.path.join(output_dir, f"{mol_name}_pred_density.png")
    fig.write_image(pred_density_path)
    logger.info(f"Predicted density image saved to: {pred_density_path}")

    if static_fig:
        img_bytes = fig.to_image(format="png", scale=2)
        display(Image(img_bytes))
    else:
        fig.show()
