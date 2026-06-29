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


import math

import plotly.graph_objects as go

from ppmat.utils import logger

try:
    from IPython.display import Image
    from IPython.display import display
except ImportError:  # Optional dependency; visualization still works for files
    Image, display = None, None


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
    Downsample a regular 3D grid for visualization to keep Plotly volume
    traces responsive.
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
            val.reshape(shape[0], shape[1], shape[2])[
                ::stride, ::stride, ::stride
            ].reshape(-1)
            for val in values
        ]
    except Exception as e:
        logger.warning(f"Failed to downsample grid for visualization: {e}")
        return grid, values, False, 1

    return grid_ds, values_ds, True, stride
