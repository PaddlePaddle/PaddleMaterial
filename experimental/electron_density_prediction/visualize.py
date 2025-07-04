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

import io

import paddle
import PIL
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

plt.switch_backend("agg")
cmap = ListedColormap(["grey", "white", "red", "blue", "green", "white"])


def draw_stack(density, atom_type=None, atom_coord=None, dim=-1):
    """
    Draw a 2D density map along specific axis.
    :param density: density data, tensor of shape (batch_size, nx, ny, nz)
    :param atom_type: atom types, tensor of shape (batch_size, n_atom)
    :param atom_coord: atom coordinates, tensor of shape (batch_size, n_atom, 3)
    :param dim: axis along which to sum
    :return: an image tensor
    """
    plt.figure(figsize=(3, 3))
    plt.imshow(density.sum(axis=dim).detach().cpu().numpy(), cmap="viridis")
    plt.colorbar()
    if atom_type is not None:
        idx = [i for i in range(3) if i != dim % 3]
        coord = atom_coord.detach().cpu().numpy()
        color = cmap(atom_type.detach().cpu().numpy())
        plt.scatter(coord[:, idx[1]], coord[:, idx[0]], c=color, alpha=0.8)
    buf = io.BytesIO()
    plt.savefig(buf, format="jpg")
    buf.seek(0)
    image = PIL.Image.open(buf)
    image = paddle.vision.transforms.ToTensor()(
        image
    )  # 这个是paconvert自动改的，应该是准确的把，之前是torchvision.transforms.ToTensor()
    image = image.transpose([1, 2, 0])  # add in 0319
    plt.close()
    return image.numpy()  # modified in 0319
