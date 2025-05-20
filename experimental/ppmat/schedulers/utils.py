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


from typing import Optional

import paddle

from ppmat.utils.paddle_aux import dim2perm


def make_noise_symmetric_preserve_variance(noise: paddle.Tensor) -> paddle.Tensor:
    """Makes the noise matrix symmetric, preserving the variance. Assumes i.i.d. noise
    for each dimension.

    Args:
        noise (paddle.Tensor): Input noise matrix, must be a batched square matrix,
            i.e., have shape (batch_size, dim, dim).

    Returns:
        paddle.Tensor: The symmetric noise matrix, with the same variance as the input.
    """
    assert (
        len(tuple(noise.shape)) == 3 and tuple(noise.shape)[1] == tuple(noise.shape)[2]
    ), "Symmetric noise only works for square-matrix-shaped data."
    return (
        1
        / 2**0.5
        * (1 - paddle.eye(num_rows=3)[None])
        * (noise + noise.transpose(perm=dim2perm(noise.ndim, 1, 2)))
        + paddle.eye(num_rows=3)[None] * noise
    )


def expand(a, x_shape, left=False):
    a_dim = len(tuple(a.shape))
    if left:
        return a.reshape(*((1,) * (len(x_shape) - a_dim) + tuple(a.shape)))
    else:
        return a.reshape(*(tuple(a.shape) + (1,) * (len(x_shape) - a_dim)))


def _broadcast_like(x, like):
    """
    add broadcast dimensions to x so that it can be broadcast over ``like``
    """
    if like is None:
        return x
    return x[(...,) + (None,) * (like.ndim - x.ndim)]


def maybe_expand(
    x: paddle.Tensor, batch: Optional[paddle.Tensor], like: paddle.Tensor = None
) -> paddle.Tensor:
    """

    Args:
        x: shape (batch_size, ...)
        batch: shape (num_thingies,) with integer entries in the range [0, batch_size),
            indicating which sample each thingy belongs to
        like: shape x.shape + potential additional dimensions
    Returns:
        expanded x with shape (num_thingies,), or if given like.shape, containing value
             of x for each thingy.
        If `batch` is None, just returns `x` unmodified, to avoid pointless work if you
        have exactly one thingy per sample.
    """
    x = _broadcast_like(x, like)
    if batch is None:
        return x
    else:
        return x[batch]
