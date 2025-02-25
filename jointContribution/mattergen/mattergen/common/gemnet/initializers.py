import functools
import operator

import paddle

"""
Adapted from https://github.com/FAIR-Chem/fairchem/blob/main/src/fairchem/core/models/gemnet/initializers.py.
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found at https://github.com/FAIR-Chem/fairchem/blob/main/LICENSE.md.

"""


def _standardize(kernel):
    """
    Makes sure that N*Var(W) = 1 and E[W] = 0
    """
    eps = 1e-06
    if len(tuple(kernel.shape)) == 3:
        axis = 0, 1
    else:
        axis = 1
    var, mean = tuple(
        [
            paddle.var(kernel, axis=axis, unbiased=True, keepdim=True),
            paddle.mean(kernel, axis=axis, keepdim=True),
        ]
    )
    kernel = (kernel - mean) / (var + eps) ** 0.5
    return kernel


# def he_orthogonal_init(tensor: paddle.Tensor) -> paddle.Tensor:
#     """
#     Generate a weight matrix with variance according to He (Kaiming) initialization.
#     Based on a random (semi-)orthogonal matrix neural networks
#     are expected to learn better when features are decorrelated
#     (stated by eg. "Reducing overfitting in deep networks by decorrelating representations",
#     "Dropout: a simple way to prevent neural networks from overfitting",
#     "Exact solutions to the nonlinear dynamics of learning in deep linear neural networks")
#     """
#     init_Orthogonal = paddle.nn.initializer.Orthogonal()
#     tensor = init_Orthogonal(tensor)
#     if len(tuple(tensor.shape)) == 3:
#         fan_in = tuple(tensor.shape)[:-1].size
#     else:
#         fan_in = tuple(tensor.shape)[1]
#     with paddle.no_grad():
#         tensor.data = _standardize(tensor.data)
#         tensor.data *= (1 / fan_in) ** 0.5
#     return tensor


def he_orthogonal_init(tensor):
    """
    Generate a weight matrix with variance according to He initialization.
    Based on a random (semi-)orthogonal matrix neural networks
    are expected to learn better when features are decorrelated
    (stated by eg. "Reducing overfitting in deep networks by decorrelating
    representations",
    "Dropout: a simple way to prevent neural networks from overfitting",
    "Exact solutions to the nonlinear dynamics of learning in deep linear
    neural networks")
    """
    init_Orthogonal = paddle.nn.initializer.Orthogonal()
    init_Orthogonal(tensor)
    if len(tuple(tensor.shape)) == 3:
        fan_in = functools.reduce(operator.mul, tuple(tensor.shape)[:-1], 1)

    else:
        fan_in = tuple(tensor.shape)[0]
    stop_gradient = tensor.stop_gradient
    with paddle.no_grad():
        tensor.data = _standardize(tensor.data)
        tensor.data *= (1 / fan_in) ** 0.5
    tensor.stop_gradient = stop_gradient
    return tensor
