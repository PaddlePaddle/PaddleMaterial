import sys

import paddle

from paddle_utils import *

"""
Adapted from https://github.com/FAIR-Chem/fairchem/blob/main/src/fairchem/core/models/gemnet/layers/efficient.py.
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found at https://github.com/FAIR-Chem/fairchem/blob/main/LICENSE.md.

"""
from warnings import warn

from mattergen.common.gemnet.initializers import he_orthogonal_init


class EfficientInteractionDownProjection(paddle.nn.Layer):
    """
    Down projection in the efficient reformulation.

    Parameters
    ----------
        emb_size_interm: int
            Intermediate embedding size (down-projection size).
        kernel_initializer: callable
            Initializer of the weight matrix.
    """

    def __init__(self, num_spherical: int, num_radial: int, emb_size_interm: int):
        super().__init__()
        self.num_spherical = num_spherical
        self.num_radial = num_radial
        self.emb_size_interm = emb_size_interm
        self.reset_parameters()

    def reset_parameters(self):
        self.weight = paddle.base.framework.EagerParamBase.from_tensor(
            tensor=paddle.empty(shape=(self.num_spherical, self.num_radial, self.emb_size_interm)),
            trainable=True,
        )
        he_orthogonal_init(self.weight)

    def forward(self, rbf, sph, id_ca, id_ragged_idx):
        """

        Arguments
        ---------
        rbf: paddle.Tensor, shape=(1, nEdges, num_radial)
        sph: paddle.Tensor, shape=(nEdges, Kmax, num_spherical)
        id_ca
        id_ragged_idx

        Returns
        -------
        rbf_W1: paddle.Tensor, shape=(nEdges, emb_size_interm, num_spherical)
        sph: paddle.Tensor, shape=(nEdges, Kmax, num_spherical)
            Kmax = maximum number of neighbors of the edges
        """
        num_edges = tuple(rbf.shape)[1]
        rbf_W1 = paddle.matmul(x=rbf, y=self.weight)
        rbf_W1 = rbf_W1.transpose(perm=[1, 2, 0])
        if tuple(sph.shape)[0] == 0:
            Kmax = 0
        else:
            Kmax = max(
                paddle.max(x=id_ragged_idx + 1),
                paddle.to_tensor(data=0).to(id_ragged_idx.place),
            )
        sph2 = paddle.zeros(shape=[num_edges, Kmax, self.num_spherical], dtype=sph.dtype)
        sph2[id_ca, id_ragged_idx] = sph
        sph2 = paddle.transpose(x=sph2, perm=dim2perm(sph2.ndim, 1, 2))
        return rbf_W1, sph2


class EfficientInteractionBilinear(paddle.nn.Layer):
    """
    Efficient reformulation of the bilinear layer and subsequent summation.

    Parameters
    ----------
        units_out: int
            Embedding output size of the bilinear layer.
        kernel_initializer: callable
            Initializer of the weight matrix.
    """

    def __init__(self, emb_size: int, emb_size_interm: int, units_out: int):
        super().__init__()
        self.emb_size = emb_size
        self.emb_size_interm = emb_size_interm
        self.units_out = units_out
        self.reset_parameters()

    def reset_parameters(self):
        out_0 = paddle.empty(shape=(self.emb_size, self.emb_size_interm, self.units_out))
        out_0.stop_gradient = not True
        self.weight = paddle.base.framework.EagerParamBase.from_tensor(tensor=out_0)
        he_orthogonal_init(self.weight)

    def forward(self, basis, m, id_reduce, id_ragged_idx):
        """

        Arguments
        ---------
        basis
        m: quadruplets: m = m_db , triplets: m = m_ba
        id_reduce
        id_ragged_idx

        Returns
        -------
            m_ca: paddle.Tensor, shape=(nEdges, units_out)
                Edge embeddings.
        """
        rbf_W1, sph = basis
        nEdges = tuple(rbf_W1.shape)[0]
        if nEdges == 0:
            warn(f"Zero graph edges found in {self.__class__}")
            return paddle.zeros(shape=(0, 0))
        Kmax = max(
            paddle.max(x=id_ragged_idx) + 1,
            paddle.to_tensor(data=0).to(id_ragged_idx.place),
        )
        m2 = paddle.zeros(shape=[nEdges, Kmax, self.emb_size], dtype=m.dtype)
        m2[id_reduce, id_ragged_idx] = m
        sum_k = paddle.matmul(x=sph, y=m2)
        rbf_W1_sum_k = paddle.matmul(x=rbf_W1, y=sum_k)
        m_ca = paddle.matmul(x=rbf_W1_sum_k.transpose(perm=[2, 0, 1]), y=self.weight)
        m_ca = paddle.sum(x=m_ca, axis=0)
        return m_ca
