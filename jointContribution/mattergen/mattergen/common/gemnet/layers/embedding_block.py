import sys

import paddle

from paddle_utils import *

"""
Adapted from https://github.com/FAIR-Chem/fairchem/blob/main/src/fairchem/core/models/gemnet/layers/embedding_block.py.
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found at https://github.com/FAIR-Chem/fairchem/blob/main/LICENSE.md.

"""
import numpy as np

from mattergen.common.gemnet.layers.base_layers import Dense
from mattergen.common.utils.globals import MAX_ATOMIC_NUM


class IdentityEmbedding(paddle.nn.Identity):
    """Embedding layer that just returns the input"""

    def __init__(self, emb_size):
        super().__init__()
        self.emb_size = emb_size


class AtomEmbedding(paddle.nn.Layer):
    """
    Initial atom embeddings based on the atom type

    Parameters
    ----------
        emb_size: int
            Atom embeddings size
    """

    def __init__(self, emb_size, with_mask_type=False):
        super().__init__()
        self.emb_size = emb_size
        self.embeddings = paddle.nn.Embedding(
            num_embeddings=MAX_ATOMIC_NUM + int(with_mask_type), embedding_dim=emb_size
        )
        init_Uniform = paddle.nn.initializer.Uniform(low=-np.sqrt(3), high=np.sqrt(3))
        init_Uniform(self.embeddings.weight)

    def forward(self, Z):
        """
        Returns
        -------
            h: torch.Tensor, shape=(nAtoms, emb_size)
                Atom embeddings.
        """
        h = self.embeddings(Z - 1)
        return h


class EdgeEmbedding(paddle.nn.Layer):
    """
    Edge embedding based on the concatenation of atom embeddings and subsequent dense layer.

    Parameters
    ----------
        emb_size: int
            Embedding size after the dense layer.
        activation: str
            Activation function used in the dense layer.
    """

    def __init__(self, atom_features, edge_features, out_features, activation=None):
        super().__init__()
        in_features = 2 * atom_features + edge_features
        self.dense = Dense(in_features, out_features, activation=activation, bias=False)

    def forward(self, h, m_rbf, idx_s, idx_t):
        """

        Arguments
        ---------
        h
        m_rbf: shape (nEdges, nFeatures)
            in embedding block: m_rbf = rbf ; In interaction block: m_rbf = m_st
        idx_s
        idx_t

        Returns
        -------
            m_st: torch.Tensor, shape=(nEdges, emb_size)
                Edge embeddings.
        """
        h_s = h[idx_s]
        h_t = h[idx_t]
        m_st = paddle.concat(x=[h_s, h_t, m_rbf], axis=-1)
        m_st = self.dense(m_st)
        return m_st
