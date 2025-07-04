import numpy as np
import paddle

from .base_layers import Dense


class AtomEmbedding(paddle.nn.Layer):
    """
    Initial atom embeddings based on the atom type

    Parameters
    ----------
        emb_size: int
            Atom embeddings size
    """

    def __init__(self, num_embeddings, emb_size):
        super().__init__()

        self.emb_size = emb_size
        self.embeddings = paddle.nn.Embedding(
            num_embeddings=num_embeddings, embedding_dim=emb_size
        )
        init_Uniform = paddle.nn.initializer.Uniform(low=-np.sqrt(3), high=np.sqrt(3))
        init_Uniform(self.embeddings.weight)

    def forward(self, Z):
        """
        Returns
        -------
            h: paddle.Tensor, shape=(nAtoms, emb_size)
                Atom embeddings.
        """
        h = self.embeddings(Z)

        return h


class EdgeEmbedding(paddle.nn.Layer):
    """
    Edge embedding based on the concatenation of atom embeddings and subsequent
    dense layer.

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
            m_st: paddle.Tensor, shape=(nEdges, emb_size)
                Edge embeddings.
        """
        h_s = h[idx_s]
        h_t = h[idx_t]
        m_st = paddle.concat(x=[h_s, h_t, m_rbf], axis=-1)
        m_st = self.dense(m_st)
        return m_st
