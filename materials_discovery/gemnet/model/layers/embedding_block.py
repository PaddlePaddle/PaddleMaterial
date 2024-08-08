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

    def __init__(self, emb_size, name=None):
        super().__init__()
        self.emb_size = emb_size
        self.embeddings = paddle.nn.Embedding(num_embeddings=93, embedding_dim=emb_size)
        init_Uniform = paddle.nn.initializer.Uniform(low=-np.sqrt(3), high=np.sqrt(3))
        init_Uniform(self.embeddings.weight)

    def forward(self, Z):
        """
        Returns
        -------
            h: Tensor, shape=(nAtoms, emb_size)
                Atom embeddings.
        """
        h = self.embeddings(Z - 1)
        return h


class EdgeEmbedding(paddle.nn.Layer):
    """
    Edge embedding based on the concatenation of atom embeddings and subsequent dense layer.

    Parameters
    ----------
        atom_features: int
            Embedding size of the atom embeddings.
        edge_features: int
            Embedding size of the edge embeddings.
        out_features: int
            Embedding size after the dense layer.
        activation: str
            Activation function used in the dense layer.
    """

    def __init__(
        self, atom_features, edge_features, out_features, activation=None, name=None
    ):
        super().__init__()
        in_features = 2 * atom_features + edge_features
        self.dense = Dense(in_features, out_features, activation=activation, bias=False)

    def forward(self, h, m_rbf, idnb_a, idnb_c):
        """
        Returns
        -------
            m_ca: Tensor, shape=(nEdges, emb_size)
                Edge embeddings.
        """
        h_a = h[idnb_a]
        h_c = h[idnb_c]
        m_ca = paddle.concat(x=[h_a, h_c, m_rbf], axis=-1)
        m_ca = self.dense(m_ca)
        return m_ca
