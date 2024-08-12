import sys

import paddle

from ..initializers import he_orthogonal_init

sys.path.append("/home/chenxiaoxu02/workspaces/gemnet_paddle/utils")


class EfficientInteractionDownProjection(paddle.nn.Layer):
    """
    Down projection in the efficient reformulation.

    Parameters
    ----------
        num_spherical: int
            Same as the setting in the basis layers.
        num_radial: int
            Same as the setting in the basis layers.
        emb_size_interm: int
            Intermediate embedding size (down-projection size).
    """

    def __init__(
        self,
        num_spherical: int,
        num_radial: int,
        emb_size_interm: int,
        name="EfficientDownProj",
    ):
        super().__init__()
        self.num_spherical = num_spherical
        self.num_radial = num_radial
        self.emb_size_interm = emb_size_interm
        self.reset_parameters()

    def reset_parameters(self):
        out_2 = paddle.create_parameter(
            shape=paddle.empty(
                shape=(self.num_spherical, self.num_radial, self.emb_size_interm)
            ).shape,
            dtype=paddle.empty(
                shape=(self.num_spherical, self.num_radial, self.emb_size_interm)
            )
            .numpy()
            .dtype,
            default_initializer=paddle.nn.initializer.Assign(
                paddle.empty(
                    shape=(self.num_spherical, self.num_radial, self.emb_size_interm)
                )
            ),
        )
        out_2.stop_gradient = not True
        self.weight = out_2
        he_orthogonal_init(self.weight)

    def forward(self, tbf):
        """
        Returns
        -------
            (rbf_W1, sph): tuple
            - rbf_W1: Tensor, shape=(nEdges, emb_size_interm, num_spherical)
            - sph: Tensor, shape=(nEdges, Kmax, num_spherical)
        """
        rbf_env, sph = tbf
        rbf_W1 = paddle.matmul(x=rbf_env, y=self.weight)
        rbf_W1 = rbf_W1.transpose(perm=[1, 2, 0])
        x = sph
        perm_2 = list(range(x.ndim))
        perm_2[1] = 2
        perm_2[2] = 1
        sph = paddle.transpose(x=x, perm=perm_2)
        return rbf_W1, sph


class EfficientInteractionHadamard(paddle.nn.Layer):
    """
    Efficient reformulation of the hadamard product and subsequent summation.

    Parameters
    ----------
        emb_size_interm: int
            Intermediate embedding size (down-projection size).
        emb_size: int
            Embedding size.
    """

    def __init__(self, emb_size_interm: int, emb_size: int, name="EfficientHadamard"):
        super().__init__()
        self.emb_size_interm = emb_size_interm
        self.emb_size = emb_size
        self.reset_parameters()

    def reset_parameters(self):
        out_3 = paddle.empty(shape=(self.emb_size, 1, self.emb_size_interm))
        out_3.stop_gradient = not True
        out_4 = paddle.create_parameter(
            shape=out_3.shape,
            dtype=out_3.numpy().dtype,
            default_initializer=paddle.nn.initializer.Assign(out_3),
        )
        out_4.stop_gradient = not True
        self.weight = out_4
        he_orthogonal_init(self.weight)

    def forward(self, basis, m, id_reduce, Kidx):
        """
        Returns
        -------
            m_ca: Tensor, shape=(nEdges, emb_size)
                Edge embeddings.
        """
        rbf_W1, sph = basis
        nEdges = tuple(rbf_W1.shape)[0]
        if tuple(sph.shape)[2] == 0:
            Kmax = 0
        else:
            Kmax = paddle.maximum(paddle.max(x=Kidx + 1), paddle.to_tensor(data=0))
        m2 = paddle.zeros(shape=[nEdges, Kmax, self.emb_size], dtype=m.dtype)
        m2[id_reduce, Kidx] = m
        sum_k = paddle.matmul(x=sph, y=m2)
        rbf_W1_sum_k = paddle.matmul(x=rbf_W1, y=sum_k)
        m_ca = paddle.matmul(x=self.weight, y=rbf_W1_sum_k.transpose(perm=[2, 1, 0]))[
            :, 0
        ]
        x = m_ca
        perm_3 = list(range(x.ndim))
        perm_3[0] = 1
        perm_3[1] = 0
        m_ca = paddle.transpose(x=x, perm=perm_3)
        return m_ca


class EfficientInteractionBilinear(paddle.nn.Layer):
    """
    Efficient reformulation of the bilinear layer and subsequent summation.

    Parameters
    ----------
        emb_size: int
            Edge embedding size.
        emb_size_interm: int
            Intermediate embedding size (down-projection size).
        units_out: int
            Embedding output size of the bilinear layer.
        kernel_initializer: callable
            Initializer of the weight matrix.
    """

    def __init__(
        self,
        emb_size: int,
        emb_size_interm: int,
        units_out: int,
        name="EfficientBilinear",
    ):
        super().__init__()
        self.emb_size = emb_size
        self.emb_size_interm = emb_size_interm
        self.units_out = units_out
        self.reset_parameters()

    def reset_parameters(self):
        out_5 = paddle.empty(
            shape=(self.emb_size, self.emb_size_interm, self.units_out)
        )
        out_5.stop_gradient = not True
        out_6 = paddle.create_parameter(
            shape=out_5.shape,
            dtype=out_5.numpy().dtype,
            default_initializer=paddle.nn.initializer.Assign(out_5),
        )
        out_6.stop_gradient = not True
        self.weight = out_6
        he_orthogonal_init(self.weight)

    def forward(self, basis, m, id_reduce, Kidx):
        """
        Returns
        -------
            m_ca: Tensor, shape=(nEdges, units_out)
                Edge embeddings.
        """
        rbf_W1, sph = basis
        nEdges = tuple(rbf_W1.shape)[0]
        Kmax = (
            0
            if tuple(sph.shape)[2] == 0
            else paddle.maximum(paddle.max(x=Kidx + 1), paddle.to_tensor(data=0))
        )
        m2 = paddle.zeros(shape=[nEdges, Kmax, self.emb_size], dtype=m.dtype)
        m2[id_reduce, Kidx] = m
        sum_k = paddle.matmul(x=sph, y=m2)
        rbf_W1_sum_k = paddle.matmul(x=rbf_W1, y=sum_k)
        m_ca = paddle.matmul(x=rbf_W1_sum_k.transpose(perm=[2, 0, 1]), y=self.weight)
        m_ca = paddle.sum(x=m_ca, axis=0)
        return m_ca
