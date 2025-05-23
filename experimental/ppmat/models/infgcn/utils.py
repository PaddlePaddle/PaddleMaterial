import paddle


class BroadcastGTOTensor(paddle.nn.Layer):
    """
    Broadcast between spherical tensors of the Gaussian Type Orbitals (GTOs):

    .. math::
        \\{a_{clm}, 1\\le c\\le c_{max}, 0\\le\\ell\\le\\ell_{max}, -\\ell\\le m\\le\\ell\\}

    For efficiency reason, the feature tensor is indexed by l, c, m.
    For example, for lmax = 3, cmax = 2, we have a tensor of 1s2s 1p2p 1d2d 1f2f.
    Currently, we support the following broadcasting:
    lc -> lcm;
    m -> lcm.
    """

    def __init__(self, lmax, cmax, src="lc", dst="lcm"):
        super(BroadcastGTOTensor, self).__init__()
        assert src in ["lc", "m"]
        assert dst in ["lcm"]
        self.src = src
        self.dst = dst
        self.lmax = lmax
        self.cmax = cmax
        if src == "lc":
            self.src_dim = (lmax + 1) * cmax
        else:
            self.src_dim = (lmax + 1) ** 2
        self.dst_dim = (lmax + 1) ** 2 * cmax
        if src == "lc":
            indices = self._generate_lc2lcm_indices()
        else:
            indices = self._generate_m2lcm_indices()
        self.register_buffer(name="indices", tensor=indices)

    def _generate_lc2lcm_indices(self):
        """
        lc -> lcm
        .. math::
            1s2s 1p2p → 1s2s 1p_x1p_y1p_z2p_x2p_y2p_z
        [0, 1, 2, 2, 2, 3, 3, 3]

        :return: (lmax+1)^2 * cmax
        """
        indices = [
            (l * self.cmax + c)
            for l in range(self.lmax + 1)
            for c in range(self.cmax)
            for _ in range(2 * l + 1)
        ]
        return paddle.to_tensor(data=indices, dtype="int64")

    def _generate_m2lcm_indices(self):
        """
        m -> lcm
        .. math::
            s p_x p_y p_z → 1s2s 1p_x1p_y1p_z2p_x2p_y2p_z
        [0, 0, 1, 2, 3, 1, 2, 3]

        :return: (lmax+1)^2 * cmax
        """
        indices = [
            (l * l + m)
            for l in range(self.lmax + 1)
            for _ in range(self.cmax)
            for m in range(2 * l + 1)
        ]
        return paddle.to_tensor(data=indices, dtype="int64")

    def forward(self, x):
        """
        Apply broadcasting to x.
        :param x: (..., src_dim)
        :return: (..., dst_dim)
        """
        assert (
            x.shape[-1] == self.src_dim
        ), f"Input dimension mismatch! Should be {self.src_dim}, but got {x.shape[-1]} instead!"
        if self.src == self.dst:
            return x
        return x[..., self.indices]
