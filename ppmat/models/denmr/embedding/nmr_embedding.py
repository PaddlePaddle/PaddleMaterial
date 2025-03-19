import paddle
import paddle.nn as nn


class RBFEncoder(nn.Layer):
    def __init__(self, min, max, bins):
        super(RBFEncoder, self).__init__()
        self.centers = self.create_parameter(
            shape=[bins],
            default_initializer=nn.initializer.Assign(paddle.linspace(min, max, bins)),
        )
        self.centers.stop_gradient = True
        self.sigma = (max - min) / (bins - 1)  # adaptive bandwidth

    def forward(self, x):
        # x: (...,)
        diff = x.unsqueeze(-1) - self.centers  # (..., bins)
        return paddle.exp(-0.5 * (diff / self.sigma).pow(2))


class RBFEncoder_Jcouple(nn.Layer):
    def __init__(self, min1=0, max1=26, bins1=131, min2=27, max2=58, bins2=32):
        super(RBFEncoder_Jcouple, self).__init__()

        centers1 = paddle.linspace(min1, max1, bins1)
        sigma1 = (max1 - min1) / (bins1 - 1)  # 20/99 ≈ 0.202

        centers2 = paddle.linspace(min2, max2, bins2)
        sigma2 = (max2 - min2) / (bins2 - 1)  # 30/29 ≈ 1.034

        # 合并参数
        self.centers = self.create_parameter(
            shape=[bins1 + bins2],
            default_initializer=nn.initializer.Assign(
                paddle.concat([centers1, centers2])
            ),
        )
        self.centers.stop_gradient = True
        self.sigma = self.create_parameter(
            shape=[bins1 + bins2],
            default_initializer=nn.initializer.Assign(
                paddle.concat(
                    [paddle.full([bins1], sigma1), paddle.full([bins2], sigma2)]
                )
            ),
        )
        self.sigma.stop_gradient = True

    def forward(self, x):
        diff = x.unsqueeze(-1) - self.centers  # (..., 130)
        return paddle.exp(-0.5 * (diff / self.sigma).pow(2))


class H1nmr_embedding(nn.Layer):
    def __init__(
        self,
        split_dim=64,
        peakwidth_dim=40,
        integral_dim=32,
        H_shift_min=-1,
        H_shift_max=10,
        H_shift_bin=111,
        min_j=0,
        max_j=58,
        j_bins1=131,
        j_bins2=32,
        hidden=1024,
        dim=1024,
        drop_prob=0.1,
    ):
        super(H1nmr_embedding, self).__init__()

        self.shift_emb = RBFEncoder(
            min=H_shift_min, max=H_shift_max, bins=H_shift_bin
        )  # Covering common 1H ranges

        self.peakwidth_emb = nn.Embedding(70, peakwidth_dim, padding_idx=0)

        self.split_emb = nn.Embedding(
            116, split_dim, padding_idx=0
        )  # Supports 116 split patterns

        self.integral_emb = nn.Embedding(26, integral_dim, padding_idx=0)

        self.J_emb = RBFEncoder_Jcouple(
            min1=min_j, max1=26, bins1=j_bins1, min2=27, max2=max_j, bins2=j_bins2
        )

        self.d_model = (
            split_dim + peakwidth_dim + integral_dim + H_shift_bin + j_bins1 + j_bins2
        )

        self.peak_fuser = peak_fuser(self.d_model, dim, drop_prob)

    def forward(self, h1nmr, src_mask):

        hnmr = h1nmr

        h_shift, peakwidth, split, integral, j_couple = (
            hnmr[:, :, 0],
            hnmr[:, :, 1],
            hnmr[:, :, 2],
            hnmr[:, :, 3],
            hnmr[:, :, 4:],
        )

        h_shift_emb = self.shift_emb(h_shift) * src_mask.unsqueeze(-1)
        peakwidth_emb = self.peakwidth_emb(peakwidth.astype("int64"))
        split_emb = self.split_emb(split.astype("int64"))
        integral_emb = self.integral_emb((integral + 1).astype("int64"))

        J_emb = self.J_emb(j_couple)
        J_emb = paddle.sum(J_emb, axis=-2) * src_mask.unsqueeze(-1)

        hnmr_emb = paddle.concat(
            [h_shift_emb, peakwidth_emb, split_emb, integral_emb, J_emb], axis=-1
        )
        hnmr_emb = self.peak_fuser(hnmr_emb)

        return hnmr_emb


class C13nmr_embedding(nn.Layer):
    def __init__(
        self,
        C_shift_min=-15,
        C_shift_max=229,
        C_bins=245,
        hidden=512,
        dim=256,
        drop_prob=0.1,
    ):
        super(C13nmr_embedding, self).__init__()

        self.shift_emb = RBFEncoder(min=C_shift_min, max=C_shift_max, bins=C_bins)

        self.peak_fuser = peak_fuser(C_bins, dim, drop_prob)

    def forward(self, c13nmr, src_mask):

        cnmr = c13nmr

        c_shift_emb = self.shift_emb(cnmr) * src_mask.unsqueeze(-1)

        cnmr_emb = self.peak_fuser(c_shift_emb)

        return cnmr_emb


class peak_fuser(nn.Layer):
    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(peak_fuser, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(), nn.Dropout(drop_prob)
        )

    def forward(self, x):
        return self.net(x)
