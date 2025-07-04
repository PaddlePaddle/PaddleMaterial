import paddle

from ppmat.models.common.e3nn import o3


class Extract(paddle.nn.Layer):
    def __init__(self, irreps_in, irreps_outs, instructions, squeeze_out: bool = False):
        r"""Extract sub sets of irreps

        Parameters
        ----------
        irreps_in : `e3nn.o3.Irreps`
            representation of the input

        irreps_outs : list of `e3nn.o3.Irreps`
            list of representation of the outputs

        instructions : list of tuple of int
            list of tuples, one per output continaing each ``len(irreps_outs[i])`` int

        squeeze_out : bool, default False
            if ``squeeze_out`` and only one output exists, a ``paddle.Tensor`` will be returned instead of a
            ``Tuple[paddle.Tensor]``
        """
        super().__init__()
        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_outs = tuple(o3.Irreps(irreps) for irreps in irreps_outs)
        self.instructions = instructions
        self.squeeze_out = squeeze_out

        assert len(self.irreps_outs) == len(self.instructions)
        for irreps_out, ins in zip(self.irreps_outs, self.instructions):
            assert len(irreps_out) == len(ins)

    def forward(self, x):
        assert x.shape[-1] == self.irreps_in.dim, "invalid input shape"

        out = []
        for irreps in self.irreps_outs:
            out.append(paddle.zeros(list(x.shape[:-1]) + [irreps.dim], dtype=x.dtype))
        for i, (irreps_out, ins) in enumerate(zip(self.irreps_outs, self.instructions)):
            if ins == tuple(range(len(self.irreps_in))):
                out[i] = paddle.assign(x)
            else:
                for s_out, i_in in zip(irreps_out.slices(), ins):
                    i_start = self.irreps_in[:i_in].dim
                    i_len = self.irreps_in[i_in].dim
                    x_slice = x.slice([-1], [i_start], [i_start + i_len])
                    if len(out[i].shape) == 1:
                        out[i][s_out.start : s_out.stop] = x_slice
                    else:
                        idx = [slice(None)] * (
                            len(out[i].shape) - 1
                        )  
                        idx.append(
                            slice(s_out.start, s_out.stop)
                        )  
                        out[i][tuple(idx)] = x_slice

        if self.squeeze_out and len(out) == 1:
            return out[0]
        return tuple(out)


class ExtractIr(Extract):
    def __init__(self, irreps_in, ir):
        r"""Extract ``ir`` from irreps

        Parameters
        ----------
        irreps_in : `e3nn.o3.Irreps`
            representation of the input

        ir : `e3nn.o3.Irrep`
            representation to extract
        """
        ir = o3.Irrep(ir)
        irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps([mul_ir for mul_ir in irreps_in if mul_ir.ir == ir])
        instructions = [
            tuple(i for i, mul_ir in enumerate(irreps_in) if mul_ir.ir == ir)
        ]

        super().__init__(irreps_in, [self.irreps_out], instructions, squeeze_out=True)
