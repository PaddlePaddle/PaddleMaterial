import collections

import paddle

from ppmat.models.common.e3nn import o3
from ppmat.models.common.e3nn.math import germinate_formulas
from ppmat.models.common.e3nn.math import orthonormalize
from ppmat.models.common.e3nn.math import reduce_permutation
from ppmat.models.common.e3nn.util import explicit_default_types
from ppmat.models.common.e3nn.util.codegen import CodeGenMixin

_TP = collections.namedtuple("tp", "op, args")
_INPUT = collections.namedtuple("input", "tensor, start, stop")


def _wigner_nj(
    *irrepss, normalization="component", filter_ir_mid=None, dtype=None, device=None
):
    irrepss = [o3.Irreps(irreps) for irreps in irrepss]
    if filter_ir_mid is not None:
        filter_ir_mid = [o3.Irrep(ir) for ir in filter_ir_mid]

    if len(irrepss) == 1:
        (irreps,) = irrepss
        ret = []
        e = paddle.eye(irreps.dim, dtype=dtype)
        i = 0
        for mul, ir in irreps:
            for _ in range(mul):
                sl = slice(i, i + ir.dim)
                ret += [(ir, _INPUT(0, sl.start, sl.stop), e[sl])]
                i += ir.dim
        return ret

    *irrepss_left, irreps_right = irrepss
    ret = []
    for ir_left, path_left, C_left in _wigner_nj(
        *irrepss_left,
        normalization=normalization,
        filter_ir_mid=filter_ir_mid,
        dtype=dtype,
    ):
        i = 0
        for mul, ir in irreps_right:
            for ir_out in ir_left * ir:
                if filter_ir_mid is not None and ir_out not in filter_ir_mid:
                    continue

                C = o3.wigner_3j(ir_out.l, ir_left.l, ir.l, dtype=dtype)
                if normalization == "component":
                    C *= ir_out.dim**0.5
                if normalization == "norm":
                    C *= ir_left.dim**0.5 * ir.dim**0.5

                C = paddle.einsum(
                    "jk,ijl->ikl", C_left.reshape([-1, C_left.shape[-1]]), C
                )
                C = C.reshape(
                    [ir_out.dim] + [irreps.dim for irreps in irrepss_left] + [ir.dim]
                )

                for u in range(mul):
                    E = paddle.zeros(
                        [ir_out.dim]
                        + [irreps.dim for irreps in irrepss_left]
                        + [irreps_right.dim],
                        dtype=dtype,
                    )
                    sl = slice(i + u * ir.dim, i + (u + 1) * ir.dim)
                    E[..., sl] = C
                    ret += [
                        (
                            ir_out,
                            _TP(
                                op=(ir_left, ir, ir_out),
                                args=(
                                    path_left,
                                    _INPUT(len(irrepss_left), sl.start, sl.stop),
                                ),
                            ),
                            E,
                        )
                    ]
            i += mul * ir.dim

    return sorted(ret, key=lambda x: x[0])


def _get_ops(path):
    if isinstance(path, _INPUT):
        return
    assert isinstance(path, _TP)
    yield path.op
    for op in _get_ops(path.args[0]):
        yield op


class ReducedTensorProducts(CodeGenMixin, paddle.nn.Layer):
    def __init__(
        self, formula, filter_ir_out=None, filter_ir_mid=None, eps=1e-9, **irreps
    ):
        super().__init__()

        if filter_ir_out is not None:
            try:
                filter_ir_out = [o3.Irrep(ir) for ir in filter_ir_out]
            except ValueError:
                raise ValueError(
                    f"filter_ir_out (={filter_ir_out}) must be an iterable of e3nn.o3.Irrep"
                )

        if filter_ir_mid is not None:
            try:
                filter_ir_mid = [o3.Irrep(ir) for ir in filter_ir_mid]
            except ValueError:
                raise ValueError(
                    f"filter_ir_mid (={filter_ir_mid}) must be an iterable of e3nn.o3.Irrep"
                )

        f0, formulas = germinate_formulas(formula)

        irreps = {i: o3.Irreps(irs) for i, irs in irreps.items()}

        for i in irreps:
            if len(i) != 1:
                raise TypeError(f"got an unexpected keyword argument '{i}'")

        for _sign, p in formulas:
            f = "".join(f0[i] for i in p)
            for i, j in zip(f0, f):
                if i in irreps and j in irreps and irreps[i] != irreps[j]:
                    raise RuntimeError(f"irreps of {i} and {j} should be the same")
                if i in irreps:
                    irreps[j] = irreps[i]
                if j in irreps:
                    irreps[i] = irreps[j]

        for i in f0:
            if i not in irreps:
                raise RuntimeError(f"index {i} has no irreps associated to it")

        for i in irreps:
            if i not in f0:
                raise RuntimeError(
                    f"index {i} has an irreps but does not appear in the fomula"
                )

        base_perm, _ = reduce_permutation(
            f0, formulas, dtype="float64", **{i: irs.dim for i, irs in irreps.items()}
        )

        Ps = collections.defaultdict(list)

        for ir, path, base_o3 in _wigner_nj(
            *[irreps[i] for i in f0], filter_ir_mid=filter_ir_mid, dtype="float64"
        ):
            if filter_ir_out is None or ir in filter_ir_out:
                Ps[ir].append((path, base_o3))

        outputs = []
        change_of_basis = []
        irreps_out = []

        P = base_perm.reshape([base_perm.shape[0], -1])
        PP = paddle.matmul(P, P.t())

        for ir in Ps:
            mul = len(Ps[ir])
            paths = [path for path, _ in Ps[ir]]
            base_o3 = paddle.stack([R for _, R in Ps[ir]])

            R = base_o3.reshape([base_o3.shape[0], ir.dim, -1])

            proj_s = []
            for j in range(ir.dim):
                RR = paddle.matmul(R[:, j], R[:, j].t())
                RP = paddle.matmul(R[:, j], P.t())

                prob = paddle.concat(
                    [
                        paddle.concat([RR, -RP], axis=1),
                        paddle.concat([-RP.t(), PP], axis=1),
                    ],
                    axis=0,
                )

                eigenvalues, eigenvectors = paddle.linalg.eigh(prob)
                X = eigenvectors[:, eigenvalues < eps][:mul].t()
                proj_s.append(paddle.matmul(X.t(), X))

                break

            for p in proj_s:
                assert (
                    paddle.max(paddle.abs(p - proj_s[0])) < eps
                ), f"found different solutions for irrep {ir}"

            X, _ = orthonormalize(proj_s[0], eps)

            for x in X:
                C = paddle.einsum("u,ui...->i...", x, base_o3)
                correction = (ir.dim / paddle.sum(C.pow(2))) ** 0.5
                C = correction * C

                outputs.append(
                    [
                        ((correction * v).item(), p)
                        for v, p in zip(x, paths)
                        if abs(v) > eps
                    ]
                )
                change_of_basis.append(C)
                irreps_out.append((1, ir))

        dtype, _ = explicit_default_types(None, None)
        self.change_of_basis = paddle.concat(change_of_basis).astype(dtype)

        tps = set()
        for vp_list in outputs:
            for v, p in vp_list:
                for op in _get_ops(p):
                    tps.add(op)

        self.outputs = outputs
        self.tps = list(tps)

        self.tensor_products = paddle.nn.LayerDict()
        for i, op in enumerate(self.tps):
            tp = o3.TensorProduct(op[0], op[1], op[2], [(0, 0, 0, "uuu", False)])
            self.tensor_products[f"tp{i}"] = tp

        self.irreps_in = [irreps[i] for i in f0]
        self.irreps_out = o3.Irreps(irreps_out).simplify()
        self.f0 = f0
        self.eps = eps

    def forward(self, *xs):
        values = dict()

        def evaluate(path):
            if path in values:
                return values[path]

            if isinstance(path, _INPUT):
                out = xs[path.tensor]
                if (path.start, path.stop) != (0, self.irreps_in[path.tensor].dim):
                    out = paddle.slice(out, [-1], [path.start], [path.stop])
            if isinstance(path, _TP):
                x1 = evaluate(path.args[0])
                x2 = evaluate(path.args[1])
                tp_idx = self.tps.index(path.op)
                out = self.tensor_products[f"tp{tp_idx}"](x1, x2)
            values[path] = out
            return out

        outs = []
        for vp_list in self.outputs:
            v, p = vp_list[0]
            out = evaluate(p)
            if abs(v - 1.0) > self.eps:
                out = v * out
            for v, p in vp_list[1:]:
                t = evaluate(p)
                if abs(v - 1.0) > self.eps:
                    t = v * t
                out = out + t
            outs.append(out)

        return paddle.concat(outs, axis=-1)
