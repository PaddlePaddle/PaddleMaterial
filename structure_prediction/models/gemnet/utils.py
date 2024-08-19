import json
from typing import Optional

import paddle


def read_json(path):
    """ """
    if not path.endswith(".json"):
        raise UserWarning(f"Path {path} is not a json-path.")
    with open(path, "r") as f:
        content = json.load(f)
    return content


def update_json(path, data):
    """ """
    if not path.endswith(".json"):
        raise UserWarning(f"Path {path} is not a json-path.")
    content = read_json(path)
    content.update(data)
    write_json(path, content)


def write_json(path, data):
    """ """
    if not path.endswith(".json"):
        raise UserWarning(f"Path {path} is not a json-path.")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def read_value_json(path, key):
    """ """
    content = read_json(path)
    if key in content.keys():
        return content[key]
    else:
        return None


def _broadcast(src: paddle.Tensor, other: paddle.Tensor, dim: int):
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand(other.shape)
    return src


def _scatter_sum(
    src: paddle.Tensor,
    index: paddle.Tensor,
    dim: int = -1,
    out: Optional[paddle.Tensor] = None,
    dim_size: Optional[int] = None,
) -> paddle.Tensor:
    index = _broadcast(index, src, dim)
    if out is None:
        size = list(src.shape)
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = paddle.zeros(size, dtype=src.dtype)
    return paddle.put_along_axis(
        arr=out, indices=index, values=src, axis=dim, reduce="add"
    )


def _scatter_add(
    src: paddle.Tensor,
    index: paddle.Tensor,
    dim: int = -1,
    out: Optional[paddle.Tensor] = None,
    dim_size: Optional[int] = None,
) -> paddle.Tensor:
    return _scatter_sum(src, index, dim, out, dim_size)


def _scatter_mean(
    src: paddle.Tensor,
    index: paddle.Tensor,
    dim: int = -1,
    out: Optional[paddle.Tensor] = None,
    dim_size: Optional[int] = None,
) -> paddle.Tensor:
    out = _scatter_sum(src, index, dim, out, dim_size)
    dim_size = out.shape[dim]

    index_dim = dim
    if index_dim < 0:
        index_dim = index_dim + src.dim()
    if index.dim() <= index_dim:
        index_dim = index.dim() - 1

    ones = paddle.ones(index.shape, dtype=src.dtype)
    count = _scatter_sum(ones, index, index_dim, None, dim_size)
    count[count < 1] = 1
    count = _broadcast(count, out, dim)
    if out.is_floating_point():
        out = paddle.divide(out, count)
        # out.true_divide_(count)
    else:
        out = paddle.floor_divide(out, count)
        # out.div_(count, rounding_mode='floor')
    return out


def scatter(
    src: paddle.Tensor,
    index: paddle.Tensor,
    dim: int = -1,
    out: Optional[paddle.Tensor] = None,
    dim_size: Optional[int] = None,
    reduce: str = "sum",
) -> paddle.Tensor:
    """
    Implement paddle version API like torch_scatter.scatter
    """
    if reduce == "sum" or reduce == "add":
        return _scatter_sum(src, index, dim, out, dim_size)
    elif reduce == "mean":
        return _scatter_mean(src, index, dim, out, dim_size)
    else:
        raise ValueError("Only support add or mean")


def ragged_range(sizes):
    """Multiple concatenated ranges.

    Examples
    --------
        sizes = [1 4 2 3]
        Return: [0  0 1 2 3  0 1  0 1 2]
    """
    assert sizes.dim() == 1
    if sizes.sum() == 0:
        return paddle.empty(shape=[0], dtype=sizes.dtype)
    sizes_nonzero = sizes > 0
    if not paddle.all(x=sizes_nonzero):
        sizes = paddle.masked_select(x=sizes, mask=sizes_nonzero)
    id_steps = paddle.ones(shape=sizes.sum(), dtype="int64")
    id_steps[0] = 0
    insert_index = sizes[:-1].cumsum(axis=0)
    insert_val = (1 - sizes)[:-1]
    id_steps[insert_index] = insert_val
    res = id_steps.cumsum(axis=0)
    return res


def repeat_blocks(
    sizes, repeats, continuous_indexing=True, start_idx=0, block_inc=0, repeat_inc=0
):
    """Repeat blocks of indices.
    Adapted from
    https://stackoverflow.com/questions/51154989/numpy-vectorized-function-to- \
    repeat-blocks-of-consecutive-elements

    continuous_indexing: Whether to keep increasing the index after each block
    start_idx: Starting index
    block_inc: Number to increment by after each block,
               either global or per block. Shape: len(sizes) - 1
    repeat_inc: Number to increment by after each repetition,
                either global or per block

    Examples
    --------
        sizes = [1,3,2] ; repeats = [3,2,3] ; continuous_indexing = False
        Return: [0 0 0  0 1 2 0 1 2  0 1 0 1 0 1]
        sizes = [1,3,2] ; repeats = [3,2,3] ; continuous_indexing = True
        Return: [0 0 0  1 2 3 1 2 3  4 5 4 5 4 5]
        sizes = [1,3,2] ; repeats = [3,2,3] ; continuous_indexing = True ;
        repeat_inc = 4
        Return: [0 4 8  1 2 3 5 6 7  4 5 8 9 12 13]
        sizes = [1,3,2] ; repeats = [3,2,3] ; continuous_indexing = True ;
        start_idx = 5
        Return: [5 5 5  6 7 8 6 7 8  9 10 9 10 9 10]
        sizes = [1,3,2] ; repeats = [3,2,3] ; continuous_indexing = True ;
        block_inc = 1
        Return: [0 0 0  2 3 4 2 3 4  6 7 6 7 6 7]
        sizes = [0,3,2] ; repeats = [3,2,3] ; continuous_indexing = True
        Return: [0 1 2 0 1 2  3 4 3 4 3 4]
        sizes = [2,3,2] ; repeats = [2,0,2] ; continuous_indexing = True
        Return: [0 1 0 1  5 6 5 6]
    """
    assert sizes.dim() == 1
    assert all(sizes >= 0)
    sizes_nonzero = sizes > 0
    if not paddle.all(x=sizes_nonzero):
        assert block_inc == 0
        sizes = paddle.masked_select(x=sizes, mask=sizes_nonzero)
        if isinstance(repeats, paddle.Tensor):
            repeats = paddle.masked_select(x=repeats, mask=sizes_nonzero)
        if isinstance(repeat_inc, paddle.Tensor):
            repeat_inc = paddle.masked_select(x=repeat_inc, mask=sizes_nonzero)
    if isinstance(repeats, paddle.Tensor):
        assert all(repeats >= 0)
        insert_dummy = repeats[0] == 0
        if insert_dummy:
            one = paddle.ones(shape=[1], dtype=sizes.dtype)
            zero = paddle.zeros(shape=[1], dtype=sizes.dtype)
            sizes = paddle.concat(x=(one, sizes))
            repeats = paddle.concat(x=(one, repeats))
            if isinstance(block_inc, paddle.Tensor):
                block_inc = paddle.concat(x=(zero, block_inc))
            if isinstance(repeat_inc, paddle.Tensor):
                repeat_inc = paddle.concat(x=(zero, repeat_inc))
    else:
        assert repeats >= 0
        insert_dummy = False
    r1 = paddle.repeat_interleave(x=paddle.arange(end=len(sizes)), repeats=repeats)
    N = (sizes * repeats).sum()
    id_ar = paddle.ones(shape=N, dtype="int64")
    id_ar[0] = 0
    insert_index = sizes[r1[:-1]].cumsum(axis=0)
    insert_val = (1 - sizes)[r1[:-1]]
    if isinstance(repeats, paddle.Tensor) and paddle.any(x=repeats == 0):
        diffs = r1[1:] - r1[:-1]
        indptr = paddle.concat(
            x=(paddle.zeros(shape=[1], dtype=sizes.dtype), diffs.cumsum(axis=0))
        )
        if continuous_indexing:
            # insert_val += segment_csr(sizes[: r1[-1]], indptr, reduce="sum")
            raise NotImplementedError
        if isinstance(block_inc, paddle.Tensor):
            # insert_val += segment_csr(block_inc[: r1[-1]], indptr, reduce="sum")
            raise NotImplementedError
        else:
            insert_val += block_inc * (indptr[1:] - indptr[:-1])
            if insert_dummy:
                insert_val[0] -= block_inc
    else:
        idx = r1[1:] != r1[:-1]
        if continuous_indexing:
            insert_val[idx] = 1
        idx = paddle.where(condition=idx)[0].flatten()
        insert_val[idx] += block_inc
    if isinstance(repeat_inc, paddle.Tensor):
        insert_val += repeat_inc[r1[:-1]]
        if isinstance(repeats, paddle.Tensor):
            repeat_inc_inner = repeat_inc[repeats > 0][:-1]
        else:
            repeat_inc_inner = repeat_inc[:-1]
    else:
        insert_val += repeat_inc
        repeat_inc_inner = repeat_inc
    if isinstance(repeats, paddle.Tensor):
        repeats_inner = repeats[repeats > 0][:-1]
    else:
        repeats_inner = repeats
    idx = r1[1:] != r1[:-1]
    idx = paddle.where(condition=idx)[0].flatten()
    insert_val[idx] -= repeat_inc_inner * repeats_inner
    id_ar[insert_index] = insert_val
    if insert_dummy:
        id_ar = id_ar[1:]
        if continuous_indexing:
            id_ar[0] -= 1
    id_ar[0] += start_idx
    res = id_ar.cumsum(axis=0)
    return res


def calculate_interatomic_vectors(R, id_s, id_t, offsets_st):
    """
    Calculate the vectors connecting the given atom pairs,
    considering offsets from periodic boundary conditions (PBC).

    Parameters
    ----------
        R: Tensor, shape = (nAtoms, 3)
            Atom positions.
        id_s: Tensor, shape = (nEdges,)
            Indices of the source atom of the edges.
        id_t: Tensor, shape = (nEdges,)
            Indices of the target atom of the edges.
        offsets_st: Tensor, shape = (nEdges,)
            PBC offsets of the edges.
            Subtract this from the correct direction.

    Returns
    -------
        (D_st, V_st): tuple
            D_st: Tensor, shape = (nEdges,)
                Distance from atom t to s.
            V_st: Tensor, shape = (nEdges,)
                Unit direction from atom t to s.
    """
    Rs = R[id_s]
    Rt = R[id_t]
    if offsets_st is None:
        V_st = Rt - Rs
    else:
        V_st = Rt - Rs + offsets_st
    D_st = paddle.sqrt(x=paddle.sum(x=V_st**2, axis=1))
    V_st = V_st / D_st[..., None]
    return D_st, V_st


def inner_product_normalized(x, y):
    """
    Calculate the inner product between the given normalized vectors,
    giving a result between -1 and 1.
    """
    return paddle.sum(x=x * y, axis=-1).clip(min=-1, max=1)
