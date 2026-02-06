import paddle


def _broadcast(src, other, dim):
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = paddle.expand_as(src, other)
    return src


def scatter_sum(
    src,
    index,
    dim=-1,
    out=None,
    dim_size=None,
):
    index = _broadcast(index, src, dim)
    if out is None:
        size = list(src.shape)
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max().item()) + 1
        out = paddle.zeros(size, dtype=src.dtype)
    
    if dim < 0:
        dim = src.ndim + dim
    
    for i in range(out.shape[dim]):
        mask = (index == i)
        if dim == 0:
            out[i] = paddle.where(mask, src, paddle.zeros_like(src)).sum(axis=0)
        else:
            masked_src = paddle.where(mask, src, paddle.zeros_like(src))
            if dim == 1:
                out[:, i] = masked_src.sum(axis=dim)
            elif dim == 2:
                out[:, :, i] = masked_src.sum(axis=dim)
    
    return out


def scatter_mean(
    src,
    index,
    dim=-1,
    out=None,
    dim_size=None,
):
    out = scatter_sum(src, index, dim, out, dim_size)
    dim_size = out.shape[dim]

    index_dim = dim
    if index_dim < 0:
        index_dim = index_dim + src.ndim
    if index.ndim <= index_dim:
        index_dim = index.ndim - 1

    ones = paddle.ones(index.shape, dtype=src.dtype)
    count = scatter_sum(ones, index, index_dim, None, dim_size)
    count = paddle.where(count < 1, paddle.ones_like(count), count)
    count = _broadcast(count, out, dim)
    
    if paddle.is_floating_point(out):
        out = out / count
    else:
        out = out // count
    
    return out


def scatter_std(
    src,
    index,
    dim=-1,
    out=None,
    dim_size=None,
    unbiased=True,
):
    if out is not None:
        dim_size = out.shape[dim]

    if dim < 0:
        dim = src.ndim + dim

    count_dim = dim
    if index.ndim <= dim:
        count_dim = index.ndim - 1

    ones = paddle.ones(index.shape, dtype=src.dtype)
    count = scatter_sum(ones, index, count_dim, dim_size=dim_size)

    index = _broadcast(index, src, dim)
    tmp = scatter_sum(src, index, dim, dim_size=dim_size)
    count_broadcast = _broadcast(count, tmp, dim)
    count_broadcast = paddle.clip(count_broadcast, min=1)
    mean = tmp / count_broadcast

    var = src - paddle.take_along_axis(mean, index, axis=dim)
    var = var * var
    out = scatter_sum(var, index, dim, out, dim_size)

    if unbiased:
        count_broadcast = paddle.clip(count_broadcast - 1, min=1)
    out = paddle.sqrt(out / (count_broadcast + 1e-6))

    return out
