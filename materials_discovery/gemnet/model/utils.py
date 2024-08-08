import json
import paddle 

from typing import Optional

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


def _scatter_sum(src: paddle.Tensor, index: paddle.Tensor, dim: int = -1,
                out: Optional[paddle.Tensor] = None,
                dim_size: Optional[int] = None) -> paddle.Tensor:
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
    return paddle.put_along_axis(arr=out, indices=index, values=src, axis=dim, reduce='add')
 

def _scatter_add(src: paddle.Tensor, index: paddle.Tensor, dim: int = -1,
                out: Optional[paddle.Tensor] = None,
                dim_size: Optional[int] = None) -> paddle.Tensor:
    return _scatter_sum(src, index, dim, out, dim_size)


def _scatter_mean(src: paddle.Tensor, index: paddle.Tensor, dim: int = -1,
                 out: Optional[paddle.Tensor] = None,
                 dim_size: Optional[int] = None) -> paddle.Tensor:
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


def scatter(src: paddle.Tensor, index: paddle.Tensor, dim: int = -1,
            out: Optional[paddle.Tensor] = None, dim_size: Optional[int] = None,
            reduce: str = "sum") -> paddle.Tensor:
    """
    Implement paddle version API like torch_scatter.scatter
    """
    if reduce == 'sum' or reduce == 'add':
        return _scatter_sum(src, index, dim, out, dim_size)
    elif reduce == 'mean':
        return _scatter_mean(src, index, dim, out, dim_size)
    else:
        raise ValueError('Only support add or mean')
