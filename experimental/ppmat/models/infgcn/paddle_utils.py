import paddle


def view(self, *args, **kwargs):
    if args:
        if len(args) == 1 and isinstance(args[0], (tuple, list, str)):
            return paddle.view(self, args[0])
        else:
            return paddle.view(self, list(args))
    elif kwargs:
        return paddle.view(self, shape_or_dtype=list(kwargs.values())[0])


setattr(paddle.Tensor, "view", view)


def dim2perm(ndim, dim0, dim1):
    perm = list(range(ndim))
    perm[dim0], perm[dim1] = perm[dim1], perm[dim0]
    return perm


class Embedding(paddle.nn.Embedding):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.padding_idx = self._padding_idx


setattr(paddle.nn, "Embedding", Embedding)

"""
# split_tensor_func
def split_tensor_func(self, split_size, dim=0):
    if isinstance(split_size, int):
        return paddle.split(self, self.shape[dim] // split_size, dim)
    else:
        return paddle.split(self, split_size, dim)
"""


def split_tensor_func(self, split_size, dim=0):
    total_size = self.shape[dim]

    if isinstance(split_size, int):
        sections = []
        for i in range(0, total_size, split_size):
            sections.append(min(split_size, total_size - i))
        return paddle.split(self, sections, dim)
    else:
        return paddle.split(self, split_size, dim)


setattr(paddle.Tensor, "split", split_tensor_func)
