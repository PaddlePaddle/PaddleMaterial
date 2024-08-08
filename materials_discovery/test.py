# # import paddle 
# # import torch 
# # import torch_scatter 
# # import numpy as np 
# # from typing import Optional

# # # # def _scatter_sum(src: paddle.Tensor, index: paddle.Tensor, dim: int = -1,
# # # #                 out: Optional[paddle.Tensor] = None,
# # # #                 dim_size: Optional[int] = None) -> paddle.Tensor:
# # # #     index = broadcast(index, src, dim)
# # # #     if out is None:
# # # #         size = list(src.shape)
# # # #         if dim_size is not None:
# # # #             size[dim] = dim_size
# # # #         elif index.numel() == 0:
# # # #             size[dim] = 0
# # # #         else:
# # # #             size[dim] = int(index.max()) + 1
# # # #         out = paddle.zeros(size, dtype=src.dtype)
# # # #         return out.scatter_add_(dim, index, src)
# # # #     else:
# # # #         return out.scatter_add_(dim, index, src)

# # # # def paddle_scatter(src, index, dim, out=None, dim_size=None, reduce='add'):
# # # #     """Implement paddle version API like torch_scatter.scatter
# # # #     """
# # # #     if reduce not in ('add', 'mean'):
# # # #         raise ValueError('The paddle_scatter only support add or mean reduce type.')

# # # #     index = _broadcast(index, src, dim)
# # # #     if out is None:
# # # #         size = list(src.shape)
# # # #         if dim_size is not None:
# # # #             size[dim] = dim_size
# # # #         elif index.numel() == 0:
# # # #             size[dim] = 0
# # # #         else:
# # # #             size[dim] = int(index.max()) + 1
# # # #         out = paddle.zeros(size, dtype=src.dtype)
# # # #     return paddle.put_along_axis(arr=out, indices=index, values=src, axis=dim, reduce=reduce, include_self=False)




# # def _broadcast(src: paddle.Tensor, other: paddle.Tensor, dim: int):
# #     if dim < 0:
# #         dim = other.dim() + dim
# #     if src.dim() == 1:
# #         for _ in range(0, dim):
# #             src = src.unsqueeze(0)
# #     for _ in range(src.dim(), other.dim()):
# #         src = src.unsqueeze(-1)
# #     src = src.expand(other.shape)
# #     return src


# # def _scatter_sum(src: paddle.Tensor, index: paddle.Tensor, dim: int = -1,
# #                 out: Optional[paddle.Tensor] = None,
# #                 dim_size: Optional[int] = None) -> paddle.Tensor:
# #     index = _broadcast(index, src, dim)
# #     if out is None:
# #         size = list(src.shape)
# #         if dim_size is not None:
# #             size[dim] = dim_size
# #         elif index.numel() == 0:
# #             size[dim] = 0
# #         else:
# #             size[dim] = int(index.max()) + 1
# #         out = paddle.zeros(size, dtype=src.dtype)
# #     return paddle.put_along_axis(arr=out, indices=index, values=src, axis=dim, reduce='add')
 

# # def _scatter_add(src: paddle.Tensor, index: paddle.Tensor, dim: int = -1,
# #                 out: Optional[paddle.Tensor] = None,
# #                 dim_size: Optional[int] = None) -> paddle.Tensor:
# #     return _scatter_sum(src, index, dim, out, dim_size)


# # def _scatter_mean(src: paddle.Tensor, index: paddle.Tensor, dim: int = -1,
# #                  out: Optional[paddle.Tensor] = None,
# #                  dim_size: Optional[int] = None) -> paddle.Tensor:
# #     out = _scatter_sum(src, index, dim, out, dim_size)
# #     dim_size = out.shape[dim]

# #     index_dim = dim
# #     if index_dim < 0:
# #         index_dim = index_dim + src.dim()
# #     if index.dim() <= index_dim:
# #         index_dim = index.dim() - 1

# #     ones = paddle.ones(index.shape, dtype=src.dtype)
# #     count = _scatter_sum(ones, index, index_dim, None, dim_size)
# #     count[count < 1] = 1
# #     count = _broadcast(count, out, dim)
# #     if out.is_floating_point():
# #         out = paddle.divide(out, count)
# #         # out.true_divide_(count)
# #     else:
# #         out = paddle.floor_divide(out, count)
# #         # out.div_(count, rounding_mode='floor')
# #     return out


# # def paddle_scatter(src: paddle.Tensor, index: paddle.Tensor, dim: int = -1,
# #             out: Optional[paddle.Tensor] = None, dim_size: Optional[int] = None,
# #             reduce: str = "sum") -> paddle.Tensor:
# #     r"""
# #     """
# #     if reduce == 'sum' or reduce == 'add':
# #         return _scatter_sum(src, index, dim, out, dim_size)
# #     elif reduce == 'mean':
# #         return _scatter_mean(src, index, dim, out, dim_size)
# #     else:
# #         raise ValueError('Only support add or mean')

# # x = np.array([[10, 30, 20], [60, 40, 50]])
# # src = np.array([[1,2],[3,4]])
# # indices = np.zeros([2,2]).astype(np.int64)

# # px = paddle.to_tensor(x)
# # psrc = paddle.to_tensor(src)
# # pindices = paddle.to_tensor(indices)
# # # print(psrc, pindices)
# # pout = paddle_scatter(src=psrc, dim=0, index=pindices, out=None, dim_size=2, reduce='mean')
# # print(pout)


# # tx = torch.tensor(x)
# # tsrc = torch.tensor(src)
# # tindices = torch.tensor(indices)
# # # tout = torch.scatter(input=tx, src=tsrc, index=tindices, dim=0, reduce='add')
# # # print(tout)

# # # print(tsrc, tindices)
# # tsout = torch_scatter.scatter(src=tsrc, dim=0, index=tindices, out=None, dim_size=2, reduce='mean')
# # print(tsout)




# import paddle 
# x = paddle.create_parameter(shape=[128, 1], dtype='float32')
# # init_Orthogonal = paddle.nn.initializer.Orthogonal()
# # init_Orthogonal(x)
# # print(x)

# v = paddle.var(x, axis=1)
# print(v)


# import torch
# x = torch.tensor(x.numpy())
# # init_Orthogonal = paddle.nn.initializer.Orthogonal()
# # init_Orthogonal(x)
# # print(x)

# v = torch.var(x, axis=1)
# print(v)


import paddle


paddle.base.core.set_prim_eager_enabled(True)

class MyNet(paddle.nn.Layer):
    def __init__(self):
        super(MyNet, self).__init__()
        self.weight = self.create_parameter(shape=(2,2), dtype=paddle.float32, is_bias=False)
        self.bias = self.create_parameter(shape=(2,2), dtype=paddle.float32, is_bias=True)
        self.add_parameter("weight", self.weight)
        self.add_parameter("bias", self.bias)

    def forward(self, x):
        y = paddle.matmul(x, self.weight) + self.bias
        return paddle.tanh(y)


x = paddle.randn(shape=(2,2), dtype=paddle.float32)
net = MyNet()
y = net(x)


grad1 = paddle.grad(y, x)
grad2 = paddle.grad(grad1, x)
loss = paddle.norm(grad2, p=2)


opt = paddle.optimizer.Adam(parameters=net.parameters())
loss.backward()
opt.update()