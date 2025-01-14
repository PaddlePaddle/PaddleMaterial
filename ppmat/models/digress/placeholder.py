import paddle


class PlaceHolder:
    """
    辅助类，用于封装 (X, E, y) 并提供 type_as / mask 等方法。
    """

    def __init__(self, X, E, y):
        self.X = X
        self.E = E
        self.y = y

    def type_as(self, x: paddle.Tensor):
        self.X = self.X.astype(x.dtype)
        self.E = self.E.astype(x.dtype)
        self.y = self.y.astype(x.dtype)
        return self

    def mask(self, node_mask, collapse=False):
        # x_mask = node_mask.unsqueeze(-1)
        x_mask = paddle.unsqueeze(node_mask, axis=-1)  # (bs, n, 1)
        e_mask1 = paddle.unsqueeze(x_mask, axis=2)  # (bs, n, 1, 1)
        e_mask2 = paddle.unsqueeze(x_mask, axis=1)  # (bs, 1, n, 1)

        if collapse:
            # self.X = torch.argmax(self.X, dim=-1)
            self.X = paddle.argmax(self.X, axis=-1)  # (bs,n)
            self.E = paddle.argmax(self.E, axis=-1)  # (bs,n,n)

            # self.X[node_mask == 0] = -1
            zero_mask = node_mask == 0
            self.X = paddle.where(
                zero_mask, paddle.full_like(self.X, fill_value=-1), self.X
            )

            # e_mask => (bs,n,n) shape (由 e_mask1 * e_mask2 => (bs,n,n,1)?)
            e_mask = paddle.squeeze(e_mask1 * e_mask2, axis=-1)  # (bs,n,n)
            self.E = paddle.where(
                e_mask == 0, paddle.full_like(self.E, fill_value=-1), self.E
            )
        else:
            # self.X = self.X * x_mask
            self.X = self.X * x_mask
            # self.E = self.E * e_mask1 * e_mask2
            self.E = self.E * e_mask1 * e_mask2

            if not paddle.allclose(self.E, paddle.transpose(self.E, perm=[0, 2, 1, 3])):
                raise ValueError("E is not symmetric after masking.")
        return self
