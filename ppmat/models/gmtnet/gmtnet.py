import paddle
class GMTNet(paddle.nn.Layer):
    def __init__(self, args):
        super().__init__()
        self.fc = paddle.nn.Linear(92, 9)
        paddle.nn.initializer.KaimingUniform(negative_slope=0.01)(self.fc.weight)
        paddle.nn.initializer.Constant(0.0)(self.fc.bias)
    def forward(self, data, feat_mask, equality):
        x = data.x
        x_mean = x.mean(axis=0, keepdim=True)
        return self.fc(x_mean)
