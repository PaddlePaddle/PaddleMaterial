import paddle

"""
@author : Hyunwoong
@when : 2019-12-18
@homepage : https://github.com/gusdnd852
"""


class PositionwiseFeedForward(paddle.nn.Layer):
    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = paddle.nn.Linear(in_features=d_model, out_features=hidden)
        self.linear2 = paddle.nn.Linear(in_features=hidden, out_features=d_model)
        self.relu = paddle.nn.ReLU()
        self.dropout = paddle.nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
