# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle.nn as nn


class Net(nn.Layer):
    def __init__(self, atom_fea_len=90, h_fea_len=180, n_h=1):
        super(Net, self).__init__()
        self.cgnf_to_fc = nn.Linear(atom_fea_len, h_fea_len)
        self.cgnf_to_fc_softplus = nn.Softplus()
        self.final_fea = 0
        if n_h > 1:
            self.fcs = nn.LayerList(
                [nn.Linear(h_fea_len, h_fea_len) for _ in range(n_h - 1)])
            self.softpluses = nn.LayerList([nn.Softplus() for _ in range(n_h - 1)])
        self.fc_out = nn.Linear(h_fea_len, 2)
        self.logsoftmax = nn.LogSoftmax(axis=1)
        self.dropout = nn.Dropout()

    def forward(self, struc_fea):
        comp_fea = self.cgnf_to_fc(struc_fea)
        comp_fea = self.cgnf_to_fc_softplus(comp_fea)
        comp_fea = self.dropout(comp_fea)
        if hasattr(self, "fcs") and hasattr(self, "softpluses"):
            for fc, softplus in zip(self.fcs, self.softpluses):
                comp_fea = softplus(fc(comp_fea))
        self.final_fea = comp_fea
        out = self.fc_out(comp_fea)
        out = self.logsoftmax(out)
        return out
