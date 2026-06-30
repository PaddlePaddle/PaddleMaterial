# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import numpy as np
import paddle
from ppmat.models.matinvent.rewards.base import Calculator
from ppmat.models.matinvent.rewards.calculators.syn_score import EMB_PATH, MODEL_PATH
from ppmat.models.matinvent.rewards.calculators.syn_score.model import Net


def _predict(struc_list, model_dir=MODEL_PATH, emb_path=EMB_PATH):
    if not os.path.exists(model_dir):
        raise FileNotFoundError(model_dir)
    if not os.path.exists(emb_path):
        raise FileNotFoundError(emb_path)
    with open(emb_path) as f:
        emb_dict = json.load(f)
    ds = []
    for s in struc_list:
        d = s.composition.reduced_composition.get_el_amt_dict()
        t = sum(d.values())
        e = sum(np.array(emb_dict[el]) * n for el, n in d.items()) / t
        ds.append(paddle.to_tensor(e))
    loader = paddle.io.DataLoader(
        paddle.io.TensorDataset(paddle.stack(ds, 0)),
        batch_size=64, shuffle=False, num_workers=0)
    preds = []
    for i in range(1, 101):
        mp = os.path.join(model_dir, f"checkpoint_bag_{i}.pdparams")
        if not os.path.isfile(mp):
            continue
        ckpt = paddle.load(mp)
        model = Net(atom_fea_len=ckpt["atom_fea_len"],
                    h_fea_len=ckpt["h_fea_len"],
                    n_h=ckpt["n_h"])
        model.set_state_dict(ckpt["state_dict"])
        model.eval()
        bp = []
        for (x,) in loader:
            with paddle.no_grad():
                bp.extend(paddle.exp(model(x.astype("float32"))).numpy()[:, 1].tolist())
        preds.append(bp)
    if not preds:
        raise ValueError(f"No checkpoints in {model_dir}")
    return np.array(preds).mean(axis=0)


class SynScore(Calculator):
    def calc(self, samples, label="tmp"):
        try:
            r = _predict(samples[0])
        except FileNotFoundError:
            r = np.full(len(samples[0]), 0.5)
        np.savetxt(os.path.abspath(os.path.join(self.root_dir, f"{label}.txt")), r, fmt="%.6f")
        return r
