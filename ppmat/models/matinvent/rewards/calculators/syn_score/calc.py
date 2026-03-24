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

"""
Synthesizability score calculator.

Uses a pre-trained neural network ensemble to predict crystal synthesizability.
"""

import argparse
import json
import os
from typing import List
from typing import Tuple

import numpy as np
import paddle
import paddle.nn as nn
from pymatgen.core.structure import Structure

from ppmat.models.matinvent.rewards.base import Calculator
from ppmat.models.matinvent.rewards.calculators.syn_score import EMB_PATH
from ppmat.models.matinvent.rewards.calculators.syn_score import MODEL_PATH


class Net(nn.Layer):
    """Neural network model for synthesizability prediction."""

    def __init__(self, atom_fea_len=90, h_fea_len=180, n_h=1):
        super(Net, self).__init__()
        self.cgnf_to_fc = nn.Linear(atom_fea_len, h_fea_len)
        self.cgnf_to_fc_softplus = nn.Softplus()
        self.final_fea = 0
        if n_h > 1:
            self.fcs = nn.LayerList(
                [nn.Linear(h_fea_len, h_fea_len) for _ in range(n_h - 1)]
            )
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


class Normalizer(object):
    """Normalize a Tensor and restore it later."""

    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = paddle.mean(tensor)
        self.std = paddle.std(x=tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {"mean": self.mean, "std": self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict["mean"]
        self.std = state_dict["std"]


def collate_pool(dataset_list):
    """Collate function for data loader."""
    batch_struc_fea, batch_target = [], []
    batch_cif_ids = []
    for i, (struc_fea, target, cif_id) in enumerate(dataset_list):
        batch_struc_fea.append(struc_fea)
        batch_target.append(target)
        batch_cif_ids.append(cif_id)
    return (
        paddle.stack(batch_struc_fea, axis=0),
        paddle.stack(batch_target, axis=0),
        batch_cif_ids,
    )


def get_dataset(struc_list, emb_path):
    """Convert structure list to dataset with element embeddings."""
    if not os.path.exists(emb_path):
        raise FileNotFoundError(
            f"Element embedding file not found: {emb_path}\n"
            "Please copy element_emb.json from matinvent/rewards/calculators/syn_score/"
        )

    with open(emb_path) as f:
        emb_dict = json.load(f)
    comp = []
    comp_emb = []
    for struc in struc_list:
        _emb = np.zeros(90)
        _num = 0
        ele_dict = struc.composition.reduced_composition.get_el_amt_dict()
        for element, number in ele_dict.items():
            _num += number
            _emb += np.array(emb_dict[element]) * number
        comp_emb.append(_emb / _num)
        comp.append(struc.composition.reduced_formula)
    data = []
    for _comp, _emb in zip(comp, comp_emb):
        data.append((paddle.to_tensor(_emb), paddle.to_tensor([0]), _comp))
    return data


def validate(val_loader, model, test=True, device=None):
    """Validate/predict using the model."""
    if test:
        test_preds = []
    model.eval()
    for i, (input, target, batch_cif_ids) in enumerate(val_loader):
        with paddle.no_grad():
            input_var = input
            if device is not None:
                input_var = input_var.to(device)
        output = model(input_var.astype("float32"))
        if test:
            test_pred = paddle.exp(output.numpy())
            assert test_pred.shape[1] == 2
            test_preds += test_pred[:, 1].tolist()
    return test_preds


def predict(struc_list, model_dir=MODEL_PATH, emb_path=EMB_PATH, device=None):
    """Predict synthesizability scores using ensemble of models.

    Args:
        struc_list: List of pymatgen Structure objects
        model_dir: Directory containing model checkpoints
        emb_path: Path to element embeddings JSON
        device: Device to run models on

    Returns:
        Array of synthesizability scores (0-1 range)
    """
    if not os.path.exists(model_dir):
        raise FileNotFoundError(
            f"Model directory not found: {model_dir}\n"
            "Please copy model_pt/ from matinvent/rewards/calculators/syn_score/"
        )

    if device is None:
        device = paddle.get_device()

    dataset_test = get_dataset(struc_list, emb_path)
    collate_fn = collate_pool
    test_loader = paddle.io.DataLoader(
        dataset=dataset_test,
        batch_size=64,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )
    pred_list = []

    # Load ensemble of models (default 100 models)
    for i in range(1, 101):
        modelpath = os.path.join(model_dir, f"checkpoint_bag_{i}.pth.tar")
        if os.path.isfile(modelpath):
            model_checkpoint = paddle.load(path=str(modelpath))
            model_args = argparse.Namespace(**model_checkpoint["args"])
        else:
            # Skip if model file doesn't exist
            continue

        model = Net(
            atom_fea_len=model_args.atom_fea_len,
            h_fea_len=model_args.h_fea_len,
            n_h=model_args.n_h,
        )
        if device is not None:
            model.to(device)

        normalizer = Normalizer(paddle.zeros([3]))
        if os.path.isfile(modelpath):
            checkpoint = paddle.load(path=str(modelpath))
            model.set_state_dict(checkpoint["state_dict"])
            normalizer.load_state_dict(checkpoint["normalizer"])

        preds = validate(test_loader, model, test=True, device=device)
        pred_list.append(preds)

    if len(pred_list) == 0:
        raise ValueError(
            f"No model checkpoints found in {model_dir}. "
            "Please copy model_pt/ from matinvent/rewards/calculators/syn_score/"
        )

    pred_array = np.array(pred_list)
    syn_score = pred_array.mean(axis=0)
    return syn_score


class SynScore(Calculator):
    """Synthesizability score calculator.

    Uses an ensemble of 100 pre-trained neural networks to predict
    the synthesizability likelihood of crystal structures.
    """

    def __init__(self, root_dir: str, task: str = "syn_score") -> None:
        """Initialize SynScore calculator.

        Args:
            root_dir: Directory for output files
            task: Must be "syn_score"
        """
        super().__init__(root_dir, task)

    def calc(
        self, samples: Tuple[List[Structure], str], label: str = "tmp"
    ) -> np.ndarray:
        """Calculate synthesizability scores.

        Args:
            samples: Tuple of (structure_list, path)
            label: Label for output file

        Returns:
            Array of synthesizability scores (0-1 range, higher = more synthesizable)
        """
        struc_list = samples[0]
        out_path = os.path.join(self.root_dir, f"{label}.txt")
        out_path = os.path.abspath(out_path)

        try:
            results = predict(struc_list)
        except FileNotFoundError as e:
            # Return default scores if models not available
            import warnings

            warnings.warn(
                f"SynScore model files not found: {e}\n"
                "Returning default scores of 0.5. "
                "Please copy element_emb.json and model_pt/ "
                "from matinvent/rewards/calculators/syn_score/"
            )
            results = np.full(len(struc_list), 0.5)

        np.savetxt(out_path, results, fmt="%.6f")
        return results
