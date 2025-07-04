# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This code is heavily adapted from：
# https://github.com/huggingface/transformers/tree/main/src/transformers/callbacks.py


import dataclasses
import json
from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainerState:
    """A class containing the [`Trainer`] inner state that will be saved.

    <Tip>

    In all this class, one step is to be understood as one update step. When using
    gradient accumulation, one update step may require several forward and backward
    passes: if you use `gradient_accumulation_steps=n`, then one update step requires
    going through *n* batches.

    </Tip>

    Args:
        epoch (int, optional): Only set during training, will represent the epoch the
            training is at. Defaults to 0.
        global_step (int, optional): During training, represents the number of update
            steps completed. Defaults to 0.
        step_in_train_epoch (int, optional): During training, represents the number of
            update steps completed within the current epoch. Defaults to 0.
        step_in_eval_epoch (int, optional): During evaluation, represents the number of
            update steps completed within the current epoch. Defaults to 0.
        max_steps_in_train_epoch (int, optional): The number of update steps to do
            during the current training epoch. Defaults to 0.
        max_steps_in_eval_epoch (int, optional): The number of update steps to do
            during the current evaluation epoch. Defaults to 0.
        cur_metric (Optional[float]): The current metric value. Defaults to None.
        best_metric (Optional[float]): The best metric value. Defaults to None.
        best_epoch (Optional[int]): The epoch where the best metric was reached.
            Defaults to None.
    """

    # training state, updated during training
    epoch: int = 0
    global_step: int = 0
    step_in_train_epoch: int = 0
    step_in_eval_epoch: int = 0

    max_steps_in_train_epoch: int = 0
    max_steps_in_eval_epoch: int = 0

    cur_metric: Optional[float] = None
    best_metric: Optional[float] = None
    best_epoch: Optional[int] = None

    def to_dict(self):
        return dataclasses.asdict(self)

    @staticmethod
    def from_dict(dict_data):
        return TrainerState(**dict_data)

    def save_to_json(self, json_path: str):
        """Save the content of this instance in JSON format inside `json_path`."""
        json_string = (
            json.dumps(dataclasses.asdict(self), indent=2, sort_keys=True) + "\n"
        )
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(json_string)

    @classmethod
    def load_from_json(json_path: str):
        """Create an instance from the content of `json_path`."""
        with open(json_path, encoding="utf-8") as f:
            text = f.read()
        return TrainerState(**json.loads(text))
