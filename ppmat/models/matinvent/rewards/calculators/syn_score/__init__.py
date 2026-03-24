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


Uses a pre-trained neural network model to predict crystal synthesizability scores.
Requires model weights and element embeddings to be present in the calculator directory.
"""

import os

EMB_PATH = os.path.join(os.path.dirname(__file__), "element_emb.json")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model_pt")
EMB_PATH = os.path.abspath(EMB_PATH)
MODEL_PATH = os.path.abspath(MODEL_PATH)

from ppmat.models.matinvent.rewards.calculators.syn_score.calc import SynScore  # noqa
from ppmat.models.matinvent.rewards.calculators.syn_score.model import Net  # noqa

__all__ = ["SynScore", "Net"]
