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

from typing import Any
from typing import Dict

import paddle


def _all_reduce_sum_(t: paddle.Tensor) -> paddle.Tensor:
    try:
        import paddle.distributed as dist

        if dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
    except Exception:
        pass
    return t


class StreamingMetricBase:
    """
    Generic streaming-metric interface: supports multi-input / multi-output /
    cross-step accumulation, and optional per-stage execution (train/eval/sample).
    Any new model metric can be reused by the Trainer as long as it implements
    these methods.
    """

    # Optional: late-binding runtime objects (dataset infos, encoders, etc.)
    def bind(self, **runtime_objs):
        """Inject runtime dependencies (dataset_infos / train_smiles / clip / ...)."""
        return

    # Required: called once per step to accumulate internal state
    def update_step(self, *, result: Dict[str, Any], batch: Any, stage: str):
        """
        Args:
           result: model forward output (should contain some of pred_dict / loss_dict /
            label_dict)
           batch: the original batch object of this step (usually a dict)
           stage: "train" | "eval" | "sample"
        """
        raise NotImplementedError

    # Required: compute final scalars at the end of an epoch, return {name: float}
    def compute_epoch(self, *, stage: str) -> Dict[str, float]:
        raise NotImplementedError

    # Required: clear internal accumulators (typically called at epoch end)
    def reset(self):
        raise NotImplementedError
