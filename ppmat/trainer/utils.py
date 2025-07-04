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

from __future__ import annotations

from typing import Dict
from typing import Sequence
from typing import Union

import paddle
import paddle.distributed as dist
from packaging import version

from ppmat.utils import logger


def log_paddle_version():
    # log paddlepaddle's version
    if version.Version(paddle.__version__) != version.Version("0.0.0"):
        paddle_version = paddle.__version__
        if version.Version(paddle.__version__) < version.Version("3.0.0"):
            logger.warning(
                f"Detected paddlepaddle version is '{paddle_version}', "
                "currently it is recommended to use release 3.0.0 or develop "
                "version."
            )
    else:
        paddle_version = f"develop({paddle.version.commit[:7]})"

    logger.info(f"Using paddlepaddle {paddle_version}")


def compute_batch_size(
    input_dict: Dict[str, Union[paddle.Tensor, Sequence[paddle.Tensor]]]
) -> int:
    """Compute batch size from given input dict.

    NOTE: Returned `batch_size` might be inaccurate, but it won't affect the correctness
    of the training results because `batch_size` is now only used for timing.

    Args:
        input_dict (Dict[str, Union[paddle.Tensor, Sequence[paddle.Tensor]]]): Given
        input dict.

    Returns:
        int: Batch size of input dict.
    """
    for _, value in input_dict.items():
        if hasattr(value, "shape"):
            return value.shape[0]
        elif hasattr(value, "__len__"):  # Might be inaccurate here.
            return len(value)
    raise ValueError("Unsupported type of input dict value.")


def scale_shared_grads(module):
    """Divide the gradients of the layers that are shared across multiple blocks by the
    number the weights are shared for
    """
    with paddle.no_grad():

        def scale_grad(param, scale_factor):
            if param.grad is None:
                return
            g_data = param.grad
            new_grads = g_data / scale_factor
            param.grad = new_grads  # .copy_(new_grads)

        if isinstance(module, dist.parallel.DataParallel):
            module = module._layers
        if hasattr(module, "model") and hasattr(module.model, "shared_parameters"):
            for layer, num_blocks in module.model.shared_parameters:
                scale_grad(layer, num_blocks)
