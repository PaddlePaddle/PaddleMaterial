# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import paddle


class LiFlowMSE:
    """Mean squared error of predicted and reference atomic velocities."""

    def __call__(self, prediction, target):
        return paddle.mean((prediction - target) ** 2)
