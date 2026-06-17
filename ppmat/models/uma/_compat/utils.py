# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

from functools import wraps


def conditional_grad(dec):
    """Enable gradients only when model is configured for autograd forces."""

    def decorator(func):
        @wraps(func)
        def cls_method(self, *args, **kwargs):
            f = func
            if getattr(self, "regress_forces", False) and not getattr(
                self, "direct_forces", False
            ):
                f = dec(func)
            return f(self, *args, **kwargs)

        return cls_method

    return decorator
