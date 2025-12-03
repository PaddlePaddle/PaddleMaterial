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

__version__ = "0.5.1"
from typing import Dict

_OPT_DEFAULTS: Dict[str, bool] = dict(
    specialized_code=True, optimize_einsums=True, jit_script_fx=True
)


def set_optimization_defaults(**kwargs) -> None:
    """Globally set the default optimization settings.

    Parameters
    ----------
    **kwargs
        Keyword arguments to set the default optimization settings.
    """
    for k, v in kwargs.items():
        if k not in _OPT_DEFAULTS:
            raise ValueError(f"Unknown optimization option: {k}")
        _OPT_DEFAULTS[k] = v


def get_optimization_defaults() -> Dict[str, bool]:
    """Get the global default optimization settings."""
    return dict(_OPT_DEFAULTS)


from . import io as io
from . import nn as nn
from . import o3 as o3
