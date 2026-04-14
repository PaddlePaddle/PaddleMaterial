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

"""conftest — stub out heavy optional deps so mofdiff tests run on CPU
without pgl, ase, pymatgen, etc. installed."""

import importlib.util
import os
import sys
import types

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _ensure_package(dotted, fs_path):
    """Register *dotted* as a package backed by *fs_path* if absent."""
    if dotted in sys.modules:
        return sys.modules[dotted]
    pkg = types.ModuleType(dotted)
    pkg.__path__ = [os.path.join(_ROOT, fs_path)]
    pkg.__package__ = dotted
    sys.modules[dotted] = pkg
    return pkg


def _load_module_from_file(dotted, rel_path):
    """Load a single .py file as *dotted* without running parent __init__.py."""
    if dotted in sys.modules:
        return sys.modules[dotted]
    full = os.path.join(_ROOT, rel_path)
    spec = importlib.util.spec_from_file_location(dotted, full)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[dotted] = mod
    spec.loader.exec_module(mod)
    return mod


# 1. Stub heavy third-party deps
for name in ("pgl", "ase", "pymatgen", "matgl", "jarvis"):
    if name not in sys.modules:
        sys.modules[name] = types.ModuleType(name)

# 2. Register minimal ppmat package tree (no __init__.py execution)
_ensure_package("ppmat", "ppmat")
_ensure_package("ppmat.utils", "ppmat/utils")
_ensure_package("ppmat.models", "ppmat/models")
_ensure_package("ppmat.models.mofdiff", "ppmat/models/mofdiff")

# 3. Load scatter utility (the only real dependency of mofdiff.py)
_load_module_from_file("ppmat.utils.scatter", "ppmat/utils/scatter.py")

# 4. Load the mofdiff module itself
_load_module_from_file(
    "ppmat.models.mofdiff.mofdiff", "ppmat/models/mofdiff/mofdiff.py"
)
