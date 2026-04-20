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

from paddle_geometric.utils import scatter as scatter_org


# The parameter may need to be converted to an integer
# before calling the original function, so add the function here
def scatter(src, index, dim=0, dim_size=None, reduce="sum"):
    return scatter_org(src, index, dim=dim, dim_size=dim_size, reduce=reduce)
