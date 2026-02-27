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

def has_element_in_range(lst, lower_bound, upper_bound):
    """
    检查给定列表 lst 中是否存在元素在指定的区间 [lower_bound, upper_bound] 内。

    参数:
    - lst: 输入的列表
    - lower_bound: 区间的下界
    - upper_bound: 区间的上界

    返回:
    - 存在元素在指定区间内时返回 True, 否则返回 False
    """
    for element in lst:
        if lower_bound <= element <= upper_bound:
            return True
    return False


def normalize_func(src_list, norm_range=[-100, 100]):
    # lihao implecation for list normalization
    # input: src_list, normalization range
    # output: tgt_list after normalization
    
    src_max, src_min = max(src_list), min(src_list)
    norm_min, norm_max = norm_range[0], norm_range[1]
    if src_max == 0: src_max = 1
    if src_min == 0: src_min = -1
    
    tgt_list = []
    for i in range(len(src_list)):
        if src_list[i] >= 0:
            tgt_list.append(src_list[i] * norm_max / src_max)
        else:
            tgt_list.append(src_list[i] * norm_min / src_min)
    
    assert len(src_list) == len(tgt_list)
    return tgt_list

if __name__ == "__main__":
    src = [-50, 0, 1, 50]
    norm_range = [-100, 100]
    tgt = normalize_func(src, norm_range)
    print(tgt)
