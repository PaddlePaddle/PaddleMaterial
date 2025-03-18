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
import numpy as np


class ConcatData(object):
    """This class is used to concatenate numpy data when grouping batches

    Args:
        data (np.ndarray): Data to be concatenated with other data.
    """

    def __init__(self, data: np.ndarray) -> None:

        self.data = data

    @staticmethod
    def batch(data_list):
        data_list = [data.data for data in data_list]
        data = np.concatenate(data_list, axis=0)
        return data

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return str(self.__dict__)
