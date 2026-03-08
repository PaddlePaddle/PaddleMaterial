# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

import os
import numpy as np
import pandas as pd
import paddle
from paddle.io import Dataset
from paddle_geometric.data import Data
from pathlib import Path
from typing import Dict, List, Optional, Any

from ppmat.utils import ColoredTqdm as tqdm
from ppmat.utils import PlaceEnv
from ppmat.utils.compound_tools import get_atom_feature_dims, get_bond_feature_dims
from ppmat.datasets.build_ecd import build_ecformer_sample_builder
from ppmat.datasets.build_ecd import build_ecformer_downloader
from ppmat.datasets.build_ecd import GetAtomBondAngleDataset

_cache = ()

class ECDDataset(Dataset):
    """
    ECDFormer ECD 光谱预测数据集
    
    数据来源：https://paddle-org.bj.bcebos.com/paddlematerials/datasets/ECD/ECD.tar.gz
    """
    
    url = "https://paddle-org.bj.bcebos.com/paddlematerials/datasets/ECD/ECD.tar.gz"
    md5 = "aa86eddee2397dbc37c4b7b9a45b1e27"
    
    def __init__(
        self,
        data_path: str,
        split: Optional[str] = None,  # 'train'/'val'/'test'
        data_count: Optional[int] = None,
        sample_builder_cfg: Optional[Dict] = None,
        downloader_cfg: Optional[Dict] = None,
        download: bool = True,
        force_download: bool = False,
        use_geometry_enhanced: bool = True,
        use_column_info: bool = False,
    ):
        super().__init__()
        
        self.data_path = Path(data_path)
        self.split = split
        self.data_count = data_count
        self.use_geometry_enhanced = use_geometry_enhanced
        self.use_column_info = use_column_info
        
        # 构建组件
        self.sample_builder = build_ecformer_sample_builder(sample_builder_cfg)
        self.downloader = build_ecformer_downloader(downloader_cfg)
        
        # 处理下载
        if force_download or (not self._check_files() and download):
            self.downloaded_root = self.downloader.download(
                self.url, self.md5, force_download=force_download
            )
            self.data_path = self.downloaded_root
        
        # 加载数据
        self._load_data()
    
    def _check_files(self):
        """检查必要的文件是否存在"""
        npy_path = self.data_path / 'ecd_column_charity_new_smiles.npy'
        csv_path = self.data_path / 'ecd_info.csv'
        
        if not npy_path.exists():
            return False
        if not csv_path.exists():
            return False
        return True

    def _load_data(self):
        """加载所有数据"""
        # 1. 加载 npy 文件
        npy_path = self.data_path / 'ecd_column_charity_new_smiles.npy'
        if not npy_path.exists():
            raise FileNotFoundError(f"npy file not found: {npy_path}")
        
        self.ecd_dataset = np.load(npy_path, allow_pickle=True).tolist()
        
        # 2. 加载 csv 文件
        csv_path = self.data_path / 'ecd_info.csv'
        if not csv_path.exists():
            raise FileNotFoundError(f"csv file not found: {csv_path}")
        
        self.ecd_info = pd.read_csv(csv_path, encoding='gbk')
        
        # 3. 提取数据
        self.dataset_all = [item['info'] for item in self.ecd_dataset]
        self.smiles_all = [item['smiles'] for item in self.ecd_dataset]
        self.index_all = self.ecd_info['Unnamed: 0'].values
        
        # 4. 构建手性对映射
        self._build_chiral_mapping()
        
        # 5. 构建图数据集
        self._build_graph_dataset()
    
    def _build_chiral_mapping(self):
        """构建手性对映体映射"""
        self.hand_idx_dict = {}
        self.line_idx_dict = {}
        
        for i, itm in enumerate(self.ecd_dataset):
            self.line_idx_dict[i] = {
                'hand_id': itm['hand_id'],
                'unnamed_id': itm['id'],
                'smiles': itm['smiles']
            }
            
            if itm['hand_id'] not in self.hand_idx_dict:
                self.hand_idx_dict[itm['hand_id']] = []
            self.hand_idx_dict[itm['hand_id']].append({
                'line_number': i,
                'unnamed_id': itm['id'],
                'smiles': itm['smiles']
            })
    
    @PlaceEnv(paddle.CPUPlace())
    def _build_graph_dataset(self):
        """构建图数据集"""       
        global _cache

        if len(_cache) > 0:
            self.graph_atom_bond, self.graph_bond_angle = _cache
            return

        self.graph_atom_bond, self.graph_bond_angle = GetAtomBondAngleDataset(
            sample_path=str(self.data_path),
            dataset_all=self.dataset_all,
            index_all=self.index_all,
            hand_idx_dict=self.hand_idx_dict,
            line_idx_dict=self.line_idx_dict
        )

        _cache = (self.graph_atom_bond, self.graph_bond_angle)
        
        assert len(self.graph_atom_bond) == len(self.graph_bond_angle)
    
    def __len__(self):
        return len(self.graph_atom_bond)
    
    @PlaceEnv(paddle.CPUPlace())
    def __getitem__(self, idx):
        """返回 (atom_bond_graph, bond_angle_graph)"""
        return self.graph_atom_bond[idx], self.graph_bond_angle[idx]