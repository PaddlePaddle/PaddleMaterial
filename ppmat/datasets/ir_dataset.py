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

import numpy as np
import paddle
from paddle.io import Dataset
from pathlib import Path
from typing import Dict, Optional

from ppmat.utils import PlaceEnv
from ppmat.datasets.build_ir import (
    build_ir_sample_builder,
    build_ir_downloader,
    read_ir_spectra_by_ids,
    Construct_IR_Dataset,
)

_cache = {}


class IRDataset(Dataset):
    """
    ECFormer IR Spectrum Prediction Dataset
    
    Supports three preloading modes:
    - '100': Small dataset with 100 samples (default, for quick testing)
    - '10000': Medium dataset with 10,000 samples
    - 'all': All samples (may be very large)
    """
    
    url = "https://paddle-org.bj.bcebos.com/paddlematerials/datasets/IR/IR.tar.gz"
    md5 = "e1ea5624cf9b92b3657933245196f5dc"
    
    def __init__(
        self,
        data_path: str,
        mode: str = '100',
        split: Optional[str] = None,
        data_count: Optional[int] = None,
        sample_builder_cfg: Optional[Dict] = None,
        downloader_cfg: Optional[Dict] = None,
        download: bool = True,
        force_download: bool = False,
        use_geometry_enhanced: bool = True,
        use_cache: bool = True,
    ):
        super().__init__()
        
        self.data_path = Path(data_path)
        self.mode = mode
        self.split = split
        self.data_count = data_count
        self.use_geometry_enhanced = use_geometry_enhanced
        self.use_cache = use_cache
        
        cache_key = f"{data_path}_{mode}_{use_geometry_enhanced}"
        
        # If cache is enabled and hit, return directly
        if use_cache and cache_key in _cache:
            cached_data = _cache[cache_key]
            self.graph_atom_bond = cached_data['atom_bond']
            self.graph_bond_angle = cached_data['bond_angle']
            self.smiles_list = cached_data.get('smiles', [])
            return
        
        # Build components
        self.sample_builder = build_ir_sample_builder(sample_builder_cfg)
        self.downloader = build_ir_downloader(downloader_cfg)
        
        # Handle download
        if force_download or (not self._check_files() and download):
            self.downloaded_root = self.downloader.download(
                self.url, self.md5, force_download=force_download
            )
            self.data_path = self.downloaded_root
        
        # Load data
        self._load_data()
        
        # Store in cache
        if use_cache:
            _cache[cache_key] = {
                'atom_bond': self.graph_atom_bond,
                'bond_angle': self.graph_bond_angle,
                'smiles': self.smiles_list
            }
    
    def _check_files(self):
        """Check if necessary files exist"""
        meta_path = self.data_path / f'ir_column_charity_{self.mode}.npy'
        spectra_path = self.data_path / 'qm9_ir_spec'
        
        if not meta_path.exists():
            return False
        if not spectra_path.exists():
            return False
        return True

    def _load_data(self):
        """Load all data"""
        # 1. Load metadata file
        meta_path = self.data_path / f'ir_column_charity_{self.mode}.npy'
        if not meta_path.exists():
            raise FileNotFoundError(f"IR meta file {meta_path} not found")
        
        data = np.load(meta_path, allow_pickle=True).item()
        dataset_all = data['dataset_all']
        smiles_all = data['smiles_all']
        index_all = data['index_all']
        
        print(f"Loaded meta data: {len(index_all)} samples")
        
        # 2. Read IR spectra on demand
        spectra_path = self.data_path / 'qm9_ir_spec'
        self.ir_sequences = read_ir_spectra_by_ids(str(spectra_path), index_all)
        
        print(f"Loaded {len(self.ir_sequences)} IR spectra")
        
        # 3. Construct graph data
        descriptor_path = self.data_path / 'descriptor_all_column.npy'
        if not descriptor_path.exists():
            descriptor_path = None
        
        total_graph_atom_bond, total_graph_bond_angle = Construct_IR_Dataset(
            dataset_all, index_all, descriptor_path
        )
        
        # 4. Attach spectrum information to graph data
        self.graph_atom_bond = []
        self.graph_bond_angle = []
        self.smiles_list = []
        
        for i, itm in enumerate(self.ir_sequences):
            atom_bond = total_graph_atom_bond[i]
            
            atom_bond.sequence = paddle.to_tensor([itm['seq_40']])
            atom_bond.ir_id = paddle.to_tensor(int(itm['id']))
            atom_bond.peak_num = paddle.to_tensor([itm['peak_num']])
            atom_bond.peak_position = paddle.to_tensor([itm['peak_position']])
            atom_bond.peak_height = paddle.to_tensor([itm['peak_height']])
            atom_bond.query_mask = itm['query_mask']
            
            self.graph_atom_bond.append(atom_bond)
            self.graph_bond_angle.append(total_graph_bond_angle[i])
            self.smiles_list.append(smiles_all[i])
        
        print(f"Final dataset size: {len(self.graph_atom_bond)}")
    
    def __len__(self):
        return len(self.graph_atom_bond)
    
    @PlaceEnv(paddle.CPUPlace())
    def __getitem__(self, idx):
        """Returns (atom_bond_graph, bond_angle_graph)"""
        return self.graph_atom_bond[idx], self.graph_bond_angle[idx]