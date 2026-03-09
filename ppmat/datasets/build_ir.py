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
import copy
import importlib
import json
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import paddle
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from paddle_geometric.data import Data

from ppmat.utils import download as download_utils
from ppmat.utils import logger
from ppmat.utils import ColoredTqdm as tqdm
from ppmat.utils.compound_tools import (
    atom_id_names, bond_id_names, bond_angle_float_names
)


def _locate_class(class_name: str):
    if "." in class_name:
        mod, cls = class_name.rsplit(".", 1)
        return getattr(importlib.import_module(mod), cls)
    return globals()[class_name]


def _parse_factory_cfg(
    cfg: Optional[Dict[str, Any] | str],
    *,
    default_class_name: str,
) -> Tuple[str, Dict[str, Any]]:
    """解析工厂配置，兼容多种格式"""
    if cfg is None:
        return default_class_name, {}

    if isinstance(cfg, str):
        return cfg, {}

    if not isinstance(cfg, dict):
        raise TypeError(f"cfg must be None, str, or dict, got {type(cfg).__name__}")

    cfg = copy.deepcopy(cfg)
    class_name = cfg.pop("__class_name__", None) or cfg.pop("class_name", None) or cfg.pop("type", None)
    if not class_name:
        raise ValueError("Factory cfg must include class name key")

    init_params = (
        cfg.pop("__init_params__", None) or
        cfg.pop("init_params", None) or
        cfg.pop("params", None) or {}
    )
    if not isinstance(init_params, dict):
        raise TypeError(f"init_params must be dict, got {type(init_params).__name__}")

    if cfg:
        raise ValueError(f"Unsupported keys in cfg: {list(cfg.keys())}")
    return class_name, init_params


class IRStrictIndexSampleBuilder:
    """按严格索引构建IR样本"""
    def build(self, data_dir: Path, meta_file: str, spectra_dir: str, data_count: Optional[int] = None):
        """构建样本列表"""
        samples = []
        meta_path = data_dir / meta_file
        data = np.load(meta_path, allow_pickle=True).item()
        
        index_all = data['index_all'][:data_count] if data_count else data['index_all']
        
        for idx in index_all:
            samples.append({
                'id': int(idx),
                'smiles': data['smiles_all'][data['index_all'].index(idx)] if hasattr(data['index_all'], 'index') else None,
                'spectrum_path': str(Path(spectra_dir) / f"{idx}.json")
            })
        return samples


class DefaultIRDatasetDownloader:
    """IR 数据集下载器"""
    def __init__(self, datasets_home: Optional[str] = None):
        self.datasets_home = datasets_home or download_utils.DATASETS_HOME

    def download(self, url: str, md5: Optional[str] = None, force_download: bool = False) -> Path:
        if force_download:
            downloaded_root = download_utils.get_path_from_url(
                url, self.datasets_home, md5sum=md5, check_exist=False, decompress=True
            )
        else:
            downloaded_root = download_utils.get_datasets_path_from_url(url, md5)
        return Path(downloaded_root)


def build_ir_downloader(cfg: Optional[Dict[str, Any] | str]):
    """构建下载器"""
    class_name, init_params = _parse_factory_cfg(cfg, default_class_name="DefaultIRDatasetDownloader")
    cls = _locate_class(class_name)
    downloader = cls(**init_params)
    if not hasattr(downloader, 'download'):
        raise TypeError(f"Downloader {class_name} must implement 'download' method")
    logger.debug(f"Use downloader: {class_name}")
    return downloader


def build_ir_sample_builder(cfg: Optional[Dict[str, Any] | str]):
    """构建样本构建器"""
    class_name, init_params = _parse_factory_cfg(cfg, default_class_name="IRStrictIndexSampleBuilder")
    cls = _locate_class(class_name)
    builder = cls(**init_params)
    if not hasattr(builder, 'build'):
        raise TypeError(f"Sample builder {class_name} must implement 'build' method")
    logger.debug(f"Use sample builder: {class_name}")
    return builder


# ==================== IR 特定工具函数 ====================

IR_WAVELENGTH_MIN = 500
IR_WAVELENGTH_MAX = 4000
IR_STEP = 100
DEFAULT_MAX_PEAKS = 15


def get_key_padding_mask(tokens):
    """生成query padding mask"""
    key_padding_mask = paddle.zeros(tokens.shape)
    key_padding_mask[tokens == -1] = -paddle.inf
    return key_padding_mask


def x_bin_position(real_x, distance=IR_STEP):
    """将实际波数转换为箱ID"""
    return int((real_x - IR_WAVELENGTH_MIN) / distance)


def Construct_IR_Dataset(dataset, data_index, descriptor_path=None):
    """
    从原始特征构建IR图数据
    """
    graph_atom_bond = []
    graph_bond_angle = []
    
    all_descriptor = None
    if descriptor_path and os.path.exists(descriptor_path):
        all_descriptor = np.load(descriptor_path)

    for i in tqdm(range(len(dataset)), desc="Constructing IR graphs"):
        data = dataset[i]
        
        # 收集原子特征
        atom_feature = []
        for name in atom_id_names:
            if name in data:
                atom_feature.append(data[name])
            else:
                if i == 0:
                    warnings.warn(f"Feature {name} not found in data, using zeros")
                num_atoms = data.get('atomic_num', np.zeros(1)).shape[0]
                atom_feature.append(np.zeros(num_atoms))
        
        # 收集键特征
        bond_feature = []
        for name in bond_id_names:
            if name in data:
                bond_feature.append(data[name])
            else:
                if i == 0:
                    warnings.warn(f"Bond feature {name} not found, using zeros")
                num_bonds = data.get('bond_dir', np.zeros(1)).shape[0]
                bond_feature.append(np.zeros(num_bonds))
        
        # 转换为Tensor
        atom_feature = paddle.to_tensor(np.array(atom_feature).T, dtype='int64')
        bond_feature = paddle.to_tensor(np.array(bond_feature).T, dtype='int64')
        
        bond_float_feature = paddle.to_tensor(data.get('bond_length', np.zeros(data['edges'].shape[0])).astype(paddle.get_default_dtype()))
        bond_angle_feature = paddle.to_tensor(data.get('bond_angle', np.zeros(data.get('BondAngleGraph_edges', np.zeros((0,2))).shape[0])).astype(paddle.get_default_dtype()))
        
        edge_index = paddle.to_tensor(data['edges'].T, dtype='int64')
        bond_index = paddle.to_tensor(data.get('BondAngleGraph_edges', np.zeros((0,2))).T, dtype='int64')
        
        data_index_int = paddle.to_tensor(np.array(int(data_index[i])), dtype='int64')
        num_atoms = atom_feature.shape[0]
        
        # 合并键特征
        bond_feature = paddle.concat(
            [bond_feature.astype(bond_float_feature.dtype), 
             bond_float_feature.reshape([-1, 1])], 
            axis=1
        )
        
        # 处理键角特征 - 确保输出6维！
        if bond_angle_feature.shape[0] > 0:
            # 基础特征：bond_angle
            features = [bond_angle_feature.reshape([-1, 1])]
            
            # 如果有描述符，添加5个描述符特征
            if all_descriptor is not None and i < all_descriptor.shape[0]:
                TPSA = paddle.ones([bond_angle_feature.shape[0]]) * all_descriptor[i, 820] / 100
                RASA = paddle.ones([bond_angle_feature.shape[0]]) * all_descriptor[i, 821]
                RPSA = paddle.ones([bond_angle_feature.shape[0]]) * all_descriptor[i, 822]
                MDEC = paddle.ones([bond_angle_feature.shape[0]]) * all_descriptor[i, 1568]
                MATS = paddle.ones([bond_angle_feature.shape[0]]) * all_descriptor[i, 457]
                
                features.extend([
                    TPSA.reshape([-1, 1]),
                    RASA.reshape([-1, 1]),
                    RPSA.reshape([-1, 1]),
                    MDEC.reshape([-1, 1]),
                    MATS.reshape([-1, 1])
                ])
            else:
                # 如果没有描述符，用0填充剩下的5维
                for _ in range(5):
                    features.append(paddle.zeros([bond_angle_feature.shape[0], 1]))
            
            # 拼接成 [E_ba, 6]
            bond_angle_feature = paddle.concat(features, axis=1)
        else:
            # 如果没有键角，创建全0的 [0, 6]
            bond_angle_feature = paddle.zeros([0, 6])
        
        data_atom_bond = Data(
            x=atom_feature,
            edge_index=edge_index,
            edge_attr=bond_feature,
            data_index=data_index_int,
        )
        
        data_bond_angle = Data(
            edge_index=bond_index,
            edge_attr=bond_angle_feature if bond_angle_feature.shape[0] > 0 else paddle.zeros([0, 1]),
            num_nodes=num_atoms,
        )
        
        graph_atom_bond.append(data_atom_bond)
        graph_bond_angle.append(data_bond_angle)

    return graph_atom_bond, graph_bond_angle


def read_ir_spectra_by_ids(sample_path, index_all, max_peak=DEFAULT_MAX_PEAKS):
    """
    按需读取IR光谱文件
    """
    ir_final_list = []
    
    for fileid in tqdm(index_all, desc="Reading IR spectra by ID"):
        filepath = os.path.join(sample_path, f"{fileid}.json")
        
        try:
            with open(filepath, 'r') as f:
                raw_ir_info = json.load(f)
            
            ir_x = raw_ir_info['x']
            ir_y = raw_ir_info['y_40']
            
            peaks_raw, _ = find_peaks(x=ir_y, height=0.1, distance=100)
            peaks_raw = peaks_raw.tolist()
            
            peak_num = min(len(peaks_raw), max_peak)
            
            if peak_num > 0:
                if len(peaks_raw) > max_peak:
                    peaks = peaks_raw[len(peaks_raw)-max_peak:]
                else:
                    peaks = peaks_raw
                
                peak_position_list = [x_bin_position(ir_x[i]) for i in peaks]
                peak_height_list = [ir_y[i] for i in peaks]
            else:
                peak_position_list = []
                peak_height_list = []
            
            peak_position_list = peak_position_list + [-1] * (max_peak - len(peak_position_list))
            peak_height_list = peak_height_list + [-1] * (max_peak - len(peak_height_list))
            
            query_padding_mask = get_key_padding_mask(paddle.to_tensor(peak_position_list))
            
            tmp_dict = {
                'id': fileid,
                'seq_40': ir_y,
                'peak_num': peak_num,
                'peak_position': peak_position_list,
                'peak_height': peak_height_list,
                'query_mask': query_padding_mask.unsqueeze(0),
            }
            ir_final_list.append(tmp_dict)
            
        except Exception as e:
            warnings.warn(f"Error processing {fileid}.json: {e}")
            continue
    
    ir_final_list.sort(key=lambda x: x['id'])
    return ir_final_list


def GetIRDataset(
    sample_path,
    dataset_all,
    index_all,
):
    """
    核心函数：构建并返回IR图数据集
    """
    # 1. 读取IR光谱序列
    ir_sequences = read_ir_spectra_by_ids(sample_path, index_all)

    # 2. 构建图数据
    total_graph_atom_bond, total_graph_bond_angle = Construct_IR_Dataset(
        dataset_all, index_all, sample_path
    )
    print("Case Before Process = ", len(total_graph_atom_bond), len(total_graph_bond_angle))

    # 3. 将光谱信息附加到图数据上
    dataset_graph_atom_bond, dataset_graph_bond_angle = [], []

    for i, itm in enumerate(ir_sequences):
        atom_bond = total_graph_atom_bond[i]

        # 附加光谱信息
        atom_bond.sequence = paddle.to_tensor([itm['seq_40']])
        atom_bond.ir_id = paddle.to_tensor(int(itm['id']))
        atom_bond.peak_num = paddle.to_tensor([itm['peak_num']])
        atom_bond.peak_position = paddle.to_tensor([itm['peak_position']])
        atom_bond.peak_height = paddle.to_tensor([itm['peak_height']])
        atom_bond.query_mask = itm['query_mask']

        dataset_graph_atom_bond.append(atom_bond)
        dataset_graph_bond_angle.append(total_graph_bond_angle[i])

    total_num = len(dataset_graph_atom_bond)
    print("Case After Process = ", len(dataset_graph_atom_bond), len(dataset_graph_bond_angle))
    print('=================== Data prepared ================\n')

    return dataset_graph_atom_bond, dataset_graph_bond_angle