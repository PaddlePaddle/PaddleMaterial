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
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List
import paddle
import pandas as pd
import numpy as np

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


class StrictIndexSampleBuilder:
    """按严格索引构建样本（适用于 ECD 数据集）"""
    def build(self, data_dir: Path, index_file: str, sample_path: str, data_count: Optional[int] = None):
        import pandas as pd
        samples = []
        df = pd.read_csv(data_dir / index_file, encoding='gbk')
        ids = df['Unnamed: 0'].values[:data_count] if data_count else df['Unnamed: 0'].values
        for idx in ids:
            samples.append({
                'id': int(idx),
                'smiles': df[df['Unnamed: 0'] == idx]['SMILES'].values[0],
                'spectrum_path': str(Path(sample_path) / f"{idx}.csv")
            })
        return samples


class DefaultECDDatasetDownloader:
    """ECD 数据集下载器"""
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

def build_ecformer_downloader(cfg: Optional[Dict[str, Any] | str]):
    """构建下载器"""
    class_name, init_params = _parse_factory_cfg(cfg, default_class_name="DefaultECDDatasetDownloader")
    cls = _locate_class(class_name)
    downloader = cls(**init_params)
    if not hasattr(downloader, 'download'):
        raise TypeError(f"Downloader {class_name} must implement 'download' method")
    logger.debug(f"Use downloader: {class_name}")
    return downloader

def get_key_padding_mask(tokens):
    """生成query padding mask"""
    key_padding_mask = paddle.zeros(tokens.shape)
    key_padding_mask[tokens == -1] = -paddle.inf
    return key_padding_mask

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

def get_sequence_peak(sequence):
    # input- seq: List
    # output- peak_list contains peak position
    peak_list = []
    for i in range(1, len(sequence)-1):
        if sequence[i-1]<sequence[i] and sequence[i]>sequence[i+1]:
            peak_list.append(i)
        if sequence[i-1]>sequence[i] and sequence[i]<sequence[i+1]:
            peak_list.append(i)
    return peak_list

def read_total_ecd(sample_path, fix_length=20):
    """
    读取所有ECD光谱文件，提取峰值信息
    完全复用原型程序的read_total_ecd逻辑
    """
    filepaths = [
        os.path.join(sample_path, "500ECD/data/"),
        os.path.join(sample_path, "501-2000ECD/data/"),
        os.path.join(sample_path, "2k-6kECD/data/"),
        os.path.join(sample_path, "6k-8kECD/data/"),
        os.path.join(sample_path, "8k-11kECD/data/"),
    ]
    
    ecd_dict = {}
    ecd_original_dict = {}

    for filepath in filepaths:
        if not os.path.exists(filepath):
            continue
        files = os.listdir(filepath)
        for file in files:
            if not file.endswith(".csv"):
                continue
            fileid = int(file[:-4])
            single_file_path = os.path.join(filepath, file)
            ECD_info = pd.read_csv(single_file_path).to_dict(orient='list')
            wavelengths_o, mdegs_o = ECD_info['Wavelength (nm)'], ECD_info['ECD (Mdeg)']
            
            wavelengths = [int(i) for i in wavelengths_o]
            # 将小值置零
            mdegs = [int(i) if abs(i) > 1 else 0 for i in mdegs_o]
            
            # 去除前后零值
            begin, end = 0, 0
            for i in range(len(mdegs)):
                if mdegs[i] != 0:
                    begin = i
                    break
            for i in range(len(mdegs) - 1, 0, -1):
                if mdegs[i] != 0:
                    end = i
                    break
            
            ecd_dict[fileid] = {
                'wavelengths': wavelengths[begin: end + 1],
                'ecd': mdegs[begin: end + 1],
            }
            ecd_original_dict[fileid] = {
                'wavelengths': wavelengths,
                'ecd': mdegs,
            }

    # 处理光谱序列，提取峰值
    ecd_final_list = []
    for key, itm in ecd_dict.items():
        # 等间隔采样
        distance = int(len(itm['ecd']) / (fix_length - 1))
        sequence_org = [itm['ecd'][i] for i in range(0, len(itm['ecd']), distance)][:fix_length]
        
        # 归一化
        sequence = normalize_func(sequence_org, norm_range=[-100, 100])
        
        # padding到固定长度
        if len(sequence) < fix_length:
            sequence.extend([0] * (fix_length - len(sequence)))
            sequence_org.extend([0] * (fix_length - len(sequence_org)))
        assert len(sequence) == fix_length

        # 生成峰值掩码
        peak_mask = [0] * len(sequence)
        for i in range(1, len(sequence) - 1):
            if sequence[i - 1] < sequence[i] and sequence[i] > sequence[i + 1]:
                if peak_mask[i - 1] != 2:
                    peak_mask[i - 1] = 1
                peak_mask[i] = 2
                if peak_mask[i + 1] != 2:
                    peak_mask[i + 1] = 1
            if sequence[i - 1] > sequence[i] and sequence[i] < sequence[i + 1]:
                if peak_mask[i - 1] != 2:
                    peak_mask[i - 1] = 1
                peak_mask[i] = 2
                if peak_mask[i + 1] != 2:
                    peak_mask[i + 1] = 1

        # 提取峰值位置
        peak_position_list = get_sequence_peak(sequence)
        peak_number = len(peak_position_list)
        assert peak_number < 9, f"Peak number {peak_number} >= 9"

        # 峰值符号
        peak_height_list = []
        for i in peak_position_list:
            peak_height_list.append(1 if sequence[i] >= 0 else 0)

        # padding到9个峰
        peak_position_list = peak_position_list + [-1] * (9 - peak_number)
        peak_height_list = peak_height_list + [-1] * (9 - peak_number)
        query_padding_mask = get_key_padding_mask(paddle.to_tensor(peak_position_list))

        tmp_dict = {
            'id': key,
            'seq': [0] + sequence,
            'seq_original': sequence_org,
            'seq_mask': peak_mask,
            'peak_num': peak_number,
            'peak_position': peak_position_list,
            'peak_height': peak_height_list,
            'query_mask': query_padding_mask.unsqueeze(0),
        }
        ecd_final_list.append(tmp_dict)

    ecd_final_list.sort(key=lambda x: x['id'])
    return ecd_final_list, ecd_original_dict


def Construct_dataset(dataset, data_index, path):
    """
    从原始特征构建图数据
    完全复用原型程序的Construct_dataset逻辑
    """
    graph_atom_bond = []
    graph_bond_angle = []

    all_descriptor = np.load(os.path.join(path, 'descriptor_all_column.npy'))  # (25847, 1826)

    for i in tqdm(range(len(dataset)), desc="Constructing graphs"):
        data = dataset[i]
        
        # 收集原子特征
        atom_feature = []
        for name in atom_id_names:
            atom_feature.append(data[name])
        
        # 收集键特征
        bond_feature = []
        for name in bond_id_names[0:3]:
            bond_feature.append(data[name])
        
        # 转换为Tensor
        atom_feature = paddle.to_tensor(np.array(atom_feature).T, dtype='int64')
        bond_feature = paddle.to_tensor(np.array(bond_feature).T, dtype='int64')
        bond_float_feature = paddle.to_tensor(data['bond_length'].astype(paddle.get_default_dtype()))
        bond_angle_feature = paddle.to_tensor(data['bond_angle'].astype(paddle.get_default_dtype()))
        edge_index = paddle.to_tensor(data['edges'].T, dtype='int64')
        bond_index = paddle.to_tensor(data['BondAngleGraph_edges'].T, dtype='int64')
        data_index_int = paddle.to_tensor(np.array(data_index[i]), dtype='int64')

        # 添加描述符特征（与原型程序完全一致）
        TPSA = paddle.ones([bond_angle_feature.shape[0]]) * all_descriptor[i, 820] / 100
        RASA = paddle.ones([bond_angle_feature.shape[0]]) * all_descriptor[i, 821]
        RPSA = paddle.ones([bond_angle_feature.shape[0]]) * all_descriptor[i, 822]
        MDEC = paddle.ones([bond_angle_feature.shape[0]]) * all_descriptor[i, 1568]
        MATS = paddle.ones([bond_angle_feature.shape[0]]) * all_descriptor[i, 457]

        # 合并特征
        bond_feature = paddle.concat(
            [bond_feature.astype(bond_float_feature.dtype), 
             bond_float_feature.reshape([-1, 1])], 
            axis=1
        )

        bond_angle_feature = paddle.concat(
            [bond_angle_feature.reshape([-1, 1]), TPSA.reshape([-1, 1])], 
            axis=1
        )
        bond_angle_feature = paddle.concat([bond_angle_feature, RASA.reshape([-1, 1])], axis=1)
        bond_angle_feature = paddle.concat([bond_angle_feature, RPSA.reshape([-1, 1])], axis=1)
        bond_angle_feature = paddle.concat([bond_angle_feature, MDEC.reshape([-1, 1])], axis=1)
        bond_angle_feature = paddle.concat([bond_angle_feature, MATS.reshape([-1, 1])], axis=1)

        # 创建Data对象
        data_atom_bond = Data(
            x=atom_feature,
            edge_index=edge_index,
            edge_attr=bond_feature,
            data_index=data_index_int,
        )
        data_bond_angle = Data(
            edge_index=bond_index,
            edge_attr=bond_angle_feature,
            num_nodes=atom_feature.shape[0]
        )
        
        graph_atom_bond.append(data_atom_bond)
        graph_bond_angle.append(data_bond_angle)

    return graph_atom_bond, graph_bond_angle




def GetAtomBondAngleDataset(
    sample_path,
    dataset_all,
    index_all,
    hand_idx_dict,
    line_idx_dict
):
    """
    核心函数：构建并返回切好的图数据集
    
    Args:
        sample_path: ECD光谱文件路径
        dataset_all: 从npy加载的info列表
        index_all: 索引列表
        hand_idx_dict: 手性对映射
        line_idx_dict: 行号映射
    
    Returns:
        dataset_graph_atom_bond: atom-bond图列表
        dataset_graph_bond_angle: bond-angle图列表
    """
    # 1. 读取ECD光谱序列
    ecd_sequences, ecd_original_sequences = read_total_ecd(sample_path)

    # 2. 构建图数据
    total_graph_atom_bond, total_graph_bond_angle = Construct_dataset(
        dataset_all, index_all, sample_path
    )
    print("Case Before Process = ", len(total_graph_atom_bond), len(total_graph_bond_angle))

    # 3. 将光谱序列信息附加到图数据上
    dataset_graph_atom_bond, dataset_graph_bond_angle = [], []

    for itm in ecd_sequences:
        line_num = itm['id'] - 1
        atom_bond = total_graph_atom_bond[line_num]

        # 附加光谱信息
        atom_bond.sequence = paddle.to_tensor([itm['seq']])
        atom_bond.ecd_id = paddle.to_tensor(itm['id'])
        atom_bond.seq_mask = paddle.to_tensor([itm['seq_mask']])
        atom_bond.seq_original = paddle.to_tensor([itm['seq_original']])
        atom_bond.peak_num = paddle.to_tensor([itm['peak_num']])
        atom_bond.peak_position = paddle.to_tensor([itm['peak_position']])
        atom_bond.peak_height = paddle.to_tensor([itm['peak_height']])
        atom_bond.query_mask = itm['query_mask']

        dataset_graph_atom_bond.append(atom_bond)
        dataset_graph_bond_angle.append(total_graph_bond_angle[line_num])

        # 4. 对映体增强：添加对映体样本
        hand_id, unnamed_id = line_idx_dict[line_num]['hand_id'], line_idx_dict[line_num]['unnamed_id']
        another_line_num = -1
        
        for alternative in hand_idx_dict[hand_id]:
            if alternative['unnamed_id'] != unnamed_id:
                another_line_num = alternative['line_number']
                break
                
        assert another_line_num != -1, f"cannot find the hand info of {line_num}"

        # 对映体：光谱取反
        atom_bond_oppo = total_graph_atom_bond[another_line_num]
        atom_bond_oppo.sequence = paddle.neg(paddle.to_tensor([itm['seq']]))
        atom_bond_oppo.ecd_id = paddle.to_tensor(another_line_num + 1)
        atom_bond_oppo.seq_mask = paddle.to_tensor([itm['seq_mask']])
        atom_bond_oppo.seq_original = paddle.neg(paddle.to_tensor([itm['seq_original']]))
        atom_bond_oppo.peak_num = paddle.to_tensor([itm['peak_num']])
        atom_bond_oppo.peak_position = paddle.to_tensor([itm['peak_position']])
        atom_bond_oppo.peak_height = paddle.to_tensor([itm['peak_height']])
        atom_bond_oppo.query_mask = itm['query_mask']

        dataset_graph_atom_bond.append(atom_bond_oppo)
        dataset_graph_bond_angle.append(total_graph_bond_angle[another_line_num])

    total_num = len(dataset_graph_atom_bond)
    print("Case After Process = ", len(dataset_graph_atom_bond), len(dataset_graph_bond_angle))
    print('=================== Data prepared ================\n')

    return dataset_graph_atom_bond, dataset_graph_bond_angle


def build_ecformer_sample_builder(cfg: Optional[Dict[str, Any] | str]):
    """构建样本构建器"""
    class_name, init_params = _parse_factory_cfg(cfg, default_class_name="StrictIndexSampleBuilder")
    cls = _locate_class(class_name)
    builder = cls(**init_params)
    if not hasattr(builder, 'build'):
        raise TypeError(f"Sample builder {class_name} must implement 'build' method")
    logger.debug(f"Use sample builder: {class_name}")
    return builder
