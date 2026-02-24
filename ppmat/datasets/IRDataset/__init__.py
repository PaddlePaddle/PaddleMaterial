# IRDataset.py
"""
IR光谱预测数据集模块
支持预加载的npy文件，包含缓存机制，默认使用100样本的小数据集
"""

import os
import numpy as np
import pandas as pd
import paddle
from paddle.io import Dataset, DataLoader
from paddle_geometric.data import Data
from tqdm import tqdm
import pickle
import json
import warnings

import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem

from .compound_tools import mol_to_geognn_graph_data_MMFF3d
from .compound_tools import get_atom_feature_dims, get_bond_feature_dims
from .colored_tqdm import ColoredTqdm as tqdm
from .place_env import PlaceEnv

# ----------------常量定义----------------
ATOM_ID_NAMES = [
    "atomic_num", "chiral_tag", "degree", "explicit_valence",
    "formal_charge", "hybridization", "implicit_valence",
    "is_aromatic", "total_numHs",
]

BOND_ID_NAMES = ["bond_dir", "bond_type", "is_in_ring"]

BOND_ANGLE_FLOAT_NAMES = ['bond_angle', 'TPSA', 'RASA', 'RPSA', 'MDEC', 'MATS']

# 获取特征维度
FULL_ATOM_FEATURE_DIMS = get_atom_feature_dims(ATOM_ID_NAMES)
FULL_BOND_FEATURE_DIMS = get_bond_feature_dims(BOND_ID_NAMES)

# IR光谱参数
IR_WAVELENGTH_MIN = 500
IR_WAVELENGTH_MAX = 4000
IR_STEP = 100  # 波数步长，用于离散化
IR_NUM_POSITION_CLASSES = (IR_WAVELENGTH_MAX - IR_WAVELENGTH_MIN) // IR_STEP  # 36

# 默认最大峰数
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
    类似于ECD的Construct_dataset，但针对IR任务
    
    Args:
        dataset: list of dict, 每个元素是分子的info字典
        data_index: list or array, 索引列表
        descriptor_path: str, 描述符文件路径（IR可能不需要，保留接口）
    
    Returns:
        graph_atom_bond: list of Data, atom-bond图
        graph_bond_angle: list of Data, bond-angle图
    """
    graph_atom_bond = []
    graph_bond_angle = []
    
    # IR任务可能不需要描述符，但如果需要可以加载
    all_descriptor = None
    if descriptor_path and os.path.exists(descriptor_path):
        all_descriptor = np.load(descriptor_path)

    for i in tqdm(range(len(dataset)), desc="Constructing IR graphs"):
        data = dataset[i]
        
        # 收集原子特征
        atom_feature = []
        for name in ATOM_ID_NAMES:
            if name in data:
                atom_feature.append(data[name])
            else:
                # 如果某些特征缺失，用0填充
                # 注意：根据实际数据调整
                if i == 0:  # 只在第一次警告
                    warnings.warn(f"Feature {name} not found in data, using zeros")
                num_atoms = data.get('atomic_num', np.zeros(1)).shape[0]
                atom_feature.append(np.zeros(num_atoms))
        
        # 收集键特征
        bond_feature = []
        for name in BOND_ID_NAMES:
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
        
        # 键长特征（IR可能不需要，但保留）
        bond_float_feature = paddle.to_tensor(data.get('bond_length', np.zeros(data['edges'].shape[0])).astype(np.float32))
        
        # 键角特征（IR可能不需要，但保留）
        bond_angle_feature = paddle.to_tensor(data.get('bond_angle', np.zeros(data.get('BondAngleGraph_edges', np.zeros((0,2))).shape[0])).astype(np.float32))
        
        # 边索引
        edge_index = paddle.to_tensor(data['edges'].T, dtype='int64')
        bond_index = paddle.to_tensor(data.get('BondAngleGraph_edges', np.zeros((0,2))).T, dtype='int64')
        

        data_index_int = paddle.to_tensor(np.array(int(data_index[i])), dtype='int64')
        
        # 获取原子数（键角图的节点数）
        num_atoms = atom_feature.shape[0]
        
        # 合并键特征
        bond_feature = paddle.concat(
            [bond_feature.astype(bond_float_feature.dtype), 
             bond_float_feature.reshape([-1, 1])], 
            axis=1
        )
        
        # 处理键角特征（如果有）
        if bond_angle_feature.shape[0] > 0:
            # 如果有描述符，可以添加
            if all_descriptor is not None and i < all_descriptor.shape[0]:
                TPSA = paddle.ones([bond_angle_feature.shape[0]]) * all_descriptor[i, 820] / 100
                RASA = paddle.ones([bond_angle_feature.shape[0]]) * all_descriptor[i, 821]
                RPSA = paddle.ones([bond_angle_feature.shape[0]]) * all_descriptor[i, 822]
                MDEC = paddle.ones([bond_angle_feature.shape[0]]) * all_descriptor[i, 1568]
                MATS = paddle.ones([bond_angle_feature.shape[0]]) * all_descriptor[i, 457]
                
                bond_angle_feature = paddle.concat(
                    [bond_angle_feature.reshape([-1, 1]), TPSA.reshape([-1, 1])], axis=1
                )
                bond_angle_feature = paddle.concat([bond_angle_feature, RASA.reshape([-1, 1])], axis=1)
                bond_angle_feature = paddle.concat([bond_angle_feature, RPSA.reshape([-1, 1])], axis=1)
                bond_angle_feature = paddle.concat([bond_angle_feature, MDEC.reshape([-1, 1])], axis=1)
                bond_angle_feature = paddle.concat([bond_angle_feature, MATS.reshape([-1, 1])], axis=1)
            else:
                # 如果没有描述符，直接reshape
                bond_angle_feature = bond_angle_feature.reshape([-1, 1])
        
        # 创建Data对象
        data_atom_bond = Data(
            x=atom_feature,
            edge_index=edge_index,
            edge_attr=bond_feature,
            data_index=data_index_int,
        )
        
        data_bond_angle = Data(
            edge_index=bond_index,
            edge_attr=bond_angle_feature if bond_angle_feature.shape[0] > 0 else paddle.zeros([0, 1]),
            num_nodes=num_atoms,  # 键角图的节点数等于原子数
        )
        
        graph_atom_bond.append(data_atom_bond)
        graph_bond_angle.append(data_bond_angle)

    return graph_atom_bond, graph_bond_angle


def read_ir_spectra_by_ids(sample_path, index_all, max_peak=DEFAULT_MAX_PEAKS):
    """
    按需读取IR光谱文件
    
    Args:
        sample_path: str, IR光谱JSON文件目录
        index_all: list, 需要读取的文件ID列表
        max_peak: int, 最大峰数
    
    Returns:
        ir_final_list: list of dict, 包含峰值信息的字典列表
    """
    ir_final_list = []
    
    for fileid in tqdm(index_all, desc="Reading IR spectra by ID"):
        filepath = os.path.join(sample_path, f"{fileid}.json")
        
        try:
            with open(filepath, 'r') as f:
                raw_ir_info = json.load(f)
            
            ir_x = raw_ir_info['x']
            ir_y = raw_ir_info['y_40']
            
            from scipy.signal import find_peaks
            peaks_raw, _ = find_peaks(x=ir_y, height=0.1, distance=100)
            peaks_raw = peaks_raw.tolist()
            
            # 处理峰值（与原作完全一致）
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


def load_ir_meta_file(mode='100'):
    """
    加载预生成的IR元数据文件
    
    Args:
        mode: str, 可选 '100', '10000', 'all'
    
    Returns:
        dataset_all: list, 图特征数据
        smiles_all: list, SMILES列表
        index_all: list, 索引列表
    """
    valid_modes = {'100', '10000', 'all'}
    if mode not in valid_modes:
        warnings.warn(f"Invalid mode {mode}, using '100'")
        mode = '100'
    
    filename = f'ir_column_charity_{mode}.npy'
    
    if not os.path.exists(filename):
        # 尝试在dataset/IR目录下查找
        alt_path = os.path.join('dataset', 'IR', filename)
        if os.path.exists(alt_path):
            filename = alt_path
        else:
            raise FileNotFoundError(f"IR meta file {filename} not found")
    
    print(f"Loading IR meta file: {filename}")
    data = np.load(filename, allow_pickle=True).item()
    
    return data['dataset_all'], data['smiles_all'], data['index_all']


class IRDataset(Dataset):
    """
    IR光谱预测数据集类
    
    支持三种预加载模式：
    - '100': 100个样本的小数据集（默认，用于快速测试）
    - '10000': 1万个样本的中等数据集
    - 'all': 全部样本（可能很大）
    
    关键优化：按需读取JSON文件，只读取index_all中指定的文件！
    """
    
    _cache = {}
    
    @PlaceEnv(paddle.CPUPlace())
    def __init__(self,
                 path: str = "dataset/IR",
                 mode: str = '100',
                 use_geometry_enhanced: bool = True,
                 force_reload: bool = False,
                 cache: bool = True):
        
        self.path = path
        self.mode = mode
        self.use_geometry_enhanced = use_geometry_enhanced
        self.cache_enabled = cache
        
        cache_key = f"{path}_{mode}_{use_geometry_enhanced}"
        
        if cache and not force_reload and cache_key in self._cache:
            print(f"Loading IR dataset from cache: {mode}")
            cached_data = self._cache[cache_key]
            self.graph_atom_bond = cached_data['atom_bond']
            self.graph_bond_angle = cached_data['bond_angle']
            self.smiles_list = cached_data.get('smiles', [])
            return
        
        print(f"First-time loading IR dataset (mode={mode})...")
        
        # 1. 加载元数据文件（获取index_all）
        meta_path = os.path.join(path, f'ir_column_charity_{mode}.npy')
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"IR meta file {meta_path} not found")
        
        data = np.load(meta_path, allow_pickle=True).item()
        dataset_all = data['dataset_all']
        smiles_all = data['smiles_all']
        index_all = data['index_all']
        
        print(f"Loaded meta data: {len(index_all)} samples")
        
        # 2. 按需读取IR光谱（只读取index_all中的文件！）
        spectra_path = os.path.join(path, 'qm9_ir_spec')
        self.ir_sequences = read_ir_spectra_by_ids(spectra_path, index_all)
        
        print(f"Loaded {len(self.ir_sequences)} IR spectra")
        
        # 3. 构建图数据
        descriptor_path = os.path.join(path, 'descriptor_all_column.npy')
        if not os.path.exists(descriptor_path):
            descriptor_path = None
        
        total_graph_atom_bond, total_graph_bond_angle = Construct_IR_Dataset(
            dataset_all, index_all, descriptor_path
        )
        
        # 4. 将光谱信息附加到图数据
        self.graph_atom_bond = []
        self.graph_bond_angle = []
        self.smiles_list = []
        
        # 注意：ir_sequences已经按id排序，index_all也是按id排序的
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
        
        if cache:
            self._cache[cache_key] = {
                'atom_bond': self.graph_atom_bond,
                'bond_angle': self.graph_bond_angle,
                'smiles': self.smiles_list
            }
    def __len__(self):
        """返回数据集大小"""
        return len(self.graph_atom_bond)

    def __getitem__(self, idx):
        return (self.graph_atom_bond[idx], self.graph_bond_angle[idx])

# 自定义DataLoader
class IRDataLoader(DataLoader):
    """IR数据集专用DataLoader"""
    
    def __init__(self, dataset, batch_size=64, shuffle=True, num_workers=0, **kwargs):
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self._collate_fn,
            **kwargs
        )
    
    @staticmethod
    def _collate_fn(batch):
        """
        自定义collate函数
        
        Args:
            batch: list of (atom_bond, bond_angle, smiles) tuples
        
        Returns:
            batched_atom_bond: Batch
            batched_bond_angle: Batch
            smiles_list: list of str
        """
        from paddle_geometric.data import Batch
        
        atom_bond_list = [item[0] for item in batch]
        bond_angle_list = [item[1] for item in batch]
        #smiles_list = [item[2] for item in batch]
        
        batch_atom_bond = Batch.from_data_list(atom_bond_list)
        batch_bond_angle = Batch.from_data_list(bond_angle_list)
        # Data解包到Tensor字典
        x, edge_index, edge_attr, query_mask =batch_atom_bond.x,batch_atom_bond.edge_index,batch_atom_bond.edge_attr,batch_atom_bond.query_mask
        ba_edge_index, ba_edge_attr = batch_bond_angle.edge_index,batch_bond_angle.edge_attr
        batch_data = batch_atom_bond.batch
        pos_gt = batch_atom_bond.peak_position 
        height_gt = batch_atom_bond.peak_height
        num_gt = batch_atom_bond.peak_num      
        return \
        {
        "x"               : x             ,
        "edge_index"      : edge_index    ,
        "edge_attr"       : edge_attr     ,
        "batch_data"      : batch_data    ,
        "ba_edge_index"   : ba_edge_index ,
        "ba_edge_attr"    : ba_edge_attr  ,
        "query_mask"      : query_mask
        },     \
        {
        "peak_number_gt"  : num_gt        ,
        "peak_position_gt": pos_gt        ,
        "peak_height_gt"  : height_gt
        }
        return batched_atom_bond, batched_bond_angle

