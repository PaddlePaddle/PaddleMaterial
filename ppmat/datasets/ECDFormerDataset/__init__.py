# __init__.py
"""
ECDFormer数据集加载模块
"""

import os
import numpy as np
import pandas as pd
import paddle
from paddle.io import Dataset
from paddle_geometric.data import Data

from .compound_tools import get_atom_feature_dims, get_bond_feature_dims
from .util_func import normalize_func
from .eval_func import get_sequence_peak
from .colored_tqdm import ColoredTqdm as tqdm
from .place_env import PlaceEnv
from .dataloader import ECDFormerDataset_DataLoader


# ----------------Commonly-used Parameters----------------
atom_id_names = [
    "atomic_num", "chiral_tag", "degree", "explicit_valence",
    "formal_charge", "hybridization", "implicit_valence",
    "is_aromatic", "total_numHs",
]
bond_id_names = ["bond_dir", "bond_type", "is_in_ring"]
full_atom_feature_dims = get_atom_feature_dims(atom_id_names)
full_bond_feature_dims = get_bond_feature_dims(bond_id_names)
bond_angle_float_names = ['bond_angle', 'TPSA', 'RASA', 'RPSA', 'MDEC', 'MATS']
column_specify={
    'ADH':[1,5,0,0],'ODH':[1,5,0,1],'IC':[0,5,1,2],'IA':[0,5,1,3],'OJH':[1,5,0,4],
    'ASH':[1,5,0,5],'IC3':[0,3,1,6],'IE':[0,5,1,7],'ID':[0,5,1,8],'OD3':[1,3,0,9],
    'IB':[0,5,1,10],'AD':[1,10,0,11],'AD3':[1,3,0,12],'IF':[0,5,1,13],'OD':[1,10,0,14],
    'AS':[1,10,0,15],'OJ3':[1,3,0,16],'IG':[0,5,1,17],'AZ':[1,10,0,18],'IAH':[0,5,1,19],
    'OJ':[1,10,0,20],'ICH':[0,5,1,21],'OZ3':[1,3,0,22],'IF3':[0,3,1,23],'IAU':[0,1.6,1,24]
}
bond_float_names = []


def get_key_padding_mask(tokens):
    """生成query padding mask"""
    key_padding_mask = paddle.zeros(tokens.shape)
    key_padding_mask[tokens == -1] = -paddle.inf
    return key_padding_mask


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
        bond_float_feature = paddle.to_tensor(data['bond_length'].astype(np.float32))
        bond_angle_feature = paddle.to_tensor(data['bond_angle'].astype(np.float32))
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


_cache = None

class ECDFormerDataset(Dataset):
    """
    ECDFormer数据集类
    返回 (atom_bond_graph, bond_angle_graph)
    """
    @PlaceEnv(paddle.CPUPlace())
    def __init__(self,
                 path: str = "dataset/ECD",
                 Use_geometry_enhanced: bool = True,
                 Use_column_info: bool = False):
        global _cache

        if _cache:
            self.graph_atom_bond, self.graph_bond_angle = _cache
            return
        
        # 保存参数
        self.path = path
        self.Use_geometry_enhanced = Use_geometry_enhanced
        self.Use_column_info = Use_column_info

        # 1. 加载npy文件
        print(f"Loading ECDFormer dataset from {path}")
        self.ecd_dataset = np.load(
            os.path.join(path, 'ecd_column_charity_new_smiles.npy'),
            allow_pickle=True
        ).tolist()
        
        # 2. 加载csv文件
        self.ecd_info = pd.read_csv(
            os.path.join(path, 'ecd_info.csv'),
            encoding='gbk'
        )

        # 3. 提取info列表和索引
        self.dataset_all = [item['info'] for item in self.ecd_dataset]
        self.index_all = self.ecd_info['Unnamed: 0'].values

        # 4. 构建手性对映射
        self.unnamed_idx_dict, self.hand_idx_dict, self.line_idx_dict = {}, {}, {}
        for i, itm in enumerate(self.ecd_dataset):
            self.line_idx_dict[i] = {
                'hand_id': itm['hand_id'],
                'unnamed_id': itm['id'],
                'smiles': itm['smiles']
            }
            
            if itm['id'] not in self.unnamed_idx_dict:
                self.unnamed_idx_dict[itm['id']] = {
                    'line_number': i,
                    'hand_id': itm['hand_id'],
                    'smiles': itm['smiles']
                }
            else:
                raise AssertionError(f"Duplicate unnamed id: {itm['id']}")
                
            if itm['hand_id'] not in self.hand_idx_dict:
                self.hand_idx_dict[itm['hand_id']] = []
            self.hand_idx_dict[itm['hand_id']].append({
                'line_number': i,
                'unnamed_id': itm['id'],
                'smiles': itm['smiles']
            })

        # 5. 构建图数据集（核心调用）
        self.graph_atom_bond, self.graph_bond_angle = GetAtomBondAngleDataset(
            sample_path=path,
            dataset_all=self.dataset_all,
            index_all=self.index_all,
            hand_idx_dict=self.hand_idx_dict,
            line_idx_dict=self.line_idx_dict
        )
        
        _cache = (self.graph_atom_bond, self.graph_bond_angle)
        assert len(self.graph_atom_bond) == len(self.graph_bond_angle), \
            "Mismatch between atom_bond and bond_angle graph lengths"

    def __len__(self):
        return len(self.graph_atom_bond)

    def __getitem__(self, idx):
        """
        返回:
            atom_bond_graph: paddle_geometric.data.Data
            bond_angle_graph: paddle_geometric.data.Data
        """
        return self.graph_atom_bond[idx], self.graph_bond_angle[idx]