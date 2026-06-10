# MACE-MP-0 训练数据集说明

## 数据集概述
MACE-MP-0 在 **Materials Project Trajectory（MPtrj）** 数据集上训练，该数据集包含 Materials Project 中约 10 年 DFT 计算的结构弛豫和静态计算轨迹。

## 数据集详情

| 项目 | 说明 |
|------|------|
| 名称 | Materials Project Trajectory（MPtrj） |
| 来源 | Materials Project |
| 规模 | **1,580,000+** 结构，覆盖 **146,000+** 种独特材料 |
| 元素覆盖 | **89 种元素**（H 至 Bi，含镧系） |
| 理论级别 | PBE + U（Perdew-Burke-Ernzerhof + Hubbard U 修正） |
| 标签类型 | 总能量（eV/atom）、原子力（eV/Å）、维里应力（kBar）、磁矩（如有） |
| 数据结构 | 晶体结构 + 能量/力/应力张量 |

## 下载方式

### 方式一：通过 Matbench Discovery 下载（推荐）
Matbench Discovery 项目已整理 MPtrj 数据集，提供直接下载链接：

- **官方页面**：https://matbench-discovery.materialsproject.org/data/mptrj
- **数据下载**：https://figshare.com/articles/dataset/MPTrj/23714600
- **训练集划分**：已按 95% / 2.5% / 2.5% 划分为训练/验证/测试集

```bash
# 使用 wget 下载
wget https://figshare.com/ndownloader/files/41619375 -O mptrj.zip

# 或使用 Python
import urllib.request
url = "https://figshare.com/ndownloader/files/41619375"
urllib.request.urlretrieve(url, "mptrj.zip")
```

### 方式二：通过 Materials Project API 自行收集
通过 `pymatgen` 和 `mp-api` 从 Materials Project 数据库获取原始数据：

```bash
pip install mp-api pymatgen
```

```python
from mp_api.client import MPRester
from pymatgen.core import Structure

with MPRester("YOUR_API_KEY") as mpr:
    # 获取包含轨迹计算的材料条目
    docs = mpr.materials.trajectory.search(
        fields=["structure", "energy_per_atom", "forces", "stress"]
    )
```

- **API 文档**：https://docs.materialsproject.org/
- **获取 API Key**：https://materialsproject.org/api

### 方式三：通过 MACE 官方仓库预处理脚本
MACE 官方仓库提供了从 Materials Project 原始数据构建训练集的脚本：

```bash
git clone https://github.com/ACEsuit/mace.git
cd mace/scripts

# 运行数据预处理脚本
python preprocess_mptrj.py \
    --mp_api_key YOUR_API_KEY \
    --output_dir ./mptrj_processed \
    --r_max 6.0
```

## 数据预处理

### 1. 格式转换
原始 MPtrj 数据为 pymatgen `Structure` 对象，需转换为训练框架支持的格式：

```python
from pymatgen.io.ase import AseAtomsAdaptor
from ase.io import write

# 将 pymatgen Structure 转为 ASE Atoms
atoms = AseAtomsAdaptor.get_atoms(structure)
atoms.info["energy"] = energy_per_atom * len(atoms)  # 总能量（eV）
atoms.arrays["forces"] = forces  # 力（eV/Å）
atoms.info["stress"] = stress  # 应力（kBar）

# 保存为 ASE 支持的格式
write("train.extxyz", atoms)
```

### 2. 数据划分
使用与原始 MACE-MP-0 一致的划分方式：

| 划分 | 比例 | 用途 |
|------|------|------|
| 训练集 | 95% | 模型训练 |
| 验证集 | 2.5% | 超参数调优和早停 |
| 测试集 | 2.5% | 最终评估和对齐精度 |

```python
import numpy as np
from sklearn.model_selection import train_test_split

# 按材料 ID 划分（确保同一材料的不同构型不会分散到不同集合）
train_ids, temp_ids = train_test_split(
    material_ids, test_size=0.05, random_state=42, shuffle=True
)
valid_ids, test_ids = train_test_split(
    temp_ids, test_size=0.5, random_state=42, shuffle=True
)
```

### 3. 数据过滤
原始 MPtrj 包含部分低质量数据，建议进行以下过滤：

```python
# 过滤条件（与原始 MACE-MP-0 一致）
filtered_data = [
    d for d in data
    if d["max_force"] < 50.0  # 最大原子力 < 50 eV/Å
    and d["energy_per_atom"] > -20.0  # 能量合理范围
    and len(d["structure"]) <= 500  # 单胞原子数 <= 500
]
```

## 数据存储格式

### 推荐格式：extended XYZ（.extxyz）
每行一个结构，包含能量、力、应力信息：

```
64
Lattice="10.0 0.0 0.0 0.0 10.0 0.0 0.0 0.0 10.0" Properties=species:S:1:pos:R:3:forces:R:3 energy=-456.789 stress="0.1 0.0 0.0 0.0 0.1 0.0 0.0 0.0 0.1" pbc="T T T"
Si 0.0 0.0 0.0 -0.001 0.002 -0.001
Si 2.5 0.0 0.0 0.001 -0.002 0.001
...
```

### 字段说明
| 字段 | 单位 | 说明 |
|------|------|------|
| energy | eV | 体系总能量 |
| forces | eV/Å | 每个原子的力矢量 |
| stress | kBar | Voigt 表示的应力张量（6 分量） |
| Lattice | Å | 晶胞参数矩阵 |
| pbc | bool | 周期性边界条件设置 |

## 数据验证

### 与原始 MACE-MP-0 训练集的一致性检查
| 检查项 | 原始值 | 验证方法 |
|--------|--------|----------|
| 结构总数 | ~1.58M | 统计处理后的文件数量 |
| 元素种类 | 89 | 遍历所有结构的元素种类 |
| 能量范围 | [-12, 5] eV/atom | 绘制能量分布直方图 |
| 最大力 | < 50 eV/Å | 检查力的大小分布 |
| 晶胞体积 | [10, 10000] Å³ | 过滤异常大/小晶胞 |

### 快速验证脚本
```bash
python scripts/validate_mptrj.py --data_path data/mptrj_processed
```

## 参考链接
- **Materials Project 官网**：https://materialsproject.org
- **MPtrj 数据页面**：https://matbench-discovery.materialsproject.org/data/mptrj
- **MACE 官方预处理脚本**：https://github.com/ACEsuit/mace/tree/main/scripts
- **原始论文数据集描述**：Batatia et al., arXiv:2401.00096, Section 2.1
