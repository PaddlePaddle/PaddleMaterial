# SGEQUI 数据目录说明

## 数据说明

### wyckoff_positions/
包含所有230个空间群的Wyckoff位置和非对称单元(ASU)几何信息。

**主要文件**: `clean_wyckoffs_in_asu_v6.json`
- **大小**: 234KB
- **格式**: JSON
- **内容**: 
  - 230个空间群的数据
  - 每个空间群的Wyckoff位置
  - ASU几何约束信息
  - 顶点、维度、体积等几何数据

### wyckoff_shape_decomposition.pkl
Wyckoff形状分解字典，用于采样时的均匀分布生成。

**主要文件**: `wyckoff_shape_decomposition.pkl`
- **大小**: 129KB
- **格式**: Python pickle
- **用途**: 在采样过程中均匀生成Wyckoff位置

## 配置方式

### 1. 自动检测（推荐）
代码会按以下优先级自动检测数据目录：
1. `ppmat/models/sgequidiff/resources/` (模块内位置，当前使用）
2. 相对于项目根目录的 `data/sgequidiff/` (向后兼容）

### 2. 环境变量
可以通过环境变量指定自定义数据目录：
```bash
export SGEQUI_DATA_DIR=/your/custom/path
```