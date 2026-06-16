# Dataset: MPtrj

MACE-MP-0 使用 **Materials Project Trajectory (MPtrj)** 数据集进行训练。

## Dataset Properties

| Property | Value |
|----------|-------|
| Name | Materials Project Trajectory |
| Size | ~1.58M structures, 146K+ materials |
| Elements | 89 elements (H to Bi) |
| Theory Level | PBE+U |
| Targets | energy, forces, stress |

## Integration

MPtrj 数据集已在 PaddleMaterials 套件内集成，训练时直接通过配置文件指定：

```yaml
dataset:
  name: MPtrjDataset
  train_split: train
  valid_split: valid
  test_split: test
  r_max: 6.0
```

## Data Format

| Target | Shape | Unit |
|--------|-------|------|
| energy | scalar | eV/atom |
| forces | [n_atoms, 3] | eV/Å |
| stress | [3, 3] | kBar |

## Split Ratio

| Split | Ratio |
|-------|-------|
| Train | 95% |
| Validation | 2.5% |
| Test | 2.5% |

## Source

- Materials Project: https://materialsproject.org
- Matbench Discovery: https://matbench-discovery.materialsproject.org
