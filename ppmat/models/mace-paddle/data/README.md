# Data Directory

## Data Format

MACE-Paddle supports the same data format as the original MACE repository. Data should be provided in XYZ format with additional energy and force information.

### Example XYZ file:

```
2
Energy=-10.0
O 0.0 0.0 0.0
H 0.0 0.0 1.0
```

## Data Placement

Place your training and test data in this directory. For example:

- `data/train.xyz` - Training data
- `data/test.xyz` - Test data
- `data/valid.xyz` - Validation data

## Sample Data

A small sample dataset is provided in `data/sample_data/` for testing purposes.

## Data Preparation

1. Convert your molecular dynamics trajectories to XYZ format
2. Include energy and force information in the file headers
3. Split your data into training, validation, and test sets
4. Place the files in the appropriate locations
