# InfGCN field examples

These compressed files are unchanged samples from the published test splits used by
the InfGCN configurations.

| File | Dataset | Test entry | Predictor option |
|---|---|---:|---|
| `000002.CHGCAR.lz4` | QM9 ES | `1` (stored as index + 1) | `--input_format=chgcar` |
| `mp-1546.json.xz` | MP ES cubic | `mp-1546` | `--input_format=json` |
| `004437.cube.lz4` | OMol25 MC 5k | `4437` | `--input_format=cube` |

The source field values are reference data. Field prediction uses the structure and
native grid geometry stored in each file and writes its prediction as CUBE.
