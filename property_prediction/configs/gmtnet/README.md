# GMTNet dielectric prediction

This configuration integrates GMTNet into the PaddleMaterials public
property-prediction path. GMTNet predicts a `3 x 3` dielectric tensor from a
crystal structure.

## Data and training

Prepare the normalized GMTNet dielectric data with the repository tool:

```bash
python -m ppmat.datasets.prepare_gmtnet_dataset --help
```

The canonical split is packaged as
`ppmat.datasets.gmtnet_dielectric_split_seed32.json`. Use the public trainer
entry point and this configuration for training or evaluation; the former
standalone synthetic `train.py` smoke script is not a public GMTNet workflow.

```bash
python property_prediction/train.py \
  -c property_prediction/configs/gmtnet/config.yaml
```

GMTNet's inference path requires forward gradients internally. Keep
`Predict.eval_with_no_grad: false` for this model, even during CPU evaluation.

## Converted checkpoint prediction

```python
from property_prediction.predict import PropertyPredictor

predictor = PropertyPredictor(
    config_path="property_prediction/configs/gmtnet/config.yaml",
    checkpoint_path="property_prediction/configs/gmtnet/paddle_model.pdparams",
)
```

`predictor.from_structures(structure)` accepts one pymatgen `Structure`.
An ordered non-empty list of structures is also supported. Low-level GMTNet
Mapping inputs contain `graph`, `feature_mask`, and `matrix_equal` and can be
passed to `predictor.model.predict`.

## CIF contract

GMTNet CIF prediction uses a model-specific precision-preserving reader:

```python
Structure.from_file(
    path,
    primitive=False,
    sort=False,
    merge_tol=0.0,
    frac_tolerance=0.0,
)
```

No primitive/conventional-cell conversion, site sort, site merge, or parser
coordinate idealization is applied. The output tensor is expressed in the
Cartesian frame of the parsed CIF structure. A CIF cannot recover floating
point coordinates that were not encoded in its text.

For a CIF directory, all files are parsed before any prediction begins. A
malformed member therefore raises without producing partial directory
predictions. Existing directory enumeration order is retained and is not a
lexicographic-order guarantee.

## Known numerical limits

The float32 edge-order candidate and rotation canonicalization candidate were
audited but are not applied. Equivalent edge-array orders can produce small
float32 prediction differences, and symmetry-derived constraints remain
sensitive to numerically equivalent rotation representations. These limits do
not alter the precision-preserving CIF parsing contract.
