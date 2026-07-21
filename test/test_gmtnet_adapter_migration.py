from unittest.mock import patch

from pymatgen.core import Structure

from ppmat.models import GMTNetGraphConverter


def test_gmtnet_converter_mapping_passthrough():
    converter = GMTNetGraphConverter()
    mapping = {
        "graph": object(),
        "feature_mask": object(),
        "matrix_equal": object(),
    }
    assert converter.build_prediction_input(mapping) is mapping
    assert converter.build_prediction_input([mapping]) == [mapping]


def test_gmtnet_precision_cif_loader_arguments(tmp_path):
    converter = GMTNetGraphConverter()
    cif_path = tmp_path / "sample.cif"
    cif_path.write_text("data_test\n")
    with patch.object(Structure, "from_file", return_value=object()) as loader:
        assert converter.load_structure_from_cif(cif_path) is loader.return_value
    loader.assert_called_once_with(
        cif_path,
        primitive=False,
        sort=False,
        merge_tol=0.0,
        frac_tolerance=0.0,
    )
