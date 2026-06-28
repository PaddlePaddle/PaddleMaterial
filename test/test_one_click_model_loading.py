from __future__ import annotations

import ast
import re
from pathlib import Path

from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
INFGCN_CONFIG_DIR = ROOT / "electronic_structure/configs/infgcn"


def _models_module_ast():
    return ast.parse((ROOT / "ppmat/models/__init__.py").read_text())


def _literal_assign(name: str):
    for node in _models_module_ast().body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)
    raise AssertionError(f"{name} assignment not found")


def test_infgcn_and_diffnmr_are_registered_for_one_click_loading():
    registry = _literal_assign("MODEL_REGISTRY")
    config_registry = _literal_assign("MODEL_CONFIG_REGISTRY")

    for model_name in ["infgcn_qm9", "diffnmr_msdnmr_nless15"]:
        assert model_name in registry
        assert model_name in config_registry
        assert registry[model_name].startswith("https://paddle-org.bj.bcebos.com/")
        assert registry[model_name].endswith(".zip")
        assert (ROOT / config_registry[model_name]).exists()


def test_model_package_helpers_resolve_standard_zip_layout(tmp_path):
    from ppmat.models import get_model_config_path_from_package
    from ppmat.models import get_model_file_path_from_package

    cache_dir = tmp_path / "infgcn_qm9"
    package_dir = cache_dir / "infgcn_qm9"
    checkpoints_dir = package_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True)
    config_path = package_dir / "infgcn_qm9.yaml"
    weight_path = checkpoints_dir / "infgcn_qm9.pdparams"
    config_path.write_text("Model: {}\n")
    weight_path.write_bytes(b"fake")

    assert Path(
        get_model_config_path_from_package("infgcn_qm9", str(cache_dir))
    ) == config_path
    assert Path(
        get_model_file_path_from_package(cache_dir, "infgcn_qm9.pdparams")
    ) == weight_path


def test_infgcn_predict_cli_accepts_one_click_model_arguments():
    source = (ROOT / "electronic_structure/predict.py").read_text()

    assert '"--model_name"' in source
    assert '"--weights_name"' in source
    assert "build_model_from_name" in source


def test_electronic_structure_models_use_builtin_scatter():
    for relative_path in [
        "ppmat/models/infgcn/infgcn.py",
        "ppmat/models/mateno/mateno.py",
    ]:
        source = (ROOT / relative_path).read_text()
        assert "from paddle_scatter import scatter" not in source
        assert "from ppmat.utils.scatter import scatter" in source


def test_all_infgcn_configs_are_parseable_and_complete():
    config_paths = sorted(INFGCN_CONFIG_DIR.glob("*.yaml"))
    assert [path.name for path in config_paths] == [
        "infgcn_md17_benzene.yaml",
        "infgcn_md17_ethane.yaml",
        "infgcn_md17_ethanol.yaml",
        "infgcn_md17_malonaldehyde.yaml",
        "infgcn_md17_phenol.yaml",
        "infgcn_md17_resorcinol.yaml",
        "infgcn_mp.yaml",
        "infgcn_omol25_MC_5k_trimmed.yaml",
        "infgcn_qm9.yaml",
    ]

    required_model_params = {
        "n_atom_type",
        "num_radial",
        "num_spherical",
        "radial_embed_size",
        "radial_hidden_size",
        "cutoff",
        "grid_cutoff",
    }

    for config_path in config_paths:
        cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
        assert cfg["Model"]["__class_name__"] == "InfGCN", config_path.name
        assert required_model_params.issubset(
            cfg["Model"]["__init_params__"]
        ), config_path.name

        dataset_cfg = cfg["Dataset"]
        for split in ["train", "val", "test"]:
            split_cfg = dataset_cfg[split]
            dataset = split_cfg["dataset"]
            assert dataset["__class_name__"] in {
                "DensityDataset",
                "SmallDensityDataset",
            }, config_path.name
            assert "root" in dataset["__init_params__"], config_path.name
            assert "sampler" in split_cfg, config_path.name
            assert "loader" in split_cfg, config_path.name
            assert isinstance(
                split_cfg["loader"]["use_shared_memory"], bool
            ), config_path.name
            assert split_cfg["loader"]["collate_fn"] in {
                "DensityCollator",
                "DensityVoxelCollator",
            }, config_path.name


def test_infgcn_readme_commands_and_config_links_are_clean():
    readme_path = INFGCN_CONFIG_DIR / "README.md"
    readme = readme_path.read_text()

    assert "--model_name infgcn_qm9" in readme
    assert "--weights_name infgcn_qm9.pdparams" in readme
    assert "conda run" not in readme
    assert "/home/" not in readme
    assert ".pt" not in readme

    hrefs = re.findall(r'href="([^"]*configs/infgcn/[^"]+\.yaml)"', readme)
    assert hrefs
    for href in hrefs:
        assert (readme_path.parent / href).resolve().exists(), href


def test_diffnmr_sample_readme_documents_one_click_sample_command():
    readme = (
        ROOT / "spectrum_elucidation/configs/diffnmr/README.md"
    ).read_text()

    assert "--model_name='diffnmr_msdnmr_nless15'" in readme
    assert "--weights_name='DiffNMR_nless15_best.pdparams'" in readme
