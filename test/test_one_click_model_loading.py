from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
INFGCN_CONFIG_DIR = ROOT / "electronic_structure/configs/infgcn"
INFGCN_MODEL_NAMES = [
    "infgcn_md17_benzene",
    "infgcn_md17_ethane",
    "infgcn_md17_ethanol",
    "infgcn_md17_malonaldehyde",
    "infgcn_md17_phenol",
    "infgcn_md17_resorcinol",
    "infgcn_mp",
    "infgcn_omol25_mc_5k_trimmed",
    "infgcn_qm9",
]


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

    for model_name in [*INFGCN_MODEL_NAMES, "diffnmr_msdnmr_nless15"]:
        assert model_name in registry
        assert registry[model_name].startswith("https://paddle-org.bj.bcebos.com/")
        assert registry[model_name].endswith(".zip")
        assert registry[model_name].endswith(f"{model_name}.zip")


def test_models_init_keeps_one_click_surface_minimal():
    source = (ROOT / "ppmat/models/__init__.py").read_text()

    forbidden_names = [
        "MODEL_CONFIG_REGISTRY",
        "MODEL_SUPPORT_REGISTRY",
        "_repo_root",
        "_resolve_repo_path",
        "get_model_config_path_from_name",
        "get_model_package_path_from_name",
        "get_model_file_path_from_package",
        "get_model_config_path_from_package",
    ]
    for name in forbidden_names:
        assert name not in source


def test_model_package_helpers_resolve_standard_zip_layout(tmp_path):
    from ppmat.utils.io import find_config_file_in_package
    from ppmat.utils.io import find_file_in_package

    cache_dir = tmp_path / "infgcn_qm9"
    package_dir = cache_dir / "infgcn_qm9"
    checkpoints_dir = package_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True)
    config_path = package_dir / "infgcn_qm9.yaml"
    weight_path = checkpoints_dir / "best.pdparams"
    config_path.write_text("Model: {}\n")
    weight_path.write_bytes(b"fake")

    assert (
        Path(find_config_file_in_package("infgcn_qm9", str(cache_dir))) == config_path
    )
    assert Path(find_file_in_package(cache_dir, "best.pdparams")) == weight_path


def test_model_package_helpers_use_ppmat_logger():
    import ppmat.utils.io as io_utils

    find_file_source = inspect.getsource(io_utils.find_file_in_package)
    find_config_source = inspect.getsource(io_utils.find_config_file_in_package)

    assert "from ppmat.utils import logger" in (ROOT / "ppmat/utils/io.py").read_text()
    assert "logger." in find_file_source
    assert "logger." in find_config_source


def test_model_package_helpers_live_in_io_module():
    assert not (ROOT / "ppmat/utils/model_package.py").exists()


def test_diffnmr_train_smiles_uses_existing_datadir_cache(tmp_path):
    import numpy as np

    from ppmat.datasets.msd_nmr_dataset import get_train_smiles

    cache_dir = tmp_path / "msd_nmr_nless15_cache" / "train"
    cache_dir.mkdir(parents=True)
    smiles_path = cache_dir / "train_smiles_no_h.npy"
    expected_smiles = np.array(["CCO", "CO"])
    np.save(smiles_path, expected_smiles)

    cfg = {
        "datadir": str(tmp_path),
        "data_flag": "n<15",
        "build_graph_cfg": {"__init_params__": {"remove_h": True}},
    }
    dataset_infos = SimpleNamespace(atom_decoder=["C", "N", "O", "F"])

    train_smiles = get_train_smiles(cfg, dataloader=[], dataset_infos=dataset_infos)

    np.testing.assert_array_equal(train_smiles, expected_smiles)


def test_build_model_from_name_uses_package_config_discovery():
    import ppmat.models as models

    source = inspect.getsource(models.build_model_from_name)

    assert "path = osp.join(path, model_name)" in source
    assert "find_config_file_in_package(model_name, path)" in source
    assert 'config_path = osp.join(path, f"{model_name}.yaml")' not in source
    assert "os.listdir(path)" not in source


def test_infgcn_predict_uses_config_defaults_for_cli_options():
    source = (ROOT / "electronic_structure/predict.py").read_text()
    field_source = (ROOT / "ppmat/predictor/field.py").read_text()
    cfg = OmegaConf.to_container(
        OmegaConf.load(INFGCN_CONFIG_DIR / "infgcn_qm9.yaml"),
        resolve=True,
    )

    assert "FieldPredictor" in source
    assert "def apply_predict_config" in field_source
    assert cfg["Predict"]["grid_batch_size"] == 20000
    assert cfg["Predict"]["output_dir"] == "output/infgcn_qm9/vis_val0"
    assert cfg["Predict"]["save_pred_cube"] is True
    assert cfg["Predict"]["save_true_cube"] is True
    assert cfg["Predict"]["cube_dir"] == "output/infgcn_qm9/cubes"


def test_infgcn_predict_config_fills_unset_cli_options():
    from electronic_structure.predict import apply_predict_config

    args = SimpleNamespace(
        split=None,
        index=None,
        data_root=None,
        split_file=None,
        atom_file=None,
        output_dir=None,
        grid_batch_size=123,
        skip_vis=None,
        save_true_cube=None,
        save_pred_cube=None,
        save_html=None,
        cube_dir=None,
        show_plot=None,
        mol_pattern=None,
        mol_grid_shape=None,
        mol_grid_padding=None,
        mol_true_cube_dir=None,
    )
    cfg = {
        "Predict": {
            "split": "validation",
            "index": 3,
            "output_dir": "from_config",
            "grid_batch_size": 456,
            "save_pred_cube": True,
        }
    }

    apply_predict_config(args, cfg)

    assert args.split == "validation"
    assert args.index == 3
    assert args.output_dir == "from_config"
    assert args.grid_batch_size == 123
    assert args.save_pred_cube is True
    assert args.save_true_cube is False


def test_infgcn_predict_cli_accepts_one_click_model_arguments():
    source = (ROOT / "electronic_structure/predict.py").read_text()

    assert '"--model_name"' in source
    assert '"--weights_name"' in source
    assert "FieldPredictor(" in source


def test_field_predictor_is_shared_predictor_entrypoint():
    import ppmat.predictor as predictor

    field_source = (ROOT / "ppmat/predictor/field.py").read_text()
    entry_source = (ROOT / "electronic_structure/predict.py").read_text()

    assert hasattr(predictor, "FieldPredictor")
    assert "class FieldPredictor" in field_source
    assert "from ppmat.predictor import FieldPredictor" in entry_source
    assert "from ppmat.models import MODEL_REGISTRY" not in entry_source
    assert "from ppmat.datasets import DensityDataset" not in entry_source


def test_field_predictor_reuses_base_and_keeps_helpers_outside_predictor():
    field_source = (ROOT / "ppmat/predictor/field.py").read_text()
    io_source = (ROOT / "ppmat/utils/io.py").read_text()
    visualization_source = (ROOT / "ppmat/utils/visualization.py").read_text()

    assert "from ppmat.predictor.base import BasePredictor" in field_source
    assert "class FieldPredictor(BasePredictor):" in field_source
    assert not (ROOT / "ppmat/utils/field_io.py").exists()
    assert not (ROOT / "ppmat/utils/field_visualization.py").exists()

    for helper_name in [
        "draw_volume",
        "safe_write_image",
        "maybe_downsample_volume",
        "read_cube_density",
        "write_cube_generic",
        "prepare_info_cube",
    ]:
        assert f"def {helper_name}" not in field_source

    for helper_name in ["read_cube_density", "write_cube_generic", "prepare_info_cube"]:
        assert f"def {helper_name}" in io_source

    for helper_name in ["draw_volume", "safe_write_image", "maybe_downsample_volume"]:
        assert f"def {helper_name}" in visualization_source

    top_level_vis_imports = "\n".join(
        line
        for line in visualization_source.splitlines()
        if line.startswith("import ") or line.startswith("from ")
    )
    assert "import matplotlib.pyplot as plt" not in top_level_vis_imports
    assert "import imageio" not in top_level_vis_imports
    assert "import rdkit" not in top_level_vis_imports
    assert "from rdkit" not in top_level_vis_imports

    assert "def _save_cubes" in field_source
    assert "def _save_visualizations" in field_source
    assert "FieldPredictor._save_cubes(" in field_source
    assert "FieldPredictor._save_visualizations(" in field_source


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

        predict_cfg = cfg["Predict"]
        for key in [
            "split",
            "index",
            "output_dir",
            "grid_batch_size",
            "save_true_cube",
            "save_pred_cube",
            "save_html",
            "cube_dir",
            "mol_grid_shape",
            "mol_grid_padding",
        ]:
            assert key in predict_cfg, config_path.name


def test_infgcn_readme_commands_and_config_links_are_clean():
    readme_path = INFGCN_CONFIG_DIR / "README.md"
    readme = readme_path.read_text()

    assert "--model_name infgcn_qm9" in readme
    assert "--weights_name best.pdparams" in readme
    assert "conda run" not in readme
    assert "/home/" not in readme
    assert ".pt" not in readme
    assert "_t_2026" not in readme
    assert "_s_42.zip" not in readme

    hrefs = re.findall(r'href="([^"]*configs/infgcn/[^"]+\.yaml)"', readme)
    assert hrefs
    for href in hrefs:
        assert (readme_path.parent / href).resolve().exists(), href


def test_diffnmr_sample_readme_documents_one_click_sample_command():
    readme = (ROOT / "spectrum_elucidation/configs/diffnmr/README.md").read_text()
    sample_csv = ROOT / "spectrum_elucidation/configs/diffnmr/sample.csv"

    assert "--model_name='diffnmr_msdnmr_nless15'" in readme
    assert "--weights_name='DiffNMR_nless15_best.pdparams'" in readme
    assert "data/MSD_nmr/test.csv" in readme
    assert "### Sampling Sample" not in readme
    assert "Sampler.sample_batch_iters=1" not in readme
    assert "Sampler.data.sampler.__init_params__.batch_size=1" not in readme
    assert "--checkpoint_path='./checkpoints'" in readme
    assert sample_csv.exists()
    assert sample_csv.read_text().splitlines()[0] == "smiles,tokenized_input,atom_count"


def test_diffnmr_config_uses_standard_checkpoint_paths():
    config_path = ROOT / "spectrum_elucidation/configs/diffnmr/DiffNMR.yaml"
    source = config_path.read_text()
    cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)

    assert "./pretrained/" not in source
    assert cfg["Sampler"]["pretrained_model_path"].startswith("./checkpoints/")
    assert cfg["Model"]["__init_params__"]["encoder_cfg"]["pretrained_path"].startswith(
        "./checkpoints/"
    )
    assert cfg["Model"]["__init_params__"]["decoder_cfg"]["pretrained_path"].startswith(
        "./checkpoints/"
    )
    assert cfg["CLIP"]["__init_params__"]["spectrum_encoder"][
        "pretrained_model_path"
    ].startswith("./checkpoints/")
    assert cfg["CLIP"]["__init_params__"]["graph_encoder"][
        "pretrained_model_path"
    ].startswith("./checkpoints/")


def test_molecular_sampler_resolves_diffnmr_checkpoint_paths(tmp_path):
    from ppmat.sample.molecular_sampler import MolecularSampler

    package_dir = tmp_path / "diffnmr_msdnmr_nless15"
    package_ckpt_dir = package_dir / "checkpoints"
    package_ckpt_dir.mkdir(parents=True)
    package_weight = package_ckpt_dir / "DiffNMR_NMRNet_nless15_best.pdparams"
    package_weight.write_bytes(b"fake")

    package_config = {
        "Model": {
            "__init_params__": {
                "encoder_cfg": {
                    "pretrained_path": (
                        "./checkpoints/DiffNMR_NMRNet_nless15_best.pdparams"
                    )
                }
            }
        }
    }

    MolecularSampler._resolve_pretrained_paths(
        package_config,
        config_base_dir=str(package_dir),
        checkpoint_dir=None,
    )

    assert package_config["Model"]["__init_params__"]["encoder_cfg"][
        "pretrained_path"
    ] == str(package_weight)

    custom_ckpt_dir = tmp_path / "custom_checkpoints"
    custom_ckpt_dir.mkdir()
    custom_weight = custom_ckpt_dir / "DiffNMR_DiffGraphFormer_nless15_best.pdparams"
    custom_weight.write_bytes(b"fake")
    custom_config = {
        "CLIP": {
            "__init_params__": {
                "graph_encoder": {
                    "pretrained_model_path": (
                        "./checkpoints/DiffNMR_DiffGraphFormer_nless15_best.pdparams"
                    )
                }
            }
        }
    }

    MolecularSampler._resolve_pretrained_paths(
        custom_config,
        config_base_dir=str(tmp_path / "config_dir"),
        checkpoint_dir=str(custom_ckpt_dir),
    )

    assert custom_config["CLIP"]["__init_params__"]["graph_encoder"][
        "pretrained_model_path"
    ] == str(custom_weight)


def test_molecular_sampler_allows_zero_saved_chains(monkeypatch):
    import paddle

    import ppmat.sample.molecular_sampler as molecular_sampler
    from ppmat.sample.molecular_sampler import MolecularSampler

    class FakeData:
        def __init__(self, X, E, y=None):
            self.X = X
            self.E = E
            self.y = y

        def mask(self, node_mask, collapse=False):
            if collapse:
                return FakeData(
                    paddle.argmax(self.X, axis=-1),
                    paddle.argmax(self.E, axis=-1),
                    self.y,
                )
            return self

    class FakeModel:
        T = 1
        limit_dist = None

    def fake_noise(limit_dist, node_mask):
        del limit_dist
        batch_size, n_max = node_mask.shape
        return FakeData(
            paddle.ones([batch_size, n_max, 1], dtype="float32"),
            paddle.ones([batch_size, n_max, n_max, 1], dtype="float32"),
            paddle.zeros([batch_size, 1], dtype="float32"),
        )

    def fake_step(model, **kwargs):
        del model
        batch_size = kwargs["X_t"].shape[0]
        n_max = kwargs["X_t"].shape[1]
        sampled = FakeData(
            paddle.ones([batch_size, n_max, 1], dtype="float32"),
            paddle.ones([batch_size, n_max, n_max, 1], dtype="float32"),
            paddle.zeros([batch_size, 1], dtype="float32"),
        )
        discrete = FakeData(
            paddle.zeros([batch_size, n_max], dtype="int64"),
            paddle.zeros([batch_size, n_max, n_max], dtype="int64"),
        )
        return sampled, discrete

    monkeypatch.setattr(
        molecular_sampler.scheduling_diffnmr,
        "sample_discrete_feature_noise",
        fake_noise,
    )
    monkeypatch.setattr(molecular_sampler.scheduling_diffnmr, "step", fake_step)

    sampler = object.__new__(MolecularSampler)
    sampler.visualization_tools = None

    mol_list, mol_true = sampler.sample_batch(
        model=FakeModel(),
        batch_id=0,
        batch_size=1,
        batch_condition=[],
        number_chain_steps=1,
        keep_chain=0,
        visual_num=0,
        batch_X=paddle.ones([1, 1, 1], dtype="float32"),
        batch_E=paddle.ones([1, 1, 1, 1], dtype="float32"),
        batch_y=paddle.zeros([1, 1], dtype="float32"),
        iter_idx=0,
        num_nodes=paddle.to_tensor([1], dtype="int64"),
    )

    assert len(mol_list) == 1
    assert len(mol_true) == 1


def test_diffnmr_sample_entrypoint_supports_config_overrides():
    source = (ROOT / "spectrum_elucidation/sample.py").read_text()
    sampler_source = (ROOT / "ppmat/sample/molecular_sampler.py").read_text()

    assert "parse_known_args()" in source
    assert "config_overrides=dynamic_args" in source
    assert "config_overrides: Optional[List[str]] = None" in sampler_source
    assert "OmegaConf.merge(config, cli_config)" in sampler_source
    assert "_apply_package_support_files" not in sampler_source
    assert "_replace_with_package_file" not in sampler_source


def test_diffnmr_uses_molecular_sampler_from_sample_package():
    source = (ROOT / "spectrum_elucidation/sample.py").read_text()
    sampler_path = ROOT / "ppmat/sample/molecular_sampler.py"
    legacy_source = (ROOT / "ppmat/sampler/base_sampler.py").read_text()

    assert sampler_path.exists()
    assert "from ppmat.sample import MolecularSampler" in source
    assert "class MolecularSampler" in sampler_path.read_text()
    assert "class MolecularSampler" not in legacy_source
