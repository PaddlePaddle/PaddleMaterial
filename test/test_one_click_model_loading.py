from __future__ import annotations

import ast
import importlib.util
import inspect
import re
import sys
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


def test_diffnmr_dataset_infos_can_skip_train_smiles():
    from ppmat.datasets.msd_nmr_dataset import MSDnmrinfos

    cfg = {
        "data_flag": "n<15",
        "build_graph_cfg": {"__init_params__": {"remove_h": True}},
        "load_train_smiles": False,
    }

    dataset_infos = MSDnmrinfos(
        dataloaders=SimpleNamespace(train_dataloader=None),
        cfg=cfg,
    )

    assert dataset_infos.train_smiles is None
    assert dataset_infos.atom_decoder == ["C", "N", "O", "F", "P", "S", "Cl", "Br", "I"]


def test_build_model_from_name_uses_package_config_discovery():
    import ppmat.models as models

    source = inspect.getsource(models.build_model_from_name)

    assert "path = osp.join(path, model_name)" in source
    assert "find_config_file_in_package(model_name, path)" in source
    assert 'config_path = osp.join(path, f"{model_name}.yaml")' not in source
    assert "os.listdir(path)" not in source


def test_infgcn_predict_uses_config_defaults_for_cli_options():
    source = (ROOT / "electronic_structure/predict.py").read_text()
    field_source = (ROOT / "ppmat/predictor/field_predictor.py").read_text()
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

    field_source = (ROOT / "ppmat/predictor/field_predictor.py").read_text()
    entry_source = (ROOT / "electronic_structure/predict.py").read_text()

    assert not (ROOT / "ppmat/predictor/field.py").exists()
    assert hasattr(predictor, "FieldPredictor")
    assert "class FieldPredictor" in field_source
    assert "from ppmat.predictor import FieldPredictor" in entry_source
    assert "from ppmat.predictor.field_predictor import apply_predict_config" in (
        entry_source
    )
    assert "from ppmat.models import MODEL_REGISTRY" not in entry_source
    assert "from ppmat.datasets import DensityDataset" not in entry_source


def test_field_predictor_reuses_base_and_keeps_helpers_outside_predictor():
    field_source = (ROOT / "ppmat/predictor/field_predictor.py").read_text()
    io_source = (ROOT / "ppmat/utils/io.py").read_text()
    visualization_source = (ROOT / "ppmat/utils/visualization.py").read_text()

    assert "from ppmat.predictor.base import BasePredictor" in field_source
    assert "class FieldPredictor(BasePredictor):" in field_source
    assert "self._load_model()" in field_source
    assert "def _load_model(self):" in field_source
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

    top_level_imports = "\n".join(
        line
        for line in visualization_source.splitlines()
        if line.startswith("import ") or line.startswith("from ")
    )
    assert "import imageio" in top_level_imports
    assert "import matplotlib.pyplot as plt" in top_level_imports
    assert "import networkx as nx" in top_level_imports
    assert "import plotly.graph_objects as go" in top_level_imports
    assert "import rdkit" in top_level_imports
    assert "def _rdkit_modules" not in visualization_source
    assert "def _matplotlib_pyplot" not in visualization_source
    assert "def _networkx" not in visualization_source
    assert "def _imageio" not in visualization_source

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
    sample_csv = ROOT / "spectrum_elucidation/configs/diffnmr/example/sample.csv"

    assert "--model_name='diffnmr_msdnmr_nless15'" in readme
    assert "--weights_name='DiffNMR_nless15_best.pdparams'" in readme
    assert "bundled one-row validation example" in readme
    assert "Sampler.data.dataset.__init_params__.path" in readme
    assert (
        "Sampler.data.dataset.__init_params__.path='./data/MSD_nmr/test.csv'" in readme
    )
    assert "### Sampling Sample" not in readme
    assert "--checkpoint_path='./checkpoints'" in readme
    assert sample_csv.exists()
    assert sample_csv.read_text().splitlines()[0] == "smiles,tokenized_input,atom_count"
    assert (
        sample_csv.read_text()
        .splitlines()[1]
        .startswith('CSc1ccc(C(C)C(=O)O)cc1F,"{""1HNMR"":')
    )


def test_diffnmr_package_sample_defaults_to_bundled_example():
    config = OmegaConf.to_container(
        OmegaConf.load(ROOT / "spectrum_elucidation/configs/diffnmr/DiffNMR.yaml"),
        resolve=False,
    )

    assert config["Sampler"]["name"] == "diffnmr"
    assert config["Sampler"]["retrival_database_path"] == (
        "./assets/retrival_database/"
        "msd_nmr_nless15_retrieval_molecular_representations.csv"
    )
    sampler_params = config["Sampler"]["data"]["dataset"]["__init_params__"]
    assert sampler_params["path"] == "./example/sample.csv"
    assert sampler_params["vocab_peakwidth_path"] == (
        "./assets/vocab/nless15/H1_statistic/delta_distribution.csv"
    )
    assert sampler_params["vocab_split_path"] == (
        "./assets/vocab/nless15/H1_statistic/split_type_distribution.csv"
    )
    assert sampler_params["cache_path"] == "./output/diffnmr_example_cache"
    assert sampler_params["overwrite"] is True
    assert config["Sampler"]["data"]["sampler"]["__init_params__"]["batch_size"] == 1
    assert config["Sampler"]["sample_batch_iters"] == 1
    assert config["Sampler"]["visual_num"] == 1
    assert config["Sampler"]["chains_to_save"] == 0


def test_molecular_sampler_entrypoint_keeps_diffnmr_imports_lazy():
    source = (ROOT / "ppmat/sampler/molecular_sampler.py").read_text()

    forbidden_snippets = [
        "import paddle",
        "from ppmat.datasets",
        "from ppmat.metrics",
        "from ppmat.models.diffnmr",
        "from ppmat.schedulers",
        "DiffNMRStreamingAdapter",
        "ExtraMolecularFeatures",
        "MolecularVisualization",
        "scheduling_diffnmr",
        "graphs_from_mol",
    ]
    for snippet in forbidden_snippets:
        assert snippet not in source
    assert "importlib.import_module" in source
    assert "diffnmr_sampler:DiffNMRSampler" in source
    assert "MODEL_NAME_TO_SAMPLER" in source


def test_molecular_sampler_module_source_load_does_not_load_diffnmr():
    sys.modules.pop("ppmat.models.diffnmr.diffnmr", None)
    sys.modules.pop("ppmat.sampler.diffnmr_sampler", None)

    module_path = ROOT / "ppmat/sampler/molecular_sampler.py"
    spec = importlib.util.spec_from_file_location(
        "molecular_sampler_under_test", module_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.SAMPLER_REGISTRY["diffnmr"].endswith("diffnmr_sampler:DiffNMRSampler")
    assert "ppmat.models.diffnmr.diffnmr" not in sys.modules
    assert "ppmat.sampler.diffnmr_sampler" not in sys.modules


def test_molecular_sampler_dispatches_diffnmr_config(tmp_path, monkeypatch):
    import ppmat.sampler.molecular_sampler as molecular_sampler

    class FakeSampler:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    def fake_import_module(module_name):
        assert module_name == "fake_sampler_module"
        return SimpleNamespace(FakeSampler=FakeSampler)

    monkeypatch.setattr(
        molecular_sampler,
        "SAMPLER_REGISTRY",
        {"diffnmr": "fake_sampler_module:FakeSampler"},
    )
    monkeypatch.setattr(
        molecular_sampler.importlib, "import_module", fake_import_module
    )

    config_path = tmp_path / "DiffNMR.yaml"
    checkpoint_path = tmp_path / "checkpoints"
    checkpoint_path.mkdir()
    config_path.write_text(
        "Model:\n" "  __class_name__: DiffNMR\n" "Sampler:\n" "  name: diffnmr\n"
    )

    sampler = molecular_sampler.MolecularSampler(
        config_path=str(config_path),
        checkpoint_path=str(checkpoint_path),
        config_overrides=["Sampler.name=diffnmr"],
    )

    assert isinstance(sampler, FakeSampler)
    assert sampler.kwargs["config_path"] == str(config_path)
    assert sampler.kwargs["checkpoint_path"] == str(checkpoint_path)
    assert sampler.kwargs["config_overrides"] == ["Sampler.name=diffnmr"]


def test_molecular_sampler_dispatches_registered_diffnmr_name(monkeypatch):
    import ppmat.sampler.molecular_sampler as molecular_sampler

    class FakeSampler:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(
        molecular_sampler,
        "SAMPLER_REGISTRY",
        {"diffnmr": "fake_sampler_module:FakeSampler"},
    )
    monkeypatch.setattr(
        molecular_sampler.importlib,
        "import_module",
        lambda module_name: SimpleNamespace(FakeSampler=FakeSampler),
    )

    sampler = molecular_sampler.MolecularSampler(model_name="diffnmr_msdnmr_nless15")

    assert isinstance(sampler, FakeSampler)
    assert sampler.kwargs["model_name"] == "diffnmr_msdnmr_nless15"


def test_molecular_sampler_updates_visualization_output_dir_for_save_path(tmp_path):
    from ppmat.sampler.diffnmr_sampler import DiffNMRSampler

    sampler = object.__new__(DiffNMRSampler)
    sampler.sample_config = {"data": {}}
    sampler.output_dir = "old_output"
    sampler.visualization_tools = SimpleNamespace(result_path="old_output/graph/")
    sampler.model = SimpleNamespace(eval=lambda: None)
    sampler.flag_retrival_sampling = False
    sampler.num_candidates = 1
    sampler.metric_dict_sample = {"Accuracy"}

    save_path = tmp_path / "sample_output"
    sampler.sample_epoch = lambda *args, **kwargs: {"Accuracy": 1.0}

    result = sampler.sample_by_dataloader(
        save_path=str(save_path),
        data_loader=[],
    )

    assert result == {"Accuracy": 1.0}
    assert sampler.output_dir == str(save_path)
    assert sampler.visualization_tools.result_path == str(save_path / "graph")


def test_molecular_sampler_compute_metric_reuses_sample_metrics(tmp_path):
    from ppmat.sampler.diffnmr_sampler import DiffNMRSampler

    sampler = object.__new__(DiffNMRSampler)
    sampler.sample_config = {}
    sampler.output_dir = "old_output"
    sampler.visualization_tools = None
    sampler.sample_by_dataloader = lambda save_path: {"Accuracy": 0.5}

    assert sampler.compute_metric(save_path=str(tmp_path)) == {"Accuracy": 0.5}
    assert sampler.output_dir == str(tmp_path)


def test_molecular_sampler_clamps_keep_chain_to_batch_size():
    import paddle

    from ppmat.sampler.diffnmr_sampler import DiffNMRSampler

    sampler = object.__new__(DiffNMRSampler)
    assert sampler._clamp_keep_chain(5, 1) == 1
    assert sampler._clamp_keep_chain(0, 1) == 0
    assert sampler._clamp_keep_chain(3, paddle.to_tensor([2, 2], dtype="int64")) == 2


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
    from ppmat.sampler.diffnmr_sampler import DiffNMRSampler

    package_dir = tmp_path / "diffnmr_msdnmr_nless15"
    package_ckpt_dir = package_dir / "checkpoints"
    package_assets_dir = package_dir / "assets"
    package_vocab_dir = package_assets_dir / "vocab/nless15/H1_statistic"
    package_retrieval_dir = package_assets_dir / "retrival_database"
    package_ckpt_dir.mkdir(parents=True)
    package_vocab_dir.mkdir(parents=True)
    package_retrieval_dir.mkdir(parents=True)
    package_weight = package_ckpt_dir / "DiffNMR_NMRNet_nless15_best.pdparams"
    package_vocab = package_vocab_dir / "delta_distribution.csv"
    package_retrieval = (
        package_retrieval_dir
        / "msd_nmr_nless15_retrieval_molecular_representations.csv"
    )
    package_weight.write_bytes(b"fake")
    package_vocab.write_text("Value,Count\n0.03,1\n")
    package_retrieval.write_text("smiles,mol_rep\n")

    package_config = {
        "Model": {
            "__init_params__": {
                "encoder_cfg": {
                    "pretrained_path": (
                        "./checkpoints/DiffNMR_NMRNet_nless15_best.pdparams"
                    )
                }
            }
        },
        "Dataset": {
            "train": {
                "dataset": {
                    "__init_params__": {
                        "vocab_peakwidth_path": (
                            "./assets/vocab/nless15/H1_statistic/"
                            "delta_distribution.csv"
                        ),
                    }
                }
            }
        },
        "Sampler": {
            "retrival_database_path": (
                "./assets/retrival_database/"
                "msd_nmr_nless15_retrieval_molecular_representations.csv"
            )
        },
    }

    DiffNMRSampler._resolve_package_paths(
        package_config,
        config_base_dir=str(package_dir),
        checkpoint_dir=None,
    )

    assert package_config["Model"]["__init_params__"]["encoder_cfg"][
        "pretrained_path"
    ] == str(package_weight)
    assert package_config["Dataset"]["train"]["dataset"]["__init_params__"][
        "vocab_peakwidth_path"
    ] == str(package_vocab)
    assert package_config["Sampler"]["retrival_database_path"] == str(package_retrieval)

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

    DiffNMRSampler._resolve_package_paths(
        custom_config,
        config_base_dir=str(tmp_path / "config_dir"),
        checkpoint_dir=str(custom_ckpt_dir),
    )

    assert custom_config["CLIP"]["__init_params__"]["graph_encoder"][
        "pretrained_model_path"
    ] == str(custom_weight)


def test_molecular_sampler_downloads_diffnmr_vocab_asset_when_package_missing(
    tmp_path, monkeypatch
):
    import ppmat.sampler.diffnmr_sampler as diffnmr_sampler
    from ppmat.sampler.diffnmr_sampler import DIFFNMR_VOCAB_URL
    from ppmat.sampler.diffnmr_sampler import DiffNMRSampler

    package_dir = tmp_path / "diffnmr_msdnmr_nless15"
    package_dir.mkdir()
    vocab_root = tmp_path / "downloaded_vocab"
    vocab_dir = vocab_root / "nless15/H1_statistic"
    vocab_dir.mkdir(parents=True)
    delta_path = vocab_dir / "delta_distribution.csv"
    split_path = vocab_dir / "split_type_distribution.csv"
    delta_path.write_text("Value,Count\n0.03,1\n")
    split_path.write_text("Type,Count\nt,1\n")
    calls = []

    def fake_get_assets_path_from_url(url):
        calls.append(url)
        return str(vocab_root)

    monkeypatch.setattr(
        diffnmr_sampler.download,
        "get_assets_path_from_url",
        fake_get_assets_path_from_url,
    )
    config = {
        "Sampler": {
            "data": {
                "dataset": {
                    "__init_params__": {
                        "vocab_peakwidth_path": (
                            "./assets/vocab/nless15/H1_statistic/"
                            "delta_distribution.csv"
                        ),
                        "vocab_split_path": (
                            "./assets/vocab/nless15/H1_statistic/"
                            "split_type_distribution.csv"
                        ),
                    }
                }
            }
        }
    }

    DiffNMRSampler._resolve_package_paths(
        config,
        config_base_dir=str(package_dir),
        checkpoint_dir=None,
    )

    params = config["Sampler"]["data"]["dataset"]["__init_params__"]
    assert params["vocab_peakwidth_path"] == str(delta_path)
    assert params["vocab_split_path"] == str(split_path)
    assert calls == [DIFFNMR_VOCAB_URL]


def test_molecular_sampler_downloads_diffnmr_retrieval_asset_when_package_missing(
    tmp_path, monkeypatch
):
    import ppmat.sampler.diffnmr_sampler as diffnmr_sampler
    from ppmat.sampler.diffnmr_sampler import DIFFNMR_RETRIEVAL_URLS
    from ppmat.sampler.diffnmr_sampler import DiffNMRSampler

    package_dir = tmp_path / "diffnmr_msdnmr_nless15"
    package_dir.mkdir()
    retrieval_root = tmp_path / "downloaded_retrieval"
    retrieval_dir = retrieval_root / "retrival_database"
    retrieval_dir.mkdir(parents=True)
    retrieval_path = (
        retrieval_dir / "msd_nmr_nless15_retrieval_molecular_representations.csv"
    )
    retrieval_path.write_text("smiles,molecularRep\nC,[0.0 1.0]\n")
    calls = []

    def fake_get_assets_path_from_url(url):
        calls.append(url)
        return str(retrieval_root)

    monkeypatch.setattr(
        diffnmr_sampler.download,
        "get_assets_path_from_url",
        fake_get_assets_path_from_url,
    )
    config = {
        "Sampler": {
            "retrival_database_path": (
                "./assets/retrival_database/"
                "msd_nmr_nless15_retrieval_molecular_representations.csv"
            )
        }
    }

    DiffNMRSampler._resolve_package_paths(
        config,
        config_base_dir=str(package_dir),
        checkpoint_dir=None,
    )

    assert config["Sampler"]["retrival_database_path"] == str(retrieval_path)
    assert calls == [DIFFNMR_RETRIEVAL_URLS[retrieval_path.name]]


def test_molecular_sampler_allows_zero_saved_chains(monkeypatch):
    import paddle

    import ppmat.sampler.diffnmr_sampler as diffnmr_sampler
    from ppmat.sampler.diffnmr_sampler import DiffNMRSampler

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
        diffnmr_sampler.scheduling_diffnmr,
        "sample_discrete_feature_noise",
        fake_noise,
    )
    monkeypatch.setattr(diffnmr_sampler.scheduling_diffnmr, "step", fake_step)

    sampler = object.__new__(DiffNMRSampler)
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
    sampler_source = (ROOT / "ppmat/sampler/diffnmr_sampler.py").read_text()

    assert "parse_known_args()" in source
    assert "config_overrides=dynamic_args" in source
    assert "config_overrides: Optional[List[str]] = None" in sampler_source
    assert "OmegaConf.merge(config, cli_config)" in sampler_source
    assert "_apply_package_support_files" not in sampler_source
    assert "_replace_with_package_file" not in sampler_source


def test_diffnmr_uses_molecular_sampler_from_sampler_package():
    source = (ROOT / "spectrum_elucidation/sample.py").read_text()
    sampler_path = ROOT / "ppmat/sampler/molecular_sampler.py"
    diffnmr_sampler_path = ROOT / "ppmat/sampler/diffnmr_sampler.py"
    legacy_sample_dir = ROOT / "ppmat/sample"

    assert sampler_path.exists()
    assert diffnmr_sampler_path.exists()
    assert not legacy_sample_dir.exists()
    assert "from ppmat.sampler import MolecularSampler" in source
    assert "class MolecularSampler" in sampler_path.read_text()
    assert "class DiffNMRSampler" in diffnmr_sampler_path.read_text()
