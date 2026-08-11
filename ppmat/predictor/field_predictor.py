# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import copy
from collections.abc import Mapping
from functools import partial
from pathlib import Path
from typing import Optional
from typing import Sequence

import numpy as np
import paddle
from cvve import GridField
from cvve import GridSpec
from cvve import Structure
from pymatgen.core import Element
from tqdm import tqdm

import ppmat.datasets as datasets
from ppmat.datasets import DensityDataset
from ppmat.datasets import MD17DensityDataset
from ppmat.datasets.build_field import BuildField
from ppmat.datasets.build_molecule import BuildMolecule
from ppmat.models import build_graph_converter
from ppmat.predictor.base import BasePredictor
from ppmat.utils import logger
from ppmat.utils.crystal import atomic_number_from_symbol
from ppmat.utils.io import write_cube
from ppmat.utils.misc import set_random_seed
from ppmat.visualization import VolumeVisualizer


def to_numpy(value):
    return (
        value.detach().cpu().numpy()
        if isinstance(value, paddle.Tensor)
        else np.asarray(value)
    )


def read_cube_density(path, field_converter):
    field = BuildField(
        format="cube",
        name=field_converter.name,
        value_unit=field_converter.value_unit,
    )(path, validate_coordinate_unit=False)
    grid = field.grid
    structure = field.structure
    return (
        np.asarray(field.flat, dtype=np.float32),
        np.asarray(grid.cartesian_coordinates(), dtype=np.float32),
        {
            "shape": list(grid.shape),
            "cell": np.asarray(grid.cell_vectors, dtype=np.float32),
            "origin": np.asarray(grid.origin, dtype=np.float32),
            "atom_numbers": np.asarray(
                [atomic_number_from_symbol(symbol) for symbol in structure.symbols],
                dtype=np.int64,
            ),
            "atom_coord_ref": np.asarray(
                structure.cartesian_positions(), dtype=np.float32
            ),
            "coordinate_unit": grid.length_unit,
            "density_unit": grid.value_unit,
        },
    )


def write_cube_from_atom_types(
    destination, atom_type, atom_coord, density, info, idx2atom_num
):
    atom_numbers = np.asarray(
        [idx2atom_num[int(atom)] for atom in to_numpy(atom_type).reshape(-1)],
        dtype=np.int64,
    )
    write_cube(destination, atom_numbers, to_numpy(atom_coord), to_numpy(density), info)


def _as_paddle_tensor(value, dtype):
    if isinstance(value, paddle.Tensor):
        return value.astype(dtype)
    return paddle.to_tensor(np.asarray(value), dtype=dtype)


def _graph_node_feature(graph, name):
    node_feat = getattr(graph, "node_feat", None)
    if not isinstance(node_feat, Mapping) or name not in node_feat:
        raise KeyError(f"Field graph requires node_feat[{name!r}].")
    return node_feat[name]


def build_visualization_field(graph, grid_coord, values, info, name, atom_numbers):
    """Build the CVVE field consumed by the generic volume visualizer."""

    shape = tuple(int(size) for size in info["shape"])
    coordinates = to_numpy(grid_coord).reshape(*shape, 3)
    origin = coordinates[0, 0, 0]
    vectors = np.stack(
        [
            coordinates[1, 0, 0] - origin,
            coordinates[0, 1, 0] - origin,
            coordinates[0, 0, 1] - origin,
        ]
    )
    coordinate_unit = info.get("coordinate_unit") or "unknown"
    grid = GridSpec(
        shape=shape,
        origin=origin,
        vectors=vectors,
        length_unit=coordinate_unit,
        value_unit=info.get("density_unit") or "unknown",
    )
    atom_coord = to_numpy(_graph_node_feature(graph, "cart_coords"))
    structure = Structure(
        symbols=[Element.from_Z(int(number)).symbol for number in atom_numbers],
        positions=atom_coord,
        position_unit=coordinate_unit,
    )
    return GridField(
        data=to_numpy(values).reshape(shape),
        grid=grid,
        structure=structure,
        name=name,
        kind="density",
    )


def parse_grid_shape(grid_shape):
    """Normalize a scalar or three-dimensional grid shape."""

    if isinstance(grid_shape, Sequence) and not isinstance(grid_shape, str):
        parts = list(grid_shape)
    else:
        parts = [part.strip() for part in str(grid_shape).split(",") if part.strip()]
    if len(parts) == 1:
        n = int(parts[0])
        if n <= 1:
            raise ValueError(f"Invalid grid_shape {grid_shape}: dimensions must be > 1")
        return [n, n, n]
    if len(parts) == 3:
        shape = [int(part) for part in parts]
        if any(size <= 1 for size in shape):
            raise ValueError(f"Invalid grid_shape {grid_shape}: dimensions must be > 1")
        return shape
    raise ValueError(f"Invalid grid_shape {grid_shape}: expected N or Nx,Ny,Nz")


def collect_mol_files(mol_file_path):
    mol_path = Path(mol_file_path).expanduser()
    if mol_path.is_file():
        if mol_path.suffix.lower() != ".mol":
            raise ValueError(f"Expected a .mol file, but got: {mol_path}")
        return [mol_path]
    if not mol_path.is_dir():
        raise FileNotFoundError(f"MOL input path not found: {mol_path}")

    files = sorted(
        path
        for path in mol_path.iterdir()
        if path.is_file() and path.suffix.lower() == ".mol"
    )
    if not files:
        raise FileNotFoundError(f"No .mol files found in directory: {mol_path}")
    return files


def use_reference_atom_coordinates(
    graph,
    reference_info,
    idx2atom_num,
    sample_name,
    graph_converter,
):
    """Use verified CUBE atom coordinates for reference-grid inference."""

    reference_atom_numbers = np.asarray(reference_info.get("atom_numbers"))
    reference_atom_coord = np.asarray(
        reference_info.get("atom_coord_ref"), dtype=np.float32
    )
    expected_atom_numbers = np.asarray(
        [
            idx2atom_num[int(atom_type)]
            for atom_type in to_numpy(_graph_node_feature(graph, "x")).reshape(-1)
        ],
        dtype=np.int64,
    )
    if (
        reference_atom_numbers.shape != expected_atom_numbers.shape
        or not np.array_equal(reference_atom_numbers, expected_atom_numbers)
    ):
        raise ValueError(
            f"Atom order in the reference CUBE does not match {sample_name}: "
            f"expected {expected_atom_numbers.tolist()}, got "
            f"{reference_atom_numbers.tolist()}."
        )
    if reference_atom_coord.shape != (len(expected_atom_numbers), 3):
        raise ValueError(
            f"Invalid reference atom coordinates for {sample_name}: "
            f"{reference_atom_coord.shape}."
        )
    return graph_converter.from_arrays(
        expected_atom_numbers,
        reference_atom_coord,
        node_features={
            "x": to_numpy(_graph_node_feature(graph, "x")).reshape(-1),
        },
    )


def resolve_true_cube_for_mol(mol_path, reference_cube_dir=None):
    if reference_cube_dir is None:
        return None

    base = sanitize_base_name(mol_path.name)
    base_density = f"{base[:-3]}Density" if base.endswith("Opt") else f"{base}Density"
    root = Path(reference_cube_dir).expanduser()

    stems = [base, f"{base}_true", base_density]
    exts = [
        ".cube",
        ".cub",
        ".cube.lz4",
        ".cube.gz",
        ".cube.xz",
        ".cub.lz4",
        ".cub.gz",
        ".cub.xz",
    ]
    name_candidates = []
    for s in stems:
        for ext in exts:
            name_candidates.append(f"{s}{ext}")

    seen = set()
    uniq_candidates = []
    for name in name_candidates:
        if name not in seen:
            uniq_candidates.append(name)
            seen.add(name)

    if not root.is_dir():
        raise FileNotFoundError(f"Reference CUBE directory not found: {root}")
    for name in uniq_candidates:
        path = root / name
        if path.is_file():
            return path
    return None


def build_mol_sample(
    mol_path,
    atom_name2idx,
    grid_shape,
    grid_padding,
    field_converter: BuildField,
    graph_converter,
):
    molecule = BuildMolecule(format="mol_file", sanitize=False)(mol_path)
    if molecule is None:
        raise ValueError(f"RDKit failed to parse MOL file: {mol_path}")
    if molecule.GetNumConformers() == 0:
        raise ValueError(f"MOL file does not contain atom coordinates: {mol_path}")
    atom_coord_np = np.asarray(
        molecule.GetConformer().GetPositions(),
        dtype=np.float32,
    )
    atom_symbols = [atom.GetSymbol() for atom in molecule.GetAtoms()]
    atom_numbers = np.asarray(
        [atom.GetAtomicNum() for atom in molecule.GetAtoms()],
        dtype=np.int64,
    )
    # MOL conformer coordinates are consumed directly in angstrom.
    coordinate_unit = "angstrom"
    grid_padding = float(grid_padding)

    atom_type_idx = []
    missing = set()
    for sym in atom_symbols:
        idx = atom_name2idx.get(sym)
        if idx is None:
            missing.add(sym)
        else:
            atom_type_idx.append(idx)
    if missing:
        raise ValueError(
            "Found atoms not covered by the atom vocabulary in "
            f"{mol_path}: {sorted(missing)}"
        )

    g = graph_converter.from_arrays(
        atom_numbers,
        atom_coord_np,
        node_features={"x": np.asarray(atom_type_idx, dtype=np.int64)},
    )
    if g is None:
        raise ValueError(f"No radius edges were found for {mol_path}.")

    shape = [int(size) for size in grid_shape]
    min_coord = atom_coord_np.min(axis=0)
    max_coord = atom_coord_np.max(axis=0)
    span = np.maximum(
        max_coord - min_coord, np.array([1e-3, 1e-3, 1e-3], dtype=np.float32)
    )
    axis_len = span + 2.0 * float(grid_padding)
    center = 0.5 * (min_coord + max_coord)
    origin = center - 0.5 * axis_len

    grid = BuildField.build_grid_one(
        {
            "shape": shape,
            "voxel_vectors": np.diag(axis_len / np.asarray(shape, dtype=np.float32)),
            "origin": origin,
        },
        coordinate_unit,
    )
    info = {
        "shape": list(grid.shape),
        "cell": np.asarray(grid.cell_vectors, dtype=np.float32),
        "origin": np.asarray(grid.origin, dtype=np.float32),
        "file_name": mol_path.name,
        "coordinate_unit": grid.length_unit,
        "density_unit": field_converter.value_unit or "unknown",
    }

    return (
        g,
        None,
        np.asarray(grid.cartesian_coordinates(), dtype=np.float32),
        info,
    )


def sanitize_base_name(sample_name):
    base_name = Path(sample_name).name
    compression_suffixes = {".lz4", ".zst", ".gz", ".xz"}
    data_suffixes = {".cube", ".cub", ".chgcar", ".json", ".mol"}
    suffix = Path(base_name).suffix
    while suffix.lower() in compression_suffixes:
        base_name = base_name[: -len(suffix)]
        suffix = Path(base_name).suffix
    if suffix.lower() in data_suffixes:
        base_name = base_name[: -len(suffix)]
    return base_name


class FieldPredictor(BasePredictor):
    """Predict scalar fields on molecular or crystal grids.

    The predictor owns inference orchestration and output formatting. Models only
    need to follow the electronic-structure batch contract and return their field
    tensor through ``output["pred_dict"][target_name]``.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        weights_name: Optional[str] = None,
        config_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
        config_overrides: Optional[Sequence[str]] = None,
        seed: int = 42,
    ):
        super().__init__(
            model_name=model_name,
            weights_name=weights_name,
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            work_dir="",
            device=device,
            config_overrides=config_overrides,
        )
        set_random_seed(seed)
        self.load_inference_model()
        model_config = self.config.get("Model", {}).get("__init_params__", {})
        self.target_name = getattr(
            self.model, "target_name", model_config.get("target_name", "density")
        )
        self.field_converter = BuildField(
            **self._get_build_config("build_field_cfg"),
        )
        # Fill the base-class converter slot instead of shadowing its
        # ``graph_converter`` method: field configs name this
        # ``Predict.build_graph_cfg`` rather than ``Predict.graph_converter``.
        self.graph_converter_fn = build_graph_converter(
            self._get_build_config("build_graph_cfg"), vocab=self.vocab
        )
        if not hasattr(self.graph_converter_fn, "from_arrays"):
            raise TypeError(
                "Predict.build_graph_cfg must build an array-compatible graph "
                "converter."
            )
        model_cutoff = getattr(
            self.model,
            "atom_graph_cutoff",
            model_config.get("atom_graph_cutoff"),
        )
        if model_cutoff is not None and not np.isclose(
            self.graph_converter_fn.cutoff,
            float(model_cutoff),
        ):
            raise ValueError(
                "Predict build_graph_cfg cutoff must match the model "
                "atom_graph_cutoff."
            )
        if self.field_converter.name != self.target_name:
            raise ValueError(
                f"Predict.build_field_cfg.name is {self.field_converter.name!r}, "
                f"but the model target_name is {self.target_name!r}."
            )
        logger.info(f"Model loaded successfully on {self.device}.")

    def _get_build_config(self, name):
        build_config = self.predict_config.get(name)
        if build_config is None:
            raise KeyError(f"Predict.{name} is required for field prediction.")
        if not isinstance(build_config, Mapping):
            raise TypeError(f"Predict.{name} must be a mapping.")
        return dict(build_config)

    @staticmethod
    def _canonical_split(split: str) -> str:
        """Return the dataset's own split name, accepting the ``val`` alias."""

        if split == "val":
            split = "validation"
        if split not in {"train", "validation", "test"}:
            raise ValueError(
                f"Unsupported split '{split}'. Expected train, validation, or test."
            )
        return split

    def _resolve_grid_batch_size(self, grid_batch_size: Optional[int]) -> int:
        if grid_batch_size is None:
            grid_batch_size = self.predict_config.get("grid_batch_size", 4096)
        grid_batch_size = int(grid_batch_size)
        if grid_batch_size <= 0:
            raise ValueError("grid_batch_size must be a positive integer.")
        return grid_batch_size

    def _extract_prediction(self, output) -> paddle.Tensor:
        if not isinstance(output, Mapping):
            raise TypeError(
                "Field models must return a mapping containing a 'pred_dict'."
            )
        pred_dict = output.get("pred_dict")
        if not isinstance(pred_dict, Mapping):
            raise TypeError("Field model output['pred_dict'] must be a mapping.")
        if self.target_name not in pred_dict:
            raise KeyError(
                f"Prediction key '{self.target_name}' is missing from pred_dict; "
                f"available keys: {list(pred_dict)}."
            )
        prediction = pred_dict[self.target_name]
        if not isinstance(prediction, paddle.Tensor):
            raise TypeError(
                f"Prediction '{self.target_name}' must be a paddle.Tensor, "
                f"but got {type(prediction)}."
            )
        if len(prediction.shape) == 2 and prediction.shape[0] == 1:
            prediction = prediction.squeeze(0)
        if len(prediction.shape) != 1:
            raise ValueError(
                "Field prediction for one structure must have shape [num_grid] or "
                f"[1, num_grid], but got {list(prediction.shape)}."
            )
        return prediction

    def _predict_grid(
        self,
        graph,
        grid_coord: paddle.Tensor,
        info: dict,
        grid_batch_size: int,
    ) -> paddle.Tensor:
        num_grid = int(grid_coord.shape[1])
        if num_grid == 0:
            raise ValueError("grid_coord must contain at least one grid point.")
        starts = range(0, num_grid, grid_batch_size)
        if num_grid > grid_batch_size:
            starts = tqdm(
                starts,
                total=(num_grid + grid_batch_size - 1) // grid_batch_size,
                desc="Grid inference",
                leave=False,
            )

        predictions = []
        self.model.eval()
        with paddle.no_grad():
            for start in starts:
                grid = grid_coord[:, start : start + grid_batch_size]
                batch = {
                    "graph": graph,
                    "density_mask": None,
                    "grid_coord": grid,
                    "info": {
                        "cell": info["cell"].unsqueeze(0),
                    },
                }
                output = self.post_process(
                    self.model(
                        batch,
                        return_loss=False,
                        return_prediction=True,
                    )
                )
                prediction = self._extract_prediction(output)
                expected_size = int(grid.shape[1])
                if prediction.shape[0] != expected_size:
                    raise ValueError(
                        "Field model returned an unexpected number of grid values: "
                        f"expected {expected_size}, got {prediction.shape[0]}."
                    )
                predictions.append(prediction)
        return paddle.concat(predictions, axis=0)

    def from_data(
        self,
        graph,
        grid_coord,
        info,
        density=None,
        grid_batch_size: Optional[int] = None,
    ):
        """Predict one field sample already represented as graph and grid data.

        Args:
            graph: PGL graph with ``x`` and ``pos`` node features.
            grid_coord: Grid coordinates with shape ``[num_grid, 3]`` or
                ``[1, num_grid, 3]``.
            info: Metadata mapping containing the lattice ``cell`` and explicit
                ``coordinate_unit`` (``"angstrom"`` or ``"bohr"``).
            density: Optional reference density used only for metrics.
            grid_batch_size: Maximum grid points processed per forward pass.

        Returns:
            A dict containing the predicted field and, when ``density`` is given,
            the mean squared error and normalized mean absolute error (NMAE).
        """
        paddle.set_device(self.device)
        if not isinstance(info, Mapping):
            raise TypeError(f"info must be a mapping, but got {type(info)}.")
        info = dict(info)
        if "coordinate_unit" not in info:
            raise KeyError("Field prediction requires info['coordinate_unit'].")
        info["coordinate_unit"] = str(info["coordinate_unit"])
        if "density_unit" not in info:
            raise KeyError("Field prediction requires info['density_unit'].")
        density_unit = info["density_unit"]
        if not isinstance(density_unit, str) or not density_unit.strip():
            raise ValueError("info['density_unit'] must be a non-empty string.")
        density_unit = density_unit.strip()
        info["density_unit"] = density_unit
        if "cell" not in info:
            raise KeyError("Field prediction requires info['cell'].")
        info["cell"] = _as_paddle_tensor(info["cell"], "float32")
        if list(info["cell"].shape) != [3, 3]:
            raise ValueError(
                f"info['cell'] must have shape [3, 3], but got "
                f"{list(info['cell'].shape)}."
            )
        info["cell"] = info["cell"].to(self.device)

        grid_coord = _as_paddle_tensor(grid_coord, "float32")
        if len(grid_coord.shape) == 2:
            grid_coord = grid_coord.unsqueeze(0)
        elif len(grid_coord.shape) != 3 or grid_coord.shape[0] != 1:
            raise ValueError(
                "grid_coord must have shape [num_grid, 3] or [1, num_grid, 3], "
                f"but got {list(grid_coord.shape)}."
            )
        if grid_coord.shape[-1] != 3:
            raise ValueError("The last dimension of grid_coord must be 3.")

        atom_types = to_numpy(_graph_node_feature(graph, "x")).reshape(-1)
        atom_coord = to_numpy(_graph_node_feature(graph, "cart_coords"))
        if atom_types.shape[0] != atom_coord.shape[0] or atom_coord.ndim != 2:
            raise ValueError("Graph atom types and positions are inconsistent.")
        if atom_coord.shape[1] != 3:
            raise ValueError("Graph positions must have shape [num_atoms, 3].")
        grid_coord = grid_coord.to(self.device)
        grid_batch_size = self._resolve_grid_batch_size(grid_batch_size)
        prediction = self._predict_grid(
            graph,
            grid_coord,
            info,
            grid_batch_size,
        )

        prediction_cpu = prediction.detach().cpu()
        result_info = {
            key: value.detach().cpu() if isinstance(value, paddle.Tensor) else value
            for key, value in info.items()
        }
        result = {
            self.target_name: prediction_cpu,
            "grid_coord": grid_coord.squeeze(0).detach().cpu(),
            "info": result_info,
        }
        if density is not None:
            density = _as_paddle_tensor(density, prediction.dtype)
            density = density.reshape([-1]).to(self.device)
            if density.shape[0] != prediction.shape[0]:
                raise ValueError(
                    "Reference density and prediction must contain the same number "
                    f"of grid points, but got {density.shape[0]} and "
                    f"{prediction.shape[0]}."
                )
            difference = prediction - density
            result["reference_density"] = density.detach().cpu()
            result["loss"] = float(paddle.mean(difference**2))
            loss_eps = float(getattr(self.model, "loss_eps", 1e-8))
            denominator = paddle.sum(paddle.abs(density)) + loss_eps
            result["nmae"] = float(paddle.sum(paddle.abs(difference)) / denominator)
        return result

    def _dataset_config(self, split: str):
        split = self._canonical_split(split)
        split_key = "val" if split == "validation" else split
        dataset_config = copy.deepcopy(
            self.config.get("Dataset", {}).get(split_key, {}).get("dataset")
        )
        if dataset_config is None:
            raise KeyError(f"Dataset.{split_key}.dataset is not defined in config.")
        return split, dataset_config

    @staticmethod
    def _build_dataset(dataset_config, dataset_params):
        class_name = dataset_config.get("__class_name__", "DensityDataset")
        dataset_class = getattr(datasets, class_name, None)
        if not isinstance(dataset_class, type) or not issubclass(
            dataset_class,
            (DensityDataset, MD17DensityDataset),
        ):
            raise ValueError(f"Unsupported field dataset class: {class_name}")
        return dataset_class(**dataset_params)

    def _get_cube_writer(self, dataset):
        idx2atom_num = getattr(dataset, "idx2atom_num", None)
        if idx2atom_num is None:
            idx2atom_num = self.vocab["atom"]["id_to_atomic_number"]
        return partial(write_cube_from_atom_types, idx2atom_num=idx2atom_num)

    def from_dataset(
        self,
        split: str = "test",
        index: int = 0,
        save_path: Optional[str] = None,
        data_root: Optional[str] = None,
        split_file_path: Optional[str] = None,
        grid_batch_size: Optional[int] = None,
        save_true_cube: bool = False,
        visualize: bool = False,
        save_html: bool = False,
        show_plot: bool = False,
    ):
        """Predict one sample from a configured field dataset."""
        split, dataset_config = self._dataset_config(split)
        dataset_params = copy.deepcopy(dataset_config.get("__init_params__", {}))
        dataset_params["split"] = split
        dataset_class_name = dataset_config.get("__class_name__", "DensityDataset")
        if data_root is not None:
            configured_path = Path(dataset_params["path"])
            suffix_parts = (
                configured_path.parts[-2:]
                if dataset_class_name == "MD17DensityDataset"
                else configured_path.parts[-1:]
            )
            dataset_params["path"] = str(
                Path(data_root).expanduser().joinpath(*suffix_parts)
            )
        if split_file_path is not None:
            dataset_params["path"] = split_file_path
        # Configured train/eval sampling must not truncate a prediction grid.
        # Inference memory is bounded by grid_batch_size instead.
        dataset_params.pop("grid_sampler_cfg", None)
        dataset_params["vocab"] = self.vocab

        dataset = self._build_dataset(dataset_config, dataset_params)
        if index < 0 or index >= len(dataset):
            raise IndexError(
                f"Index {index} is outside dataset split '{split}' with "
                f"{len(dataset)} samples."
            )

        sample = dataset[index]
        graph = sample["graph"]
        density = sample["density"]
        grid_coord = sample["grid_coord"]
        info = sample["info"]
        sample_name = info.get("file_name", f"{split}_{index}")
        logger.info(f"Starting prediction for sample: {sample_name}")
        result = self.from_data(
            graph,
            grid_coord,
            info,
            density=density,
            grid_batch_size=grid_batch_size,
        )
        saved_paths = self._save_outputs(
            save_path=save_path,
            cube_writer=self._get_cube_writer(dataset),
            sample_name=sample_name,
            graph=graph,
            density=density,
            prediction=result[self.target_name],
            info=info,
            grid_coord=grid_coord,
            save_true_cube=save_true_cube,
            visualize=visualize,
            save_html=save_html,
            show_plot=show_plot,
        )
        if saved_paths:
            result["saved_paths"] = saved_paths
        self._log_metrics(sample_name, result)
        return result

    def from_mol_file(
        self,
        mol_file_path: str,
        save_path: Optional[str] = None,
        grid_shape="80,80,80",
        grid_padding: float = 6.0,
        reference_cube_dir: Optional[str] = None,
        grid_batch_size: Optional[int] = None,
        save_true_cube: bool = False,
        visualize: bool = False,
        save_html: bool = False,
        show_plot: bool = False,
    ):
        """Predict from one MOL file or every MOL file in a directory.

        ``save_path`` is an output directory. When provided, predicted CUBE files
        are always written; visualization flags add PNG or HTML files there.
        """
        input_path = Path(mol_file_path).expanduser()
        mol_files = collect_mol_files(input_path)
        grid_shape = parse_grid_shape(grid_shape)
        if grid_padding < 0:
            raise ValueError("grid_padding must be non-negative.")

        atom_name2idx = self.vocab["atom"]["token_to_id"]
        idx2atom_num = self.vocab["atom"]["id_to_atomic_number"]
        cube_writer = partial(write_cube_from_atom_types, idx2atom_num=idx2atom_num)

        logger.info(
            f"MOL inference: {len(mol_files)} file(s), "
            f"grid_shape={grid_shape}, grid_padding={grid_padding}."
        )
        sample_iter = (
            tqdm(mol_files, desc="MOL inference") if len(mol_files) > 1 else mol_files
        )
        results = []
        for mol_path in sample_iter:
            graph, density, grid_coord, info = build_mol_sample(
                mol_path,
                atom_name2idx,
                grid_shape,
                grid_padding,
                self.field_converter,
                self.graph_converter_fn,
            )
            reference_cube_path = resolve_true_cube_for_mol(
                mol_path,
                reference_cube_dir,
            )
            if reference_cube_path is not None:
                density, grid_coord, reference_info = read_cube_density(
                    reference_cube_path,
                    self.field_converter,
                )
                graph = use_reference_atom_coordinates(
                    graph,
                    reference_info,
                    idx2atom_num,
                    mol_path.name,
                    self.graph_converter_fn,
                )
                info = dict(reference_info)
                info["file_name"] = mol_path.name
                info["reference_cube_file"] = str(reference_cube_path)
                logger.info(
                    f"Using reference CUBE for {mol_path.name}: "
                    f"{reference_cube_path}"
                )
            elif reference_cube_dir is not None:
                logger.warning(f"No matching reference CUBE for {mol_path.name}.")

            sample_name = info.get("file_name", mol_path.name)
            result = self.from_data(
                graph,
                grid_coord,
                info,
                density=density,
                grid_batch_size=grid_batch_size,
            )
            saved_paths = self._save_outputs(
                save_path=save_path,
                cube_writer=cube_writer,
                sample_name=sample_name,
                graph=graph,
                density=density,
                prediction=result[self.target_name],
                info=info,
                grid_coord=grid_coord,
                save_true_cube=save_true_cube,
                visualize=visualize,
                save_html=save_html,
                show_plot=show_plot,
            )
            if saved_paths:
                result["saved_paths"] = saved_paths
            self._log_metrics(sample_name, result)
            results.append(result)

        return results[0] if input_path.is_file() else results

    @staticmethod
    def _log_metrics(sample_name, result):
        if "loss" in result:
            logger.info(
                f"Prediction completed for {sample_name}, "
                f"MSE: {float(result['loss']):.6f}, "
                f"NMAE: {float(result['nmae']):.6f}."
            )
        else:
            logger.info(
                f"Prediction completed for {sample_name} (no reference density)."
            )

    def _save_outputs(
        self,
        save_path,
        cube_writer,
        sample_name,
        graph,
        density,
        prediction,
        info,
        grid_coord,
        save_true_cube,
        visualize,
        save_html,
        show_plot,
    ):
        needs_output_path = save_true_cube or visualize or save_html or show_plot
        if save_path is None:
            if needs_output_path:
                raise ValueError(
                    "save_path is required when saving reference data or "
                    "visualizations."
                )
            return {}

        self._validate_output_grid(info, grid_coord, prediction, density)
        output_dir = Path(save_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        sample_tag = sanitize_base_name(sample_name)
        saved_paths = self._save_cubes(
            output_dir,
            cube_writer,
            sample_name,
            sample_tag,
            graph,
            density,
            prediction,
            info,
            grid_coord,
            save_true_cube,
        )
        if visualize or save_html or show_plot:
            saved_paths["visualizations"] = self._save_visualizations(
                output_dir,
                sample_tag,
                graph,
                density,
                prediction,
                info,
                grid_coord,
                visualize,
                save_html,
                show_plot,
            )
        return saved_paths

    @staticmethod
    def _validate_output_grid(info, grid_coord, prediction, density):
        shape = info.get("shape")
        if shape is None or len(shape) != 3:
            raise ValueError("Field output requires a three-dimensional grid shape.")
        shape = [int(size) for size in shape]
        expected_size = int(np.prod(shape))
        grid_size = int(grid_coord.shape[-2])
        prediction_size = int(prediction.shape[0])
        if grid_size != expected_size or prediction_size != expected_size:
            raise ValueError(
                "Grid shape, coordinates, and prediction size are inconsistent: "
                f"shape={shape} ({expected_size}), grid={grid_size}, "
                f"prediction={prediction_size}."
            )
        if density is not None and int(density.reshape([-1]).shape[0]) != expected_size:
            raise ValueError(
                "Reference density size is inconsistent with grid shape: "
                f"shape={shape} ({expected_size}), "
                f"density={density.reshape([-1]).shape[0]}."
            )

    def _save_cubes(
        self,
        output_dir,
        cube_writer,
        sample_name,
        sample_tag,
        graph,
        density,
        prediction,
        info,
        grid_coord,
        save_true_cube,
    ):
        atom_type = to_numpy(_graph_node_feature(graph, "x")).reshape(-1)
        atom_coord = to_numpy(_graph_node_feature(graph, "cart_coords"))
        cube_info = info
        saved_paths = {}

        prediction_path = output_dir / f"{sample_tag}_pred.cube"
        cube_writer(
            prediction_path,
            atom_type,
            atom_coord,
            to_numpy(prediction),
            cube_info,
        )
        saved_paths["prediction_cube"] = str(prediction_path)
        logger.info(f"Saved predicted density CUBE to: {prediction_path}")

        if save_true_cube:
            if density is None:
                logger.warning(
                    f"Skipping reference CUBE for {sample_name}: no reference "
                    "density is available."
                )
            else:
                reference_path = output_dir / f"{sample_tag}_true.cube"
                cube_writer(
                    reference_path,
                    atom_type,
                    atom_coord,
                    to_numpy(density),
                    cube_info,
                )
                saved_paths["reference_cube"] = str(reference_path)
                logger.info(f"Saved reference density CUBE to: {reference_path}")
        return saved_paths

    def _save_visualizations(
        self,
        output_dir,
        sample_tag,
        graph,
        density,
        prediction,
        info,
        grid_coord,
        visualize,
        save_html,
        show_plot,
    ):
        prediction = to_numpy(prediction)
        visualizer = VolumeVisualizer()
        atom_ids = to_numpy(_graph_node_feature(graph, "x")).reshape(-1)
        id_to_atomic_number = self.vocab["atom"]["id_to_atomic_number"]
        atom_numbers = np.asarray(
            [id_to_atomic_number[int(atom_id)] for atom_id in atom_ids]
        )
        fields = []
        if density is not None:
            reference = to_numpy(density)
            fields.extend(
                [
                    ("true_density", "DFT electron density", reference, 0.05, 3.5, 5),
                    (
                        "diff_density",
                        "Electron Density Difference",
                        reference - prediction,
                        -0.06,
                        0.06,
                        4,
                    ),
                ]
            )
        fields.append(
            (
                "pred_density",
                "Predicted Electron Density",
                prediction,
                0.05,
                3.5,
                5,
            )
        )

        saved_paths = []
        for suffix, title, values, isomin, isomax, surface_count in fields:
            field = build_visualization_field(
                graph,
                grid_coord,
                values,
                info,
                suffix,
                atom_numbers,
            )
            _, _, metadata = visualizer.render_arrays(field)
            if metadata["downsampled"]:
                logger.warning(
                    "Downsampled volume grid from "
                    f"{metadata['original_points']} to "
                    f"{metadata['rendered_points']} points for visualization "
                    f"(stride={metadata['stride']})."
                )
            figure = visualizer.render(
                field,
                isomin=isomin,
                isomax=isomax,
                surface_count=surface_count,
                title=title,
            )
            if visualize:
                image_path = output_dir / f"{sample_tag}_{suffix}.png"
                saved_paths.append(str(visualizer.save_png(figure, image_path)))
            if save_html:
                html_path = output_dir / f"{sample_tag}_{suffix}.html"
                saved_paths.append(str(visualizer.save_html(figure, html_path)))
            if show_plot:
                figure.show()
        return saved_paths
