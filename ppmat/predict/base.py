import os
import os.path as osp
from typing import Optional

from omegaconf import OmegaConf
from pymatgen.core import Structure
from tqdm import tqdm

from ppmat.datasets.transform import build_post_transforms
from ppmat.models import build_graph_converter
from ppmat.models import build_model
from ppmat.models import build_model_from_name
from ppmat.utils import logger
from ppmat.utils import save_load


class PPMatPredictor:
    """PPMaterial predictor.

    This class provides an interface for predicting properties of crystalline
    structures using pre-trained deep learning models.

    Supports two initialization modes:

    1. **Automatic Model Loading**
       Specify `model_name` and `weights_name` to automatically download
       and load pre-trained weights from the `MODEL_REGISTRY`.

    2. **Custom Model Loading**
       Provide explicit `config_path` and `checkpoint_path` to load
       custom-trained models from local files.

    Args:
        model_name (Optional[str], optional):
            Name of the pre-defined model architecture
            from the `MODEL_REGISTRY` registry.
            When specified, associated weights
            will be automatically downloaded. Defaults to None.

        weights_name (Optional[str], optional):
            Specific pre-trained weight identifier.
            Used only when `model_name` is provided. Valid options include:
            - 'best.pdparams' (highest validation performance)
            - 'latest.pdparams' (most recent training checkpoint)
            - Custom weight files ending with '.pdparams'
            Defaults to None.

        config_path (Optional[str], optional):
            Path to model configuration file (YAML)
            for custom models. Required when not using predefined `model_name`.
            Defaults to None.

        checkpoint_path (Optional[str], optional):
            Path to model checkpoint file
            (.pdparams) for custom models. Required when not using predefined
            `model_name`. Defaults to None.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        weights_name: Optional[str] = None,
        config_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
    ):

        self.model_name = model_name
        self.weights_name = weights_name
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path

    def load_inference_model(
        self,
        ase_calc: bool = False,
        device: Optional[str] = "cpu",
    ):
        # if model_name is not None,
        # then config_path and checkpoint_path must be provided
        if self.model_name is None:
            assert self.config_path is not None and self.checkpoint_path is not None, (
                "config_path and checkpoint_path must be provided "
                "when model_name is None."
            )
            logger.info(
                f"Loading configuration from {self.config_path} "
                f"and model from {self.checkpoint_path}."
            )

            config = OmegaConf.load(self.config_path)
            config = OmegaConf.to_container(config, resolve=True)

            model_config = config.get("Model", None)
            assert model_config is not None, "Model config must be provided."
            # TODO: support more models
            if ase_calc:
                if model_config["__class_name__"] == "CHGNet":
                    # CHGNet by default predicts energy per atom;
                    # convert it to total energy
                    model_config["__init_params__"]["is_intensive"] = False
                    logger.warning(
                        "CHGNet by default predicts energy per atom; "
                        "change 'is_intensive' to False to "
                        "predict total energy for ASE integration."
                    )
                else:
                    raise NotImplementedError(
                        f"The model '{model_config.get('__class_name__')}' "
                        f"is not yet supported with ASE integration.\n"
                        f"Please ensure that the model predicts total energy, "
                        f"or manually adjust parameter according to the model.\n"
                        f"If this model should be supported, "
                        f"please add a special handling case here."
                    )
            model = build_model(model_config)
            save_load.load_pretrain(model, self.checkpoint_path)
        else:
            logger.info("Since model_name is given, downloading it...")
            model, config = build_model_from_name(self.model_name, self.weights_name)

        self.model = model
        self.config = config

        self.model.eval()

        self.predict_config = config.get("Predict", None)
        self.eval_with_no_grad = self.predict_config.get("eval_with_no_grad", True)

        if self.predict_config is not None:
            graph_converter_config = self.predict_config.get("graph_converter", None)
            if graph_converter_config is not None:
                self.graph_converter_fn = build_graph_converter(graph_converter_config)
        else:
            self.graph_converter_fn = None

        self.post_transforms_cfg = self.predict_config.get("post_transforms", None)
        if self.post_transforms_cfg is not None:
            self.post_transforms = build_post_transforms(self.post_transforms_cfg)
        else:
            self.post_transforms = None

    def collect_structures(
        self,
        file_path: str,
    ):
        """
        Supported formats include:
            CIF, POSCAR/CONTCAR, CHGCAR, LOCPOT, vasprun.xml, CSSR,
            Netcdf and pymatgen's JSON-serialized structures.

        Args:
            file_path (str):
                The path of the input file or directory.
        """
        if osp.isdir(file_path):
            all_files = [
                osp.join(file_path, f)
                for f in os.listdir(file_path)
                if f.endswith(".cif")
            ]
        else:
            all_files = [file_path]
        logger.info(f"Load {len(all_files)} structures from {file_path}")

        # Read raw file
        structures = []
        for file in tqdm(all_files):
            try:
                # read file by pymatgen package
                structure = Structure.from_file(file)
                structures.append(structure)
            except Exception as e:
                logger.warning("Error reading file: {}, skip it.\n{}".format(file, e))
        logger.info("Successfully read raw files and convert to pymatgen format")
        return all_files, structures

    def graph_converter(self, structure):
        if self.graph_converter_fn is None:
            return structure
        return self.graph_converter_fn(structure)

    def post_process(self, data):
        if self.post_transforms is None:
            return data
        return self.post_transforms(data)

    def init_save_dir(
        self,
        file_path: Optional[str] = None,
        save_path: Optional[str] = None,
    ):
        if save_path is not None:
            self.save_path = save_path
        else:
            # The save_path is not provided,
            # infer save_path from file_path.
            if file_path is not None:
                self.save_path = (
                    file_path if osp.isdir(file_path) else osp.dirname(file_path)
                )
            else:
                self.save_path = "."
