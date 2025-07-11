from typing import Optional

import paddle
from omegaconf import OmegaConf

from ppmat.datasets.transform import build_post_transforms
from ppmat.models import build_graph_converter
from ppmat.models import build_model
from ppmat.models import build_model_from_name
from ppmat.utils import logger
from ppmat.utils import save_load


class PPMatPredictor:
    def __init__(
        self,
        model_name: Optional[str] = None,
        weights_name: Optional[str] = None,
        config_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
    ):
        """
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
            if ase_calc and model_config["__class_name__"] == "CHGNet":
                # CHGNet by default predicts energy per atom;
                # convert it to total energy
                model_config["__init_params__"]["is_intensive"] = False
            elif ase_calc:
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

    def from_structures(self, structures):
        data = self.graph_converter(structures)
        if self.eval_with_no_grad:
            with paddle.no_grad():
                out = self.model.predict(data)
        else:
            out = self.model.predict(data)
        out = self.post_process(out)
        return out

    def graph_converter(self, structure):
        if self.graph_converter_fn is None:
            return structure
        return self.graph_converter_fn(structure)

    def post_process(self, data):
        if self.post_transforms is None:
            return data
        return self.post_transforms(data)
