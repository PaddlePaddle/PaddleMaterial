import fnmatch
import os
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from functools import cached_property
from pathlib import Path
from typing import Any
from typing import Literal

import numpy as np
from hydra import compose
from hydra import initialize_config_dir
from omegaconf import DictConfig


### checked
def find_local_files(local_path: str, glob: str = "*", relative: bool = False) -> list[str]: # noqa
    """
    Find files in the given directory or blob storage path, and return the list of files
    matching the given glob pattern. If relative is True, the returned paths are 
    relative to the given directory or blob storage path.

    Args:
        blob_or_local_path: path to the directory or blob storage path
        glob: glob pattern to match. By default, all files are returned.
        relative: whether to return relative paths. By default, absolute paths are 
        returned.

    Returns:
        list of paths to files matching the given glob pattern.
    """
    local_files = [x for x in Path(local_path).rglob("*") if os.path.isfile(x)]
    files_list = [(str(x.relative_to(local_path)) if relative else str(x)) for x in local_files]  # noqa
    return fnmatch.filter(files_list, glob)


### checked
@dataclass(frozen=True)
class MatterGenCheckpointInfo:
    model_path: str
    load_epoch: int | Literal["best", "latest"] | None = "latest"
    config_overrides: list[str] = field(default_factory=list)
    split: str = "val"
    strict_checkpoint_loading: bool = True

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["model_path"] = str(self.model_path)
        return d

    @classmethod
    def from_dict(cls, d) -> "MatterGenCheckpointInfo":
        d = d.copy()
        d["model_path"] = Path(d["model_path"])
        if "load_data" in d:
            del d["load_data"]
        return cls(**d)

    @property
    def config(self) -> DictConfig:
        with initialize_config_dir(str(self.model_path)):
            cfg = compose(config_name="config", overrides=self.config_overrides)
            return cfg

    @cached_property
    def checkpoint_path(self) -> str:
        """
        Search for checkpoint files in the given directory, and return the path
        to the checkpoint with the given epoch number or the best checkpoint if load_epoch is "best".
        "Best" is selected via the lowest validation loss, which is stored in the checkpoint filename.
        Assumes that the checkpoint filenames are of the form "epoch=1-val_loss=0.1234.ckpt" or 'last.ckpt'.

        Returns:
            Path to the checkpoint file to load.
        """
        model_path = str(self.model_path)
        ckpts = find_local_files(local_path=model_path, glob="*.pdparams")
        assert len(ckpts) > 0, f"No checkpoints found at {model_path}"
        if self.load_epoch == "latest":
            assert any(
                [x.endswith("latest.pdparams") for x in ckpts]
            ), "No latest.pdparams found in checkpoints."
            return [x for x in ckpts if x.endswith("latest.pdparams")][0]
        if self.load_epoch == "best":
            assert any(
                [x.endswith("best.pdparams") for x in ckpts]
            ), "No best.pdparams found in checkpoints."
            return [x for x in ckpts if x.endswith("best.pdparams")][0]
        ckpts = [
            x
            for x in ckpts
            if not x.endswith("latest.pdparams") and not x.endswith("best.pdparams")
        ]
        ckpt_paths = [Path(x) for x in ckpts]
        ckpt_epochs = np.array(
            [
                int(ckpt.parts[-1].split(".pdparams")[0].split("-")[0].split("=")[1])
                for ckpt in ckpt_paths
            ]
        )
        ckpt_val_losses = np.array(
            [
                (
                    float(ckpt.parts[-1].replace(".pdparams", "").split("-")[1].split("=")[1])
                    if "loss_val" in ckpt.parts[-1]
                    else 99999999.9
                )
                for ckpt in ckpt_paths
            ]
        )
        if self.load_epoch == "best":
            ckpt_ix = ckpt_val_losses.argmin()
        elif isinstance(self.load_epoch, int):
            assert (
                self.load_epoch in ckpt_epochs
            ), f"Epoch {self.load_epoch} not found in checkpoints."
            ckpt_ix = (ckpt_epochs == self.load_epoch).nonzero()[0][0].item()
        else:
            raise ValueError(f"Unrecognized load_epoch {self.load_epoch}")
        ckpt = ckpts[ckpt_ix]
        return ckpt
