import os.path as osp
import pickle
from typing import Callable
from typing import Dict
from typing import Optional

import numpy as np
import paddle
import pandas as pd

from ppmat.datasets.structure_converter import Structure2Graph
from ppmat.datasets.utils import build_structure_from_str
from ppmat.utils import logger


class MP20Dataset(paddle.io.Dataset):
    def __init__(
        self,
        path: str,
        niggli: bool = True,
        primitive: bool = False,
        converter_cfg: Dict = None,
        transforms: Optional[Callable] = None,
        cache: bool = True,
    ):
        super().__init__()
        self.path = path
        self.niggli = niggli
        self.primitive = primitive
        self.converter_cfg = converter_cfg
        self.transforms = transforms
        self.cache = cache

        self.csv_data = self.read_csv(path)
        self.num_samples = len(self.csv_data["cif"])

        # when cache is True, load cached structures from cache file
        cache_path = osp.join(path.rsplit(".", 1)[0] + "_strucs.pkl")
        if self.cache and osp.exists(cache_path):
            with open(cache_path, "rb") as f:
                self.structures = pickle.load(f)
            logger.info(
                f"Load {len(self.structures)} cached structures from {cache_path}"
            )
        else:
            # build structures from cif
            self.structures = build_structure_from_str(
                self.csv_data["cif"], niggli=niggli, primitive=primitive
            )
            logger.info(f"Build {len(self.structures)} structures")
            if self.cache:
                with open(cache_path, "wb") as f:
                    pickle.dump(self.structures, f)
                logger.info(
                    f"Save {len(self.structures)} built structures to {cache_path}"
                )

        # build graphs from structures
        if converter_cfg is not None:
            # load cached graphs from cache file
            cache_path = osp.join(path.rsplit(".", 1)[0] + "_graphs.pkl")
            if self.cache and osp.exists(cache_path):
                with open(cache_path, "rb") as f:
                    self.graphs = pickle.load(f)
                logger.info(f"Load {len(self.graphs)} cached graphs from {cache_path}")
                assert len(self.graphs) == len(self.structures)
            else:
                # build graphs from structures
                self.converter = Structure2Graph(**self.converter_cfg)
                self.graphs = self.converter(self.structures)
                logger.info(f"Convert {len(self.graphs)} structures into graphs")
                if self.cache:
                    with open(cache_path, "wb") as f:
                        pickle.dump(self.graphs, f)
                    logger.info(
                        f"Save {len(self.graphs)} converted graphs to {cache_path}"
                    )
        else:
            self.graphs = None

    def read_csv(self, path):
        data = pd.read_csv(path)
        logger.info(f"Read {len(data)} structures from {path}")
        data = {key: data[key].tolist() for key in data if "Unnamed" not in key}
        return data

    def __getitem__(self, idx):
        data = {}
        if self.graphs is not None:
            data["graph"] = self.graphs[idx]

        data["formation_energy_per_atom"] = np.array(
            [self.csv_data["formation_energy_per_atom"][idx]]
        ).astype("float32")
        data["band_gap"] = np.array([self.csv_data["band_gap"][idx]]).astype("float32")

        data = self.transforms(data) if self.transforms is not None else data
        return data

    def __len__(self):
        return self.num_samples
