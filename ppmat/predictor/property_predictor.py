# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import os.path as osp
from collections import defaultdict
from typing import Optional
from typing import Sequence

import paddle
import pandas as pd
from pymatgen.core import Structure
from tqdm import tqdm

from ppmat.datasets.build_molecule import BuildMolecule
from ppmat.predictor.base import BasePredictor
from ppmat.utils import logger


class PropertyPredictor(BasePredictor):
    """Property predictor.

    This class provides an interface for predicting properties of crystalline
    structures using pre-trained deep learning models. Supports two initialization
    modes:

    1. **Automatic Model Loading**
       Specify `model_name` and `weights_name` to automatically download
       and load pre-trained weights from the `MODEL_REGISTRY`.

    2. **Custom Model Loading**
       Provide explicit `config_path` and `checkpoint_path` to load
       custom-trained models from local files.

    Args:
        model_name (Optional[str], optional): Name of the pre-defined model architecture
            from the `MODEL_REGISTRY` registry. When specified, associated weights
            will be automatically downloaded. Defaults to None.

        weights_name (Optional[str], optional): Specific pre-trained weight identifier.
            Used only when `model_name` is provided. Valid options include:
            - 'best.pdparams' (highest validation performance)
            - 'latest.pdparams' (most recent training checkpoint)
            - Custom weight files ending with '.pdparams'
            Defaults to None.

        config_path (Optional[str], optional): Path to model configuration file (YAML)
            for custom models. Required when not using predefined `model_name`.
            Defaults to None.
        checkpoint_path (Optional[str], optional): Path to a model checkpoint file
            (.pdparams) for custom models. If omitted, `Predict.checkpoint_path` from
            the config is used. Defaults to None.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        weights_name: Optional[str] = None,
        config_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
        config_overrides: Optional[Sequence[str]] = None,
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
        self.load_inference_model()

    def from_structures(self, structures):
        data = self.graph_converter(structures)
        return self._run_model(data)

    def from_cif_file(self, cif_file_path, save_path=None):
        if save_path is not None:
            assert save_path.endswith(".csv"), "save_path must end with .csv"
        if osp.isdir(cif_file_path):
            cif_files = [
                osp.join(cif_file_path, f)
                for f in os.listdir(cif_file_path)
                if f.endswith(".cif")
            ]
            results = []
            for cif_file in tqdm(cif_files):
                structure = Structure.from_file(cif_file)
                result = self.from_structures(structure)
                results.append(result)
            if save_path is not None:

                keys = list(results[0].keys())
                result_properties = defaultdict(list)
                for key in keys:
                    for r in results:
                        result_properties[key].append(r[key])

                # save cif_files and result to csv file
                df = pd.DataFrame({"cif_file": cif_files, **result_properties})
                df.to_csv(save_path, index=False)
                logger.info(f"Saved the prediction result to {save_path}")

            return results
        else:
            structure = Structure.from_file(cif_file_path)
            result = self.from_structures(structure)

            keys = list(result.keys())
            result_properties = defaultdict(list)
            for key in keys:
                result_properties[key].append(result[key])

            if save_path is not None:
                df = pd.DataFrame({"cif_file": [cif_file_path], **result_properties})
                df.to_csv(save_path, index=False)
                logger.info(f"Saved the prediction result to {save_path}")

            return result

    def from_molecule(self, molecule_data, molecule_format):
        """Predict properties from molecular data.

        Follows the standard PaddleMaterials molecular pipeline:
        ``BuildMolecule -> graph_converter -> predict``.
        """
        mol = BuildMolecule(format=molecule_format)(molecule_data)

        try:
            conf = mol.GetConformer()
        except ValueError:
            conf = None
        if conf is None or not conf.Is3D():
            from rdkit import Chem as RDChem
            from rdkit.Chem import AllChem

            if molecule_format == "smiles":
                mol = RDChem.AddHs(mol)
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.MMFFOptimizeMolecule(mol)

        if self.graph_converter_fn is not None:
            data = self.graph_converter_fn(mol)
        else:
            conf = mol.GetConformer()
            num_atoms = mol.GetNumAtoms()
            z = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
            pos = [
                [
                    conf.GetAtomPosition(i).x,
                    conf.GetAtomPosition(i).y,
                    conf.GetAtomPosition(i).z,
                ]
                for i in range(num_atoms)
            ]
            data = {
                "z": paddle.to_tensor(z, dtype=paddle.int64),
                "pos": paddle.to_tensor(pos, dtype=paddle.get_default_dtype()),
                "batch": paddle.zeros([num_atoms], dtype=paddle.int64),
            }

        return self._run_model(data)

    def from_xyz_file(self, xyz_file_path, save_path=None):
        """Predict molecular properties from XYZ file(s).

        Reads each ``.xyz`` file via RDKit, then delegates to
        :meth:`from_molecule` (``BuildMolecule → graph_converter → predict``).

        Args:
            xyz_file_path: Path to a single ``.xyz`` file or a directory
                of ``.xyz`` files.
            save_path: Optional CSV path.

        Returns:
            Single result dict or list of result dicts.
        """
        from rdkit import Chem

        if save_path is not None:
            assert save_path.endswith(".csv"), "save_path must end with .csv"

        if osp.isdir(xyz_file_path):
            xyz_files = sorted(
                [
                    osp.join(xyz_file_path, f)
                    for f in os.listdir(xyz_file_path)
                    if f.endswith(".xyz")
                ]
            )
        else:
            xyz_files = [xyz_file_path]

        results = []
        for xyz_path in tqdm(xyz_files, desc="Predict"):
            with open(xyz_path, "r") as f:
                xyz_block = f.read()
            mol = Chem.MolFromXYZBlock(xyz_block)
            if mol is None:
                raise ValueError(f"Failed to parse XYZ file: {xyz_path}")
            out = self.from_molecule(mol, "rdmol")
            results.append(out)

        if save_path is not None and results:
            keys = list(results[0].keys())
            props = defaultdict(list)
            for key in keys:
                for r in results:
                    props[key].append(r[key])
            df = pd.DataFrame(
                {"xyz_file": [osp.basename(f) for f in xyz_files], **props}
            )
            df.to_csv(save_path, index=False)
            logger.info(f"Saved prediction results to {save_path}")

        return results if len(results) > 1 else results[0]
