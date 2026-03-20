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

"""
Atom and bond feature encoding utilities.

This module provides featurizers for encoding atoms and bonds in molecular graphs.
"""

import itertools
from collections import defaultdict
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional

import numpy as np
import paddle
from rdkit import Chem


def one_hot_encoding(
    value, allowable_set: List, encode_unknown: bool = False
) -> List[bool]:
    """One-hot encoding for a value.

    Args:
        value: Value to encode
        allowable_set: List of allowable values
        encode_unknown: If True, add an extra element for unknown values

    Returns:
        List of boolean values where at most one value is True
    """
    if encode_unknown and (value not in allowable_set):
        # Add None as the last element for unknown values
        return [v == value for v in allowable_set] + [True]
    else:
        return [v == value for v in allowable_set]


# ============================================================================
# Atom Featurization Functions
# ============================================================================


def atom_type_one_hot(
    atom: Chem.Atom,
    allowable_set: Optional[List[str]] = None,
    encode_unknown: bool = False,
) -> List[bool]:
    """One hot encoding for the type of an atom.

    Args:
        atom: RDKit atom instance
        allowable_set: Atom types to consider. Default: 43 types from GDI-NN
        encode_unknown: If True, map inputs not in the allowable set to the additional last element

    Returns:
        List of boolean values where at most one value is True
    """
    if allowable_set is None:
        allowable_set = [
            "C",
            "N",
            "O",
            "S",
            "F",
            "Si",
            "P",
            "Cl",
            "Br",
            "Mg",
            "Na",
            "Ca",
            "Fe",
            "As",
            "Al",
            "I",
            "B",
            "V",
            "K",
            "Tl",
            "Yb",
            "Sb",
            "Sn",
            "Ag",
            "Pd",
            "Co",
            "Se",
            "Ti",
            "Zn",
            "H",
            "Li",
            "Ge",
            "Cu",
            "Au",
            "Ni",
            "Cd",
            "In",
            "Mn",
            "Zr",
            "Cr",
            "Pt",
            "Hg",
            "Pb",
        ]
    return one_hot_encoding(atom.GetSymbol(), allowable_set, encode_unknown)


def atom_degree_one_hot(
    atom: Chem.Atom,
    allowable_set: Optional[List[int]] = None,
    encode_unknown: bool = False,
) -> List[bool]:
    """One hot encoding for the degree of an atom.

    Args:
        atom: RDKit atom instance
        allowable_set: Atom degrees to consider. Default: 0-10
        encode_unknown: If True, map inputs not in the allowable set to the additional last element

    Returns:
        List of boolean values where at most one value is True
    """
    if allowable_set is None:
        allowable_set = list(range(11))
    return one_hot_encoding(atom.GetDegree(), allowable_set, encode_unknown)


def atom_implicit_valence_one_hot(
    atom: Chem.Atom,
    allowable_set: Optional[List[int]] = None,
    encode_unknown: bool = False,
) -> List[bool]:
    """One hot encoding for the implicit valence of an atom.

    Args:
        atom: RDKit atom instance
        allowable_set: Atom implicit valences to consider. Default: 0-6
        encode_unknown: If True, map inputs not in the allowable set to the additional last element

    Returns:
        List of boolean values where at most one value is True
    """
    if allowable_set is None:
        allowable_set = list(range(7))
    return one_hot_encoding(atom.GetImplicitValence(), allowable_set, encode_unknown)


def atom_formal_charge(atom: Chem.Atom) -> List[float]:
    """Get formal charge for an atom.

    Args:
        atom: RDKit atom instance

    Returns:
        List containing one float value
    """
    return [float(atom.GetFormalCharge())]


def atom_num_radical_electrons(atom: Chem.Atom) -> List[float]:
    """Get the number of radical electrons for an atom.

    Args:
        atom: RDKit atom instance

    Returns:
        List containing one float value
    """
    return [float(atom.GetNumRadicalElectrons())]


def atom_hybridization_one_hot(
    atom: Chem.Atom, allowable_set: Optional[List] = None, encode_unknown: bool = False
) -> List[bool]:
    """One hot encoding for the hybridization of an atom.

    Args:
        atom: RDKit atom instance
        allowable_set: Atom hybridizations to consider. Default: SP, SP2, SP3, SP3D, SP3D2
        encode_unknown: If True, map inputs not in the allowable set to the additional last element

    Returns:
        List of boolean values where at most one value is True
    """
    if allowable_set is None:
        allowable_set = [
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2,
        ]
    return one_hot_encoding(atom.GetHybridization(), allowable_set, encode_unknown)


def atom_is_aromatic(atom: Chem.Atom) -> List[float]:
    """Get whether the atom is aromatic.

    Args:
        atom: RDKit atom instance

    Returns:
        List containing one float value
    """
    return [float(atom.GetIsAromatic())]


def atom_total_num_H_one_hot(
    atom: Chem.Atom,
    allowable_set: Optional[List[int]] = None,
    encode_unknown: bool = False,
) -> List[bool]:
    """One hot encoding for the total number of Hs of an atom.

    Args:
        atom: RDKit atom instance
        allowable_set: Total number of Hs to consider. Default: 0-4
        encode_unknown: If True, map inputs not in the allowable set to the additional last element

    Returns:
        List of boolean values where at most one value is True
    """
    if allowable_set is None:
        allowable_set = list(range(5))
    return one_hot_encoding(atom.GetTotalNumHs(), allowable_set, encode_unknown)


# ============================================================================
# Bond Featurization Functions
# ============================================================================


def bond_type_one_hot(
    bond: Chem.Bond, allowable_set: Optional[List] = None, encode_unknown: bool = False
) -> List[bool]:
    """One hot encoding for the type of a bond.

    Args:
        bond: RDKit bond instance
        allowable_set: Bond types to consider. Default: SINGLE, DOUBLE, TRIPLE, AROMATIC
        encode_unknown: If True, map inputs not in the allowable set to the additional last element

    Returns:
        List of boolean values where at most one value is True
    """
    if allowable_set is None:
        allowable_set = [
            Chem.rdchem.BondType.SINGLE,
            Chem.rdchem.BondType.DOUBLE,
            Chem.rdchem.BondType.TRIPLE,
            Chem.rdchem.BondType.AROMATIC,
        ]
    return one_hot_encoding(bond.GetBondType(), allowable_set, encode_unknown)


def bond_is_conjugated(bond: Chem.Bond) -> List[float]:
    """Get whether the bond is conjugated.

    Args:
        bond: RDKit bond instance

    Returns:
        List containing one float value
    """
    return [float(bond.GetIsConjugated())]


def bond_is_in_ring(bond: Chem.Bond) -> List[float]:
    """Get whether the bond is in a ring.

    Args:
        bond: RDKit bond instance

    Returns:
        List containing one float value
    """
    return [float(bond.IsInRing())]


def bond_stereo_one_hot(
    bond: Chem.Bond, allowable_set: Optional[List] = None, encode_unknown: bool = False
) -> List[bool]:
    """One hot encoding for the stereo configuration of a bond.

    Args:
        bond: RDKit bond instance
        allowable_set: Stereo configurations to consider. Default: STEREONONE, STEREOZ, STEREOE,
                      STEREOCIS, STEREOTRANS, STEREOANY
        encode_unknown: If True, map inputs not in the allowable set to the additional last element

    Returns:
        List of boolean values where at most one value is True
    """
    if allowable_set is None:
        allowable_set = [
            Chem.rdchem.BondStereo.STEREONONE,
            Chem.rdchem.BondStereo.STEREOZ,
            Chem.rdchem.BondStereo.STEREOE,
            Chem.rdchem.BondStereo.STEREOCIS,
            Chem.rdchem.BondStereo.STEREOTRANS,
            Chem.rdchem.BondStereo.STEREOANY,
        ]
    return one_hot_encoding(bond.GetStereo(), allowable_set, encode_unknown)


# ============================================================================
# ConcatFeaturizer
# ============================================================================


class ConcatFeaturizer:
    """Concatenate the evaluation results of multiple functions as a single feature.

    Args:
        func_list: List of functions for computing features from an atom or bond.
                  Each function should return a list of float or bool values.
    """

    def __init__(self, func_list: List[Callable]):
        self.func_list = func_list

    def __call__(self, x) -> List:
        """Featurize the input data.

        Args:
            x: RDKit atom or bond instance

        Returns:
            List of feature values
        """
        return list(itertools.chain.from_iterable([func(x) for func in self.func_list]))


# ============================================================================
# Base Featurizer Classes
# ============================================================================


class BaseAtomFeaturizer:
    """An abstract class for atom featurizers.

    Loop over all atoms in a molecule and featurize them with the featurizer_funcs.

    Args:
        featurizer_funcs: Dictionary mapping feature name to featurization function
        feat_sizes: Dictionary mapping feature name to the size of the corresponding feature
    """

    def __init__(
        self,
        featurizer_funcs: Dict[str, Callable],
        feat_sizes: Optional[Dict[str, int]] = None,
    ):
        self.featurizer_funcs = featurizer_funcs
        self._feat_sizes = feat_sizes if feat_sizes is not None else {}

    def feat_size(self, feat_name: Optional[str] = None) -> int:
        """Get the feature size for feat_name.

        Args:
            feat_name: Feature for query

        Returns:
            Feature size for the feature with name feat_name
        """
        if feat_name is None:
            assert (
                len(self.featurizer_funcs) == 1
            ), "feat_name should be provided if there are more than one features"
            feat_name = list(self.featurizer_funcs.keys())[0]

        if feat_name not in self.featurizer_funcs:
            raise ValueError(
                f"feat_name {feat_name} not in {list(self.featurizer_funcs.keys())}"
            )

        if feat_name not in self._feat_sizes:
            # Compute feature size by applying the function to a test atom
            atom = Chem.MolFromSmiles("C").GetAtomWithIdx(0)
            self._feat_sizes[feat_name] = len(self.featurizer_funcs[feat_name](atom))

        return self._feat_sizes[feat_name]

    def __call__(self, mol: Chem.Mol) -> Dict[str, paddle.Tensor]:
        """Featurize all atoms in a molecule.

        Args:
            mol: RDKit molecule instance

        Returns:
            Dictionary mapping feature name to feature tensor of shape [num_atoms, feat_size]
        """
        num_atoms = mol.GetNumAtoms()
        atom_features = defaultdict(list)

        # Compute features for each atom
        for i in range(num_atoms):
            atom = mol.GetAtomWithIdx(i)
            for feat_name, feat_func in self.featurizer_funcs.items():
                atom_features[feat_name].append(feat_func(atom))

        # Stack the features and convert to tensors
        processed_features = {}
        for feat_name, feat_list in atom_features.items():
            feat = np.stack(feat_list).astype(np.float32)
            processed_features[feat_name] = paddle.to_tensor(feat)

        return processed_features


class BaseBondFeaturizer:
    """An abstract class for bond featurizers.

    Loop over all bonds in a molecule and featurize them with the featurizer_funcs.

    Args:
        featurizer_funcs: Dictionary mapping feature name to featurization function
        feat_sizes: Dictionary mapping feature name to the size of the corresponding feature
        self_loop: Whether to add self-loops
    """

    def __init__(
        self,
        featurizer_funcs: Dict[str, Callable],
        feat_sizes: Optional[Dict[str, int]] = None,
        self_loop: bool = False,
    ):
        self.featurizer_funcs = featurizer_funcs
        self._feat_sizes = feat_sizes if feat_sizes is not None else {}
        self.self_loop = self_loop

    def feat_size(self, feat_name: Optional[str] = None) -> int:
        """Get the feature size for feat_name.

        Args:
            feat_name: Feature for query

        Returns:
            Feature size for the feature with name feat_name
        """
        if feat_name is None:
            assert (
                len(self.featurizer_funcs) == 1
            ), "feat_name should be provided if there are more than one features"
            feat_name = list(self.featurizer_funcs.keys())[0]

        if feat_name not in self.featurizer_funcs:
            raise ValueError(
                f"feat_name {feat_name} not in {list(self.featurizer_funcs.keys())}"
            )

        if feat_name not in self._feat_sizes:
            # Compute feature size by applying the function to a test bond
            mol = Chem.MolFromSmiles("CO")
            bond = mol.GetBondWithIdx(0)
            self._feat_sizes[feat_name] = len(self.featurizer_funcs[feat_name](bond))

        return self._feat_sizes[feat_name]

    def __call__(self, mol: Chem.Mol) -> Dict[str, paddle.Tensor]:
        """Featurize all bonds in a molecule.

        Args:
            mol: RDKit molecule instance

        Returns:
            Dictionary containing:
                - 'src': Source node indices
                - 'dst': Destination node indices
                - Feature tensors for each feature name
        """
        num_atoms = mol.GetNumAtoms()
        num_bonds = mol.GetNumBonds()

        # Collect bond information
        bonds = []
        bond_features = defaultdict(list)

        for i in range(num_bonds):
            bond = mol.GetBondWithIdx(i)
            src = bond.GetBeginAtomIdx()
            dst = bond.GetEndAtomIdx()

            # Compute features for this bond
            for feat_name, feat_func in self.featurizer_funcs.items():
                feat = feat_func(bond)
                # Add the same features for both directions (bidirectional graph)
                bond_features[feat_name].extend([feat, feat.copy()])

            # Add edges in both directions
            bonds.append((src, dst))
            bonds.append((dst, src))

        # Add self-loops if needed
        if self.self_loop:
            for i in range(num_atoms):
                bonds.append((i, i))
                for feat_name in self.featurizer_funcs.keys():
                    # Self-loop features: all zeros except a marker
                    feat_size = self.feat_size(feat_name)
                    self_loop_feat = [0.0] * feat_size
                    bond_features[feat_name].append(self_loop_feat)

        # Convert to arrays
        if len(bonds) == 0:
            # Handle edge case of single atom
            src = np.array([0], dtype=np.int64)
            dst = np.array([0], dtype=np.int64)
        else:
            src = np.array([b[0] for b in bonds], dtype=np.int64)
            dst = np.array([b[1] for b in bonds], dtype=np.int64)

        # Stack features
        result = {"src": paddle.to_tensor(src), "dst": paddle.to_tensor(dst)}

        for feat_name, feat_list in bond_features.items():
            feat = np.stack(feat_list).astype(np.float32)
            result[feat_name] = paddle.to_tensor(feat)

        return result


# ============================================================================
# Canonical Featurizers
# ============================================================================


class CanonicalAtomFeaturizer(BaseAtomFeaturizer):
    """A default featurizer for atoms.

    This implementation matches GDI-NN's CanonicalAtomFeaturizer for compatibility.

    The atom features include:
    * One hot encoding of the atom type (43 types)
    * One hot encoding of the atom degree (0-10)
    * One hot encoding of the implicit valence (0-6)
    * Formal charge of the atom
    * Number of radical electrons of the atom
    * One hot encoding of the atom hybridization (SP, SP2, SP3, SP3D, SP3D2)
    * Whether the atom is aromatic
    * One hot encoding of the number of total Hs on the atom (0-4)

    Total: 74 dimensions

    Args:
        atom_data_field: Name for storing atom features, default to 'h'
    """

    def __init__(self, atom_data_field: str = "h"):
        super(CanonicalAtomFeaturizer, self).__init__(
            featurizer_funcs={
                atom_data_field: ConcatFeaturizer(
                    [
                        atom_type_one_hot,
                        atom_degree_one_hot,
                        atom_implicit_valence_one_hot,
                        atom_formal_charge,
                        atom_num_radical_electrons,
                        atom_hybridization_one_hot,
                        atom_is_aromatic,
                        atom_total_num_H_one_hot,
                    ]
                )
            }
        )


class CanonicalBondFeaturizer(BaseBondFeaturizer):
    """A default featurizer for bonds.

    The bond features include:
    * One hot encoding of the bond type (SINGLE, DOUBLE, TRIPLE, AROMATIC)
    * Whether the bond is conjugated
    * Whether the bond is in a ring
    * One hot encoding of the stereo configuration (STEREONONE, STEREOZ, STEREOE,
      STEREOCIS, STEREOTRANS, STEREOANY)

    Total: 12 dimensions

    Args:
        bond_data_field: Name for storing bond features, default to 'e'
        self_loop: Whether to add self-loops
    """

    def __init__(self, bond_data_field: str = "e", self_loop: bool = False):
        super(CanonicalBondFeaturizer, self).__init__(
            featurizer_funcs={
                bond_data_field: ConcatFeaturizer(
                    [
                        bond_type_one_hot,
                        bond_is_conjugated,
                        bond_is_in_ring,
                        bond_stereo_one_hot,
                    ]
                )
            },
            self_loop=self_loop,
        )
