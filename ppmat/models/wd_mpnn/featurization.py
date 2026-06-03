from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from itertools import zip_longest
from typing import List
from typing import Tuple
from typing import Union

import numpy as np
import paddle
from rdkit import Chem

from ppmat.utils.ext_rdkit import make_mol
from ppmat.utils.ext_rdkit import make_polymer_mol


def tag_atoms_in_repeating_unit(mol):
    """
    Tags atoms that are part of the core units, as well as atoms serving to identify
    attachment points. In addition, create a map of bond types based on what bonds are
    connected to R groups in the input.
    """
    atoms = [a for a in mol.GetAtoms()]
    neighbor_map = {}
    r_bond_types = {}

    for atom in atoms:
        if "*" in atom.GetSmarts():
            neighbors = atom.GetNeighbors()
            assert len(neighbors) == 1
            neighbor_idx = neighbors[0].GetIdx()
            r_tag = atom.GetSmarts().strip("[]").replace(":", "")
            neighbor_map[r_tag] = neighbor_idx
            atom.SetBoolProp("core", False)
            bond = mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor_idx)
            r_bond_types[r_tag] = bond.GetBondType()
        else:
            atom.SetBoolProp("core", True)

    for atom in atoms:
        if atom.GetIdx() in neighbor_map.values():
            r_tags = [k for k, v in neighbor_map.items() if v == atom.GetIdx()]
            atom.SetProp("R", "".join(r_tags))
        else:
            atom.SetProp("R", "")

    return mol, r_bond_types


def remove_wildcard_atoms(rwmol):
    indices = [a.GetIdx() for a in rwmol.GetAtoms() if "*" in a.GetSmarts()]
    while len(indices) > 0:
        rwmol.RemoveAtom(indices[0])
        indices = [a.GetIdx() for a in rwmol.GetAtoms() if "*" in a.GetSmarts()]
    Chem.SanitizeMol(rwmol, Chem.SanitizeFlags.SANITIZE_ALL)
    return rwmol


def parse_polymer_rules(rules):
    polymer_info = []
    counter = Counter()

    if "~" in rules[-1]:
        Xn = float(rules[-1].split("~")[1])
        rules[-1] = rules[-1].split("~")[0]
    else:
        Xn = 1.0

    for rule in rules:
        if rule == "":
            continue
        if len(rule.split(":")) != 3:
            raise ValueError(f'incorrect format for input information "{rule}"')
        idx1, idx2 = rule.split(":")[0].split("-")
        w12 = float(rule.split(":")[1])
        w21 = float(rule.split(":")[2])
        polymer_info.append((idx1, idx2, w12, w21))
        counter[idx1] += float(w21)
        counter[idx2] += float(w12)

    for k, v in counter.items():
        if np.isclose(v, 1.0) is False:
            raise ValueError(
                f"sum of weights of incoming stochastic edges should be 1 -- found {v} for [*:{k}]"
            )
    return polymer_info, 1.0 + np.log10(Xn)


# ---------------------------------------------------------------------------
# Featurization configuration
# ---------------------------------------------------------------------------


@dataclass
class Featurization_parameters:
    max_atomic_num: int = 100
    explicit_h: bool = False
    add_h: bool = False
    polymer: bool = False
    reaction: bool = False
    reaction_mode: str = None
    extra_atom_fdim: int = 0
    extra_bond_fdim: int = 0

    def __post_init__(self):
        self.ATOM_FEATURES = {
            "atomic_num": list(range(self.max_atomic_num)),
            "degree": [0, 1, 2, 3, 4, 5],
            "formal_charge": [-1, -2, 1, 2, 0],
            "chiral_tag": [0, 1, 2, 3],
            "num_Hs": [0, 1, 2, 3, 4],
            "hybridization": [
                Chem.rdchem.HybridizationType.SP,
                Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3,
                Chem.rdchem.HybridizationType.SP3D,
                Chem.rdchem.HybridizationType.SP3D2,
            ],
        }
        self.ATOM_FDIM = (
            sum(len(choices) + 1 for choices in self.ATOM_FEATURES.values()) + 2
        )
        self.BOND_FDIM = 14


_DEFAULT_CONFIG = None


def _get_default_config() -> Featurization_parameters:
    global _DEFAULT_CONFIG
    if _DEFAULT_CONFIG is None:
        _DEFAULT_CONFIG = Featurization_parameters()
    return _DEFAULT_CONFIG


# ---------------------------------------------------------------------------
# Dimension helpers
# ---------------------------------------------------------------------------


def get_atom_fdim(
    config: Featurization_parameters = None, overwrite_default_atom: bool = False
) -> int:
    if config is None:
        config = _get_default_config()
    return (not overwrite_default_atom) * config.ATOM_FDIM + config.extra_atom_fdim


def get_bond_fdim(
    config: Featurization_parameters = None,
    atom_messages: bool = False,
    overwrite_default_bond: bool = False,
    overwrite_default_atom: bool = False,
) -> int:
    if config is None:
        config = _get_default_config()
    return (
        (not overwrite_default_bond) * config.BOND_FDIM
        + config.extra_bond_fdim
        + (not atom_messages)
        * get_atom_fdim(config, overwrite_default_atom=overwrite_default_atom)
    )


# ---------------------------------------------------------------------------
# Feature functions
# ---------------------------------------------------------------------------


def onek_encoding_unk(value: int, choices: List[int]) -> List[int]:
    """
    Creates a one-hot encoding with an extra category for uncommon values.
    """
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1
    return encoding


def atom_features(
    atom: Chem.rdchem.Atom,
    functional_groups: List[int] = None,
    config: Featurization_parameters = None,
) -> List[Union[bool, int, float]]:
    """
    Builds a feature vector for an atom.
    """
    if config is None:
        config = _get_default_config()
    if atom is None:
        features = [0] * config.ATOM_FDIM
    else:
        features = (
            onek_encoding_unk(
                atom.GetAtomicNum() - 1, config.ATOM_FEATURES["atomic_num"]
            )
            + onek_encoding_unk(atom.GetTotalDegree(), config.ATOM_FEATURES["degree"])
            + onek_encoding_unk(
                atom.GetFormalCharge(), config.ATOM_FEATURES["formal_charge"]
            )
            + onek_encoding_unk(
                int(atom.GetChiralTag()), config.ATOM_FEATURES["chiral_tag"]
            )
            + onek_encoding_unk(
                int(atom.GetTotalNumHs()), config.ATOM_FEATURES["num_Hs"]
            )
            + onek_encoding_unk(
                int(atom.GetHybridization()), config.ATOM_FEATURES["hybridization"]
            )
            + [1 if atom.GetIsAromatic() else 0]
            + [atom.GetMass() * 0.01]
        )
        if functional_groups is not None:
            features += functional_groups
    return features


def bond_features(
    bond: Chem.rdchem.Bond, config: Featurization_parameters = None
) -> List[Union[bool, int, float]]:
    """
    Builds a feature vector for a bond.
    """
    if config is None:
        config = _get_default_config()
    if bond is None:
        fbond = [1] + [0] * (config.BOND_FDIM - 1)
    else:
        bt = bond.GetBondType()
        fbond = [
            0,  # bond is not None
            bt == Chem.rdchem.BondType.SINGLE,
            bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE,
            bt == Chem.rdchem.BondType.AROMATIC,
            (bond.GetIsConjugated() if bt is not None else 0),
            (bond.IsInRing() if bt is not None else 0),
        ]
        fbond += onek_encoding_unk(int(bond.GetStereo()), list(range(6)))
    return fbond


# ---------------------------------------------------------------------------
# MolGraph
# ---------------------------------------------------------------------------


class MolGraph:
    """
    A MolGraph represents the graph structure and featurization of a single molecule.
    """

    def __init__(
        self,
        mol: Union[str, Chem.Mol, Tuple[Chem.Mol, Chem.Mol]],
        atom_features_extra: np.ndarray = None,
        bond_features_extra: np.ndarray = None,
        overwrite_default_atom_features: bool = False,
        overwrite_default_bond_features: bool = False,
        config: Featurization_parameters = None,
    ):
        """
        :param mol: A SMILES or an RDKit molecule.
        :param atom_features_extra: Additional atom features as numpy array.
        :param bond_features_extra: Additional bond features as numpy array.
        :param overwrite_default_atom_features: Whether to overwrite default atom features.
        :param overwrite_default_bond_features: Whether to overwrite default bond features.
        :param config: A Featurization_parameters instance (uses default if None).
        """
        if config is None:
            config = _get_default_config()
        self.config = config

        self.is_polymer = config.polymer
        self.polymer_info = []
        self.is_reaction = config.reaction
        self.is_explicit_h = config.explicit_h
        self.is_adding_hs = config.add_h
        self.reaction_mode = config.reaction_mode

        # Convert SMILES to RDKit molecule if necessary
        if type(mol) == str:
            if self.is_reaction:
                raise NotImplementedError(
                    "Reaction mode is not supported in this port."
                )
            elif self.is_polymer:
                mol = (
                    make_polymer_mol(
                        mol.split("|")[0],
                        self.is_explicit_h,
                        self.is_adding_hs,
                        fragment_weights=mol.split("|")[1:-1],
                    ),
                    mol.split("<")[1:],
                )
            else:
                mol = make_mol(mol, self.is_explicit_h, self.is_adding_hs)

        self.n_atoms = 0
        self.n_bonds = 0
        self.degree_of_polym = 1
        self.f_atoms = []
        self.f_bonds = []
        self.w_bonds = []
        self.w_atoms = []
        self.a2b = []
        self.b2a = []
        self.b2revb = []
        self.overwrite_default_atom_features = overwrite_default_atom_features
        self.overwrite_default_bond_features = overwrite_default_bond_features

        # =============
        # Standard mode
        # =============
        if not self.is_reaction and not self.is_polymer:
            self.f_atoms = [
                atom_features(atom, config=config) for atom in mol.GetAtoms()
            ]
            self.w_atoms = [1.0] * len(mol.GetAtoms())
            if atom_features_extra is not None:
                if overwrite_default_atom_features:
                    self.f_atoms = [descs.tolist() for descs in atom_features_extra]
                else:
                    self.f_atoms = [
                        f_atoms + descs.tolist()
                        for f_atoms, descs in zip(self.f_atoms, atom_features_extra)
                    ]

            self.n_atoms = len(self.f_atoms)
            if (
                atom_features_extra is not None
                and len(atom_features_extra) != self.n_atoms
            ):
                raise ValueError(
                    f"The number of atoms in {Chem.MolToSmiles(mol)} is different from the length of "
                    f"the extra atom features"
                )

            for _ in range(self.n_atoms):
                self.a2b.append([])

            for a1 in range(self.n_atoms):
                for a2 in range(a1 + 1, self.n_atoms):
                    bond = mol.GetBondBetweenAtoms(a1, a2)

                    if bond is None:
                        continue

                    f_bond = bond_features(bond, config=config)
                    if bond_features_extra is not None:
                        descr = bond_features_extra[bond.GetIdx()].tolist()
                        if overwrite_default_bond_features:
                            f_bond = descr
                        else:
                            f_bond += descr

                    self.f_bonds.append(self.f_atoms[a1] + f_bond)
                    self.f_bonds.append(self.f_atoms[a2] + f_bond)

                    b1 = self.n_bonds
                    b2 = b1 + 1
                    self.a2b[a2].append(b1)
                    self.b2a.append(a1)
                    self.a2b[a1].append(b2)
                    self.b2a.append(a2)
                    self.b2revb.append(b2)
                    self.b2revb.append(b1)
                    self.w_bonds.extend([1.0, 1.0])
                    self.n_bonds += 2

            if (
                bond_features_extra is not None
                and len(bond_features_extra) != self.n_bonds / 2
            ):
                raise ValueError(
                    f"The number of bonds in {Chem.MolToSmiles(mol)} is different from the length of "
                    f"the extra bond features"
                )

        # ============
        # Polymer mode
        # ============
        if not self.is_reaction and self.is_polymer:
            m = mol[0]
            rules = mol[1]
            self.polymer_info, self.degree_of_polym = parse_polymer_rules(rules)
            rwmol = Chem.rdchem.RWMol(m)
            rwmol, r_bond_types = tag_atoms_in_repeating_unit(rwmol)

            # Get atom features
            self.f_atoms = [
                atom_features(atom, config=config)
                for atom in rwmol.GetAtoms()
                if atom.GetBoolProp("core") is True
            ]
            self.w_atoms = [
                atom.GetDoubleProp("w_frag")
                for atom in rwmol.GetAtoms()
                if atom.GetBoolProp("core") is True
            ]

            if atom_features_extra is not None:
                if overwrite_default_atom_features:
                    self.f_atoms = [descs.tolist() for descs in atom_features_extra]
                else:
                    self.f_atoms = [
                        f_atoms + descs.tolist()
                        for f_atoms, descs in zip(self.f_atoms, atom_features_extra)
                    ]

            self.n_atoms = len(self.f_atoms)
            if (
                atom_features_extra is not None
                and len(atom_features_extra) != self.n_atoms
            ):
                raise ValueError(
                    f"The number of atoms in {Chem.MolToSmiles(rwmol)} is different from the length of "
                    f"the extra atom features"
                )

            rwmol = remove_wildcard_atoms(rwmol)

            for _ in range(self.n_atoms):
                self.a2b.append([])

            # Bond features for separate monomers
            for a1 in range(self.n_atoms):
                for a2 in range(a1 + 1, self.n_atoms):
                    bond = rwmol.GetBondBetweenAtoms(a1, a2)

                    if bond is None:
                        continue

                    f_bond = bond_features(bond, config=config)
                    if bond_features_extra is not None:
                        descr = bond_features_extra[bond.GetIdx()].tolist()
                        if overwrite_default_bond_features:
                            f_bond = descr
                        else:
                            f_bond += descr

                    self.f_bonds.append(self.f_atoms[a1] + f_bond)
                    self.f_bonds.append(self.f_atoms[a2] + f_bond)

                    b1 = self.n_bonds
                    b2 = b1 + 1
                    self.a2b[a2].append(b1)
                    self.b2a.append(a1)
                    self.a2b[a1].append(b2)
                    self.b2a.append(a2)
                    self.b2revb.append(b2)
                    self.b2revb.append(b1)
                    self.w_bonds.extend([1.0, 1.0])
                    self.n_bonds += 2

            # Bond features for bonds between repeating units
            rwmol_copy = deepcopy(rwmol)
            _ = [a.SetBoolProp("OrigMol", True) for a in rwmol.GetAtoms()]
            _ = [a.SetBoolProp("OrigMol", False) for a in rwmol_copy.GetAtoms()]
            cm = Chem.CombineMols(rwmol, rwmol_copy)
            cm = Chem.RWMol(cm)

            for r1, r2, w_bond12, w_bond21 in self.polymer_info:
                a1 = None
                a2 = None
                _a2 = None
                for atom in cm.GetAtoms():
                    if (
                        f"*{r1}" in atom.GetProp("R")
                        and atom.GetBoolProp("OrigMol") is True
                    ):
                        a1 = atom.GetIdx()
                    if f"*{r2}" in atom.GetProp("R"):
                        if atom.GetBoolProp("OrigMol") is True:
                            a2 = atom.GetIdx()
                        elif atom.GetBoolProp("OrigMol") is False:
                            _a2 = atom.GetIdx()

                if a1 is None:
                    raise ValueError(f"cannot find atom attached to [*:{r1}]")
                if a2 is None or _a2 is None:
                    raise ValueError(f"cannot find atom attached to [*:{r2}]")

                order1 = r_bond_types[f"*{r1}"]
                order2 = r_bond_types[f"*{r2}"]
                if order1 != order2:
                    raise ValueError(
                        f"two atoms are trying to be bonded with different bond types: "
                        f"{order1} vs {order2}"
                    )
                cm.AddBond(a1, _a2, order=order1)
                Chem.SanitizeMol(cm, Chem.SanitizeFlags.SANITIZE_ALL)

                bond = cm.GetBondBetweenAtoms(a1, _a2)
                f_bond = bond_features(bond, config=config)
                if bond_features_extra is not None:
                    descr = bond_features_extra[bond.GetIdx()].tolist()
                    if overwrite_default_bond_features:
                        f_bond = descr
                    else:
                        f_bond += descr

                self.f_bonds.append(self.f_atoms[a1] + f_bond)
                self.f_bonds.append(self.f_atoms[a2] + f_bond)

                b1 = self.n_bonds
                b2 = b1 + 1
                self.a2b[a2].append(b1)
                self.b2a.append(a1)
                self.a2b[a1].append(b2)
                self.b2a.append(a2)
                self.b2revb.append(b2)
                self.b2revb.append(b1)
                self.w_bonds.extend([w_bond12, w_bond21])
                self.n_bonds += 2

                cm.RemoveBond(a1, _a2)
                Chem.SanitizeMol(cm, Chem.SanitizeFlags.SANITIZE_ALL)

            if (
                bond_features_extra is not None
                and len(bond_features_extra) != self.n_bonds / 2
            ):
                raise ValueError(
                    f"The number of bonds in {Chem.MolToSmiles(rwmol)} is different from the length of "
                    f"the extra bond features"
                )


# ---------------------------------------------------------------------------
# BatchMolGraph
# ---------------------------------------------------------------------------


class BatchMolGraph:
    """
    A BatchMolGraph represents the graph structure and featurization of a batch of molecules.
    """

    def __init__(self, mol_graphs: List[MolGraph]):
        self.overwrite_default_atom_features = mol_graphs[
            0
        ].overwrite_default_atom_features
        self.overwrite_default_bond_features = mol_graphs[
            0
        ].overwrite_default_bond_features

        config = mol_graphs[0].config
        self.atom_fdim = get_atom_fdim(
            config, overwrite_default_atom=self.overwrite_default_atom_features
        )
        self.bond_fdim = get_bond_fdim(
            config,
            overwrite_default_bond=self.overwrite_default_bond_features,
            overwrite_default_atom=self.overwrite_default_atom_features,
        )

        self.n_atoms = 1  # start at 1 b/c need index 0 as padding
        self.n_bonds = 1
        self.a_scope = []
        self.b_scope = []
        self.degree_of_polym = []

        # All start with zero padding
        f_atoms = [[0] * self.atom_fdim]
        f_bonds = [[0] * self.bond_fdim]
        a2b = [[]]
        w_atoms = [0]
        w_bonds = [0]
        b2a = [0]
        b2revb = [0]

        for mol_graph in mol_graphs:
            f_atoms.extend(mol_graph.f_atoms)
            f_bonds.extend(mol_graph.f_bonds)
            w_atoms.extend(mol_graph.w_atoms)
            w_bonds.extend(mol_graph.w_bonds)

            for a in range(mol_graph.n_atoms):
                a2b.append([b + self.n_bonds for b in mol_graph.a2b[a]])

            for b in range(mol_graph.n_bonds):
                b2a.append(self.n_atoms + mol_graph.b2a[b])
                b2revb.append(self.n_bonds + mol_graph.b2revb[b])

            self.a_scope.append((self.n_atoms, mol_graph.n_atoms))
            self.b_scope.append((self.n_bonds, mol_graph.n_bonds))
            self.n_atoms += mol_graph.n_atoms
            self.n_bonds += mol_graph.n_bonds

            self.degree_of_polym.append(mol_graph.degree_of_polym)

        self.max_num_bonds = max(1, max(len(in_bonds) for in_bonds in a2b))

        self.f_atoms = paddle.to_tensor(f_atoms, dtype="float32")
        self.f_bonds = paddle.to_tensor(f_bonds, dtype="float32")
        self.w_atoms = paddle.to_tensor(w_atoms, dtype="float32")
        self.w_bonds = paddle.to_tensor(w_bonds, dtype="float32")
        self.a2b = paddle.to_tensor(
            [
                a2b[a] + [0] * (self.max_num_bonds - len(a2b[a]))
                for a in range(self.n_atoms)
            ],
            dtype="int64",
        )
        self.b2a = paddle.to_tensor(b2a, dtype="int64")
        self.b2revb = paddle.to_tensor(b2revb, dtype="int64")
        self.b2b = None
        self.a2a = None

    def get_components(self, atom_messages: bool = False):
        """
        Returns the components of the BatchMolGraph.
        """
        if atom_messages:
            f_bonds = self.f_bonds[
                :,
                -get_bond_fdim(
                    config=None,
                    atom_messages=atom_messages,
                    overwrite_default_atom=self.overwrite_default_atom_features,
                    overwrite_default_bond=self.overwrite_default_bond_features,
                ) :,
            ]
        else:
            f_bonds = self.f_bonds

        return (
            self.f_atoms,
            f_bonds,
            self.w_atoms,
            self.w_bonds,
            self.a2b,
            self.b2a,
            self.b2revb,
            self.a_scope,
            self.b_scope,
            self.degree_of_polym,
        )

    def get_b2b(self) -> paddle.Tensor:
        if self.b2b is None:
            b2b = self.a2b[self.b2a]
            revmask = (b2b != self.b2revb.unsqueeze(1).expand_as(b2b)).astype("int64")
            self.b2b = b2b * revmask
        return self.b2b

    def get_a2a(self) -> paddle.Tensor:
        if self.a2a is None:
            self.a2a = self.b2a[self.a2b]
        return self.a2a


# ---------------------------------------------------------------------------
# mol2graph
# ---------------------------------------------------------------------------


def mol2graph(
    mols: Union[List[str], List[Chem.Mol], List[Tuple[Chem.Mol, Chem.Mol]]],
    atom_features_batch: List[np.array] = (None,),
    bond_features_batch: List[np.array] = (None,),
    overwrite_default_atom_features: bool = False,
    overwrite_default_bond_features: bool = False,
    config: Featurization_parameters = None,
) -> BatchMolGraph:
    """
    Converts a list of SMILES or RDKit molecules to a BatchMolGraph.

    :param mols: A list of SMILES or a list of RDKit molecules.
    :param atom_features_batch: A list of 2D numpy arrays with additional atom features.
    :param bond_features_batch: A list of 2D numpy arrays with additional bond features.
    :param overwrite_default_atom_features: Whether to overwrite default atom descriptors.
    :param overwrite_default_bond_features: Whether to overwrite default bond descriptors.
    :param config: A Featurization_parameters instance (uses default if None).
    :return: A BatchMolGraph containing the combined molecular graph.
    """
    return BatchMolGraph(
        [
            MolGraph(
                mol,
                af,
                bf,
                overwrite_default_atom_features=overwrite_default_atom_features,
                overwrite_default_bond_features=overwrite_default_bond_features,
                config=config,
            )
            for mol, af, bf in zip_longest(
                mols, atom_features_batch, bond_features_batch
            )
        ]
    )
