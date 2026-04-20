from collections import Counter

import numpy as np
from rdkit import Chem


def make_mol(s: str, keep_h: bool, add_h: bool):
    """
    Builds an RDKit molecule from a SMILES string.

    :param s: SMILES string.
    :param keep_h: Boolean whether to keep hydrogens in the input smiles.
    :param add_h: Boolean whether to add hydrogens.
    :return: RDKit molecule.
    """
    if keep_h:
        mol = Chem.MolFromSmiles(s, sanitize=False)
        Chem.SanitizeMol(
            mol,
            sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL
            ^ Chem.SanitizeFlags.SANITIZE_ADJUSTHS,
        )
    else:
        mol = Chem.MolFromSmiles(s)
    if add_h:
        mol = Chem.AddHs(mol)
    return mol


def make_polymer_mol(smiles: str, keep_h: bool, add_h: bool, fragment_weights: list):
    """
    Builds an RDKit molecule from a polymer SMILES string with fragment weights.

    :param smiles: SMILES string (fragments separated by '.').
    :param keep_h: Boolean whether to keep hydrogens in the input smiles.
    :param add_h: Boolean whether to add hydrogens.
    :param fragment_weights: List of monomer fractions for each fragment.
    :return: RDKit molecule.
    """
    num_frags = len(smiles.split("."))
    if len(fragment_weights) != num_frags:
        raise ValueError(
            f"number of input monomers/fragments ({num_frags}) does not match number of "
            f"input number of weights ({len(fragment_weights)})"
        )

    mols = []
    for s, w in zip(smiles.split("."), fragment_weights):
        m = make_mol(s, keep_h, add_h)
        for a in m.GetAtoms():
            a.SetDoubleProp("w_frag", float(w))
        mols.append(m)

    mol = mols.pop(0)
    while len(mols) > 0:
        m2 = mols.pop(0)
        mol = Chem.CombineMols(mol, m2)

    return mol


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
