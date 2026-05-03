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

"""
CIF Tokenizer for CrystalLLM.

Ported from lantunes/CrystaLLM (MIT License).
A regex-based tokenizer for Crystallographic Information Files (CIF).
"""

import os
import re

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(THIS_DIR, "spacegroups.txt"), "rt") as f:
    SPACE_GROUPS = [sg.strip() for sg in f.readlines() if sg.strip()]

ATOMS = [
    "Si",
    "C",
    "Pb",
    "I",
    "Br",
    "Cl",
    "Eu",
    "O",
    "Fe",
    "Sb",
    "In",
    "S",
    "N",
    "U",
    "Mn",
    "Lu",
    "Se",
    "Tl",
    "Hf",
    "Ir",
    "Ca",
    "Ta",
    "Cr",
    "K",
    "Pm",
    "Mg",
    "Zn",
    "Cu",
    "Sn",
    "Ti",
    "B",
    "W",
    "P",
    "H",
    "Pd",
    "As",
    "Co",
    "Np",
    "Tc",
    "Hg",
    "Pu",
    "Al",
    "Tm",
    "Tb",
    "Ho",
    "Nb",
    "Ge",
    "Zr",
    "Cd",
    "V",
    "Sr",
    "Ni",
    "Rh",
    "Th",
    "Na",
    "Ru",
    "La",
    "Re",
    "Y",
    "Er",
    "Ce",
    "Pt",
    "Ga",
    "Li",
    "Cs",
    "F",
    "Ba",
    "Te",
    "Mo",
    "Gd",
    "Pr",
    "Bi",
    "Sc",
    "Ag",
    "Rb",
    "Dy",
    "Yb",
    "Nd",
    "Au",
    "Os",
    "Pa",
    "Sm",
    "Be",
    "Ac",
    "Xe",
    "Kr",
    "He",
    "Ne",
    "Ar",
]

DIGITS = [str(d) for d in range(10)]

KEYWORDS = [
    "_cell_length_b",
    "_atom_site_occupancy",
    "_atom_site_attached_hydrogens",
    "_cell_length_a",
    "_cell_angle_beta",
    "_symmetry_equiv_pos_as_xyz",
    "_cell_angle_gamma",
    "_atom_site_fract_x",
    "_symmetry_space_group_name_H-M",
    "_symmetry_Int_Tables_number",
    "_chemical_formula_structural",
    "_chemical_name_systematic",
    "_atom_site_fract_y",
    "_atom_site_symmetry_multiplicity",
    "_chemical_formula_sum",
    "_atom_site_label",
    "_atom_site_type_symbol",
    "_cell_length_c",
    "_atom_site_B_iso_or_equiv",
    "_symmetry_equiv_pos_site_id",
    "_cell_volume",
    "_atom_site_fract_z",
    "_cell_angle_alpha",
    "_cell_formula_units_Z",
    "loop_",
    "data_",
]

EXTENDED_KEYWORDS = [
    "_atom_type_symbol",
    "_atom_type_electronegativity",
    "_atom_type_radius",
    "_atom_type_ionic_radius",
    "_atom_type_oxidation_number",
]

UNK_TOKEN = "<unk>"


class CIFTokenizer:
    """Regex-based tokenizer for CIF (Crystallographic Information File) strings.

    Vocabulary structure:
        - 89 atomic element symbols
        - 10 digits (0-9)
        - 31 CIF keywords (26 standard + 5 extended)
        - 13 symbols (x, y, z, ., (, ), +, -, /, ', comma, space, newline)
        - 227 space group symbols (with _sg suffix for disambiguation)
        - 1 unknown token (<unk>)
        Total: 371 tokens (+ 1 UNK = 372 unique IDs)
    """

    def __init__(self):
        self._tokens = list(self.atoms())
        self._tokens.extend(self.digits())
        self._tokens.extend(self.keywords())
        self._tokens.extend(self.symbols())

        space_groups = list(self.space_groups())
        # Append _sg suffix to disambiguate from atoms
        # (e.g., "Pm" atom vs "Pm" space group)
        space_groups_sg = [sg + "_sg" for sg in space_groups]
        self._tokens.extend(space_groups_sg)

        self._escaped_tokens = [re.escape(token) for token in self._tokens]
        self._escaped_tokens.sort(key=len, reverse=True)

        self._tokens_with_unk = list(self._tokens)
        self._tokens_with_unk.append(UNK_TOKEN)

        self._token_to_id = {ch: i for i, ch in enumerate(self._tokens_with_unk)}
        self._id_to_token = {i: ch for i, ch in enumerate(self._tokens_with_unk)}
        # Map space group IDs back to names without _sg suffix for decoding
        for sg in space_groups_sg:
            self._id_to_token[self._token_to_id[sg]] = sg.replace("_sg", "")

    @staticmethod
    def atoms():
        return ATOMS

    @staticmethod
    def digits():
        return DIGITS

    @staticmethod
    def keywords():
        kws = list(KEYWORDS)
        kws.extend(EXTENDED_KEYWORDS)
        return kws

    @staticmethod
    def symbols():
        return ["x", "y", "z", ".", "(", ")", "+", "-", "/", "'", ",", " ", "\n"]

    @staticmethod
    def space_groups():
        return SPACE_GROUPS

    @property
    def vocab_size(self):
        return len(self._tokens_with_unk)

    @property
    def token_to_id(self):
        return dict(self._token_to_id)

    @property
    def id_to_token(self):
        return dict(self._id_to_token)

    def encode(self, tokens):
        """Encode a list of string tokens to integer IDs."""
        return [self._token_to_id[t] for t in tokens]

    def decode(self, ids):
        """Decode a list of integer IDs to a string."""
        return "".join([self._id_to_token[i] for i in ids])

    def tokenize_cif(self, cif_string, single_spaces=True):
        """Tokenize a CIF string into a list of string tokens.

        Args:
            cif_string: Raw CIF text.
            single_spaces: If True, collapse multiple spaces/tabs to single space.

        Returns:
            List of string tokens.
        """
        # Disambiguate space group names from atom symbols
        spacegroups = "|".join(SPACE_GROUPS)
        cif_string = re.sub(
            rf"(_symmetry_space_group_name_H-M *\b({spacegroups}))\n",
            r"\1_sg\n",
            cif_string,
        )

        token_pattern = "|".join(self._escaped_tokens)
        full_pattern = f"({token_pattern}|\\w+|[\\.,;!?])"

        if single_spaces:
            cif_string = re.sub(r"[ \t]+", " ", cif_string)
        tokens = re.findall(full_pattern, cif_string)

        # Replace unrecognized tokens with UNK
        tokens = [token if token in self._tokens else UNK_TOKEN for token in tokens]

        return tokens
