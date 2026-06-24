# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

NUM_ELEMENTS: int = 98
NUM_SPACE_GROUPS: int = 230
MAX_WYCKOFF_SITES: int = 27


lattice_parameter_ranges = {
    "mp_20": {
        "min_lattice_length": 2.0,
        "max_lattice_length": 133.0,
        "min_lattice_angle": 60.0,
        "max_lattice_angle": 135.0,
    },
    "mpts_52": {
        "min_lattice_length": 0.98,
        "max_lattice_length": 189.5,
        "min_lattice_angle": 60.0,
        "max_lattice_angle": 135.0,
    },
}
max_atoms_per_dataset = {
    "mp_20": 20,
    "mpts_52": 52,
}

chemical_symbols = [
    # 0
    "X",
    # 1
    "H", "He",
    # 2
    "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    # 3
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    # 4
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr",
    # 5
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "I", "Xe",
    # 6
    "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy",
    "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt",
    "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn",
    # 7
    "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf",
    "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
    "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
]

from ppmat.utils.crystal import OFFSET_LIST

spgroup_data = {
    1: "aP", 2: "aP",
    3: "mP", 4: "mP", 5: "mC", 6: "mP", 7: "mP", 8: "mC", 9: "mC",
    10: "mP", 11: "mP", 12: "mC", 13: "mP", 14: "mP", 15: "mC",
    16: "oP", 17: "oP", 18: "oP", 19: "oP", 20: "oC", 21: "oC",
    22: "oF", 23: "oI", 24: "oI",
    25: "oP", 26: "oP", 27: "oP", 28: "oP", 29: "oP",
    30: "oP", 31: "oP", 32: "oP", 33: "oP", 34: "oP",
    35: "oC", 36: "oC", 37: "oC",
    38: "oA", 39: "oA", 40: "oA", 41: "oA",
    42: "oF", 43: "oF",
    44: "oI", 45: "oI", 46: "oI",
    47: "oP", 48: "oP", 49: "oP", 50: "oP", 51: "oP",
    52: "oP", 53: "oP", 54: "oP", 55: "oP",
    56: "oP", 57: "oP", 58: "oP", 59: "oP", 60: "oP", 61: "oP", 62: "oP",
    63: "oC", 64: "oC", 65: "oC", 66: "oC", 67: "oC", 68: "oC",
    69: "oF", 70: "oF",
    71: "oI", 72: "oI", 73: "oI", 74: "oI",
    75: "tP", 76: "tP", 77: "tP", 78: "tP",
    79: "tI", 80: "tI",
    81: "tP",
    82: "tI",
    83: "tP", 84: "tP", 85: "tP", 86: "tP",
    87: "tI", 88: "tI",
    89: "tP", 90: "tP", 91: "tP", 92: "tP",
    93: "tP", 94: "tP", 95: "tP", 96: "tP",
    97: "tI", 98: "tI",
    99: "tP", 100: "tP", 101: "tP", 102: "tP",
    103: "tP", 104: "tP", 105: "tP", 106: "tP",
    107: "tI", 108: "tI", 109: "tI", 110: "tI",
    111: "tP", 112: "tP", 113: "tP", 114: "tP",
    115: "tP", 116: "tP", 117: "tP", 118: "tP",
    119: "tI", 120: "tI", 121: "tI", 122: "tI",
    123: "tP", 124: "tP", 125: "tP", 126: "tP",
    127: "tP", 128: "tP", 129: "tP", 130: "tP",
    131: "tP", 132: "tP", 133: "tP", 134: "tP",
    135: "tP", 136: "tP", 137: "tP", 138: "tP",
    139: "tI", 140: "tI", 141: "tI", 142: "tI",
    143: "hP", 144: "hP", 145: "hP",
    146: "hR",
    147: "hP", 148: "hR",
    149: "hP", 150: "hP", 151: "hP", 152: "hP",
    153: "hP", 154: "hP",
    155: "hR",
    156: "hP", 157: "hP", 158: "hP", 159: "hP",
    160: "hR", 161: "hR",
    162: "hP", 163: "hP", 164: "hP", 165: "hP",
    166: "hR", 167: "hR",
    168: "hP", 169: "hP", 170: "hP", 171: "hP",
    172: "hP", 173: "hP", 174: "hP",
    175: "hP", 176: "hP",
    177: "hP", 178: "hP", 179: "hP", 180: "hP",
    181: "hP", 182: "hP", 183: "hP", 184: "hP",
    185: "hP", 186: "hP", 187: "hP", 188: "hP",
    189: "hP", 190: "hP",
    191: "hP", 192: "hP", 193: "hP", 194: "hP",
    195: "cP", 196: "cF", 197: "cI",
    198: "cP", 199: "cI",
    200: "cP", 201: "cP",
    202: "cF", 203: "cF",
    204: "cI",
    205: "cP",
    206: "cI",
    207: "cP", 208: "cP",
    209: "cF", 210: "cF",
    211: "cI",
    212: "cP", 213: "cP",
    214: "cI",
    215: "cP",
    216: "cF",
    217: "cI",
    218: "cP",
    219: "cF",
    220: "cI",
    221: "cP", 222: "cP", 223: "cP", 224: "cP",
    225: "cF", 226: "cF", 227: "cF", 228: "cF",
    229: "cI", 230: "cI",
}
