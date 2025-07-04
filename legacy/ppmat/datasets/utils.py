import numpy as np
from p_tqdm import p_map
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure

from ppmat.utils.crystal import lattices_to_params_shape_numpy

from rdkit import Chem, RDLogger
from rdkit.Chem.rdchem import Mol

def build_structure_from_str(crystal_str, niggli=True, primitive=False, num_cpus=None):
    """Build crystal structure of pymatgen object from cif string."""

    def build_one(crystal_str):
        crystal = Structure.from_str(crystal_str, fmt="cif")
        if primitive:
            crystal = crystal.get_primitive_structure()
        if niggli:
            crystal = crystal.get_reduced_structure()
        canonical_crystal = Structure(
            lattice=Lattice.from_parameters(*crystal.lattice.parameters),
            species=crystal.species,
            coords=crystal.frac_coords,
            coords_are_cartesian=False,
        )
        return canonical_crystal

    if isinstance(crystal_str, str):
        return build_one(crystal_str)
    elif isinstance(crystal_str, list):
        canonical_crystal = p_map(build_one, crystal_str, num_cpus=num_cpus)
        return canonical_crystal
    else:
        raise TypeError("crystal_str must be str or list.")


def build_structure_from_array(
    crystal_array, niggli=True, primitive=False, num_cpus=None
):
    """Build crystal from cif string."""

    def build_one(crystal_array):

        frac_coords = crystal_array["frac_coords"]
        atom_types = crystal_array["atom_types"]

        if "lengths" in crystal_array and "angles" in crystal_array:
            lengths = crystal_array["lengths"]
            angles = crystal_array["angles"]
        else:
            lattices = crystal_array["lattices"]
            if isinstance(lattices, list):
                lattices = np.asarray(lattices)
                lengths, angles = lattices_to_params_shape_numpy(lattices)

        if isinstance(lengths, np.ndarray):
            lengths = lengths.tolist()
        if isinstance(angles, np.ndarray):
            angles = angles.tolist()

        crystal = Structure(
            lattice=Lattice.from_parameters(*(lengths + angles)),
            species=atom_types,
            coords=frac_coords,
            coords_are_cartesian=False,
        )
        if primitive:
            crystal = crystal.get_primitive_structure()
        if niggli:
            crystal = crystal.get_reduced_structure()
        canonical_crystal = Structure(
            lattice=Lattice.from_parameters(*crystal.lattice.parameters),
            species=crystal.species,
            coords=crystal.frac_coords,
            coords_are_cartesian=False,
        )
        return canonical_crystal

    if isinstance(crystal_array, dict):
        return build_one(crystal_array)
    elif isinstance(crystal_array, list):
        canonical_crystal = p_map(build_one, crystal_array, num_cpus=num_cpus)
        return canonical_crystal
    else:
        raise TypeError("crystal_array must be dict or list.")


def build_structure_from_file(file_names, niggli=True, primitive=False, num_cpus=None):
    """Build crystal structure of pymatgen object from cif string."""

    def build_one(file_names):
        crystal = Structure.from_file(file_names, primitive=primitive)
        if niggli:
            crystal = crystal.get_reduced_structure()
            crystal = Structure(
                lattice=Lattice.from_parameters(*crystal.lattice.parameters),
                species=crystal.species,
                coords=crystal.frac_coords,
                coords_are_cartesian=False,
            )
        return crystal

    if isinstance(file_names, str):
        return build_one(file_names)
    elif isinstance(file_names, list):
        canonical_crystal = p_map(build_one, file_names, num_cpus=num_cpus)
        return canonical_crystal
    else:
        raise TypeError("file_names must be str or list.")


def build_structure_from_dict(
    crystal_dict, niggli=True, primitive=False, num_cpus=None
):
    """Build crystal structure of pymatgen object from cif string."""

    def build_one(crystal_dict):
        crystal = Structure.from_dict(crystal_dict)
        if primitive:
            crystal = crystal.get_primitive_structure()
        if niggli:
            crystal = crystal.get_reduced_structure()
        canonical_crystal = Structure(
            lattice=Lattice.from_parameters(*crystal.lattice.parameters),
            species=crystal.species,
            coords=crystal.frac_coords,
            coords_are_cartesian=False,
        )
        return canonical_crystal

    if isinstance(crystal_dict, dict):
        return build_one(crystal_dict)
    elif isinstance(crystal_dict, list):
        canonical_crystal = p_map(build_one, crystal_dict, num_cpus=num_cpus)
        return canonical_crystal
    else:
        raise TypeError("crystal_dict must be str or list.")


def abs_cap(val, max_abs_val=1):
    """
    Returns the value with its absolute value capped at max_abs_val.
    Particularly useful in passing values to trignometric functions where
    numerical errors may result in an argument > 1 being passed in.
    https://github.com/materialsproject/pymatgen/blob/b789d74639aa851d7e5ee427a765d9fd5a8d1079/pymatgen/util/num.py#L15
    Args:
        val (float): Input value.
        max_abs_val (float): The maximum absolute value for val. Defaults to 1.
    Returns:
        val if abs(val) < 1 else sign of val * max_abs_val.
    """
    return max(min(val, max_abs_val), -max_abs_val)


def lattice_params_to_matrix(a, b, c, alpha, beta, gamma):
    """Converts lattice from abc, angles to matrix.
    https://github.com/materialsproject/pymatgen/blob/b789d74639aa851d7e5ee427a765d9fd5a8d1079/pymatgen/core/lattice.py#L311
    """
    angles_r = np.radians([alpha, beta, gamma])
    cos_alpha, cos_beta, cos_gamma = np.cos(angles_r)
    sin_alpha, sin_beta, sin_gamma = np.sin(angles_r)

    val = (cos_alpha * cos_beta - cos_gamma) / (sin_alpha * sin_beta)
    val = abs_cap(val)
    gamma_star = np.arccos(val)

    vector_a = [a * sin_beta, 0.0, a * cos_beta]
    vector_b = [
        -b * sin_alpha * np.cos(gamma_star),
        b * sin_alpha * np.sin(gamma_star),
        b * cos_alpha,
    ]
    vector_c = [0.0, 0.0, float(c)]
    return np.array([vector_a, vector_b, vector_c])

def build_molecules_from_smiles(smiles_str, remove_h=None, num_cpus=None):
    """Build molecules of rdkit object from smiles."""
    
    RDLogger.DisableLog('rdApp.*')
    
    def build_one(smiles_str):
        mol = Chem.MolFromSmiles(smiles_str)
        return mol
    
    if isinstance(smiles_str, str):
        return build_one(crystal_str)
    elif isinstance(smiles_str, list):
        mols = p_map(build_one, smiles_str, num_cpus=num_cpus)
        return mols
    else:
        raise TypeError("crystal_str must be str or list.")

def numericalize_text(text, vocab_to_id, dim):
    """
    将给定的文本转换为对应的 token id 。

    参数:
        text (str): 输入文本为一个字符串，单词以空格分隔。
        vocab_to_id (dict): 词汇表字典，将单词映射为唯一的 id。
        dim (int): 返回的每个 token id 列表的长度。

    返回:
        list of list: 对应的数值化 token id ，文本对应一个长度为 dim 的 token id 。
    """

    # 如果输入文本为空，则返回长度为 dim 且全为 0 的列表
    if not text:
        token_ids = [0] * dim
    else:
        # 将文本按空格进行分割，生成单词列表
        words = text.split(" ")

        # 使用词汇表字典将每个单词转换为对应的 id，如果不在词汇表中则使用 <unk> 的 id
        token_ids = [vocab_to_id.get(word, vocab_to_id["<unk>"]) for word in words]

        # 如果 token_ids 长度小于 dim，则在后面补充 0
        if len(token_ids) < dim:
            token_ids += [0] * (dim - len(token_ids))
        # 如果 token_ids 长度超过 dim，则截断到 dim 长度
        else:
            token_ids = token_ids[:dim]

    return token_ids