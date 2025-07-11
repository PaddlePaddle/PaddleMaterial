from typing import Literal
from typing import Optional

import numpy as np
from p_tqdm import p_map
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure

from ppmat.utils.crystal import lattices_to_params_shape_numpy


class BuildStructure:
    """Build crystal structure from different formats, including cif string, array,
    cif file, and dict.

    Args:
        format (Literal["cif_str", "array", "cif_file", "dict"]): The format of the
            crystal data.
            - "cif_str": Crystal data in CIF format as a string.
            - "array": Crystal data in array format.
            - "cif_file": Crystal data in CIF file path.
            - "dict": Crystal data in dictionary format.
        primitive (bool, optional): Whether to return the primitive or conventional
            unit cell. Defaults to False.
        niggli (bool, optional): Whether to reduce the lattice using Niggli's
            algorithm. Defaults to True.
        num_cpus (Optional[int], optional): Number of CPUs to use for parallel
            processing. Defaults to None.
    """

    def __init__(
        self,
        format: Literal["cif_str", "array", "cif_file", "dict"],
        primitive: bool = False,
        niggli: bool = True,
        num_cpus: Optional[int] = None,
    ):

        self.format = format
        self.niggli = niggli
        self.primitive = primitive
        self.num_cpus = num_cpus if num_cpus is not None else 1

    @staticmethod
    def build_one(crystal_data, format, primitive=False, niggli=True):
        if format == "cif_str":
            crystal = Structure.from_str(crystal_data, fmt="cif")
        elif format == "array":

            frac_coords = crystal_data["frac_coords"]
            atom_types = crystal_data["atom_types"]

            if "lengths" in crystal_data and "angles" in crystal_data:
                lengths = crystal_data["lengths"]
                angles = crystal_data["angles"]
            else:
                lattices = crystal_data["lattices"]
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
        elif format == "cif_file":
            crystal = Structure.from_file(crystal_data)
        elif format == "dict":
            crystal = Structure.from_dict(crystal_data)
        else:
            raise ValueError(f"Invalid format specified: {format}")

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

    def __call__(self, crystals_data):
        if isinstance(crystals_data, list):
            canonical_crystal = p_map(
                BuildStructure.build_one,
                crystals_data,
                [self.format] * len(crystals_data),
                [self.primitive] * len(crystals_data),
                [self.niggli] * len(crystals_data),
                num_cpus=self.num_cpus,
            )
            return canonical_crystal
        else:
            return BuildStructure.build_one(
                crystals_data,
                self.format,
                self.primitive,
                self.niggli,
                num_cpus=self.num_cpus,
            )
