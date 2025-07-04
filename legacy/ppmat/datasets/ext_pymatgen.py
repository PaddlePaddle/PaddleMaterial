from __future__ import annotations

import abc

import numpy as np
import paddle
import pgl
from pymatgen.core import Element
from pymatgen.core import Molecule
from pymatgen.core import Structure
from pymatgen.optimization.neighbors import find_points_in_spheres


def get_element_list(train_structures: list[Structure | Molecule]) -> tuple[str, ...]:
    """Get the tuple of elements in the training set for atomic features.

    Args:
        train_structures: pymatgen Molecule/Structure object

    Returns:
        Tuple of elements covered in training set
    """
    elements: set[str] = set()
    for s in train_structures:
        elements.update(s.composition.get_el_amt_dict().keys())
    return tuple(sorted(elements, key=lambda el: Element(el).Z))


class GraphConverter(metaclass=abc.ABCMeta):
    """Abstract base class for converters from input crystals/molecules to graphs."""

    @abc.abstractmethod
    def get_graph(self, structure) -> tuple[pgl.Graph, paddle.Tensor, list]:
        """Args:
        structure: Input crystals or molecule.

        Returns:
        Graph object, state_attr
        """

    def get_graph_from_processed_structure_tensor(
        self,
        structure,
        src_id,
        dst_id,
        images,
        lattice_matrix,
        element_types,
        frac_coords,
        is_atoms: bool = False,
    ) -> tuple[pgl.Graph, paddle.Tensor, list]:
        """Construct a pgl graph from processed structure and bond information.

        Args:
            structure: Input crystals or molecule of pymatgen structure or molecule
                types.
            src_id: site indices for starting point of bonds.
            dst_id: site indices for destination point of bonds.
            images: the periodic image offsets for the bonds.
            lattice_matrix: lattice information of the structure.
            element_types: Element symbols of all atoms in the structure.
            frac_coords: Fractional coordinates of all atoms in the structure. Note:
                Cartesian coordinates for molecule
            is_atoms: whether the input structure object is ASE atoms object or not.

        Returns:
            Graph object, state_attr

        """
        u, v = paddle.to_tensor(data=src_id), paddle.to_tensor(data=dst_id)
        g = pgl.Graph((u, v), num_nodes=len(structure))
        pbc_offset = paddle.to_tensor(data=images, dtype="float32")
        g.edge_feat["pbc_offset"] = pbc_offset
        lattice = paddle.to_tensor(data=np.array(lattice_matrix), dtype="float32")
        element_to_index = {elem: idx for idx, elem in enumerate(element_types)}
        node_type = (
            np.array([element_types.index(site.specie.symbol) for site in structure])
            if is_atoms is False
            else np.array(
                [element_to_index[elem] for elem in structure.get_chemical_symbols()]
            )
        )
        g.node_feat["node_type"] = paddle.to_tensor(data=node_type, dtype="int32")
        g.node_feat["frac_coords"] = paddle.to_tensor(data=frac_coords, dtype="float32")
        state_attr = np.array([0.0, 0.0]).astype("float32")
        return g, lattice, state_attr

    def get_graph_from_processed_structure(
        self,
        structure,
        src_id,
        dst_id,
        images,
        lattice_matrix,
        element_types,
        frac_coords,
        is_atoms: bool = False,
    ) -> tuple[pgl.Graph, paddle.Tensor, list]:
        """Construct a pgl graph from processed structure and bond information.

        Args:
            structure: Input crystals or molecule of pymatgen structure or molecule
                types.
            src_id: site indices for starting point of bonds.
            dst_id: site indices for destination point of bonds.
            images: the periodic image offsets for the bonds.
            lattice_matrix: lattice information of the structure.
            element_types: Element symbols of all atoms in the structure.
            frac_coords: Fractional coordinates of all atoms in the structure. Note:
                Cartesian coordinates for molecule
            is_atoms: whether the input structure object is ASE atoms object or not.

        Returns:
            Graph object, state_attr

        """
        # u, v = src_id, dst_id
        edges = [(u, v) for u, v in zip(src_id, dst_id)]
        g = pgl.Graph(edges, num_nodes=len(structure))
        pbc_offset = np.array(images, dtype="float32")
        g.edge_feat["pbc_offset"] = pbc_offset
        lattice = np.array(lattice_matrix, dtype="float32")
        element_to_index = {elem: idx for idx, elem in enumerate(element_types)}
        node_type = (
            np.array([element_types.index(site.specie.symbol) for site in structure])
            if is_atoms is False
            else np.array(
                [element_to_index[elem] for elem in structure.get_chemical_symbols()]
            )
        )
        g.node_feat["node_type"] = np.array(node_type, dtype="int32")
        g.node_feat["frac_coords"] = np.array(frac_coords, dtype="float32")
        state_attr = np.array([0.0, 0.0]).astype("float32")
        return g, lattice, state_attr


class Structure2Graph(GraphConverter):
    """Construct a PGL graph from Pymatgen Structure."""

    def __init__(self, element_types: tuple[str, ...], cutoff: float = 5.0):
        """Parameters
        ----------
        element_types: List of elements present in dataset for graph conversion. This
            ensures all graphs are
            constructed with the same dimensionality of features.
        cutoff: Cutoff radius for graph representation
        """
        self.element_types = tuple(element_types)
        self.cutoff = cutoff

    def get_graph(self, structure: Structure) -> tuple[pgl.Graph, paddle.Tensor, list]:
        """Get a PGL graph from an input Structure.

        :param structure: pymatgen structure object
        :return:
            g: PGL graph
            lat: lattice for periodic systems
            state_attr: state features
        """
        numerical_tol = 1e-08
        pbc = np.array([1, 1, 1], dtype=int)
        element_types = self.element_types
        lattice_matrix = structure.lattice.matrix
        cart_coords = structure.cart_coords
        src_id, dst_id, images, bond_dist = find_points_in_spheres(
            cart_coords,
            cart_coords,
            r=self.cutoff,
            pbc=pbc,
            lattice=lattice_matrix,
            tol=numerical_tol,
        )
        exclude_self = (src_id != dst_id) | (bond_dist > numerical_tol)
        src_id, dst_id, images, bond_dist = (
            src_id[exclude_self],
            dst_id[exclude_self],
            images[exclude_self],
            bond_dist[exclude_self],
        )
        g, lat, state_attr = super().get_graph_from_processed_structure(
            structure,
            src_id,
            dst_id,
            images,
            [lattice_matrix],
            element_types,
            structure.frac_coords,
        )
        return g, lat, state_attr
