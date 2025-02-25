from typing import Protocol
from typing import Sequence

import paddle
from pymatgen.core import Composition

from mattergen.common.data.chemgraph import ChemGraph
from mattergen.common.utils.data_utils import compute_lattice_polar_decomposition
from mattergen.common.utils.data_utils import get_element_symbol
from mattergen.common.utils.globals import MAX_ATOMIC_NUM


class Transform(Protocol):
    def __call__(self, sample: ChemGraph) -> ChemGraph:
        ...


def symmetrize_lattice(sample: ChemGraph) -> ChemGraph:
    return sample.replace(cell=compute_lattice_polar_decomposition(sample.cell))


def set_chemical_system(sample: ChemGraph) -> ChemGraph:
    chemsys = (
        paddle.eye(num_rows=MAX_ATOMIC_NUM + 1)[sample.atomic_numbers].sum(axis=0) > 0
    ).astype(dtype="float32")[None]
    return sample.replace(chemical_system=chemsys)


def set_chemical_system_string(sample: ChemGraph) -> ChemGraph:
    return sample.replace(
        chemical_system=Composition(
            {get_element_symbol(Z=i.item()): (1) for i in sample.atomic_numbers}
        ).chemical_system
    )


class SetProperty:
    def __init__(self, property_name: str, value: (float | Sequence[str])):
        self.property_name = property_name
        self.value = (
            paddle.to_tensor(data=value, dtype="float32")
            if isinstance(value, float) or isinstance(value, int)
            else value
        )

    def __call__(self, sample: ChemGraph) -> ChemGraph:
        return sample.replace(**{self.property_name: self.value})
