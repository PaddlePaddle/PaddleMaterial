from typing import Any, Optional

from paddle_geometric.data import Data
from paddle_geometric.datasets.motif_generator import MotifGenerator
from paddle_geometric.utils import from_networkx


class CustomMotif(MotifGenerator):
    r"""Generates a motif based on a custom structure coming from a
    :class:`paddle_geometric.data.Data` or :class:`networkx.Graph` object.

    Args:
        structure (paddle_geometric.data.Data or networkx.Graph): The structure
            to use as a motif.
    """
    def __init__(self, structure: Any):
        super().__init__()

        self.structure: Optional[Data] = None

        if isinstance(structure, Data):
            self.structure = structure
        else:
            try:
                import networkx as nx
                if isinstance(structure, nx.Graph):
                    self.structure = from_networkx(structure)
            except ImportError:
                pass

        if self.structure is None:
            raise ValueError(f"Expected a motif structure of type "
                             f"'paddle_geometric.data.Data' or 'networkx.Graph'"
                             f"(got {type(structure)})")

    def __call__(self) -> Data:
        assert isinstance(self.structure, Data)
        return self.structure
