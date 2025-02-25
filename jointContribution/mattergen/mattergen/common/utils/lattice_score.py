import paddle
from paddle_scatter import scatter_add

from paddle_utils import *  # noqa


def edge_score_to_lattice_score_frac_symmetric(
    score_d: paddle.Tensor,
    edge_index: paddle.Tensor,
    edge_vectors: paddle.Tensor,
    batch: paddle.Tensor,
) -> paddle.Tensor:
    """Converts a score per edge into a score for the atom coordinates and/or the
    lattice matrix via the chain rule. This method explicitly takes into account the
    fact that the cartesian coordinates depend on the lattice via the fractional
    coordinates. Moreover, we make sure to get a symmetric update:
    D_cart_norm @ Phi @ D_cart_norm^T, where Phi is a |E| x |E| diagonal matrix with
    the predicted edge scores

    Args:
        score_d (torch.Tensor, [num_edges,]): A score per edge in the graph.
        edge_index (torch.Tensor, [2, num_edges]): The edge indices in the graph.
        edge_vectors (torch.Tensor, [num_edges, 3]): The vectors connecting the source
            of each edge to the target.
        lattice_matrix (torch.Tensor, [num_nodes, 3, 3]): The lattice matrices for
            each crystal in num_nodes.
        batch (torch.Tensor, [num_nodes,]): The pointer indicating for each atom which
            molecule in the batch it belongs to.

    Returns:
        torch.Tensor: The predicted lattice score.
    """
    batch_edge = batch[edge_index[0]]
    unit_edge_vectors_cart = edge_vectors / edge_vectors.norm(axis=-1, keepdim=True)
    score_lattice = scatter_add(
        score_d[:, None, None]
        * (unit_edge_vectors_cart[:, :, None] @ unit_edge_vectors_cart[:, None, :]),
        batch_edge,
        dim=0,
        dim_size=batch.max() + 1,
    )
    score_lattice = score_lattice.transpose([0, -1, -2])
    return score_lattice
