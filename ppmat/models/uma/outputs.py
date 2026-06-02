from __future__ import annotations

"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from typing import Literal

import paddle

from ._compat import gp_utils


def get_l_component_range(x: paddle.Tensor, l_min: int, l_max: int) -> paddle.Tensor:
    """Extract spherical harmonic components for L in [l_min, l_max] from node embeddings.

    The node embeddings are assumed to be organized as [N, (lmax+1)^2, C] where the
    second dimension contains spherical harmonic coefficients ordered by L:
    - L=0: index 0 (1 component)
    - L=1: indices 1-3 (3 components)
    - L=2: indices 4-8 (5 components)
    - etc.

    Args:
        x: Node embeddings tensor of shape [N, (lmax+1)^2, C].
        l_min: Lowest angular momentum quantum number to include (0, 1, 2, ...).
        l_max: Highest angular momentum quantum number to include (>= l_min).

    Returns:
        Tensor of shape [N, (l_max+1)^2 - l_min^2, C] containing the concatenated
        spherical harmonic components for all L in [l_min, l_max].
    """
    start_idx = l_min * l_min
    num_components = (l_max + 1) ** 2 - l_min**2
    return x.narrow(1, start_idx, num_components)


def reduce_node_to_system(
    node_values: paddle.Tensor, batch: paddle.Tensor, num_systems: int
) -> tuple[paddle.Tensor, paddle.Tensor]:
    """Reduce node-level values to system-level by summing over nodes in each system.

    Handles graph-parallel (GP) reduction when GP is initialized.

    Args:
        node_values: Node-level values of shape [N, ...] where N is the number of nodes.
        batch: Batch indices mapping each node to its system, shape [N].
        num_systems: Total number of systems in the batch.

    Returns:
        A tuple of (reduced, unreduced) where:
        - reduced: System-level values after GP reduction (if applicable), shape [num_systems, ...].
        - unreduced: System-level values before GP reduction, useful for autograd, shape [num_systems, ...].
    """
    output_shape = (num_systems,) + node_values.shape[1:]
    system_values = paddle.zeros(output_shape, dtype=paddle.float64)
    if node_values.dim() == 1:
        system_values.index_add_(0, batch, node_values.astype(system_values.dtype))
    else:
        flat_node = node_values.view(node_values.shape[0], -1)
        flat_system = system_values.view(num_systems, -1)
        flat_system.index_add_(0, batch, flat_node.astype(flat_system.dtype))
        system_values = flat_system.view(output_shape)
    if gp_utils.initialized():
        reduced = gp_utils.reduce_from_model_parallel_region(system_values)
    else:
        reduced = system_values
    return reduced, system_values


def compute_energy(
    emb: dict[str, paddle.Tensor],
    energy_block: paddle.nn.Module,
    batch: paddle.Tensor,
    num_systems: int,
    natoms: (paddle.Tensor | None) = None,
    reduce: Literal["sum", "mean"] = "sum",
) -> tuple[paddle.Tensor, paddle.Tensor]:
    """Compute system-level energy from node embeddings and an energy block.

    Extracts the L=0 (scalar) component from node embeddings, applies the energy
    block to produce per-node energies, reduces to system-level energies, and
    optionally normalizes by the number of atoms per system.

    Args:
        emb: Embedding dictionary containing "node_embedding" of shape [N, (lmax+1)^2, C].
        energy_block: Module that maps scalar node features [N, C] to per-node energies [N, 1].
        batch: Batch indices mapping each node to its system, shape [N].
        num_systems: Total number of systems in the batch.
        natoms: Number of atoms per system, shape [num_systems]. Required when reduce="mean".
        reduce: How to aggregate node energies into system energies. "sum" returns the total
            energy; "mean" divides by natoms to return the average energy per atom.

    Returns:
        A tuple of (energy, energy_part) where:
        - energy: System-level energy after GP reduction and reduce, shape [num_systems].
        - energy_part: System-level energy before GP reduction (for autograd), shape [num_systems].
    """
    scalar_embedding = get_l_component_range(
        emb["node_embedding"], l_min=0, l_max=0
    ).squeeze(1)
    node_energy = energy_block(scalar_embedding)
    node_energy_flat = node_energy.view(-1)
    energy, energy_part = reduce_node_to_system(node_energy_flat, batch, num_systems)
    if reduce == "sum":
        pass
    elif reduce == "mean":
        if natoms is None:
            raise ValueError("natoms must be provided when reduce='mean'")
        energy = energy / natoms.astype(energy.dtype)
    else:
        raise ValueError(f"reduce can only be sum or mean, got: {reduce}")
    return energy, energy_part


def compute_forces(
    energy_part: paddle.Tensor, pos: paddle.Tensor, training: bool = True
) -> paddle.Tensor:
    """Compute forces as negative gradient of energy with respect to positions.

    Args:
        energy_part: System-level energy before GP reduction, shape [num_systems].
        pos: Atomic positions, shape [N, 3].
        training: Whether to create graph for higher-order gradients.

    Returns:
        Forces tensor of shape [N, 3].
    """
    (grad,) = paddle.grad(outputs=energy_part.sum(), inputs=pos, create_graph=training)
    forces = paddle.neg(x=grad)
    if gp_utils.initialized():
        forces = gp_utils.reduce_from_model_parallel_region(forces)
    return forces


def compute_forces_and_stress(
    energy_part: paddle.Tensor,
    pos: paddle.Tensor,
    cell: paddle.Tensor,
    batch: paddle.Tensor,
    training: bool = True,
) -> tuple[paddle.Tensor, paddle.Tensor]:
    """Compute forces and stress from energy using autograd.

    Forces are computed as the negative gradient of energy with respect to positions.
    Stress is computed by reconstructing the virial tensor from the position and cell
    gradients, equivalent to the strain-derivative approach.

    The virial is:
        V = (g_r^T r + r^T g_r) / 2 + (cell^T g_h + g_h^T cell) / 2

    where g_r = dE/dpos and g_h = dE/dcell. This matches dE/dε for a symmetric
    strain ε applied as r' = r(I + ε), h' = h(I + ε).

    Args:
        energy_part: System-level energy before GP reduction, shape [num_systems].
        pos: Atomic positions, shape [N, 3].
        cell: Unit cell vectors, shape [num_systems, 3, 3].
        batch: Batch indices mapping each node to its system, shape [N].
        training: Whether to create graph for higher-order gradients.

    Returns:
        A tuple of (forces, stress) where:
        - forces: Shape [N, 3].
        - stress: Shape [num_systems, 9] (flattened 3x3 tensor).
    """
    grads = paddle.grad(
        outputs=[energy_part.sum()], inputs=[pos, cell], create_graph=training
    )
    if gp_utils.initialized():
        grads = gp_utils.reduce_from_model_parallel_region(
            grads[0]
        ), gp_utils.reduce_from_model_parallel_region(grads[1])
    num_systems = cell.shape[0]
    forces = paddle.neg(x=grads[0])
    pos_virial_per_atom = grads[0].unsqueeze(2) * pos.unsqueeze(1)
    pos_virial, _ = reduce_node_to_system(pos_virial_per_atom, batch, num_systems)
    cell_virial = cell.mT @ grads[1]
    virial = (pos_virial + pos_virial.mT + cell_virial + cell_virial.mT) / 2
    volume = paddle.linalg.det(x=cell).abs().unsqueeze(-1)
    stress = virial / volume.view(-1, 1, 1)
    stress = stress.view(-1, 9)
    return forces, stress


def compute_hessian_vmap(
    forces_flat: paddle.Tensor, pos: paddle.Tensor, create_graph: bool
) -> paddle.Tensor:
    """Compute Hessian using vectorized mapping (vmap)."""

    def compute_grad_component(vec):
        return paddle.grad(
            outputs=-1 * forces_flat,
            inputs=pos,
            grad_outputs=vec,
            retain_graph=True,
            create_graph=create_graph,
        )[0]

    basis = paddle.eye(forces_flat.shape[0], dtype=forces_flat.dtype)
    if hasattr(paddle, "vmap"):
        hessian = paddle.vmap(compute_grad_component)(basis)
    else:
        hessian = paddle.stack([compute_grad_component(basis[i]) for i in range(basis.shape[0])], axis=0)

    n_dof = int(forces_flat.numel())
    return hessian.reshape([n_dof, n_dof])


def compute_hessian_loop(
    forces_flat: paddle.Tensor, pos: paddle.Tensor, create_graph: bool
) -> paddle.Tensor:
    """Compute Hessian using a loop over force components.

    Iteratively computes gradients of each force component with respect to
    positions. This is a fallback when vmap is not desired or unavailable.

    Args:
        forces_flat: Flattened forces tensor, shape [N*3].
        pos: Atomic positions, shape [N, 3].
        create_graph: Whether to create graph for third-order derivatives.

    Returns:
        Hessian matrix of shape [N*3, N*3].
    """
    n_forces = len(forces_flat)
    hessian = paddle.zeros([n_forces, n_forces], dtype=forces_flat.dtype)
    for i in range(n_forces):
        hessian[:, i] = paddle.grad(
            outputs=-forces_flat[i],
            inputs=pos,
            retain_graph=i < n_forces - 1,
            create_graph=create_graph,
        )[0].flatten()
    return hessian


def compute_hessian(
    forces: paddle.Tensor, pos: paddle.Tensor, vmap: bool = True, training: bool = False
) -> paddle.Tensor:
    """Compute Hessian matrix as second derivative of energy w.r.t. positions.

    The Hessian is computed as the negative gradient of forces with respect to
    positions: H = -∇_pos(forces) = ∇²_pos(energy).

    Args:
        forces: Force tensor, shape [N, 3].
        pos: Atomic positions, shape [N, 3].
        vmap: Whether to use vectorized mapping (faster but higher memory).
        training: Whether to create graph for third-order derivatives.

    Returns:
        Hessian matrix of shape [1, N*3, N*3] (batch dim always 1 since
        hessian requires single-system batches).

    Note:
        Graph parallel (GP) mode is not fully supported. The Hessian should
        be computed after forces have been reduced across GP ranks.
    """
    if gp_utils.initialized():
        raise NotImplementedError(
            "Hessian computation is not currently supported with graph parallel mode. Please compute Hessian on a single rank after force reduction."
        )
    forces_flat = forces.flatten()
    if vmap:
        hessian = compute_hessian_vmap(forces_flat, pos, create_graph=training)
    else:
        hessian = compute_hessian_loop(forces_flat, pos, create_graph=training)
    return hessian.unsqueeze(0)
