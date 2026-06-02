from __future__ import annotations

import paddle


def initialized() -> bool:
    """Graph-parallel is disabled in PaddleMaterials UMA local mode."""
    return False


def get_gp_world_size() -> int:
    return 1


def get_gp_rank() -> int:
    return 0


def reduce_from_model_parallel_region(input: paddle.Tensor) -> paddle.Tensor:
    return input


def gather_from_model_parallel_region(input: paddle.Tensor, natoms: int) -> paddle.Tensor:
    del natoms
    return input


def gather_from_model_parallel_region_sum_grad(
    input: paddle.Tensor, natoms: int
) -> paddle.Tensor:
    del natoms
    return input
