import paddle


def to_discrete_time(t: paddle.Tensor, N: int, T: float) -> paddle.int64:
    """Convert continuous time to integer timestep.

    Args:
        t: continuous time between 0 and T
        N: number of timesteps
        T: max time
    Returns:
        Integer timesteps between 0 and N-1
    """
    return (t * (N - 1) / T).astype(dtype="int64")
