import paddle
from mattergen.diffusion.wrapped.wrapped_normal_loss import (
    get_pbc_offsets, wrapped_normal_score)
from mattergen.diffusion.wrapped.wrapped_sde import wrap_at_boundary


def test_wrapped_normal_score_isotropic():
    variance = paddle.rand(shape=(1,)) * 5
    max_offsets = 3
    num_atoms = 1
    batch = paddle.zeros(shape=num_atoms, dtype="int64")
    cell = paddle.to_tensor(
        data=[[[3.4641, 0.0, 2.0], [-1.1196, 1.6572, 0], [0.0, 0.0, 3.0]]]
    )
    lattice_offsets = get_pbc_offsets(cell, max_offsets)
    mean = paddle.zeros(shape=(3,))
    shifted_means = mean[None, None] + lattice_offsets
    normal_distributions = paddle.distribution.Normal(
        loc=shifted_means[0], scale=variance.sqrt().item()
    )
    noisy_frac_coords = paddle.rand(shape=(num_atoms, 3))
    noisy_cart_coords = wrap_at_boundary(noisy_frac_coords, 1.0)
    noisy_cart_coords.stop_gradient = not True
    comp_scores = wrapped_normal_score(
        noisy_cart_coords,
        mean[None],
        cell,
        variance.tile(repeat_times=num_atoms),
        batch,
        max_offsets,
    )
>>>>>>    mix = torch.distributions.Categorical(
        probs=paddle.ones(shape=tuple(shifted_means.shape)[1])
    )
    comp = paddle.distribution.Independent(
        base=normal_distributions, reinterpreted_batch_rank=1
    )
>>>>>>    gmm = torch.distributions.MixtureSameFamily(mix, comp)
    gmm.log_prob(noisy_cart_coords).backward()
    coord_score = noisy_cart_coords.grad
    assert paddle.allclose(x=coord_score, y=comp_scores, atol=1e-05).item()
