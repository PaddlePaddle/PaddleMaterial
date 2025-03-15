import math

import numpy as np
import paddle
import paddle.nn.functional as F

from .utils import diffgraphformer_utils as utils


def sum_except_batch(x):
    x_reshaped = paddle.reshape(x, [x.shape[0], -1])
    return paddle.sum(x_reshaped, axis=-1)


def assert_correctly_masked(variable, node_mask):
    mask_int = node_mask.astype("int64")
    masked = variable * (1 - mask_int).astype(variable.dtype)
    if paddle.max(paddle.abs(masked)).item() >= 1e-4:
        raise ValueError("Variables not masked properly.")


def sample_gaussian(size):
    return paddle.randn(shape=size)


def sample_gaussian_with_mask(size, node_mask):
    x = paddle.randn(shape=size)
    x = x.astype(node_mask.dtype)
    x_masked = x * node_mask
    return x_masked


def clip_noise_schedule(alphas2, clip_value=0.001):
    """
    For a noise schedule given by alpha^2, this clips alpha_t / alpha_t-1.
    This may help improve stability during sampling.
    """
    alphas2 = np.concatenate([np.ones(1), alphas2], axis=0)
    alphas_step = alphas2[1:] / alphas2[:-1]

    alphas_step = np.clip(alphas_step, a_min=clip_value, a_max=1.0)
    alphas2 = np.cumprod(alphas_step, axis=0)

    return alphas2


def cosine_beta_schedule(timesteps, s=0.008, raise_to_power: float = 1):
    """
    Cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = np.clip(betas, a_min=0, a_max=0.999)
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)

    if raise_to_power != 1:
        alphas_cumprod = np.power(alphas_cumprod, raise_to_power)

    return alphas_cumprod


def cosine_beta_schedule_discrete(timesteps, s=0.008):
    """Cosine schedule as proposed in https://openreview.net/forum?id=-NEXDKk8gZ."""
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(0.5 * np.pi * ((x / steps) + s) / (1 + s)) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
    betas = 1 - alphas
    return betas.squeeze()


def custom_beta_schedule_discrete(timesteps, average_num_nodes=50, s=0.008):
    """
    Cosine schedule with modifications for a discrete setting.
    """
    steps = timesteps + 2
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(0.5 * np.pi * ((x / steps) + s) / (1 + s)) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = alphas_cumprod[1:] / alphas_cumprod[:-1]
    betas = 1 - alphas

    assert timesteps >= 100

    p = 4 / 5  # 1 - 1 / num_edge_classes
    num_edges = average_num_nodes * (average_num_nodes - 1) / 2

    # First 100 steps: only a few updates per graph
    updates_per_graph = 1.2
    beta_first = updates_per_graph / (p * num_edges)

    betas[betas < beta_first] = beta_first
    return np.array(betas)


def gaussian_KL(q_mu, q_sigma):
    """
    KL divergence between a normal distribution (q) and
    the standard normal distribution.
    """
    # 原: sum_except_batch((torch.log(1 / q_sigma) + 0.5*(q_sigma**2 + q_mu**2) - 0.5))
    inside = paddle.log(1.0 / q_sigma) + 0.5 * (q_sigma**2 + q_mu**2) - 0.5
    return sum_except_batch(inside)


def cdf_std_gaussian(x):
    return 0.5 * (1.0 + paddle.erf(x / math.sqrt(2)))


def SNR(gamma):
    """Computes signal to noise ratio (alpha^2/sigma^2) given gamma."""
    return paddle.exp(-gamma)


def inflate_batch_array(array, target_shape):
    """
    Inflates the batch array (array) with only a single axis (batch_size, ...)
    to match the target shape.
    """
    shape0 = array.shape[0]
    new_shape = [shape0] + [1] * (len(target_shape) - 1)
    return paddle.reshape(array, new_shape)


def sigma(gamma, target_shape):
    """Computes sigma given gamma."""
    sig = paddle.sqrt(F.sigmoid(gamma))
    return inflate_batch_array(sig, target_shape)


def alpha(gamma, target_shape):
    """Computes alpha given gamma."""
    sig = paddle.sqrt(F.sigmoid(-gamma))
    return inflate_batch_array(sig, target_shape)


def check_mask_correct(variables, node_mask):
    for var in variables:
        if var.numel() > 0:  # 说明张量非空
            assert_correctly_masked(var, node_mask)


def check_tensor_same_size(*args):
    for i, arg in enumerate(args):
        if i == 0:
            continue
        if args[0].shape != arg.shape:
            raise ValueError("Tensors have different shapes.")


def sigma_and_alpha_t_given_s(
    gamma_t: paddle.Tensor, gamma_s: paddle.Tensor, target_size: paddle.shape
):
    """
    Computes sigma_t_given_s and alpha_t_given_s for sampling.
    """
    part = paddle.softplus(gamma_s) - paddle.softplus(gamma_t)
    sigma2_t = -paddle.expm1(part)
    sigma2_t_given_s = inflate_batch_array(sigma2_t, target_size)

    log_alpha2_t = F.logsigmoid(-gamma_t)
    log_alpha2_s = F.logsigmoid(-gamma_s)
    log_alpha2_t_given_s = log_alpha2_t - log_alpha2_s

    alpha_t_given_s_ = paddle.exp(0.5 * log_alpha2_t_given_s)
    alpha_t_given_s_ = inflate_batch_array(alpha_t_given_s_, target_size)

    sigma_t_given_s = paddle.sqrt(sigma2_t_given_s)

    return sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s_


def reverse_tensor(x):
    idx = paddle.arange(x.shape[0] - 1, -1, -1, dtype="int64")
    return paddle.index_select(x, index=idx, axis=0)


def sample_feature_noise(X_size, E_size, y_size, node_mask):
    """
    Standard normal noise for all features. Output size: X.size(), E.size(), y.size().
    """
    epsX = sample_gaussian(X_size)
    epsE = sample_gaussian(E_size)
    epsy = sample_gaussian(y_size)

    float_mask = node_mask.astype("float32")
    epsX = epsX.astype(float_mask.dtype)
    epsE = epsE.astype(float_mask.dtype)
    epsy = epsy.astype(float_mask.dtype)

    # Get upper triangular part of edge noise, without main diagonal
    upper_triangular_mask = paddle.zeros_like(epsE)

    row_idx, col_idx = np.triu_indices(n=epsE.shape[1], k=1)
    row_idx_t = paddle.to_tensor(row_idx, dtype="int64")
    col_idx_t = paddle.to_tensor(col_idx, dtype="int64")

    for b in range(epsE.shape[0]):
        upper_triangular_mask[b, row_idx_t, col_idx_t, :] = 1.0

    epsE = epsE * upper_triangular_mask
    epsE_T = paddle.transpose(epsE, perm=[0, 2, 1, 3])
    epsE = epsE + epsE_T

    # assert (epsE == torch.transpose(epsE, 1, 2)).all()
    # Paddle :
    eq_ = paddle.all(epsE == epsE_T)
    if not eq_.item():
        raise ValueError("epsE is not symmetric!")

    return utils.PlaceHolder(X=epsX, E=epsE, y=epsy).mask(node_mask)


def sample_normal(mu_X, mu_E, mu_y, sigma_, node_mask):
    """
    Samples from a Normal distribution.
    """
    eps = sample_feature_noise(mu_X.shape, mu_E.shape, mu_y.shape, node_mask)
    eps = eps.astype(mu_X.dtype)  # 如果需要与 mu_X 同 dtype

    X = mu_X + sigma_ * eps.X
    E = mu_E + paddle.unsqueeze(sigma_, 1) * eps.E
    y = mu_y + paddle.squeeze(sigma_, axis=1) * eps.y
    return utils.PlaceHolder(X=X, E=E, y=y)


def check_issues_norm_values(gamma_func, norm_val1, norm_val2, num_stdevs=8):
    """
    Check if 1 / norm_value is still larger than 10 * standard deviation.
    """
    zeros = paddle.zeros([1, 1], dtype="float32")
    gamma_0 = gamma_func(zeros)
    # sigma_0:
    sig0 = sigma(gamma_0, zeros.shape).item()
    max_norm_value = max(norm_val1, norm_val2)
    if sig0 * num_stdevs > 1.0 / max_norm_value:
        raise ValueError(
            f"Value for normalization {max_norm_value} too large "
            f"with sigma_0={sig0:.5f}."
        )


def sample_discrete_features(probX, probE, node_mask):
    """Sample features from multinomial distribution with given probabilities
        (probX, probE).

    Args:
        probX: node features with shape (bs, n, dx_out)
        probE: edge features with shape (bs, n, n, de_out)
        node_mask: node mask
    """
    bs, n, _ = probX.shape

    # Noise X
    # The masked rows should define probability distributions as well
    probX = paddle.where(
        node_mask.unsqueeze(-1), probX, paddle.full_like(probX, 1 / probX.shape[-1])
    )

    # Flatten the probability tensor to sample with multinomial
    probX = probX.reshape([bs * n, -1])  # (bs * n, dx_out)

    # Sample X
    X_t = paddle.multinomial(probX, num_samples=1).reshape([bs, n])  # (bs, n)

    # Noise E
    # The masked rows should define probability distributions as well
    inverse_edge_mask = ~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2)).unsqueeze(-1)
    diag_mask = paddle.eye(n).unsqueeze(0).expand([bs, -1, -1]).unsqueeze(-1)

    probE = paddle.where(
        inverse_edge_mask, paddle.full_like(probE, 1 / probE.shape[-1]), probE
    )
    probE = paddle.where(
        diag_mask.astype(paddle.bool),
        paddle.full_like(probE, 1 / probE.shape[-1]),
        probE,
    )

    probE = probE.reshape([bs * n * n, -1])  # (bs * n * n, de_out)

    # Sample E
    E_t = paddle.multinomial(probE, num_samples=1).reshape([bs, n, n])  # (bs, n, n)
    E_t = paddle.triu(E_t, diagonal=1)
    E_t = E_t + paddle.transpose(E_t, [0, 2, 1])

    # Create a placeholder for y, since it's not used in this function
    y = paddle.zeros([bs, 0], dtype=X_t.dtype)

    return utils.PlaceHolder(X=X_t, E=E_t, y=y)


def compute_posterior_distribution(M, M_t, Qt_M, Qsb_M, Qtb_M):
    """
    M, M_t: shape (bs, N, d) or (bs, N) and flattened
    compute xt @ Qt.T * x0 @ Qsb / x0 @ Qtb @ xt.T
    """
    # Flatten
    bs = M.shape[0]
    M_flat = paddle.reshape(M, [bs, -1, M.shape[-1]])  # e.g. (bs, N, d)
    M_t_flat = paddle.reshape(M_t, [bs, -1, M_t.shape[-1]]).astype("float32")

    Qt_M_T = paddle.transpose(Qt_M, perm=[0, 2, 1])  # (bs, d, d)

    left_term = paddle.matmul(M_t_flat, Qt_M_T)  # (bs, N, d)
    right_term = paddle.matmul(M_flat, Qsb_M)  # (bs, N, d)
    product = left_term * right_term  # (bs, N, d)

    denom = paddle.matmul(M_flat, Qtb_M)  # (bs, N, d)
    denom = paddle.sum(denom * M_t_flat, axis=-1)  # (bs, N)

    denom_ = paddle.unsqueeze(denom, axis=-1)  # (bs, N, 1)
    # avoid zero div
    zero_mask = denom_ == 0.0
    denom_ = paddle.where(zero_mask, paddle.ones_like(denom_), denom_)

    prob = product / denom_
    return prob


def compute_batched_over0_posterior_distribution_(X_t, Qt, Qsb, Qtb):
    """
    Compute xt @ Qt.T * x0 @ Qsb / x0 @ Qtb @ xt.T for each possible value of x0
    """
    X_t = X_t.astype("float32")
    Qt_T = paddle.transpose(Qt, perm=[0, 2, 1]).astype("float32")

    left_term = paddle.matmul(X_t, Qt_T)  # (bs, N, d_t-1)
    left_term = paddle.unsqueeze(left_term, axis=2)  # (bs, N, 1, d_t-1)

    right_term = paddle.unsqueeze(Qsb, axis=1)  # (bs, 1, d0, d_t-1)
    numerator = left_term * right_term  # (bs, N, d0, d_t-1)

    denominator = paddle.matmul(
        Qtb, paddle.transpose(X_t, perm=[0, 2, 1])
    )  # (bs, d0, N)
    denominator = paddle.transpose(denominator, perm=[0, 2, 1])  # (bs, N, d0)
    denominator = paddle.unsqueeze(denominator, axis=-1)  # (bs, N, d0, 1)

    zero_mask = denominator == 0.0
    denominator = paddle.where(zero_mask, paddle.ones_like(denominator), denominator)

    return numerator / denominator


def compute_batched_over0_posterior_distribution(X_t, Qt, Qsb, Qtb):
    """
    Flatten edge features to (bs, N, dt).
    Then compute the posterior distribution. (Same logic as the '_' version)
    """
    # Flatten
    X_t_f = X_t.flatten(start_axis=1, stop_axis=-2).astype("float32")  # (bs, N, dt)
    Qt_T = paddle.transpose(Qt, perm=[0, 2, 1])  # (bs, dt, d_t-1)

    left_term = paddle.matmul(X_t_f, Qt_T)  # (bs, N, d_t-1)
    left_term = paddle.unsqueeze(left_term, axis=2)  # (bs, N, 1, d_t-1)
    right_term = paddle.unsqueeze(Qsb, axis=1)  # (bs, 1, d0, d_t-1)
    numerator = left_term * right_term  # (bs, N, d0, d_t-1)

    X_t_transposed = paddle.transpose(X_t_f, perm=[0, 2, 1])  # (bs, dt, N)
    prod = paddle.matmul(Qtb, X_t_transposed)  # (bs, d0, N)
    prod = paddle.transpose(prod, perm=[0, 2, 1])  # (bs, N, d0)

    denominator = paddle.unsqueeze(prod, axis=-1)  # (bs, N, d0, 1)
    zero_mask = denominator == 0
    denominator = paddle.where(
        zero_mask, paddle.full_like(denominator, 1e-6), denominator
    )

    return numerator / denominator


def mask_distributions(true_X, true_E, pred_X, pred_E, node_mask):
    """
    Set masked rows to arbitrary distributions, so they don't contribute to loss.
    Then renormalize.
    """
    dtype_ = true_X.dtype

    row_X = paddle.zeros([true_X.shape[-1]], dtype=dtype_)
    row_X[0] = 1.0
    row_E = paddle.zeros([true_E.shape[-1]], dtype=dtype_)
    row_E[0] = 1.0

    n_ = node_mask.shape[1]
    diag_mask = paddle.eye(n_, dtype="int32").astype("bool")
    diag_mask_bs = diag_mask.unsqueeze(0).expand([node_mask.shape[0], n_, n_])

    mask_bool = node_mask.astype("bool")
    mask_not = paddle.logical_not(mask_bool)

    row_X_bc = row_X.unsqueeze(0).unsqueeze(0)  # shape (1,1,dx)
    row_X_bc = paddle.expand(
        row_X_bc, [mask_not.shape[0], mask_not.shape[1], row_X.shape[0]]
    )

    true_X = paddle.where(paddle.unsqueeze(mask_not, axis=-1), row_X_bc, true_X)
    pred_X = paddle.where(paddle.unsqueeze(mask_not, axis=-1), row_X_bc, pred_X)

    # Edge
    mask_2d = paddle.unsqueeze(mask_bool, axis=1) & paddle.unsqueeze(mask_bool, axis=2)
    inv_mask_2d = paddle.logical_not(mask_2d)
    # + diag => unify
    comb_mask = paddle.logical_or(inv_mask_2d, diag_mask_bs)

    row_E_bc = row_E.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # shape (1,1,1,de)
    row_E_bc = paddle.expand(
        row_E_bc,
        [comb_mask.shape[0], comb_mask.shape[1], comb_mask.shape[2], row_E.shape[0]],
    )

    true_E = paddle.where(paddle.unsqueeze(comb_mask, axis=-1), row_E_bc, true_E)
    pred_E = paddle.where(paddle.unsqueeze(comb_mask, axis=-1), row_E_bc, pred_E)

    # + 1e-7
    eps_ = 1e-7
    true_X = true_X + eps_
    pred_X = pred_X + eps_
    true_E = true_E + eps_
    pred_E = pred_E + eps_

    # normalize
    sum_true_X = paddle.sum(true_X, axis=-1, keepdim=True)
    sum_pred_X = paddle.sum(pred_X, axis=-1, keepdim=True)
    sum_true_E = paddle.sum(true_E, axis=-1, keepdim=True)
    sum_pred_E = paddle.sum(pred_E, axis=-1, keepdim=True)

    true_X = true_X / sum_true_X
    pred_X = pred_X / sum_pred_X
    true_E = true_E / sum_true_E
    pred_E = pred_E / sum_pred_E

    return true_X, true_E, pred_X, pred_E


def posterior_distributions(X, E, y, X_t, E_t, y_t, Qt, Qsb, Qtb):
    """
    Compute posterior distribution for X, E.
    """
    prob_X = compute_posterior_distribution(
        M=X, M_t=X_t, Qt_M=Qt.X, Qsb_M=Qsb.X, Qtb_M=Qtb.X
    )  # (bs, n, dx)
    prob_E = compute_posterior_distribution(
        M=E, M_t=E_t, Qt_M=Qt.E, Qsb_M=Qsb.E, Qtb_M=Qtb.E
    )  # (bs, n*n, de)
    return utils.PlaceHolder(X=prob_X, E=prob_E, y=y_t)


def sample_discrete_feature_noise(limit_dist, node_mask):
    """
    Sample from the limit distribution of the diffusion process
    (multinomial with prob = limit_dist).
    """
    bs, n_max = node_mask.shape
    # x_limit => shape (bs, n_max, dx)
    x_limit = paddle.unsqueeze(limit_dist.X, axis=0)  # (1, dx)
    x_limit = paddle.unsqueeze(x_limit, axis=0)  # (1,1,dx)
    x_limit = paddle.expand(x_limit, [bs, n_max, x_limit.shape[-1]])  # (bs, n_max, dx)

    e_limit = paddle.unsqueeze(limit_dist.E, axis=0)  # (1, de)
    e_limit = paddle.unsqueeze(e_limit, axis=0)  # (1,1,de)
    e_limit = paddle.unsqueeze(e_limit, axis=0)  # (1,1,1,de)
    e_limit = paddle.expand(
        e_limit, [bs, n_max, n_max, e_limit.shape[-1]]
    )  # (bs, n_max, n_max, de)

    y_limit = paddle.unsqueeze(limit_dist.y, axis=0)  # (1, dy)
    if y_limit.shape[-1] == 0:
        y_limit = paddle.zeros([bs, 0], dtype=y_limit.dtype)
    else:
        y_limit = paddle.expand(y_limit, [bs, y_limit.shape[-1]])  # (bs, dy)

    # multinomial for X
    # flatten => (bs*n_max, dx)
    X_probs_flat = paddle.reshape(x_limit, [bs * n_max, -1])
    X_idx = paddle.multinomial(X_probs_flat, num_samples=1)
    X_idx = paddle.reshape(X_idx, [bs, n_max])  # (bs, n_max)

    # multinomial for E
    E_probs_flat = paddle.reshape(e_limit, [bs * n_max * n_max, -1])
    E_idx = paddle.multinomial(E_probs_flat, num_samples=1)
    E_idx = paddle.reshape(E_idx, [bs, n_max, n_max])

    U_y = paddle.zeros([bs, 0], dtype=X_idx.dtype)

    # one_hot
    X_onehot = F.one_hot(X_idx, num_classes=x_limit.shape[-1]).astype("float32")
    E_onehot = F.one_hot(E_idx, num_classes=e_limit.shape[-1]).astype("float32")

    # Get upper triangular part for E
    row_idx, col_idx = np.triu_indices(n=n_max, k=1)
    row_idx_t = paddle.to_tensor(row_idx, dtype="int64")
    col_idx_t = paddle.to_tensor(col_idx, dtype="int64")

    E_upper = paddle.zeros_like(E_onehot)
    for b in range(bs):
        E_upper[b, row_idx_t, col_idx_t] = E_onehot[b, row_idx_t, col_idx_t]

    E_sym = E_upper + paddle.transpose(E_upper, perm=[0, 2, 1, 3])
    # check symmetry
    eq_ = paddle.all(E_sym == paddle.transpose(E_sym, perm=[0, 2, 1, 3]))
    if not eq_.item():
        raise ValueError("Discrete feature noise E is not symmetric!")

    ph = utils.PlaceHolder(X=X_onehot, E=E_sym, y=U_y)
    return ph.mask(node_mask)
