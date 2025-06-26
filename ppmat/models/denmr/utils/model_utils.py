import paddle
import rdkit
from paddle.nn import functional as F
from rdkit import Chem
from tqdm import tqdm
import copy
from typing import List, Dict, Tuple, Union

from ppmat.utils import logger

from .. import diffusion_utils
from . import diffgraphformer_utils as utils


# -------------------------
# Noise & Q
# -------------------------
def apply_noise(model, X, E, y, node_mask):
    """
    Sample noise and apply it to the data.
    """
    t_int = paddle.randint(
        low=1, high=model.T + 1, shape=[X.shape[0], 1], dtype="int64"
    ).astype("float32")
    s_int = t_int - 1

    t_float = t_int / model.T  # nomarlize for stablizing training diffusion model
    s_float = s_int / model.T

    beta_t = model.noise_schedule(t_normalized=t_float)
    alpha_s_bar = model.noise_schedule.get_alpha_bar(t_normalized=s_float)
    alpha_t_bar = model.noise_schedule.get_alpha_bar(t_normalized=t_float)

    Qtb = model.transition_model.get_Qt_bar(alpha_t_bar)
    assert (abs(Qtb.X.sum(axis=2) - 1.0) < 1e-4).all(), Qtb.X.sum(axis=2) - 1
    assert (abs(Qtb.E.sum(axis=2) - 1.0) < 1e-4).all()

    probX = paddle.matmul(X, Qtb.X)  # (bs, n, dx_out)
    probE = paddle.matmul(E, Qtb.E.unsqueeze(1))  # (bs, n, n, de_out)

    sampled_t = diffusion_utils.sample_discrete_features(
        probX=probX, probE=probE, node_mask=node_mask
    )

    X_t = F.one_hot(sampled_t.X, num_classes=model.Xdim_output).astype("int64")
    E_t = F.one_hot(sampled_t.E, num_classes=model.Edim_output).astype("int64")
    assert (X.shape == X_t.shape) and (E.shape == E_t.shape)

    z_t = utils.PlaceHolder(X=X_t, E=E_t, y=y).type_as(X_t).mask(node_mask)

    noisy_data = {
        "t_int": t_int,
        "t": t_float,
        "beta_t": beta_t,
        "alpha_s_bar": alpha_s_bar,
        "alpha_t_bar": alpha_t_bar,
        "X_t": z_t.X,
        "E_t": z_t.E,
        "y_t": z_t.y,
        "node_mask": node_mask,
    }
    return noisy_data


def compute_extra_data(model, noisy_data, isPure=False):
    #  mix extra_features with domain_features and
    # noisy_data into X/E/y final inputs. domain_features
    extra_features = model.extra_features(noisy_data)
    extra_molecular_features = model.domain_features(noisy_data)

    extra_X = concat_without_empty(
        [extra_features.X, extra_molecular_features.X], axis=-1
    )
    extra_E = concat_without_empty(
        [extra_features.E, extra_molecular_features.E], axis=-1
    )
    extra_y = concat_without_empty(
        [extra_features.y, extra_molecular_features.y], axis=-1
    )

    if not isPure:
        t = noisy_data["t"]
        extra_y = concat_without_empty([extra_y, t], axis=1)

    return utils.PlaceHolder(X=extra_X, E=extra_E, y=extra_y)


def concat_without_empty(tensor_lst, axis=-1):
    new_lst = [t.astype("float32") for t in tensor_lst if 0 not in t.shape]
    if new_lst == []:
        return utils.return_empty(tensor_lst[0])
    return paddle.concat(new_lst, axis=axis)


# -------------------------
# KL prior
# -------------------------
def kl_prior(model, X, E, node_mask):
    """
    KL between q(zT|x) and prior p(zT)=Uniform(...)
    """
    bs = X.shape[0]
    ones = paddle.ones([bs, 1], dtype="float32")
    Ts = model.T * ones
    alpha_t_bar = model.noise_schedule.get_alpha_bar(t_int=Ts)  # (bs,1)

    Qtb = model.transition_model.get_Qt_bar(alpha_t_bar)
    probX = paddle.matmul(X, Qtb.X)  # (bs,n,dx_out)
    probE = paddle.matmul(E, Qtb.E.unsqueeze(1))  # (bs,n,n,de_out)

    # limit distribution
    limit_X = model.limit_dist.X.unsqueeze(0).unsqueeze(0)  # shape (1,1,dx_out)
    limit_X = paddle.expand(limit_X, [bs, X.shape[1], model.Xdim_output])

    limit_E = model.limit_dist.E.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    limit_E = paddle.expand(limit_E, [bs, E.shape[1], E.shape[2], model.Edim_output])

    # mask
    limit_dist_X, limit_dist_E, probX, probE = diffusion_utils.mask_distributions(
        true_X=limit_X.clone(),
        true_E=limit_E.clone(),
        pred_X=probX,
        pred_E=probE,
        node_mask=node_mask,
    )

    kl_distance_X = F.kl_div(
        input=paddle.log(probX + 1e-10), label=limit_dist_X, reduction="none"
    )
    kl_distance_E = F.kl_div(
        input=paddle.log(probE + 1e-10), label=limit_dist_E, reduction="none"
    )
    klX_sum = diffusion_utils.sum_except_batch(kl_distance_X)
    klE_sum = diffusion_utils.sum_except_batch(kl_distance_E)
    return klX_sum + klE_sum


def compute_val_loss(
    model, pred, noisy_data, X, E, y, node_mask, condition, test=False
):
    """
    compute NLL (VLB estimation) at validation/test stage
    """
    t = noisy_data["t"]

    # 1. log p(N) = number of nodes prior
    N = paddle.sum(node_mask, axis=1).astype("int64")
    log_pN = model.node_dist.log_prob(N)

    # 2. KL(q(z_T|x), p(z_T)) => uniform prior
    kl_prior_ = kl_prior(model, X, E, node_mask)

    # 3. Stepwise diffusion loss
    loss_all_t = compute_Lt(model, X, E, y, pred, noisy_data, node_mask, test)

    # 4. reconstruction loss
    prob0 = reconstruction_logp(model, t, X, E, node_mask, condition)
    loss_term_0_x = X * paddle.log(prob0.X + 1e-10)  # avoid log(0)
    loss_term_0_e = E * paddle.log(prob0.E + 1e-10)
    # sum val_X_logp and val_E_logp
    loss_term_0 = model.val_X_logp(loss_term_0_x) + model.val_E_logp(loss_term_0_e)

    # combine
    nlls = -log_pN + kl_prior_ + loss_all_t - loss_term_0
    # shape: (bs, ), taking the mean for the batch
    nll = (model.test_nll if test else model.val_nll)(nlls)

    return nll


def compute_Lt(model, X, E, y, pred, noisy_data, node_mask, test):
    pred_probs_X = F.softmax(pred.X, axis=-1)
    pred_probs_E = F.softmax(pred.E, axis=-1)
    pred_probs_y = F.softmax(pred.y, axis=-1)

    Qtb = model.transition_model.get_Qt_bar(noisy_data["alpha_t_bar"])
    Qsb = model.transition_model.get_Qt_bar(noisy_data["alpha_s_bar"])
    Qt = model.transition_model.get_Qt(noisy_data["beta_t"])

    bs, n, _ = X.shape
    # compute true posterior distribution
    prob_true = diffusion_utils.posterior_distributions(
        X=X,
        E=E,
        y=y,
        X_t=noisy_data["X_t"],
        E_t=noisy_data["E_t"],
        y_t=noisy_data["y_t"],
        Qt=Qt,
        Qsb=Qsb,
        Qtb=Qtb,
    )
    prob_true.E = paddle.reshape(prob_true.E, [bs, n, n, -1])

    # compute predicted posterior distribution
    prob_pred = diffusion_utils.posterior_distributions(
        X=pred_probs_X,
        E=pred_probs_E,
        y=pred_probs_y,
        X_t=noisy_data["X_t"],
        E_t=noisy_data["E_t"],
        y_t=noisy_data["y_t"],
        Qt=Qt,
        Qsb=Qsb,
        Qtb=Qtb,
    )
    prob_pred.E = paddle.reshape(prob_pred.E, [bs, n, n, -1])

    # mask
    (
        prob_true_X,
        prob_true_E,
        prob_pred.X,
        prob_pred.E,
    ) = diffusion_utils.mask_distributions(
        true_X=prob_true.X,
        true_E=prob_true.E,
        pred_X=prob_pred.X,
        pred_E=prob_pred.E,
        node_mask=node_mask,
    )

    # KL
    kl_x = (model.test_X_kl if test else model.val_X_kl)(
        prob_true_X, paddle.log(prob_pred.X + 1e-10)
    )
    kl_e = (model.test_E_kl if test else model.val_E_kl)(
        prob_true_E, paddle.log(prob_pred.E + 1e-10)
    )
    return model.T * (kl_x + kl_e)


def reconstruction_logp(model, t, X, E, node_mask, condition):
    """
    L0: - log p(X,E|z0)
    sample randomly from X0, E0, then perform a forward pass
    """
    t_zeros = paddle.zeros_like(t)
    beta_0 = model.noise_schedule(t_zeros)
    Q0 = model.transition_model.get_Qt(beta_t=beta_0)

    probX0 = paddle.matmul(X, Q0.X)
    # E => broadcast
    probE0 = paddle.matmul(E, Q0.E.unsqueeze(1))

    sampled0 = diffusion_utils.sample_discrete_features(
        probX0, probE0, node_mask
    )  # TODO
    X0 = F.one_hot(sampled0.X, num_classes=model.Xdim_output)
    E0 = F.one_hot(sampled0.E, num_classes=model.Edim_output)
    y0 = sampled0.y
    assert (X.shape == X0.shape) and (E.shape == E0.shape)

    sampled_0 = utils.PlaceHolder(X=X0, E=E0, y=y0).mask(
        node_mask
    )  # TODO new add for step4

    # noisy_data
    noisy_data = {
        "X_t": sampled_0.X,
        "E_t": sampled_0.E,
        "y_t": sampled_0.y,
        "node_mask": node_mask,
        "t": paddle.zeros([X0.shape[0], 1]).astype(y0.dtype),
    }

    extra_data = compute_extra_data(model, noisy_data)

    # input_X
    input_X = paddle.concat(
        [noisy_data["X_t"].astype("float32"), extra_data.X], axis=2
    ).astype(dtype="float32")

    # input_E
    input_E = paddle.concat(
        [noisy_data["E_t"].astype("float32"), extra_data.E], axis=3
    ).astype(dtype="float32")

    # partial input_y for decoder
    input_y = paddle.hstack([noisy_data["y_t"].astype("float32"), extra_data.y]).astype(
        dtype="float32"
    )

    ###########################################################
    from ppmat.models.denmr.base_model import MultiModalDecoder

    if model.__class__ == MultiModalDecoder:
        pred0 = model.forward_MultiModalModel(
            input_X, input_E, input_y, node_mask, condition  # TODO : uniform
        )
    else:
        # prepare the extra feature for encoder input without noisy
        z_t = utils.PlaceHolder(X=X0, E=E0, y=y0).type_as(X).mask(node_mask)
        extra_data_pure = compute_extra_data(
            model,
            {"X_t": z_t.X, "E_t": z_t.E, "y_t": z_t.y, "node_mask": node_mask},
            isPure=True,
        )
        # prepare the input data for encoder combining extra features
        input_X_pure = paddle.concat(
            [z_t.X.astype("float32"), extra_data_pure.X], axis=2
        ).astype(dtype="float32")
        input_E_pure = paddle.concat(
            [z_t.E.astype("float32"), extra_data_pure.E], axis=3
        ).astype(dtype="float32")
        input_y_pure = paddle.hstack(
            x=(z_t.y.astype("float32"), extra_data_pure.y)
        ).astype(dtype="float32")
        # obtain the condition vector from output of encoder
        conditionVec = model.encoder(
            input_X_pure, input_E_pure, input_y_pure, node_mask
        )
        # complete input_y for decoder
        input_y = paddle.hstack(x=(input_y, conditionVec)).astype(dtype="float32")

        # forward of decoder with encoder output as condition vector of input of decoder
        pred0 = model.decoder(input_X, input_E, input_y, node_mask)  # TODO: uniform
    ############################################################

    probX0 = F.softmax(pred0.X, axis=-1)
    probE0 = F.softmax(pred0.E, axis=-1)
    proby0 = F.softmax(pred0.y, axis=-1)

    ones_X = paddle.ones([model.Xdim_output], dtype=probX0.dtype)
    ones_E = paddle.ones([model.Edim_output], dtype=probE0.dtype)

    node_mask_3d = node_mask.unsqueeze(-1)
    probX0 = paddle.where(~node_mask_3d, ones_X, probX0)

    edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
    edge_mask_4d = edge_mask.unsqueeze(-1)
    probE0 = paddle.where(~edge_mask_4d, ones_E, probE0)

    diag_mask = paddle.eye(probE0.shape[1], dtype="int64").astype("bool")
    diag_mask = diag_mask.unsqueeze(0).expand([probE0.shape[0], -1, -1])
    diag_mask_4d = diag_mask.unsqueeze(-1)
    probE0 = paddle.where(diag_mask_4d, ones_E, probE0)

    return utils.PlaceHolder(X=probX0, E=probE0, y=proby0)


# -------------------------
# sample => sample_batch
# -------------------------

################################################################################
# Helper function:  reverse‑diffusion sampling for **one** batch               #
################################################################################

@paddle.no_grad()
def sample_batch(
    model,
    batch_id: int,
    batch_size: int,
    batch_condition: List[paddle.Tensor],
    number_chain_steps: int,
    keep_chain: int,
    visual_num: int,
    batch_X: paddle.Tensor,
    batch_E: paddle.Tensor,
    batch_y: paddle.Tensor,
    iter_idx: int,
    num_nodes: Union[int, paddle.Tensor] = None,
    flag_useformula: bool = False,
    return_onehot: bool = False,
) -> Union[
    Tuple[List, List],
    Tuple[List, List, paddle.Tensor, paddle.Tensor],
]:
    """Reverse–diffusion sampling in **Paddle dynamic graph**.

    Parameters
    ----------
    model : DiffusionModelLike
        The generator. Must expose attributes `T`, `node_dist`, `limit_dist`,
        and method `sample_p_zs_given_zt`.
    batch_id : int
        Index of the current batch – used only for logging / visualisation.
    batch_size : int
        Number of graphs to sample in this call.
    batch_condition : list[paddle.Tensor]
        Four‑branch conditioning vector (¹H‑NMR, ¹H peaks, ¹³C‑NMR, ¹³C peaks).
    number_chain_steps : int
        How many intermediate frames to keep for visualisation.
    keep_chain : int
        Number of graph chains to retain (B‑dim truncation).
    visual_num : int
        Number of final samples to render via `visualization_tools`.
    batch_X / batch_E : paddle.Tensor
        One‑hot ground‑truth node / edge feature tensors (used for guidance or
        as *oracle formula* when ``flag_useformula`` is True).
    batch_y : paddle.Tensor
        Additional labels (if any) required by the model.
    iter_idx : int
        Current iteration index for obtain candidates for retrival.
    num_nodes : int | paddle.Tensor | None
        Number of nodes per graph. When *None* the model samples from its own
        learned distribution.
    flag_useformula : bool
        If *True* force the sampled node features to exactly equal the
        provided one‑hot `batch_X` (for strict formula reconstruction).
    return_onehot : bool
        Whether to return the *padded* one‑hot tensors (`X_hot`, `E_hot`) in
        addition to discrete index lists – required by molVec retrieval.

    Returns
    -------
    If ``return_onehot`` is **False** (default):
        (molecule_list, molecule_list_true)
    If ``return_onehot`` is **True**:
        (molecule_list, molecule_list_true, X_hot, E_hot)

    Where
    ``molecule_list[i] == [atom_index_vector, bond_matrix]`` and
    ``molecule_list_true`` follows the same structure for ground‑truth.
    """
    
    # 1. Determine node counts and create a boolean mask for padded positions
    if num_nodes is None:
        # Sample number of nodes from the model's learned distribution
        n_nodes = model.node_dist.sample_n(batch_size)
    elif isinstance(num_nodes, int):
        n_nodes = paddle.full([batch_size], num_nodes, dtype="int64")
    else:
        n_nodes = paddle.to_tensor(num_nodes)  # assume Tensor

    n_max: int = int(paddle.max(n_nodes).item()) # ***largest graph size***

    # `node_mask[b, i] == True` if node *i* is real for graph *b*
    arange = paddle.arange(n_max).unsqueeze(0).expand([batch_size, n_max])
    node_mask = arange < n_nodes.unsqueeze(1)

    # 2. Initialise z_T with (categorical) noise and prepare trajectory buffers
    # z(n_samples, n_nodes, n_features)
    z_T = diffusion_utils.sample_discrete_feature_noise(
        limit_dist=model.limit_dist, node_mask=node_mask
    )
    X_t, E_t, y_t = z_T.X, z_T.E, z_T.y

    chain_X = paddle.zeros([number_chain_steps, keep_chain, n_max], dtype="int64")
    chain_E = paddle.zeros(
        [number_chain_steps, keep_chain, n_max, n_max], dtype="int64"
    )

    # 3. Main reverse‑diffusion loop: t = T → 1 (s = t‑1)
    for s_int in tqdm(
        range(model.T - 1, -1, -1),
        desc=f"Batch {batch_id} RepeatIter {iter_idx} sampling {model.T}→0",
        unit="step",
    ):
        s_arr = paddle.full([batch_size, 1], float(s_int))
        t_arr = s_arr + 1.0
        s_norm, t_norm = s_arr / model.T, t_arr / model.T

        # One reverse‑diffusion step
        sampled_s, discrete_sampled_s = sample_p_zs_given_zt(
            model,
            s=s_norm,
            t=t_norm,
            X_t=X_t,
            E_t=E_t,
            y_t=y_t,
            node_mask=node_mask,
            conditionVec=batch_condition,
            batch_X=batch_X,
            batch_E=batch_E,
            batch_y=batch_y,
        )
        X_t, E_t, y_t = sampled_s.X, sampled_s.E, sampled_s.y
        if flag_useformula == True:
            # Force atom types to match the provided formula (oracle guidance)
            X_t = batch_X

        # save intermediate frames for the first `keep_chain` graphs
        write_index = (s_int * number_chain_steps) // model.T
        chain_X[write_index] = discrete_sampled_s.X[:keep_chain]
        chain_E[write_index] = discrete_sampled_s.E[:keep_chain]

    # 4. Collapse padding → obtain discrete indices; optionally keep one‑hot 
    # Make a *clone* of `sampled_s` so that collapsing will not overwrite the
    # one‑hot information we still need for molVec retrieval.
    sampled_copy = copy.deepcopy(sampled_s)
    
    # 4‑a. Get discrete indices from the cloned tensor (padding removed)
    sampled_collapse = sampled_copy.mask(node_mask, collapse=True)
    X_idx, E_idx = sampled_collapse.X, sampled_collapse.E                # [B, …]
    if flag_useformula:
        # Ensure indices follow the oracle molecular formula when required
        X_idx = paddle.argmax(batch_X, axis=-1)

    # 4‑b. Optionally obtain **un‑collapsed** one‑hot tensors for retrieval.
    if return_onehot:
        # Call mask *without* collapse on the ORIGINAL `sampled_s`, which still
        # contains one‑hot embeddings; shape stays [B, n_max, feat]
        X_hot = sampled_s.mask(node_mask).X.numpy()
        E_hot = sampled_s.mask(node_mask).E.numpy()
        if flag_useformula:
            # When formula guidance is enabled, the node one‑hot should exactly
            # match the provided ground‑truth.
            X_hot = batch_X.numpy()
        X_hot = [X_hot[i] for i in range(X_hot.shape[0])]
        E_hot = [E_hot[i] for i in range(E_hot.shape[0])]
    else:
        X_hot = E_hot = None

    # 5. Assemble Python lists for downstream RDKit / metrics 
    mol_list, mol_true = [], []
    n_nodes_np = n_nodes.numpy()
    batch_X_idx = paddle.argmax(batch_X, axis=-1).numpy()
    batch_E_idx = paddle.argmax(batch_E, axis=-1).numpy()
    for i in range(batch_size):
        n = n_nodes_np[i]
        mol_list.append([
            X_idx[i, :n].numpy(),
            E_idx[i, :n, :n].numpy(),
        ])
        mol_true.append([
            batch_X_idx[i, :n],
            batch_E_idx[i, :n, :n],
        ])

    # 6. Optional visualisation via model.visualization_tools
    if model.visualization_tools is not None:
        # 6.a Prepare the chain for visualization and saving
        if keep_chain > 0:
            # pick the last frame of the chain add the top index of chain_X/E(index 0)
            final_X_chain = X_idx[:keep_chain]
            final_E_chain = E_idx[:keep_chain]
            chain_X[0] = final_X_chain  # Overwrite last frame with the resulting X, E
            chain_E[0] = final_E_chain

            # revers time sequence for visualization
            chain_X = diffusion_utils.reverse_tensor(chain_X)
            chain_E = diffusion_utils.reverse_tensor(chain_E)

            # Repeat last frame to see final sample better
            chain_X = paddle.concat([chain_X, chain_X[-1:].tile([10, 1, 1])], axis=0)
            chain_E = paddle.concat([chain_E, chain_E[-1:].tile([10, 1, 1, 1])], axis=0)
            assert chain_X.shape[0] == (number_chain_steps + 10)

        # 6.b use visulize tools
        num_mols = chain_X.shape[1]
        # draw animation of diffusion process of generated molecules
        for i in range(num_mols):
            chain_X_np = chain_X[:, i, :].numpy()
            chain_E_np = chain_E[:, i, :, :].numpy()
            model.visualization_tools.visualize_chain(
                batch_id, i, chain_X_np, chain_E_np
            )
        # draw picture of predicted and true molecules
        model.visualization_tools.visualizeNmr(
            batch_id,
            mol_list,
            mol_true,
            visual_num,
        )
    
    if return_onehot:
        return mol_list, mol_true, X_hot, E_hot
    return mol_list, mol_true


@paddle.no_grad()
def sample_p_zs_given_zt(
    model, s, t, X_t, E_t, y_t, node_mask, conditionVec, batch_X, batch_E, batch_y
):
    """
    sample from p(z_s | z_t) : take one step of reverse diffusion
    """
    beta_t = model.noise_schedule(t_normalized=t)
    alpha_s_bar = model.noise_schedule.get_alpha_bar(t_normalized=s)
    alpha_t_bar = model.noise_schedule.get_alpha_bar(t_normalized=t)

    # retrieve transitions matrix
    Qtb = model.transition_model.get_Qt_bar(alpha_t_bar)
    Qsb = model.transition_model.get_Qt_bar(alpha_s_bar)
    Qt = model.transition_model.get_Qt(beta_t)

    # prepare neural net input
    noisy_data = {
        "X_t": X_t,
        "E_t": E_t,
        "y_t": y_t,
        "t": t,
        "node_mask": node_mask,
    }
    extra_data = compute_extra_data(model, noisy_data)

    # input_X for decoder
    input_X = paddle.concat(
        [noisy_data["X_t"].astype("float32"), extra_data.X.astype(dtype="float32")],
        axis=2,
    )

    # input_E for decoder
    input_E = paddle.concat(
        [noisy_data["E_t"].astype("float32"), extra_data.E.astype(dtype="float32")],
        axis=3,
    )

    # partial input_y for decoder
    input_y = paddle.hstack(
        [noisy_data["y_t"].astype("float32"), extra_data.y.astype(dtype="float32")]
    )

    from ppmat.models.denmr.base_model import MultiModalDecoder

    if model.__class__ == MultiModalDecoder:
        pred = model.forward_MultiModalModel(
            input_X, input_E, input_y, node_mask, conditionVec
        )
    else:
        # prepare the extra feature for encoder input without noisy
        batch_values = (
            utils.PlaceHolder(X=batch_X, E=batch_E, y=batch_y)
            .type_as(batch_X)
            .mask(node_mask)
        )
        extra_data_pure = compute_extra_data(
            model,
            {
                "X_t": batch_values.X,
                "E_t": batch_values.E,
                "y_t": batch_values.y,
                "node_mask": node_mask,
            },
            isPure=True,
        )
        # prepare the input data for encoder combining extra features
        input_X_pure = paddle.concat(
            [batch_values.X.astype("float32"), extra_data_pure.X], axis=2
        ).astype(dtype="float32")
        input_E_pure = paddle.concat(
            [batch_values.E.astype("float32"), extra_data_pure.E], axis=3
        ).astype(dtype="float32")
        input_y_pure = paddle.hstack(
            x=(batch_values.y.astype("float32"), extra_data_pure.y)
        ).astype(dtype="float32")
        # obtain the condition vector from output of encoder
        conditionVec = model.encoder(
            input_X_pure, input_E_pure, input_y_pure, node_mask
        )
        # complete input_y for decoder
        input_y = paddle.hstack(x=(input_y, conditionVec)).astype(dtype="float32")

        # forward of decoder with encoder output as condition vector of input of decoder
        pred = model.decoder(input_X, input_E, input_y, node_mask)

    pred_X = F.softmax(pred.X, axis=-1)
    pred_E = F.softmax(pred.E, axis=-1)

    # compute posterior distribution
    p_s_and_t_given_0_X = diffusion_utils.compute_batched_over0_posterior_distribution(
        X_t=X_t, Qt=Qt.X, Qsb=Qsb.X, Qtb=Qtb.X
    )
    p_s_and_t_given_0_E = diffusion_utils.compute_batched_over0_posterior_distribution(
        X_t=E_t, Qt=Qt.E, Qsb=Qsb.E, Qtb=Qtb.E
    )

    # compute node probability
    weighted_X = pred_X.unsqueeze(-1) * p_s_and_t_given_0_X
    unnormalized_prob_X = paddle.sum(weighted_X, axis=2)
    unnormalized_prob_X = paddle.where(
        paddle.sum(unnormalized_prob_X, axis=-1, keepdim=True) == 0,
        paddle.to_tensor(1e-5, dtype=unnormalized_prob_X.dtype),
        unnormalized_prob_X,
    )
    prob_X = unnormalized_prob_X / paddle.sum(
        unnormalized_prob_X, axis=-1, keepdim=True
    )

    # compute edge probability
    pred_E = pred_E.reshape([X_t.shape[0], -1, pred.E.shape[-1]])
    weighted_E = pred_E.unsqueeze(-1) * p_s_and_t_given_0_E
    unnormalized_prob_E = paddle.sum(weighted_E, axis=-2)
    unnormalized_prob_E = paddle.where(
        paddle.sum(unnormalized_prob_E, axis=-1, keepdim=True) == 0,
        paddle.to_tensor(1e-5, dtype=unnormalized_prob_E.dtype),
        unnormalized_prob_E,
    )
    prob_E = unnormalized_prob_E / paddle.sum(
        unnormalized_prob_E, axis=-1, keepdim=True
    )
    prob_E = prob_E.reshape([X_t.shape[0], X_t.shape[1], X_t.shape[1], -1])

    assert ((prob_X.sum(axis=-1) - 1).abs().max() < 1e-4).all()
    assert ((prob_E.sum(axis=-1) - 1).abs() < 1e-4).all()

    # sample from p(z_s | z_t)
    sampled_s = diffusion_utils.sample_discrete_features(prob_X, prob_E, node_mask)
    X_s = F.one_hot(sampled_s.X, num_classes=model.Xdim_output)
    E_s = F.one_hot(sampled_s.E, num_classes=model.Edim_output)

    assert (E_s == paddle.transpose(E_s, [0, 1, 2])).all()
    assert (X_t.shape == X_s.shape) and (E_t.shape == E_s.shape)

    out_one_hot = utils.PlaceHolder(X=X_s, E=E_s, y=paddle.zeros([y_t.shape[0], 0]))
    out_discrete = utils.PlaceHolder(X=X_s, E=E_s, y=paddle.zeros([y_t.shape[0], 0]))

    return out_one_hot.mask(node_mask), out_discrete.mask(node_mask, collapse=True)


# -----------------------
# molecule visualization/comparision
# -----------------------
def mol_from_graphs(atom_decoder, node_list, adjacency_matrix):
    """
    Convert discrete graph (atom indices, adjacency) to rdkit Mol
    """
    mol = Chem.RWMol()

    node_to_idx = {}
    for i, nd in enumerate(node_list):
        if nd == -1:
            continue
        a = Chem.Atom(atom_decoder[int(nd)])
        molIdx = mol.AddAtom(a)
        node_to_idx[i] = molIdx

    for ix, row in enumerate(adjacency_matrix):
        for iy, bond in enumerate(row):
            if iy <= ix:
                continue
            if bond == 1:
                bond_type = Chem.rdchem.BondType.SINGLE
            elif bond == 2:
                bond_type = Chem.rdchem.BondType.DOUBLE
            elif bond == 3:
                bond_type = Chem.rdchem.BondType.TRIPLE
            elif bond == 4:
                bond_type = Chem.rdchem.BondType.AROMATIC
            else:
                continue
            mol.AddBond(node_to_idx[ix], node_to_idx[iy], bond_type)

    try:
        mol = mol.GetMol()
    except rdkit.Chem.KekulizeException:
        print("Can't kekulize molecule")
        mol = None
    return mol
