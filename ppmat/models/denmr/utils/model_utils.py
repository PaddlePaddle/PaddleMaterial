import paddle
import rdkit
from paddle.nn import functional as F
from rdkit import Chem
from tqdm import tqdm

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

    t_float = t_int / model.T
    s_float = s_int / model.T

    beta_t = model.noise_schedule(t_normalized=t_float)
    alpha_s_bar = model.noise_schedule.get_alpha_bar(t_normalized=s_float)
    alpha_t_bar = model.noise_schedule.get_alpha_bar(t_normalized=t_float)

    Qtb = model.transition_model.get_Qt_bar(alpha_t_bar)

    probX = paddle.matmul(X, Qtb.X)  # (bs, n, dx_out)
    probE = paddle.matmul(E, Qtb.E.unsqueeze(1))  # (bs, n, n, de_out)

    sampled_t = diffusion_utils.sample_discrete_features(
        probX=probX, probE=probE, node_mask=node_mask
    )

    X_t = F.one_hot(sampled_t.X, num_classes=model.Xdim_output).astype("int64")
    E_t = F.one_hot(sampled_t.E, num_classes=model.Edim_output).astype("int64")

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
    new_lst = [t.astype("float") for t in tensor_lst if 0 not in t.shape]
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

    sampled0 = diffusion_utils.sample_discrete_features(probX0, probE0, node_mask)
    X0 = F.one_hot(sampled0.X, num_classes=model.Xdim_output)
    E0 = F.one_hot(sampled0.E, num_classes=model.Edim_output)
    y0 = sampled0.y
    assert (X.shape == X0.shape) and (E.shape == E0.shape)

    # noisy_data
    noisy_data = {
        "X_t": X0,
        "E_t": E0,
        "y_t": y0,
        "node_mask": node_mask,
        "t": paddle.zeros([X0.shape[0], 1]).astype("float32"),
    }
    extra_data = compute_extra_data(model, noisy_data)

    # input_X
    input_X = paddle.concat(
        [noisy_data["X_t"].astype("float"), extra_data.X], axis=2
    ).astype(dtype="float32")

    # input_E
    input_E = paddle.concat(
        [noisy_data["E_t"].astype("float"), extra_data.E], axis=3
    ).astype(dtype="float32")

    # partial input_y for decoder
    input_y = paddle.hstack([noisy_data["y_t"].astype("float"), extra_data.y]).astype(
        dtype="float32"
    )

    ###########################################################
    from ppmat.models.denmr.base_model import MultiModalDecoder

    if model.__class__ == MultiModalDecoder:
        if model.add_condition:
            batch_length = input_X.shape[0]
            conditionVec = condition
            y_condition = conditionVec.reshape(batch_length, model.vocabDim)
        else:
            y_condition = paddle.zeros(shape=[X.shape[0], 1024]).cuda(blocking=True)
        pred0 = model.forward_MultiModalModel(
            input_X, input_E, input_y, node_mask, y_condition  # TODO : uniform
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
            [z_t.X.astype("float"), extra_data_pure.X], axis=2
        ).astype(dtype="float32")
        input_E_pure = paddle.concat(
            [z_t.E.astype("float"), extra_data_pure.E], axis=3
        ).astype(dtype="float32")
        input_y_pure = paddle.hstack(
            x=(z_t.y.astype("float"), extra_data_pure.y)
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


@paddle.no_grad()
def sample_batch(
    model,
    batch_id: int,
    batch_size: int,
    batch_condition,
    number_chain_steps: int,
    keep_chain: int,
    visual_num: int,
    batch_X,
    batch_E,
    batch_y,
    num_nodes=None,
):
    """
    sample: reverse diffusion
    """
    if num_nodes is None:
        n_nodes = model.node_dist.sample_n(batch_size)
    elif isinstance(num_nodes, int):
        n_nodes = paddle.full([batch_size], num_nodes, dtype="int64")
    else:
        n_nodes = paddle.to_tensor(num_nodes)  # assume Tensor
    n_max = int(paddle.max(n_nodes).item())

    # build masks
    arange = paddle.arange(n_max).unsqueeze(0).expand([batch_size, n_max])
    node_mask = arange < n_nodes.unsqueeze(1)

    # sample noise -- z hase saize (n_samples, n_nodes, n_features)
    z_T = diffusion_utils.sample_discrete_feature_noise(
        limit_dist=model.limit_dist, node_mask=node_mask
    )
    X, E, y = z_T.X, z_T.E, z_T.y

    assert number_chain_steps < model.T
    chain_X = paddle.zeros([number_chain_steps, keep_chain, X.shape[1]], dtype="int64")
    chain_E = paddle.zeros(
        [number_chain_steps, keep_chain, E.shape[1], E.shape[2]], dtype="int64"
    )

    # Iterate sample p(z_s | z_t) over diffusion steps, t = 1, ..., T, with s = t - 1
    for s_int in tqdm(
        reversed(range(model.T)), desc="Sampling", unit=" diffuision step"
    ):
        s_array = paddle.full([batch_size, 1], float(s_int))
        t_array = s_array + 1.0
        s_norm = s_array / model.T
        t_norm = t_array / model.T

        # Sample z_s
        sampled_s, discrete_sampled_s = sample_p_zs_given_zt(
            model,
            s=s_norm,
            t=t_norm,
            X_t=X,
            E_t=E,
            y_t=y,
            node_mask=node_mask,
            conditionVec=batch_condition,
            batch_X=batch_X,
            batch_E=batch_E,
            batch_y=batch_y,
        )

        # save the first keep_chain graphs
        write_index = (s_int * number_chain_steps) // model.T
        chain_X[write_index] = discrete_sampled_s.X[:keep_chain]
        chain_E[write_index] = discrete_sampled_s.E[:keep_chain]

    # sample results of last diffusion step
    sampled_s = sampled_s.mask(node_mask, collapse=True)
    X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

    # log sample result info
    msg = f"Sampling: {len(batch_X)} molecules with {model.T} diffusion steps generated"
    logger.message(msg)

    # Prepare the chain for visualization and saving
    if keep_chain > 0:
        final_X_chain = X[:keep_chain]
        final_E_chain = E[:keep_chain]

        chain_X[0] = final_X_chain  # Overwrite last frame with the resulting X, E
        chain_E[0] = final_E_chain

        chain_X = diffusion_utils.reverse_tensor(chain_X)
        chain_E = diffusion_utils.reverse_tensor(chain_E)

        # Repeat last frame to see final sample better
        chain_X = paddle.concat([chain_X, chain_X[-1:].tile([10, 1, 1])], axis=0)
        chain_E = paddle.concat([chain_E, chain_E[-1:].tile([10, 1, 1, 1])], axis=0)
        assert chain_X.shape[0] == (number_chain_steps + 10)

    batch_X = paddle.argmax(batch_X, axis=-1)
    batch_E = paddle.argmax(batch_E, axis=-1)

    # Prepare the samples for visualization and saving
    molecule_list = []
    molecule_list_True = []
    n_nodes_np = n_nodes.numpy()
    for i in range(batch_size):
        n = n_nodes_np[i]
        atom_types = X[i, :n].numpy()
        edge_types = E[i, :n, :n].numpy()

        atom_types_true = batch_X[i, :n].numpy()
        edge_types_true = batch_E[i, :n, :n].numpy()

        molecule_list.append([atom_types, edge_types])
        molecule_list_True.append([atom_types_true, edge_types_true])

    # visualization
    if model.visualization_tools is not None:
        num_molecules = chain_X.shape[1]
        for i in range(num_molecules):
            chain_X_np = chain_X[:, i, :].numpy()
            chain_E_np = chain_E[:, i, :, :].numpy()

            model.visualization_tools.visualize_chain(
                batch_id, i, chain_X_np, chain_E_np
            )
        msg = "Sampling:"
        msg += f" {num_molecules} of {batch_size} generated molecules"
        msg += " visualization complete"
        logger.message(msg)

        model.visualization_tools.visualizeNmr(
            batch_id,
            molecule_list,
            molecule_list_True,
            visual_num,
        )
        msg = "Sampling:"
        msg += f" {visual_num} of {batch_size} generated and true molecules"
        msg += " visualization complete"
        logger.message(msg)

    return molecule_list, molecule_list_True


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

    Qtb = model.transition_model.get_Qt_bar(alpha_t_bar)
    Qsb = model.transition_model.get_Qt_bar(alpha_s_bar)
    Qt = model.transition_model.get_Qt(beta_t)

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
        [noisy_data["X_t"].astype("float"), extra_data.X], axis=2
    ).astype(dtype="float32")

    # input_E for decoder
    input_E = paddle.concat(
        [noisy_data["E_t"].astype("float"), extra_data.E], axis=3
    ).astype(dtype="float32")

    # partial input_y for decoder
    input_y = paddle.hstack([noisy_data["y_t"].astype("float"), extra_data.y]).astype(
        dtype="float32"
    )

    from ppmat.models.denmr.base_model import MultiModalDecoder

    if model.__class__ == MultiModalDecoder:
        pred = model.forward_MultiModalModel(
            input_X, input_E, input_y, node_mask, conditionVec
        )
    else:
        # prepare the extra feature for encoder input without noisy
        z_t = (
            utils.PlaceHolder(X=batch_X, E=batch_E, y=batch_y)
            .type_as(batch_X)
            .mask(node_mask)
        )
        extra_data_pure = compute_extra_data(
            model,
            {"X_t": z_t.X, "E_t": z_t.E, "y_t": z_t.y, "node_mask": node_mask},
            isPure=True,
        )
        # prepare the input data for encoder combining extra features
        input_X_pure = paddle.concat(
            [z_t.X.astype("float"), extra_data_pure.X], axis=2
        ).astype(dtype="float32")
        input_E_pure = paddle.concat(
            [z_t.E.astype("float"), extra_data_pure.E], axis=3
        ).astype(dtype="float32")
        input_y_pure = paddle.hstack(
            x=(z_t.y.astype("float"), extra_data_pure.y)
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
    pred_E = F.softmax(pred.E, axis=-1).reshape([X_t.shape[0], -1, pred.E.shape[-1]])

    p_s_and_t_given_0_X = diffusion_utils.compute_batched_over0_posterior_distribution(
        X_t=X_t, Qt=Qt.X, Qsb=Qsb.X, Qtb=Qtb.X
    )
    p_s_and_t_given_0_E = diffusion_utils.compute_batched_over0_posterior_distribution(
        X_t=E_t, Qt=Qt.E, Qsb=Qsb.E, Qtb=Qtb.E
    )

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

    # 采样
    sampled_s = diffusion_utils.sample_discrete_features(prob_X, prob_E, node_mask)
    X_s = F.one_hot(sampled_s.X, num_classes=model.Xdim_output)
    E_s = F.one_hot(sampled_s.E, num_classes=model.Edim_output)

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
