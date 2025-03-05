import copy
import os
import random

import paddle
import paddle.nn as nn
import rdkit
from einops import rearrange
from einops import repeat
from paddle.nn import functional as F
from rdkit import Chem
from tqdm import tqdm

from ppmat.metrics.abstract_metrics import NLL
from ppmat.metrics.abstract_metrics import SumExceptBatchKL
from ppmat.metrics.abstract_metrics import SumExceptBatchMetric
from ppmat.metrics.train_metrics import TrainLossDiscrete
from ppmat.models.denmr.diffusion_prior import DiffusionPriorNetwork
from ppmat.models.denmr.diffusion_prior import EmptyLayer
from ppmat.models.denmr.diffusion_prior import NoiseScheduler
from ppmat.models.denmr.diffusion_prior import freeze_model_and_make_eval_
from ppmat.models.denmr.encoder import Encoder
from ppmat.utils import save_load
from ppmat.utils import logger

from . import diffusion_utils
from .graph_transformer import GraphTransformer
from .graph_transformer import GraphTransformer_C
from .noise_schedule import DiscreteUniformTransition
from .noise_schedule import MarginalUniformTransition
from .noise_schedule import PredefinedNoiseScheduleDiscrete
from .utils import digressutils as utils
from .utils import model_utils as m_utils
from .utils.diffusionprior_utils import default
from .utils.diffusionprior_utils import exists
from .utils.diffusionprior_utils import l2norm


class MolecularGraphTransformer(nn.Layer):
    def __init__(
        self,
        config,
        dataset_infos,
        train_metrics,
        sampling_metrics,
        visualization_tools,
        extra_features,
        domain_features,
    ) -> None:
        super().__init__()

        #############################################################
        # configure general variables settings
        #############################################################
        self.name = config["__name__"]
        self.model_dtype = paddle.get_default_dtype()
        self.T = config["diffusion_model"]["diffusion_steps"]
        self.visualization_tools = visualization_tools

        #############################################################
        # configure datasets inter-varibles
        #############################################################
        input_dims = dataset_infos.input_dims
        output_dims = dataset_infos.output_dims
        self.dataset_info = dataset_infos
        self.extra_features = extra_features
        self.domain_features = domain_features

        #############################################################
        # configure noise scheduler
        #############################################################
        self.noise_schedule = PredefinedNoiseScheduleDiscrete(
            config["diffusion_model"]["diffusion_noise_schedule"],
            timesteps=self.T,
        )

        #############################################################
        # configure model
        #############################################################
        self.con_input_dim = copy.deepcopy(input_dims)
        self.con_input_dim["X"] -= 8
        self.con_input_dim["y"] = 1024
        self.con_output_dim = dataset_infos.output_dims

        self.encoder = GraphTransformer_C(
            n_layers=config["encoder"]["num_layers"],
            input_dims=self.con_input_dim,
            hidden_mlp_dims=config["encoder"]["hidden_mlp_dims"],
            hidden_dims=config["encoder"]["hidden_dims"],
            output_dims=self.con_output_dim,
            act_fn_in=nn.ReLU(),
            act_fn_out=nn.ReLU(),
        )

        self.decoder = GraphTransformer(
            n_layers=config["decoder"]["num_layers"],
            input_dims=input_dims,
            hidden_mlp_dims=config["decoder"]["hidden_mlp_dims"],
            hidden_dims=config["decoder"]["hidden_dims"],
            output_dims=output_dims,
            act_fn_in=nn.ReLU(),
            act_fn_out=nn.ReLU(),
        )

        #############################################################
        # configure loss calculation with initialization of transition model
        #############################################################
        self.Xdim = input_dims["X"]
        self.Edim = input_dims["E"]
        self.ydim = input_dims["y"]
        self.Xdim_output = output_dims["X"]
        self.Edim_output = output_dims["E"]
        self.ydim_output = output_dims["y"]
        self.node_dist = dataset_infos.nodes_dist

        # Transition Model
        if config["diffusion_model"]["transition"] == "uniform":
            self.transition_model = DiscreteUniformTransition(
                x_classes=self.Xdim_output,
                e_classes=self.Edim_output,
                y_classes=self.ydim_output,
            )
            x_limit = paddle.ones([self.Xdim_output]) / self.Xdim_output
            e_limit = paddle.ones([self.Edim_output]) / self.Edim_output
            y_limit = paddle.ones([self.ydim_output]) / self.ydim_output
            self.limit_dist = utils.PlaceHolder(X=x_limit, E=e_limit, y=y_limit)

        elif config["diffusion_model"]["transition"] == "marginal":
            node_types = self.dataset_info.node_types.astype(self.model_dtype)
            x_marginals = node_types / paddle.sum(node_types)

            edge_types = self.dataset_info.edge_types.astype(self.model_dtype)
            e_marginals = edge_types / paddle.sum(edge_types)
            logger.info(f"Marginal distribution of classes: {x_marginals.tolist()} for nodes, ")
            logger.info(f"{e_marginals.tolist()} for edges")

            self.transition_model = MarginalUniformTransition(
                x_marginals=x_marginals,
                e_marginals=e_marginals,
                y_classes=self.ydim_output,
            )
            self.limit_dist = utils.PlaceHolder(
                X=x_marginals,
                E=e_marginals,
                y=paddle.ones([self.ydim_output]) / self.ydim_output,
            )

        # configure loss
        self.train_loss = TrainLossDiscrete(config["diffusion_model"]["lambda_train"])

        #############################################################
        # configure training setting and other properties
        #############################################################
        self.best_val_nll = 1e8
        self.val_counter = 0
        self.vocabDim = config["vocab_dim"]
        self.number_chain_steps = config["diffusion_model"]["number_chain_steps"]

        #############################################################
        # configure generated datas for visualization after forward
        #############################################################
        self.val_nll = NLL()
        self.val_X_kl = SumExceptBatchKL()
        self.val_E_kl = SumExceptBatchKL()
        self.val_X_logp = SumExceptBatchMetric()
        self.val_E_logp = SumExceptBatchMetric()
        self.val_y_collection = []
        self.val_atomCount = []
        self.val_data_X = []
        self.val_data_E = []

        self.test_nll = NLL()
        self.test_X_kl = SumExceptBatchKL()
        self.test_E_kl = SumExceptBatchKL()
        self.test_X_logp = SumExceptBatchMetric()
        self.test_E_logp = SumExceptBatchMetric()
        self.test_y_collection = []
        self.test_atomCount = []
        self.test_data_X = []
        self.test_data_E = []

        self.train_metrics = train_metrics
        self.sampling_metrics = sampling_metrics

        #############################################################
        # configure key data container
        #############################################################
        self.val_y_collection = []
        self.val_atomCount = []
        self.val_data_X = []
        self.val_data_E = []
        self.test_y_collection = []
        self.test_atomCount = []
        self.test_data_X = []
        self.test_data_E = []

    def forward(self, batch, is_val = False):
        batch_graph, other_data = batch

        # transfer to dense graph from sparse graph
        if batch_graph.edges.T.numel() == 0:
            print("Found a batch with no edges. Skipping.")
            return None
        dense_data, node_mask = utils.to_dense(
            batch_graph.node_feat["feat"],
            batch_graph.edges.T,
            batch_graph.edge_feat["feat"],
            batch_graph.graph_node_id,
        )
        dense_data = dense_data.mask(node_mask)
        X, E = dense_data.X, dense_data.E

        # add noise to the inputs (X, E)
        noisy_data = self.apply_noise(
            dense_data.X, dense_data.E, other_data["y"], node_mask
        )
        extra_data = self.compute_extra_data(noisy_data)

        # input_X
        input_X = paddle.concat(
            [noisy_data["X_t"].astype("float"), extra_data.X], axis=2
        ).astype(dtype="float32")

        # input_E
        input_E = paddle.concat(
            [noisy_data["E_t"].astype("float"), extra_data.E], axis=3
        ).astype(dtype="float32")

        # input_y with encoder output as condition vector of input of decoder
        input_y = paddle.hstack(
            [noisy_data["y_t"].astype("float"), extra_data.y]
        ).astype(dtype="float32")

        y_condition = paddle.zeros(shape=[input_X.shape[0], 1024]).cuda(blocking=True)

        conditionVec = self.encoder(X, E, y_condition, node_mask)
        input_y = paddle.hstack(x=(input_y, conditionVec)).astype(dtype="float32")

        # forward of decoder with encoder output as condition vector of input of decoder
        pred = self.decoder(input_X, input_E, input_y, node_mask)

        # compute loss
        loss = self.train_loss(
            masked_pred_X=pred.X,
            masked_pred_E=pred.E,
            pred_y=pred.y,
            true_X=X,
            true_E=E,
            true_y=other_data["y"],
        )

        self.train_metrics(
            masked_pred_X=pred.X,
            masked_pred_E=pred.E,
            true_X=X,
            true_E=E,
            log=False,
        )
        return loss

    def apply_noise(self, X, E, y, node_mask):
        """
        Sample noise and apply it to the data.
        """
        t_int = paddle.randint(
            low=1, high=self.T + 1, shape=[X.shape[0], 1], dtype="int64"
        ).astype("float32")
        s_int = t_int - 1

        t_float = t_int / self.T
        s_float = s_int / self.T

        beta_t = self.noise_schedule(t_normalized=t_float)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s_float)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_float)

        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar)

        # probX = X @ Qtb.X => paddle.matmul(X, Qtb.X)
        probX = paddle.matmul(X, Qtb.X)  # (bs, n, dx_out)
        probE = paddle.matmul(E, Qtb.E.unsqueeze(1))  # (bs, n, n, de_out)

        sampled_t = diffusion_utils.sample_discrete_features(
            probX=probX, probE=probE, node_mask=node_mask
        )

        X_t = F.one_hot(sampled_t.X, num_classes=self.Xdim_output).astype("int64")
        E_t = F.one_hot(sampled_t.E, num_classes=self.Edim_output).astype("int64")

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

    def compute_extra_data(self, noisy_data):
        #  mix extra_features with domain_features and
        # noisy_data into X/E/y final inputs. domain_features
        extra_features = self.extra_features(noisy_data)
        extra_molecular_features = self.domain_features(noisy_data)

        extra_X = self.concat_without_empty(
            [extra_features.X, extra_molecular_features.X], axis=-1
        )
        extra_E = self.concat_without_empty(
            [extra_features.E, extra_molecular_features.E], axis=-1
        )
        extra_y = self.concat_without_empty(
            [extra_features.y, extra_molecular_features.y], axis=-1
        )

        t = noisy_data["t"]
        extra_y = self.concat_without_empty([extra_y, t], axis=1)

        return utils.PlaceHolder(X=extra_X, E=extra_E, y=extra_y)

    def concat_without_empty(self, tensor_lst, axis=-1):
        new_lst = [t.astype("float") for t in tensor_lst if 0 not in t.shape]
        if new_lst == []:
            return utils.return_empty(tensor_lst[0])
        return paddle.concat(new_lst, axis=axis)

    @paddle.no_grad()
    def sample(self, batch, i):
        batch_graph, other_data = batch
        # transfer to dense graph from sparse graph
        dense_data, node_mask = utils.to_dense(
            batch_graph.node_feat["feat"],
            batch_graph.edges.T,
            batch_graph.edge_feat["feat"],
            batch_graph.graph_node_id,
        )
        dense_data = dense_data.mask(node_mask)
        X, E = dense_data.X, dense_data.E

        # add noise to the inputs (X, E)
        noisy_data = self.apply_noise(
            dense_data.X, dense_data.E, other_data["y"], node_mask
        )
        extra_data = self.compute_extra_data(noisy_data)

        # input_X
        input_X = paddle.concat(
            [noisy_data["X_t"].astype("float"), extra_data.X], axis=2
        ).astype("float32")

        # input_E
        input_E = paddle.concat(
            [noisy_data["E_t"].astype("float"), extra_data.E], axis=3
        ).astype("float32")

        # input_y
        input_y = paddle.concat(
            [noisy_data["y_t"].astype("float"), extra_data.y], axis=1
        ).astype("float32")
        
        y_condition = paddle.zeros(shape=[X.shape[0], 1024]).cuda(blocking=True)
        
        conditionVec = self.encoder(X, E, y_condition, node_mask)
        input_y = paddle.hstack(x=(input_y, conditionVec)).astype(dtype="float32")

        # forward of decoder with encoder output as condition vector of input of decoder
        pred = self.decoder(input_X, input_E, input_y, node_mask)

        # evaluate the loss especially in the inference stage
        loss = self.train_loss(
            masked_pred_X=pred.X,
            masked_pred_E=pred.E,
            pred_y=pred.y,
            true_X=X,
            true_E=E,
            true_y=other_data["y"],
        )

        batch_length = other_data["y"].shape[0]
        conditionAll = other_data["conditionVec"]
        conditionAll = conditionAll.reshape(batch_length, self.vocabDim)

        nll = self.compute_val_loss(
            pred,
            noisy_data,
            dense_data.X,
            dense_data.E,
            other_data["y"],
            node_mask,
            condition=conditionAll,
            test=False,
        )
        loss["nll"] = nll
        
        # save the data for visualization
        self.val_y_collection.append(other_data["conditionVec"])
        self.val_atomCount.append(paddle.to_tensor(other_data["atom_count"]))
        self.val_data_X.append(X)
        self.val_data_E.append(E)

        return loss

    def compute_val_loss(
        self, pred, noisy_data, X, E, y, node_mask, condition, test=False
    ):
        """
        计算 validation/test 阶段的 NLL (variational lower bound 估计)
        """
        t = noisy_data["t"]

        # 1. log p(N) = number of nodes 先验
        N = paddle.sum(node_mask, axis=1).astype("int64")
        log_pN = self.node_dist.log_prob(N)

        # 2. KL(q(z_T|x), p(z_T)) => uniform prior
        kl_prior = self.kl_prior(X, E, node_mask)

        # 3. 逐步扩散损失
        loss_all_t = self.compute_Lt(X, E, y, pred, noisy_data, node_mask, test)

        # 4. 重构损失
        prob0 = self.reconstruction_logp(t, X, E, node_mask, condition)
        loss_term_0_x = X * paddle.log(prob0.X + 1e-10)  # avoid log(0)
        loss_term_0_e = E * paddle.log(prob0.E + 1e-10)

        # 这里 val_X_logp / val_E_logp 进行加和
        loss_term_0 = self.val_X_logp(loss_term_0_x) + self.val_E_logp(loss_term_0_e)

        # combine
        nlls = -log_pN + kl_prior + loss_all_t - loss_term_0
        # shape: (bs, ), 对batch做均值
        nll = (self.test_nll if test else self.val_nll)(nlls)

        return nll

    def kl_prior(self, X, E, node_mask):
        """
        KL between q(zT|x) and prior p(zT)=Uniform(...)
        """
        bs = X.shape[0]
        ones = paddle.ones([bs, 1], dtype="float32")
        Ts = self.T * ones
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_int=Ts)  # (bs,1)

        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar)
        probX = paddle.matmul(X, Qtb.X)  # (bs,n,dx_out)
        probE = paddle.matmul(E, Qtb.E.unsqueeze(1))  # (bs,n,n,de_out)

        # limit分布
        limit_X = self.limit_dist.X.unsqueeze(0).unsqueeze(0)  # shape (1,1,dx_out)
        limit_X = paddle.expand(limit_X, [bs, X.shape[1], self.Xdim_output])

        limit_E = self.limit_dist.E.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        limit_E = paddle.expand(limit_E, [bs, E.shape[1], E.shape[2], self.Edim_output])

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

    def compute_Lt(self, X, E, y, pred, noisy_data, node_mask, test):
        """
        逐步扩散的 KL 估计
        """
        pred_probs_X = F.softmax(pred.X, axis=-1)
        pred_probs_E = F.softmax(pred.E, axis=-1)
        pred_probs_y = F.softmax(pred.y, axis=-1)

        Qtb = self.transition_model.get_Qt_bar(noisy_data["alpha_t_bar"])
        Qsb = self.transition_model.get_Qt_bar(noisy_data["alpha_s_bar"])
        Qt = self.transition_model.get_Qt(noisy_data["beta_t"])

        bs, n, _ = X.shape
        # 计算真实后验分布
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

        # 计算预测后验分布
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
        
        prob_true_X, prob_true_E, prob_pred.X, prob_pred.E = \
            diffusion_utils.mask_distributions(
                true_X=prob_true.X,
                true_E=prob_true.E,
                pred_X=prob_pred.X,
                pred_E=prob_pred.E,
                node_mask=node_mask,
            )

        # KL
        kl_x = (self.test_X_kl if test else self.val_X_kl)(
            prob_true.X, paddle.log(prob_pred.X + 1e-10)
        )
        kl_e = (self.test_E_kl if test else self.val_E_kl)(
            prob_true.E, paddle.log(prob_pred.E + 1e-10)
        )
        return self.T * (kl_x + kl_e)

    def reconstruction_logp(self, t, X, E, node_mask, condition):
        """
        L0: - log p(X,E|z0)
        这里随机从 X0, E0 采样, 再前向
        """
        t_zeros = paddle.zeros_like(t)
        beta_0 = self.noise_schedule(t_zeros)
        Q0 = self.transition_model.get_Qt(beta_t=beta_0)

        probX0 = paddle.matmul(X, Q0.X)
        # E => broadcast
        probE0 = paddle.matmul(E, Q0.E.unsqueeze(1))

        sampled0 = diffusion_utils.sample_discrete_features(probX0, probE0, node_mask)
        X0 = F.one_hot(sampled0.X, num_classes=self.Xdim_output)
        E0 = F.one_hot(sampled0.E, num_classes=self.Edim_output)
        y0 = sampled0.y  

        # noisy_data
        noisy_data = {
            "X_t": X0,
            "E_t": E0,
            "y_t": y0,
            "node_mask": node_mask,
            "t": paddle.zeros([X0.shape[0], 1]).astype("float32"),
        }
        extra_data = self.compute_extra_data(noisy_data)

        # input_X
        input_X = paddle.concat(
            [noisy_data["X_t"].astype("float"), extra_data.X], axis=2
        ).astype(dtype="float32")

        # input_E
        input_E = paddle.concat(
            [noisy_data["E_t"].astype("float"), extra_data.E], axis=3
        ).astype(dtype="float32")

        # input_y with encoder output as condition vector of input of decoder
        input_y = paddle.hstack(
            [noisy_data["y_t"].astype("float"), extra_data.y]
        ).astype(dtype="float32")

        y0 = paddle.zeros(shape=[input_X.shape[0], 1024]).cuda(blocking=True)

        conditionVec = self.encoder(X0, E0, y0, node_mask)
        input_y = paddle.hstack(x=(input_y, conditionVec)).astype(dtype="float32")

        # forward of decoder with encoder output as condition vector of input of decoder
        pred0 = self.decoder(input_X, input_E, input_y, node_mask)

        probX0 = F.softmax(pred0.X, axis=-1)
        probE0 = F.softmax(pred0.E, axis=-1)
        proby0 = F.softmax(pred0.y, axis=-1)

        # mask
        probX0[~node_mask] = 1.0 / probX0.shape[-1]
        # E -> (bs, n, n, de_out)
        # 屏蔽 ~mask
        expand_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
        probE0[~expand_mask] = 1.0 / probE0.shape[-1]

        diag_mask = paddle.eye(probE0.shape[1]).astype("bool")
        diag_mask = diag_mask.unsqueeze(0).expand([probE0.shape[0], -1, -1])
        probE0[diag_mask] = 1.0 / probE0.shape[-1]

        # 返回概率
        return utils.PlaceHolder(X=probX0, E=probE0, y=proby0)

    @paddle.no_grad()
    def sample_batch(
        self,
        batch_id: int,
        batch_size: int,
        batch_condition,
        keep_chain: int,
        number_chain_steps: int,
        save_final: int,
        batch_X,
        batch_E,
        num_nodes=None,
    ):
        """
        采样: 反向扩散
        """
        if num_nodes is None:
            n_nodes = self.node_dist.sample_n(batch_size, None)  # device
        elif isinstance(num_nodes, int):
            n_nodes = paddle.full([batch_size], num_nodes, dtype="int64")
        else:
            n_nodes = num_nodes  # assume Tensor
        n_max = int(paddle.max(n_nodes).item())

        # node_mask
        arange = paddle.arange(n_max).unsqueeze(0).expand([batch_size, n_max])
        node_mask = arange < n_nodes.unsqueeze(1)

        # z_T
        z_T = diffusion_utils.sample_discrete_feature_noise(
            limit_dist=self.limit_dist, node_mask=node_mask
        )
        X, E, y = z_T.X, z_T.E, z_T.y

        chain_X = paddle.zeros(
            [number_chain_steps, keep_chain, X.shape[1]], dtype="int64"
        )
        chain_E = paddle.zeros(
            [number_chain_steps, keep_chain, E.shape[1], E.shape[2]], dtype="int64"
        )

        # 逐步还原
        for s_int in reversed(range(self.T)):
            s_array = paddle.full([batch_size, 1], float(s_int))
            t_array = s_array + 1.0
            s_norm = s_array / self.T
            t_norm = t_array / self.T

            # Sample z_s
            sampled_s, discrete_sampled_s = self.sample_p_zs_given_zt(
                s=s_norm,
                t=t_norm,
                X_t=X,
                E_t=E,
                y_t=y,
                node_mask=node_mask,
                conditionVec=batch_condition,
                batch_X=batch_X,
                batch_E=batch_E,
            )
            X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

            write_index = (s_int * number_chain_steps) // self.T
            if write_index >= 0 and write_index < number_chain_steps:
                chain_X[write_index] = discrete_sampled_s.X[:keep_chain]
                chain_E[write_index] = discrete_sampled_s.E[:keep_chain]

        # 最终 mask
        sampled_s = sampled_s.mask(node_mask, collapse=True)
        X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

        # batch_X, batch_E
        batch_X = paddle.argmax(batch_X, axis=-1)
        batch_E = paddle.argmax(batch_E, axis=-1)

        # 组装 output
        molecule_list = []
        molecule_list_True = []
        n_nodes_np = n_nodes.numpy()

        for i in range(batch_size):
            n = n_nodes_np[i]
            atom_types = X[i, :n].cpu()
            edge_types = E[i, :n, :n].cpu()

            atom_types_true = batch_X[i, :n].cpu()
            edge_types_true = batch_E[i, :n, :n].cpu()

            molecule_list.append([atom_types, edge_types])
            molecule_list_True.append([atom_types_true, edge_types_true])

        # 可视化
        if self.visualization_tools is not None:
            num_molecules = chain_X.shape[1]
            for i in range(num_molecules):
                # chain_X与chain_E => numpy
                chain_X_np = chain_X[:, i, :].numpy()
                chain_E_np = chain_E[:, i, :, :].numpy()

                self.visualization_tools.visualize_chain(
                    batch_id, i, chain_X_np, chain_E_np
                )
                logger.message(f"{i+1}/{num_molecules} complete")

            self.visualization_tools.visualizeNmr(
                batch_id,
                molecule_list,
                molecule_list_True,
                save_final,
            )

        return molecule_list, molecule_list_True

    @paddle.no_grad()
    def sample_p_zs_given_zt(
        self, s, t, X_t, E_t, y_t, node_mask, conditionVec, batch_X, batch_E
    ):
        """
        从 p(z_s | z_t) 采样: 反向扩散一步
        """
        beta_t = self.noise_schedule(t_normalized=t)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t)

        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar)
        Qsb = self.transition_model.get_Qt_bar(alpha_s_bar)
        Qt = self.transition_model.get_Qt(beta_t)

        # forward
        noisy_data = {
            "X_t": X_t,
            "E_t": E_t,
            "y_t": y_t,
            "t": t,
            "node_mask": node_mask,
        }
        extra_data = self.compute_extra_data(noisy_data)
        
        # input_X
        input_X = paddle.concat(
            [noisy_data["X_t"].astype("float"), extra_data.X], axis=2
        ).astype(dtype="float32")

        # input_E
        input_E = paddle.concat(
            [noisy_data["E_t"].astype("float"), extra_data.E], axis=3
        ).astype(dtype="float32")

        # input_y with encoder output as condition vector of input of decoder
        input_y = paddle.hstack(
            [noisy_data["y_t"].astype("float"), extra_data.y]
        ).astype(dtype="float32")

        y_condition = paddle.zeros(shape=[input_X.shape[0], 1024]).cuda(blocking=True)
        
        conditionVec = self.encoder(batch_X, batch_E, y_condition, node_mask)
        input_y = paddle.hstack(x=(input_y, conditionVec)).astype(dtype="float32") 
        
        # forward of decoder with encoder output as condition vector of input of decoder
        pred = self.decoder(input_X, input_E, input_y, node_mask)


        pred_X = F.softmax(pred.X, axis=-1)
        pred_E = F.softmax(pred.E, axis=-1).reshape(
            [X_t.shape[0], -1, pred.E.shape[-1]]
        )

        p_s_and_t_given_0_X = (
            diffusion_utils.compute_batched_over0_posterior_distribution(
                X_t=X_t, Qt=Qt.X, Qsb=Qsb.X, Qtb=Qtb.X
            )
        )
        p_s_and_t_given_0_E = (
            diffusion_utils.compute_batched_over0_posterior_distribution(
                X_t=E_t, Qt=Qt.E, Qsb=Qsb.E, Qtb=Qtb.E
            )
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
        X_s = F.one_hot(sampled_s.X, num_classes=self.Xdim_output)
        E_s = F.one_hot(sampled_s.E, num_classes=self.Edim_output)

        out_one_hot = utils.PlaceHolder(X=X_s, E=E_s, y=paddle.zeros([y_t.shape[0], 0]))
        out_discrete = utils.PlaceHolder(
            X=X_s, E=E_s, y=paddle.zeros([y_t.shape[0], 0])
        )

        return out_one_hot.mask(node_mask), out_discrete.mask(node_mask, collapse=True)

    # -----------------------
    # 分子可视化/对比
    # -----------------------
    def mol_from_graphs(self, node_list, adjacency_matrix):
        """
        将离散图 (atom indices, adjacency) 转为 rdkit Mol
        """
        atom_decoder = self.dataset_info.atom_decoder
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


class ContrastiveModel(nn.Layer):
    def __init__(
        self,
        graph_encoder: dict,
        nmr_encoder: dict,
        **kwargs,
    ):
        super().__init__()
        self.name = kwargs.get("__name__")
        self.text_encoder = Encoder(
            enc_voc_size=nmr_encoder["enc_voc_size"],
            max_len=nmr_encoder["max_len"],
            d_model=nmr_encoder["d_model"],
            ffn_hidden=nmr_encoder["ffn_hidden"],
            n_head=nmr_encoder["n_head"],
            n_layers=nmr_encoder["n_layers"],
            drop_prob=nmr_encoder["drop_prob"],
        )

        if nmr_encoder["projector"]["__name__"] == "Linear":
            self.text_encoder_projector = paddle.nn.Linear(
                in_features=nmr_encoder["max_len"] * nmr_encoder["d_model"],
                out_features=nmr_encoder["projector"]["outfeatures"],
            )

        self.con_input_dim = graph_encoder["input_dims"]
        self.con_input_dim["X"] = graph_encoder["input_dims"]["X"] - 8
        self.con_input_dim["y"] = 1024  # to be write in yaml
        self.con_output_dim = graph_encoder["output_dims"]
        self.graph_encoder = GraphTransformer_C(
            n_layers=graph_encoder["n_layers_GT"],
            input_dims=self.con_input_dim,
            hidden_mlp_dims=graph_encoder["hidden_mlp_dims"],
            hidden_dims=graph_encoder["hidden_dims"],
            output_dims=self.con_output_dim,
            act_fn_in=paddle.nn.ReLU(),
            act_fn_out=paddle.nn.ReLU(),
        )
        for param in self.graph_encoder.parameters():
            param.stop_gradient = True
        self.graph_encoder.eval()
        self.vocabDim = graph_encoder["vocab_dim"]
        self.tem = 2

        # TODO: need to revise pretrained parameters for model
        # for Constrastive Learning & Prior Training
        if graph_encoder["pretrained_model_path"] is not None:
            save_load.load_pretrain(
                self.graph_encoder, graph_encoder["pretrained_model_path"]
            )
        # for Prior Training
        if (
            "pretrained_model_path" in nmr_encoder
            and nmr_encoder["pretrained_model_path"] is not None
        ):
            save_load.load_pretrain(
                self.text_encoder, nmr_encoder["pretrained_model_path"]
            )

    def make_src_mask(self, src):
        src_mask = (src != 0).unsqueeze(axis=1).unsqueeze(axis=2)
        return src_mask

    def forward_(self, node_mask, X_condition, E_condtion, conditionVec):
        assert isinstance(
            conditionVec, paddle.Tensor
        ), "conditionVec should be a tensor, but got type {}".format(type(conditionVec))
        srcMask = self.make_src_mask(conditionVec)  # .to(self.device)
        conditionVecNmr = self.text_encoder(conditionVec, srcMask)
        conditionVecNmr = conditionVecNmr.reshape([conditionVecNmr.shape[0], -1])
        conditionVecNmr = self.text_encoder_projector(conditionVecNmr)
        y_condition = paddle.zeros(shape=[X_condition.shape[0], 1024]).cuda(
            blocking=True
        )
        conditionVecM = self.graph_encoder(
            X_condition, E_condtion, y_condition, node_mask
        )
        return conditionVecM, conditionVecNmr

    def forward(self, batch):
        batch_graph, other_data = batch
        batch_length = batch_graph.num_graph
        # transfer to dense graph from sparse graph
        if batch_graph.edges.T.numel() == 0:
            print("Found a batch with no edges. Skipping.")
            return None
        dense_data, node_mask = utils.to_dense(
            batch_graph.node_feat["feat"],
            batch_graph.edges.T.contiguous(),
            batch_graph.edge_feat["feat"],
            batch_graph.graph_node_id,
        )
        dense_data = dense_data.mask(node_mask)
        X, E = dense_data.X, dense_data.E
        conditionAll = other_data["conditionVec"]
        conditionAll = conditionAll.reshape([batch_length, self.vocabDim])
        conditionVecM, conditionVecNmr = self.forward_(node_mask, X, E, conditionAll)

        V1_f = conditionVecM  # 假设 V1 是从图像（或者其他模态）得到的特征
        V2_f = conditionVecNmr  # 假设 V2 是从文本（或者其他模态）得到的特征

        V1_e = paddle.nn.functional.normalize(x=V1_f, p=2, axis=1)
        V2_e = paddle.nn.functional.normalize(x=V2_f, p=2, axis=1)
        logits = paddle.matmul(x=V1_e, y=V2_e.T) * paddle.exp(
            x=paddle.to_tensor(data=self.tem, place=V1_e.place)
        )
        n = V1_f.shape[0]
        labels = paddle.arange(end=n)
        loss_fn = paddle.nn.CrossEntropyLoss()
        loss_v1 = loss_fn(logits, labels)
        loss_v2 = loss_fn(logits.T, labels)
        loss = (loss_v1 + loss_v2) / 2

        return {"loss": loss}


class DiffusionPriorModel(nn.Layer):
    def __init__(
        self,
        config,
        model: nn.Layer,
        clip: nn.Layer,
        graph_embed_scale=None,
        timesteps: int = 1000,
        cond_drop_prob: float = 0.0,
        condition_on_text_encodings: bool = True,
    ):
        super().__init__()

        # 初始化TrainerDiffPrior类的实例变量
        self.config = config
        self.model = model
        self.clip = clip

        self.sample_timesteps = config["sample_timesteps"]
        self.noise_scheduler = NoiseScheduler(
            beta_schedule=config["beta_schedule"],
            timesteps=config["timesteps"],
            loss_type=config["loss_type"],
        )
        if exists(clip):
            freeze_model_and_make_eval_(clip)
            self.clip = clip
        else:
            self.clip = None
        self.net = model
        self.graph_embed_dim = default(
            config["graph_embed_dim"], lambda: clip.dim_latent
        )

        assert (
            model.dim == self.graph_embed_dim
        ), f"your diffusion prior network has a dimension of {model.dim}, \
            but you set your image embedding dimension (keyword graph_embed_dim) \
            on DiffusionPrior to {self.graph_embed_dim}"
        assert (
            not exists(clip)
            or clip.text_encoder_projector.weight.shape[1] == self.graph_embed_dim
        ), f"you passed in a CLIP to the diffusion prior with latent dimensions of \
            {clip.dim_latent}, but your image embedding dimension \
            (keyword graph_embed_dim) for the DiffusionPrior was set \
            to {self.graph_embed_dim}"

        self.text_cond_drop_prob = default(
            config["text_cond_drop_prob"], config["cond_drop_prob"]
        )
        self.graph_cond_drop_prob = default(
            config["graph_cond_drop_prob"], config["cond_drop_prob"]
        )

        self.can_classifier_guidance = (
            self.text_cond_drop_prob > 0.0 and self.graph_cond_drop_prob > 0.0
        )
        self.condition_on_text_encodings = config["condition_on_text_encodings"]

        # in paper, they do not predict the noise, but predict x0 directly for image
        # embedding, claiming empirically better results. I'll just offer both.
        self.predict_x_start = config["predict_x_start"]
        self.predict_v = config["predict_v"]

        # @crowsonkb 's suggestion - https://github.com/lucidrains/DALLE2-pytorch/issues/60#issue-1226116132

        self.graph_embed_scale = default(
            graph_embed_scale, config["graph_embed_dim"] ** 0.5
        )

        # whether to force an l2norm, similar to clipping denoised, when sampling

        self.sampling_clamp_l2norm = config["sampling_clamp_l2norm"]
        self.sampling_final_clamp_l2norm = config["sampling_final_clamp_l2norm"]

        self.training_clamp_l2norm = config["training_clamp_l2norm"]
        self.init_graph_embed_l2norm = config["init_graph_embed_l2norm"]

        # device tracker, todo: maybe could be deleted
        self.register_buffer(
            name="_dummy", tensor=paddle.to_tensor(data=[True]), persistable=False
        )

    def forward(
        self,
        text=None,
        moleculargraph=None,
        text_embed=None,
        moleculargraph_embed=None,
        text_encodings=None,
        *args,
        **kwargs,
    ):
        assert exists(text) ^ exists(
            text_embed
        ), "either text or text embedding must be supplied"
        assert exists(moleculargraph) ^ exists(
            moleculargraph_embed
        ), "either moleculegraph or moleculegraph embedding must be supplied"
        assert not (
            self.condition_on_text_encodings
            and (not exists(text_encodings) and not exists(text))
        ), "cannot use both conditions at once"

        if exists(moleculargraph):
            moleculargraph_embed, _ = self.clip.graph_encoder(moleculargraph)

        # calculate text conditionings, based on what is passed in
        if exists(text):
            text_embed, text_encodings = self.clip.text_encoder(text)

        text_cond = dict(text_embed=text_embed)

        if self.condition_on_text_encodings:
            assert exists(
                text_encodings
            ), "text encodings must be present for diffusion prior if specified"
            text_cond = {**text_cond, "text_encodings": text_encodings}

        # timestep conditioning from ddpm
        batch = tuple(moleculargraph_embed.shape)[0]
        times = self.noise_scheduler.sample_random_times(batch)

        # scale image embed (Katherine)
        moleculargraph_embed *= self.graph_embed_scale

        # calculate forward loss
        return self.p_losses(
            moleculargraph_embed, times, text_cond=text_cond, *args, **kwargs
        )

    def generate_embed_vector(self, batch):
        batch_graph, other_data = batch
        batch_length = batch_graph.num_graph
        # transfer to dense graph from sparse graph
        if batch_graph.edges.T.numel() == 0:
            print("Found a batch with no edges. Skipping.")
            return None
        dense_data, node_mask = utils.to_dense(
            batch_graph.node_feat["feat"],
            batch_graph.edges.T.contiguous(),
            batch_graph.edge_feat["feat"],
            batch_graph.graph_node_id,
        )
        dense_data = dense_data.mask(node_mask)
        graph_X, graph_E = (
            dense_data.X,
            dense_data.E,
        )
        graph_y = paddle.zeros(shape=[graph_X.shape[0], 1024]).cuda(blocking=True)

        clip_graph_embeds = self.clip.graph_encoder(
            graph_X, graph_E, graph_y, node_mask
        )

        text_conditionVec = other_data["conditionVec"]
        text_conditionVec = text_conditionVec.reshape(
            [batch_length, self.config["CLIP"]["nmr_encoder"]["max_len"]]
        )

        assert isinstance(
            text_conditionVec, paddle.Tensor
        ), "nmr_text_conditionVec should be a tensor, but got type {}".format(
            type(text_conditionVec)
        )
        text_srcMask = self.clip.make_src_mask(text_conditionVec)

        clip_text_embeds = self.clip.text_encoder(text_conditionVec, text_srcMask)
        clip_text_embeds = clip_text_embeds.reshape([clip_text_embeds.shape[0], -1])
        clip_text_embeds = self.clip.text_encoder_projector(clip_text_embeds)

        return clip_graph_embeds, clip_text_embeds

    def p_losses(self, moleculargraph_embed, times, text_cond, noise=None):
        noise = default(
            noise,
            lambda: paddle.randn(
                shape=moleculargraph_embed.shape, dtype=moleculargraph_embed.dtype
            ),
        )

        moleculargraph_embed_noisy = self.noise_scheduler.q_sample(
            x_start=moleculargraph_embed, t=times, noise=noise
        )

        self_cond = None
        if self.net.self_cond and random.random() < 0.5:
            with paddle.no_grad():
                self_cond = self.net(
                    moleculargraph_embed_noisy, times, **text_cond
                ).detach()

        pred = self.net(
            moleculargraph_embed_noisy,
            times,
            self_cond=self_cond,
            text_cond_drop_prob=self.text_cond_drop_prob,
            graph_cond_drop_prob=self.graph_cond_drop_prob,
            **text_cond,
        )

        if self.predict_x_start and self.training_clamp_l2norm:
            pred = self.l2norm_clamp_embed(pred)

        if self.predict_v:
            target = self.noise_scheduler.calculate_v(
                moleculargraph_embed, times, noise
            )
        elif self.predict_x_start:
            target = moleculargraph_embed
        else:
            target = noise

        loss = self.noise_scheduler.loss_fn(pred, target)

        return {"loss": loss}

    def l2norm_clamp_embed(self, graph):
        return l2norm(graph) * self.graph_embed_scale

    @paddle.no_grad()
    def sample(
        self, text, mask, num_samples_per_batch=2, cond_scale=1.0, timesteps=None
    ):
        timesteps = default(timesteps, self.sample_timesteps)

        # in the paper, what they did was
        # sample 2 image embeddings, choose the top 1 similarity, as judged by CLIP
        text = repeat(text, "b ... -> (b r) ...", r=num_samples_per_batch)
        mask = repeat(mask, "b ... -> (b r) ...", r=num_samples_per_batch)

        batch_size = tuple(text.shape)[0]
        graph_embed_dim = self.graph_embed_dim

        text_embeds = self.clip.text_encoder(text, mask)
        text_embeds = text_embeds.reshape([text_embeds.shape[0], -1])
        text_embeds = self.clip.text_encoder_projector(text_embeds)

        text_cond = dict(text_embed=text_embeds)

        if self.condition_on_text_encodings:
            text_encodings = None  # TODO: revise this
            text_cond = {**text_cond, "text_encodings": text_encodings}

        graph_embeds = self.p_sample_loop(
            (batch_size, graph_embed_dim),
            text_cond=text_cond,
            cond_scale=cond_scale,
            timesteps=timesteps,
        )

        # retrieve original unscaled image embed

        text_embeds = text_cond["text_embed"]

        text_embeds = rearrange(
            text_embeds, "(b r) d -> b r d", r=num_samples_per_batch
        )
        graph_embeds = rearrange(
            graph_embeds, "(b r) d -> b r d", r=num_samples_per_batch
        )

        text_image_sims = paddle.einsum(
            "b r d, b r d -> b r", l2norm(text_embeds), l2norm(graph_embeds)
        )
        top_sim_indices = text_image_sims.topk(k=1)[1]

        top_sim_indices = repeat(top_sim_indices, "b 1 -> b 1 d", d=graph_embed_dim)

        top_graph_embeds = graph_embeds.take_along_axis(
            axis=1, indices=top_sim_indices, broadcast=False
        )
        return rearrange(top_graph_embeds, "b 1 d -> b d")

    @paddle.no_grad()
    def p_sample_loop(self, *args, timesteps=None, **kwargs):
        timesteps = default(timesteps, self.noise_scheduler.num_timesteps)
        assert timesteps <= self.noise_scheduler.num_timesteps
        is_ddim = timesteps < self.noise_scheduler.num_timesteps
        if not is_ddim:
            normalized_graph_embed = self.p_sample_loop_ddpm(*args, **kwargs)
        else:
            normalized_graph_embed = self.p_sample_loop_ddim(
                *args, **kwargs, timesteps=timesteps
            )
        graph_embed = normalized_graph_embed / self.graph_embed_scale
        return graph_embed

    @paddle.no_grad()
    def p_sample_loop_ddpm(self, shape, text_cond, cond_scale=1.0):
        batch = shape[0]
        graph_embed = paddle.randn(shape=shape)
        x_start = None
        if self.init_graph_embed_l2norm:
            graph_embed = l2norm(graph_embed) * self.graph_embed_scale
        for i in tqdm(
            reversed(range(0, self.noise_scheduler.num_timesteps)),
            desc="sampling loop time step",
            total=self.noise_scheduler.num_timesteps,
        ):
            times = paddle.full(shape=(batch,), fill_value=i, dtype="int64")
            self_cond = x_start if self.net.self_cond else None
            graph_embed, x_start = self.p_sample(
                graph_embed,
                times,
                text_cond=text_cond,
                self_cond=self_cond,
                cond_scale=cond_scale,
            )
        if self.sampling_final_clamp_l2norm and self.predict_x_start:
            graph_embed = self.l2norm_clamp_embed(graph_embed)
        return graph_embed

    @paddle.no_grad()
    def p_sample_loop_ddim(
        self, shape, text_cond, *, timesteps, eta=1.0, cond_scale=1.0
    ):
        batch, alphas, total_timesteps = (
            shape[0],
            self.noise_scheduler.alphas_cumprod_prev,
            self.noise_scheduler.num_timesteps,
        )
        times = paddle.linspace(start=-1.0, stop=total_timesteps, num=timesteps + 1)[
            :-1
        ]
        times = list(reversed(times.astype(dtype="int32").tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))
        graph_embed = paddle.randn(shape=shape)
        x_start = None
        if self.init_graph_embed_l2norm:
            graph_embed = l2norm(graph_embed) * self.graph_embed_scale
        for time, time_next in tqdm(time_pairs, desc="sampling loop time step"):
            alpha = alphas[time]
            alpha_next = alphas[time_next]
            time_cond = paddle.full(shape=(batch,), fill_value=time, dtype="int64")
            self_cond = x_start if self.net.self_cond else None
            pred = self.net.forward_with_cond_scale(
                graph_embed,
                time_cond,
                self_cond=self_cond,
                cond_scale=cond_scale,
                **text_cond,
            )
            if self.predict_v:
                x_start = self.noise_scheduler.predict_start_from_v(
                    graph_embed, t=time_cond, v=pred
                )
            elif self.predict_x_start:
                x_start = pred
            else:
                x_start = self.noise_scheduler.predict_start_from_noise(
                    graph_embed, t=time_cond, noise=pred
                )
            if not self.predict_x_start:
                x_start.clip_(min=-1.0, max=1.0)
            if self.predict_x_start and self.sampling_clamp_l2norm:
                x_start = self.l2norm_clamp_embed(x_start)
            pred_noise = self.noise_scheduler.predict_noise_from_start(
                graph_embed, t=time_cond, x0=x_start
            )
            if time_next < 0:
                graph_embed = x_start
                continue
            c1 = (
                eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            )
            c2 = (1 - alpha_next - paddle.square(x=c1)).sqrt()
            noise = (
                paddle.randn(shape=graph_embed.shape, dtype=graph_embed.dtype)
                if time_next > 0
                else 0.0
            )
            graph_embed = x_start * alpha_next.sqrt() + c1 * noise + c2 * pred_noise
        if self.predict_x_start and self.sampling_final_clamp_l2norm:
            graph_embed = self.l2norm_clamp_embed(graph_embed)
        return graph_embed

    @paddle.no_grad()
    def p_sample(
        self, x, t, text_cond=None, self_cond=None, clip_denoised=True, cond_scale=1.0
    ):
        (
            b,
            *_,
        ) = x.shape
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            x=x,
            t=t,
            text_cond=text_cond,
            self_cond=self_cond,
            clip_denoised=clip_denoised,
            cond_scale=cond_scale,
        )
        noise = paddle.randn(shape=x.shape, dtype=x.dtype)
        nonzero_mask = (1 - (t == 0).astype(dtype="float32")).reshape(
            b, *((1,) * (len(tuple(x.shape)) - 1))
        )
        pred = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        return pred, x_start

    def l2norm_clamp_embed(self, graph_embed):
        return l2norm(graph_embed) * self.graph_embed_scale

    def p_mean_variance(
        self, x, t, text_cond, self_cond=None, clip_denoised=False, cond_scale=1.0
    ):
        assert not (
            cond_scale != 1.0 and not self.can_classifier_guidance
        ), "the model was not trained with conditional dropout, and thus one cannot \
            use classifier free guidance (cond_scale anything other than 1)"
        pred = self.net.forward_with_cond_scale(
            x, t, cond_scale=cond_scale, self_cond=self_cond, **text_cond
        )
        if self.predict_v:
            x_start = self.noise_scheduler.predict_start_from_v(x, t=t, v=pred)
        elif self.predict_x_start:
            x_start = pred
        else:
            x_start = self.noise_scheduler.predict_start_from_noise(x, t=t, noise=pred)
        if clip_denoised and not self.predict_x_start:
            x_start.clip_(min=-1.0, max=1.0)
        if self.predict_x_start and self.sampling_clamp_l2norm:
            x_start = l2norm(x_start) * self.graph_embed_scale
        (
            model_mean,
            posterior_variance,
            posterior_log_variance,
        ) = self.noise_scheduler.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start


class MultiModalDecoder(nn.Layer):
    def __init__(
        self,
        config,
        dataset_infos,
        train_metrics,
        sampling_metrics,
        visualization_tools,
        extra_features,
        domain_features,
    ) -> None:
        super().__init__()

        #############################################################
        # configure general variables settings
        #############################################################
        self.name = config["__name__"]
        self.model_dtype = paddle.get_default_dtype()
        self.T = config["graph_decoder"]["diffusion_model"]["diffusion_steps"]
        self.visualization_tools = visualization_tools

        #############################################################
        # configure datasets inter-varibles
        #############################################################
        input_dims = dataset_infos.input_dims
        output_dims = dataset_infos.output_dims
        self.dataset_info = dataset_infos
        self.extra_features = extra_features
        self.domain_features = domain_features

        #############################################################
        # configure noise scheduler
        #############################################################
        self.noise_schedule = PredefinedNoiseScheduleDiscrete(
            config["graph_decoder"]["diffusion_model"]["diffusion_noise_schedule"],
            timesteps=self.T,
        )

        #############################################################
        # configure model
        #############################################################
        self.add_condition = True

        # set nmr encoder model #####################################
        self.encoder = Encoder(
            enc_voc_size=config["nmr_encoder"]["enc_voc_size"],
            max_len=config["nmr_encoder"]["max_len"],
            d_model=config["nmr_encoder"]["d_model"],
            ffn_hidden=config["nmr_encoder"]["ffn_hidden"],
            n_head=config["nmr_encoder"]["n_head"],
            n_layers=config["nmr_encoder"]["n_layers"],
            drop_prob=config["nmr_encoder"]["drop_prob"],
        )
        # load nmr encoder model from pretrained model
        state_dict = paddle.load(config["nmr_encoder"]["pretrained_path"])
        encoder_state_dict = {
            k[len("encoder.") :]: v
            for k, v in state_dict.items()
            if k.startswith("encoder.")
        }
        self.encoder.set_state_dict(encoder_state_dict)

        # set nmr encoder projector head model #####################
        state_dict = paddle.load(config["nmr_encoder"]["pretrained_path"])
        if config["nmr_encoder"]["projector"]["__name__"] == "Linear":
            self.encoder_projector = paddle.nn.Linear(
                in_features=config["nmr_encoder"]["max_len"] * config["nmr_encoder"]["d_model"],
                out_features=config["nmr_encoder"]["projector"]["outfeatures"],
            )
            encoder_projector_state_dict = {
                k[len("linear_layer.") :]: v
                for k, v in state_dict.items()
                if k.startswith("linear_layer.")
            }
            self.encoder_projector.set_state_dict(encoder_projector_state_dict)
        # TODO: add support for LSTM
        # elif config["encoder"]["__name__"] == "LSTM":
        #    encoder_projector_state_dict = {
        #        k[len("LSTM.") :]: v
        #        for k, v in state_dict.items()
        #        if k.startswith("lstm.")
        #    }
        #    self.encoder_projector.set_state_dict(encoder_projector_state_dict)

        # set graph decoder model ###################################
        self.decoder = GraphTransformer(
            n_layers=config["graph_decoder"]["num_layers"],
            input_dims=input_dims,
            hidden_mlp_dims=config["graph_decoder"]["hidden_mlp_dims"],
            hidden_dims=config["graph_decoder"]["hidden_dims"],
            output_dims=output_dims,
            act_fn_in=nn.ReLU(),
            act_fn_out=nn.ReLU(),
        )
        # load graph decoder model from pretrained model
        state_dict = paddle.load(config["graph_decoder"]["pretrained_path"])
        decoder_state_dict = {
            k[len("decoder.") :]: v
            for k, v in state_dict.items()
            if k.startswith("decoder.")
        }
        self.decoder.set_state_dict(decoder_state_dict)

        # set connector model #######################################
        self.connector_flag = False
        if config.get("connector") and config["connector"]["__name__"] == "DiffPrior":
            self.connector_flag = True
            self.connector = DiffusionPriorModel(
                config= config["connector"],
                model = DiffusionPriorNetwork(
                    dim=config["connector"]["prior_network"]["dim"],
                    num_timesteps=config["connector"]["prior_network"]["num_timesteps"],
                    num_time_embeds=config["connector"]["prior_network"]["num_time_embeds"],
                    num_graph_embeds=config["connector"]["prior_network"]["num_graph_embeds"],
                    num_text_embeds=config["connector"]["prior_network"]["num_text_embeds"],
                    max_text_len=config["connector"]["prior_network"]["max_text_len"],
                    self_cond=config["connector"]["prior_network"]["self_cond"],
                    depth=config["connector"]["prior_network"]["depth"],
                    dim_head=config["connector"]["prior_network"]["dim_head"],
                    heads=config["connector"]["prior_network"]["heads"],
                ),
                clip=ContrastiveModel(**config["clip"]),
            )
            state_dict = paddle.load(config["connector"]["pretrained_path"])
            connector_state_dict = {
                k[len("connector.") :]: v
                for k, v in state_dict.items()
                if k.startswith("connector.")
            }
            self.connector.set_state_dict(connector_state_dict)
        else:
            self.connector = EmptyLayer()


        #############################################################
        # configure loss calculation with initialization of transition model
        #############################################################
        self.Xdim = input_dims["X"]
        self.Edim = input_dims["E"]
        self.ydim = input_dims["y"]
        self.Xdim_output = output_dims["X"]
        self.Edim_output = output_dims["E"]
        self.ydim_output = output_dims["y"]
        self.node_dist = dataset_infos.nodes_dist

        # Transition Model
        if config["graph_decoder"]["diffusion_model"]["transition"] == "uniform":
            self.transition_model = DiscreteUniformTransition(
                x_classes=self.Xdim_output,
                e_classes=self.Edim_output,
                y_classes=self.ydim_output,
            )
            x_limit = paddle.ones([self.Xdim_output]) / self.Xdim_output
            e_limit = paddle.ones([self.Edim_output]) / self.Edim_output
            y_limit = paddle.ones([self.ydim_output]) / self.ydim_output
            self.limit_dist = utils.PlaceHolder(X=x_limit, E=e_limit, y=y_limit)

        elif config["graph_decoder"]["diffusion_model"]["transition"] == "marginal":
            node_types = self.dataset_info.node_types.astype(self.model_dtype)
            x_marginals = node_types / paddle.sum(node_types)

            edge_types = self.dataset_info.edge_types.astype(self.model_dtype)
            e_marginals = edge_types / paddle.sum(edge_types)
            logger.info(f"Marginal distribution of classes: {x_marginals.tolist()} for nodes, ")
            logger.info(f"{e_marginals.tolist()} for edges")

            self.transition_model = MarginalUniformTransition(
                x_marginals=x_marginals,
                e_marginals=e_marginals,
                y_classes=self.ydim_output,
            )
            self.limit_dist = utils.PlaceHolder(
                X=x_marginals,
                E=e_marginals,
                y=paddle.ones([self.ydim_output]) / self.ydim_output,
            )

        self.train_loss = TrainLossDiscrete(
            config["graph_decoder"]["diffusion_model"]["lambda_train"]
        )

        #############################################################
        # configure training setting and other properties
        #############################################################
        self.best_val_nll = 1e8
        self.val_counter = 0
        self.vocabDim = config['graph_decoder']['vocab_dim']
        self.number_chain_steps = config["graph_decoder"]["diffusion_model"]["number_chain_steps"]

        #############################################################
        # configure for visualization and test
        #############################################################
        self.val_nll = NLL()
        self.val_X_kl = SumExceptBatchKL()
        self.val_E_kl = SumExceptBatchKL()
        self.val_X_logp = SumExceptBatchMetric()
        self.val_E_logp = SumExceptBatchMetric()
        self.val_y_collection = []
        self.val_atomCount = []
        self.val_data_X = []
        self.val_data_E = []

        self.test_nll = NLL()
        self.test_X_kl = SumExceptBatchKL()
        self.test_E_kl = SumExceptBatchKL()
        self.test_X_logp = SumExceptBatchMetric()
        self.test_E_logp = SumExceptBatchMetric()
        self.test_y_collection = []
        self.test_atomCount = []
        self.test_x = []
        self.test_e = []

        self.train_metrics = train_metrics
        self.sampling_metrics = sampling_metrics
        
        #############################################################
        # configure key data container
        #############################################################
        self.val_y_collection = []
        self.val_atomCount = []
        self.val_data_X = []
        self.val_data_E = []
        self.test_y_collection = []
        self.test_atomCount = []
        self.test_data_X = []
        self.test_data_E = []

    def preprocess_data(self, batch_graph, other_data):
        dense_data, node_mask = utils.to_dense(
            batch_graph.node_feat["feat"],
            batch_graph.edges.T,
            batch_graph.edge_feat["feat"],
            batch_graph.graph_node_id,
        )
        dense_data = dense_data.mask(node_mask)

        # add noise to the inputs (X, E)
        noisy_data = m_utils.apply_noise(
            self, dense_data.X, dense_data.E, other_data["y"], node_mask
        )
        extra_data = m_utils.compute_extra_data(self, noisy_data)

        # concate data
        input_X = paddle.concat(
            [noisy_data["X_t"].astype("float"), extra_data.X], axis=2
        ).astype(dtype="float32")
        input_E = paddle.concat(
            [noisy_data["E_t"].astype("float"), extra_data.E], axis=3
        ).astype(dtype="float32")
        input_y = paddle.hstack(
            [noisy_data["y_t"].astype("float"), extra_data.y]
        ).astype(dtype="float32")

        return dense_data, noisy_data, node_mask, extra_data, input_X, input_E, input_y

    def make_src_mask(self, src):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        return src_mask

    def forward_MultiModalModel(self, X, E, y, node_mask, conditionVec):
        assert isinstance(
            conditionVec, paddle.Tensor
        ), "conditionVec should be a tensor, but got type {}".format(type(conditionVec))

        srcMask = self.make_src_mask(conditionVec).astype("float32")
        if self.connector_flag is True:
            with paddle.no_grad():
                conditionVec = self.connector.sample(conditionVec, srcMask)
        else:
            conditionVec = self.encoder(conditionVec, srcMask)
            conditionVec = conditionVec.reshape([conditionVec.shape[0], -1])
            conditionVec = self.encoder_projector(conditionVec)

        y = paddle.concat([y, conditionVec], axis=1).astype("float32")

        output = self.decoder(X, E, y, node_mask)
        return output

    def forward(self, batch):
        batch_graph, other_data = batch

        # transfer to dense graph from sparse graph
        if batch_graph.edges.T.numel() == 0:
            print("Found a batch with no edges. Skipping.")
            return None

        # process data
        (
            dense_data,
            noisy_data,
            node_mask,
            extra_data,
            input_X,
            input_E,
            input_y,
        ) = self.preprocess_data(batch_graph, other_data)
        X, E = dense_data.X, dense_data.E

        # set condition
        if self.add_condition:
            batch_length = X.shape[0]
            conditionVec = other_data["conditionVec"]
            y_condition = conditionVec.reshape(batch_length, self.vocabDim)
        else:
            y_condition = paddle.zeros(shape=[X.shape[0], 1024]).cuda(blocking=True)

        # forward of the model
        pred = self.forward_MultiModalModel(
            input_X, input_E, input_y, node_mask, y_condition
        )

        # compute loss
        loss = self.train_loss(
            masked_pred_X=pred.X,
            masked_pred_E=pred.E,
            pred_y=pred.y,
            true_X=X,
            true_E=E,
            true_y=other_data["y"],
        )
        # log metrics to do move to another location
        self.train_metrics(
            masked_pred_X=pred.X,
            masked_pred_E=pred.E,
            true_X=X,
            true_E=E,
            log=False,
        )
        return loss

    @paddle.no_grad()
    def sample(self, batch, i):
        batch_graph, other_data = batch
        
        # transfer to dense graph from sparse graph
        if batch_graph.edges.T.numel() == 0:
            print("Found a batch with no edges. Skipping.")
            return None

        # process data
        (
            dense_data,
            noisy_data,
            node_mask,
            extra_data,
            input_X,
            input_E,
            input_y,
        ) = self.preprocess_data(batch_graph, other_data)
        X, E = dense_data.X, dense_data.E

        # set condition
        if self.add_condition:
            batch_length = X.shape[0]
            conditionVec = other_data["conditionVec"]
            y_condition = conditionVec.reshape(batch_length, self.vocabDim)
        else:
            y_condition = paddle.zeros(shape=[X.shape[0], 1024]).cuda(blocking=True)
        
        # forward of the model
        pred = self.forward_MultiModalModel(
            input_X, input_E, input_y, node_mask, y_condition
        )

        # evaluate the loss especially in the inference stage
        loss = self.train_loss(
            masked_pred_X=pred.X,
            masked_pred_E=pred.E,
            pred_y=pred.y,
            true_X=X,
            true_E=E,
            true_y=other_data["y"],
        )

        batch_length = other_data["y"].shape[0]
        conditionAll = other_data["conditionVec"]
        conditionAll = conditionAll.reshape(batch_length, self.vocabDim)

        nll = m_utils.compute_val_loss(
            self,
            pred,
            noisy_data,
            dense_data.X,
            dense_data.E,
            other_data["y"],
            node_mask,
            condition=conditionAll,
            test=False,
        )
        loss["nll"] = nll
        
        # save the data for visualization
        self.val_y_collection.append(other_data["conditionVec"])
        self.val_atomCount.append(paddle.to_tensor(other_data["atom_count"]))
        self.val_data_X.append(X)
        self.val_data_E.append(E)

        return loss

    def apply_noise(self, X, E, y, node_mask):
        """
        Sample noise and apply it to the data.
        """
        t_int = paddle.randint(
            low=1, high=self.T + 1, shape=[X.shape[0], 1], dtype="int64"
        ).astype("float32")
        s_int = t_int - 1

        t_float = t_int / self.T
        s_float = s_int / self.T

        beta_t = self.noise_schedule(t_normalized=t_float)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s_float)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_float)

        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar)

        # probX = X @ Qtb.X => paddle.matmul(X, Qtb.X)
        probX = paddle.matmul(X, Qtb.X)  # (bs, n, dx_out)
        probE = paddle.matmul(E, Qtb.E.unsqueeze(1))  # (bs, n, n, de_out)

        sampled_t = diffusion_utils.sample_discrete_features(
            probX=probX, probE=probE, node_mask=node_mask
        )

        X_t = F.one_hot(sampled_t.X, num_classes=self.Xdim_output).astype("int64")
        E_t = F.one_hot(sampled_t.E, num_classes=self.Edim_output).astype("int64")

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

    def compute_extra_data(self, noisy_data):
        #  mix extra_features with domain_features and
        # noisy_data into X/E/y final inputs. domain_features
        extra_features = self.extra_features(noisy_data)
        extra_molecular_features = self.domain_features(noisy_data)

        extra_X = self.concat_without_empty(
            [extra_features.X, extra_molecular_features.X], axis=-1
        )
        extra_E = self.concat_without_empty(
            [extra_features.E, extra_molecular_features.E], axis=-1
        )
        extra_y = self.concat_without_empty(
            [extra_features.y, extra_molecular_features.y], axis=-1
        )

        t = noisy_data["t"]
        extra_y = self.concat_without_empty([extra_y, t], axis=1)

        return utils.PlaceHolder(X=extra_X, E=extra_E, y=extra_y)

    def concat_without_empty(self, tensor_lst, axis=-1):
        new_lst = [t.astype("float") for t in tensor_lst if 0 not in t.shape]
        if new_lst == []:
            return utils.return_empty(tensor_lst[0])
        return paddle.concat(new_lst, axis=axis)

    @paddle.no_grad()
    def sample_batch(
        self,
        batch_id: int,
        batch_size: int,
        batch_condition,
        keep_chain: int,
        number_chain_steps: int,
        save_final: int,
        batch_X,
        batch_E,
        num_nodes=None,
    ):
        """
        采样: 反向扩散
        """
        if num_nodes is None:
            n_nodes = self.node_dist.sample_n(batch_size, None)  # device
        elif isinstance(num_nodes, int):
            n_nodes = paddle.full([batch_size], num_nodes, dtype="int64")
        else:
            n_nodes = num_nodes  # assume Tensor
        n_max = int(paddle.max(n_nodes).item())

        # node_mask
        arange = paddle.arange(n_max).unsqueeze(0).expand([batch_size, n_max])
        node_mask = arange < n_nodes.unsqueeze(1)

        # z_T
        z_T = diffusion_utils.sample_discrete_feature_noise(
            limit_dist=self.limit_dist, node_mask=node_mask
        )
        X, E, y = z_T.X, z_T.E, z_T.y

        chain_X = paddle.zeros(
            [number_chain_steps, keep_chain, X.shape[1]], dtype="int64"
        )
        chain_E = paddle.zeros(
            [number_chain_steps, keep_chain, E.shape[1], E.shape[2]], dtype="int64"
        )

        # 逐步还原
        for s_int in reversed(range(self.T)):
            s_array = paddle.full([batch_size, 1], float(s_int))
            t_array = s_array + 1.0
            s_norm = s_array / self.T
            t_norm = t_array / self.T

            # Sample z_s
            sampled_s, discrete_sampled_s = m_utils.sample_p_zs_given_zt(
                self,
                s=s_norm,
                t=t_norm,
                X_t=X,
                E_t=E,
                y_t=y,
                node_mask=node_mask,
                conditionVec=batch_condition,
                batch_X=batch_X,
                batch_E=batch_E,
            )
            X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

            write_index = (s_int * number_chain_steps) // self.T
            if write_index >= 0 and write_index < number_chain_steps:
                chain_X[write_index] = discrete_sampled_s.X[:keep_chain]
                chain_E[write_index] = discrete_sampled_s.E[:keep_chain]

        # 最终 mask
        sampled_s = sampled_s.mask(node_mask, collapse=True)
        X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

        # batch_X, batch_E
        batch_X = paddle.argmax(batch_X, axis=-1)
        batch_E = paddle.argmax(batch_E, axis=-1)

        # 组装 output
        molecule_list = []
        molecule_list_True = []
        n_nodes_np = n_nodes.numpy()

        for i in range(batch_size):
            n = n_nodes_np[i]
            atom_types = X[i, :n].cpu()
            edge_types = E[i, :n, :n].cpu()

            atom_types_true = batch_X[i, :n].cpu()
            edge_types_true = batch_E[i, :n, :n].cpu()

            molecule_list.append([atom_types, edge_types])
            molecule_list_True.append([atom_types_true, edge_types_true])

        # 可视化
        if self.visualization_tools is not None:
            num_molecules = chain_X.shape[1]
            for i in range(num_molecules):
                # chain_X与chain_E => numpy
                chain_X_np = chain_X[:, i, :].numpy()
                chain_E_np = chain_E[:, i, :, :].numpy()

                self.visualization_tools.visualize_chain(
                    batch_id, i, chain_X_np, chain_E_np
                )
                logger.message(f"{i+1}/{num_molecules} complete")

            self.visualization_tools.visualizeNmr(
                batch_id,
                molecule_list,
                molecule_list_True,
                save_final,
            )

        return molecule_list, molecule_list_True


if __name__ == "__main__":

    paddle.set_device("gpu")

    # setup CLIP, which contains a transformer and a vision encoder

    clip = ContrastiveModel(
        n_layers_GT=5,
        input_dims={"X": 17, "E": 5, "y": 512},
        hidden_mlp_dims={"X": 256, "E": 128, "y": 256},
        hidden_dims={
            "dx": 256,
            "de": 64,
            "dy": 256,
            "n_head": 8,
            "dim_ffX": 256,
            "dim_ffE": 128,
            "dim_ffy": 256,
        },
        output_dims={"X": 9, "E": 5, "y": 0},
        act_fn_in=paddle.nn.ReLU(),
        act_fn_out=paddle.nn.ReLU(),
        enc_voc_size=5450,
        max_len=256,
        d_model=256,
        ffn_hidden=1024,
        n_head=8,
        n_layers_TE=3,
        drop_prob=0.0,
    )
    for param in clip.graph_encoder.parameters():
        param.stop_gradient = True
    clip.graph_encoder.eval()
    for param in clip.text_encoder.parameters():
        param.stop_gradient = True
    clip.text_encoder.eval()

    prior_network = DiffusionPriorNetwork(
        dim=512,
        num_timesteps=None,
        num_time_embeds=1,
        num_graph_embeds=1,
        num_text_embeds=1,
        max_text_len=256,
        self_cond=False,
        depth=6,
        dim_head=64,
        heads=8,
    )
    ################# for precision aligement #########################
    prior_network.set_state_dict(
        paddle.load(
            "/home/liuxuwei01/PaddleScience-Material/tmp/priorNetwork_paddle.pdparams"
        )
    )
    ###################################################################

    # diffusion prior network, which contains the CLIP and network (with transformer)
    # above
    from omegaconf import OmegaConf

    config_path = "./molecule_generation/configs/diffusionPrior_CHnmr.yaml"
    config = OmegaConf.load(config_path)
    config["Model"]["condition_on_text_encodings"] = False
    config["Model"]["timesteps"] = 100
    config["Model"]["cond_drop_prob"] = 0.2
    diffusion_prior = DiffusionPriorModel(
        config=config["Model"],
        model=prior_network,
        clip=clip,
    )

    # mock data

    nmr_conditionVec = paddle.randint(0, 2, shape=[4, 256])
    nmr_srcMask = paddle.randint(0, 100000, shape=[4, 1, 1, 256], dtype="int32").astype(
        "bool"
    )
    graph_X = paddle.randn([4, 15, 9])
    graph_E = paddle.randn([4, 15, 15, 5])
    graph_y = paddle.randn([4, 1024])
    graph_node_mask = paddle.randint(0, 2, shape=[4, 15], dtype="int32").astype("bool")

    clip_text_embeds = diffusion_prior.clip.text_encoder(nmr_conditionVec, nmr_srcMask)
    clip_text_embeds = clip_text_embeds.reshape([clip_text_embeds.shape[0], -1])
    clip_text_embeds = clip.linear_layer(clip_text_embeds)
    clip_graph_embeds = diffusion_prior.clip.graph_encoder(
        graph_X, graph_E, graph_y, graph_node_mask
    )

    # feed text and images into diffusion prior network
    ################# for precision aligement #########################
    import numpy as np

    clip_text_embeds = paddle.to_tensor(
        np.load("/home/liuxuwei01/PaddleScience-Material/tmp/clip_text_embeds.npy")
    )
    clip_graph_embeds = paddle.to_tensor(
        np.load("/home/liuxuwei01/PaddleScience-Material/tmp/clip_graph_embeds.npy")
    )
    ###################################################################

    loss = diffusion_prior(
        text_embed=clip_text_embeds, moleculargraph_embed=clip_graph_embeds
    )

    print(clip_graph_embeds)
    print(clip_text_embeds)
    print(loss)
    loss.backward()

    result_sample = diffusion_prior.sample(nmr_conditionVec, nmr_srcMask)
    print(result_sample)
