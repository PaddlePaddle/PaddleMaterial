import copy
import os

import paddle
import paddle.nn as nn
import rdkit

# from paddle.nn import TransformerEncoderLayer
from paddle.nn import TransformerEncoder as Encoder
from paddle.nn import functional as F
from rdkit import Chem

from ppmat.metrics.abstract_metrics import NLL
from ppmat.metrics.abstract_metrics import SumExceptBatchKL
from ppmat.metrics.abstract_metrics import SumExceptBatchMetric
from ppmat.metrics.train_metrics import TrainLossDiscrete

from . import diffusion_utils
from .graph_transformer import GraphTransformer
from .graph_transformer import GraphTransformer_C
from .noise_schedule import DiscreteUniformTransition
from .noise_schedule import MarginalUniformTransition
from .noise_schedule import PredefinedNoiseScheduleDiscrete
from .utils import digressutils as utils


class MolecularGraphTransformer(paddle.nn.Layer):
    def __init__(
        self,
        cfg,
        dataset_infos,
        train_metrics,
        sampling_metrics,
        visualization_tools,
        extra_features,
        domain_features,
    ) -> None:
        super().__init__()

        #############################################################
        # # for testing

        # input_dims = {"X": 17, "E": 5, "y": 525}  # dataset_infos.input_dims
        # output_dims = {"X": 9, "E": 5, "y": 0}  # dataset_infos.output_dims
        # self.encoder = GraphTransformer_C(
        #     n_layers=cfg["encoder"]["num_layers"],
        #     input_dims=input_dims,
        #     hidden_mlp_dims=cfg["encoder"]["hidden_mlp_dims"],
        #     hidden_dims=cfg["encoder"]["hidden_dims"],
        #     output_dims=output_dims,
        #     act_fn_in=nn.ReLU(),
        #     act_fn_out=nn.ReLU(),
        #  )
        #
        # con_input_dim = input_dims
        # con_input_dim["X"] = input_dims["X"] - 8
        # con_input_dim["y"] = 1024
        # con_output_dim = output_dims
        # self.decoder = GraphTransformer(
        #     n_layers=cfg["decoder"]["num_layers"],
        #     input_dims=con_input_dim,
        #     hidden_mlp_dims=cfg["decoder"]["hidden_mlp_dims"],
        #     hidden_dims=cfg["decoder"]["hidden_dims"],
        #     output_dims=con_output_dim,
        #     act_fn_in=nn.ReLU(),
        #     act_fn_out=nn.ReLU(),
        # )
        #############################################################

        #############################################################
        # configure general variables settings
        #############################################################
        # self.cfg = cfg
        self.name = cfg["__name__"]
        self.model_dtype = paddle.get_default_dtype()
        self.T = cfg["diffusion_model"]["diffusion_steps"]
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
        # configure model
        #############################################################
        input_dims = dataset_infos.input_dims
        self.con_input_dim = copy.deepcopy(input_dims)
        self.con_input_dim["X"] = input_dims["X"] - 8
        self.con_input_dim["y"] = 1024
        self.con_output_dim = dataset_infos.output_dims

        self.encoder = GraphTransformer_C(
            n_layers=cfg["encoder"]["num_layers"],
            input_dims=self.con_input_dim,
            hidden_mlp_dims=cfg["encoder"]["hidden_mlp_dims"],
            hidden_dims=cfg["encoder"]["hidden_dims"],
            output_dims=self.con_output_dim,
            act_fn_in=nn.ReLU(),
            act_fn_out=nn.ReLU(),
        )

        self.decoder = GraphTransformer(
            n_layers=cfg["decoder"]["num_layers"],
            input_dims=input_dims,
            hidden_mlp_dims=cfg["decoder"]["hidden_mlp_dims"],
            hidden_dims=cfg["decoder"]["hidden_dims"],
            output_dims=output_dims,
            act_fn_in=nn.ReLU(),
            act_fn_out=nn.ReLU(),
        )

        #############################################################
        # configure noise scheduler
        #############################################################
        self.noise_scheduler = PredefinedNoiseScheduleDiscrete(
            cfg["diffusion_model"]["diffusion_noise_schedule"],
            timesteps=self.T,
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
        if cfg["diffusion_model"]["transition"] == "uniform":
            self.transition_model = DiscreteUniformTransition(
                x_classes=self.Xdim_output,
                e_classes=self.Edim_output,
                y_classes=self.ydim_output,
            )
            x_limit = paddle.ones([self.Xdim_output]) / self.Xdim_output
            e_limit = paddle.ones([self.Edim_output]) / self.Edim_output
            y_limit = paddle.ones([self.ydim_output]) / self.ydim_output
            self.limit_dist = utils.PlaceHolder(X=x_limit, E=e_limit, y=y_limit)

        elif cfg["diffusion_model"]["transition"] == "marginal":
            node_types = self.dataset_info.node_types.astype(self.model_dtype)
            x_marginals = node_types / paddle.sum(node_types)

            edge_types = self.dataset_info.edge_types.astype(self.model_dtype)
            e_marginals = edge_types / paddle.sum(edge_types)
            print(
                f"Marginal distribution of classes: {x_marginals} for nodes, "
                f"{e_marginals} for edges"
            )

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
        self.train_loss = TrainLossDiscrete(cfg["diffusion_model"]["lambda_train"])

        #############################################################
        # configure training setting and other properties
        #############################################################
        self.start_epoch_time = None
        self.best_val_nll = 1e8
        self.val_counter = 0
        self.vocabDim = 256
        self.number_chain_steps = cfg["diffusion_model"]["number_chain_steps"]
        # self.log_every_steps = self.cfg["Global"]["log_every_steps"]

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

    def forward(self, batch):
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
        input_X = paddle.concat([noisy_data["X_t"], extra_data.X], axis=2).astype(
            "float32"
        )

        # input_E
        input_E = paddle.concat([noisy_data["E_t"], extra_data.E], axis=3).astype(
            "float32"
        )

        # input_y with encoder output as condition vector of input of decoder
        input_y = paddle.concat([noisy_data["y_t"], extra_data.y], axis=1).astype(
            "float32"
        )
        y_condition = paddle.zeros(shape=[X.shape[0], 1024]).cuda(blocking=True)
        conditionVec = self.encoder(X, E, y_condition, node_mask)
        input_y = paddle.hstack(x=(input_y, conditionVec)).astype(dtype="float32")

        # forward of decoder with encoder output as condition vector of input of decoder
        pred = self.decoder(input_X, input_E, input_y, node_mask)

        # compute loss
        # TODO: move loss out!
        loss = self.train_loss(
            masked_pred_X=pred.X,
            masked_pred_E=pred.E,
            pred_y=pred.y,
            true_X=X,
            true_E=E,
            true_y=other_data["y"],
            log=False,
        )

        # log metrics to do move to another location
        # if self.log_count % 80 == 0:
        #     print(f"train_loss: {loss}")
        self.train_metrics(
            masked_pred_X=pred.X,
            masked_pred_E=pred.E,
            true_X=X,
            true_E=E,
            log=False,
        )
        return {"loss": loss}

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

        beta_t = self.noise_scheduler(t_normalized=t_float)
        alpha_s_bar = self.noise_scheduler.get_alpha_bar(t_normalized=s_float)
        alpha_t_bar = self.noise_scheduler.get_alpha_bar(t_normalized=t_float)

        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, device=None)

        # probX = X @ Qtb.X => paddle.matmul(X, Qtb.X)
        probX = paddle.matmul(X, Qtb.X)  # (bs, n, dx_out)
        probE = paddle.matmul(E, Qtb.E.unsqueeze(1))  # (bs, n, n, de_out)

        sampled_t = diffusion_utils.sample_discrete_features(
            probX=probX, probE=probE, node_mask=node_mask
        )

        X_t = F.one_hot(sampled_t.X, num_classes=self.Xdim_output)
        E_t = F.one_hot(sampled_t.E, num_classes=self.Edim_output)

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
        new_lst = [t for t in tensor_lst if 0 not in t.shape]
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

        pred = self.forward(noisy_data, extra_data, node_mask, X, E)

        # input_X
        input_X = paddle.concat([noisy_data["X_t"], extra_data.X], axis=2).astype(
            "float32"
        )

        # input_E
        input_E = paddle.concat([noisy_data["E_t"], extra_data.E], axis=3).astype(
            "float32"
        )

        # input_y
        input_y = paddle.concat([noisy_data["y_t"], extra_data.y], axis=1).astype(
            "float32"
        )
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
            log=i % self.log_every_steps == 0,
        )
        if i % 10 == 0:
            print(f"val_loss:{loss}")

        batch_length = dense_data.num_graphs
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

        # save the data for visualization
        self.val_y_collection.append(other_data["conditionVec"])
        self.val_atomCount.append(other_data["atom_count"])
        self.val_data_X.append(X)
        self.val_data_E.append(E)

        return {"loss": nll}

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
        alpha_t_bar = self.noise_scheduler.get_alpha_bar(t_int=Ts)  # (bs,1)

        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, None)
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
            x=paddle.log(probX + 1e-10), target=limit_dist_X, reduction="none"
        )
        kl_distance_E = F.kl_div(
            x=paddle.log(probE + 1e-10), target=limit_dist_E, reduction="none"
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

        Qtb = self.transition_model.get_Qt_bar(noisy_data["alpha_t_bar"], None)
        Qsb = self.transition_model.get_Qt_bar(noisy_data["alpha_s_bar"], None)
        Qt = self.transition_model.get_Qt(noisy_data["beta_t"], None)

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
        beta_0 = self.noise_scheduler(t_zeros)
        Q0 = self.transition_model.get_Qt(beta_t=beta_0, device=None)

        probX0 = paddle.matmul(X, Q0.X)
        # E => broadcast
        probE0 = paddle.matmul(E, Q0.E.unsqueeze(1))

        sampled0 = diffusion_utils.sample_discrete_features(probX0, probE0, node_mask)
        X0 = F.one_hot(sampled0.X, num_classes=self.Xdim_output).astype("float32")
        E0 = F.one_hot(sampled0.E, num_classes=self.Edim_output).astype("float32")
        y0 = sampled0.y  # 这里是空?

        # noisy_data
        noisy_data = {
            "X_t": X0,
            "E_t": E0,
            "y_t": y0,
            "node_mask": node_mask,
            "t": paddle.zeros([X0.shape[0], 1]).astype("float32"),
        }
        extra_data = self.compute_extra_data(noisy_data)
        pred0 = self.forward(noisy_data, extra_data, node_mask, X, E)

        probX0 = F.softmax(pred0.X, axis=-1)
        probE0 = F.softmax(pred0.E, axis=-1)
        proby0 = F.softmax(pred0.y, axis=-1)

        # mask
        probX0[~node_mask] = 1.0 / probX0.shape[-1]
        # E -> (bs, n, n, de_out)
        # 屏蔽 ~mask
        expand_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
        probE0[~expand_mask] = 1.0 / probE0.shape[-1]

        diag_mask = paddle.eye(probE0.shape[1], dtype="bool")
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
            current_path = os.getcwd()
            num_molecules = chain_X.shape[1]
            for i in range(num_molecules):
                result_path = os.path.join(
                    current_path,
                    f"chains/{self.cfg.general.name}",
                    f"epochXX/chains/molecule_{batch_id + i}",
                )
                os.makedirs(result_path, exist_ok=True)
                # chain_X与chain_E => numpy
                chain_X_np = chain_X[:, i, :].numpy()
                chain_E_np = chain_E[:, i, :, :].numpy()

                self.visualization_tools.visualize_chain(
                    result_path, chain_X_np, chain_E_np
                )
                print(f"\r {i+1}/{num_molecules} complete", end="", flush=True)
            print("\n")

            # graph
            result_path = os.path.join(
                current_path, f"graphs/{self.name}/epochXX_b{batch_id}/"
            )
            result_path_true = os.path.join(
                current_path, f"graphs/{self.name}/True_epochXX_b{batch_id}/"
            )
            self.visualization_tools.visualizeNmr(
                result_path,
                result_path_true,
                molecule_list,
                molecule_list_True,
                save_final,
            )

        return molecule_list, molecule_list_True

    @paddle.no_grad()
    def forward_sample(self, noisy_data, extra_data, node_mask, batch_X, batch_E):
        """
        用于 sampling 时的推断：同上，但不记录梯度
        """
        X = paddle.concat([noisy_data["X_t"], extra_data.X], axis=2).astype("float32")
        E = paddle.concat([noisy_data["E_t"], extra_data.E], axis=3).astype("float32")
        y = paddle.concat([noisy_data["y_t"], extra_data.y], axis=1).astype("float32")
        return self.model(X, E, y, node_mask, batch_X, batch_E)

    @paddle.no_grad()
    def sample_p_zs_given_zt(
        self, s, t, X_t, E_t, y_t, node_mask, conditionVec, batch_X, batch_E
    ):
        """
        从 p(z_s | z_t) 采样: 反向扩散一步
        """
        beta_t = self.r(t_normalized=t)
        alpha_s_bar = self.noise_scheduler.get_alpha_bar(t_normalized=s)
        alpha_t_bar = self.noise_scheduler.get_alpha_bar(t_normalized=t)

        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, None)
        Qsb = self.transition_model.get_Qt_bar(alpha_s_bar, None)
        Qt = self.transition_model.get_Qt(beta_t, None)

        # forward
        noisy_data = {
            "X_t": X_t,
            "E_t": E_t,
            "y_t": y_t,
            "t": t,
            "node_mask": node_mask,
        }
        extra_data = self.compute_extra_data(noisy_data)
        pred = self.forward_sample(noisy_data, extra_data, node_mask, batch_X, batch_E)

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
        X_s = F.one_hot(sampled_s.X, num_classes=self.Xdim_output).astype("float32")
        E_s = F.one_hot(sampled_s.E, num_classes=self.Edim_output).astype("float32")

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


class ContrastGraphTransformer(paddle.nn.Layer):
    def __init__(
        self,
        n_layers_GT: int,
        input_dims: dict,
        hidden_mlp_dims: dict,
        hidden_dims: dict,
        output_dims: dict,
        act_fn_in: paddle.nn.ReLU(),
        act_fn_out: paddle.nn.ReLU(),
        enc_voc_size,
        max_len,
        d_model,
        ffn_hidden,
        n_head,
        n_layers_TE,
        drop_prob,
        device,
    ):
        super().__init__()
        self.transEn = Encoder(
            enc_voc_size=enc_voc_size,
            max_len=max_len,
            d_model=d_model,
            ffn_hidden=ffn_hidden,
            n_head=n_head,
            n_layers=n_layers_TE,
            drop_prob=drop_prob,
            device=device,
        )
        self.linear_layer = paddle.nn.Linear(
            in_features=max_len * d_model, out_features=512
        )
        self.con_input_dim = input_dims
        self.con_input_dim["X"] = input_dims["X"] - 8
        self.con_input_dim["y"] = 1024
        self.con_output_dim = output_dims
        self.conditionEn = GraphTransformer_C(
            n_layers=n_layers_GT,
            input_dims=self.con_input_dim,
            hidden_mlp_dims=hidden_mlp_dims,
            hidden_dims=hidden_dims,
            output_dims=self.con_output_dim,
            act_fn_in=act_fn_in,
            act_fn_out=act_fn_out,
        )
        self.device = device
        checkpoint = paddle.load(
            path=str("/home/liuxuwei01/molecular2molecular/src/epoch-438.ckpt")
        )
        state_dict = checkpoint["state_dict"]
        print(state_dict.keys())
        conditionEn_state_dict = {
            k[len("model.conditionEn.") :]: v
            for k, v in state_dict.items()
            if k.startswith("model.conditionEn.")
        }
        self.conditionEn.set_state_dict(state_dict=conditionEn_state_dict)
        print("conditionEn parameters loaded successfully.")
        for param in self.conditionEn.parameters():
            param.stop_gradient = not False
        self.conditionEn.eval()

    def make_src_mask(self, src):
        src_mask = (src != 0).unsqueeze(axis=1).unsqueeze(axis=2)
        return src_mask

    def forward(self, X, E, y, node_mask, X_condition, E_condtion, conditionVec):
        assert isinstance(
            conditionVec, paddle.Tensor
        ), "conditionVec should be a tensor, but got type {}".format(type(conditionVec))
        srcMask = self.make_src_mask(conditionVec).to(self.device)
        conditionVecNmr = self.transEn(conditionVec, srcMask)
        conditionVecNmr = conditionVecNmr.view(conditionVecNmr.shape[0], -1)
        conditionVecNmr = self.linear_layer(conditionVecNmr)
        y_condition = paddle.zeros(shape=[X_condition.shape[0], 1024]).cuda(
            blocking=True
        )
        conditionVecM = self.conditionEn(
            X_condition, E_condtion, y_condition, node_mask
        )
        return conditionVecM, conditionVecNmr


class ConditionGraphTransformer(nn.Layer):
    def __init__(
        self,
        n_layers_GT: int,
        input_dims: dict,
        hidden_mlp_dims: dict,
        hidden_dims: dict,
        output_dims: dict,
        act_fn_in: paddle.nn.ReLU(),
        act_fn_out: paddle.nn.ReLU(),
        enc_voc_size,
        max_len,
        d_model,
        ffn_hidden,
        n_head,
        n_layers_TE,
        drop_prob,
        device,
    ):
        super().__init__()
        self.GT = GraphTransformer(
            n_layers=n_layers_GT,
            input_dims=input_dims,
            hidden_mlp_dims=hidden_mlp_dims,
            hidden_dims=hidden_dims,
            output_dims=output_dims,
            act_fn_in=act_fn_in,
            act_fn_out=act_fn_out,
        )
        self.transEn = Encoder(
            enc_voc_size=enc_voc_size,
            max_len=max_len,
            d_model=d_model,
            ffn_hidden=ffn_hidden,
            n_head=n_head,
            n_layers=n_layers_TE,
            drop_prob=drop_prob,
            device=device,
        )
        self.linear_layer = paddle.nn.Linear(
            in_features=max_len * d_model, out_features=512
        )
        self.device = device

        checkpoint = paddle.load(
            "/home/liuxuwei01/molecular2molecular/src/epoch-438.ckpt"
        )
        state_dict = checkpoint["state_dict"]
        GT_state_dict = {
            k[len("model.GT.") :]: v
            for k, v in state_dict.items()
            if k.startswith("model.GT.")
        }
        self.GT.set_state_dict(GT_state_dict)

        checkpoint = paddle.load(
            "/home/liuxuwei01/molecular2molecular/src/epoch-35.ckpt"
        )
        state_dict = checkpoint["state_dict"]
        linear_layer_state_dict = {
            k[len("model.linear_layer.") :]: v
            for k, v in state_dict.items()
            if k.startswith("model.linear_layer.")
        }
        self.linear_layer.set_state_dict(linear_layer_state_dict)

        checkpoint = paddle.load(
            "/home/liuxuwei01/molecular2molecular/src/epoch-35.ckpt"
        )
        state_dict = checkpoint["state_dict"]
        transEn_state_dict = {
            k[len("model.transEn.") :]: v
            for k, v in state_dict.items()
            if k.startswith("model.transEn.")
        }
        self.transEn.set_state_dict(transEn_state_dict)

    def make_src_mask(self, src):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        return src_mask

    def forward(self, X, E, y, node_mask, conditionVec):
        assert isinstance(
            conditionVec, paddle.Tensor
        ), "conditionVec should be a tensor, but got type {}".format(type(conditionVec))

        srcMask = self.make_src_mask(conditionVec).astype("float32")
        conditionVec = self.transEn(conditionVec, srcMask)
        conditionVec = conditionVec.reshape([conditionVec.shape[0], -1])
        conditionVec = self.linear_layer(conditionVec)

        y = paddle.concat([y, conditionVec], axis=1).astype("float32")

        return self.GT(X, E, y, node_mask)


if __name__ == "__main__":
    import argparse

    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="./molecule_generation/configs/digress_CHnmr.yaml",
        help="Path to config file",
    )
    args, dynamic_args = parser.parse_known_args()
    config = OmegaConf.load(args.config)

    model = MolecularGraphTransformer(config["Model"])
