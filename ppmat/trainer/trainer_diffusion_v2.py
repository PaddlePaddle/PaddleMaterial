import os
import sys
import time
import random
import rdkit
from rdkit import Chem
from rdkit.Chem import Draw, DataStructs, RDKFingerprint

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

import numpy as np

from ppmat.models.digress.molecularGT import molecularGT
from ppmat.models.digress.contrastGT import molecularGT
from ppmat.models.digress.conditionGT import ConditionGT

from ppmat.models.digress.noise_schedule import DiscreteUniformTransition, PredefinedNoiseScheduleDiscrete, MarginalUniformTransition
from ppmat.models.digress import diffusion_utils
from ppmat.metrics.TrainLossDiscrete import TrainLossDiscrete
from ppmat.metrics.abstract_metrics import SumExceptBatchMetric, SumExceptBatchKL, NLL
from ppmat.utils import digressutils as utils

class DiscreteDenoisingDiffusionMolecular_condition(nn.Layer):
    def __init__(self, cfg, dataset_infos, train_metrics, sampling_metrics, visualization_tools, extra_features, domain_features):
        super().__init__()
        # ---------------------
        # 1. 初始化配置信息
        # ---------------------
        input_dims = dataset_infos.input_dims
        output_dims = dataset_infos.output_dims
        nodes_dist = dataset_infos.nodes_dist

        self.cfg = cfg
        self.name = cfg.general.name
        self.model_dtype = 'float32'
        self.T = cfg.model.diffusion_steps

        self.Xdim = input_dims['X']
        self.Edim = input_dims['E']
        self.ydim = input_dims['y']
        self.Xdim_output = output_dims['X']
        self.Edim_output = output_dims['E']
        self.ydim_output = output_dims['y']
        self.node_dist = nodes_dist

        self.dataset_info = dataset_infos

        # 训练损失 & 验证 / 测试指标
        self.train_loss = TrainLossDiscrete(self.cfg.model.lambda_train)
        self.val_nll = NLL()
        self.val_X_kl = SumExceptBatchKL()
        self.val_E_kl = SumExceptBatchKL()
        self.val_X_logp = SumExceptBatchMetric()
        self.val_E_logp = SumExceptBatchMetric()

        self.test_nll = NLL()
        self.test_X_kl = SumExceptBatchKL()
        self.test_E_kl = SumExceptBatchKL()
        self.test_X_logp = SumExceptBatchMetric()
        self.test_E_logp = SumExceptBatchMetric()

        # 一些容器
        self.val_y_collection = []
        self.val_atomCount = []
        self.val_data_X = []
        self.val_data_E = []
        self.test_y_collection = []
        self.test_atomCount = []
        self.test_data_X = []
        self.test_data_E = []

        # 训练 / 采样指标
        self.train_metrics = train_metrics
        self.sampling_metrics = sampling_metrics

        # 可视化、额外特征
        self.visualization_tools = visualization_tools
        self.extra_features = extra_features
        self.domain_features = domain_features

        # 2. 构造网络
        self.model = molecularGT(
            n_layers_GT=cfg.model.n_layers,
            input_dims=input_dims,
            hidden_mlp_dims=cfg.model.hidden_mlp_dims,
            hidden_dims=cfg.model.hidden_dims,
            output_dims=output_dims,
            act_fn_in=nn.ReLU(),
            act_fn_out=nn.ReLU()
        )

        # 3. 噪声日程表
        self.noise_schedule = PredefinedNoiseScheduleDiscrete(
            cfg.model.diffusion_noise_schedule, timesteps=cfg.model.diffusion_steps
        )

        # 4. Transition Model
        if cfg.model.transition == 'uniform':
            self.transition_model = DiscreteUniformTransition(
                x_classes=self.Xdim_output,
                e_classes=self.Edim_output,
                y_classes=self.ydim_output
            )
            x_limit = paddle.ones([self.Xdim_output]) / self.Xdim_output
            e_limit = paddle.ones([self.Edim_output]) / self.Edim_output
            y_limit = paddle.ones([self.ydim_output]) / self.ydim_output
            self.limit_dist = utils.PlaceHolder(X=x_limit, E=e_limit, y=y_limit)

        elif cfg.model.transition == 'marginal':
            node_types = self.dataset_info.node_types.astype('float32')
            x_marginals = node_types / paddle.sum(node_types)

            edge_types = self.dataset_info.edge_types.astype('float32')
            e_marginals = edge_types / paddle.sum(edge_types)
            print(f"Marginal distribution of classes: {x_marginals} for nodes, {e_marginals} for edges")

            self.transition_model = MarginalUniformTransition(
                x_marginals=x_marginals, e_marginals=e_marginals, y_classes=self.ydim_output
            )
            self.limit_dist = utils.PlaceHolder(
                X=x_marginals, E=e_marginals,
                y=paddle.ones([self.ydim_output]) / self.ydim_output
            )

        # 其余属性
        self.start_epoch_time = None
        self.best_val_nll = 1e8
        self.val_counter = 0
        self.vocabDim = 256
        self.number_chain_steps = cfg.general.number_chain_steps
        self.log_every_steps = cfg.general.log_every_steps

    def train_step(self, data, i: int):
        """
        训练阶段: 前向 + 计算损失 + 更新指标
        data: 一批图数据 (已 batch 化)
        i: 当前 iteration
        return: Paddle的loss张量
        """
        # 转 dense & mask
        if data.edge_index.numel() == 0:
            print("Found a batch with no edges. Skipping.")
            return None

        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        dense_data = dense_data.mask(node_mask)
        X, E = dense_data.X, dense_data.E

        # 给 (X, E) 加噪
        noisy_data = self.apply_noise(X, E, data.y, node_mask)
        extra_data = self.compute_extra_data(noisy_data)

        # 前向
        pred = self.forward(noisy_data, extra_data, node_mask, X, E)
        loss = self.train_loss(
            masked_pred_X=pred.X, masked_pred_E=pred.E, pred_y=pred.y,
            true_X=X, true_E=E, true_y=data.y,
            log=(i % self.log_every_steps == 0)
        )
        # 记录
        if i % 80 == 0:
            print(f"train_loss: {loss}")

        self.train_metrics(
            masked_pred_X=pred.X, masked_pred_E=pred.E,
            true_X=X, true_E=E,
            log=(i % self.log_every_steps == 0)
        )
        return loss

    def val_step(self, data, i: int):
        """
        验证阶段: 计算验证损失 + KL + 记录必要信息
        """
        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        dense_data = dense_data.mask(node_mask)
        X, E = dense_data.X, dense_data.E

        # 加噪
        noisy_data = self.apply_noise(dense_data.X, dense_data.E, data.y, node_mask)
        extra_data = self.compute_extra_data(noisy_data)

        # 条件信息
        batch_length = data.num_graphs
        conditionAll = data.conditionVec
        conditionAll = paddle.reshape(conditionAll, [batch_length, self.vocabDim])

        # 前向
        pred = self.forward(noisy_data, extra_data, node_mask, X, E)
        # 记录
        self.val_y_collection.append(data.conditionVec)
        self.val_atomCount.append(data.atom_count)
        self.val_data_X.append(X)
        self.val_data_E.append(E)

        # 计算 train_loss 中的NLL
        loss = self.train_loss(
            masked_pred_X=pred.X, masked_pred_E=pred.E, pred_y=pred.y,
            true_X=X, true_E=E, true_y=data.y,
            log=(i % self.log_every_steps == 0)
        )
        if i % 10 == 0:
            print(f"val_loss: {loss}")

        # compute_val_loss
        nll = self.compute_val_loss(
            pred=pred, noisy_data=noisy_data,
            X=X, E=E, y=data.y,
            node_mask=node_mask, condition=conditionAll,
            test=False
        )
        return nll

    def test_step(self, data, i: int):
        """
        测试阶段 step
        """
        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        dense_data = dense_data.mask(node_mask)
        X, E = dense_data.X, dense_data.E

        # 加噪
        noisy_data = self.apply_noise(X, E, data.y, node_mask)
        extra_data = self.compute_extra_data(noisy_data)

        batch_length = data.num_graphs
        conditionAll = data.conditionVec
        conditionAll = paddle.reshape(conditionAll, [batch_length, self.vocabDim])

        # forward
        pred = self.forward(noisy_data, extra_data, node_mask, X, E)
        self.test_y_collection.append(data.conditionVec)
        self.test_atomCount.append(data.atom_count)
        self.test_data_X.append(X)
        self.test_data_E.append(E)

        # 计算NLL
        nll = self.compute_val_loss(
            pred=pred, noisy_data=noisy_data,
            X=X, E=E, y=data.y,
            node_mask=node_mask, condition=conditionAll,
            test=True
        )
        return nll

    # ----------------------
    #  前向 & 模型推断相关
    # ----------------------
    def forward(self, noisy_data, extra_data, node_mask, X, E):
        """
        替代LightningModule的 forward:
        将 (noisy_data, extra_data) 拼接为网络输入, 调用 self.model
        """
        # 将去噪输入拼到一起
        X_ = paddle.concat([noisy_data['X_t'], extra_data.X], axis=2)  # (bs, n, dX + extra)
        E_ = paddle.concat([noisy_data['E_t'], extra_data.E], axis=3)  # (bs, n, n, dE + extra)
        y_ = paddle.concat([noisy_data['y_t'], extra_data.y], axis=1)  # (bs, dy + extra)

        return self.model(X_, E_, y_, node_mask, X, E)

    @paddle.no_grad()
    def forward_sample(self, noisy_data, extra_data, node_mask, batch_X, batch_E):
        """
        用于 sampling 时的推断：同上，但不记录梯度
        """
        X = paddle.concat([noisy_data['X_t'], extra_data.X], axis=2).astype('float32')
        E = paddle.concat([noisy_data['E_t'], extra_data.E], axis=3).astype('float32')
        y = paddle.concat([noisy_data['y_t'], extra_data.y], axis=1).astype('float32')
        return self.model(X, E, y, node_mask, batch_X, batch_E)

    # ----------------------
    #  Noise / Posterior 相关
    # ----------------------
    def apply_noise(self, X, E, y, node_mask):
        """
        1. 在 [1, T] 范围内随机抽样 t
        2. 将 X,E 按照 alpha_t_bar 变为 X_t, E_t
        """
        lowest_t = 1
        t_int = paddle.randint(lowest_t, self.T + 1, shape=[X.shape[0], 1], dtype='int64').astype('float32')  # (bs,1)
        s_int = t_int - 1

        t_float = t_int / self.T
        s_float = s_int / self.T

        # 取得 beta_t, alpha_s_bar, alpha_t_bar
        beta_t = self.noise_schedule(t_normalized=t_float)  # (bs,1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s_float)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_float)

        # 计算 Q_t^bar
        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, device=None)  # 不显式 device, Paddle 中 tensor 会自动管理
        probX = paddle.matmul(X, Qtb.X)  # (bs,n,dx_out)
        # 对 E 做 broadcast
        probE = paddle.matmul(E, Qtb.E.unsqueeze(1))  # (bs,n,n,de_out)

        # Sample x_t, e_t
        sampled_t = diffusion_utils.sample_discrete_features(probX=probX, probE=probE, node_mask=node_mask)
        X_t = F.one_hot(sampled_t.X, num_classes=self.Xdim_output)
        E_t = F.one_hot(sampled_t.E, num_classes=self.Edim_output)

        z_t = utils.PlaceHolder(X=X_t, E=E_t, y=y).astype('float32').mask(node_mask)

        noisy_data = {
            't_int': t_int, 't': t_float,
            'beta_t': beta_t,
            'alpha_s_bar': alpha_s_bar,
            'alpha_t_bar': alpha_t_bar,
            'X_t': z_t.X, 'E_t': z_t.E, 'y_t': z_t.y,
            'node_mask': node_mask
        }
        return noisy_data

    def compute_val_loss(self, pred, noisy_data, X, E, y, node_mask, condition, test=False):
        """
        计算 validation/test 阶段的 NLL (variational lower bound 估计)
        """
        t = noisy_data['t']

        # 1. log p(N) = number of nodes 先验
        N = paddle.sum(node_mask, axis=1).astype('int64')
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
        nlls = - log_pN + kl_prior + loss_all_t - loss_term_0
        # shape: (bs, ), 对batch做均值
        nll = (self.test_nll if test else self.val_nll)(nlls)

        return nll

    def kl_prior(self, X, E, node_mask):
        """
        KL between q(zT|x) and prior p(zT)=Uniform(...) 
        """
        bs = X.shape[0]
        ones = paddle.ones([bs, 1], dtype='float32')
        Ts = self.T * ones
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_int=Ts)  # (bs,1)

        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, None)
        probX = paddle.matmul(X, Qtb.X)   # (bs,n,dx_out)
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
            node_mask=node_mask
        )

        kl_distance_X = F.kl_div(
            x=paddle.log(probX + 1e-10),
            target=limit_dist_X,
            reduction='none'
        )
        kl_distance_E = F.kl_div(
            x=paddle.log(probE + 1e-10),
            target=limit_dist_E,
            reduction='none'
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

        Qtb = self.transition_model.get_Qt_bar(noisy_data['alpha_t_bar'], None)
        Qsb = self.transition_model.get_Qt_bar(noisy_data['alpha_s_bar'], None)
        Qt  = self.transition_model.get_Qt(noisy_data['beta_t'], None)

        bs, n, _ = X.shape
        # 计算真实后验分布
        prob_true = diffusion_utils.posterior_distributions(
            X=X, E=E, y=y,
            X_t=noisy_data['X_t'], E_t=noisy_data['E_t'], y_t=noisy_data['y_t'],
            Qt=Qt, Qsb=Qsb, Qtb=Qtb
        )
        prob_true.E = paddle.reshape(prob_true.E, [bs, n, n, -1])

        # 计算预测后验分布
        prob_pred = diffusion_utils.posterior_distributions(
            X=pred_probs_X, E=pred_probs_E, y=pred_probs_y,
            X_t=noisy_data['X_t'], E_t=noisy_data['E_t'], y_t=noisy_data['y_t'],
            Qt=Qt, Qsb=Qsb, Qtb=Qtb
        )
        prob_pred.E = paddle.reshape(prob_pred.E, [bs, n, n, -1])

        # mask
        prob_true_X, prob_true_E, prob_pred.X, prob_pred.E = diffusion_utils.mask_distributions(
            true_X=prob_true.X, true_E=prob_true.E,
            pred_X=prob_pred.X, pred_E=prob_pred.E,
            node_mask=node_mask
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
        Q0 = self.transition_model.get_Qt(beta_t=beta_0, device=None)

        probX0 = paddle.matmul(X, Q0.X)
        # E => broadcast
        probE0 = paddle.matmul(E, Q0.E.unsqueeze(1))

        sampled0 = diffusion_utils.sample_discrete_features(probX0, probE0, node_mask)
        X0 = F.one_hot(sampled0.X, num_classes=self.Xdim_output).astype('float32')
        E0 = F.one_hot(sampled0.E, num_classes=self.Edim_output).astype('float32')
        y0 = sampled0.y  # 这里是空?

        # noisy_data
        noisy_data = {
            'X_t': X0, 'E_t': E0, 'y_t': y0, 'node_mask': node_mask,
            't': paddle.zeros([X0.shape[0], 1]).astype('float32')
        }
        extra_data = self.compute_extra_data(noisy_data)
        pred0 = self.forward(noisy_data, extra_data, node_mask, X, E)

        probX0 = F.softmax(pred0.X, axis=-1)
        probE0 = F.softmax(pred0.E, axis=-1)
        proby0 = F.softmax(pred0.y, axis=-1)

        # mask
        probX0[~node_mask] = 1. / probX0.shape[-1]
        # E -> (bs, n, n, de_out)
        # 屏蔽 ~mask
        expand_mask = (node_mask.unsqueeze(1) * node_mask.unsqueeze(2))
        probE0[~expand_mask] = 1. / probE0.shape[-1]

        diag_mask = paddle.eye(probE0.shape[1], dtype='bool')
        diag_mask = diag_mask.unsqueeze(0).expand([probE0.shape[0], -1, -1])
        probE0[diag_mask] = 1. / probE0.shape[-1]

        # 返回概率
        return utils.PlaceHolder(X=probX0, E=probE0, y=proby0)

    # -----------------------
    # Sampling 相关
    # -----------------------
    @paddle.no_grad()
    def sample_batch(self, batch_id: int, batch_size: int, batch_condition, keep_chain: int,
                     number_chain_steps: int, save_final: int, batch_X, batch_E, num_nodes=None):
        """
        采样: 反向扩散
        """
        if num_nodes is None:
            n_nodes = self.node_dist.sample_n(batch_size, None)  # device
        elif isinstance(num_nodes, int):
            n_nodes = paddle.full([batch_size], num_nodes, dtype='int64')
        else:
            n_nodes = num_nodes  # assume Tensor
        n_max = int(paddle.max(n_nodes).item())

        # node_mask
        arange = paddle.arange(n_max).unsqueeze(0).expand([batch_size, n_max])
        node_mask = arange < n_nodes.unsqueeze(1)

        # z_T
        z_T = diffusion_utils.sample_discrete_feature_noise(
            limit_dist=self.limit_dist,
            node_mask=node_mask
        )
        X, E, y = z_T.X, z_T.E, z_T.y

        chain_X = paddle.zeros([number_chain_steps, keep_chain, X.shape[1]], dtype='int64')
        chain_E = paddle.zeros([number_chain_steps, keep_chain, E.shape[1], E.shape[2]], dtype='int64')

        # 逐步还原
        for s_int in reversed(range(self.T)):
            s_array = paddle.full([batch_size,1], float(s_int))
            t_array = s_array + 1.
            s_norm = s_array / self.T
            t_norm = t_array / self.T

            # Sample z_s
            sampled_s, discrete_sampled_s = self.sample_p_zs_given_zt(
                s=s_norm, t=t_norm, X_t=X, E_t=E, y_t=y, node_mask=node_mask,
                conditionVec=batch_condition, batch_X=batch_X, batch_E=batch_E
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
                    f'chains/{self.cfg.general.name}/epochXX/chains/molecule_{batch_id + i}'
                )
                os.makedirs(result_path, exist_ok=True)
                # chain_X与chain_E => numpy
                chain_X_np = chain_X[:, i, :].numpy()
                chain_E_np = chain_E[:, i, :, :].numpy()

                self.visualization_tools.visualize_chain(
                    result_path,
                    chain_X_np, chain_E_np
                )
                print(f"\r {i+1}/{num_molecules} complete", end='', flush=True)
            print('\n')

            # graph
            result_path = os.path.join(current_path, f'graphs/{self.name}/epochXX_b{batch_id}/')
            result_path_true = os.path.join(current_path, f'graphs/{self.name}/True_epochXX_b{batch_id}/')
            self.visualization_tools.visualizeNmr(result_path, result_path_true, molecule_list, molecule_list_True, save_final)

        return molecule_list, molecule_list_True

    @paddle.no_grad()
    def sample_p_zs_given_zt(self, s, t, X_t, E_t, y_t, node_mask, conditionVec, batch_X, batch_E):
        """
        从 p(z_s | z_t) 采样: 反向扩散一步
        """
        beta_t = self.noise_schedule(t_normalized=t)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t)

        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, None)
        Qsb = self.transition_model.get_Qt_bar(alpha_s_bar, None)
        Qt  = self.transition_model.get_Qt(beta_t, None)

        # forward
        noisy_data = {'X_t': X_t, 'E_t': E_t, 'y_t': y_t, 't': t, 'node_mask': node_mask}
        extra_data = self.compute_extra_data(noisy_data)
        pred = self.forward_sample(noisy_data, extra_data, node_mask, batch_X, batch_E)

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
            unnormalized_prob_X
        )
        prob_X = unnormalized_prob_X / paddle.sum(unnormalized_prob_X, axis=-1, keepdim=True)

        weighted_E = pred_E.unsqueeze(-1) * p_s_and_t_given_0_E
        unnormalized_prob_E = paddle.sum(weighted_E, axis=-2)
        unnormalized_prob_E = paddle.where(
            paddle.sum(unnormalized_prob_E, axis=-1, keepdim=True) == 0,
            paddle.to_tensor(1e-5, dtype=unnormalized_prob_E.dtype),
            unnormalized_prob_E
        )
        prob_E = unnormalized_prob_E / paddle.sum(unnormalized_prob_E, axis=-1, keepdim=True)
        prob_E = prob_E.reshape([X_t.shape[0], X_t.shape[1], X_t.shape[1], -1])

        # 采样
        sampled_s = diffusion_utils.sample_discrete_features(prob_X, prob_E, node_mask)
        X_s = F.one_hot(sampled_s.X, num_classes=self.Xdim_output).astype('float32')
        E_s = F.one_hot(sampled_s.E, num_classes=self.Edim_output).astype('float32')

        out_one_hot = utils.PlaceHolder(X=X_s, E=E_s, y=paddle.zeros([y_t.shape[0], 0]))
        out_discrete = utils.PlaceHolder(X=X_s, E=E_s, y=paddle.zeros([y_t.shape[0], 0]))

        return out_one_hot.mask(node_mask), out_discrete.mask(node_mask, collapse=True)

    # -----------------------
    # 额外特征
    # -----------------------
    def compute_extra_data(self, noisy_data):
        """将 extra_features 与 domain_features 结合到X/E/y最终输入中。"""
        extra_features = self.extra_features(noisy_data)
        extra_molecular_features = self.domain_features(noisy_data)

        extra_X = paddle.concat([extra_features.X, extra_molecular_features.X], axis=-1)
        extra_E = paddle.concat([extra_features.E, extra_molecular_features.E], axis=-1)
        extra_y = paddle.concat([extra_features.y, extra_molecular_features.y], axis=-1)

        t = noisy_data['t']
        extra_y = paddle.concat([extra_y, t], axis=1)

        return utils.PlaceHolder(X=extra_X, E=extra_E, y=extra_y)

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

class CLIP_molecule_nmr(nn.Layer):
    def __init__(self, cfg, dataset_infos, train_metrics, sampling_metrics, visualization_tools, extra_features,
                 domain_features):
        """
        Paddle版本的CLIP_molecule_nmr模型，不再继承pl.LightningModule，而是普通的nn.Layer。
        """
        super().__init__()

        input_dims = dataset_infos.input_dims
        output_dims = dataset_infos.output_dims
        nodes_dist = dataset_infos.nodes_dist

        self.cfg = cfg
        self.name = cfg.general.name
        self.model_dtype = 'float32'      # 原先是torch.float32
        self.T = cfg.model.diffusion_steps

        # 以下是一些网络结构相关超参
        self.enc_voc_size = 5450
        self.max_len = 256
        self.d_model = 256
        self.ffn_hidden = 1024
        self.n_head = 8
        self.n_layers_TE = 3
        self.drop_prob = 0.0

        # Paddle 不需要手动获取 device，一般通过 paddle.set_device("gpu") 或 "cpu"
        # self.device = paddle.device.get_device()

        self.Xdim = input_dims['X']
        self.Edim = input_dims['E']
        self.ydim = input_dims['y']
        self.Xdim_output = output_dims['X']
        self.Edim_output = output_dims['E']
        self.ydim_output = output_dims['y']
        self.node_dist = nodes_dist

        self.dataset_info = dataset_infos
        self.tem = 2  # 温度/缩放参数
        self.val_loss = []

        self.train_metrics = train_metrics
        self.sampling_metrics = sampling_metrics
        self.visualization_tools = visualization_tools
        self.extra_features = extra_features
        self.domain_features = domain_features

        # 构造 backbone (contrastGT)
        self.model = contrastGT(
            n_layers_GT=cfg.model.n_layers,
            input_dims=input_dims,
            hidden_mlp_dims=cfg.model.hidden_mlp_dims,
            hidden_dims=cfg.model.hidden_dims,
            output_dims=output_dims,
            act_fn_in=nn.ReLU(),
            act_fn_out=nn.ReLU(),
            enc_voc_size=self.enc_voc_size, 
            max_len=self.max_len,
            d_model=self.d_model,
            ffn_hidden=self.ffn_hidden,
            n_head=self.n_head,
            n_layers_TE=self.n_layers_TE,
            drop_prob=self.drop_prob
            # device=self.device  # 在 Paddle 下通常不需显式传
        )

        # 噪声日程表
        self.noise_schedule = PredefinedNoiseScheduleDiscrete(
            cfg.model.diffusion_noise_schedule,
            timesteps=cfg.model.diffusion_steps
        )

        # Transition Model
        if cfg.model.transition == 'uniform':
            self.transition_model = DiscreteUniformTransition(
                x_classes=self.Xdim_output, 
                e_classes=self.Edim_output,
                y_classes=self.ydim_output
            )
            x_limit = paddle.ones([self.Xdim_output]) / self.Xdim_output
            e_limit = paddle.ones([self.Edim_output]) / self.Edim_output
            y_limit = paddle.ones([self.ydim_output]) / self.ydim_output
            self.limit_dist = utils.PlaceHolder(X=x_limit, E=e_limit, y=y_limit)

        elif cfg.model.transition == 'marginal':
            node_types = self.dataset_info.node_types.astype('float32')
            x_marginals = node_types / paddle.sum(node_types)

            edge_types = self.dataset_info.edge_types.astype('float32')
            e_marginals = edge_types / paddle.sum(edge_types)
            print(f"Marginal distribution of the classes: {x_marginals} for nodes, {e_marginals} for edges")

            self.transition_model = MarginalUniformTransition(
                x_marginals=x_marginals, e_marginals=e_marginals, y_classes=self.ydim_output
            )
            self.limit_dist = utils.PlaceHolder(
                X=x_marginals, 
                E=e_marginals,
                y=paddle.ones([self.ydim_output]) / self.ydim_output
            )

        self.train_iterations = None
        self.val_iterations = None
        self.log_every_steps = cfg.general.log_every_steps
        self.number_chain_steps = cfg.general.number_chain_steps
        self.best_val_nll = 1e8
        self.val_counter = 0
        self.vocabDim = 256

    def forward(self, noisy_data, extra_data, node_mask, X, E, condition):
        """
        前向逻辑：和LightningModule的 forward 不同，这里是普通方法
        """
        # 拼接
        X_ = paddle.concat([noisy_data['X_t'], extra_data.X], axis=2).astype('float32')
        E_ = paddle.concat([noisy_data['E_t'], extra_data.E], axis=3).astype('float32')
        y_ = paddle.concat([noisy_data['y_t'], extra_data.y], axis=1).astype('float32')

        # 把 condition 转到 Paddle Tensor
        # 如果是 int64 -> 'int64', or as needed
        condition_tensor = paddle.to_tensor(condition, dtype='int64')

        # 调用 self.model
        return self.model(X_, E_, y_, node_mask, X, E, condition_tensor)

    # --------------------------
    # 训练阶段
    # --------------------------
    def train_step(self, data, i):
        """
        原Lightning的 training_step => 自定义函数
        data: batch data
        i: iteration index
        return: dict(loss=...)
        """
        if data.edge_index.size(0) == 0:
            print("Found a batch with no edges. Skipping.")
            return None

        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        dense_data = dense_data.mask(node_mask)
        X, E = dense_data.X, dense_data.E

        # 加噪
        noisy_data = self.apply_noise(X, E, data.y, node_mask)
        extra_data = self.compute_extra_data(noisy_data)

        batch_length = data.num_graphs
        conditionAll = data.conditionVec
        conditionAll = paddle.reshape(conditionAll, [batch_length, self.vocabDim])

        # 前向
        predV1, predV2 = self.forward(noisy_data, extra_data, node_mask, X, E, conditionAll)

        # L2 normalize
        V1_e = F.normalize(predV1, p=2, axis=1)
        V2_e = F.normalize(predV2, p=2, axis=1)

        # 矩阵相乘 => (bs, bs)
        # 原: torch.matmul(V1_e, V2_e.T)*exp(torch.tensor(self.tem))
        temperature = paddle.to_tensor(self.tem, dtype=V1_e.dtype)
        logits = paddle.matmul(V1_e, V2_e, transpose_y=True) * paddle.exp(temperature)

        # 交叉熵损失
        n = V1_e.shape[0]
        labels = paddle.arange(0, n, dtype='int64')  # (bs,)
        loss_fn = nn.CrossEntropyLoss()

        # loss_v1
        loss_v1 = loss_fn(logits, labels)
        # loss_v2 => 对称
        loss_v2 = loss_fn(logits.transpose([1, 0]), labels)

        loss = (loss_v1 + loss_v2) / 2.0

        if i % 100 == 0:
            print(f"train_loss: {loss.numpy()}")

        return {'loss': loss}

    # --------------------------
    # 验证阶段
    # --------------------------
    def val_step(self, data, i):
        """
        类似Lightning的 validation_step
        """
        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        dense_data = dense_data.mask(node_mask)
        X, E = dense_data.X, dense_data.E

        # 加噪
        noisy_data = self.apply_noise(X, E, data.y, node_mask)
        extra_data = self.compute_extra_data(noisy_data)

        batch_length = data.num_graphs
        conditionAll = data.conditionVec
        conditionAll = paddle.reshape(conditionAll, [batch_length, self.vocabDim])

        predV1, predV2 = self.forward(noisy_data, extra_data, node_mask, X, E, conditionAll)
        V1_e = F.normalize(predV1, p=2, axis=1)
        V2_e = F.normalize(predV2, p=2, axis=1)

        temperature = paddle.to_tensor(self.tem, dtype=V1_e.dtype)
        logits = paddle.matmul(V1_e, V2_e, transpose_y=True) * paddle.exp(temperature)

        n = V1_e.shape[0]
        labels = paddle.arange(0, n, dtype='int64')
        loss_fn = nn.CrossEntropyLoss()

        loss_v1 = loss_fn(logits, labels)
        loss_v2 = loss_fn(logits.transpose([1,0]), labels)
        loss = (loss_v1 + loss_v2) / 2.0

        self.val_loss.append(loss)

        if i % 8 == 0:
            print(f"val_loss: {loss.numpy()}")

        return {'loss': loss}

    # --------------------------
    # 测试阶段
    # --------------------------
    def test_step(self, data, i):
        """
        类似Lightning的 test_step
        """
        # 可根据需求实现
        pass

    # --------------------------
    # apply_noise
    # --------------------------
    def apply_noise(self, X, E, y, node_mask):
        """
        Sample noise and apply it to the data.
        """
        bs = X.shape[0]
        # t_int in [1, T]
        t_int = paddle.randint(low=1, high=self.T+1, shape=[bs,1], dtype='int64')
        t_int = t_int.astype('float32') 
        s_int = t_int - 1

        t_float = t_int / self.T
        s_float = s_int / self.T

        beta_t = self.noise_schedule(t_normalized=t_float)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s_float)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_float)

        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, device=None)
        # probX = X @ Qtb.X => paddle.matmul(X, Qtb.X)
        probX = paddle.matmul(X, Qtb.X)  # (bs, n, dx_out)
        probE = paddle.matmul(E, Qtb.E.unsqueeze(1)) # (bs, n, n, de_out)

        sampled_t = diffusion_utils.sample_discrete_features(probX=probX, probE=probE, node_mask=node_mask)

        X_t = F.one_hot(sampled_t.X, num_classes=self.Xdim_output)
        E_t = F.one_hot(sampled_t.E, num_classes=self.Edim_output)

        z_t = utils.PlaceHolder(X=X_t, E=E_t, y=y).astype('float32').mask(node_mask)

        noisy_data = {
            't_int': t_int, 't': t_float,
            'beta_t': beta_t,
            'alpha_s_bar': alpha_s_bar,
            'alpha_t_bar': alpha_t_bar,
            'X_t': z_t.X, 'E_t': z_t.E, 'y_t': z_t.y,
            'node_mask': node_mask
        }
        return noisy_data

    def compute_extra_data(self, noisy_data):
        """
        生成额外特征（extra_features + domain_features），并拼接 t
        """
        extra_features = self.extra_features(noisy_data)
        extra_molecular_features = self.domain_features(noisy_data)

        extra_X = paddle.concat([extra_features.X, extra_molecular_features.X], axis=-1)
        extra_E = paddle.concat([extra_features.E, extra_molecular_features.E], axis=-1)
        extra_y = paddle.concat([extra_features.y, extra_molecular_features.y], axis=-1)

        t = noisy_data['t']
        extra_y = paddle.concat([extra_y, t], axis=1)

        return utils.PlaceHolder(X=extra_X, E=extra_E, y=extra_y)

    # --------------------------
    # 可选的一些回调
    # --------------------------
    def on_train_epoch_start(self):
        print("Starting train epoch...")

    def on_train_epoch_end(self):
        # 这里可以做一些清理或日志记录
        sys.stdout.flush()

    def on_validation_epoch_start(self):
        self.val_loss = []

    def on_validation_epoch_end(self):
        val_loss_sum = paddle.add_n([v for v in self.val_loss])  # or sum(self.val_loss)
        # sum(...) => 需要是相同dtype
        val_loss_val = val_loss_sum.numpy()[0] if len(val_loss_sum.shape) > 0 else val_loss_sum.numpy()
        print(f"Epoch {0} : Val Loss {val_loss_val:.2f}")  # 或 self.current_epoch

    def on_test_epoch_start(self):
        pass

    def on_test_epoch_end(self):
        print("Done testing.")

class DiscreteDenoisingDiffusionCondition(nn.Layer):
    def __init__(self, cfg, dataset_infos, train_metrics, sampling_metrics, visualization_tools,
                 extra_features, domain_features):
        super().__init__()

        input_dims = dataset_infos.input_dims
        output_dims = dataset_infos.output_dims
        nodes_dist = dataset_infos.nodes_dist

        self.cfg = cfg
        self.name = cfg.general.name
        self.model_dtype = 'float32'
        self.T = cfg.model.diffusion_steps

        self.enc_voc_size = 5450
        self.max_len = 256
        self.d_model = 256
        self.ffn_hidden = 1024
        self.n_head = 8
        self.n_layers_TE = 3
        self.drop_prob = 0.0
        self.device = "gpu"  # Paddle 通常用 paddle.set_device("gpu") 来设置设备

        self.Xdim = input_dims['X']
        self.Edim = input_dims['E']
        self.ydim = input_dims['y']
        self.Xdim_output = output_dims['X']
        self.Edim_output = output_dims['E']
        self.ydim_output = output_dims['y']
        self.node_dist = nodes_dist

        self.dataset_info = dataset_infos

        # 训练损失 & 一些指标
        self.train_loss = TrainLossDiscrete(self.cfg.model.lambda_train)

        self.val_nll = NLL()
        self.val_X_kl = SumExceptBatchKL()
        self.val_E_kl = SumExceptBatchKL()
        self.val_X_logp = SumExceptBatchMetric()
        self.val_E_logp = SumExceptBatchMetric()
        self.val_y_collection = []
        self.val_atomCount = []
        self.val_x = []
        self.val_e = []

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

        self.visualization_tools = visualization_tools
        self.extra_features = extra_features
        self.domain_features = domain_features

        # 构建 ConditionGT 模型
        self.model = ConditionGT(
            n_layers_GT=cfg.model.n_layers,
            input_dims=input_dims,
            hidden_mlp_dims=cfg.model.hidden_mlp_dims,
            hidden_dims=cfg.model.hidden_dims,
            output_dims=output_dims,
            act_fn_in=nn.ReLU(),
            act_fn_out=nn.ReLU(),
            enc_voc_size=self.enc_voc_size,
            max_len=self.max_len,
            d_model=self.d_model,
            ffn_hidden=self.ffn_hidden,
            n_head=self.n_head,
            n_layers_TE=self.n_layers_TE,
            drop_prob=self.drop_prob,
            device=self.device
        )

        # 噪声日程表
        self.noise_schedule = PredefinedNoiseScheduleDiscrete(
            cfg.model.diffusion_noise_schedule,
            timesteps=cfg.model.diffusion_steps
        )

        # Transition Model
        if cfg.model.transition == 'uniform':
            self.transition_model = DiscreteUniformTransition(
                x_classes=self.Xdim_output,
                e_classes=self.Edim_output,
                y_classes=self.ydim_output
            )
            x_limit = paddle.ones([self.Xdim_output], dtype='float32') / self.Xdim_output
            e_limit = paddle.ones([self.Edim_output], dtype='float32') / self.Edim_output
            y_limit = paddle.ones([self.ydim_output], dtype='float32') / self.ydim_output
            self.limit_dist = utils.PlaceHolder(X=x_limit, E=e_limit, y=y_limit)

        elif cfg.model.transition == 'marginal':
            node_types = paddle.to_tensor(self.dataset_info.node_types, dtype='float32')
            x_marginals = node_types / paddle.sum(node_types)
            edge_types = paddle.to_tensor(self.dataset_info.edge_types, dtype='float32')
            e_marginals = edge_types / paddle.sum(edge_types)
            print(f"Marginal distribution of the classes: {x_marginals} for nodes, {e_marginals} for edges")
            self.transition_model = MarginalUniformTransition(
                x_marginals=x_marginals,
                e_marginals=e_marginals,
                y_classes=self.ydim_output
            )
            y_limit = paddle.ones([self.ydim_output], dtype='float32') / self.ydim_output
            self.limit_dist = utils.PlaceHolder(X=x_marginals, E=e_marginals, y=y_limit)

        # 其余属性
        self.start_epoch_time = None
        self.train_iterations = None
        self.val_iterations = None
        self.log_every_steps = cfg.general.log_every_steps
        self.number_chain_steps = cfg.general.number_chain_steps
        self.best_val_nll = 1e8
        self.val_counter = 0
        self.vocabDim = 256

    # -------------------------
    # 优化器 (可选)
    # -------------------------
    def configure_optimizers(self):
        return paddle.optimizer.AdamW(
            parameters=self.parameters(),
            learning_rate=self.cfg.train.lr,
            weight_decay=self.cfg.train.weight_decay
        )

    # -------------------------
    # 训练循环 (Lightning => 手动 train_step)
    # -------------------------
    def train_step(self, data, i):
        """
        在外部训练循环中:
          for i, batch_data in enumerate(train_dataloader):
              loss_dict = model.train_step(batch_data, i)
              loss = loss_dict['loss']
              loss.backward()
              optimizer.step()
              optimizer.clear_grad()
        """
        if data.edge_index.size(1) == 0:
            print("Found a batch with no edges. Skipping.")
            return None

        # to_dense
        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        dense_data = dense_data.mask(node_mask)
        X, E = dense_data.X, dense_data.E

        # 加噪
        noisy_data = self.apply_noise(X, E, data.y, node_mask)
        extra_data = self.compute_extra_data(noisy_data)

        batch_length = data.num_graphs
        conditionAll = data.conditionVec
        conditionAll = paddle.reshape(conditionAll, [batch_length, self.vocabDim])

        # forward
        pred = self.forward(noisy_data, extra_data, node_mask, conditionAll)
        # compute loss
        loss_val = self.train_loss(
            masked_pred_X=pred.X, masked_pred_E=pred.E, pred_y=pred.y,
            true_X=X, true_E=E, true_y=data.y,
            log=(i % self.log_every_steps == 0)
        )
        if i % 80 == 0:
            print(f"train_loss: {loss_val.numpy()[0]}")
        # train_metrics
        self.train_metrics(
            masked_pred_X=pred.X, masked_pred_E=pred.E, true_X=X, true_E=E,
            log=(i % self.log_every_steps == 0)
        )
        sys.stdout.flush()
        return {'loss': loss_val}

    # -------------------------
    # 验证循环 => val_step
    # -------------------------
    def val_step(self, data, i):
        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        dense_data = dense_data.mask(node_mask)
        X, E = dense_data.X, dense_data.E

        # apply noise
        noisy_data = self.apply_noise(X, E, data.y, node_mask)
        extra_data = self.compute_extra_data(noisy_data)

        batch_length = data.num_graphs
        conditionAll = data.conditionVec
        conditionAll = paddle.reshape(conditionAll, [batch_length, self.vocabDim])

        pred = self.forward(noisy_data, extra_data, node_mask, conditionAll)

        self.val_y_collection.append(data.conditionVec)
        self.val_atomCount.append(data.atom_count)
        self.val_x.append(X)
        self.val_e.append(E)

        # 计算 training loss
        loss_val = self.train_loss(
            masked_pred_X=pred.X, masked_pred_E=pred.E, pred_y=pred.y,
            true_X=X, true_E=E, true_y=data.y,
            log=(i % self.log_every_steps == 0)
        )
        if i % 10 == 0:
            print(f"val_loss: {loss_val.numpy()[0]}")

        # 进一步计算NLL
        nll = self.compute_val_loss(
            pred, noisy_data, X, E, data.y, node_mask,
            condition=conditionAll, test=False
        )
        return {'loss': nll}

    # -------------------------
    # 测试循环 => test_step
    # -------------------------
    def test_step(self, data, i):
        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        dense_data = dense_data.mask(node_mask)
        X, E = dense_data.X, dense_data.E

        noisy_data = self.apply_noise(X, E, data.y, node_mask)
        extra_data = self.compute_extra_data(noisy_data)

        batch_length = data.num_graphs
        conditionAll = data.conditionVec
        conditionAll = paddle.reshape(conditionAll, [batch_length, self.vocabDim])

        pred = self.forward(noisy_data, extra_data, node_mask, conditionAll)

        self.test_y_collection.append(data.conditionVec)
        self.test_atomCount.append(data.atom_count)
        self.test_x.append(X)
        self.test_e.append(E)

        nll = self.compute_val_loss(
            pred, noisy_data, X, E, data.y, node_mask,
            condition=conditionAll, test=True
        )
        return {'loss': nll}

    # -------------------------
    # 噪声 & Q
    # -------------------------
    def apply_noise(self, X, E, y, node_mask):
        """
        随机选择 t in [1, T], 根据 alpha_t_bar 计算 X_t, E_t
        """
        bs = X.shape[0]
        lowest_t = 1
        t_int = paddle.randint(lowest_t, self.T+1, shape=[bs,1], dtype='int64').astype('float32')
        s_int = t_int - 1

        t_float = t_int / self.T
        s_float = s_int / self.T

        beta_t = self.noise_schedule(t_normalized=t_float)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s_float)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_float)

        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, device=self.device)

        # probX = X @ Qtb.X => paddle.matmul
        probX = paddle.matmul(X, Qtb.X)
        probE = paddle.matmul(E, Qtb.E.unsqueeze(1))

        sampled_t = diffusion_utils.sample_discrete_features(probX, probE, node_mask)
        X_t = F.one_hot(sampled_t.X, num_classes=self.Xdim_output)
        E_t = F.one_hot(sampled_t.E, num_classes=self.Edim_output)

        z_t = utils.PlaceHolder(X=X_t, E=E_t, y=y).astype('float32').mask(node_mask)

        noisy_data = {
            't_int': t_int,
            't': t_float,
            'beta_t': beta_t,
            'alpha_s_bar': alpha_s_bar,
            'alpha_t_bar': alpha_t_bar,
            'X_t': z_t.X,
            'E_t': z_t.E,
            'y_t': z_t.y,
            'node_mask': node_mask
        }
        return noisy_data

    def compute_val_loss(self, pred, noisy_data, X, E, y, node_mask, condition, test=False):
        """
        估计 variational lower bound
        """
        t = noisy_data['t']
        N = paddle.sum(node_mask, axis=1).astype('int64')
        log_pN = self.node_dist.log_prob(N)

        # kl_prior => uniform
        kl_prior_ = self.kl_prior(X, E, node_mask)

        # diffusion loss
        loss_all_t = self.compute_Lt(X, E, y, pred, noisy_data, node_mask, test)

        # reconstruction
        prob0 = self.reconstruction_logp(t, X, E, node_mask, condition)
        # val_X_logp, val_E_logp
        loss_term_0 = self.val_X_logp(X * paddle.log(prob0.X + 1e-10)) + \
                      self.val_E_logp(E * paddle.log(prob0.E + 1e-10))

        nlls = - log_pN + kl_prior_ + loss_all_t - loss_term_0
        nll = (self.test_nll if test else self.val_nll)(nlls)

        # 在 Paddle 下若需 wandb，需要 import wandb 并保证 wandb.run != None
        # ...
        return nll

    # -------------------------
    # forward => 组装拼接 + 调用 ConditionGT
    # -------------------------
    def forward(self, noisy_data, extra_data, node_mask, condition):
        """
        注：Lightning的 forward => 这里是普通 forward
        """
        X_ = paddle.concat([noisy_data['X_t'], extra_data.X], axis=2).astype('float32')
        E_ = paddle.concat([noisy_data['E_t'], extra_data.E], axis=3).astype('float32')
        y_ = paddle.concat([noisy_data['y_t'], extra_data.y], axis=1).astype('float32')

        condition_t = paddle.to_tensor(condition, dtype='int64')
        return self.model(X_, E_, y_, node_mask, condition_t)

    def forward_sample(self, noisy_data, extra_data, node_mask, condition):
        """
        用于sample时: 只是不记录梯度
        """
        X_ = paddle.concat([noisy_data['X_t'], extra_data.X], axis=2).astype('float32')
        E_ = paddle.concat([noisy_data['E_t'], extra_data.E], axis=3).astype('float32')
        y_ = paddle.concat([noisy_data['y_t'], extra_data.y], axis=1).astype('float32')
        condition_t = paddle.to_tensor(condition, dtype='int64')
        return self.model(X_, E_, y_, node_mask, condition_t)

    # -------------------------
    # KL prior
    # -------------------------
    def kl_prior(self, X, E, node_mask):
        bs = X.shape[0]
        ones = paddle.ones([bs,1], dtype='float32')
        Ts = self.T * ones
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_int=Ts)

        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device)
        probX = paddle.matmul(X, Qtb.X)
        probE = paddle.matmul(E, Qtb.E.unsqueeze(1))
        # limit
        limit_X = paddle.expand(self.limit_dist.X.unsqueeze(0).unsqueeze(0), [bs, X.shape[1], -1])
        limit_E = paddle.expand(self.limit_dist.E.unsqueeze(0).unsqueeze(0).unsqueeze(0),
                                [bs, E.shape[1], E.shape[2], -1])
        # mask
        limit_dist_X, limit_dist_E, probX_m, probE_m = diffusion_utils.mask_distributions(
            true_X=limit_X.clone(), true_E=limit_E.clone(),
            pred_X=probX, pred_E=probE,
            node_mask=node_mask
        )
        kl_distance_X = F.kl_div(paddle.log(probX_m + 1e-10), limit_dist_X, reduction='none')
        kl_distance_E = F.kl_div(paddle.log(probE_m + 1e-10), limit_dist_E, reduction='none')

        return diffusion_utils.sum_except_batch(kl_distance_X) + diffusion_utils.sum_except_batch(kl_distance_E)

    def compute_Lt(self, X, E, y, pred, noisy_data, node_mask, test):
        """
        逐步KL
        """
        pred_probs_X = F.softmax(pred.X, axis=-1)
        pred_probs_E = F.softmax(pred.E, axis=-1)
        pred_probs_y = F.softmax(pred.y, axis=-1)

        Qtb = self.transition_model.get_Qt_bar(noisy_data['alpha_t_bar'], self.device)
        Qsb = self.transition_model.get_Qt_bar(noisy_data['alpha_s_bar'], self.device)
        Qt  = self.transition_model.get_Qt(noisy_data['beta_t'], self.device)

        bs, n, _ = X.shape
        prob_true = diffusion_utils.posterior_distributions(
            X=X, E=E, y=y,
            X_t=noisy_data['X_t'], E_t=noisy_data['E_t'], y_t=noisy_data['y_t'],
            Qt=Qt, Qsb=Qsb, Qtb=Qtb
        )
        prob_true.E = paddle.reshape(prob_true.E, [bs, n, n, -1])

        prob_pred = diffusion_utils.posterior_distributions(
            X=pred_probs_X, E=pred_probs_E, y=pred_probs_y,
            X_t=noisy_data['X_t'], E_t=noisy_data['E_t'], y_t=noisy_data['y_t'],
            Qt=Qt, Qsb=Qsb, Qtb=Qtb
        )
        prob_pred.E = paddle.reshape(prob_pred.E, [bs, n, n, -1])

        # mask
        prob_true_X, prob_true_E, prob_pred_X, prob_pred_E = diffusion_utils.mask_distributions(
            true_X=prob_true.X, true_E=prob_true.E,
            pred_X=prob_pred.X, pred_E=prob_pred.E,
            node_mask=node_mask
        )

        kl_x = (self.test_X_kl if test else self.val_X_kl)(prob_true_X, paddle.log(prob_pred_X + 1e-10))
        kl_e = (self.test_E_kl if test else self.val_E_kl)(prob_true_E, paddle.log(prob_pred_E + 1e-10))
        return self.T * (kl_x + kl_e)

    def reconstruction_logp(self, t, X, E, node_mask, condition):
        """
        L0: -log p(X,E | z0)
        """
        t_zeros = paddle.zeros_like(t)
        beta_0 = self.noise_schedule(t_zeros)
        Q0 = self.transition_model.get_Qt(beta_t=beta_0, device=self.device)

        probX0 = paddle.matmul(X, Q0.X)
        probE0 = paddle.matmul(E, Q0.E.unsqueeze(1))

        sampled0 = diffusion_utils.sample_discrete_features(probX0, probE0, node_mask)
        X0 = F.one_hot(sampled0.X, num_classes=self.Xdim_output).astype('float32')
        E0 = F.one_hot(sampled0.E, num_classes=self.Edim_output).astype('float32')
        y0 = sampled0.y

        # forward
        sampled_0 = utils.PlaceHolder(X=X0, E=E0, y=y0).mask(node_mask)
        zeros_t = paddle.zeros([X0.shape[0],1], dtype='float32')
        noised_data = {'X_t': sampled_0.X, 'E_t': sampled_0.E, 'y_t': sampled_0.y,
                       'node_mask': node_mask, 't': zeros_t}
        extra_data = self.compute_extra_data(noised_data)
        pred0 = self.forward(noised_data, extra_data, node_mask, condition)

        probX0 = F.softmax(pred0.X, axis=-1)
        probE0 = F.softmax(pred0.E, axis=-1)
        proby0 = F.softmax(pred0.y, axis=-1)

        # 屏蔽无效节点
        probX0 = paddle.where(~node_mask.unsqueeze(-1), paddle.ones_like(probX0)*1.0, probX0)
        # E
        node_mask_2d = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
        probE0 = paddle.where(~node_mask_2d.unsqueeze(-1), paddle.ones_like(probE0)*1.0, probE0)

        diag_mask = paddle.eye(probE0.shape[1], dtype='bool')
        diag_mask = diag_mask.unsqueeze(0)
        diag_mask = diag_mask.expand([probE0.shape[0], -1, -1])
        probE0 = paddle.where(diag_mask.unsqueeze(-1), paddle.ones_like(probE0)*1.0, probE0)

        return utils.PlaceHolder(X=probX0, E=probE0, y=proby0)

    # -------------------------
    # 采样 => sample_batch
    # -------------------------
    @paddle.no_grad()
    def sample_batch(self, batch_id: int, batch_size: int, batch_condition, keep_chain: int, 
                     number_chain_steps: int, save_final: int, batch_X, batch_E, num_nodes=None):

        # 这里是反向扩散采样逻辑
        # 与Lightning下相同，只需把 torch.* -> paddle.* 并注意张量形状
        ...

    def mol_from_graphs(self, node_list, adjacency_matrix):
        """
        Convert graphs to rdkit molecules
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
        except Chem.KekulizeException:
            print("Can't kekulize molecule")
            mol = None
        return mol

    # -------------------------
    # 如果想仿Lightning的回调
    # -------------------------
    def on_train_epoch_start(self):
        print("Starting train epoch...")
        self.start_epoch_time = time.time()
        self.train_loss.reset()
        self.train_metrics.reset()

    def on_train_epoch_end(self):
        to_log = self.train_loss.log_epoch_metrics()
        print(f"Epoch XX: X_CE: {to_log['train_epoch/x_CE'] :.3f}, E_CE: {to_log['train_epoch/E_CE'] :.3f}, "
              f"y_CE: {to_log['train_epoch/y_CE'] :.3f}, Time: {time.time() - self.start_epoch_time:.1f}s")
        epoch_at_metrics, epoch_bond_metrics = self.train_metrics.log_epoch_metrics()
        print(f"Train epoch end: {epoch_at_metrics} -- {epoch_bond_metrics}")

    def on_validation_epoch_start(self):
        self.val_nll.reset()
        self.val_X_kl.reset()
        self.val_E_kl.reset()
        self.val_X_logp.reset()
        self.val_E_logp.reset()
        self.sampling_metrics.reset()
        self.val_y_collection = []
        self.val_atomCount = []
        self.val_x = []
        self.val_e = []

    def on_validation_epoch_end(self):
        metrics = [
            self.val_nll.compute(), 
            self.val_X_kl.compute() * self.T,
            self.val_E_kl.compute() * self.T,
            self.val_X_logp.compute(),
            self.val_E_logp.compute()
        ]
        print(f"Val NLL {metrics[0]:.2f} | Val Atom KL {metrics[1]:.2f} | Val Edge KL {metrics[2]:.2f}")

    def on_test_epoch_start(self):
        print("Starting test...")
        self.test_nll.reset()
        self.test_X_kl.reset()
        self.test_E_kl.reset()
        self.test_X_logp.reset()
        self.test_E_logp.reset()
        self.test_y_collection = []
        self.test_atomCount = []
        self.test_x = []
        self.test_e = []

    def on_test_epoch_end(self):
        print("Done testing.")