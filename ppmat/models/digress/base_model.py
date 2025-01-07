import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from collections import defaultdict
import os
import time
from rdkit import Chem

from .graph_transformer import GraphTransformer, GraphTransformer_C
from .noise_schedule import PredefinedNoiseScheduleDiscrete, DiscreteUniformTransition, MarginalUniformTransition
from ppmat.metrics.train_metrics import TrainLossDiscrete
from ppmat.metrics.abstract_metrics import NLL, SumExceptBatchKL, SumExceptBatchMetric
from . import diffusion_utils
from ppmat.utils import digressutils as utils

from .graph_transformer import GraphTransformer, GraphTransformer_C

from paddle.nn import TransformerEncoderLayer
from paddle.nn import TransformerEncoder as Encoder

class MolecularGraphTransformer(paddle.nn.Layer):
    def __init__(
        self,
        cfg: dict,
        encoder_cfg: dict,
        decoder_cfg: dict,
        dataset_infos,
        train_metrics,
        sampling_metrics,
        visualization_tools,
        extra_features,
        domain_features,
        **kwargs
    ) -> None:
        super().__init__()
        
        # configure infos
        
        self.cfg = cfg
        self.name = cfg.general.name
        self.model_dtype = "float32"
        self.T = cfg.model.diffusion_steps
        
        self.con_input_dim = dataset_infos.input_dims
        self.con_input_dim['X'] = dataset_infos.input_dims['X'] - 8
        self.con_input_dim['y'] = 1024
        self.con_output_dim = dataset_infos.output_dims
        self.node_dist = dataset_infos.nodes_dist
        
        self.Xdim = dataset_infos.input_dims['X']
        self.Edim = dataset_infos.input_dims['E']
        self.ydim = dataset_infos.input_dims['y']
        self.Xdim_output = dataset_infos.output_dims['X']
        self.Edim_output = dataset_infos.output_dims['E']
        self.ydim_output = dataset_infos.output_dims['y']
        
        self.dataset_info = dataset_infos
        
        # configure loss of training validation and testing
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
        
        # container
        self.val_y_collection = []
        self.val_atomCount = []
        self.val_data_X = []
        self.val_data_E = []
        self.test_y_collection = []
        self.test_atomCount = []
        self.test_data_X = []
        self.test_data_E = []
        
        # configure metrics of training sampling
        self.train_metrics = train_metrics
        self.sampling_metrics = sampling_metrics
        
        # visualization
        self.visualization_tools = visualization_tools
        
        # extra features
        self.extra_features = extra_features
        self.domain_features = domain_features
        
        # configure model
        self.encoder = GraphTransformer(**cfg.encoder_cfg)
        self.decoder = GraphTransformer_C(**cfg.decoder_cfg)
        
        # noise scheduler
        self.noise_scheduler = PredefinedNoiseScheduleDiscrete(
            cfg.model.diffusion_noise_schedule, timesteps=self.T
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
            print(f"Marginal distribution of classes: {x_marginals} for nodes, {e_marginals} for edges")

            self.transition_model = MarginalUniformTransition(
                x_marginals=x_marginals, e_marginals=e_marginals, y_classes=self.ydim_output
            )
            self.limit_dist = utils.PlaceHolder(
                X=x_marginals, E=e_marginals,
                y=paddle.ones([self.ydim_output]) / self.ydim_output
            )
        
        # other properties
        self.start_epoch_time = None
        self.best_val_nll = 1e8
        self.val_counter = 0
        self.vocabDim = 256
        self.number_chain_steps = cfg.general.number_chain_steps
        self.log_every_steps = cfg.general.log_every_steps

    def forward(self, data, i:int):
        
        # transfer to dense graph from sparse graph
        if data.edge_index.numel() == 0:
            print("Found a batch with no edges. Skipping.")
            return None

        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        dense_data = dense_data.mask(node_mask)
        X, E = dense_data.X, dense_data.E

        # add noise to the inputs (X, E)
        noisy_data = self.apply_noise(X, E, data.y, node_mask)
        extra_data = self.compute_extra_data(noisy_data)
        
        # forward pass
        pred = self.forward_core(noisy_data, extra_data, node_mask, X, E)
        loss = self.train_loss(
            masked_pred_X=pred.X, masked_pred_E=pred.E, pred_y=pred.y,
            true_X=X, true_E=E, true_y=data.y,
            log=(i % self.log_every_steps == 0)
        )
        
        # log metrics
        if i % 80 == 0:
            print(f"train_loss: {loss}")
        self.train_metrics(
            masked_pred_X=pred.X, masked_pred_E=pred.E,
            true_X=X, true_E=E,
            log=(i % self.log_every_steps == 0)
        )
        
        return loss
 
    def forward_core(self, noisy_data, extra_data, node_mask, X, E):
        
        X = paddle.concat([noisy_data['X_t'], extra_data.X], axis=2).astype('float32')
        
        E = paddle.concat([noisy_data['E_t'], extra_data.E], axis=3).astype('float32')
        
        y = paddle.concat([noisy_data['y_t'], extra_data.y], axis=1).astype('float32')
        y_condition = paddle.zeros(shape=[X.shape[0], 1024]).cuda(blocking=True)
        conditionVec = self.conditionEn(X, E, y_condition, node_mask)
        y = paddle.hstack(x=(y, conditionVec)).astype(dtype='float32')
        
        return self.encoder(X, E, y, node_mask)

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
        #  mix extra_features with domain_features and noisy_data into X/E/y final inputs. domain_features
        extra_features = self.extra_features(noisy_data)
        extra_molecular_features = self.domain_features(noisy_data)

        extra_X = paddle.concat([extra_features.X, extra_molecular_features.X], axis=-1)
        extra_E = paddle.concat([extra_features.E, extra_molecular_features.E], axis=-1)
        extra_y = paddle.concat([extra_features.y, extra_molecular_features.y], axis=-1)

        t = noisy_data['t']
        extra_y = paddle.concat([extra_y, t], axis=1)

        return utils.PlaceHolder(X=extra_X, E=extra_E, y=extra_y)
    
    @paddle.no_grad()
    def sample(self, batch, step_lr=1e-05, is_save_traj=False):
        
    
    

"""
class ContrastGraphTransformer(paddle.nn.Layer):
    def __init__(self, n_layers_GT: int, input_dims: dict, hidden_mlp_dims:
        dict, hidden_dims: dict, output_dims: dict, act_fn_in: paddle.nn.
        ReLU(), act_fn_out: paddle.nn.ReLU(), enc_voc_size, max_len,
        d_model, ffn_hidden, n_head, n_layers_TE, drop_prob, device):
        super().__init__()
        self.transEn = Encoder(enc_voc_size=enc_voc_size, max_len=max_len,
            d_model=d_model, ffn_hidden=ffn_hidden, n_head=n_head, n_layers
            =n_layers_TE, drop_prob=drop_prob, device=device)
        self.linear_layer = paddle.nn.Linear(in_features=max_len * d_model,
            out_features=512)
        self.con_input_dim = input_dims
        self.con_input_dim['X'] = input_dims['X'] - 8
        self.con_input_dim['y'] = 1024
        self.con_output_dim = output_dims
        self.conditionEn = GraphTransformer_C(n_layers=n_layers_GT,
            input_dims=self.con_input_dim, hidden_mlp_dims=hidden_mlp_dims,
            hidden_dims=hidden_dims, output_dims=self.con_output_dim,
            act_fn_in=act_fn_in, act_fn_out=act_fn_out)
        self.device = device
        checkpoint = paddle.load(path=str(
            '/home/liuxuwei01/molecular2molecular/src/epoch-438.ckpt'))
        state_dict = checkpoint['state_dict']
        print(state_dict.keys())
        conditionEn_state_dict = {k[len('model.conditionEn.'):]: v for k, v in
            state_dict.items() if k.startswith('model.conditionEn.')}
        self.conditionEn.set_state_dict(state_dict=conditionEn_state_dict)
        print('conditionEn parameters loaded successfully.')
        for param in self.conditionEn.parameters():
            param.stop_gradient = not False
        self.conditionEn.eval()

    def make_src_mask(self, src):
        src_mask = (src != 0).unsqueeze(axis=1).unsqueeze(axis=2)
        return src_mask

    def forward(self, X, E, y, node_mask, X_condition, E_condtion, conditionVec
        ):
        assert isinstance(conditionVec, paddle.Tensor
            ), 'conditionVec should be a tensor, but got type {}'.format(type
            (conditionVec))
        srcMask = self.make_src_mask(conditionVec).to(self.device)
        conditionVecNmr = self.transEn(conditionVec, srcMask)
        conditionVecNmr = conditionVecNmr.view(conditionVecNmr.shape[0], -1)
        conditionVecNmr = self.linear_layer(conditionVecNmr)
        y_condition = paddle.zeros(shape=[X_condition.shape[0], 1024]).cuda(
            blocking=True)
        conditionVecM = self.conditionEn(X_condition, E_condtion,
            y_condition, node_mask)
        return conditionVecM, conditionVecNmr

class ConditionGraphTransformer(nn.Layer):

    def __init__(
        self, 
        n_layers_GT: int, 
        input_dims: dict, 
        hidden_mlp_dims:dict, 
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
        device
    ):
        super().__init__()
        self.GT = GraphTransformer(
            n_layers = n_layers_GT, 
            input_dims = input_dims, 
            hidden_mlp_dims = hidden_mlp_dims, 
            hidden_dims = hidden_dims, 
            output_dims = output_dims, 
            act_fn_in = act_fn_in,
            act_fn_out = act_fn_out
        )
        self.transEn = Encoder(
            enc_voc_size = enc_voc_size, 
            max_len = max_len,
            d_model = d_model, 
            ffn_hidden = ffn_hidden, 
            n_head = n_head, 
            n_layers =n_layers_TE, 
            drop_prob = drop_prob, 
            device = device
        )
        self.linear_layer = paddle.nn.Linear(
            in_features = max_len * d_model,
            out_features = 512
        )
        self.device = device

        checkpoint = paddle.load('/home/liuxuwei01/molecular2molecular/src/epoch-438.ckpt')
        state_dict = checkpoint['state_dict']
        GT_state_dict = {k[len('model.GT.'):]: v for k, v in state_dict.items() if k.startswith('model.GT.')}
        self.GT.set_state_dict(GT_state_dict)

        checkpoint = paddle.load('/home/liuxuwei01/molecular2molecular/src/epoch-35.ckpt')
        state_dict = checkpoint['state_dict']
        linear_layer_state_dict = {k[len('model.linear_layer.'):]: v for k, v in state_dict.items() if k.startswith('model.linear_layer.')}
        self.linear_layer.set_state_dict(linear_layer_state_dict)

        checkpoint = paddle.load('/home/liuxuwei01/molecular2molecular/src/epoch-35.ckpt')
        state_dict = checkpoint['state_dict']
        transEn_state_dict = {k[len('model.transEn.'):]: v for k, v in state_dict.items() if k.startswith('model.transEn.')}
        self.transEn.set_state_dict(transEn_state_dict)

    def make_src_mask(self, src):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        return src_mask

    def forward(self, X, E, y, node_mask, conditionVec):
        assert isinstance(conditionVec, paddle.Tensor), "conditionVec should be a tensor, but got type {}".format(type(conditionVec))

        srcMask = self.make_src_mask(conditionVec).astype('float32')
        conditionVec = self.transEn(conditionVec, srcMask)
        conditionVec = conditionVec.reshape([conditionVec.shape[0], -1])
        conditionVec = self.linear_layer(conditionVec)

        y = paddle.concat([y, conditionVec], axis=1).astype('float32')

        return self.GT(X, E, y, node_mask)

"""