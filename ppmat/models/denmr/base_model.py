import copy
import random

import paddle
import paddle.nn as nn
from einops import rearrange
from einops import repeat
from tqdm import tqdm

from ppmat.metrics.abstract_metrics import NLL
from ppmat.metrics.abstract_metrics import SumExceptBatchKL
from ppmat.metrics.abstract_metrics import SumExceptBatchMetric
from ppmat.metrics.train_metrics import TrainLossDiscrete
from ppmat.models.denmr.diffusion_prior import DiffusionPriorNetwork
from ppmat.models.denmr.diffusion_prior import NoiseScheduler
from ppmat.models.denmr.diffusion_prior import freeze_model_and_make_eval_
from ppmat.models.denmr.encoder import Encoder
from ppmat.utils import logger
from ppmat.utils import save_load

from .graph_transformer import GraphTransformer
from .graph_transformer import GraphTransformer_C
from .noise_schedule import DiscreteUniformTransition
from .noise_schedule import MarginalUniformTransition
from .noise_schedule import PredefinedNoiseScheduleDiscrete
from .utils import diffgraphformer_utils as utils
from .utils import model_utils as m_utils
from .utils.diffprior_utils import default
from .utils.diffprior_utils import exists
from .utils.diffprior_utils import l2norm


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

        # configure general variables settings
        self.name = config["__name__"]
        self.model_dtype = paddle.get_default_dtype()
        self.T = config["diffusion_model"]["diffusion_steps"]
        self.visualization_tools = visualization_tools

        # configure datasets inter-varibles
        input_dims = dataset_infos.input_dims
        output_dims = dataset_infos.output_dims
        self.dataset_info = dataset_infos
        self.extra_features = extra_features
        self.domain_features = domain_features

        # configure noise scheduler
        self.noise_schedule = PredefinedNoiseScheduleDiscrete(
            config["diffusion_model"]["diffusion_noise_schedule"],
            timesteps=self.T,
        )

        # configure model
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

        # configure loss calculation with initialization of transition model
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
            logger.info(
                f"Marginal distribution of classes: {x_marginals.tolist()} for nodes, "
            )
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

        # configure training setting and other properties
        self.best_val_nll = 1e8
        self.val_counter = 0
        self.vocabDim = config["vocab_dim"]
        self.number_chain_steps = config["diffusion_model"]["number_chain_steps"]

        # configure generated datas for visualization after forward
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

        # configure key data container
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
        noisy_data = m_utils.apply_noise(
            self, dense_data.X, dense_data.E, other_data["y"], node_mask
        )
        extra_data = m_utils.compute_extra_data(self, noisy_data)

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

        metric = self.train_metrics(
            masked_pred_X=pred.X,
            masked_pred_E=pred.E,
            true_X=X,
            true_E=E,
            log=True,
        )
        return loss, metric

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
        noisy_data = m_utils.apply_noise(
            self, dense_data.X, dense_data.E, other_data["y"], node_mask
        )
        extra_data = m_utils.compute_extra_data(self, noisy_data)

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
        self.con_input_dim["y"] = 1024
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
        conditionVecGraph, conditionVecNmr = self.forward_(
            node_mask, X, E, conditionAll
        )

        V1_f = (
            conditionVecGraph  # Assuming V1 is a feature obtained from molecular graph
        )
        V2_f = conditionVecNmr  # Assume V2 is a feature obtained from NMR text

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

        self.config = config
        self.model = model
        self.clip = clip

        self.sample_timesteps = default(config["sample_timesteps"], timesteps)
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

        self.cond_drop_prob = default(config["cond_drop_prob"], cond_drop_prob)
        self.text_cond_drop_prob = default(
            config["text_cond_drop_prob"], self.cond_drop_prob
        )
        self.graph_cond_drop_prob = default(
            config["graph_cond_drop_prob"], self.cond_drop_prob
        )

        self.can_classifier_guidance = (
            self.text_cond_drop_prob > 0.0 and self.graph_cond_drop_prob > 0.0
        )
        self.condition_on_text_encodings = default(
            config["condition_on_text_encodings"], condition_on_text_encodings
        )

        self.predict_x_start = config["predict_x_start"]
        self.predict_v = config["predict_v"]

        self.graph_embed_scale = default(
            graph_embed_scale, config["graph_embed_dim"] ** 0.5
        )

        self.sampling_clamp_l2norm = config["sampling_clamp_l2norm"]
        self.sampling_final_clamp_l2norm = config["sampling_final_clamp_l2norm"]

        self.training_clamp_l2norm = config["training_clamp_l2norm"]
        self.init_graph_embed_l2norm = config["init_graph_embed_l2norm"]

        # TODO: maybe could remove this dummy buffer
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

        self.name = config["__name__"]
        self.model_dtype = paddle.get_default_dtype()
        self.T = config["graph_decoder"]["diffusion_model"]["diffusion_steps"]
        self.visualization_tools = visualization_tools

        input_dims = dataset_infos.input_dims
        output_dims = dataset_infos.output_dims
        self.dataset_info = dataset_infos
        self.extra_features = extra_features
        self.domain_features = domain_features

        self.noise_schedule = PredefinedNoiseScheduleDiscrete(
            config["graph_decoder"]["diffusion_model"]["diffusion_noise_schedule"],
            timesteps=self.T,
        )

        self.add_condition = True

        # set nmr encoder model
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

        # set nmr encoder projector head model
        state_dict = paddle.load(config["nmr_encoder"]["pretrained_path"])
        if config["nmr_encoder"]["projector"]["__name__"] == "Linear":
            self.encoder_projector = paddle.nn.Linear(
                in_features=config["nmr_encoder"]["max_len"]
                * config["nmr_encoder"]["d_model"],
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

        # set graph decoder model
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

        # set connector model
        self.connector_flag = False
        if config.get("connector") and config["connector"]["__name__"] == "DiffPrior":
            self.connector_flag = True
            self.connector = DiffusionPriorModel(
                config=config["connector"],
                model=DiffusionPriorNetwork(
                    dim=config["connector"]["prior_network"]["dim"],
                    num_timesteps=config["connector"]["prior_network"]["num_timesteps"],
                    num_time_embeds=config["connector"]["prior_network"][
                        "num_time_embeds"
                    ],
                    num_graph_embeds=config["connector"]["prior_network"][
                        "num_graph_embeds"
                    ],
                    num_text_embeds=config["connector"]["prior_network"][
                        "num_text_embeds"
                    ],
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
            self.connector = nn.Identity()

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
            logger.info(
                f"Marginal distribution of classes: {x_marginals.tolist()} for nodes, "
            )
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

        self.best_val_nll = 1e8
        self.val_counter = 0
        self.vocabDim = config["graph_decoder"]["vocab_dim"]
        self.number_chain_steps = config["graph_decoder"]["diffusion_model"][
            "number_chain_steps"
        ]

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
