import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ppmat.models.denmr.embedding.nmr_embedding import C13nmr_embedding
from ppmat.models.denmr.embedding.nmr_embedding import H1nmr_embedding


class H1nmr_encoder(nn.Layer):
    def __init__(self, d_model, dim_feedforward, n_head, num_layers, drop_prob):
        super(H1nmr_encoder, self).__init__()

        # for src padding mask
        self.num_heads = n_head

        self.embed = H1nmr_embedding(dim=d_model, drop_prob=drop_prob)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            dropout=drop_prob,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x, src_mask):
        # input format: [batch, len_peak, feat_dim]
        x_emb = self.embed(x, src_mask)

        # process for src_key_padding_mask
        pad_mask = src_mask == 0
        bsz, src_len, _ = x_emb.shape
        pad_mask = pad_mask.reshape([bsz, 1, 1, src_len]).expand(
            [-1, self.num_heads, src_len, -1]
        )

        out = self.encoder(src=x_emb, src_mask=pad_mask)
        return out


class C13nmr_encoder(nn.Layer):
    def __init__(self, d_model, dim_feedforward, n_head, num_layers, drop_prob):
        super(C13nmr_encoder, self).__init__()

        # for src padding mask
        self.num_heads = n_head

        self.embed = C13nmr_embedding(dim=d_model, drop_prob=drop_prob)
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            dropout=drop_prob,
            normalize_before=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x, src_mask):
        # input format: [batch, len_peak, feat_dim]
        x_emb = self.embed(x, src_mask)

        # process for src_key_padding_mask
        pad_mask = src_mask == 0
        bsz, src_len, _ = x_emb.shape
        pad_mask = pad_mask.reshape([bsz, 1, 1, src_len]).expand(
            [-1, self.num_heads, src_len, -1]
        )

        out = self.encoder(src=x_emb, src_mask=pad_mask)
        return out


class MaskedAttentionPool(nn.Layer):
    def __init__(self, dim):
        super(MaskedAttentionPool, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )  # 移除了Softmax，需手动处理

    def forward(self, x, mask=None):
        # x: [batch, seq_len, dim]
        # mask: [batch, seq_len] （1: valid，0: pad）
        attn_scores = self.attention(x)  # [batch, seq_len, 1]

        # add mask processing
        if mask is not None:
            # Set the attention scores at padding positions to -∞,
            # resulting in zero weight after Softmax
            attn_scores = attn_scores.masked_fill(
                mask.unsqueeze(-1) == 0, -float("inf")
            )

        attn_weights = F.softmax(attn_scores, axis=1)  # [batch, seq_len, 1]
        return (x * attn_weights).sum(axis=1)  # [batch, dim]


class NMR_fusion(nn.Layer):
    def __init__(
        self,
        dim_h=1024,
        dim_c=256,
        hidden_dim=512,
        n_head=8,
        out_dim=512,
        bi_crossattn_fusion_mode="",
        pool_mode="",
        crossmodal_fusion_mode="",
    ):
        super(NMR_fusion, self).__init__()

        # projection layer
        self.proj_h = nn.Linear(dim_h, hidden_dim)
        self.proj_c = nn.Linear(dim_c, hidden_dim)
        # Bidirectional cross-attention
        self.cross_attn_ab = nn.MultiHeadAttention(hidden_dim, num_heads=n_head)
        self.cross_attn_ba = nn.MultiHeadAttention(hidden_dim, num_heads=n_head)

        self.bi_crossattn_fusion_mode = bi_crossattn_fusion_mode
        self.pool_mode = pool_mode
        self.crossmodal_fusion = crossmodal_fusion_mode

        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.gate_linear = nn.Linear(hidden_dim, 1)
        self.attn_pool = MaskedAttentionPool(dim=self.hidden_dim)
        self.weighted_sum = nn.Linear(1024, 1)
        self.concat_linear = nn.Linear(1024, 512)

        # for src padding mask
        self.num_heads = n_head

    def masked_mean_pool(self, tensor, mask):
        # tensor: [batch, seq_len, dim]
        # mask: [batch, seq_len] (1: valid，0: pad)
        lengths = mask.sum(axis=1, keepdim=True)  # [batch, 1]
        masked = tensor * mask.unsqueeze(-1)  # zero out padding positions
        return masked.sum(axis=1) / (lengths + 1e-6)  # [batch, dim]

    def forward(self, tensor_Hnmr, mask_H, tensor_Cnmr, mask_C):

        max_len_H = mask_H.sum(axis=-1).max().item()
        mask_H = mask_H[:, : int(max_len_H)]
        tensor_Hnmr = tensor_Hnmr[:, : int(max_len_H), :]
        max_len_C = mask_C.sum(axis=-1).max().item()
        mask_C = mask_C[:, : int(max_len_C)]
        tensor_Cnmr = tensor_Cnmr[:, : int(max_len_C), :]

        # project to uniform dimension
        H_aligned = self.proj_h(tensor_Hnmr)
        C_aligned = self.proj_c(tensor_Cnmr)

        # bidirectonal cross-attention
        pad_mask_H = mask_H == 0
        bsz_H, src_len_H, _ = H_aligned.shape
        pad_mask_C = mask_C == 0
        bsz_C, src_len_C, _ = C_aligned.shape
        pad_mask_H = pad_mask_H.reshape([bsz_H, 1, 1, src_len_H]).expand(
            [-1, self.num_heads, src_len_C, -1]
        )
        pad_mask_C = pad_mask_C.reshape([bsz_C, 1, 1, src_len_C]).expand(
            [-1, self.num_heads, src_len_H, -1]
        )

        attn_H2C = self.cross_attn_ab(
            query=H_aligned,
            key=C_aligned,
            value=C_aligned,
            attn_mask=pad_mask_C,
        )
        attn_C2H = self.cross_attn_ba(
            query=C_aligned,
            key=H_aligned,
            value=H_aligned,
            attn_mask=pad_mask_H,
        )

        # combine the cross-attention output of two modalities with the origin features
        if self.bi_crossattn_fusion_mode == "concat":
            # Method 1: Concatenate outputs from two directions
            fused_H = paddle.concat(
                [H_aligned, attn_H2C], axis=-1
            )  # [batch, seq_a, 2*hidden_dim]
            fused_C = paddle.concat(
                [C_aligned, attn_C2H], axis=-1
            )  # [batch, seq_b, 2*hidden_dim]

        elif self.bi_crossattn_fusion_mode == "add":
            # Method 2: Residual connection
            fused_H = H_aligned + attn_H2C  # [batch, seq_a, hidden_dim]
            fused_C = C_aligned + attn_C2H  # [batch, seq_b, hidden_dim]

        elif self.bi_crossattn_fusion_mode == "gated":
            # Method 3: Gated Fusion (Adaptive Weights)
            gate_H = F.sigmoid(self.gate_linear(attn_H2C))  # 计算权重
            fused_H = (1 - gate_H) * H_aligned + gate_H * attn_H2C
            gate_C = F.sigmoid(self.gate_linear(attn_C2H))  # 计算权重
            fused_C = (1 - gate_C) * C_aligned + gate_C * attn_C2H

        else:
            fused_H = attn_H2C
            fused_C = attn_C2H

        # Intra-modal Aggregation
        if self.pool_mode == "mean_pool":
            # method 1：average pooling
            global_H = self.masked_mean_pool(fused_H, mask_H)

            global_C = self.masked_mean_pool(fused_C, mask_C)  # [batch, 256]

        elif self.pool_mode == "attn_pool":
            # Apply attention pooling to each of the two modalities separately
            global_H = self.attn_pool(fused_H, mask_H)
            global_C = self.attn_pool(fused_C, mask_C)  # [batch, 256]

        # cross-modal fusion
        if self.crossmodal_fusion == "concat_linear":
            merged = paddle.concat([global_H, global_C], axis=-1)  # [batch, 512]
            global_output = self.concat_linear(merged)  # 压缩到 [batch, 256]

        elif self.crossmodal_fusion == "weighted_sum":
            merged = paddle.concat([global_H, global_C], axis=-1)
            # 或者
            # merged = global_H + global_C

            gate = F.sigmoid(self.weighted_sum(merged))  # [batch,1]
            global_output = gate * global_H + (1 - gate) * global_C

        return global_output


class NMR_encoder(nn.Layer):
    def __init__(
        self, dim_H, dimff_H, dim_C, dimff_C, hidden_dim, n_head, num_layers, drop_prob
    ):
        super(NMR_encoder, self).__init__()
        self.H1nmr_encoder = H1nmr_encoder(
            d_model=dim_H,
            dim_feedforward=dimff_H,
            n_head=n_head,
            num_layers=num_layers,
            drop_prob=drop_prob,
        )

        self.C13nmr_encoder = C13nmr_encoder(
            d_model=dim_C,
            dim_feedforward=dimff_C,
            n_head=n_head,
            num_layers=num_layers,
            drop_prob=drop_prob,
        )

        self.NMR_fusion = NMR_fusion(
            dim_H,
            dim_C,
            hidden_dim,
            n_head,
            bi_crossattn_fusion_mode="add",
            pool_mode="attn_pool",
            crossmodal_fusion_mode="concat_linear",
        )

    def create_mask(self, batch_size, max_seq_len, num_peak):

        mask = paddle.zeros([batch_size, max_seq_len], dtype="float32")
        for i, length in enumerate(num_peak):
            mask[i, :length] = 1
        return mask

    def forward(self, condition):
        H1nmr, num_H_peak, C13nmr, num_C_peak = condition

        batch_size, max_seq_len_H, _ = H1nmr.shape
        mask_H = self.create_mask(batch_size, max_seq_len_H, num_H_peak)
        _, max_seq_len_C = C13nmr.shape
        mask_C = self.create_mask(batch_size, max_seq_len_C, num_C_peak)

        h_feat = self.H1nmr_encoder(H1nmr, mask_H)  # [batch, h_seq, h_dim]
        c_feat = self.C13nmr_encoder(C13nmr, mask_C)  # [batch, c_seq, c_dim]

        fused_feat = self.NMR_fusion(
            h_feat, mask_H, c_feat, mask_C
        )  # [batch, fusion_dim]

        return fused_feat
