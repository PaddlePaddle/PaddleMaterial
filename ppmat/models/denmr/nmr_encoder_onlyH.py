import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ppmat.models.denmr.embedding.nmr_embedding import C13nmr_embedding
from ppmat.models.denmr.embedding.nmr_embedding import H1nmr_embedding


class H1nmr_encoder(nn.Layer):
    def __init__(
        self,
        d_model,
        dim_feedforward,
        n_head,
        num_layers,
        drop_prob,
        peakwidthemb_num,
        integralemb_num,
    ):
        super(H1nmr_encoder, self).__init__()

        # for src padding mask
        self.num_heads = n_head

        self.embed = H1nmr_embedding(
            dim=d_model,
            drop_prob=drop_prob,
            peakwidthemb_num=peakwidthemb_num,
            integralemb_num=integralemb_num,
        )

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
        pad_mask = src_mask == 1
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
        pad_mask = src_mask == 1
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

        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.attn_pool = MaskedAttentionPool(dim=self.hidden_dim)

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

        fused_H = H_aligned
        fused_C = C_aligned

        # Apply attention pooling to each of the two modalities separately
        global_H = self.attn_pool(fused_H, mask_H)
        global_C = self.attn_pool(fused_C, mask_C)

        return global_H, global_C


class NMR_encoder_H(nn.Layer):
    def __init__(
        self,
        dim_H,
        dimff_H,
        dim_C,
        dimff_C,
        hidden_dim,
        n_head,
        num_layers,
        drop_prob,
        peakwidthemb_num,
        integralemb_num,
    ):
        super(NMR_encoder_H, self).__init__()
        self.H1nmr_encoder = H1nmr_encoder(
            d_model=dim_H,
            dim_feedforward=dimff_H,
            n_head=n_head,
            num_layers=num_layers,
            drop_prob=drop_prob,
            peakwidthemb_num=peakwidthemb_num,
            integralemb_num=integralemb_num,
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
            bi_crossattn_fusion_mode="gated",
            pool_mode="attn_pool",
            crossmodal_fusion_mode="weighted_sum",
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

        global_H, global_C = self.NMR_fusion(
            h_feat, mask_H, c_feat, mask_C
        )  # [batch, fusion_dim]

        return global_H, global_C
