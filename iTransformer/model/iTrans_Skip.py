import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
import numpy as np

def parse_skip_rates(skip_rates):
    if isinstance(skip_rates, list):
        return [int(s) for s in skip_rates]

    if isinstance(skip_rates, tuple):
        return [int(s) for s in skip_rates]

    if isinstance(skip_rates, int):
        return [skip_rates]

    if isinstance(skip_rates, str):
        skip_rates = skip_rates.replace("[", "")
        skip_rates = skip_rates.replace("]", "")
        skip_rates = skip_rates.replace(" ", "")
        return [int(s) for s in skip_rates.split(",") if s != ""]

    raise ValueError(f"Unsupported skip_rates type: {type(skip_rates)}")


class MultiSkipInvertedEmbedding(nn.Module):
    """
    iTransformer用のMulti-Skip Embedding

    入力:
        x: [B, L, N]

    出力:
        skip_tokens: [B, N, M, D]
        skip_mask  : [B, M]
    """

    def __init__(self, seq_len, d_model, skip_rates=[2], dropout=0.1):
        super().__init__()

        self.seq_len = seq_len
        self.d_model = d_model
        self.skip_rates = skip_rates
        self.num_skip = sum(skip_rates)

        self.max_len = max(
            (seq_len + s - 1) // s
            for s in skip_rates
        )

        self.value_embedding = nn.Linear(self.max_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, L, N]
        B, L, N = x.shape

        outputs = []
        masks = []

        for skip in self.skip_rates:
            for offset in range(skip):

                seq = x[:, offset::skip, :]
                # [B, L_s, N]

                valid_len = seq.shape[1]

                if valid_len < self.max_len:
                    pad_len = self.max_len - valid_len

                    pad = torch.zeros(
                        B,
                        pad_len,
                        N,
                        device=x.device,
                        dtype=x.dtype
                    )

                    seq = torch.cat([seq, pad], dim=1)

                # [B, L_s, N] -> [B, N, L_s]
                seq = seq.permute(0, 2, 1)

                # [B, N, max_len] -> [B, N, D]
                token = self.value_embedding(seq)

                outputs.append(token)

                masks.append(
                    torch.ones(
                        B,
                        device=x.device,
                        dtype=torch.bool
                    )
                )

        # [B, M, N, D] -> [B, N, M, D]
        skip_tokens = torch.stack(outputs, dim=1).permute(0, 2, 1, 3)

        # [B, M]
        skip_mask = torch.stack(masks, dim=1)

        skip_tokens = self.dropout(skip_tokens)

        return skip_tokens, skip_mask
    

class InvertedSTICLN(nn.Module):
    """
    iTransformer用STICLN

    h:
        original variable tokens
        [B, N, D]

    s:
        skip variable tokens
        [B, N, M, D]

    skip_mask:
        [B, M]
    """

    def __init__(self, d_model, dropout=0.1):
        super().__init__()

        self.norm = nn.LayerNorm(
            d_model,
            elementwise_affine=False
        )

        self.cond_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 2 * d_model)
        )

        nn.init.zeros_(self.cond_proj[-1].weight)
        nn.init.zeros_(self.cond_proj[-1].bias)

        self.base_gamma = nn.Parameter(torch.ones(d_model))
        self.base_beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, h, s, skip_mask, skip_weights=None):
        # h: [B, N, D]
        # s: [B, N, M, D]
        # skip_mask: [B, M]

        B, N, D = h.shape
        M = s.shape[2]

        mask_float = skip_mask.unsqueeze(1).unsqueeze(-1).float()
        # [B, 1, M, 1]

        if skip_weights is not None:
            weight = skip_weights.view(1, 1, M, 1)
            mask_float = mask_float * weight

        # 各変数ごとにskip情報を集約
        cond = (s * mask_float).sum(dim=2) / mask_float.sum(dim=2).clamp_min(1.0)
        # [B, N, D]

        gamma_beta = self.cond_proj(cond)
        # [B, N, 2D]

        delta_gamma, delta_beta = gamma_beta.chunk(2, dim=-1)

        gamma = self.base_gamma.view(1, 1, D) + delta_gamma
        beta = self.base_beta.view(1, 1, D) + delta_beta

        h_norm = self.norm(h)

        out = gamma * h_norm + beta

        return out
    

class Model(nn.Module):
    """
    iTransformer + SkipTimeformer idea
    """

    def __init__(self, configs):
        super(Model, self).__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm

        self.enc_in = configs.enc_in
        self.c_out = configs.c_out

        # -------------------------------
        # Skip settings
        # -------------------------------
        self.skip_rates = parse_skip_rates(
            getattr(configs, "skip_rates", "2")
        )

        self.num_skip = sum(self.skip_rates)

        self.use_skip_weight = bool(
            getattr(configs, "use_skip_weight", 1)
        )

        if self.use_skip_weight:
            self.skip_logits = nn.Parameter(
                torch.zeros(self.num_skip)
            )
        else:
            self.register_buffer(
                "skip_logits",
                torch.zeros(self.num_skip)
            )

        # -------------------------------
        # Original iTransformer Embedding
        # -------------------------------
        self.enc_embedding = DataEmbedding_inverted(
            configs.seq_len,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout
        )

        self.class_strategy = configs.class_strategy

        # -------------------------------
        # Multi-Skip Inverted Embedding
        # -------------------------------
        self.multi_skip_embedding = MultiSkipInvertedEmbedding(
            seq_len=configs.seq_len,
            d_model=configs.d_model,
            skip_rates=self.skip_rates,
            dropout=configs.dropout
        )

        # -------------------------------
        # skip_msa over [B, N*M, D]
        # -------------------------------
        self.skip_msa = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=configs.output_attention
                        ),
                        configs.d_model,
                        configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                )
                for _ in range(
                    int(getattr(configs, "skip_msa_layers", 1))
                )
            ],
            norm_layer=nn.LayerNorm(configs.d_model)
        )

        # -------------------------------
        # STICLN
        # -------------------------------
        self.sticln = InvertedSTICLN(
            d_model=configs.d_model,
            dropout=configs.dropout
        )

        # -------------------------------
        # iTransformer Encoder
        # -------------------------------
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=configs.output_attention
                        ),
                        configs.d_model,
                        configs.n_heads
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model)
        )

        self.projector = nn.Linear(
            configs.d_model,
            configs.pred_len,
            bias=True
        )

    def get_skip_weights(self):
        if self.use_skip_weight:
            return torch.softmax(
                self.skip_logits.detach(),
                dim=0
            )

        return torch.ones(
            self.num_skip,
            device=self.skip_logits.device
        ) / self.num_skip

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):

        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means

            stdev = torch.sqrt(
                torch.var(
                    x_enc,
                    dim=1,
                    keepdim=True,
                    unbiased=False
                ) + 1e-5
            )

            x_enc /= stdev

        _, _, N = x_enc.shape
        # x_enc: [B, L, N]

        # -------------------------------
        # Original iTransformer embedding
        # -------------------------------
        enc_out = self.enc_embedding(
            x_enc,
            x_mark_enc
        )
        # enc_out: [B, N(+covariates), D]

        # 予測対象の変数tokenだけ取り出す
        h_data = enc_out[:, :N, :]
        # [B, N, D]

        h_extra = enc_out[:, N:, :]
        # timestampなどのcovariate tokenがある場合

        # -------------------------------
        # Multi-Skip Inverted Embedding
        # -------------------------------
        skip_tokens, skip_mask = self.multi_skip_embedding(
            x_enc
        )
        # skip_tokens: [B, N, M, D]
        # skip_mask  : [B, M]

        B, N, M, D = skip_tokens.shape

        # [B, N, M, D] -> [B, N*M, D]
        skip_tokens_flat = skip_tokens.reshape(
            B,
            N * M,
            D
        )

        # -------------------------------
        # skip MSA
        # -------------------------------
        skip_out, skip_attns = self.skip_msa(
            skip_tokens_flat,
            attn_mask=None
        )
        # [B, N*M, D]

        skip_out = skip_out.reshape(
            B,
            N,
            M,
            D
        )
        # [B, N, M, D]

        # -------------------------------
        # Skip weights
        # -------------------------------
        if self.use_skip_weight:
            skip_weights = torch.softmax(
                self.skip_logits,
                dim=0
            )
        else:
            skip_weights = torch.ones(
                M,
                device=x_enc.device,
                dtype=x_enc.dtype
            ) / M

        # -------------------------------
        # STICLN as main path
        # -------------------------------
        h_data = self.sticln(
            h_data,
            skip_out,
            skip_mask,
            skip_weights=skip_weights
        )
        # [B, N, D]

        # covariate tokenがある場合は戻す
        if h_extra.shape[1] > 0:
            enc_out = torch.cat(
                [h_data, h_extra],
                dim=1
            )
        else:
            enc_out = h_data

        # -------------------------------
        # iTransformer Encoder
        # -------------------------------
        enc_out, attns = self.encoder(
            enc_out,
            attn_mask=None
        )

        # -------------------------------
        # Projection
        # -------------------------------
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N]
        # [B, pred_len, N]

        if self.use_norm:
            dec_out = dec_out * (
                stdev[:, 0, :]
                .unsqueeze(1)
                .repeat(1, self.pred_len, 1)
            )

            dec_out = dec_out + (
                means[:, 0, :]
                .unsqueeze(1)
                .repeat(1, self.pred_len, 1)
            )

        all_attns = {
            "skip_attns": skip_attns,
            "itransformer_attns": attns
        }

        return dec_out, all_attns

    def forward(
        self,
        x_enc,
        x_mark_enc,
        x_dec,
        x_mark_dec,
        mask=None
    ):
        dec_out, attns = self.forecast(
            x_enc,
            x_mark_enc,
            x_dec,
            x_mark_dec
        )

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns

        return dec_out[:, -self.pred_len:, :]