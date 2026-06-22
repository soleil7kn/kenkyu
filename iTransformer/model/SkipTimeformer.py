import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding


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


class MultiSkipEmbedding(nn.Module):
    """
    Multi-skip Sequence Token Embedding

    Input:
        x: [B, T, D]

    Output:
        z:    [B, M, L, D]
        mask: [B, M, L]

    M = sum(skip_rates)
    L = max subsequence length after padding
    """

    def __init__(self, skip_rates=[2]):
        super().__init__()
        self.skip_rates = skip_rates

    def forward(self, x):
        B, T, D = x.shape

        outputs = []
        masks = []

        max_len = max(
            (T + s - 1) // s
            for s in self.skip_rates
        )

        for skip in self.skip_rates:
            for offset in range(skip):

                seq = x[:, offset::skip, :]
                valid_len = seq.shape[1]

                mask = torch.ones(
                    B,
                    valid_len,
                    device=x.device,
                    dtype=torch.bool
                )

                if valid_len < max_len:
                    pad_len = max_len - valid_len

                    pad_seq = torch.zeros(
                        B,
                        pad_len,
                        D,
                        device=x.device,
                        dtype=x.dtype
                    )

                    pad_mask = torch.zeros(
                        B,
                        pad_len,
                        device=x.device,
                        dtype=torch.bool
                    )

                    seq = torch.cat([seq, pad_seq], dim=1)
                    mask = torch.cat([mask, pad_mask], dim=1)

                outputs.append(seq)
                masks.append(mask)

        z = torch.stack(outputs, dim=1)       # [B, M, L, D]
        mask = torch.stack(masks, dim=1)      # [B, M, L]

        return z, mask


class STICLN(nn.Module):
    """
    Simplified Skip-Time Interaction Conditional Layer Normalization

    h:
        original sequence representation
        [B, T, D]

    s:
        skip-time representation after MSA
        [B, M, L, D]

    skip_mask:
        valid mask for skip tokens
        [B, M, L]

    skip_weights:
        optional learnable skip weights
        [M]
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

        # Stable initialization:
        # delta_gamma = 0, delta_beta = 0 at the beginning.
        nn.init.zeros_(self.cond_proj[-1].weight)
        nn.init.zeros_(self.cond_proj[-1].bias)

        self.base_gamma = nn.Parameter(
            torch.ones(d_model)
        )

        self.base_beta = nn.Parameter(
            torch.zeros(d_model)
        )

    def forward(self, h, s, skip_mask, skip_weights=None):
        # h: [B, T, D]
        # s: [B, M, L, D]
        # skip_mask: [B, M, L]
        # skip_weights: [M] or None

        B, T, D = h.shape

        mask_float = skip_mask.unsqueeze(-1).float()
        # [B, M, L, 1]

        if skip_weights is not None:
            weight = skip_weights.view(1, -1, 1, 1)
            weighted_mask = mask_float * weight
        else:
            weighted_mask = mask_float

        # Aggregate multi-skip information into one condition vector.
        cond = (s * weighted_mask).sum(dim=(1, 2)) / weighted_mask.sum(dim=(1, 2)).clamp_min(1.0)
        # cond: [B, D]

        gamma_beta = self.cond_proj(cond)
        delta_gamma, delta_beta = gamma_beta.chunk(2, dim=-1)
        # [B, D], [B, D]

        gamma = self.base_gamma.view(1, 1, D) + delta_gamma.unsqueeze(1)
        beta = self.base_beta.view(1, 1, D) + delta_beta.unsqueeze(1)

        h_norm = self.norm(h)

        out = gamma * h_norm + beta

        return out


def build_encoder(configs, num_layers=None):
    if num_layers is None:
        num_layers = configs.e_layers

    num_layers = max(1, int(num_layers))

    return Encoder(
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
            for _ in range(num_layers)
        ],
        norm_layer=nn.LayerNorm(configs.d_model)
    )


class Model(nn.Module):
    """
    Skip-Timeformer-like architecture based on Figure 3.

    Main flow:
        Input
        -> Embedding
        -> Multi-skip Sequence Token Embedding
        -> MSA over all multi-skip tokens
        -> STICLN(h_prev, skip_out, skip_mask)
        -> Dropout
        -> Projection
        -> Output

    Note:
        WeightedMSTPooling is removed.
        Learnable skip weights are preserved as an extension and are used
        when constructing the STICLN condition vector.
    """

    def __init__(self, configs):
        super().__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        self.enc_in = configs.enc_in
        self.c_out = configs.c_out

        # --------------------------------
        # Skip setting
        # --------------------------------
        self.skip_rates = parse_skip_rates(
            getattr(configs, "skip_rates", [2])
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

        # --------------------------------
        # Optional input normalization
        # --------------------------------
        self.use_norm = bool(
            getattr(configs, "use_norm", 1)
        )

        # --------------------------------
        # Embedding
        # --------------------------------
        self.enc_embedding = DataEmbedding(
            configs.enc_in,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout
        )

        # --------------------------------
        # Multi-skip token embedding
        # --------------------------------
        self.multi_skip = MultiSkipEmbedding(
            skip_rates=self.skip_rates
        )

        # --------------------------------
        # MSA over all multi-skip tokens
        # --------------------------------
        # This corresponds to applying MSA to multi-skip sequence token embedding.
        # [B, M, L, D] -> [B, M*L, D] -> MSA -> [B, M, L, D]
        skip_msa_layers = int(
            getattr(
                configs,
                "skip_msa_layers",
                getattr(configs, "skip_interaction_layers", configs.e_layers)
            )
        )

        if skip_msa_layers <= 0:
            skip_msa_layers = configs.e_layers

        self.skip_msa = build_encoder(
            configs,
            num_layers=skip_msa_layers
        )

        # --------------------------------
        # STICLN
        # --------------------------------
        self.sticln = STICLN(
            d_model=configs.d_model,
            dropout=configs.dropout
        )

        # --------------------------------
        # Dropout -> Projection
        # --------------------------------
        self.dropout_layer = nn.Dropout(configs.dropout)

        # Project the original sequence length to prediction length.
        self.time_projection = nn.Linear(
            configs.seq_len,
            configs.pred_len
        )

        # Project latent dimension to output variables.
        self.output_projection = nn.Linear(
            configs.d_model,
            configs.c_out
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

    def print_skip_weights(self):
        weights = self.get_skip_weights().detach().cpu()

        idx = 0
        for skip in self.skip_rates:
            for offset in range(skip):
                print(
                    f"skip={skip}, offset={offset}: {weights[idx].item():.4f}"
                )
                idx += 1

    def forecast(
        self,
        x_enc,
        x_mark_enc,
        x_dec,
        x_mark_dec
    ):

        # --------------------------------
        # Optional Normalization
        # --------------------------------
        if self.use_norm:
            means = x_enc.mean(
                dim=1,
                keepdim=True
            ).detach()

            x_enc = x_enc - means

            stdev = torch.sqrt(
                torch.var(
                    x_enc,
                    dim=1,
                    keepdim=True,
                    unbiased=False
                ) + 1e-5
            )

            x_enc = x_enc / stdev

        # --------------------------------
        # Original Embedding
        # --------------------------------
        h_prev = self.enc_embedding(
            x_enc,
            x_mark_enc
        )
        # h_prev: [B, T, D]

        B, T, D = h_prev.shape

        # --------------------------------
        # Multi-skip Sequence Token Embedding
        # --------------------------------
        skip_tokens, skip_mask = self.multi_skip(
            h_prev
        )

        # skip_tokens: [B, M, L, D]
        # skip_mask  : [B, M, L]

        B, M, L, D = skip_tokens.shape

        skip_tokens = skip_tokens * skip_mask.unsqueeze(-1).float()

        # --------------------------------
        # MSA over all multi-skip tokens
        # --------------------------------
        skip_tokens_flat = skip_tokens.reshape(
            B,
            M * L,
            D
        )

        skip_mask_flat = skip_mask.reshape(
            B,
            M * L
        )

        skip_tokens_flat = skip_tokens_flat * skip_mask_flat.unsqueeze(-1).float()

        skip_out_flat, skip_attns = self.skip_msa(
            skip_tokens_flat,
            attn_mask=None
        )

        skip_out_flat = skip_out_flat * skip_mask_flat.unsqueeze(-1).float()

        skip_out = skip_out_flat.reshape(
            B,
            M,
            L,
            D
        )

        # --------------------------------
        # Learnable skip weights
        # --------------------------------
        if self.use_skip_weight:
            skip_weights = torch.softmax(
                self.skip_logits,
                dim=0
            )
        else:
            skip_weights = None

        # --------------------------------
        # STICLN as the main path
        # --------------------------------
        h = self.sticln(
            h_prev,
            skip_out,
            skip_mask,
            skip_weights=skip_weights
        )
        # h: [B, T, D]

        # --------------------------------
        # Dropout -> Projection
        # --------------------------------
        h = self.dropout_layer(h)

        # [B, T, D] -> [B, D, T]
        h = h.transpose(1, 2)

        # [B, D, T] -> [B, D, pred_len]
        out = self.time_projection(h)

        # [B, D, pred_len] -> [B, pred_len, D]
        out = out.transpose(1, 2)

        # [B, pred_len, D] -> [B, pred_len, c_out]
        out = self.output_projection(out)

        # --------------------------------
        # Optional De-normalization
        # --------------------------------
        if self.use_norm:
            out = out * (
                stdev[:, 0, :self.c_out]
                .unsqueeze(1)
                .repeat(1, self.pred_len, 1)
            )

            out = out + (
                means[:, 0, :self.c_out]
                .unsqueeze(1)
                .repeat(1, self.pred_len, 1)
            )

        attns = {
            "skip_attns": skip_attns
        }

        if self.use_skip_weight:
            weights = torch.softmax(
                self.skip_logits.detach(),
                dim=0
            )
        else:
            weights = torch.ones(
                self.num_skip,
                device=out.device
            ) / self.num_skip

        return out, attns, weights

    def forward(
        self,
        x_enc,
        x_mark_enc,
        x_dec,
        x_mark_dec,
        mask=None
    ):

        dec_out, attns, weights = self.forecast(
            x_enc,
            x_mark_enc,
            x_dec,
            x_mark_dec
        )

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns

        return dec_out[:, -self.pred_len:, :]
