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

    def __init__(self, skip_rates=[2]):
        super().__init__()
        self.skip_rates = skip_rates

    def forward(self, x):
        # x: [B, T, D]
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

        # z: [B, M, L, D]
        z = torch.stack(outputs, dim=1)

        # mask: [B, M, L]
        mask = torch.stack(masks, dim=1)

        return z, mask


class WeightedMSTPooling(nn.Module):

    def __init__(self, num_skip, d_model, use_weight=True):
        super().__init__()

        self.num_skip = num_skip
        self.d_model = d_model
        self.use_weight = use_weight

        if self.use_weight:
            self.skip_logits = nn.Parameter(
                torch.zeros(num_skip)
            )
        else:
            self.register_buffer(
                "skip_logits",
                torch.zeros(num_skip)
            )

    def forward(self, x, mask):
        # x   : [B, M, L, D]
        # mask: [B, M, L]

        B, M, L, D = x.shape

        mask_float = mask.unsqueeze(-1).float()
        # [B, M, L, 1]

        x = x * mask_float

        pooled = x.sum(dim=2) / mask_float.sum(dim=2).clamp_min(1.0)
        # pooled: [B, M, D]

        if self.use_weight:
            weights = torch.softmax(
                self.skip_logits,
                dim=0
            )
        else:
            weights = torch.ones(
                M,
                device=x.device,
                dtype=x.dtype
            ) / M

        pooled = pooled * weights.view(1, M, 1)

        # [B, M, D] -> [B, M*D]
        pooled = pooled.reshape(B, M * D)

        return pooled, weights


class SkipTimeInteraction(nn.Module):
    """
    Skip-Time Interaction

    入力:
        x   : [B, M, L, D]
        mask: [B, M, L]

    処理:
        各時刻位置 L ごとに、M方向へAttentionする。
        つまり、skip系列同士を相互作用させる。

    変形:
        [B, M, L, D]
        -> [B, L, M, D]
        -> [B*L, M, D]
        -> Encoder
        -> [B, M, L, D]
    """

    def __init__(self, configs, num_layers=1):
        super().__init__()

        self.interaction_encoder = Encoder(
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

    def forward(self, x, mask):

        # x   : [B, M, L, D]
        # mask: [B, M, L]

        B, M, L, D = x.shape

        mask_float = mask.unsqueeze(-1).float()
        x = x * mask_float

        # [B, M, L, D] -> [B, L, M, D]
        x_inter = x.permute(0, 2, 1, 3).contiguous()

        # [B, M, L] -> [B, L, M]
        mask_inter = mask.permute(0, 2, 1).contiguous()

        # [B, L, M, D] -> [B*L, M, D]
        x_inter = x_inter.reshape(
            B * L,
            M,
            D
        )

        # [B, L, M] -> [B*L, M]
        mask_inter = mask_inter.reshape(
            B * L,
            M
        )

        # padding部分を0にする
        x_inter = x_inter * mask_inter.unsqueeze(-1).float()

        # skip方向のAttention
        x_inter, attns = self.interaction_encoder(
            x_inter,
            attn_mask=None
        )

        # padding部分を再度0にする
        x_inter = x_inter * mask_inter.unsqueeze(-1).float()

        # [B*L, M, D] -> [B, L, M, D]
        x_inter = x_inter.reshape(
            B,
            L,
            M,
            D
        )

        # [B, L, M, D] -> [B, M, L, D]
        x_inter = x_inter.permute(0, 2, 1, 3).contiguous()

        return x_inter, attns


class Model(nn.Module):

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

        self.use_skip_interaction = bool(
            getattr(configs, "use_skip_interaction", 1)
        )

        self.skip_interaction_layers = int(
            getattr(configs, "skip_interaction_layers", 1)
        )

        # --------------------------------
        # Normalization
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
        # Temporal Encoder
        # --------------------------------
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

        # --------------------------------
        # Multi-Skip Token
        # --------------------------------
        self.multi_skip = MultiSkipEmbedding(
            skip_rates=self.skip_rates
        )

        # --------------------------------
        # Skip-Time Interaction
        # --------------------------------
        if self.use_skip_interaction:
            self.skip_interaction = SkipTimeInteraction(
                configs,
                num_layers=self.skip_interaction_layers
            )

            # いきなりSTIFを強く入れると性能が崩れる可能性があるため、
            # 小さいゲートから始める
            self.stif_gate = nn.Parameter(
                torch.tensor(-2.0)
            )

        # --------------------------------
        # Weighted MST Pooling
        # --------------------------------
        self.weighted_pooling = WeightedMSTPooling(
            num_skip=self.num_skip,
            d_model=configs.d_model,
            use_weight=self.use_skip_weight
        )

        # --------------------------------
        # Prediction Head
        # --------------------------------
        self.head = nn.Linear(
            self.num_skip * configs.d_model,
            self.pred_len * configs.c_out
        )

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
        # Embedding
        # --------------------------------
        enc_out = self.enc_embedding(
            x_enc,
            x_mark_enc
        )

        # enc_out: [B, T, D]

        # --------------------------------
        # Multi-Skip Token
        # --------------------------------
        skip_tokens, skip_mask = self.multi_skip(
            enc_out
        )

        # skip_tokens: [B, M, L, D]
        # skip_mask  : [B, M, L]

        B, M, L, D = skip_tokens.shape

        skip_tokens = skip_tokens * skip_mask.unsqueeze(-1).float()

        # --------------------------------
        # Temporal Encoder
        # --------------------------------
        skip_tokens = skip_tokens.reshape(
            B * M,
            L,
            D
        )

        skip_mask_flat = skip_mask.reshape(
            B * M,
            L
        )

        enc_out, temporal_attns = self.encoder(
            skip_tokens,
            attn_mask=None
        )

        # enc_out: [B*M, L, D]

        enc_out = enc_out.reshape(
            B,
            M,
            L,
            D
        )

        skip_mask = skip_mask_flat.reshape(
            B,
            M,
            L
        )

        enc_out = enc_out * skip_mask.unsqueeze(-1).float()

        # --------------------------------
        # Skip-Time Interaction
        # --------------------------------
        stif_attns = None

        if self.use_skip_interaction:

            stif_out, stif_attns = self.skip_interaction(
                enc_out,
                skip_mask
            )

            # gateは初期値 sigmoid(-2.0) ≒ 0.119
            # つまり最初はWeighted MSTに近く、
            # 学習が進むとSTIFの寄与を増やせる
            gate = torch.sigmoid(self.stif_gate)

            enc_out = enc_out + gate * (stif_out - enc_out)

            enc_out = enc_out * skip_mask.unsqueeze(-1).float()

        # --------------------------------
        # Weighted MST Pooling
        # --------------------------------
        pooled, weights = self.weighted_pooling(
            enc_out,
            skip_mask
        )

        # pooled: [B, M*D]
        # weights: [M]

        # --------------------------------
        # Prediction
        # --------------------------------
        out = self.head(pooled)

        out = out.view(
            B,
            self.pred_len,
            self.c_out
        )

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
            "temporal_attns": temporal_attns,
            "stif_attns": stif_attns
        }

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

    def get_skip_weights(self):

        if self.use_skip_weight:
            return torch.softmax(
                self.weighted_pooling.skip_logits.detach(),
                dim=0
            )

        return torch.ones(
            self.num_skip,
            device=self.head.weight.device
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

    def get_stif_gate(self):

        if self.use_skip_interaction:
            return torch.sigmoid(
                self.stif_gate.detach()
            )

        return None