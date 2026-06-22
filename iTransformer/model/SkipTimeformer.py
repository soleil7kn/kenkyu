import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding


def parse_skip_rates(skip_rates):
    """
    skip_ratesを安全にlist[int]へ変換する関数。
    例:
        [2]        -> [2]
        "2"        -> [2]
        "1,2,4"    -> [1, 2, 4]
        "[1,2,4]"  -> [1, 2, 4]
    """

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

                # 例: skip=2なら
                # offset=0: x[:, 0::2, :]
                # offset=1: x[:, 1::2, :]
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
    """
    Weighted Multi-Skip Token Pooling

    Encoder後のskip表現 [B, M, L, D] に対して、
    1. padding部分を除外してmean pooling
    2. skip系列ごとにlearnable weightをかける
    """

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

        # padding部分を除外
        x = x * mask_float

        # skipごとにmean pooling
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

        # weights: [M]
        pooled = pooled * weights.view(1, M, 1)

        # [B, M, D] -> [B, M*D]
        pooled = pooled.reshape(B, M * D)

        return pooled, weights


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
        # configsにskip_ratesがなければ[2]を使う
        self.skip_rates = parse_skip_rates(
            getattr(configs, "skip_rates", [1, 2, 4])
        )

        # 例:
        # skip_rates=[2]       -> M=2
        # skip_rates=[1,2]     -> M=3
        # skip_rates=[1,2,4]   -> M=7
        self.num_skip = sum(self.skip_rates)

        # weightを使うかどうか
        # ablation用にFalseにもできる
        self.use_skip_weight = getattr(
            configs,
            "use_skip_weight",
            True
        )

        # --------------------------------
        # Normalization
        # --------------------------------
        # 純粋にSkipTimeformer系のアイデアだけを見たい場合はFalse推奨。
        # 以前の良化結果を再現したい場合はTrueでもよい。
        self.use_norm = getattr(
            configs,
            "use_norm",
            False
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
        # Transformer Encoder
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

        # padding部分を明示的に0にする
        skip_tokens = skip_tokens * skip_mask.unsqueeze(-1).float()

        # --------------------------------
        # Encoder
        # --------------------------------
        skip_tokens = skip_tokens.reshape(
            B * M,
            L,
            D
        )

        skip_mask = skip_mask.reshape(
            B * M,
            L
        )

        enc_out, attns = self.encoder(
            skip_tokens,
            attn_mask=None
        )

        # enc_out: [B*M, L, D]

        # --------------------------------
        # Reshape back
        # --------------------------------
        enc_out = enc_out.reshape(
            B,
            M,
            L,
            D
        )

        skip_mask = skip_mask.reshape(
            B,
            M,
            L
        )

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

        # 学習コード側がoutput_attention=Trueを想定している場合
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns

        return dec_out[:, -self.pred_len:, :]

    def get_skip_weights(self):
        """
        学習後にskip weightを確認する用。
        例:
            print(model.get_skip_weights())
        """

        if self.use_skip_weight:
            return torch.softmax(
                self.weighted_pooling.skip_logits.detach(),
                dim=0
            )

        return torch.ones(
            self.num_skip
        ) / self.num_skip