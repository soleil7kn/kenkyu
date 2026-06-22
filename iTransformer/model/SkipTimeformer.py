import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding


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

                seq = x[:, offset::skip, :]   # [B, L_i, D]
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


class Model(nn.Module):

    def __init__(self, configs):

        super().__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        self.enc_in = configs.enc_in
        self.c_out = configs.c_out

        # --------------------------------
        # Embedding
        # --------------------------------
        # iTransformerのDataEmbedding_invertedではなく、
        # 通常の時系列方向Embeddingを使う
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
        # Multi-Skip Setting
        # --------------------------------
        self.skip_rates = [2]

        # step=[2] の場合、
        # offset=0 と offset=1 の2系列
        self.num_skip = sum(self.skip_rates)

        self.multi_skip = MultiSkipEmbedding(
            skip_rates=self.skip_rates
        )

        # skipごとの学習可能重み
        self.skip_logits = nn.Parameter(
            torch.zeros(self.num_skip)
        )

        # --------------------------------
        # Prediction Head
        # --------------------------------
        # masked mean pooling後は [B, M, D]
        # それを [B, M*D] にして予測する
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
        # Embedding
        # --------------------------------
        # use_normは入れない
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

        # padding部分を0にする
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
        # Reshape
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
        # Masked Mean Pooling
        # --------------------------------
        mask_float = skip_mask.unsqueeze(-1).float()
        # [B, M, L, 1]

        enc_out = enc_out * mask_float

        pooled = enc_out.sum(dim=2) / mask_float.sum(dim=2).clamp_min(1.0)

        # pooled: [B, M, D]

        # --------------------------------
        # Learnable Skip Weight
        # --------------------------------
        weights = torch.softmax(
            self.skip_logits,
            dim=0
        ).view(1, M, 1)

        pooled = pooled * weights

        # [B, M, D] -> [B, M*D]
        pooled = pooled.reshape(
            B,
            M * D
        )

        # --------------------------------
        # Prediction
        # --------------------------------
        out = self.head(pooled)

        out = out.view(
            B,
            self.pred_len,
            self.c_out
        )

        return out, attns

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
        else:
            return dec_out[:, -self.pred_len:, :]