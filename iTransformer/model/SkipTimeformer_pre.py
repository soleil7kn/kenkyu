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

        max_len = max(
            (T + s - 1) // s
            for s in self.skip_rates
        )

        for skip in self.skip_rates:

            for offset in range(skip):

                seq = x[:, offset::skip, :]

                if seq.shape[1] < max_len:

                    pad = max_len - seq.shape[1]

                    seq = F.pad(
                        seq,
                        (0, 0, 0, pad)
                    )

                outputs.append(seq)

        # [B, token_num, token_len, D]
        z = torch.stack(outputs, dim=1)

        return z


class Model(nn.Module):

    def __init__(self, configs):

        super().__init__()

        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        self.enc_in = configs.enc_in
        self.c_out = configs.c_out

        # Embedding
        self.enc_embedding = DataEmbedding(
            configs.enc_in,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout
        )

        # Transformer Encoder
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

        # Prediction Head
        self.skip_rates = [2]
 
        self.num_skip = sum(self.skip_rates)

        self.max_len = max(
            (configs.seq_len + s - 1) // s
            for s in self.skip_rates
        )
        
        self.skip_logits = nn.Parameter(
            torch.zeros(self.num_skip)
        )

        self.multi_skip = MultiSkipEmbedding(
            skip_rates=self.skip_rates
        )

        self.head = nn.Linear(
            self.num_skip * self.max_len * configs.d_model,
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

        enc_out = self.enc_embedding(
            x_enc,
            x_mark_enc
        )


        # --------------------------------
        # Multi Skip
        # --------------------------------

        skip_tokens = self.multi_skip(
            enc_out
        )

        # print("skip_tokens :", skip_tokens.shape)

        # [B, M, L, D]
        B, M, L, D = skip_tokens.shape

        skip_tokens = skip_tokens.reshape(
            B * M,
            L,
            D
        )


        # --------------------------------
        # Encoder
        # --------------------------------

        enc_out, attns = self.encoder(
            skip_tokens,
            attn_mask=None
        )


        # --------------------------------
        # Pooling
        # --------------------------------

        enc_out = enc_out.reshape(
            B,
            M,
            L,
            D
        )
        
        # skip重みを計算
        weights = torch.softmax(
            self.skip_logits,
            dim=0
        ).view(1, M, 1, 1)

        # skipごとに重み付け
        enc_out = enc_out * weights
        
        enc_out = enc_out.reshape(
            B,
            M * L * D
        )

        out = self.head(enc_out)

        out = out.view(
            B,
            self.pred_len,
            self.c_out
        )

        return out

    def forward(
        self,
        x_enc,
        x_mark_enc,
        x_dec,
        x_mark_dec,
        mask=None
    ):

        dec_out = self.forecast(
            x_enc,
            x_mark_enc,
            x_dec,
            x_mark_dec
        )

        return dec_out