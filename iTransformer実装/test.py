import torch
from argparse import Namespace

from model.SkipTimeformer import Model

configs = Namespace(
    pred_len=96,
    output_attention=False,

    enc_in=7,
    dec_in=7,
    c_out=7,

    d_model=512,
    n_heads=8,
    e_layers=2,
    d_layers=1,
    d_ff=2048,

    factor=1,
    dropout=0.1,
    embed='timeF',
    freq='h',
    activation='gelu'
)

model = Model(configs)

B = 4
seq_len = 96

x_enc = torch.randn(B, seq_len, 7)
x_mark_enc = torch.randn(B, seq_len, 4)

x_dec = torch.randn(B, 96, 7)
x_mark_dec = torch.randn(B, 96, 4)

y = model(
    x_enc,
    x_mark_enc,
    x_dec,
    x_mark_dec
)

print(y.shape)
