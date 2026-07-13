#!/bin/bash

model_name=TimeMixer

seq_len=96
pred_len=96

e_layers=2
down_sampling_layers=3
down_sampling_window=2

d_model=16
d_ff=32

learning_rate=0.01
train_epochs=10
patience=10

batch_size=16
num_workers=0

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ../dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_${seq_len}_${pred_len} \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len $pred_len \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --e_layers $e_layers \
  --d_model $d_model \
  --d_ff $d_ff \
  --n_heads 8 \
  --dropout 0.1 \
  --embed timeF \
  --freq h \
  --activation gelu \
  --factor 1 \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --patience $patience \
  --batch_size $batch_size \
  --num_workers $num_workers \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window \
  --channel_independence 1 \
  --decomp_method moving_avg \
  --moving_avg 25 \
  --top_k 5 \
  --use_norm 1 \
  --use_future_temporal_feature 0 \
  --des Exp \
  --itr 1