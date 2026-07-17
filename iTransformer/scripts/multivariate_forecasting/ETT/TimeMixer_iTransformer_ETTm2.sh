#!/bin/bash

export PYTHONHASHSEED=2023
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

model_name=TimeMixer_iTransformer

e_layers=2
down_sampling_layers=3
down_sampling_window=2

d_model=16
d_ff=32

refiner_d_model=64
refiner_n_heads=4
refiner_d_ff=128
refiner_e_layers=1

learning_rate=0.01
train_epochs=10
patience=10

batch_size=16
num_workers=0

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ../dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_96_96_TimeMixer_iTransformer \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 96 \
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
  --refiner_d_model $refiner_d_model \
  --refiner_n_heads $refiner_n_heads \
  --refiner_d_ff $refiner_d_ff \
  --refiner_e_layers $refiner_e_layers \
  --seed 2023 \
  --des Exp \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ../dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_96_192_TimeMixer_iTransformer \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 192 \
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
  --refiner_d_model $refiner_d_model \
  --refiner_n_heads $refiner_n_heads \
  --refiner_d_ff $refiner_d_ff \
  --refiner_e_layers $refiner_e_layers \
  --seed 2023 \
  --des Exp \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ../dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_96_336_TimeMixer_iTransformer \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 336 \
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
  --refiner_d_model $refiner_d_model \
  --refiner_n_heads $refiner_n_heads \
  --refiner_d_ff $refiner_d_ff \
  --refiner_e_layers $refiner_e_layers \
  --seed 2023 \
  --des Exp \
  --itr 1

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ../dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_96_720_TimeMixer_iTransformer \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 720 \
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
  --refiner_d_model $refiner_d_model \
  --refiner_n_heads $refiner_n_heads \
  --refiner_d_ff $refiner_d_ff \
  --refiner_e_layers $refiner_e_layers \
  --seed 2023 \
  --des Exp \
  --itr 1