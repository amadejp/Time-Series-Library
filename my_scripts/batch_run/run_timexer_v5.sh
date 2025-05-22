#!/usr/bin/env bash

model_name=TimeXer
experiment_id="EV_TimeXer_120_24_v5_strong_reg"
description="TimeXer_BaselineArch_Dropout0.2_BS16_LR1e-4"

echo "--- Running Experiment: $experiment_id ---"

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./my_data/ \
  --data_path inesctec_occupancy_ts.csv \
  --model_id $experiment_id \
  --model $model_name \
  --data custom \
  --features MS \
  --target n_active_sessions_grid \
  --seq_len 120 \
  --label_len 0 \
  --pred_len 24 \
  --patch_len 12 \
  --d_model 128 \
  --n_heads 8 \
  --e_layers 2 \
  --d_ff 512 \
  --dropout 0.25  `# Increased dropout` \
  --activation 'gelu' \
  --factor 3 \
  --embed 'fixed' \
  --freq 'h' \
  --use_norm 1 \
  --enc_in 14 \
  --dec_in 14 \
  --c_out 1 \
  --des "$description" \
  --batch_size 16 `# Smaller batch size` \
  --learning_rate 0.0001 \
  --lradj 'cosine' \
  --patience 5 \
  --train_epochs 50 \
  --itr 1 \
  --use_gpu 1 `# Set to 0 for CPU` \
  --gpu 0

echo "--- Experiment $experiment_id Finished ---"