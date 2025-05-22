#!/usr/bin/env bash

model_name=TimeXer
experiment_id="EV_TimeXer_120_24_v2_larger_model"
description="TimeXer_LargerModel_d256_e3_LR5e-5"

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
  --d_model 256 `# Increased` \
  --n_heads 8 \
  --e_layers 3  `# Increased` \
  --d_ff 1024   `# Increased (4*d_model)` \
  --dropout 0.15 `# Slightly increased dropout for larger model` \
  --activation 'gelu' \
  --factor 3 \
  --embed 'fixed' \
  --freq 'h' \
  --use_norm 1 \
  --enc_in 14 \
  --dec_in 14 \
  --c_out 1 \
  --des "$description" \
  --batch_size 32 \
  --learning_rate 0.00005 `# Decreased LR` \
  --lradj 'cosine' \
  --patience 7 `# Slightly more patience for larger model` \
  --train_epochs 60 `# Potentially more epochs` \
  --itr 1 \
  --use_gpu 1 `# Set to 0 for CPU` \
  --gpu 0

echo "--- Experiment $experiment_id Finished ---"