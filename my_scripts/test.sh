# To run on CPU, change default for argument --use-gpu in run.py

model_name=TimeXer
experiment_id="test" # Unique ID for this run
description="TimeXer_test"

# --- Define parameters needed by TimeXer.py __init__ ---
# These might have defaults in run.py, but it's safer to be explicit if unsure.

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./my_data/      `# MODIFIED: Path to your data directory` \
  --data_path inesctec_occupancy_ts.csv       `# MODIFIED: Your CSV file name` \
  --model_id $experiment_id      `# MODIFIED: Your experiment ID` \
  --model $model_name \
  --data custom \
  --features MS \
  --target n_active_sessions_grid             `# ADDED: Specify your target column name` \
  --seq_len 72                   `# MODIFIED: Your sequence length` \
  --label_len 0                  `# MODIFIED: Your label length (TimeXer shown doesn't use x_dec)` \
  --pred_len 24                  `# MODIFIED: Your prediction length` \
  \
  `# --- TimeXer specific parameters (from TimeXer.py __init__) ---` \
  --patch_len 12                 `# REQUIRED for TimeXer (e.g., 72/12=6)` \
  --d_model 64                  `# REQUIRED (Example value)` \
  --n_heads 8                    `# REQUIRED (Example value)` \
  --e_layers 2                   `# REQUIRED (Example value, bash used 1 or 3)` \
  --d_ff 128                     `# REQUIRED (Example value)` \
  --dropout 0.1                  `# REQUIRED (Example value)` \
  --activation 'gelu'            `# REQUIRED (Example value)` \
  --factor 3                     `# REQUIRED (Example value, bash used 3)` \
  --embed 'fixed'                `# REQUIRED for DataEmbedding_inverted` \
  --freq 'h'                     `# REQUIRED for DataEmbedding_inverted (your data frequency)` \
  --use_norm 1                `# REQUIRED by TimeXer's forecast methods` \
  \
  `# --- Input/Output dimensions for YOUR data ---` \
  --enc_in 14                    `# MODIFIED: 1 (target) + 13 (exog features)` \
  --dec_in 14                    `# MODIFIED: Consistent with enc_in for MS mode data loading` \
  --c_out 1                      `# MODIFIED: 1 (target occupancy)` \
  \
  `# --- Other training parameters ---` \
  --des "$description" \
  --batch_size 32                `# MODIFIED: Your preferred batch size` \
  --learning_rate 0.0001         `# Often good for transformers` \
  --patience 5 \
  --train_epochs 1 \
  --itr 1 \
  \
