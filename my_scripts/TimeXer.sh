# Ensure you are in the root directory of the cloned Time-Series-Library repository
# where run.py is located.

# To run on CPU, you generally don't need to set CUDA_VISIBLE_DEVICES.
# If PyTorch detects a GPU and you want to force CPU, you might need to
# add --use_gpu False to the python command if run.py supports it,
# or ensure your PyTorch installation is CPU-only.
# However, run.py in TSlib usually has --use_gpu True as default and
# a --gpu <id> argument. To force CPU, explicitly add --use_gpu 0 (or False if allowed).

model_name=TimeXer
experiment_id="EV_TimeXer_72_24" # Unique ID for this run
description="TimeXer_EV_Occupancy_Forecast"

# --- CRITICAL: Define ALL parameters needed by TimeXer.py __init__ ---
# These might have defaults in run.py, but it's safer to be explicit if unsure.
# Refer to the TimeXer.py __init__ and the previous discussion for the full list.

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./my_ev_data/      `# MODIFIED: Path to your data directory` \
  --data_path all_data.csv       `# MODIFIED: Your CSV file name` \
  --model_id $experiment_id      `# MODIFIED: Your experiment ID` \
  --model $model_name \
  --data custom \
  --features MS \
  --target occupancy             `# ADDED: Specify your target column name` \
  --seq_len 72                   `# MODIFIED: Your sequence length` \
  --label_len 0                  `# MODIFIED: Your label length (TimeXer shown doesn't use x_dec)` \
  --pred_len 24                  `# MODIFIED: Your prediction length` \
  \
  `# --- TimeXer specific parameters (from TimeXer.py __init__) ---` \
  --patch_len 12                 `# REQUIRED for TimeXer (e.g., 72/12=6)` \
  --d_model 128                  `# REQUIRED (Example value)` \
  --n_heads 8                    `# REQUIRED (Example value)` \
  --e_layers 2                   `# REQUIRED (Example value, bash used 1 or 3)` \
  --d_ff 256                     `# REQUIRED (Example value)` \
  --dropout 0.1                  `# REQUIRED (Example value)` \
  --activation 'gelu'            `# REQUIRED (Example value)` \
  --factor 3                     `# REQUIRED (Example value, bash used 3)` \
  --embed 'fixed'                `# REQUIRED for DataEmbedding_inverted` \
  --freq 'h'                     `# REQUIRED for DataEmbedding_inverted (your data frequency)` \
  --use_norm True                `# REQUIRED by TimeXer's forecast methods` \
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
  --train_epochs 20 \
  --itr 1 \
  \
  `# --- To run on CPU ---` \
  --use_gpu False                `# ADDED: Explicitly tell it not to use GPU`
  # If --use_gpu False is not an option in run.py, you might need to
  # ensure your environment is CPU-only or set CUDA_VISIBLE_DEVICES="-1" (system dependent)