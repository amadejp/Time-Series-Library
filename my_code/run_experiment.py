import torch
import os
import numpy as np
from types import SimpleNamespace # To create an args-like object

# --- Make sure the library is importable ---
# Add the root directory of the Time-Series-Library to sys.path if needed
# import sys
# sys.path.append('/path/to/your/Time-Series-Library')

# --- Import the necessary Experiment class ---
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
# Import models you might want to use IF they are not automatically registered in Exp_Basic's model_dict
# Usually not needed as Exp_Basic handles model loading based on args.model string
# from models import TimesNet, DLinear # etc.

# --- 1. Define Configuration Manually ---
# Create an object similar to what argparse would produce
args = SimpleNamespace(
    # Task and Model Definition
    task_name='long_term_forecast',
    is_training=1,  # 1 for training, 0 for testing only
    model='TimesNet', # Change to DLinear, PatchTST, etc.
    model_id='Occupancy_TimesNet_72_24_Script', # Custom ID for this run

    # Data Loader
    data='custom', # Essential for using custom train/val/test files
    root_path='../tslib_data/', # Path to the directory containing train.csv, val.csv, test.csv
    data_path='', # Not needed for 'custom' if files are named train/val/test.csv
    features='M', # M: multivariate, S: univariate, MS: multivariate -> univariate
    target='n_active_sessions_grid', # Name of the target column in your CSVs
    freq='h', # Frequency ('h' for hourly)
    checkpoints='../checkpoints/', # Directory to save model checkpoints

    # Forecasting Task Config
    seq_len=72,     # Input sequence length (window_size)
    label_len=36,   # Start token length (often seq_len // 2) - adjust if needed
    pred_len=24,    # Prediction horizon

    # Model Specific Parameters
    # --- UPDATE THESE based on your data and chosen model ---
    enc_in=12,      # Number of features (target + static = 1 + 11 = 12) - VERIFY THIS!
    dec_in=12,      # Usually same as enc_in for M features
    c_out=12,       # Usually same as enc_in for M features
    # --- Common Model Hyperparameters (Defaults from your previous script) ---
    d_model=64,     # Model dimension (e.g., for TimesNet, PatchTST)
    n_heads=8,      # Number of attention heads (for Transformer-based models)
    e_layers=2,     # Number of encoder layers
    d_layers=1,     # Number of decoder layers
    d_ff=128,       # Dimension of feed-forward network (e.g., for TimesNet)
    moving_avg=25,  # Window size for moving average (used in some models like DLinear)
    factor=3,       # Attention factor (e.g., for Autoformer)
    distil=True,    # Whether to use distillation (e.g., for Autoformer)
    dropout=0.1,
    embed='timeF',  # Type of time embedding
    activation='gelu',
    output_attention=False,
    # --- Model-Specific (Examples - check if relevant for your chosen model) ---
    top_k=5,        # For TimesNet TimesBlock
    num_kernels=6,  # For Inception block (if used)

    # Optimization
    num_workers=4,  # Adjust based on your system CPU cores
    itr=1,          # Number of experiment repetitions
    train_epochs=10,# Number of training epochs
    batch_size=32,
    patience=3,     # Early stopping patience
    learning_rate=0.001,
    des='Exp_From_Script', # Description for logging/saving purposes
    loss='mse',     # Loss function ('mse' or 'mae')
    lradj='type1',  # Learning rate adjustment strategy
    use_amp=False,  # Use Automatic Mixed Precision (requires compatible GPU/CUDA)

    # GPU Settings
    use_gpu=True,   # Set to True to try using GPU
    gpu=0,          # ID of the GPU to use
    use_multi_gpu=False,
    devices='0,1',  # Ignored if use_multi_gpu is False

    # Test specific (less relevant if is_training=1)
    inverse=True,  # Perform inverse scaling on results (if scaler was used)
    use_dtw=False, # Calculate DTW metric (can be slow) - Set to True if desired
)

# --- 2. Dynamic GPU Setup (Optional but Recommended) ---
args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
if args.use_gpu:
    if args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
        print(f"Configured for Multi-GPU: {args.device_ids}")
    else:
         print(f"Configured for GPU: {args.gpu}")
         torch.cuda.set_device(args.gpu) # Explicitly set the device
else:
    print("Configured for CPU.")


# --- 3. Instantiate and Run Experiment ---
print("Initializing Experiment...")
# Instantiate the experiment class with the configured arguments
exp = Exp_Long_Term_Forecast(args)

# Create the setting string for results/checkpoint paths
setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
    args.task_name,
    args.model_id,
    args.model,
    args.features,
    args.seq_len,
    args.label_len,
    args.pred_len,
    args.d_model,
    args.n_heads,
    args.e_layers,
    args.d_layers,
    args.d_ff,
    args.factor, # Adjust based on relevance to the model
    args.embed,
    args.distil, # Adjust based on relevance to the model
    args.des, 0 # Using 0 for the iteration number as we run only once here
)
print(f"Experiment Setting ID: {setting}")

if args.is_training:
    print('>>>>>>> Start Training <<<<<<<<<<<<<<<<<<<<<<<<<<')
    exp.train(setting)
    print('>>>>>>> Training Finished <<<<<<<<<<<<<<<<<<<<<<<<<<')

print('>>>>>>> Start Testing <<<<<<<<<<<<<<<<<<<<<<<<<<')
exp.test(setting) # test function loads the best model from training automatically
print('>>>>>>> Testing Finished <<<<<<<<<<<<<<<<<<<<<<<<<<')

# Optional: Release GPU memory if needed
if args.use_gpu:
    torch.cuda.empty_cache()

print("--- Script Finished ---")