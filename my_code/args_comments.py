# In run_ev_experiment.py

class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

args = DotDict()

# --- Basic Experiment settings ---
args.is_training = 1  # 1 for training, 0 for testing
args.model_id = 'EV_Occupancy_DLinear_from_CSV' # Custom name for this experiment run
args.model = 'DLinear'  # Choose model: DLinear, PatchTST, Autoformer, etc.

# --- Data Loader settings ---
args.data = 'custom'     # IMPORTANT: This tells TSlib to use Dataset_Custom for generic CSVs
args.root_path = './my_ev_data/'  # Path to the directory containing your CSV
args.data_path = 'all_data.csv'   # Name of your CSV file

args.features = 'MS'     # M: multivariate input, S: univariate target. 'MS' for multi-input, multi-output (or uni-output) + future known covariates
                         # Your input will be (occupancy + 13 exog), predicting occupancy, using future exog
args.target = 'occupancy'# Name of the target column in your CSV
args.freq = 'h'          # Frequency of data (h:hourly)
args.checkpoints = './checkpoints/' # Directory to save model checkpoints

# --- Forecasting Task settings ---
args.seq_len = 72        # Input sequence length (3 days of history)
args.label_len = 0       # Length of the "label" segment for the decoder.
                         # For DLinear/NLinear/PatchTST, 0 is common and simplest.
                         # For Transformer-based (Autoformer, Informer), often seq_len // 2.
args.pred_len = 24       # Prediction horizon (1 day)

# --- Model Define ---
# `enc_in`: Number of input features for the encoder.
# This will be your target ('occupancy') + 13 exogenous features.
# The data loader will select these columns from the CSV.
args.enc_in = 1 + 13     # 1 (target) + 13 (exogenous features) = 14

# `dec_in`: Number of input features for the decoder.
# For 'MS' mode, this is also typically target + exogenous features.
# The decoder receives future exogenous features and placeholders for the target.
args.dec_in = 1 + 13     # 1 (target placeholder) + 13 (exogenous features) = 14

# `c_out`: Number of output features (just 'occupancy').
args.c_out = 1

# Specific DLinear setting (if using DLinear)
if args.model == 'DLinear' or args.model == 'NLinear':
    args.individual = False # False: shared weights across channels. True: separate.
                            # For single target prediction, False is fine.

# Example for PatchTST (if you switch models, uncomment and adjust)
# if args.model == 'PatchTST':
#     args.patch_len = 16
#     args.stride = 8
#     args.n_heads = 4
#     args.e_layers = 2
#     args.d_model = 128
#     args.d_ff = 256
#     args.dropout = 0.1
#     args.revin = 1 # Instance Normalization

# --- Optimization settings ---
args.num_workers = 0     # For DataLoader. 0 for main process, >0 for multiprocessing
args.itr = 1             # Number of experiments to run (usually 1 for a single config)
args.train_epochs = 20   # Number of training epochs
args.batch_size = 32
args.patience = 5        # Early stopping patience
args.learning_rate = 0.001
args.loss = 'mse'        # 'mse' or 'mae'
args.lradj = 'type1'     # Learning rate adjustment strategy (e.g., 'type1', 'plateau')
args.use_amp = False     # Automatic mixed precision (set to True if GPU supports it and for speed)

# --- GPU settings ---
args.use_gpu = True if torch.cuda.is_available() else False
args.gpu = 0             # Specific GPU ID to use
args.devices = '0'       # Comma-separated list of GPU IDs for multi-GPU (e.g., '0,1')

# --- Time Feature Encoding ---
# Your CSV has a 'date' column. TSlib can generate time features from it.
# The 13 exogenous features might already BE your engineered time features.
# Option 1: Your exog_feat_1...13 ARE the time features.
#           You don't want TSlib to generate more or use its own for these.
args.embed = 'fixed' # Or 'learned'. This means time features are directly from data columns.
args.timeenc = 0     # 0: No automatic time feature embedding from the 'date' column by the TimeFeature layer.
                     #    The model will use the columns provided (occupancy + 13 exog) as is for enc_in/dec_in.
                     #    The 'date' column itself is not directly fed to the model layers unless it's part of your exog_feats.

# Option 2: Your exog_feat_1...13 are NOT time features, and you WANT TSlib
#           to generate time features from the 'date' column and add them.
# args.embed = 'timeF' # Use 'timeF' for frequency-based features, 'fixed' or 'learned' if providing them.
# args.timeenc = 1     # 1: Standard time features (month, day, weekday, hour).
                     #    If timeenc=1, these generated time features are ADDED to your enc_in/dec_in.
                     #    So, enc_in/dec_in would become (1 target + 13 exog + N_time_features_from_date_col).
                     #    You'd need to adjust enc_in and dec_in counts.
                     #    For 'h' freq, timeenc=1 adds 4 features. So enc_in = 1+13+4 = 18.
                     #    This is powerful if your exog_feats don't cover time adequately.

# **CHOOSE OPTION 1 or 2 based on your exog_feat content.**
# Assuming your 13 exog features ALREADY include RBF encoded time features:
# Stick with Option 1 (args.embed='fixed', args.timeenc=0).
# The `enc_in` and `dec_in` (1+13=14) should count ALL features you want the model to see from the CSV
# (target + your 13 engineered exogenous features).

# --- Output paths ---
# These are relative to where you run the script.
# Ensure 'checkpoints' and the root of 'results' directories exist or TSlib will create them.
args.output_path = './results/' # Base directory for results