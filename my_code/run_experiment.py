import torch
import os
import sys

from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
# If you get an import error for 'experiments', ensure the root of Time-Series-Library is in sys.path

# --- DotDict class (if not importing from TSlib utils) ---
class DotDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

# --- Args Definition (Copy from Step 3) ---
args = DotDict()
# --- Basic Experiment settings ---
args.is_training = 1
args.model_id = 'EV_Occupancy_DLinear_from_CSV'
args.model = 'DLinear'
# --- Data Loader settings ---
args.data = 'custom'
args.root_path = '../my_data/'
args.data_path = 'inesctec_occupancy_ts.csv'
args.features = 'MS'
args.target = 'n_active_sessions_grid'
args.freq = 'h'
args.checkpoints = './checkpoints/'
# --- Forecasting Task settings ---
args.seq_len = 72
args.label_len = 0
args.pred_len = 24
# --- Model Define ---
args.enc_in = 1 + 13 # occupancy + 13 exog features
args.dec_in = 1 + 13 # occupancy_placeholder + 13 exog features
args.c_out = 1       # predicting occupancy
if args.model == 'DLinear' or args.model == 'NLinear':
    args.individual = False
# --- Optimization settings ---
args.num_workers = 0
args.itr = 1
args.train_epochs = 20 # Adjust as needed
args.batch_size = 32
args.patience = 5
args.learning_rate = 0.001
args.loss = 'mse'
args.lradj = 'type1'
args.use_amp = False
# --- GPU settings ---
args.use_gpu = True if torch.cuda.is_available() else False
args.gpu = 0
args.devices = '0'
# --- Time Feature Encoding ---
# CHOSEN: Option 1 (Exogenous features already include time info)
args.embed = 'fixed'
args.timeenc = 0
# --- Output paths ---
args.output_path = './results/'
# --- (End of args definition) ---


def main():
    # Create directories if they don't exist
    os.makedirs(args.checkpoints, exist_ok=True)
    os.makedirs(args.output_path, exist_ok=True) # Base results directory

    # Initialize experiment
    # The Exp_Long_Term_Forecast class will handle data loading and splitting
    # from the CSV based on args.
    exp = Exp_Long_Term_Forecast(args)

    # --- Train ---
    if args.is_training:
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(args.model_id))
        exp.train(args.model_id) # The setting string is used for file naming

    # --- Test ---
    print('>>>>>>>start testing : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(args.model_id))
    exp.test(args.model_id, test=1) # test=1 loads the best model from training for testing

    # Clean up GPU memory
    if args.use_gpu:
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()