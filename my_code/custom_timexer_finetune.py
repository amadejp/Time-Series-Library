import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from argparse import Namespace, ArgumentParser  # For command-line arguments
import os
import time
from multiprocessing import freeze_support

from models.TimeXer_custom import Model as TimeXerModel
from data_provider.custom_dataset import PreprocessedDataset  # Assuming this is flexible
from utils.metrics import metric
from utils.tools import visual


# --- Load Data Function (Keep as is, or make more flexible for different datasets) ---
def load_data(data_type, base_path):  # Pass base_path
    X_history = np.load(f"{base_path}{data_type}/X_history_target.npy")
    X_known_past = np.load(f"{base_path}{data_type}/X_known_past_exog_features.npy")
    X_known_future = np.load(f"{base_path}{data_type}/X_known_future_exog_features.npy")
    y = np.load(f"{base_path}{data_type}/y_target.npy")
    interval_dates = np.load(f"{base_path}{data_type}/interval_starts.npy", allow_pickle=True)
    interval_dates = pd.to_datetime(interval_dates)
    return X_history, X_known_past, X_known_future, y, interval_dates


# --- Early Stopping Class (Keep as is) ---
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pth', trace_func=print):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


# --- Training and Evaluation Loop Function ---
def train_eval_loop(model, configs, train_loader, val_loader, criterion, optimizer, device,
                    early_stopping_checkpoint_path, phase_name="Training"):
    early_stopping = EarlyStopping(patience=configs.patience, verbose=True, path=early_stopping_checkpoint_path)
    print(f"--- Starting {phase_name} for setting: {configs.model_id} ---")

    for epoch in range(configs.train_epochs):
        model.train()
        train_loss_epoch = []
        epoch_time_start = time.time()

        for i, (batch_x_enc, batch_x_mark_enc, batch_x_dec, batch_x_mark_dec, batch_y) in enumerate(train_loader):
            batch_x_enc = batch_x_enc.float().to(device)  # Ensure float
            batch_x_mark_enc = batch_x_mark_enc.float().to(device)
            batch_x_dec = batch_x_dec.float().to(device)  # Placeholder, not used by TimeXer
            batch_x_mark_dec = batch_x_mark_dec.float().to(device)
            batch_y = batch_y.float().to(device)

            optimizer.zero_grad()
            outputs = model(batch_x_enc, batch_x_mark_enc, batch_x_dec, batch_x_mark_dec)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss_epoch.append(loss.item())

            if (i + 1) % 100 == 0:
                print(
                    f"\t{phase_name} Epoch {epoch + 1}, Iteration {i + 1}/{len(train_loader)} | Current Batch Loss: {loss.item():.7f}")

        avg_train_loss = np.average(train_loss_epoch) if train_loss_epoch else 0
        epoch_duration = time.time() - epoch_time_start

        model.eval()
        val_loss_epoch = []
        with torch.no_grad():
            for batch_x_enc, batch_x_mark_enc, batch_x_dec, batch_x_mark_dec, batch_y in val_loader:
                batch_x_enc = batch_x_enc.float().to(device)
                batch_x_mark_enc = batch_x_mark_enc.float().to(device)
                batch_x_dec = batch_x_dec.float().to(device)
                batch_x_mark_dec = batch_x_mark_dec.float().to(device)
                batch_y = batch_y.float().to(device)
                outputs = model(batch_x_enc, batch_x_mark_enc, batch_x_dec, batch_x_mark_dec)
                loss = criterion(outputs, batch_y)
                val_loss_epoch.append(loss.item())

        avg_val_loss = np.average(val_loss_epoch) if val_loss_epoch else 0
        print(
            f"{phase_name} Epoch [{epoch + 1}/{configs.train_epochs}] ({epoch_duration:.2f}s) -> Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print(f"Early stopping during {phase_name}")
            break
    print(f"--- {phase_name} Finished ---")
    return early_stopping.path  # Return path to best model from this phase


# --- Testing Function ---
def test_model(model, configs, test_loader, criterion, device, results_path, test_results_figures_path,
               interval_dates_test):
    print("\n--- Starting Testing ---")
    model.eval()
    preds_list = []
    trues_list = []
    first_batch_input_x_for_plot = None

    with torch.no_grad():
        for i, (batch_x_enc, batch_x_mark_enc, batch_x_dec, batch_x_mark_dec, batch_y) in enumerate(test_loader):
            batch_x_enc_device = batch_x_enc.float().to(device)
            batch_x_mark_enc_device = batch_x_mark_enc.float().to(device)
            batch_x_dec_device = batch_x_dec.float().to(device)
            batch_x_mark_dec_device = batch_x_mark_dec.float().to(device)

            outputs = model(batch_x_enc_device, batch_x_mark_enc_device, batch_x_dec_device, batch_x_mark_dec_device)
            preds_list.append(outputs.detach().cpu().numpy())
            trues_list.append(batch_y.numpy())  # batch_y is already float from loader
            if i == 0 and batch_x_enc.shape[0] > 0:
                # Assuming target is the last feature in x_enc for plotting input history
                first_batch_input_x_for_plot = batch_x_enc[0, :,
                                               configs.enc_in - 1].numpy() if configs.features == 'MS' else batch_x_enc[
                                                                                                            0, :,
                                                                                                            0].numpy()

    preds = np.concatenate(preds_list, axis=0)
    trues = np.concatenate(trues_list, axis=0)
    print('Test shapes after concatenation: preds={}, trues={}'.format(preds.shape, trues.shape))

    mae, mse, rmse, mape, mspe = metric(preds, trues)
    print(f'Test Metrics: MSE:{mse:.4f}, MAE:{mae:.4f}, RMSE:{rmse:.4f}, MAPE:{mape:.4f}, MSPE:{mspe:.4f}')

    os.makedirs(results_path, exist_ok=True)
    os.makedirs(test_results_figures_path, exist_ok=True)

    np.save(os.path.join(results_path, 'metrics.npy'), np.array([mae, mse, rmse, mape, mspe]))
    np.save(os.path.join(results_path, 'pred.npy'), preds)
    np.save(os.path.join(results_path, 'true.npy'), trues)

    if len(interval_dates_test) >= preds.shape[0]:
        pred_start_datetimes_test = interval_dates_test[:preds.shape[0]]
        np.save(os.path.join(results_path, 'pred_start_dates.npy'), np.array(pred_start_datetimes_test, dtype=object))
    else:
        print(
            f"Warning: Not enough interval_dates_test ({len(interval_dates_test)}) for all predictions ({preds.shape[0]}). Dates not saved.")

    if preds.shape[0] > 0 and trues.shape[0] > 0 and first_batch_input_x_for_plot is not None:
        try:
            # Ensure trues[0] and preds[0] are 2D [seq_len, num_targets] then take first target [:,0]
            gt_plot = np.concatenate((first_batch_input_x_for_plot, trues[0, :, 0]), axis=0)
            pd_plot = np.concatenate((first_batch_input_x_for_plot, preds[0, :, 0]), axis=0)
            visual(gt_plot, pd_plot, os.path.join(test_results_figures_path, 'sample_0_visualization.pdf'))
            print(f"Saved visualization for the first test sample to {test_results_figures_path}")
        except Exception as e:
            print(f"Could not generate visualization: {e}")
    else:
        print("Not enough data to generate visualization or first_batch_input_x_for_plot was not captured.")
    return mse, mae, rmse, mape, mspe


def main(args):
    # --- Base Configuration ---
    # These can be overridden by args if you make them command-line arguments
    win_len = 336
    stride = 24
    pred_len = 24
    patch_len = 24
    enc_in_pretrain = 14
    enc_in_finetune = 14  # Your current dataset
    num_future_exo_pretrain = 13  # Example
    num_future_exo_finetune = 13  # Your current

    # Common model architecture parameters (should be consistent or adaptable)
    d_model = 128
    n_heads = 16
    e_layers = 1
    d_ff = 1024
    dropout = 0.1
    activation = 'gelu'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- PHASE 1: PRE-TRAINING ---
    if args.do_pretrain:
        print("\n=============== STARTING PRE-TRAINING PHASE ===============")
        pretrain_model_id = f"Pretrained_TimeXer_win{win_len}_patch{patch_len}_data{args.pretrain_data_name}_" + time.strftime(
            "%Y%m%d_%H%M%S")
        pretrain_checkpoints_path = f'../checkpoints/{pretrain_model_id}/'
        os.makedirs(pretrain_checkpoints_path, exist_ok=True)

        pretrain_configs = Namespace(
            task_name='long_term_forecast', features='MS',
            seq_len=win_len, pred_len=pred_len, label_len=0, patch_len=patch_len,
            enc_in=enc_in_pretrain,  # Specific to pretrain data
            dec_in=num_future_exo_pretrain,  # Corresponds to X_known_future for pretrain
            c_out=1,  # Assuming single target for pretrain
            d_model=d_model, n_heads=n_heads, e_layers=e_layers, d_ff=d_ff, dropout=dropout,
            activation=activation, factor=3, embed='fixed', freq='h',
            use_norm=False, inverse=False,
            model_id=pretrain_model_id, model='TimeXerCustom', data=args.pretrain_data_name, des='PretrainRun',
            patience=args.pretrain_patience, learning_rate=args.pretrain_lr, train_epochs=args.pretrain_epochs,
            # Add num_known_future_features if your model uses it directly from configs
            num_known_future_features=num_future_exo_pretrain
        )

        # Load pre-training data
        # IMPORTANT: Adjust load_data and PreprocessedDataset to handle different data structures if needed
        print(f"Loading pre-training data from: {args.pretrain_data_path}")
        X_history_pt_train, X_known_past_pt_train, X_known_future_pt_train, y_pt_train, _ = load_data("train",
                                                                                                      args.pretrain_data_path)
        X_history_pt_val, X_known_past_pt_val, X_known_future_pt_val, y_pt_val, _ = load_data("val",
                                                                                              args.pretrain_data_path)

        pt_train_dataset = PreprocessedDataset(X_history_pt_train, X_known_past_pt_train, X_known_future_pt_train,
                                               y_pt_train)
        pt_val_dataset = PreprocessedDataset(X_history_pt_val, X_known_past_pt_val, X_known_future_pt_val, y_pt_val)

        pt_train_loader = DataLoader(pt_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
                                     pin_memory=True if device.type == 'cuda' else False)
        pt_val_loader = DataLoader(pt_val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
                                   pin_memory=True if device.type == 'cuda' else False)

        pretrain_model = TimeXerModel(pretrain_configs).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(pretrain_model.parameters(), lr=pretrain_configs.learning_rate)
        pretrain_checkpoint_file = os.path.join(pretrain_checkpoints_path, 'checkpoint.pth')

        best_pretrained_model_path = train_eval_loop(pretrain_model, pretrain_configs, pt_train_loader, pt_val_loader,
                                                     criterion, optimizer, device, pretrain_checkpoint_file,
                                                     phase_name="Pre-training")
        args.pretrained_model_path = best_pretrained_model_path  # Store for fine-tuning
        print(f"Pre-training finished. Best model saved at: {args.pretrained_model_path}")

    elif args.pretrained_model_path and not os.path.exists(args.pretrained_model_path):
        print(
            f"Error: Specified pretrained_model_path '{args.pretrained_model_path}' does not exist, but pre-training was skipped.")
        return

    # --- PHASE 2: FINE-TUNING (or training from scratch if no pretrain_path) ---
    if args.do_finetune or (not args.do_pretrain and not args.pretrained_model_path):
        phase_descriptor = "Fine-tuning" if args.pretrained_model_path else "Training_from_scratch"
        print(f"\n=============== STARTING {phase_descriptor.upper()} PHASE ===============")

        finetune_model_id_suffix = f"_ft_from_{os.path.basename(args.pretrained_model_path).replace('.pth', '')}" if args.pretrained_model_path else "_scratch"
        finetune_model_id = f"Finetuned_TimeXer_win{win_len}_stride{stride}{finetune_model_id_suffix}_" + time.strftime(
            "%Y%m%d_%H%M%S")

        finetune_checkpoints_path = f'../checkpoints/{finetune_model_id}/'
        finetune_results_path = f'../results/{finetune_model_id}/'
        finetune_test_figures_path = f'../test_results/{finetune_model_id}/'
        os.makedirs(finetune_checkpoints_path, exist_ok=True)

        finetune_configs = Namespace(
            task_name='long_term_forecast', features='MS',
            seq_len=win_len, pred_len=pred_len, label_len=0, patch_len=patch_len,
            enc_in=enc_in_finetune,  # Specific to fine-tune data
            dec_in=num_future_exo_finetune,  # Corresponds to X_known_future for fine-tune
            c_out=1,  # Assuming single target
            d_model=d_model, n_heads=n_heads, e_layers=e_layers, d_ff=d_ff, dropout=dropout,
            activation=activation, factor=3, embed='fixed', freq='h',
            use_norm=False, inverse=False,
            model_id=finetune_model_id, model='TimeXerCustom', data='customEV_finetune', des=f'{phase_descriptor}Run',
            patience=args.finetune_patience, learning_rate=args.finetune_lr, train_epochs=args.finetune_epochs,
            num_known_future_features=num_future_exo_finetune
        )

        # Load fine-tuning data (your current dataset)
        print(f"Loading fine-tuning data from: {args.finetune_data_path}")
        X_history_ft_train, X_known_past_ft_train, X_known_future_ft_train, y_ft_train, _ = load_data("train",
                                                                                                      args.finetune_data_path)
        X_history_ft_val, X_known_past_ft_val, X_known_future_ft_val, y_ft_val, _ = load_data("val",
                                                                                              args.finetune_data_path)
        X_history_ft_test, X_known_past_ft_test, X_known_future_ft_test, y_ft_test, interval_dates_ft_test = load_data(
            "test", args.finetune_data_path)

        ft_train_dataset = PreprocessedDataset(X_history_ft_train, X_known_past_ft_train, X_known_future_ft_train,
                                               y_ft_train)
        ft_val_dataset = PreprocessedDataset(X_history_ft_val, X_known_past_ft_val, X_known_future_ft_val, y_ft_val)
        ft_test_dataset = PreprocessedDataset(X_history_ft_test, X_known_past_ft_test, X_known_future_ft_test,
                                              y_ft_test)

        ft_train_loader = DataLoader(ft_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
                                     pin_memory=True if device.type == 'cuda' else False)
        ft_val_loader = DataLoader(ft_val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
                                   pin_memory=True if device.type == 'cuda' else False)
        ft_test_loader = DataLoader(ft_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0,
                                    pin_memory=True if device.type == 'cuda' else False)

        finetune_model = TimeXerModel(finetune_configs).to(device)
        criterion = nn.MSELoss()

        if args.pretrained_model_path:
            print(f"Loading pre-trained weights from: {args.pretrained_model_path}")
            # Load weights, be careful if model architectures differ (e.g. enc_in)
            # You might need to load selectively or ensure compatibility
            try:
                pretrained_dict = torch.load(args.pretrained_model_path, map_location=device)
                model_dict = finetune_model.state_dict()

                # Filter out unnecessary keys and adapt for size mismatches
                # Example: If embedding layers for enc_in differ, they won't load.
                # This is a simple load; for more complex cases (like different enc_in),
                # you'd need to handle those layers specifically (e.g., reinitialize them).
                # If d_model and other core arch params are the same, most weights should load.
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                                   k in model_dict and model_dict[k].size() == v.size()}
                model_dict.update(pretrained_dict)
                finetune_model.load_state_dict(model_dict, strict=False)  # strict=False to allow partial load
                print(f"Loaded {len(pretrained_dict)} matching keys from pretrained model.")
            except Exception as e:
                print(f"Error loading pretrained weights: {e}. Training from scratch for fine-tuning.")
        else:
            print("No pretrained model path provided, training from scratch for fine-tuning.")

        # Optimizer for fine-tuning (potentially different LR)
        optimizer = optim.Adam(finetune_model.parameters(), lr=finetune_configs.learning_rate)
        finetune_checkpoint_file = os.path.join(finetune_checkpoints_path, 'checkpoint.pth')

        best_finetuned_model_path = train_eval_loop(finetune_model, finetune_configs, ft_train_loader, ft_val_loader,
                                                    criterion, optimizer, device, finetune_checkpoint_file,
                                                    phase_name=phase_descriptor)

        print(f"Loading best model from {phase_descriptor} for final testing...")
        if os.path.exists(best_finetuned_model_path):
            finetune_model.load_state_dict(torch.load(best_finetuned_model_path, map_location=device))
        else:
            print(f"Warning: Best model checkpoint not found at {best_finetuned_model_path}. Using last state.")

        # Test the fine-tuned model
        mse, mae, rmse, mape, mspe = test_model(finetune_model, finetune_configs, ft_test_loader, criterion, device,
                                                finetune_results_path, finetune_test_figures_path,
                                                interval_dates_ft_test)

        result_log_file = f"result_custom_long_term_forecast_{phase_descriptor.lower()}.txt"
        with open(result_log_file, 'a') as f:
            f.write(finetune_configs.model_id + "  \n")
            f.write(f'mse:{mse:.4f}, mae:{mae:.4f}, rmse:{rmse:.4f}, mape:{mape:.4f}, mspe:{mspe:.4f}')
            f.write('\n')
            f.write('\n')
        print(f"Results logged to {result_log_file}")
        print(f"--- Experiment Finished: {finetune_configs.model_id} ---")


if __name__ == '__main__':
    freeze_support()
    parser = ArgumentParser(description='TimeXer Pre-training and Fine-tuning')
    # Pre-training Args
    parser.add_argument('--do_pretrain', action='store_true', help='Flag to perform pre-training phase.')
    parser.add_argument('--pretrain_data_path', type=str, default="../pretrain_data_placeholder/",
                        help='Path to pre-training dataset.')
    parser.add_argument('--pretrain_data_name', type=str, default="PretrainDataset",
                        help='Name for pre-training dataset.')
    parser.add_argument('--pretrain_epochs', type=int, default=50, help='Epochs for pre-training.')
    parser.add_argument('--pretrain_lr', type=float, default=1e-4, help='Learning rate for pre-training.')
    parser.add_argument('--pretrain_patience', type=int, default=10, help='Early stopping patience for pre-training.')

    # Fine-tuning Args
    parser.add_argument('--do_finetune', action='store_true', help='Flag to perform fine-tuning phase.')
    parser.add_argument('--finetune_data_path', type=str, default="../my_data/train70_val10_test20_winlen336_stride24/",
                        help='Path to fine-tuning dataset.')
    parser.add_argument('--finetune_epochs', type=int, default=100, help='Epochs for fine-tuning.')
    parser.add_argument('--finetune_lr', type=float, default=5e-5, help='Learning rate for fine-tuning.')
    parser.add_argument('--finetune_patience', type=int, default=8, help='Early stopping patience for fine-tuning.')

    # General Args
    parser.add_argument('--pretrained_model_path', type=str, default=None,
                        help='Path to a pre-trained model to load for fine-tuning (skips pre-training if set).')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and validation.')
    # Add other common args if needed: d_model, n_heads, etc. if you want to vary them via cmd line

    args = parser.parse_args()

    # Logic to ensure at least one phase is run
    if not args.do_pretrain and not args.do_finetune and not args.pretrained_model_path:
        print(
            "Neither pre-training nor fine-tuning is enabled, and no pre-trained model path provided for direct fine-tuning. Exiting.")
        print(
            "Use --do_pretrain to pretrain, --do_finetune to fine-tune (optionally with --pretrained_model_path), or just --pretrained_model_path and --do_finetune.")
    elif not args.do_pretrain and not args.do_finetune and args.pretrained_model_path:
        print(
            "A pretrained_model_path is provided, but --do_finetune is not set. Will only fine-tune if --do_finetune is also active.")
        args.do_finetune = True  # Implicitly enable fine-tuning if path is given
        print("Enabling fine-tuning by default as pretrained_model_path is set.")

    main(args)