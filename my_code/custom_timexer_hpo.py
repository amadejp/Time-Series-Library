import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from argparse import Namespace
import os
import time
import optuna
from multiprocessing import freeze_support

from models.TimeXer_custom import Model as TimeXerModel
from data_provider.custom_dataset import PreprocessedDataset
from utils.metrics import metric
from utils.tools import visual


# --- Load Data Function (unchanged) ---
def load_data(data_type, base_path="../my_data/train70_val10_test20_winlen168_stride24/"):
    X_history = np.load(f"{base_path}{data_type}/X_history_target.npy")
    X_known_past = np.load(f"{base_path}{data_type}/X_known_past_exog_features.npy")
    X_known_future = np.load(f"{base_path}{data_type}/X_known_future_exog_features.npy")
    y = np.load(f"{base_path}{data_type}/y_target.npy")
    interval_dates = np.load(f"{base_path}{data_type}/interval_starts.npy", allow_pickle=True)
    interval_dates = pd.to_datetime(interval_dates)
    return X_history, X_known_past, X_known_future, y, interval_dates


# --- Early Stopping Class (unchanged) ---
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


# --- Objective Function for Optuna ---
def objective(trial):
    # --- Suggest Hyperparameters ---
    # This section defines the search space for Optuna.
    # It will pick a value for each hyperparameter for each trial.
    d_model = trial.suggest_categorical('d_model', [128, 256, 512])
    n_heads = trial.suggest_categorical('n_heads', [4, 8, 16])

    # Ensure d_model is divisible by n_heads
    if d_model % n_heads != 0:
        # A simple way to handle this is to prune the trial if the condition is not met
        raise optuna.exceptions.TrialPruned(f"d_model {d_model} not divisible by n_heads {n_heads}")

    # For patch_len, ensure seq_len (336) is divisible by it.
    seq_len = 168
    patch_len = trial.suggest_categorical('patch_len', [12, 24, 42, 84, 168])
    if seq_len % patch_len != 0:
        raise optuna.exceptions.TrialPruned(f"seq_len {seq_len} not divisible by patch_len {patch_len}")

    configs = Namespace(
        # Model Architecture
        d_model=d_model,
        n_heads=n_heads,
        e_layers=trial.suggest_int('e_layers', 1, 4),
        d_ff=trial.suggest_categorical('d_ff', [256, 512, 1024, 2048]),
        dropout=trial.suggest_float('dropout', 0.1, 0.5, step=0.05),
        patch_len=patch_len,

        # Optimizer
        learning_rate=trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),

        # Training
        batch_size=trial.suggest_categorical('batch_size', [16, 32, 64]),

        # Fixed parameters from your original script
        task_name='long_term_forecast',
        features='MS',
        seq_len=seq_len,
        pred_len=24,
        label_len=0,
        enc_in=14,
        dec_in=13,
        c_out=1,
        activation='gelu',
        factor=3,
        embed='fixed',
        freq='h',
        use_norm=False,
        inverse=False,
        model_id="EV_TimeXer_Optimized",
        model='TimeXerCustom',
        data='customEV',
        des='OptunaRun',
        patience=8,
        train_epochs=120,  # Max epochs, early stopping will handle the rest
    )

    # --- Setup Directories and Settings String ---
    # Create a unique setting string for this trial
    setting = '{}_{}_trial{}_dm{}_nh{}_el{}_df{}_pl{}_lr{}_bs{}'.format(
        configs.model_id,
        configs.des,
        trial.number,
        configs.d_model,
        configs.n_heads,
        configs.e_layers,
        configs.d_ff,
        configs.patch_len,
        f"{configs.learning_rate:.1E}",
        configs.batch_size
    )
    checkpoints_path = f'../checkpoints/optuna/{setting}/'
    os.makedirs(checkpoints_path, exist_ok=True)
    configs.checkpoints = checkpoints_path

    # --- Load Data (same as before) ---
    X_history_train, X_known_past_train, X_known_future_train, y_train, _ = load_data("train")
    X_history_val, X_known_past_val, X_known_future_val, y_val, _ = load_data("val")

    train_dataset = PreprocessedDataset(X_history_train, X_known_past_train, X_known_future_train, y_train)
    val_dataset = PreprocessedDataset(X_history_val, X_known_past_val, X_known_future_val, y_val)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(train_dataset, batch_size=configs.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=configs.batch_size, shuffle=False, num_workers=0)

    # --- Initialize Model, Loss, Optimizer ---
    model = TimeXerModel(configs).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=configs.learning_rate)

    early_stopping = EarlyStopping(patience=configs.patience, verbose=False,
                                   path=os.path.join(checkpoints_path, 'checkpoint.pth'))

    print(f"\n--- Starting Trial {trial.number}: {setting} ---")

    # --- Training and Validation Loop ---
    for epoch in range(configs.train_epochs):
        model.train()
        train_loss_epoch = []
        for (batch_x_enc, batch_x_mark_enc, batch_x_dec, batch_x_mark_dec, batch_y) in train_loader:
            batch_x_enc, batch_x_mark_enc, batch_x_dec, batch_x_mark_dec, batch_y = [d.to(device) for d in
                                                                                     [batch_x_enc, batch_x_mark_enc,
                                                                                      batch_x_dec, batch_x_mark_dec,
                                                                                      batch_y]]

            optimizer.zero_grad()
            outputs = model(batch_x_enc, batch_x_mark_enc, batch_x_dec, batch_x_mark_dec)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss_epoch.append(loss.item())

        avg_train_loss = np.average(train_loss_epoch)

        model.eval()
        val_loss_epoch = []
        with torch.no_grad():
            for (batch_x_enc, batch_x_mark_enc, batch_x_dec, batch_x_mark_dec, batch_y) in val_loader:
                batch_x_enc, batch_x_mark_enc, batch_x_dec, batch_x_mark_dec, batch_y = [d.to(device) for d in
                                                                                         [batch_x_enc, batch_x_mark_enc,
                                                                                          batch_x_dec, batch_x_mark_dec,
                                                                                          batch_y]]
                outputs = model(batch_x_enc, batch_x_mark_enc, batch_x_dec, batch_x_mark_dec)
                loss = criterion(outputs, batch_y)
                val_loss_epoch.append(loss.item())

        avg_val_loss = np.average(val_loss_epoch)
        print(
            f"Trial {trial.number}, Epoch {epoch + 1} -> Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # --- Optuna Pruning and Early Stopping ---
        trial.report(avg_val_loss, epoch)
        if trial.should_prune():
            print(f"Trial {trial.number} pruned at epoch {epoch + 1}.")
            raise optuna.exceptions.TrialPruned()

        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch + 1}.")
            break

    # Return the best validation loss for this trial
    return early_stopping.val_loss_min


# --- Main Execution Block ---
if __name__ == '__main__':
    freeze_support()

    # --- Create and Run Optuna Study ---
    # The 'direction' is 'minimize' because we want to minimize validation loss.
    # The 'pruner' will stop unpromising trials early.
    study = optuna.create_study(
        direction='minimize',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    )

    # 'n_trials' is the number of different hyperparameter combinations to test.
    # Start with a smaller number (e.g., 20) and increase as needed.
    study.optimize(objective, n_trials=100)

    # --- Print Optimization Results ---
    print("\n--- OPTIMIZATION FINISHED ---")
    print(f"Number of finished trials: {len(study.trials)}")

    print("\nBest trial:")
    best_trial = study.best_trial
    print(f"  Value (min val_loss): {best_trial.value:.6f}")

    print("  Best Parameters:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    # You can now take the best parameters and run a final training run on the combined
    # train and validation sets, and then evaluate on the test set.