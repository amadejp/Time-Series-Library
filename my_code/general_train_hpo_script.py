import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from argparse import Namespace
import os
import time
from multiprocessing import freeze_support
import optuna  # Import Optuna

# --- Model Imports ---
from models.TimeXer import Model as TimeXerModel
from models.TemporalFusionTransformer import Model as TFTModel
from models.Informer import Model as InformerModel
from models.Autoformer import Model as AutoformerModel
from models.PatchTST import Model as PatchTSTModel
from models.DLinear import Model as DLinearModel

from utils.metrics import metric
from utils.tools import visual


# --- Custom Dataset Class (Unchanged) ---
class CustomDataset(Dataset):
    def __init__(self, x_history, x_known_past, x_known_future, y, configs):
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.x_history = torch.tensor(x_history, dtype=torch.float32)
        self.x_known_past = torch.tensor(x_known_past, dtype=torch.float32)
        self.x_known_future = torch.tensor(x_known_future, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.x_history)

    def __getitem__(self, idx):
        x_enc = self.x_history[idx]
        batch_y = self.y[idx]
        x_mark_enc = self.x_known_past[idx]
        dec_inp_mark_start = x_mark_enc[-self.label_len:, :] if self.label_len > 0 else torch.empty(0, x_mark_enc.shape[
            -1])
        x_mark_dec = torch.cat([dec_inp_mark_start, self.x_known_future[idx]], dim=0)
        dec_inp_val_start = x_enc[-self.label_len:, :] if self.label_len > 0 else torch.empty(0, x_enc.shape[-1])
        dec_inp_val_future = torch.zeros(self.pred_len, x_enc.shape[-1], dtype=torch.float32)
        x_dec = torch.cat([dec_inp_val_start, dec_inp_val_future], dim=0)
        return x_enc, x_mark_enc, x_dec, x_mark_dec, batch_y


# --- Early Stopping Class (Unchanged) ---
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
            if self.verbose: self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience: self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose: self.trace_func(
            f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def get_model_and_configs(model_name, trial=None):
    base_configs = {
        'task_name': 'long_term_forecast', 'features': 'S', 'seq_len': 336, 'pred_len': 24,
        'enc_in': 1, 'dec_in': 1, 'c_out': 1, 'freq': 'h', 'patience': 8, 'train_epochs': 100
    }

    # --- HPO Search Spaces ---
    search_spaces = {
        'TimeXer': {
            'learning_rate': lambda t: t.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
            'd_model': lambda t: t.suggest_categorical('d_model', [64, 128, 256]),
            'd_ff': lambda t: t.suggest_categorical('d_ff', [128, 256, 512]),
            'n_heads': lambda t: t.suggest_categorical('n_heads', [4, 8]),
            'dropout': lambda t: t.suggest_float('dropout', 0.05, 0.3),
            'patch_len': lambda t: t.suggest_categorical('patch_len', [12, 24, 48]),
        },
        'PatchTST': {
            'learning_rate': lambda t: t.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
            'd_model': lambda t: t.suggest_categorical('d_model', [64, 128, 256]),
            'n_heads': lambda t: t.suggest_categorical('n_heads', [8, 16]),
            'e_layers': lambda t: t.suggest_int('e_layers', 1, 3),
            'd_ff': lambda t: t.suggest_categorical('d_ff', [128, 256, 512]),
            'dropout': lambda t: t.suggest_float('dropout', 0.05, 0.3),
        },
        # Add other model search spaces here...
    }

    # --- Default Model Configurations ---
    default_configs = {
        'TimeXer': {'model': 'TimeXer', 'label_len': 0, 'd_model': 128, 'n_heads': 8, 'e_layers': 1, 'd_ff': 512,
                    'activation': 'gelu', 'embed': 'timeF', 'factor': 3, 'patch_len': 24, 'learning_rate': 0.000235,
                    'use_norm': True, 'dropout': 0.06},
        'TFT': {'model': 'TFT', 'data': 'customEV', 'label_len': 0, 'd_model': 128, 'n_heads': 8, 'e_layers': 1,
                'd_ff': 512, 'activation': 'gelu', 'embed': 'timeF', 'learning_rate': 0.0001, 'dropout': 0.1},
        'Informer': {'model': 'Informer', 'label_len': 48, 'd_model': 512, 'n_heads': 8, 'e_layers': 2, 'd_layers': 1,
                     'd_ff': 2048, 'activation': 'gelu', 'embed': 'timeF', 'factor': 3, 'learning_rate': 0.0001,
                     'dropout': 0.1},
        'Autoformer': {'model': 'Autoformer', 'label_len': 48, 'd_model': 512, 'n_heads': 8, 'e_layers': 2,
                       'd_layers': 1, 'd_ff': 2048, 'activation': 'gelu', 'embed': 'timeF', 'factor': 3,
                       'learning_rate': 0.0001, 'moving_avg': 25, 'dropout': 0.1},
        'PatchTST': {'model': 'PatchTST', 'label_len': 0, 'd_model': 128, 'n_heads': 16, 'e_layers': 3, 'd_ff': 256,
                     'activation': 'gelu', 'learning_rate': 0.0001, 'patch_len': 24, 'stride': 12, 'revin': 1,
                     'factor': 3, 'dropout': 0.1},
        'DLinear': {'model': 'DLinear', 'label_len': 0, 'learning_rate': 0.001, 'individual': False, 'dropout': 0.1}
    }

    MODEL_CLASSES = {
        'TimeXer': TimeXerModel, 'TFT': TFTModel, 'Informer': InformerModel, 'Autoformer': AutoformerModel,
        'PatchTST': PatchTSTModel, 'DLinear': DLinearModel,
    }

    if model_name not in MODEL_CLASSES:
        raise ValueError(f"Model '{model_name}' not recognized. Available: {list(MODEL_CLASSES.keys())}")

    final_configs = {**base_configs, **default_configs[model_name]}

    if trial:  # If in HPO mode
        if model_name not in search_spaces:
            print(f"Warning: HPO search space not defined for {model_name}. Using default parameters.")
        else:
            hpo_params = {name: func(trial) for name, func in search_spaces[model_name].items()}
            final_configs.update(hpo_params)

    return MODEL_CLASSES[model_name], Namespace(**final_configs)


def objective(trial, model_name, train_loader, val_loader, device):
    """Optuna objective function for a single HPO trial."""
    # Generate model and configs for this trial
    ModelClass, configs = get_model_and_configs(model_name, trial)
    configs.train_epochs = 20  # Train for fewer epochs during HPO

    model = ModelClass(configs).to(device)
    optimizer = optim.Adam(model.parameters(), lr=configs.learning_rate)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')

    for epoch in range(configs.train_epochs):
        model.train()
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

        # Validation
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
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss

        # Pruning: Report intermediate results to Optuna
        trial.report(avg_val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_val_loss


def run_experiment():
    # --- TOP-LEVEL CONTROLS ---
    model_name = 'PatchTST'  # Options: 'TimeXer', 'TFT', 'Informer', 'Autoformer', 'PatchTST', 'DLinear'
    PERFORM_HPO = False  # Set to True to run HPO, False to use defaults
    N_TRIALS = 100  # Number of HPO trials to run

    # --- Load Data Once ---
    print("--- Loading Data ---")
    data_path = "../my_data/train70_val10_test20_winlen336_stride24_workdays/"
    X_history_train, X_known_past_train, X_known_future_train, y_train = [np.load(f"{data_path}train/{fname}") for fname
                                                                          in ["X_history_target.npy",
                                                                              "X_known_past_exog_features.npy",
                                                                              "X_known_future_exog_features.npy",
                                                                              "y_target.npy"]]
    X_history_val, X_known_past_val, X_known_future_val, y_val = [np.load(f"{data_path}val/{fname}") for fname in
                                                                  ["X_history_target.npy",
                                                                   "X_known_past_exog_features.npy",
                                                                   "X_known_future_exog_features.npy", "y_target.npy"]]
    X_history_test, X_known_past_test, X_known_future_test, y_test = [np.load(f"{data_path}test/{fname}") for fname in
                                                                      ["X_history_target.npy",
                                                                       "X_known_past_exog_features.npy",
                                                                       "X_known_future_exog_features.npy",
                                                                       "y_target.npy"]]
    interval_dates_test = pd.to_datetime(np.load(f"{data_path}test/interval_starts.npy", allow_pickle=True))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- HPO Phase ---
    best_params = {}
    if PERFORM_HPO:
        print(f"--- Starting HPO for {model_name} ---")
        # We only need default configs to create the dataloaders for the objective function
        _, temp_configs = get_model_and_configs(model_name)
        train_dataset = CustomDataset(X_history_train, X_known_past_train, X_known_future_train, y_train, temp_configs)
        val_dataset = CustomDataset(X_history_val, X_known_past_val, X_known_future_val, y_val, temp_configs)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner())
        study.optimize(lambda trial: objective(trial, model_name, train_loader, val_loader, device), n_trials=N_TRIALS)

        best_params = study.best_trial.params
        print(f"--- HPO Finished ---")
        print("Best trial params:", best_params)
        print("Best value (validation loss):", study.best_value)

    # --- Final Training and Evaluation Phase ---
    print("\n--- Starting Final Training with Best/Default Parameters ---")
    ModelClass, configs = get_model_and_configs(model_name)
    if best_params:  # If HPO was run, update configs with best params
        for key, value in best_params.items():
            setattr(configs, key, value)

    model_id = f"{configs.model}_final_seq{configs.seq_len}_pred{configs.pred_len}_{time.strftime('%Y%m%d_%H%M%S')}"
    checkpoints_path = f'../checkpoints/{model_id}/'
    os.makedirs(checkpoints_path, exist_ok=True)

    train_dataset = CustomDataset(X_history_train, X_known_past_train, X_known_future_train, y_train, configs)
    val_dataset = CustomDataset(X_history_val, X_known_past_val, X_known_future_val, y_val, configs)
    test_dataset = CustomDataset(X_history_test, X_known_past_test, X_known_future_test, y_test, configs)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = ModelClass(configs).to(device)
    optimizer = optim.Adam(model.parameters(), lr=configs.learning_rate)
    criterion = nn.MSELoss()
    early_stopping = EarlyStopping(patience=configs.patience, verbose=True,
                                   path=os.path.join(checkpoints_path, 'checkpoint.pth'))

    print(f"--- Training model {model_id} with configs: ---")
    print(configs)

    for epoch in range(configs.train_epochs):
        model.train()
        # (Same training loop as before)
        for (batch_x_enc, batch_x_mark_enc, batch_x_dec, batch_x_mark_dec, batch_y) in train_loader:
            batch_x_enc, batch_x_mark_enc, batch_x_dec, batch_x_mark_dec, batch_y = [d.to(device) for d in
                                                                                     [batch_x_enc, batch_x_mark_enc,
                                                                                      batch_x_dec, batch_x_mark_dec,
                                                                                      batch_y]]
            optimizer.zero_grad();
            outputs = model(batch_x_enc, batch_x_mark_enc, batch_x_dec, batch_x_mark_dec);
            loss = criterion(outputs, batch_y);
            loss.backward();
            optimizer.step()

        model.eval()
        val_loss_epoch = []
        with torch.no_grad():
            for (batch_x_enc, batch_x_mark_enc, batch_x_dec, batch_x_mark_dec, batch_y) in val_loader:
                batch_x_enc, batch_x_mark_enc, batch_x_dec, batch_x_mark_dec, batch_y = [d.to(device) for d in
                                                                                         [batch_x_enc, batch_x_mark_enc,
                                                                                          batch_x_dec, batch_x_mark_dec,
                                                                                          batch_y]]
                outputs = model(batch_x_enc, batch_x_mark_enc, batch_x_dec, batch_x_mark_dec);
                loss = criterion(outputs, batch_y);
                val_loss_epoch.append(loss.item())

        avg_val_loss = np.average(val_loss_epoch)
        print(f"Epoch [{epoch + 1}/{configs.train_epochs}] -> Val Loss: {avg_val_loss:.4f}")
        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered.");
            break

    # --- Final Testing ---
    # (Same testing loop as before)
    print("\n--- Loading best model for final testing ---")
    best_model_path = os.path.join(checkpoints_path, 'checkpoint.pth')
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        print("Warning: Checkpoint not found.")

    # ... The rest of the testing and result saving code ...
    print(f"\n--- Final testing for {configs.model} ---")
    model.eval()
    preds_list = []
    trues_list = []
    with torch.no_grad():
        for i, (batch_x_enc, batch_x_mark_enc, batch_x_dec, batch_x_mark_dec, batch_y) in enumerate(test_loader):
            batch_x_enc, batch_x_mark_enc, batch_x_dec, batch_x_mark_dec, batch_y = [d.to(device) for d in
                                                                                     [batch_x_enc, batch_x_mark_enc,
                                                                                      batch_x_dec, batch_x_mark_dec,
                                                                                      batch_y]]
            outputs = model(batch_x_enc, batch_x_mark_enc, batch_x_dec, batch_x_mark_dec)
            preds_list.append(outputs.detach().cpu().numpy())
            trues_list.append(batch_y.detach().cpu().numpy())
    preds = np.concatenate(preds_list, axis=0)
    trues = np.concatenate(trues_list, axis=0)
    mae, mse, rmse, mape, mspe = metric(preds, trues)
    print(f'Final Test Metrics: MSE:{mse:.4f}, MAE:{mae:.4f}')


if __name__ == '__main__':
    freeze_support()
    run_experiment()
