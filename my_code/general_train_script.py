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

# --- Model Imports ---
# Corrected the import for TFT to match the library's typical structure
from models.TemporalFusionTransformer import Model as TFTModel
from models.Informer import Model as InformerModel
from models.Autoformer import Model as AutoformerModel
from models.PatchTST import Model as PatchTSTModel
from models.DLinear import Model as DLinearModel

from utils.metrics import metric
from utils.tools import visual


# --- Custom Dataset Class for Correct Data Mapping ---
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
        # x_enc: Historical target values. Shape: [seq_len, 1]
        x_enc = self.x_history[idx]

        # y: Future target values. Shape: [pred_len, 1]
        batch_y = self.y[idx]

        # x_mark_enc: Historical known features. Shape: [seq_len, 13]
        x_mark_enc = self.x_known_past[idx]

        # x_mark_dec: Future known features + historical context for decoder
        # Start with the last `label_len` part of historical known features
        dec_inp_mark_start = x_mark_enc[-self.label_len:, :] if self.label_len > 0 else torch.empty(0, x_mark_enc.shape[
            -1])
        # Concatenate with the future known features
        x_mark_dec = torch.cat([dec_inp_mark_start, self.x_known_future[idx]], dim=0)

        # x_dec: Decoder input. A mix of history and placeholder zeros.
        dec_inp_val_start = x_enc[-self.label_len:, :] if self.label_len > 0 else torch.empty(0, x_enc.shape[-1])
        # The placeholder should match the number of target variables (which is 1)
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


def get_model_and_configs(model_name):
    """Returns the model class and its configuration namespace based on the new data structure."""

    # --- Base Configurations ---
    base_configs = {
        'task_name': 'long_term_forecast',
        'features': 'S',  # Single variate predict Single variate
        'seq_len': 336,
        'pred_len': 24,
        'enc_in': 1,  # CRITICAL: Only the target variable is in x_enc
        'dec_in': 1,  # CRITICAL: Only the target variable is in x_dec
        'c_out': 1,  # CRITICAL: We are predicting a single target variable
        'freq': 'h',
        'dropout': 0.1,
        'patience': 8,
        'train_epochs': 100,
    }

    # --- Model-Specific Configurations ---
    model_configs = {
        'TFT': {
            'model': 'TFT',
            'data': 'customEV',
            'label_len': 0,
            'd_model': 128,
            'n_heads': 8,
            'e_layers': 1,
            'd_ff': 512,
            'activation': 'gelu',
            'embed': 'timeF',
            'learning_rate': 0.0001,
        },
        'Informer': {
            'model': 'Informer',
            'label_len': 48,
            'd_model': 512,
            'n_heads': 8,
            'e_layers': 2,
            'd_layers': 1,
            'd_ff': 2048,
            'activation': 'gelu',
            'embed': 'timeF',
            'factor': 3,
            'learning_rate': 0.0001,
        },
        'Autoformer': {
            'model': 'Autoformer',
            'label_len': 48,
            'd_model': 512,
            'n_heads': 8,
            'e_layers': 2,
            'd_layers': 1,
            'd_ff': 2048,
            'activation': 'gelu',
            'embed': 'timeF',
            'factor': 3,
            'learning_rate': 0.0001,
            'moving_avg': 25,
        },
        'PatchTST': {
            'model': 'PatchTST',
            'label_len': 0,
            'd_model': 128,
            'n_heads': 16,
            'e_layers': 3,
            'd_ff': 256,
            'activation': 'gelu',
            'learning_rate': 0.0001,
            'patch_len': 24,
            'stride': 12,
            'revin': 1,
            'factor': 3,
        },
        'DLinear': {
            'model': 'DLinear',
            'label_len': 0,
            'learning_rate': 0.001,
            'individual': False,
        }
    }

    MODEL_CLASSES = {
        'TFT': TFTModel,
        'Informer': InformerModel,
        'Autoformer': AutoformerModel,
        'PatchTST': PatchTSTModel,
        'DLinear': DLinearModel,
    }

    if model_name not in MODEL_CLASSES:
        raise ValueError(f"Model '{model_name}' not recognized. Available: {list(MODEL_CLASSES.keys())}")

    final_configs = {**base_configs, **model_configs[model_name]}
    configs_ns = Namespace(**final_configs)

    return MODEL_CLASSES[model_name], configs_ns


def run_experiment():
    # --- CHOOSE YOUR MODEL HERE ---
    model_name = 'PatchTST'  # Options: 'TFT', 'Informer', 'Autoformer', 'PatchTST', 'DLinear'

    ModelClass, configs = get_model_and_configs(model_name)

    # --- Setup Paths ---
    model_id = f"{configs.model}_seq{configs.seq_len}_pred{configs.pred_len}_" + time.strftime("%Y%m%d_%H%M%S")
    configs.model_id = model_id
    checkpoints_path = f'../checkpoints/{model_id}/'
    results_path = f'../results/{model_id}/'
    test_results_figures_path = f'../test_results/{model_id}/'
    os.makedirs(checkpoints_path, exist_ok=True)
    os.makedirs(results_path, exist_ok=True)
    os.makedirs(test_results_figures_path, exist_ok=True)
    configs.checkpoints = checkpoints_path

    # --- Load Data ---
    print("--- Loading Data ---")
    data_path = "../my_data/train70_val10_test20_winlen336_stride24_workdays/"
    X_history_train = np.load(f"{data_path}train/X_history_target.npy")
    X_known_past_train = np.load(f"{data_path}train/X_known_past_exog_features.npy")
    X_known_future_train = np.load(f"{data_path}train/X_known_future_exog_features.npy")
    y_train = np.load(f"{data_path}train/y_target.npy")

    X_history_val = np.load(f"{data_path}val/X_history_target.npy")
    X_known_past_val = np.load(f"{data_path}val/X_known_past_exog_features.npy")
    X_known_future_val = np.load(f"{data_path}val/X_known_future_exog_features.npy")
    y_val = np.load(f"{data_path}val/y_target.npy")

    X_history_test = np.load(f"{data_path}test/X_history_target.npy")
    X_known_past_test = np.load(f"{data_path}test/X_known_past_exog_features.npy")
    X_known_future_test = np.load(f"{data_path}test/X_known_future_exog_features.npy")
    y_test = np.load(f"{data_path}test/y_target.npy")
    interval_dates_test = pd.to_datetime(np.load(f"{data_path}test/interval_starts.npy", allow_pickle=True))

    # --- Device Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Dataset and DataLoader ---
    train_dataset = CustomDataset(X_history_train, X_known_past_train, X_known_future_train, y_train, configs)
    val_dataset = CustomDataset(X_history_val, X_known_past_val, X_known_future_val, y_val, configs)
    test_dataset = CustomDataset(X_history_test, X_known_past_test, X_known_future_test, y_test, configs)

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # --- Model, Criterion, Optimizer, Early Stopping ---
    model = ModelClass(configs).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=configs.learning_rate)
    early_stopping = EarlyStopping(patience=configs.patience, verbose=True,
                                   path=os.path.join(checkpoints_path, 'checkpoint.pth'))

    print(f"--- Starting Training for {configs.model}: {model_id} ---")

    for epoch in range(configs.train_epochs):
        model.train()
        train_loss_epoch = []
        epoch_time_start = time.time()

        for i, (batch_x_enc, batch_x_mark_enc, batch_x_dec, batch_x_mark_dec, batch_y) in enumerate(train_loader):
            batch_x_enc, batch_x_mark_enc, batch_x_dec, batch_x_mark_dec, batch_y = [d.to(device) for d in
                                                                                     [batch_x_enc, batch_x_mark_enc,
                                                                                      batch_x_dec, batch_x_mark_dec,
                                                                                      batch_y]]

            optimizer.zero_grad()
            outputs = model(batch_x_enc, batch_x_mark_enc, batch_x_dec, batch_x_mark_dec)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
            optimizer.step()
            train_loss_epoch.append(loss.item())

        avg_train_loss = np.average(train_loss_epoch)
        epoch_duration = time.time() - epoch_time_start

        # --- Validation ---
        model.eval()
        val_loss_epoch = []
        with torch.no_grad():
            for batch_x_enc, batch_x_mark_enc, batch_x_dec, batch_x_mark_dec, batch_y in val_loader:
                batch_x_enc, batch_x_mark_enc, batch_x_dec, batch_x_mark_dec, batch_y = [d.to(device) for d in
                                                                                         [batch_x_enc, batch_x_mark_enc,
                                                                                          batch_x_dec, batch_x_mark_dec,
                                                                                          batch_y]]
                outputs = model(batch_x_enc, batch_x_mark_enc, batch_x_dec, batch_x_mark_dec)
                loss = criterion(outputs, batch_y)
                val_loss_epoch.append(loss.item())

        avg_val_loss = np.average(val_loss_epoch)
        print(
            f"Epoch [{epoch + 1}/{configs.train_epochs}] ({epoch_duration:.2f}s) -> Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    # --- Testing ---
    print("\n--- Loading best model for testing ---")
    best_model_path = os.path.join(checkpoints_path, 'checkpoint.pth')
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        print(f"Warning: Checkpoint not found at {best_model_path}. Using last model state.")

    print(f"\n--- Starting Testing for {configs.model} ---")
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
    print('Test shapes: preds={}, trues={}'.format(preds.shape, trues.shape))

    mae, mse, rmse, mape, mspe = metric(preds, trues)
    print(f'Test Metrics for {configs.model}: MSE:{mse:.4f}, MAE:{mae:.4f}')

    # --- Save Results & Visualization ---
    np.save(os.path.join(results_path, 'metrics.npy'), np.array([mae, mse, rmse, mape, mspe]))
    np.save(os.path.join(results_path, 'pred.npy'), preds)
    np.save(os.path.join(results_path, 'true.npy'), trues)

    if X_history_test.shape[0] > 0:
        try:
            gt_plot = np.concatenate((X_history_test[0, :, 0], trues[0, :, 0]), axis=0)
            pd_plot = np.concatenate((X_history_test[0, :, 0], preds[0, :, 0]), axis=0)
            visual(gt_plot, pd_plot, os.path.join(test_results_figures_path, 'sample_0_visualization.pdf'))
            print(f"Saved visualization for the first test sample to {test_results_figures_path}")
        except Exception as e:
            print(f"Could not generate visualization: {e}")

    print(f"--- Experiment Finished: {model_id} ---")


if __name__ == '__main__':
    freeze_support()
    run_experiment()
