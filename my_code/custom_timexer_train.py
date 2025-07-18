import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from argparse import Namespace
import os
import time
from multiprocessing import freeze_support  # For Windows

from models.TimeXer_custom import Model as TimeXerModel
from data_provider.custom_dataset import PreprocessedDataset
from utils.metrics import metric
from utils.tools import visual


# --- Load Data Function ---
def load_data(data_type, base_path="../my_data/train70_val10_test20_winlen336_stride24_workdays/"):
    X_history = np.load(f"{base_path}{data_type}/X_history_target.npy")
    X_known_past = np.load(f"{base_path}{data_type}/X_known_past_exog_features.npy")
    X_known_future = np.load(f"{base_path}{data_type}/X_known_future_exog_features.npy")
    y = np.load(f"{base_path}{data_type}/y_target.npy")
    interval_dates = np.load(f"{base_path}{data_type}/interval_starts.npy", allow_pickle=True)
    interval_dates = pd.to_datetime(interval_dates)
    return X_history, X_known_past, X_known_future, y, interval_dates


# --- Early Stopping Class ---
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
        # Ensure the directory for the path exists
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def main():

    # Load train, validation, and test data
    X_history_train, X_known_past_train, X_known_future_train, y_train, interval_dates_train = load_data("train")
    X_history_val, X_known_past_val, X_known_future_val, y_val, interval_dates_val = load_data("val")
    X_history_test, X_known_past_test, X_known_future_test, y_test, interval_dates_test = load_data("test")

    # --- Configuration for TimeXer ---
    win_len = 336
    stride = 24
    model_id = f"CustomTimexer_winlen{win_len}_stride{stride}_workdays_" + time.strftime("%Y%m%d_%H%M%S")
    checkpoints_path = f'../checkpoints/{model_id}/'
    results_path = f'../results/{model_id}/'
    test_results_figures_path = f'../test_results/{model_id}/'

    # Create directories if they don't exist
    os.makedirs(checkpoints_path, exist_ok=True)
    os.makedirs(results_path, exist_ok=True)
    os.makedirs(test_results_figures_path, exist_ok=True)

    configs = Namespace(
        task_name='long_term_forecast',
        features='MS',
        seq_len=336,
        pred_len=24,
        label_len=0,
        patch_len=24,
        enc_in=14,
        dec_in=13,
        c_out=1,
        d_model=128,
        n_heads=16,
        e_layers=1,
        d_ff=1024,
        dropout=0.1,
        activation='gelu',
        factor=3,
        embed='fixed',
        freq='h',
        use_norm=False,
        inverse=False,
        checkpoints=checkpoints_path.rsplit('/', 1)[0] + '/',
        model_id=model_id,
        model='TimeXerCustom',
        data='customEV',
        des='CustomRun',
        patience=8,
        learning_rate=0.0004376,
        train_epochs=300,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_dataset = PreprocessedDataset(X_history_train, X_known_past_train, X_known_future_train, y_train)
    val_dataset = PreprocessedDataset(X_history_val, X_known_past_val, X_known_future_val, y_val)
    test_dataset = PreprocessedDataset(X_history_test, X_known_past_test, X_known_future_test, y_test)

    batch_size = 32
    # num_workers to 0 for Windows >>>>
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0,
                              pin_memory=True if device.type == 'cuda' else False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                            pin_memory=True if device.type == 'cuda' else False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                             pin_memory=True if device.type == 'cuda' else False)

    model = TimeXerModel(configs).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=configs.learning_rate)

    # Correct path for early stopping checkpoint
    early_stopping_checkpoint_path = os.path.join(checkpoints_path, 'checkpoint.pth')
    early_stopping = EarlyStopping(patience=configs.patience, verbose=True, path=early_stopping_checkpoint_path)

    print(f"Starting training for setting: {model_id}")
    for epoch in range(configs.train_epochs):
        model.train()
        train_loss_epoch = []
        epoch_time_start = time.time()

        for i, (batch_x_enc, batch_x_mark_enc, batch_x_dec, batch_x_mark_dec, batch_y) in enumerate(train_loader):
            batch_x_enc = batch_x_enc.to(device)
            batch_x_mark_enc = batch_x_mark_enc.to(device)
            batch_x_dec = batch_x_dec.to(device)
            batch_x_mark_dec = batch_x_mark_dec.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x_enc, batch_x_mark_enc, batch_x_dec, batch_x_mark_dec)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss_epoch.append(loss.item())

            if (i + 1) % 100 == 0:
                print(
                    f"\tEpoch {epoch + 1}, Iteration {i + 1}/{len(train_loader)} | Current Batch Loss: {loss.item():.7f}")

        avg_train_loss = np.average(train_loss_epoch) if train_loss_epoch else 0
        epoch_duration = time.time() - epoch_time_start

        model.eval()
        val_loss_epoch = []
        with torch.no_grad():
            for batch_x_enc, batch_x_mark_enc, batch_x_dec, batch_x_mark_dec, batch_y in val_loader:
                batch_x_enc = batch_x_enc.to(device)
                batch_x_mark_enc = batch_x_mark_enc.to(device)
                batch_x_dec = batch_x_dec.to(device)
                batch_x_mark_dec = batch_x_mark_dec.to(device)
                batch_y = batch_y.to(device)
                outputs = model(batch_x_enc, batch_x_mark_enc, batch_x_dec, batch_x_mark_dec)
                loss = criterion(outputs, batch_y)
                val_loss_epoch.append(loss.item())

        avg_val_loss = np.average(val_loss_epoch) if val_loss_epoch else 0
        print(
            f"Epoch [{epoch + 1}/{configs.train_epochs}] ({epoch_duration:.2f}s) -> Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    print("Loading best model for testing...")
    best_model_path = early_stopping_checkpoint_path  # Use the path from early stopping
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))  # Add map_location
    else:
        print(f"Warning: Best model checkpoint not found at {best_model_path}. Using last state of the model.")

    print("\n--- Starting Testing ---")
    model.eval()
    preds_list = []
    trues_list = []
    first_batch_input_x_for_plot = None

    with torch.no_grad():
        for i, (batch_x_enc, batch_x_mark_enc, batch_x_dec, batch_x_mark_dec, batch_y) in enumerate(test_loader):
            batch_x_enc_device = batch_x_enc.to(device)
            batch_x_mark_enc_device = batch_x_mark_enc.to(device)
            batch_x_dec_device = batch_x_dec.to(device)
            batch_x_mark_dec_device = batch_x_mark_dec.to(device)
            outputs = model(batch_x_enc_device, batch_x_mark_enc_device, batch_x_dec_device, batch_x_mark_dec_device)
            preds_list.append(outputs.detach().cpu().numpy())
            trues_list.append(batch_y.numpy())
            if i == 0 and batch_x_enc.shape[0] > 0:
                first_batch_input_x_for_plot = batch_x_enc[0, :, -1].numpy()

    preds = np.concatenate(preds_list, axis=0)
    trues = np.concatenate(trues_list, axis=0)
    print('Test shapes after concatenation: preds={}, trues={}'.format(preds.shape, trues.shape))

    mae, mse, rmse, mape, mspe = metric(preds, trues)
    print(f'Test Metrics: MSE:{mse:.4f}, MAE:{mae:.4f}, RMSE:{rmse:.4f}, MAPE:{mape:.4f}, MSPE:{mspe:.4f}')

    np.save(os.path.join(results_path, 'metrics.npy'), np.array([mae, mse, rmse, mape, mspe]))
    np.save(os.path.join(results_path, 'pred.npy'), preds)
    np.save(os.path.join(results_path, 'true.npy'), trues)

    if len(interval_dates_test) >= preds.shape[0]:
        pred_start_datetimes_test = interval_dates_test[:preds.shape[0]]
        np.save(os.path.join(results_path, 'pred_start_dates.npy'), np.array(pred_start_datetimes_test, dtype=object))
    else:
        print(
            f"Warning: Not enough interval_dates_test ({len(interval_dates_test)}) for all predictions ({preds.shape[0]}). Dates not saved.")

    if preds.shape[0] > 0 and first_batch_input_x_for_plot is not None:
        try:
            gt_plot = np.concatenate((first_batch_input_x_for_plot, trues[0, :, 0]), axis=0)
            pd_plot = np.concatenate((first_batch_input_x_for_plot, preds[0, :, 0]), axis=0)
            visual(gt_plot, pd_plot, os.path.join(test_results_figures_path, 'sample_0_visualization.pdf'))
            print(f"Saved visualization for the first test sample to {test_results_figures_path}")
        except Exception as e:
            print(f"Could not generate visualization: {e}")
    else:
        print("Not enough data to generate visualization or first_batch_input_x_for_plot was not captured.")

    result_log_file = "result_custom_long_term_forecast.txt"
    with open(result_log_file, 'a') as f:
        f.write(model_id + "  \n")
        f.write(f'mse:{mse:.4f}, mae:{mae:.4f}, rmse:{rmse:.4f}, mape:{mape:.4f}, mspe:{mspe:.4f}')
        f.write('\n')
        f.write('\n')
    print(f"Results logged to {result_log_file}")
    print(f"--- Experiment Finished: {model_id} ---")


if __name__ == '__main__':
    freeze_support()  # Important for Windows when using multiprocessing with spawn
    main()