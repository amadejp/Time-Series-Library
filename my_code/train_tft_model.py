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

# --- IMPORTANT: Ensure models.TFT is available and you've made the prerequisite change ---
# And ensure your custom dataset provider is in the path
from models.TemporalFusionTransformer import Model as TFTModel
from data_provider.custom_dataset import PreprocessedDataset
from utils.metrics import metric
from utils.tools import visual


# --- Load Data Function (same as your script) ---
def load_data(data_type, base_path="../my_data/train70_val10_test20_winlen336_stride24_workdays/"):
    """Loads preprocessed data for a given type (train, val, test)."""
    print(f"Loading data for: {data_type} from {base_path}{data_type}")
    X_history = np.load(f"{base_path}{data_type}/X_history_target.npy")
    X_known_past = np.load(f"{base_path}{data_type}/X_known_past_exog_features.npy")
    X_known_future = np.load(f"{base_path}{data_type}/X_known_future_exog_features.npy")
    y = np.load(f"{base_path}{data_type}/y_target.npy")
    interval_dates = np.load(f"{base_path}{data_type}/interval_starts.npy", allow_pickle=True)
    interval_dates = pd.to_datetime(interval_dates)
    print(f"Data shapes for {data_type}: X_history={X_history.shape}, y={y.shape}")
    return X_history, X_known_past, X_known_future, y, interval_dates


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

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
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decreases."""
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')

        # Only create a directory if a directory path is specified
        dir_name = os.path.dirname(self.path)
        if dir_name:  # This check prevents calling makedirs with an empty string
            os.makedirs(dir_name, exist_ok=True)

        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def main():
    # --- Load Data ---
    print("--- Loading Data ---")
    X_history_train, X_known_past_train, X_known_future_train, y_train, interval_dates_train = load_data("train")
    X_history_val, X_known_past_val, X_known_future_val, y_val, interval_dates_val = load_data("val")
    X_history_test, X_known_past_test, X_known_future_test, y_test, interval_dates_test = load_data("test")

    # --- Configuration for TFT ---
    win_len = 336
    stride = 24
    model_id = f"TFT_winlen{win_len}_stride{stride}_workdays_" + time.strftime("%Y%m%d_%H%M%S")
    checkpoints_path = f'../checkpoints/{model_id}/'
    results_path = f'../results/{model_id}/'
    test_results_figures_path = f'../test_results/{model_id}/'

    # Create directories if they don't exist
    os.makedirs(checkpoints_path, exist_ok=True)
    os.makedirs(results_path, exist_ok=True)
    os.makedirs(test_results_figures_path, exist_ok=True)

    configs = Namespace(
        task_name='long_term_forecast',
        # --- Model Configs ---
        model='TFT',
        data='customEV',  # This must match the key you added to datatype_dict in TFT.py
        features='MS',
        seq_len=336,
        pred_len=24,
        label_len=0,  # Your PreprocessedDataset likely handles this split already
        enc_in=14,  # Total number of features in the input encoder (1 target + 13 covariates)
        c_out=1,  # The number of final output features (just the target)
        d_model=128,
        n_heads=4,  # TFT often uses fewer heads than standard transformers
        e_layers=2,  # TFT has its own complex block structure, often 1-2 layers are enough
        d_ff=512,
        dropout=0.12,
        activation='gelu',
        embed='timeF',  # timeF is standard for TFT
        freq='h',
        # --- Training Configs ---
        patience=8,
        learning_rate=0.0002,
        train_epochs=100,
        # --- Path Configs ---
        checkpoints=checkpoints_path,
        model_id=model_id,
        des='CustomTFTRun',
    )

    # --- Device Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Dataset and DataLoader ---
    train_dataset = PreprocessedDataset(X_history_train, X_known_past_train, X_known_future_train, y_train)
    val_dataset = PreprocessedDataset(X_history_val, X_known_past_val, X_known_future_val, y_val)
    test_dataset = PreprocessedDataset(X_history_test, X_known_past_test, X_known_future_test, y_test)

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0,
                              pin_memory=True if device.type == 'cuda' else False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                            pin_memory=True if device.type == 'cuda' else False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                             pin_memory=True if device.type == 'cuda' else False)

    # --- Model, Criterion, Optimizer, Early Stopping ---
    model = TFTModel(configs).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=configs.learning_rate)

    early_stopping_checkpoint_path = os.path.join(checkpoints_path, 'checkpoint.pth')
    early_stopping = EarlyStopping(patience=configs.patience, verbose=True, path=early_stopping_checkpoint_path)

    print(f"--- Starting Training for TFT: {model_id} ---")

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

            # --- KEY CHANGE FOR TFT ---
            # The TFT model from the library returns predictions for all input features.
            # We assume the target is the last feature, based on your TimeXer setup.
            # We slice the output to get only the predictions for the target variable over the prediction horizon.
            # output shape: [batch_size, seq_len + pred_len, enc_in]
            # We need: [batch_size, pred_len, 1] for the target feature
            f_dim = -1  # Assuming target is the last dimension
            outputs = outputs[:, -configs.pred_len:, f_dim:]

            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss_epoch.append(loss.item())

            if (i + 1) % 100 == 0:
                print(f"\tEpoch {epoch + 1}, Iteration {i + 1}/{len(train_loader)} | Batch Loss: {loss.item():.7f}")

        avg_train_loss = np.average(train_loss_epoch)
        epoch_duration = time.time() - epoch_time_start

        # --- Validation ---
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

                # Apply the same slicing for validation
                f_dim = -1
                outputs = outputs[:, -configs.pred_len:, f_dim:]

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
    best_model_path = early_stopping_checkpoint_path
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        print(f"Warning: Checkpoint not found at {best_model_path}. Using last model state.")

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

            # Apply the same slicing for testing
            f_dim = -1
            outputs = outputs[:, -configs.pred_len:, f_dim:]

            preds_list.append(outputs.detach().cpu().numpy())
            trues_list.append(batch_y.numpy())
            if i == 0 and batch_x_enc.shape[0] > 0:
                first_batch_input_x_for_plot = batch_x_enc[0, :, -1].cpu().numpy()

    preds = np.concatenate(preds_list, axis=0)
    trues = np.concatenate(trues_list, axis=0)
    print('Test shapes: preds={}, trues={}'.format(preds.shape, trues.shape))

    mae, mse, rmse, mape, mspe = metric(preds, trues)
    print(f'Test Metrics: MSE:{mse:.4f}, MAE:{mae:.4f}, RMSE:{rmse:.4f}, MAPE:{mape:.4f}, MSPE:{mspe:.4f}')

    # --- Save Results ---
    np.save(os.path.join(results_path, 'metrics.npy'), np.array([mae, mse, rmse, mape, mspe]))
    np.save(os.path.join(results_path, 'pred.npy'), preds)
    np.save(os.path.join(results_path, 'true.npy'), trues)

    # ... (rest of the saving and visualization code is the same)
    if len(interval_dates_test) >= preds.shape[0]:
        pred_start_datetimes_test = interval_dates_test[:preds.shape[0]]
        np.save(os.path.join(results_path, 'pred_start_dates.npy'), np.array(pred_start_datetimes_test, dtype=object))
    else:
        print(f"Warning: Not enough test dates ({len(interval_dates_test)}) for all predictions ({preds.shape[0]}).")

    if preds.shape[0] > 0 and first_batch_input_x_for_plot is not None:
        try:
            gt_plot = np.concatenate((first_batch_input_x_for_plot, trues[0, :, 0]), axis=0)
            pd_plot = np.concatenate((first_batch_input_x_for_plot, preds[0, :, 0]), axis=0)
            visual(gt_plot, pd_plot, os.path.join(test_results_figures_path, 'sample_0_visualization.pdf'))
            print(f"Saved visualization for the first test sample to {test_results_figures_path}")
        except Exception as e:
            print(f"Could not generate visualization: {e}")
    else:
        print("Not enough data to generate visualization.")

    result_log_file = "result_tft_long_term_forecast.txt"
    with open(result_log_file, 'a') as f:
        f.write(model_id + "  \n")
        f.write(f'mse:{mse:.4f}, mae:{mae:.4f}, rmse:{rmse:.4f}, mape:{mape:.4f}, mspe:{mspe:.4f}')
        f.write('\n\n')
    print(f"Results logged to {result_log_file}")
    print(f"--- Experiment Finished: {model_id} ---")


if __name__ == '__main__':
    freeze_support()  # Recommended for Windows
    main()
