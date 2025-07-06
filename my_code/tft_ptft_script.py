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
# We only need the TemporalFusionTransformer for this script
from models.TemporalFusionTransformer import Model as TFTModel
from utils.metrics import metric
from utils.tools import visual


# --- Custom Dataset and Early Stopping Classes (Unchanged) ---
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
        # Encoder input: Concatenate target history and past known features along the feature dimension
        x_enc = torch.cat([self.x_history[idx], self.x_known_past[idx]], dim=-1)
        batch_y = self.y[idx]

        # These are the time-varying features (exogenous)
        x_mark_enc = self.x_known_past[idx]

        # --- FIX IS HERE ---
        # 1. Create the decoder's time features (x_mark_dec) to cover both the label and prediction lengths
        # It needs the features from the end of the encoder's window and the future window.
        dec_inp_mark_start = self.x_known_past[idx][-self.label_len:, :] if self.label_len > 0 else torch.empty(0,
                                                                                                                self.x_known_past.shape[
                                                                                                                    -1])
        x_mark_dec = torch.cat([dec_inp_mark_start, self.x_known_future[idx]],
                               dim=0)  # Shape will now be (72, num_exog_features)

        # 2. Create the decoder's target input (placeholders + history)
        dec_inp_val_start = self.x_history[idx][-self.label_len:, :] if self.label_len > 0 else torch.empty(0,
                                                                                                            self.x_history.shape[
                                                                                                                -1])
        dec_inp_future_target = torch.zeros(self.pred_len, self.y.shape[-1], dtype=torch.float32)
        dec_inp_target = torch.cat([dec_inp_val_start, dec_inp_future_target],
                                   dim=0)  # Shape is (72, num_target_features)

        # 3. Create the final decoder input (x_dec) by concatenating the two tensors from above.
        # This now works because both `dec_inp_target` and `x_mark_dec` have a sequence length of 72.
        x_dec = torch.cat([dec_inp_target, x_mark_dec], dim=-1)

        return x_enc, x_mark_enc, x_dec, x_mark_dec, batch_y


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


# --- Model Configuration Function (Adapted for TFT) ---
def get_model_and_configs(model_name):
    base_configs = {
        'task_name': 'long_term_forecast', 'features': 'MS', 'seq_len': 336, 'pred_len': 24,
        'c_out': 1, 'freq': 'h', 'patience': 8, 'train_epochs': 100, 'label_len': 48,
    }

    # Default configurations specifically for the TFT model
    default_configs = {
        'TFT': {
            'model': 'TFT',
            'data': 'customEV',
            'd_model': 256,
            'n_heads': 8,
            'e_layers': 2,
            'd_ff': 512,
            'dropout': 0.34,
            'activation': 'gelu',
            'embed': 'timeF',
            # enc_in will be set dynamically based on data
        }
    }

    MODEL_CLASSES = {
        'TFT': TFTModel,
    }

    if model_name not in MODEL_CLASSES:
        raise ValueError(f"Model '{model_name}' not recognized.")

    final_configs = {**base_configs, **default_configs[model_name]}
    return MODEL_CLASSES[model_name], Namespace(**final_configs)


def train_one_phase(phase_name, model, configs, train_loader, val_loader, device, learning_rate):
    """A helper function to run one phase of training (pre-training or fine-tuning)."""
    checkpoints_path = os.path.join(configs.checkpoints_base, phase_name)
    os.makedirs(checkpoints_path, exist_ok=True)

    print(f"\n--- Starting {phase_name.upper()} Phase for {configs.model} ---")
    print(f"Using learning rate: {learning_rate}")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    early_stopping = EarlyStopping(patience=configs.patience, verbose=True,
                                   path=os.path.join(checkpoints_path, 'checkpoint.pth'))

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

            # !!! KEY CHANGE FOR TFT !!!
            # The model outputs predictions for all features. We only need the target.
            # Assuming the target variable is the last one in the output.
            outputs = outputs[:, -configs.pred_len:, -1:]

            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss_epoch.append(loss.item())

        model.eval()
        val_loss_epoch = []
        with torch.no_grad():
            for (batch_x_enc, batch_x_mark_enc, batch_x_dec, batch_x_mark_dec, batch_y) in val_loader:
                batch_x_enc, batch_x_mark_enc, batch_x_dec, batch_x_mark_dec, batch_y = [d.to(device) for d in
                                                                                         [batch_x_enc, batch_x_mark_enc,
                                                                                          batch_x_dec, batch_x_mark_dec,
                                                                                          batch_y]]
                outputs = model(batch_x_enc, batch_x_mark_enc, batch_x_dec, batch_x_mark_dec)

                # !!! KEY CHANGE FOR TFT (Validation) !!!
                outputs = outputs[:, -configs.pred_len:, -1:]

                loss = criterion(outputs, batch_y)
                val_loss_epoch.append(loss.item())

        avg_val_loss = np.average(val_loss_epoch)
        print(f"Epoch [{epoch + 1}/{configs.train_epochs}] -> {phase_name} Val Loss: {avg_val_loss:.4f}")
        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print(f"Early stopping triggered for {phase_name} phase.")
            break

    best_model_path = os.path.join(checkpoints_path, 'checkpoint.pth')
    return best_model_path


def run_finetuning_experiment():
    # --- TOP-LEVEL CONTROLS ---
    model_name = 'TFT'
    pretrain_data_path = "../my_data/caltech/train70_val10_test20_winlen336_stride24_workdays/"
    finetune_data_path = "../my_data/train70_val10_test20_winlen336_stride24_workdays/"

    pretrain_lr = 1e-4
    finetune_lr = 0.0004  # CRITICAL: Use a smaller LR for fine-tuning
    batch_size = 32

    # --- Get base model configs ---
    ModelClass, configs = get_model_and_configs(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Create a unique ID and paths for this entire experiment run ---
    experiment_id = f"{configs.model}_finetune_{time.strftime('%Y%m%d_%H%M%S')}"
    configs.checkpoints_base = f'../checkpoints/{experiment_id}/'
    results_path = f'../results/{experiment_id}/'
    os.makedirs(results_path, exist_ok=True)

    # =================================================================
    # --- PHASE 1: PRE-TRAINING on the large dataset ---
    # =================================================================
    print(f"--- Loading Pre-training Data from: {pretrain_data_path} ---")
    pt_X_h_train, pt_X_kp_train, pt_X_kf_train, pt_y_train = [np.load(f"{pretrain_data_path}train/{fname}") for fname in
                                                              ["X_history_target.npy", "X_known_past_exog_features.npy",
                                                               "X_known_future_exog_features.npy", "y_target.npy"]]
    pt_X_h_val, pt_X_kp_val, pt_X_kf_val, pt_y_val = [np.load(f"{pretrain_data_path}val/{fname}") for fname in
                                                      ["X_history_target.npy", "X_known_past_exog_features.npy",
                                                       "X_known_future_exog_features.npy", "y_target.npy"]]

    # --- DYNAMICALLY SET enc_in and dec_in FOR TFT ---
    # enc_in = number of target variables + number of past covariates
    # dec_in = number of target variables + number of future covariates
    configs.enc_in = pt_X_h_train.shape[-1] + pt_X_kp_train.shape[-1]
    configs.dec_in = pt_X_h_train.shape[-1] + pt_X_kf_train.shape[-1]  # Based on CustomDataset logic
    print(f"Dynamically set model configs: enc_in={configs.enc_in}, dec_in={configs.dec_in}")

    pretrain_dataset = CustomDataset(pt_X_h_train, pt_X_kp_train, pt_X_kf_train, pt_y_train, configs)
    pretrain_val_dataset = CustomDataset(pt_X_h_val, pt_X_kp_val, pt_X_kf_val, pt_y_val, configs)
    pretrain_loader = DataLoader(pretrain_dataset, batch_size=batch_size, shuffle=True)
    pretrain_val_loader = DataLoader(pretrain_val_dataset, batch_size=batch_size, shuffle=False)

    model = ModelClass(configs).to(device)

    best_pretrain_model_path = train_one_phase(
        phase_name="pretrain", model=model, configs=configs,
        train_loader=pretrain_loader, val_loader=pretrain_val_loader,
        device=device, learning_rate=pretrain_lr
    )

    # =================================================================
    # --- PHASE 2: FINE-TUNING on the target dataset ---
    # =================================================================
    print(f"\n--- Loading Fine-tuning Data from: {finetune_data_path} ---")
    ft_X_h_train, ft_X_kp_train, ft_X_kf_train, ft_y_train = [np.load(f"{finetune_data_path}train/{fname}") for fname in
                                                              ["X_history_target.npy", "X_known_past_exog_features.npy",
                                                               "X_known_future_exog_features.npy", "y_target.npy"]]
    ft_X_h_val, ft_X_kp_val, ft_X_kf_val, ft_y_val = [np.load(f"{finetune_data_path}val/{fname}") for fname in
                                                      ["X_history_target.npy", "X_known_past_exog_features.npy",
                                                       "X_known_future_exog_features.npy", "y_target.npy"]]
    ft_X_h_test, ft_X_kp_test, ft_X_kf_test, ft_y_test = [np.load(f"{finetune_data_path}test/{fname}") for fname in
                                                          ["X_history_target.npy", "X_known_past_exog_features.npy",
                                                           "X_known_future_exog_features.npy", "y_target.npy"]]

    finetune_dataset = CustomDataset(ft_X_h_train, ft_X_kp_train, ft_X_kf_train, ft_y_train, configs)
    finetune_val_dataset = CustomDataset(ft_X_h_val, ft_X_kp_val, ft_X_kf_val, ft_y_val, configs)
    finetune_loader = DataLoader(finetune_dataset, batch_size=batch_size, shuffle=True)
    finetune_val_loader = DataLoader(finetune_val_dataset, batch_size=batch_size, shuffle=False)

    print(f"\nLoading weights from best pre-trained model: {best_pretrain_model_path}")
    # Re-initialize model with the same configs, then load weights
    model = ModelClass(configs).to(device)
    model.load_state_dict(torch.load(best_pretrain_model_path, map_location=device))

    best_finetune_model_path = train_one_phase(
        phase_name="finetune", model=model, configs=configs,
        train_loader=finetune_loader, val_loader=finetune_val_loader,
        device=device, learning_rate=finetune_lr
    )

    # =================================================================
    # --- FINAL EVALUATION on the fine-tuned test set ---
    # =================================================================
    print("\n--- Loading best fine-tuned model for final testing ---")
    model.load_state_dict(torch.load(best_finetune_model_path, map_location=device))

    test_dataset = CustomDataset(ft_X_h_test, ft_X_kp_test, ft_X_kf_test, ft_y_test, configs)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    preds_list, trues_list = [], []
    with torch.no_grad():
        for (batch_x_enc, batch_x_mark_enc, batch_x_dec, batch_x_mark_dec, batch_y) in test_loader:
            batch_x_enc, batch_x_mark_enc, batch_x_dec, batch_x_mark_dec, batch_y = [d.to(device) for d in
                                                                                     [batch_x_enc, batch_x_mark_enc,
                                                                                      batch_x_dec, batch_x_mark_dec,
                                                                                      batch_y]]
            outputs = model(batch_x_enc, batch_x_mark_enc, batch_x_dec, batch_x_mark_dec)

            # !!! KEY CHANGE FOR TFT (Final Test) !!!
            outputs = outputs[:, -configs.pred_len:, -1:]

            preds_list.append(outputs.detach().cpu().numpy())
            trues_list.append(batch_y.detach().cpu().numpy())

    preds = np.concatenate(preds_list, axis=0)
    trues = np.concatenate(trues_list, axis=0)

    mae, mse, rmse, mape, mspe = metric(preds, trues)

    print(f"\n--- EXPERIMENT FINISHED: {experiment_id} ---")
    print(f'Final Test Metrics on Fine-tuned Data: MSE:{mse:.4f}, MAE:{mae:.4f}')

    print(f"Saving results to {results_path}")
    np.save(os.path.join(results_path, 'metrics.npy'), np.array([mae, mse, rmse, mape, mspe]))
    np.save(os.path.join(results_path, 'pred.npy'), preds)
    np.save(os.path.join(results_path, 'true.npy'), trues)

    # Save visualization
    try:
        # We need the original target history for the plot ground truth
        history_for_plot = ft_X_h_test[0, :, 0]
        gt_plot = np.concatenate((history_for_plot, trues[0, :, 0]), axis=0)
        pd_plot = np.concatenate((history_for_plot, preds[0, :, 0]), axis=0)
        visual(gt_plot, pd_plot, os.path.join(results_path, 'sample_0_visualization.pdf'))
        print(f"Saved visualization for the first test sample to {results_path}")
    except Exception as e:
        print(f"Could not generate visualization: {e}")


if __name__ == '__main__':
    freeze_support()
    run_finetuning_experiment()