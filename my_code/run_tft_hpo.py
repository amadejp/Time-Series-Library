import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from argparse import Namespace
import os
import time
from multiprocessing import freeze_support
import optuna

# --- IMPORTANT: Ensure these modules are in your Python path ---
from models.TemporalFusionTransformer import Model as TFTModel
from data_provider.custom_dataset import PreprocessedDataset
from utils.metrics import metric


# --- Helper Functions from your original script (no changes needed) ---

def load_data(data_type, base_path="../my_data/train70_val10_test20_winlen336_stride24_workdays/"):
    """Loads preprocessed data for a given type (train, val, test)."""
    print(f"Loading data for: {data_type} from {base_path}{data_type}")
    X_history = np.load(f"{base_path}{data_type}/X_history_target.npy")
    X_known_past = np.load(f"{base_path}{data_type}/X_known_past_exog_features.npy")
    X_known_future = np.load(f"{base_path}{data_type}/X_known_future_exog_features.npy")
    y = np.load(f"{base_path}{data_type}/y_target.npy")
    return X_history, X_known_past, X_known_future, y


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


# --- The Objective Function for Optuna ---
def objective(trial, train_data, val_data, base_configs, device):
    """
    This function trains a model for one set of hyperparameters and returns the validation loss.
    """
    # --- 1. Define Hyperparameter Search Space ---
    trial_configs = base_configs
    trial_configs.learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    trial_configs.dropout = trial.suggest_float("dropout", 0.05, 0.4)
    trial_configs.d_model = trial.suggest_categorical("d_model", [64, 128, 256])

    # Step 1: Suggest n_heads from a superset of all possible values.
    n_heads = trial.suggest_categorical("n_heads", [2, 4, 8, 16])

    # Step 2: Check if the combination is valid. If not, prune the trial.
    if trial_configs.d_model % n_heads != 0:
        # This combination is invalid, so we stop this trial early.
        raise optuna.exceptions.TrialPruned()

    trial_configs.n_heads = n_heads
    trial_configs.d_ff = trial.suggest_categorical("d_ff", [256, 512, 1024])
    trial_configs.e_layers = trial.suggest_int("e_layers", 1, 2)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

    print(f"\n--- Starting Trial {trial.number} ---")
    print(f"  Params: lr={trial_configs.learning_rate:.6f}, dropout={trial_configs.dropout:.2f}, "
          f"d_model={trial_configs.d_model}, n_heads={trial_configs.n_heads}, d_ff={trial_configs.d_ff}, "
          f"e_layers={trial_configs.e_layers}, batch_size={batch_size}")

    # --- 2. Setup Dataloaders, Model, Optimizer for this trial ---
    # (The rest of your function remains exactly the same)
    X_history_train, X_known_past_train, X_known_future_train, y_train = train_data
    X_history_val, X_known_past_val, X_known_future_val, y_val = val_data

    train_dataset = PreprocessedDataset(X_history_train, X_known_past_train, X_known_future_train, y_train)
    val_dataset = PreprocessedDataset(X_history_val, X_known_past_val, X_known_future_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0,
                              pin_memory=True if device.type == 'cuda' else False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                            pin_memory=True if device.type == 'cuda' else False)

    model = TFTModel(trial_configs).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=trial_configs.learning_rate)

    early_stopping = EarlyStopping(patience=trial_configs.patience, verbose=False)

    # --- 3. Training and Validation Loop ---
    for epoch in range(trial_configs.train_epochs):
        model.train()
        for i, (batch_x_enc, batch_x_mark_enc, batch_x_dec, batch_x_mark_dec, batch_y) in enumerate(train_loader):
            batch_x_enc, batch_x_mark_enc, batch_x_dec, batch_x_mark_dec, batch_y = \
                map(lambda x: x.to(device), [batch_x_enc, batch_x_mark_enc, batch_x_dec, batch_x_mark_dec, batch_y])

            optimizer.zero_grad()
            outputs = model(batch_x_enc, batch_x_mark_enc, batch_x_dec, batch_x_mark_dec)
            f_dim = -1
            outputs = outputs[:, -trial_configs.pred_len:, f_dim:]
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        val_loss_epoch = []
        with torch.no_grad():
            for batch_x_enc, batch_x_mark_enc, batch_x_dec, batch_x_mark_dec, batch_y in val_loader:
                batch_x_enc, batch_x_mark_enc, batch_x_dec, batch_x_mark_dec, batch_y = \
                    map(lambda x: x.to(device), [batch_x_enc, batch_x_mark_enc, batch_x_dec, batch_x_mark_dec, batch_y])

                outputs = model(batch_x_enc, batch_x_mark_enc, batch_x_dec, batch_x_mark_dec)
                f_dim = -1
                outputs = outputs[:, -trial_configs.pred_len:, f_dim:]
                loss = criterion(outputs, batch_y)
                val_loss_epoch.append(loss.item())

        avg_val_loss = np.average(val_loss_epoch)
        print(f"  Epoch {epoch + 1}/{trial_configs.train_epochs}, Val Loss: {avg_val_loss:.6f}")

        # --- 4. Report to Optuna and Handle Pruning/Early Stopping ---
        trial.report(avg_val_loss, epoch)
        if trial.should_prune():
            print(f"  Trial {trial.number} pruned by Optuna pruner at epoch {epoch + 1}.")
            raise optuna.exceptions.TrialPruned()

        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print(f"  Early stopping triggered at epoch {epoch + 1}.")
            break

    # --- 5. Return the best validation loss for this trial ---
    return early_stopping.val_loss_min


# --- Main HPO Runner ---
def run_hpo():
    # --- Load Data Once ---
    print("--- Loading Data for HPO ---")
    train_data = load_data("train")
    val_data = load_data("val")

    # --- Base Configurations (parameters we are NOT tuning) ---
    base_configs = Namespace(
        task_name='long_term_forecast',
        model='TFT',
        data='customEV',
        features='MS',
        seq_len=336,
        pred_len=24,
        label_len=0,
        enc_in=14,
        c_out=1,
        activation='gelu',
        embed='timeF',
        freq='h',
        patience=8,  # Patience for EarlyStopping within a trial
        train_epochs=50,  # Set a reasonable max epoch for each trial
        des='TFT_HPO_Run',
    )

    # --- Device Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Optuna Study Setup ---
    study_name = "TFT_HPO_" + time.strftime("%Y%m%d_%H%M%S")
    # We use a pruner to stop unpromising trials early
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10, interval_steps=1)
    study = optuna.create_study(direction="minimize", study_name=study_name, pruner=pruner)

    # Wrap the objective function to pass the static arguments (data, configs, device)
    objective_with_args = lambda trial: objective(trial, train_data, val_data, base_configs, device)

    # --- Start the HPO ---
    print(f"\n--- Starting Optuna Study: {study_name} ---")
    n_trials = 50  # Set the number of HPO trials you want to run
    study.optimize(objective_with_args, n_trials=n_trials)

    # --- Print and Save HPO Results ---
    print("\n--- HPO Study Finished ---")
    print(f"Study statistics: ")
    print(f"  Number of finished trials: {len(study.trials)}")

    best_trial = study.best_trial
    print(f"Best trial:")
    print(f"  Value (min validation loss): {best_trial.value:.6f}")
    print(f"  Params: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    # Save the results to a file
    results_dir = "../hpo_results"
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, f"{study_name}_results.txt")
    with open(results_file, 'w') as f:
        f.write(f"Best trial value (min validation loss): {best_trial.value}\n")
        f.write("Best parameters:\n")
        for key, value in best_trial.params.items():
            f.write(f"  {key}: {value}\n")
    print(f"\nBest parameters saved to {results_file}")

    # You can also visualize the study
    try:
        fig1 = optuna.visualization.plot_optimization_history(study)
        fig1.write_image(os.path.join(results_dir, f"{study_name}_history.png"))
        fig2 = optuna.visualization.plot_param_importances(study)
        fig2.write_image(os.path.join(results_dir, f"{study_name}_importance.png"))
        print(f"Visualizations saved to {results_dir}")
        # To display in a script, you might need to call fig.show()
        # You may need to `pip install plotly kaleido` for image export
    except Exception as e:
        print(
            f"Could not generate visualizations. Install plotly and kaleido: 'pip install plotly kaleido'. Error: {e}")


if __name__ == '__main__':
    freeze_support()  # Recommended for Windows
    run_hpo()