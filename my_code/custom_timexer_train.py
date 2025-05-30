# train_script.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from argparse import Namespace

# Assuming timexer_model.py and custom_dataset.py are in the same directory or accessible
from models.TimeXer_custom import Model as TimeXerModel
from data_provider.custom_dataset import PreprocessedDataset

# --- Load Data ---
def load_data(data_type, base_path="../my_data/train70_val10_test20_winlen120/"):
    X_history = np.load(f"{base_path}{data_type}/X_history_target.npy")
    X_known_past = np.load(f"{base_path}{data_type}/X_known_past_exog_features.npy")
    X_known_future = np.load(f"{base_path}{data_type}/X_known_future_exog_features.npy")
    y = np.load(f"{base_path}{data_type}/y_target.npy")
    interval_dates = np.load(f"{base_path}{data_type}/interval_starts.npy", allow_pickle=True)
    interval_dates = pd.to_datetime(interval_dates)
    return X_history, X_known_past, X_known_future, y, interval_dates

# Load train and validation data
X_history_train, X_known_past_train, X_known_future_train, y_train, interval_dates_train = load_data("train")
X_history_val, X_known_past_val, X_known_future_val, y_val, interval_dates_val = load_data("val")

# --- Configuration for TimeXer ---
configs = Namespace(
    task_name='long_term_forecast',
    features='MS',  # Multivariate input (X_known_past+X_history), Single target prediction (y_train)
    seq_len=120,
    pred_len=24,
    label_len=0,  # Not strictly used by TimeXer if x_mark_dec is pred_len long
    patch_len=24,  # From your bash script
    enc_in=14,  # X_known_past (13) + X_history (1) = 14
    dec_in=13,  # Number of features in X_known_future (used for num_future_covariates)
    c_out=1,  # Number of target variables in y_train
    d_model=128,
    n_heads=8,
    e_layers=2,
    d_ff=512,
    dropout=0.15,
    activation='gelu',
    factor=3,
    embed='fixed',  # For DataEmbedding_inverted
    freq='h',  # For DataEmbedding_inverted, might not be critical if not using its time features
    use_norm=False,  # Or False, based on your data scaling. If data is already 0-1, maybe False.
    # num_future_covariates = 13 # Can be explicit if dec_in is not used for this
)

# --- Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Datasets and DataLoaders ---
train_dataset = PreprocessedDataset(X_history_train, X_known_past_train, X_known_future_train, y_train)
val_dataset = PreprocessedDataset(X_history_val, X_known_past_val, X_known_future_val, y_val)

batch_size = 32  # Example
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# --- Model, Criterion, Optimizer ---
model = TimeXerModel(configs).to(device)
criterion = nn.MSELoss()  # Or nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# --- Training Loop ---
epochs = 10  # Example
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for batch_x_enc, batch_x_mark_enc, batch_x_dec, batch_x_mark_dec, batch_y in train_loader:
        batch_x_enc = batch_x_enc.to(device)
        batch_x_mark_enc = batch_x_mark_enc.to(device)
        batch_x_dec = batch_x_dec.to(device)
        batch_x_mark_dec = batch_x_mark_dec.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        # Model's forward: forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None)
        outputs = model(batch_x_enc, batch_x_mark_enc, batch_x_dec, batch_x_mark_dec)

        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_x_enc, batch_x_mark_enc, batch_x_dec, batch_x_mark_dec, batch_y in val_loader:
            batch_x_enc = batch_x_enc.to(device)
            batch_x_mark_enc = batch_x_mark_enc.to(device)
            batch_x_dec = batch_x_dec.to(device)
            batch_x_mark_dec = batch_x_mark_dec.to(device)
            batch_y = batch_y.to(device)

            outputs = model(batch_x_enc, batch_x_mark_enc, batch_x_dec, batch_x_mark_dec)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

# --- Save Model (Example) ---
torch.save(model.state_dict(), "checkpoints/custom_timexer/test1.pth")