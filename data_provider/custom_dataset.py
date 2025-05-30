# custom_dataset.py
import torch
from torch.utils.data import Dataset
import numpy as np


class PreprocessedDataset(Dataset):
    def __init__(self, X_history, X_known_past, X_known_future, y_target):
        self.X_history = torch.FloatTensor(X_history)
        self.X_known_past = torch.FloatTensor(X_known_past)
        self.X_known_future = torch.FloatTensor(X_known_future)
        self.y_target = torch.FloatTensor(y_target)

    def __len__(self):
        return len(self.X_history)

    def __getitem__(self, idx):
        # Construct model inputs based on TimeXer's forward signature
        # x_enc = (batch, seq_len, enc_in_features)
        # x_mark_enc = (batch, seq_len, num_hist_exo_features)
        # x_dec = (batch, pred_len, c_out_features) -> Placeholder for TimeXer
        # x_mark_dec = (batch, pred_len, num_future_exo_features)

        # x_enc: Concatenate historical known covariates and historical target
        # Shape: (seq_len, 13 features + 1 target = 14)
        x_enc_sample = torch.cat((self.X_known_past[idx], self.X_history[idx]), dim=-1)

        # x_mark_enc: Use historical known covariates directly
        # Shape: (seq_len, 13 features)
        x_mark_enc_sample = self.X_known_past[idx]

        # x_mark_dec: Use future known covariates directly
        # Shape: (pred_len, 13 features)
        x_mark_dec_sample = self.X_known_future[idx]

        # y_target_actual for loss calculation
        # Shape: (pred_len, 1 target)
        y_sample = self.y_target[idx]

        # x_dec: Placeholder for TimeXer (not used by its forecast method)
        # Typically, it might be future targets or zeros.
        # Shape: (pred_len, 1 target)
        x_dec_sample = torch.zeros_like(y_sample)  # Or pass y_sample if preferred, won't affect TimeXer

        return x_enc_sample, x_mark_enc_sample, x_dec_sample, x_mark_dec_sample, y_sample