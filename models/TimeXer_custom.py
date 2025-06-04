import torch
import torch.nn as nn
import torch.nn.functional as F
# Make sure these layer files are accessible in your Python path
# If running from the root of Time-Series-Library, they should be.
# Otherwise, adjust paths or copy layers/SelfAttention_Family.py and layers/Embed.py
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted, PositionalEmbedding
import numpy as np


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class EnEmbedding(nn.Module):
    def __init__(self, n_vars, d_model, patch_len, dropout):
        super(EnEmbedding, self).__init__()
        self.patch_len = patch_len
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.glb_token = nn.Parameter(torch.randn(1, n_vars, 1, d_model))
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        n_vars = x.shape[1]
        glb = self.glb_token.repeat((x.shape[0], 1, 1, 1))
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        x = self.value_embedding(x) + self.position_embedding(x)
        x = torch.reshape(x, (-1, n_vars, x.shape[-2], x.shape[-1]))
        x = torch.cat([x, glb], dim=2)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        return self.dropout(x), n_vars


class Encoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)
        if self.norm is not None:
            x = self.norm(x)
        if self.projection is not None:
            x = self.projection(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        B, L, D = cross.shape  # Note: Original TimeXer uses B,L,D from cross. If cross can be empty, this needs care.
        # For our purpose, cross (ex_embed) will have B, L_ex, D_model

        x_self_attn_out = self.self_attention(x, x, x, attn_mask=x_mask, tau=tau, delta=None)[0]
        x = x + self.dropout(x_self_attn_out)
        x = self.norm1(x)

        x_glb_ori = x[:, -1, :].unsqueeze(1)  # (bs_nvars, 1, d_model)

        # Reshape x_glb_ori to (bs, nvars, d_model) to match cross's batching if cross is (bs, L_ex, D_model)
        # This part of original TimeXer assumes x and cross have compatible batching for cross-attention.
        # en_embed is (bs*n_vars, L_en, D) and ex_embed is (bs, L_ex, D)
        # We need to align them. Let's assume cross attention is done per original batch sample.
        num_original_batches = cross.shape[0]
        n_vars_processed = x.shape[0] // num_original_batches

        x_glb_reshaped_for_cross = torch.reshape(x_glb_ori,
                                                 (num_original_batches, n_vars_processed, D))  # (bs, n_vars, d_model)

        # If n_vars_processed > 1, we might average or take the first, or attend each.
        # Original TimeXer seems to do this per variable if n_vars > 1 due to reshaping of x.
        # Let's stick to the original logic flow as much as possible, assuming cross attention is on x_glb_ori.
        # The cross attention expects query (bs, L_q, D), key (bs, L_k, D), value (bs, L_v, D)
        # Here, x_glb_ori is (bs*n_vars, 1, D), cross is (bs, L_ex, D). This is a mismatch.
        # The original paper's diagram suggests GLB token attends to Exogenous features.
        # A common way is to repeat/expand cross for each var, or average x_glb_ori over vars.
        # Let's assume the intent was for cross to be (bs*n_vars, L_ex, D) or for x_glb to be (bs, 1, D).
        # Given DataEmbedding_inverted output is (bs, L_ex, D_model), we need to align.
        # Simplest: repeat cross for each var.
        cross_repeated = cross.unsqueeze(1).repeat(1, n_vars_processed, 1, 1).reshape(
            num_original_batches * n_vars_processed, cross.shape[1], cross.shape[2])

        x_glb_attn = self.dropout(self.cross_attention(
            x_glb_ori, cross_repeated, cross_repeated,  # Query, Key, Value
            attn_mask=cross_mask,  # cross_mask needs to align with cross_repeated
            tau=tau, delta=delta
        )[0])
        # x_glb_attn is (bs*n_vars, 1, D)

        x_glb = x_glb_ori + x_glb_attn
        x_glb = self.norm2(x_glb)

        y = x = torch.cat([x[:, :-1, :], x_glb], dim=1)  # (bs*n_vars, L_en, D)

        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.features = configs.features
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.use_norm = configs.use_norm
        self.patch_len = configs.patch_len
        self.patch_num = int(configs.seq_len // configs.patch_len)

        # n_vars is the number of variables for the main endogenous processing stream
        self.n_vars = 1 if configs.features == 'MS' else configs.enc_in

        self.en_embedding = EnEmbedding(self.n_vars, configs.d_model, self.patch_len, configs.dropout)

        # DataEmbedding_inverted takes seq_len for its internal patching/value embedding if x_cont is provided
        self.ex_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                   configs.dropout)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        # <<<< START MODIFICATION >>>>
        # Determine number of future covariates. For your data: 13
        # This should ideally come from data shape, but configs can guide.
        self.num_future_covariates = configs.enc_in - 1 if configs.features == 'MS' else 0
        # If features='M', future covariates might be handled differently or not exist if all are targets.
        # For your specific data (13 future covariates), set this appropriately.
        # Let's assume configs.dec_in can represent num_future_covariates if features='MS'
        if hasattr(configs, 'num_future_covariates'):  # Allow explicit setting
            self.num_future_covariates = configs.num_future_covariates
        elif configs.features == 'MS' and hasattr(configs, 'dec_in') and configs.dec_in > 0:  # Use dec_in if MS
            self.num_future_covariates = configs.dec_in  # If dec_in is set to 13
        elif configs.features == 'MS':
            self.num_future_covariates = configs.enc_in - 1  # Default for MS
        else:
            self.num_future_covariates = 0

        if self.num_future_covariates > 0:
            self.future_covariate_embedding = nn.Linear(self.num_future_covariates, configs.d_model)
        else:
            self.future_covariate_embedding = None

        self.head_nf_encoder_path_seq_len = self.patch_num + 1
        self.head_nf_future_cov_path_seq_len = 0
        if self.num_future_covariates > 0:  # Check based on actual num_future_covariates
            self.future_covariate_embedding = nn.Linear(self.num_future_covariates, configs.d_model)
            self.head_nf_future_cov_path_seq_len = configs.pred_len  # Future features exist for each pred step
        else:
            self.future_covariate_embedding = None

        self.head_nf = configs.d_model * (self.head_nf_encoder_path_seq_len + self.head_nf_future_cov_path_seq_len)

        self.head = FlattenHead(self.n_vars, self.head_nf, configs.pred_len, head_dropout=configs.dropout)

        # FlattenHead's n_vars should be self.n_vars (1 for MS, enc_in for M)
        # as it operates on the output of the main encoder stream for those variables.
        self.head = FlattenHead(self.n_vars, self.head_nf, configs.pred_len, head_dropout=configs.dropout)
        # <<<< END MODIFICATION >>>>

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # x_enc: (batch, seq_len, 14) [e.g., 13 past_known_covariates, 1 past_target]
        # x_mark_enc: (batch, seq_len, 13) [e.g., 13 past_known_covariates, possibly unscaled or different features]
        # x_mark_dec: (batch, pred_len, 13) [e.g., 13 future_known_covariates]

        # Normalization
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc_norm = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc_norm, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc_norm /= stdev
        else:
            x_enc_norm = x_enc

        # en_embedding for the target variable (last channel of x_enc for MS)
        # en_embed: (bs * self.n_vars, patch_num + 1, d_model)
        target_for_en_embed = x_enc_norm[:, :, -self.n_vars:]  # Takes last self.n_vars columns
        en_embed, n_vars_processed_by_en = self.en_embedding(target_for_en_embed.permute(0, 2, 1))

        # ex_embedding for historical covariates
        # x_cont from x_enc (all but target for MS): (bs, seq_len, enc_in - self.n_vars)
        # x_mark from x_mark_enc: (bs, seq_len, num_hist_exo_features)
        if self.features == 'MS':
            x_cont_hist_cov = x_enc_norm[:, :, :-self.n_vars]  # First enc_in - n_vars features
        else:  # For 'M', all x_enc features are processed by en_embedding. x_cont might be empty or x_mark_enc only.
            x_cont_hist_cov = torch.empty(x_enc_norm.shape[0], self.seq_len, 0).to(x_enc_norm.device)

        # Ensure x_mark_enc is not None and has features before passing to ex_embedding
        # DataEmbedding_inverted can handle x_cont or x_mark being empty if the other is not.
        if x_cont_hist_cov.shape[-1] > 0 or (x_mark_enc is not None and x_mark_enc.shape[-1] > 0):
            ex_embed = self.ex_embedding(x_cont_hist_cov, x_mark_enc if x_mark_enc is not None else torch.empty_like(
                x_cont_hist_cov[..., :0]))
        else:  # No historical exogenous features provided
            # Create a zero tensor that can be summed in the encoder if needed.
            # Shape needs to be (bs, patch_num_ex, d_model). patch_num_ex depends on ExEmbedding's internal logic.
            # For simplicity, let's make it compatible with en_embed's sequence length for now.
            # This might need adjustment based on how Encoder handles empty `cross`.
            # A robust encoder would handle `cross=None`.
            # Assuming ex_embed should have shape (bs, L_ex, D_model)
            # L_ex is determined by DataEmbedding_inverted. If no input, L_ex is effectively 0.
            ex_embed = torch.zeros(x_enc_norm.shape[0], self.patch_num + 1, configs.d_model).to(en_embed.device)

        enc_out = self.encoder(en_embed, ex_embed)  # en_embed (bs*n_vars,L,D), ex_embed (bs,L_ex,D)

        # Reshape enc_out from (bs * self.n_vars, patch_num + 1, d_model)
        # to (bs, self.n_vars, d_model, patch_num + 1) for the head
        enc_out = torch.reshape(enc_out, (-1, n_vars_processed_by_en, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)

        # <<<< START MODIFICATION >>>>
        # Process future covariates from x_mark_dec
        if self.future_covariate_embedding is not None and x_mark_dec is not None and x_mark_dec.shape[-1] > 0:
            # x_mark_dec shape: (bs, pred_len, num_future_covariates)
            # Ensure x_mark_dec has pred_len in its time dimension.
            # Your data X_known_future_train already has shape (N, 24, 13) where 24 is pred_len.
            future_covariates_raw = x_mark_dec

            embedded_future_covariates = self.future_covariate_embedding(
                future_covariates_raw)  # (bs, pred_len, d_model)

            # Reshape for concatenation: (bs, self.n_vars, d_model, pred_len)
            embedded_future_covariates_reshaped = embedded_future_covariates.permute(0, 2, 1).unsqueeze(1)
            if n_vars_processed_by_en > 1 and embedded_future_covariates_reshaped.shape[
                1] == 1:  # For M mode if future cov are global
                embedded_future_covariates_reshaped = embedded_future_covariates_reshaped.repeat(1,
                                                                                                 n_vars_processed_by_en,
                                                                                                 1, 1)

            combined_input_for_head = torch.cat([enc_out, embedded_future_covariates_reshaped], dim=-1)
        else:
            combined_input_for_head = enc_out
        # <<<< END MODIFICATION >>>>

        dec_out = self.head(combined_input_for_head)
        dec_out = dec_out.permute(0, 2, 1)  # (bs, pred_len, self.n_vars)

        # De-Normalization (applied to the target variable(s))
        if self.use_norm:
            target_stdev = stdev[:, 0, -self.n_vars:]
            target_means = means[:, 0, -self.n_vars:]
            dec_out = dec_out * target_stdev.unsqueeze(1).repeat(1, self.pred_len, 1)
            dec_out = dec_out + target_means.unsqueeze(1).repeat(1, self.pred_len, 1)

        return dec_out

    def forecast_multi(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # This method is called if self.features == 'M'.
        # The logic would be similar to forecast, but self.n_vars would be configs.enc_in.
        # For your 'MS' case, this won't be called.
        # For brevity, I'm showing the 'MS' focused changes in `forecast`.
        # If you need 'M', apply analogous changes here.
        # The main difference is that EnEmbedding processes all `configs.enc_in` features from `x_enc`.
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc_norm = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc_norm, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc_norm /= stdev
        else:
            x_enc_norm = x_enc

        en_embed, n_vars_processed_by_en = self.en_embedding(
            x_enc_norm.permute(0, 2, 1))  # self.n_vars = configs.enc_in for M

        # For M mode, x_cont_hist_cov is typically empty as all x_enc features are handled by en_embedding.
        # ex_embed is driven by x_mark_enc.
        x_cont_for_ex = torch.empty(x_enc_norm.shape[0], self.seq_len, 0).to(x_enc_norm.device)
        if x_mark_enc is not None and x_mark_enc.shape[-1] > 0:
            ex_embed = self.ex_embedding(x_cont_for_ex, x_mark_enc)
        else:
            ex_embed = torch.zeros(x_enc_norm.shape[0], self.patch_num + 1, configs.d_model).to(en_embed.device)

        enc_out = self.encoder(en_embed, ex_embed)
        enc_out = torch.reshape(enc_out, (-1, n_vars_processed_by_en, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)

        if self.future_covariate_embedding is not None and x_mark_dec is not None and x_mark_dec.shape[-1] > 0:
            future_covariates_raw = x_mark_dec
            embedded_future_covariates = self.future_covariate_embedding(future_covariates_raw)
            embedded_future_covariates_reshaped = embedded_future_covariates.permute(0, 2, 1).unsqueeze(1)
            if n_vars_processed_by_en > 1 and embedded_future_covariates_reshaped.shape[1] == 1:
                embedded_future_covariates_reshaped = embedded_future_covariates_reshaped.repeat(1,
                                                                                                 n_vars_processed_by_en,
                                                                                                 1, 1)
            combined_input_for_head = torch.cat([enc_out, embedded_future_covariates_reshaped], dim=-1)
        else:
            combined_input_for_head = enc_out

        dec_out = self.head(combined_input_for_head)
        dec_out = dec_out.permute(0, 2, 1)

        if self.use_norm:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))  # all variables
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))  # all variables

        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            if self.features == 'M':
                dec_out = self.forecast_multi(x_enc, x_mark_enc, x_dec, x_mark_dec)
            else:  # MS or S
                dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        return None