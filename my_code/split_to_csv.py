import pandas as pd
import numpy as np
import os  # For creating directories


def read_df(filename):
    df = pd.read_csv(filename, index_col='interval_start', parse_dates=True)
    # Optional: Ensure hourly frequency if needed, though the library might handle timestamps more flexibly
    # df = df.asfreq('H')
    return df


# --- Main Execution ---
if __name__ == "__main__":
    # --- Configuration ---
    data_filename = "../my_data/ts_ready.csv"  # Use a relative path
    output_data_dir = "../tslib_data"  # Directory to save processed data for the library

    target_col = 'n_active_sessions_grid'
    # window_size (seq_len in library terms) will be set in the experiment script
    # horizon (pred_len in library terms) will be set in the experiment script

    # --- Load Data ---
    try:
        df = read_df(data_filename)
    except FileNotFoundError:
        print(f"Error: {data_filename} not found. Make sure the path is correct relative to the script.")
        exit()

    print("Original DataFrame Columns:")
    print(df.columns)

    # Identify static feature columns (all except target)
    static_feature_cols = [col for col in df.columns if col != target_col]
    # Reorder columns: Put static features first, then target column last
    # This is a common convention for the library when using features='M'
    ordered_cols = static_feature_cols + [target_col]
    df = df[ordered_cols]

    print("\nReordered DataFrame Columns (target last):")
    print(df.columns)

    # --- Define Split Points ---
    train_start = pd.Timestamp('2023-11-14 01:00:00')
    train_end = pd.Timestamp('2024-10-08 00:00:00')  # End of day Oct 7th
    val_start = pd.Timestamp('2024-10-15 01:00:00')
    val_end = pd.Timestamp('2025-01-01 00:00:00')  # End of Dec 31st 2024
    test_start = pd.Timestamp('2025-01-01 01:00:00')
    test_end = pd.Timestamp('2025-02-26 00:00:00')  # End of Feb 25th

    # --- Data Splitting ---
    train_df = df.loc[train_start:train_end]
    val_df = df.loc[val_start:val_end]
    test_df = df.loc[test_start:test_end]

    print(f"\nTrain shape: {train_df.shape}")
    print(f"Validation shape: {val_df.shape}")
    print(f"Test shape: {test_df.shape}")

    # --- Save Data Splits for Time-Series-Library ---
    print(f"\nSaving data splits to {output_data_dir}...")
    os.makedirs(output_data_dir, exist_ok=True)

    # Important: Save *with* header but *without* index for Dataset_Custom
    # The first column should be the datetime string for the library's date handling

    # Add datetime column explicitly from index before saving
    train_df_save = train_df.reset_index().rename(columns={'interval_start': 'date'})
    val_df_save = val_df.reset_index().rename(columns={'interval_start': 'date'})
    test_df_save = test_df.reset_index().rename(columns={'interval_start': 'date'})

    # Ensure date format is consistent if needed, e.g., 'YYYY-MM-DD HH:MM:SS'
    train_df_save['date'] = train_df_save['date'].dt.strftime('%Y-%m-%d %H:%M:%S')
    val_df_save['date'] = val_df_save['date'].dt.strftime('%Y-%m-%d %H:%M:%S')
    test_df_save['date'] = test_df_save['date'].dt.strftime('%Y-%m-%d %H:%M:%S')

    train_file = os.path.join(output_data_dir, 'train.csv')
    val_file = os.path.join(output_data_dir, 'val.csv')
    test_file = os.path.join(output_data_dir, 'test.csv')

    train_df_save.to_csv(train_file, index=False, header=True)
    val_df_save.to_csv(val_file, index=False, header=True)
    test_df_save.to_csv(test_file, index=False, header=True)

    print(f"Saved train data to {train_file}")
    print(f"Saved validation data to {val_file}")
    print(f"Saved test data to {test_file}")