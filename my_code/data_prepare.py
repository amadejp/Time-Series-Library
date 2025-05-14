import pandas as pd
import numpy as np
import os


def read_timeseries(data_type):
    X_history = np.load(f"../my_data/{data_type}/X_history_target.npy")
    X_features = np.load(f"../my_data/{data_type}/X_future_exog_features.npy")
    y = np.load(f"../my_data/{data_type}/y_target.npy")
    interval_dates = np.load(f"../my_data/{data_type}/interval_starts.npy", allow_pickle=True)
    interval_dates = pd.to_datetime(interval_dates)

    return X_history, X_features, y, interval_dates


if __name__ == "__main__":
    X_history_train, X_features_train, y_train, interval_dates_train = read_timeseries("train")
    X_history_val, X_features_val, y_val, interval_dates_val = read_timeseries("val")
    X_history_test, X_features_test, y_test, interval_dates_test = read_timeseries("test")

    print("Train history shapes:", X_history_train.shape, X_features_train.shape, y_train.shape)
    print("Val history shapes:", X_history_val.shape, X_features_val.shape, y_val.shape)
    print("Test history shapes:", X_history_test.shape, X_features_test.shape, y_test.shape)
    print()
    print("Train features shapes:", X_history_train.shape, X_features_train.shape, y_train.shape)
    print("Val features shapes:", X_history_val.shape, X_features_val.shape, y_val.shape)
    print("Test features shapes:", X_history_test.shape, X_features_test.shape, y_test.shape)
    print()
    print("Train interval dates shapes:", X_history_train.shape, X_features_train.shape, y_train.shape)
    print("Val interval dates shapes:", X_history_val.shape, X_features_val.shape, y_val.shape)
    print("Test interval dates shapes:", X_history_test.shape, X_features_test.shape, y_test.shape)


