import os
import joblib
import torch
import pandas as pd
import numpy as np
from torch import nn

LOOK_BACK = 60  # Ventana temporal

DATE = "2025-11-03"

DATE_COL = "Date"

# Features originales del CSV
ALL_COLS = [
    "BBVA_Close",
    "BBVA_High",
    "BBVA_Low",
    "BBVA_Volume",
    "BBVA_MA_5",
    "BBVA_MA_10",
    "BBVA_MA_30",
    "BBVA_MA_60",
    "IBEX_Close",
    "IBEX_Return_1d",
    "IBEX_MA_5",
    "deposit_rate",
    "marginal_rate",
    "refinancing_rate",
    "event",
]

# Columnas numéricas a escalar (todas menos event)
NUM_FEATURES = [
    "BBVA_Close",
    "BBVA_High",
    "BBVA_Low",
    "BBVA_Volume",
    "BBVA_MA_5",
    "BBVA_MA_10",
    "BBVA_MA_30",
    "BBVA_MA_60",
    "IBEX_Close",
    "IBEX_Return_1d",
    "IBEX_MA_5",
    "deposit_rate",
    "marginal_rate",
    "refinancing_rate",
]

EVENT_COL = "event"

# Targets multi-output (día siguiente)
TARGET_COLS = ["BBVA_Close_next", "BBVA_Low_next", "BBVA_High_next"]


SAN_DATE_COL = "Date"
SAN_EVENT_COL = "event"

SAN_NUM_FEATURES = [
    "SAN_Close",
    "SAN_High",
    "SAN_Low",
    "SAN_Volume",
    "SAN_MA_5",
    "SAN_MA_10",
    "SAN_MA_30",
    "SAN_MA_60",
    "IBEX_Close",
    "IBEX_Return_1d",
    "IBEX_MA_5",
    "deposit_rate",
    "marginal_rate",
    "refinancing_rate",
]

SAN_TARGET_COLS = ["SAN_Close_next", "SAN_Low_next", "SAN_High_next"]
LOOK_BACK = 30


class LSTMModel(nn.Module):
    def __init__(self, n_features, hidden_size=64, num_layers=1, dropout=0.0):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # Salidas: [Close, ΔHigh, ΔLow]
        self.fc = nn.Linear(hidden_size, 3)

    def forward(self, x):
        out, _ = self.lstm(x)  # (batch, seq, hidden)
        raw = self.fc(out[:, -1, :])  # (batch, 3)

        close = raw[:, 0]
        delta_high = torch.relu(raw[:, 1])
        delta_low = torch.relu(raw[:, 2])

        high = close + delta_high
        low = close - delta_low

        # Garantizamos Low ≤ Close ≤ High
        return torch.stack([close, low, high], dim=1)


def predict_one_day_bbva(
    df=None, model=None, scaler_X=None, scaler_Y=None, target_date=None
):
    df = (
        pd.read_csv("data/BBVA_model_dataset_2.csv", parse_dates=["Date"])
        .sort_values("Date")
        .reset_index(drop=True)
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # try to load from a "models" folder next to this file, fallback to cwd/models
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    if not os.path.isdir(models_dir):
        models_dir = os.path.join(os.getcwd(), "models")

    # load scalers if not provided
    if scaler_X is None:
        for fname in ["bbva_scaler_X.pkl"]:
            path = os.path.join(models_dir, fname)
            print(path)
            if os.path.exists(path):
                scaler_X = joblib.load(path)
                break
        else:
            raise FileNotFoundError(
                "scaler_X not provided and no scaler_X.* found in models/"
            )

    if scaler_Y is None:
        for fname in ["bbva_scaler_Y.pkl"]:
            path = os.path.join(models_dir, fname)
            if os.path.exists(path):
                scaler_Y = joblib.load(path)
                break
        else:
            raise FileNotFoundError(
                "scaler_Y not provided and no scaler_Y.* found in models/"
            )

    # load model from models_dir if model is None
    if model is None:
        # prefer scripted/traced (.pt), then state (.pth/.pt)
        model_path = None
        for fname in ["bbva_lstm_state.pth"]:
            path = os.path.join(models_dir, fname)
            if os.path.exists(path):
                model_path = path
                break

        if model_path is None:
            # fallback to any .pth file
            for fname in os.listdir(models_dir) if os.path.isdir(models_dir) else []:
                if fname.endswith(".pth"):
                    model_path = os.path.join(models_dir, fname)
                    break

        if model_path is None:
            raise FileNotFoundError(
                "model not provided and no model file found in models/"
            )

        n_features = len(NUM_FEATURES) + 1  # +1 for event column
        state_dict = torch.load(model_path)
        model = LSTMModel(
            n_features=n_features, hidden_size=64, num_layers=2, dropout=0.0
        )
        model.load_state_dict(state_dict)
    model = model.to(device)
    target_date = pd.Timestamp(target_date)
    # print("Predicting for date: " + str(target_date.date()))
    # Últimos LOOK_BACK días antes de target_date
    df_hist = df[df[DATE_COL] < target_date].tail(LOOK_BACK)
    if len(df_hist) < LOOK_BACK:
        raise ValueError(f"No hay suficientes datos antes de {target_date.date()}")

    X_num = df_hist[NUM_FEATURES].values
    X_num_scaled = scaler_X.transform(X_num)
    X_event = df_hist[[EVENT_COL]].values
    X_full = np.hstack([X_num_scaled, X_event])

    X_t = torch.tensor(X_full[np.newaxis, :, :], dtype=torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        y_pred_scaled = model(X_t).cpu().numpy()

    y_pred = scaler_Y.inverse_transform(y_pred_scaled)[0]

    # Real del día objetivo
    df_test = df[df[DATE_COL] == target_date]
    if df_test.empty:
        y_real = [0, 0, 0]
    else:
        y_real = df_test[["BBVA_Close", "BBVA_Low", "BBVA_High"]].iloc[0].values

    abs_err = np.abs(y_pred - y_real)
    mae_mean = abs_err.mean()
    rmse_mean = float(np.sqrt(((y_pred - y_real) ** 2).mean()))

    return y_pred, y_real, mae_mean, rmse_mean


def predict_one_day_san(
    df=None, model=None, scaler_X=None, scaler_Y=None, target_date=None
):
    df = (
        pd.read_csv("data/SAN_model_dataset_2.csv", parse_dates=["Date"])
        .sort_values("Date")
        .reset_index(drop=True)
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # try to load from a "models" folder next to this file, fallback to cwd/models
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    if not os.path.isdir(models_dir):
        models_dir = os.path.join(os.getcwd(), "models")

    # load scalers if not provided
    if scaler_X is None:
        for fname in ["san_scaler_X.pkl"]:
            path = os.path.join(models_dir, fname)
            if os.path.exists(path):
                scaler_X = joblib.load(path)
                break
        else:
            raise FileNotFoundError(
                "scaler_X not provided and no scaler_X.* found in models/"
            )

    if scaler_Y is None:
        for fname in ["san_scaler_Y.pkl"]:
            path = os.path.join(models_dir, fname)
            if os.path.exists(path):
                scaler_Y = joblib.load(path)
                break
        else:
            raise FileNotFoundError(
                "scaler_Y not provided and no scaler_Y.* found in models/"
            )

    # load model from models_dir if model is None
    if model is None:
        # prefer scripted/traced (.pt), then state (.pth/.pt)
        model_path = None
        for fname in ["san_lstm_state.pth"]:
            path = os.path.join(models_dir, fname)
            if os.path.exists(path):
                model_path = path
                break

        if model_path is None:
            # fallback to any .pth file
            for fname in os.listdir(models_dir) if os.path.isdir(models_dir) else []:
                if fname.endswith(".pth"):
                    model_path = os.path.join(models_dir, fname)
                    break

        if model_path is None:
            raise FileNotFoundError(
                "model not provided and no model file found in models/"
            )

        n_features = len(SAN_NUM_FEATURES) + 1  # +1 for event column
        state_dict = torch.load(model_path)
        model = LSTMModel(
            n_features=n_features, hidden_size=64, num_layers=2, dropout=0.0
        )
        model.load_state_dict(state_dict)
    model = model.to(device)
    target_date = pd.Timestamp(target_date)

    # Últimos LOOK_BACK días antes de target_date
    df_hist = df[df[DATE_COL] < target_date].tail(LOOK_BACK)
    if len(df_hist) < LOOK_BACK:
        raise ValueError(f"No hay suficientes datos antes de {target_date.date()}")

    X_num = df_hist[SAN_NUM_FEATURES].values
    X_num_scaled = scaler_X.transform(X_num)
    X_event = df_hist[[EVENT_COL]].values
    X_full = np.hstack([X_num_scaled, X_event])

    X_t = torch.tensor(X_full[np.newaxis, :, :], dtype=torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        y_pred_scaled = model(X_t).cpu().numpy()

    y_pred = scaler_Y.inverse_transform(y_pred_scaled)[0]

    # Real del día objetivo
    df_test = df[df[DATE_COL] == target_date]
    if df_test.empty:
        y_real = [0, 0, 0]
    else:
        y_real = df_test[["SAN_Close", "SAN_Low", "SAN_High"]].iloc[0].values

    abs_err = np.abs(y_pred - y_real)
    mae_mean = abs_err.mean()
    rmse_mean = float(np.sqrt(((y_pred - y_real) ** 2).mean()))

    return y_pred, y_real, mae_mean, rmse_mean