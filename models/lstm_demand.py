"""
models/lstm_demand.py
=====================
Week 4: PyTorch LSTM Time-Series Demand Forecaster

Uses a 168-hour (1 week) sliding window to forecast the next
48 hours of electricity demand per region.

Usage:
    cd C:\\Users\\Lenovo\\Downloads\\energy-grid-analyzer
    python models/lstm_demand.py

Output:
    models/saved/lstm_demand.pt          — trained LSTM weights
    models/saved/scaler_lstm.pkl         — fitted MinMaxScaler
    models/saved/lstm_forecast.png       — forecast vs actual plot
"""

import os
import pickle
import sqlite3
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH  = os.path.join(BASE_DIR, "energy_grid.db")
SAVE_DIR = os.path.join(BASE_DIR, "models", "saved")
os.makedirs(SAVE_DIR, exist_ok=True)

# ── Hyperparameters ────────────────────────────────────────
SEQ_LEN    = 168   # input window: 1 week of hourly data
PRED_LEN   = 48    # forecast horizon: 48 hours ahead
HIDDEN     = 128   # LSTM hidden units
N_LAYERS   = 2     # stacked LSTM layers
DROPOUT    = 0.2
BATCH_SIZE = 64
EPOCHS     = 30
LR         = 1e-3
PATIENCE   = 5     # early stopping patience

# Features fed into the LSTM at each timestep
LSTM_FEATURES = [
    "demand_mwh",
    "renewable_pct",
    "fossil_pct",
    "heat_index",
    "peak_hour_flag",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ─────────────────────────────────────────────
# 1. DATASET
# ─────────────────────────────────────────────

class DemandDataset(Dataset):
    """
    Sliding window dataset.
    Each sample: X = (SEQ_LEN, n_features), y = (PRED_LEN,)
    """
    def __init__(self, data: np.ndarray, seq_len: int, pred_len: int):
        self.data     = torch.FloatTensor(data)
        self.seq_len  = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]                          # (SEQ_LEN, features)
        y = self.data[idx + self.seq_len : idx + self.seq_len + self.pred_len, 0]  # demand only
        return x, y


# ─────────────────────────────────────────────
# 2. MODEL
# ─────────────────────────────────────────────

class GridLSTM(nn.Module):
    """
    2-layer stacked LSTM with dropout.
    Input:  (batch, seq_len, n_features)
    Output: (batch, pred_len)
    """
    def __init__(self, input_size: int, hidden: int, layers: int,
                 pred_len: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden, layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, pred_len),
        )

    def forward(self, x):
        out, _ = self.lstm(x)           # (batch, seq_len, hidden)
        last    = out[:, -1, :]         # take last timestep
        last    = self.dropout(last)
        return self.head(last)          # (batch, pred_len)


# ─────────────────────────────────────────────
# 3. LOAD & PREPARE DATA
# ─────────────────────────────────────────────

def load_data(db_path: str):
    print("Loading features table...")
    conn = sqlite3.connect(db_path)
    df = pd.read_sql(
        "SELECT * FROM features ORDER BY region, datetime", conn
    )
    conn.close()
    print(f"  Loaded {len(df):,} rows")

    # Use the region with the most data for clean LSTM training
    region = df["region"].value_counts().index[0]
    df = df[df["region"] == region].sort_values("datetime").reset_index(drop=True)
    print(f"  Training on region: {region} ({len(df):,} rows)")

    # Keep only needed columns, fill nulls
    df = df[LSTM_FEATURES].fillna(0)

    return df, region


def prepare_sequences(df: pd.DataFrame):
    print("\nPreparing sequences...")

    # Scale all features to [0, 1]
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df.values)

    # Time-based split — last 15% = test
    split = int(len(data_scaled) * 0.85)
    train_data = data_scaled[:split]
    test_data  = data_scaled[split - SEQ_LEN:]   # overlap so test has full windows

    train_ds = DemandDataset(train_data, SEQ_LEN, PRED_LEN)
    test_ds  = DemandDataset(test_data,  SEQ_LEN, PRED_LEN)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"  Train sequences: {len(train_ds):,}")
    print(f"  Test sequences:  {len(test_ds):,}")
    print(f"  Input shape per sample: ({SEQ_LEN}, {len(LSTM_FEATURES)})")
    print(f"  Output shape per sample: ({PRED_LEN},)")

    return train_loader, test_loader, scaler, data_scaled, split


# ─────────────────────────────────────────────
# 4. TRAIN
# ─────────────────────────────────────────────

def train(model, train_loader, test_loader):
    print(f"\nTraining LSTM for up to {EPOCHS} epochs (early stopping patience={PATIENCE})...")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=2, factor=0.5
    )
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    patience_counter = 0
    best_weights = None
    history = {"train": [], "val": []}

    for epoch in range(1, EPOCHS + 1):
        # ── Train ──
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # ── Validate ──
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                pred = model(X_batch)
                val_losses.append(criterion(pred, y_batch).item())

        train_loss = np.mean(train_losses)
        val_loss   = np.mean(val_losses)
        scheduler.step(val_loss)

        history["train"].append(train_loss)
        history["val"].append(val_loss)

        print(f"  Epoch {epoch:02d}/{EPOCHS}  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n  Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
                break

    # Restore best weights
    model.load_state_dict(best_weights)
    print(f"\n  Best val_loss: {best_val_loss:.6f}")
    return model, history


# ─────────────────────────────────────────────
# 5. EVALUATE
# ─────────────────────────────────────────────

def evaluate(model, test_loader, scaler):
    print("\n=== Model Evaluation ===")
    model.eval()

    all_preds, all_targets = [], []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(y_batch.numpy())

    preds   = np.concatenate(all_preds,   axis=0).flatten()
    targets = np.concatenate(all_targets, axis=0).flatten()

    # Inverse scale demand (feature index 0)
    def inverse_demand(arr):
        dummy = np.zeros((len(arr), len(LSTM_FEATURES)))
        dummy[:, 0] = arr
        return scaler.inverse_transform(dummy)[:, 0]

    preds_mwh   = inverse_demand(preds)
    targets_mwh = inverse_demand(targets)

    mae  = mean_absolute_error(targets_mwh, preds_mwh)
    rmse = np.sqrt(mean_squared_error(targets_mwh, preds_mwh))
    r2   = r2_score(targets_mwh, preds_mwh)
    mape = np.mean(np.abs((targets_mwh - preds_mwh) / (targets_mwh + 1e-6))) * 100

    print(f"  MAE:  {mae:,.1f} MWh")
    print(f"  RMSE: {rmse:,.1f} MWh")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  R²:   {r2:.4f}")

    if r2 >= 0.85:
        print("  ✓ Excellent LSTM performance")
    elif r2 >= 0.70:
        print("  ✓ Good — try more epochs for improvement")
    else:
        print("  ~ Acceptable — LSTM needs more data or tuning")

    return preds_mwh, targets_mwh, r2


# ─────────────────────────────────────────────
# 6. PLOTS
# ─────────────────────────────────────────────

def plot_results(preds_mwh, targets_mwh, history, region):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Forecast vs Actual (first 500 predictions)
    n = min(500, len(preds_mwh))
    axes[0].plot(range(n), targets_mwh[:n], label="Actual",    linewidth=1, alpha=0.8)
    axes[0].plot(range(n), preds_mwh[:n],   label="Predicted", linewidth=1, alpha=0.8, linestyle="--")
    axes[0].set_title(f"LSTM Forecast vs Actual — {region}", fontsize=12)
    axes[0].set_xlabel("Hours")
    axes[0].set_ylabel("Demand (MWh)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.2)

    # Plot 2: Scatter actual vs predicted
    idx = np.random.choice(len(preds_mwh), min(2000, len(preds_mwh)), replace=False)
    axes[1].scatter(targets_mwh[idx], preds_mwh[idx], alpha=0.3, s=6, color="#378ADD")
    mn, mx = targets_mwh.min(), targets_mwh.max()
    axes[1].plot([mn, mx], [mn, mx], "r--", linewidth=1, label="Perfect")
    axes[1].set_xlabel("Actual (MWh)")
    axes[1].set_ylabel("Predicted (MWh)")
    axes[1].set_title("Actual vs Predicted Scatter")
    axes[1].legend()
    axes[1].grid(True, alpha=0.2)

    # Plot 3: Training loss curve
    axes[2].plot(history["train"], label="Train loss", linewidth=1.5)
    axes[2].plot(history["val"],   label="Val loss",   linewidth=1.5)
    axes[2].set_title("Training Loss Curve")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("MSE Loss")
    axes[2].legend()
    axes[2].grid(True, alpha=0.2)

    plt.tight_layout()
    path = os.path.join(SAVE_DIR, "lstm_forecast.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved → {path}")


# ─────────────────────────────────────────────
# 7. SAVE
# ─────────────────────────────────────────────

def save_model(model, scaler):
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, "lstm_demand.pt"))
    with open(os.path.join(SAVE_DIR, "scaler_lstm.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    print(f"  Model saved → {SAVE_DIR}/lstm_demand.pt")
    print(f"  Scaler saved → {SAVE_DIR}/scaler_lstm.pkl")


# ─────────────────────────────────────────────
# 8. MAIN
# ─────────────────────────────────────────────

def run():
    print("=" * 55)
    print("  Week 4: PyTorch LSTM Demand Forecaster")
    print("=" * 55)

    # Load
    df, region = load_data(DB_PATH)

    # Prepare
    train_loader, test_loader, scaler, _, _ = prepare_sequences(df)

    # Build model
    model = GridLSTM(
        input_size=len(LSTM_FEATURES),
        hidden=HIDDEN,
        layers=N_LAYERS,
        pred_len=PRED_LEN,
        dropout=DROPOUT,
    ).to(device)

    # Train
    model, history = train(model, train_loader, test_loader)

    # Evaluate
    preds_mwh, targets_mwh, r2 = evaluate(model, test_loader, scaler)

    # Plot
    print("\nGenerating plots...")
    plot_results(preds_mwh, targets_mwh, history, region)

    # Save
    save_model(model, scaler)

    print("\n" + "=" * 55)
    print(f"  ✓ LSTM complete! R² = {r2:.4f}")
    print(f"  Sequence length:  {SEQ_LEN}h (1 week lookback)")
    print(f"  Forecast horizon: {PRED_LEN}h (48h ahead)")
    print("  Open models/saved/lstm_forecast.png")
    print("=" * 55)


if __name__ == "__main__":
    run()