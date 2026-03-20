"""
models/xgboost_stress.py
========================
Week 3: XGBoost Grid Stress Forecaster + SHAP Explainability

Predicts demand_mwh (then derives grid stress score).
Handles null lag features via forward-fill per region.

Usage:
    cd C:\\Users\\Lenovo\\Downloads\\energy-grid-analyzer
    python models/xgboost_stress.py
"""

import os
import sqlite3
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xgboost as xgb
import shap
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH  = os.path.join(BASE_DIR, "energy_grid.db")
SAVE_DIR = os.path.join(BASE_DIR, "models", "saved")
os.makedirs(SAVE_DIR, exist_ok=True)

FEATURE_COLS = [
    "renewable_pct",
    "fossil_pct",
    "fossil_co2_intensity",
    "heat_index",
    "demand_lag_24h",
    "demand_lag_168h",
    "rolling_7d_avg",
    "per_capita_demand",
    "peak_hour_flag",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
]

TARGET_COL = "demand_mwh"   # predict demand → derive stress score


# ─────────────────────────────────────────────
# 1. LOAD & FIX NULLS
# ─────────────────────────────────────────────

def load_and_fix(db_path: str) -> pd.DataFrame:
    print("Loading features table...")
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM features ORDER BY region, datetime", conn)
    conn.close()
    print(f"  Loaded {len(df):,} rows")

    print("\nChecking null rates before fix:")
    for col in FEATURE_COLS + [TARGET_COL]:
        if col in df.columns:
            pct = df[col].isna().mean() * 100
            print(f"  {col:<28} {pct:5.1f}% null")

    # Fix lag/rolling nulls — forward fill within each region
    lag_cols = ["demand_lag_24h", "demand_lag_168h", "rolling_7d_avg"]
    for col in lag_cols:
        if col in df.columns:
            df[col] = df.groupby("region")[col].transform(
                lambda x: x.fillna(method="bfill").fillna(method="ffill")
            )

    # Fill weather nulls with region median
    weather_cols = ["heat_index", "fossil_co2_intensity"]
    for col in weather_cols:
        if col in df.columns:
            df[col] = df.groupby("region")[col].transform(
                lambda x: x.fillna(x.median())
            )

    # Fill remaining with 0
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0)

    print("\nNull rates after fix:")
    for col in FEATURE_COLS + [TARGET_COL]:
        if col in df.columns:
            pct = df[col].isna().mean() * 100
            print(f"  {col:<28} {pct:5.1f}% null")

    return df


# ─────────────────────────────────────────────
# 2. PREPARE DATA
# ─────────────────────────────────────────────

def prepare_data(df: pd.DataFrame):
    print("\nPreparing train/test split...")

    cols_needed = FEATURE_COLS + [TARGET_COL, "grid_stress_score"]
    df = df[cols_needed].dropna()
    print(f"  Rows after final dropna: {len(df):,}")

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    # Time-based split — last 20% = test
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    stress_test = df["grid_stress_score"].iloc[split_idx:]

    print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")
    print(f"  Target ({TARGET_COL}) range: {y.min():.1f} → {y.max():.1f}")

    return X_train, X_test, y_train, y_test, stress_test


# ─────────────────────────────────────────────
# 3. TRAIN
# ─────────────────────────────────────────────

def train_model(X_train, y_train, X_test, y_test):
    print("\nTraining XGBoost model...")

    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=30,
        eval_metric="rmse",
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=50,
    )

    print(f"  Best iteration: {model.best_iteration}")
    return model


# ─────────────────────────────────────────────
# 4. EVALUATE
# ─────────────────────────────────────────────

def evaluate(model, X_test, y_test):
    print("\n=== Model Evaluation (predicting demand_mwh) ===")
    y_pred = model.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-6))) * 100

    print(f"  MAE:  {mae:,.1f} MWh")
    print(f"  RMSE: {rmse:,.1f} MWh")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  R²:   {r2:.4f}")

    if r2 >= 0.90:
        print("  ✓ Excellent — R² ≥ 0.90")
    elif r2 >= 0.80:
        print("  ✓ Good — R² ≥ 0.80")
    elif r2 >= 0.60:
        print("  ~ Acceptable — consider adding more features")
    else:
        print("  ⚠ Low R² — check null rates above")

    return y_pred, r2


# ─────────────────────────────────────────────
# 5. SHAP
# ─────────────────────────────────────────────

def run_shap(model, X_test):
    print("\nComputing SHAP values...")
    X_sample = X_test.sample(min(2000, len(X_test)), random_state=42)
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X_sample, feature_names=FEATURE_COLS,
                      show=False, plot_size=None)
    plt.title("SHAP Feature Importance — Grid Demand Forecaster", fontsize=13)
    plt.tight_layout()
    path = os.path.join(SAVE_DIR, "shap_summary.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  SHAP plot saved → {path}")

    mean_shap = np.abs(shap_values).mean(axis=0)
    top5 = sorted(zip(FEATURE_COLS, mean_shap), key=lambda x: x[1], reverse=True)[:5]
    print("\n  Top 5 features by SHAP importance:")
    for feat, val in top5:
        print(f"    {feat:<28} {val:.4f}")


# ─────────────────────────────────────────────
# 6. PLOTS
# ─────────────────────────────────────────────

def plot_predictions(y_test, y_pred):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter: actual vs predicted
    idx = np.random.choice(len(y_test), min(3000, len(y_test)), replace=False)
    axes[0].scatter(np.array(y_test)[idx], y_pred[idx],
                    alpha=0.3, s=8, color="#378ADD")
    mn, mx = np.array(y_test).min(), np.array(y_test).max()
    axes[0].plot([mn, mx], [mn, mx], "r--", linewidth=1, label="Perfect")
    axes[0].set_xlabel("Actual Demand (MWh)")
    axes[0].set_ylabel("Predicted Demand (MWh)")
    axes[0].set_title("Actual vs Predicted Demand")
    axes[0].legend()

    # Line: first 500 test points over time
    n = min(500, len(y_test))
    axes[1].plot(range(n), np.array(y_test)[:n], label="Actual", alpha=0.8, linewidth=1)
    axes[1].plot(range(n), y_pred[:n], label="Predicted", alpha=0.8, linewidth=1, linestyle="--")
    axes[1].set_xlabel("Hours (test set)")
    axes[1].set_ylabel("Demand (MWh)")
    axes[1].set_title("Forecast vs Actual (first 500 hours)")
    axes[1].legend()

    plt.tight_layout()
    path = os.path.join(SAVE_DIR, "xgboost_predictions.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Prediction plot saved → {path}")


# ─────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────

def run():
    print("=" * 55)
    print("  Week 3: XGBoost Grid Stress Forecaster")
    print("=" * 55)

    df                                    = load_and_fix(DB_PATH)
    X_train, X_test, y_train, y_test, _  = prepare_data(df)
    model                                 = train_model(X_train, y_train, X_test, y_test)
    y_pred, r2                            = evaluate(model, X_test, y_test)

    run_shap(model, X_test)
    plot_predictions(y_test, y_pred)

    model.save_model(os.path.join(SAVE_DIR, "xgboost_stress.json"))
    print(f"\n  Model saved → {SAVE_DIR}/xgboost_stress.json")

    print("\n" + "=" * 55)
    print(f"  ✓ XGBoost complete! R² = {r2:.4f}")
    print("  Check models/saved/ for plots and model file")
    print("  Next: cluster_states.py and anomaly_detect.py")
    print("=" * 55)


if __name__ == "__main__":
    run()