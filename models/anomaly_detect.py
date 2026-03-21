"""
models/anomaly_detect.py
========================
Week 3: Isolation Forest Anomaly Detection

Detects unusual grid events: demand spikes, renewable crashes,
sudden stress surges. These become live alerts in the dashboard.

Usage:
    cd C:\\Users\\Lenovo\\Downloads\\energy-grid-analyzer
    python models/anomaly_detect.py

Output:
    models/saved/anomaly_detector.pkl    — trained Isolation Forest
    models/saved/scaler_anomaly.pkl      — fitted StandardScaler
    models/saved/anomaly_alerts.csv      — flagged anomaly events
    models/saved/anomaly_timeline.png    — visualization
"""

import os
import pickle
import sqlite3
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH  = os.path.join(BASE_DIR, "energy_grid.db")
SAVE_DIR = os.path.join(BASE_DIR, "models", "saved")
os.makedirs(SAVE_DIR, exist_ok=True)

# Features that best capture abnormal grid behavior
ANOMALY_FEATURES = [
    "demand_mwh",
    "grid_stress_score",
    "renewable_pct",
    "fossil_co2_intensity",
    "demand_lag_24h",
    "rolling_7d_avg",
    "heat_index",
    "peak_hour_flag",
]

# Contamination = expected % of anomalies in the data
# 2% is a reasonable assumption for grid stress events
CONTAMINATION = 0.02


# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────

def load_data(db_path: str) -> pd.DataFrame:
    print("Loading features table...")
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM features ORDER BY region, datetime", conn)
    conn.close()

    df["datetime"] = pd.to_datetime(df["datetime"])
    df[ANOMALY_FEATURES] = df[ANOMALY_FEATURES].fillna(0)

    print(f"  Loaded {len(df):,} rows")
    return df


# ─────────────────────────────────────────────
# 2. TRAIN ISOLATION FOREST
# ─────────────────────────────────────────────

def train_detector(df: pd.DataFrame):
    print("\nTraining Isolation Forest...")

    X = df[ANOMALY_FEATURES].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    detector = IsolationForest(
        n_estimators=200,
        contamination=CONTAMINATION,
        max_samples="auto",
        random_state=42,
        n_jobs=-1,
    )
    detector.fit(X_scaled)

    # Predict: -1 = anomaly, 1 = normal
    preds = detector.predict(X_scaled)
    scores = detector.decision_function(X_scaled)  # lower = more anomalous

    df = df.copy()
    df["anomaly_flag"]  = (preds == -1).astype(int)
    df["anomaly_score"] = scores

    n_anomalies = df["anomaly_flag"].sum()
    pct = n_anomalies / len(df) * 100
    print(f"  Total anomalies detected: {n_anomalies:,} ({pct:.2f}% of data)")

    # Show breakdown by region
    print("\n  Anomalies by region:")
    region_counts = df[df["anomaly_flag"] == 1].groupby("region").size().sort_values(ascending=False)
    for region, count in region_counts.items():
        total = len(df[df["region"] == region])
        print(f"    {region:<8} {count:>5} anomalies ({count/total*100:.1f}%)")

    return detector, scaler, df


# ─────────────────────────────────────────────
# 3. CLASSIFY ANOMALY TYPES
# ─────────────────────────────────────────────

def classify_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each anomaly, determine what type it is based on
    which feature deviates most from the rolling average.
    """
    print("\nClassifying anomaly types...")

    anomalies = df[df["anomaly_flag"] == 1].copy()

    # Compute z-scores for key features to find root cause
    for col in ["demand_mwh", "grid_stress_score", "renewable_pct", "fossil_co2_intensity"]:
        if col in anomalies.columns:
            mean = df[col].mean()
            std  = df[col].std() + 1e-6
            anomalies[f"z_{col}"] = (anomalies[col] - mean) / std

    def get_type(row):
        z_demand  = abs(row.get("z_demand_mwh", 0))
        z_stress  = abs(row.get("z_grid_stress_score", 0))
        z_renew   = abs(row.get("z_renewable_pct", 0))
        z_co2     = abs(row.get("z_fossil_co2_intensity", 0))

        max_z = max(z_demand, z_stress, z_renew, z_co2)
        if max_z == z_demand:
            return "Demand Spike" if row.get("z_demand_mwh", 0) > 0 else "Demand Drop"
        elif max_z == z_stress:
            return "Grid Stress Event"
        elif max_z == z_renew:
            return "Renewable Crash" if row.get("z_renewable_pct", 0) < 0 else "Renewable Surge"
        else:
            return "CO2 Spike"

    anomalies["anomaly_type"] = anomalies.apply(get_type, axis=1)

    type_counts = anomalies["anomaly_type"].value_counts()
    print("  Anomaly types:")
    for atype, count in type_counts.items():
        print(f"    {atype:<25} {count:>5}")

    return anomalies


# ─────────────────────────────────────────────
# 4. PLOTS
# ─────────────────────────────────────────────

def plot_anomalies(df: pd.DataFrame, anomalies: pd.DataFrame):
    print("\nGenerating anomaly visualization...")

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # Plot 1: Demand over time with anomalies highlighted (one region)
    region = df["region"].value_counts().index[0]  # most data
    reg_df  = df[df["region"] == region].sort_values("datetime").head(2000)
    reg_ano = anomalies[(anomalies["region"] == region)].sort_values("datetime")

    axes[0].plot(reg_df["datetime"], reg_df["demand_mwh"],
                 linewidth=0.8, color="#378ADD", alpha=0.8, label="Demand")
    if len(reg_ano) > 0:
        axes[0].scatter(reg_ano["datetime"], reg_ano["demand_mwh"],
                        color="#E24B4A", s=20, zorder=5, label="Anomaly", alpha=0.8)
    axes[0].set_title(f"Anomaly Detection — {region} (first 2000 hours)", fontsize=12)
    axes[0].set_ylabel("Demand (MWh)")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.2)

    # Plot 2: Anomaly type breakdown bar chart
    if "anomaly_type" in anomalies.columns:
        type_counts = anomalies["anomaly_type"].value_counts()
        colors = ["#E24B4A", "#BA7517", "#639922", "#378ADD", "#7F77DD"]
        type_counts.plot(kind="bar", ax=axes[1], color=colors[:len(type_counts)],
                         edgecolor="none")
        axes[1].set_title("Anomaly Types Distribution", fontsize=12)
        axes[1].set_ylabel("Count")
        axes[1].set_xlabel("")
        axes[1].tick_params(axis="x", rotation=30)
        axes[1].grid(True, axis="y", alpha=0.2)

    plt.tight_layout()
    path = os.path.join(SAVE_DIR, "anomaly_timeline.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved → {path}")


# ─────────────────────────────────────────────
# 5. SAVE OUTPUTS
# ─────────────────────────────────────────────

def save_outputs(detector, scaler, anomalies: pd.DataFrame):
    with open(os.path.join(SAVE_DIR, "anomaly_detector.pkl"), "wb") as f:
        pickle.dump(detector, f)
    with open(os.path.join(SAVE_DIR, "scaler_anomaly.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    # Save top anomaly alerts (most anomalous first)
    alert_cols = ["datetime", "region", "demand_mwh", "grid_stress_score",
                  "renewable_pct", "anomaly_score", "anomaly_type"]
    save_cols = [c for c in alert_cols if c in anomalies.columns]
    alerts = anomalies[save_cols].sort_values("anomaly_score").head(500)
    alerts.to_csv(os.path.join(SAVE_DIR, "anomaly_alerts.csv"), index=False)

    print(f"  Top 500 alerts saved → {SAVE_DIR}/anomaly_alerts.csv")
    print(f"  Models saved → {SAVE_DIR}")


# ─────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────

def run():
    print("=" * 55)
    print("  Week 3: Isolation Forest Anomaly Detection")
    print("=" * 55)

    df                     = load_data(DB_PATH)
    detector, scaler, df   = train_detector(df)
    anomalies              = classify_anomalies(df)
    plot_anomalies(df, anomalies)
    save_outputs(detector, scaler, anomalies)

    print("\n" + "=" * 55)
    print("  ✓ Anomaly detection complete!")
    print("  Open models/saved/anomaly_timeline.png")
    print("  Open models/saved/anomaly_alerts.csv for top alerts")
    print("=" * 55)


if __name__ == "__main__":
    run()