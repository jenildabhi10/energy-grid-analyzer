"""
models/cluster_states.py
========================
Week 3: K-Means State Energy Archetypes + PCA Visualization

Groups all 12 EIA regions into energy archetypes based on their
generation mix, demand patterns, and renewable transition stage.

Usage:
    cd C:\\Users\\Lenovo\\Downloads\\energy-grid-analyzer
    python models/cluster_states.py

Output:
    models/saved/kmeans_clusters.pkl      — trained KMeans model
    models/saved/scaler_clusters.pkl      — fitted StandardScaler
    models/saved/cluster_pca.png          — 2D PCA scatter plot
    models/saved/cluster_profiles.csv     — region → archetype mapping
"""

import os
import pickle
import sqlite3
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH  = os.path.join(BASE_DIR, "energy_grid.db")
SAVE_DIR = os.path.join(BASE_DIR, "models", "saved")
os.makedirs(SAVE_DIR, exist_ok=True)

# Features that best describe a region's energy profile
CLUSTER_FEATURES = [
    "renewable_pct",
    "fossil_pct",
    "fossil_co2_intensity",
    "per_capita_demand",
    "grid_stress_score",
    "rolling_7d_avg",
]

# Archetype names — assigned after inspecting cluster centers
ARCHETYPE_NAMES = {
    0: "Fossil Dependent",
    1: "Renewable Leader",
    2: "Balanced Grid",
    3: "High Demand",
    4: "Transitioning",
}

CLUSTER_COLORS = ["#E24B4A", "#639922", "#378ADD", "#BA7517", "#7F77DD"]


# ─────────────────────────────────────────────
# 1. BUILD REGION PROFILES
# ─────────────────────────────────────────────

def build_region_profiles(db_path: str) -> pd.DataFrame:
    """
    Aggregate hourly features to one row per region.
    Clustering works on region-level averages, not hourly rows.
    """
    print("Building region profiles...")
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM features", conn)
    conn.close()

    # Aggregate to region level
    profile = df.groupby("region")[CLUSTER_FEATURES].mean().reset_index()
    profile = profile.dropna()

    print(f"  Regions available: {len(profile)}")
    print(f"  Regions: {sorted(profile['region'].tolist())}")
    print("\n  Region profiles (mean values):")
    print(profile[["region"] + CLUSTER_FEATURES].to_string(index=False))

    return profile


# ─────────────────────────────────────────────
# 2. FIND OPTIMAL K (ELBOW + SILHOUETTE)
# ─────────────────────────────────────────────

def find_optimal_k(X_scaled: np.ndarray, max_k: int = 6) -> int:
    """
    Use silhouette score to find the best number of clusters.
    With only 12 regions, K=3-5 is usually optimal.
    """
    print("\nFinding optimal K...")

    scores = {}
    k_range = range(2, min(max_k + 1, len(X_scaled)))

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        scores[k] = score
        print(f"  K={k}  silhouette={score:.4f}")

    best_k = max(scores, key=scores.get)
    print(f"\n  Best K = {best_k} (silhouette={scores[best_k]:.4f})")
    return best_k


# ─────────────────────────────────────────────
# 3. TRAIN K-MEANS
# ─────────────────────────────────────────────

def train_kmeans(profile: pd.DataFrame):
    print("\nTraining K-Means...")

    X = profile[CLUSTER_FEATURES].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    best_k = find_optimal_k(X_scaled)

    # Use best_k but cap at 5 for meaningful archetypes
    k = min(best_k, 5)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)

    profile = profile.copy()
    profile["cluster"] = labels

    # Map cluster IDs to archetype names
    # Inspect cluster centers to assign meaningful names
    centers = pd.DataFrame(
        scaler.inverse_transform(km.cluster_centers_),
        columns=CLUSTER_FEATURES
    )
    print("\n  Cluster centers (original scale):")
    print(centers.round(4).to_string())

    # Auto-assign archetype names based on renewable_pct ranking
    centers["cluster_id"] = range(k)
    centers_sorted = centers.sort_values("renewable_pct", ascending=False)

    archetype_map = {}
    labels_list = ["Renewable Leader", "Transitioning", "Balanced Grid",
                   "Fossil Dependent", "High Demand"]
    for i, (_, row) in enumerate(centers_sorted.iterrows()):
        archetype_map[int(row["cluster_id"])] = labels_list[i] if i < len(labels_list) else f"Cluster {i}"

    profile["archetype"] = profile["cluster"].map(archetype_map)

    print("\n  Region → Archetype assignments:")
    for _, row in profile[["region", "cluster", "archetype"]].iterrows():
        print(f"    {row['region']:<8} → {row['archetype']}")

    return km, scaler, X_scaled, profile


# ─────────────────────────────────────────────
# 4. PCA VISUALIZATION
# ─────────────────────────────────────────────

def plot_pca(X_scaled: np.ndarray, profile: pd.DataFrame):
    print("\nGenerating PCA visualization...")

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    var1 = pca.explained_variance_ratio_[0] * 100
    var2 = pca.explained_variance_ratio_[1] * 100
    print(f"  PCA variance explained: PC1={var1:.1f}%, PC2={var2:.1f}%")

    unique_archetypes = profile["archetype"].unique()
    color_map = {a: CLUSTER_COLORS[i % len(CLUSTER_COLORS)]
                 for i, a in enumerate(unique_archetypes)}

    fig, ax = plt.subplots(figsize=(9, 6))

    for archetype in unique_archetypes:
        mask = profile["archetype"] == archetype
        ax.scatter(
            X_pca[mask, 0], X_pca[mask, 1],
            label=archetype,
            color=color_map[archetype],
            s=120, zorder=3
        )

    # Label each region point
    for i, row in profile.reset_index(drop=True).iterrows():
        ax.annotate(
            row["region"],
            (X_pca[i, 0], X_pca[i, 1]),
            textcoords="offset points",
            xytext=(8, 4),
            fontsize=9,
            color="#444"
        )

    ax.set_xlabel(f"PC1 ({var1:.1f}% variance)", fontsize=11)
    ax.set_ylabel(f"PC2 ({var2:.1f}% variance)", fontsize=11)
    ax.set_title("U.S. Energy Grid — State Archetypes (K-Means + PCA)", fontsize=13)
    ax.legend(title="Archetype", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(SAVE_DIR, "cluster_pca.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  PCA plot saved → {path}")


# ─────────────────────────────────────────────
# 5. SAVE
# ─────────────────────────────────────────────

def save_outputs(km, scaler, profile: pd.DataFrame):
    with open(os.path.join(SAVE_DIR, "kmeans_clusters.pkl"), "wb") as f:
        pickle.dump(km, f)
    with open(os.path.join(SAVE_DIR, "scaler_clusters.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    profile.to_csv(os.path.join(SAVE_DIR, "cluster_profiles.csv"), index=False)
    print(f"  Models + profiles saved → {SAVE_DIR}")


# ─────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────

def run():
    print("=" * 55)
    print("  Week 3: K-Means State Energy Archetypes")
    print("=" * 55)

    profile              = build_region_profiles(DB_PATH)
    km, scaler, X_scaled, profile = train_kmeans(profile)
    plot_pca(X_scaled, profile)
    save_outputs(km, scaler, profile)

    print("\n" + "=" * 55)
    print("  ✓ Clustering complete!")
    print("  Open models/saved/cluster_pca.png to see archetypes")
    print("=" * 55)


if __name__ == "__main__":
    run()