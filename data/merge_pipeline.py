
import os
import sqlite3
import numpy as np
import pandas as pd

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "energy_grid.db")


# ─────────────────────────────────────────────
# 1. LOAD RAW TABLES
# ─────────────────────────────────────────────

def load_tables(db_path: str) -> dict:
    """Load all raw tables from SQLite into DataFrames."""
    conn = sqlite3.connect(db_path)

    print("Loading raw tables...")

    eia_demand = pd.read_sql("SELECT * FROM eia_demand", conn)
    eia_gen    = pd.read_sql("SELECT * FROM eia_generation", conn)
    noaa       = pd.read_sql("SELECT * FROM noaa_weather", conn)
    census     = pd.read_sql("SELECT * FROM census_state_profile", conn)

    conn.close()

    print(f"  eia_demand:      {len(eia_demand):>10,} rows")
    print(f"  eia_generation:  {len(eia_gen):>10,} rows")
    print(f"  noaa_weather:    {len(noaa):>10,} rows")
    print(f"  census_profile:  {len(census):>10,} rows")

    return {
        "eia_demand": eia_demand,
        "eia_gen":    eia_gen,
        "noaa":       noaa,
        "census":     census,
    }


# ─────────────────────────────────────────────
# 2. CLEAN & PREP EACH TABLE
# ─────────────────────────────────────────────

def prep_eia_demand(df: pd.DataFrame) -> pd.DataFrame:
    """Clean EIA demand table and extract time components."""
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["date"]     = df["datetime"].dt.date.astype(str)
    df["hour"]     = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek   # 0=Mon, 6=Sun
    df["month"]    = df["datetime"].dt.month
    df["year"]     = df["datetime"].dt.year
    df["demand_mwh"] = pd.to_numeric(df["demand_mwh"], errors="coerce")
    return df.dropna(subset=["demand_mwh"])


def prep_eia_generation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot generation from long (one row per fuel type) to wide
    (one row per datetime+region with columns per fuel).
    """
    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["generation_mwh"] = pd.to_numeric(df["generation_mwh"], errors="coerce")
    df = df.dropna(subset=["generation_mwh"])

    # Normalize fuel type names
    df["fuel_type"] = df["fuel_type"].str.strip().str.lower()

    # Pivot: one column per fuel type
    gen_wide = df.pivot_table(
        index=["datetime", "region"],
        columns="fuel_type",
        values="generation_mwh",
        aggfunc="sum"
    ).reset_index()

    # Flatten column names
    gen_wide.columns.name = None

    # Ensure key fuel columns exist even if missing in data
    for fuel in ["col", "ng", "nuc", "wat", "sun", "wnd", "oth"]:
        if fuel not in gen_wide.columns:
            gen_wide[fuel] = 0.0

    # Rename to readable names
    rename_fuels = {
        "col": "gen_coal",
        "ng":  "gen_gas",
        "nuc": "gen_nuclear",
        "wat": "gen_hydro",
        "sun": "gen_solar",
        "wnd": "gen_wind",
        "oth": "gen_other",
    }
    gen_wide = gen_wide.rename(columns={k: v for k, v in rename_fuels.items() if k in gen_wide.columns})

    # Total generation
    gen_cols = [c for c in gen_wide.columns if c.startswith("gen_")]
    gen_wide["total_gen_mwh"] = gen_wide[gen_cols].sum(axis=1)

    return gen_wide


def prep_noaa(df: pd.DataFrame) -> pd.DataFrame:
    """Clean NOAA weather — daily resolution, will join on date."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)
    for col in ["tmax_c", "tmin_c", "tavg_c", "precip_mm", "wind_ms"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # Fill missing tavg with average of tmax/tmin
    if "tavg_c" not in df.columns:
        df["tavg_c"] = (df["tmax_c"] + df["tmin_c"]) / 2
    return df[["date", "region", "tavg_c", "tmax_c", "tmin_c", "precip_mm", "wind_ms"]].dropna(subset=["tavg_c"])


def prep_census(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only region-level aggregates for joining."""
    df = df.copy()
    # Aggregate to region level (multiple states per region)
    region_agg = df.groupby("region").agg(
        population=("population", "sum"),
        gdp_billions=("gdp_billions_usd", "sum"),
    ).reset_index()
    region_agg["gdp_per_capita"] = (region_agg["gdp_billions"] * 1e9) / region_agg["population"]
    return region_agg.dropna(subset=["region"])


# ─────────────────────────────────────────────
# 3. JOIN ALL TABLES
# ─────────────────────────────────────────────

def join_tables(demand: pd.DataFrame, gen: pd.DataFrame,
                noaa: pd.DataFrame, census: pd.DataFrame) -> pd.DataFrame:
    """
    Join all 4 prepped tables into one wide DataFrame.
    Base = hourly demand. Join generation on datetime+region,
    weather on date+region, census on region.
    """
    print("\nJoining tables...")

    # 1. Demand + Generation (hourly, exact match)
    df = pd.merge(demand, gen, on=["datetime", "region"], how="left")
    print(f"  After demand+generation join: {len(df):,} rows")

    # 2. Add weather (daily — join on date+region)
    df = pd.merge(df, noaa, on=["date", "region"], how="left")
    print(f"  After weather join:           {len(df):,} rows")

    # 3. Add census (static — join on region)
    df = pd.merge(df, census, on="region", how="left")
    print(f"  After census join:            {len(df):,} rows")

    return df


# ─────────────────────────────────────────────
# 4. FEATURE ENGINEERING — 15 FEATURES
# ─────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build all 15 ML features from the joined DataFrame.

    Features:
        1.  grid_stress_score      — demand / total_gen (capacity proxy)
        2.  renewable_pct          — (wind+solar+hydro) / total_gen
        3.  fossil_pct             — (coal+gas) / total_gen
        4.  fossil_co2_intensity   — weighted CO2 per MWh (kg/MWh)
        5.  heat_index             — temperature × humidity proxy
        6.  demand_lag_24h         — same hour yesterday
        7.  demand_lag_168h        — same hour last week
        8.  rolling_7d_avg         — 7-day rolling mean demand
        9.  per_capita_demand      — demand_mwh / population (normalized)
        10. renewable_growth_mom   — month-over-month renewable % change
        11. peak_hour_flag         — 1 if hour in [7-9, 17-21], else 0
        12. hour_sin               — cyclical hour encoding (sin)
        13. hour_cos               — cyclical hour encoding (cos)
        14. month_sin              — cyclical month encoding (sin)
        15. month_cos              — cyclical month encoding (cos)
    """
    print("\nEngineering features...")
    df = df.copy()
    df = df.sort_values(["region", "datetime"]).reset_index(drop=True)

    # ── Feature 1: Grid Stress Score ──────────────────────────
    # demand / total generation (values > 1 = stressed grid)
    df["grid_stress_score"] = np.where(
        df["total_gen_mwh"] > 0,
        df["demand_mwh"] / df["total_gen_mwh"],
        np.nan
    )
    df["grid_stress_score"] = df["grid_stress_score"].clip(0, 2)  # cap outliers

    # ── Feature 2: Renewable % ────────────────────────────────
    renewable_cols = [c for c in ["gen_wind", "gen_solar", "gen_hydro"] if c in df.columns]
    df["renewable_mwh"] = df[renewable_cols].sum(axis=1)
    df["renewable_pct"] = np.where(
        df["total_gen_mwh"] > 0,
        df["renewable_mwh"] / df["total_gen_mwh"],
        np.nan
    )

    # ── Feature 3: Fossil % ───────────────────────────────────
    fossil_cols = [c for c in ["gen_coal", "gen_gas"] if c in df.columns]
    df["fossil_mwh"] = df[fossil_cols].sum(axis=1)
    df["fossil_pct"] = np.where(
        df["total_gen_mwh"] > 0,
        df["fossil_mwh"] / df["total_gen_mwh"],
        np.nan
    )

    # ── Feature 4: CO2 Intensity ──────────────────────────────
    # Approximate emission factors (kg CO2 per MWh)
    # Source: EPA eGRID average factors
    CO2_COAL = 1001  # kg/MWh
    CO2_GAS  = 469   # kg/MWh
    coal = df.get("gen_coal", pd.Series(0, index=df.index)).fillna(0)
    gas  = df.get("gen_gas",  pd.Series(0, index=df.index)).fillna(0)
    total = df["total_gen_mwh"].replace(0, np.nan)
    df["fossil_co2_intensity"] = (coal * CO2_COAL + gas * CO2_GAS) / total

    # ── Feature 5: Heat Index ─────────────────────────────────
    # Simple proxy: tmax_c (higher temp → higher AC demand)
    df["heat_index"] = df.get("tmax_c", pd.Series(np.nan, index=df.index))

    # ── Features 6 & 7: Lag Features ─────────────────────────
    # Must group by region to avoid mixing regions
    df["demand_lag_24h"]  = df.groupby("region")["demand_mwh"].shift(24)
    df["demand_lag_168h"] = df.groupby("region")["demand_mwh"].shift(168)

    # ── Feature 8: Rolling 7-Day Average ─────────────────────
    df["rolling_7d_avg"] = (
        df.groupby("region")["demand_mwh"]
        .transform(lambda x: x.rolling(168, min_periods=24).mean())
    )

    # ── Feature 9: Per-Capita Demand ─────────────────────────
    df["per_capita_demand"] = np.where(
        df["population"] > 0,
        df["demand_mwh"] / (df["population"] / 1e6),  # MWh per million people
        np.nan
    )

    # ── Feature 10: Renewable Growth MoM ─────────────────────
    monthly_ren = (
        df.groupby(["region", "year", "month"])["renewable_pct"]
        .mean()
        .reset_index()
        .rename(columns={"renewable_pct": "monthly_ren_avg"})
    )
    monthly_ren["renewable_growth_mom"] = (
        monthly_ren.groupby("region")["monthly_ren_avg"]
        .pct_change()
        .fillna(0)
    )
    df = pd.merge(df, monthly_ren[["region", "year", "month", "renewable_growth_mom"]],
                  on=["region", "year", "month"], how="left")

    # ── Feature 11: Peak Hour Flag ────────────────────────────
    # Morning peak: 7–9am, Evening peak: 5–9pm
    df["peak_hour_flag"] = df["hour"].apply(
        lambda h: 1 if (7 <= h <= 9 or 17 <= h <= 21) else 0
    )

    # ── Features 12 & 13: Cyclical Hour Encoding ─────────────
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # ── Features 14 & 15: Cyclical Month Encoding ────────────
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    print(f"  Features engineered: 15")
    print(f"  Final shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

    return df


# ─────────────────────────────────────────────
# 5. SAVE TO SQLITE
# ─────────────────────────────────────────────

def save_features(df: pd.DataFrame, db_path: str):
    """Save the feature table to SQLite, replacing any existing version."""

    # Select final columns to save
    feature_cols = [
        "datetime", "region", "hour", "day_of_week", "month", "year",
        # Raw
        "demand_mwh", "total_gen_mwh",
        "gen_coal", "gen_gas", "gen_nuclear", "gen_hydro", "gen_solar", "gen_wind",
        "tavg_c", "tmax_c", "tmin_c", "precip_mm", "wind_ms",
        "population", "gdp_per_capita",
        # 15 Engineered Features
        "grid_stress_score",
        "renewable_pct",
        "fossil_pct",
        "fossil_co2_intensity",
        "heat_index",
        "demand_lag_24h",
        "demand_lag_168h",
        "rolling_7d_avg",
        "per_capita_demand",
        "renewable_growth_mom",
        "peak_hour_flag",
        "hour_sin",
        "hour_cos",
        "month_sin",
        "month_cos",
    ]

    # Only keep columns that exist
    save_cols = [c for c in feature_cols if c in df.columns]
    df_save = df[save_cols].copy()

    # Convert datetime to string for SQLite
    df_save["datetime"] = df_save["datetime"].astype(str)

    conn = sqlite3.connect(db_path)
    df_save.to_sql("features", conn, if_exists="replace", index=False)
    conn.close()

    print(f"\n✓ Saved {len(df_save):,} rows → 'features' table in {db_path}")
    print(f"  Columns saved: {len(save_cols)}")


# ─────────────────────────────────────────────
# 6. VALIDATION REPORT
# ─────────────────────────────────────────────

def print_validation(df: pd.DataFrame):
    """Print a quick data quality report after feature engineering."""
    print("\n=== Feature Validation Report ===")

    feature_cols = [
        "grid_stress_score", "renewable_pct", "fossil_pct",
        "fossil_co2_intensity", "heat_index", "demand_lag_24h",
        "demand_lag_168h", "rolling_7d_avg", "per_capita_demand",
        "renewable_growth_mom", "peak_hour_flag",
        "hour_sin", "hour_cos", "month_sin", "month_cos"
    ]

    for col in feature_cols:
        if col in df.columns:
            null_pct = df[col].isna().mean() * 100
            val_range = f"{df[col].min():.3f} → {df[col].max():.3f}" if df[col].notna().any() else "all null"
            print(f"  {col:<28} nulls: {null_pct:5.1f}%   range: {val_range}")

    print(f"\n  Regions covered: {sorted(df['region'].unique().tolist())}")
    print(f"  Date range:      {df['datetime'].min()} → {df['datetime'].max()}")


# ─────────────────────────────────────────────
# 7. MAIN
# ─────────────────────────────────────────────

def run():
    print("=" * 55)
    print("  Week 2: ETL Pipeline & Feature Engineering")
    print("=" * 55)

    # Load
    tables = load_tables(DB_PATH)

    # Prep
    demand = prep_eia_demand(tables["eia_demand"])
    gen    = prep_eia_generation(tables["eia_gen"])
    noaa   = prep_noaa(tables["noaa"])
    census = prep_census(tables["census"])

    # Join
    df = join_tables(demand, gen, noaa, census)

    # Engineer features
    df = engineer_features(df)

    # Validate
    print_validation(df)

    # Save
    save_features(df, DB_PATH)

    print("\n✓ ETL complete. Ready for Week 3: ML Models.")


if __name__ == "__main__":
    run()