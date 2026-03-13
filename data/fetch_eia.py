"""
data/fetch_eia.py
=================
Fetches electricity data from the EIA Open Data API v2.
Pulls hourly demand + generation by fuel type for all US regions.

API docs: https://www.eia.gov/opendata/documentation.php
Free key: https://www.eia.gov/opendata/register.php (instant, no approval)
"""

import os
import time
import sqlite3
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
EIA_KEY = os.getenv("EIA_API_KEY")

BASE_URL = "https://api.eia.gov/v2"

# EIA region codes → U.S. states/regions
# Full list: https://www.eia.gov/electricity/gridmonitor/about
EIA_REGIONS = [
    "CAL",  # California
    "TEX",  # Texas (ERCOT)
    "NY",   # New York
    "FLA",  # Florida
    "MIDA", # Mid-Atlantic
    "MIDW", # Midwest
    "NE",   # New England
    "NW",   # Northwest
    "SE",   # Southeast
    "SW",   # Southwest
    "CAR",  # Carolinas
    "TEN",  # Tennessee
]


def fetch_hourly_demand(region: str, start: str, end: str) -> pd.DataFrame:
    """
    Pull hourly electricity demand (MWh) for a given EIA region.

    Args:
        region: EIA region code, e.g. "CAL"
        start:  ISO date string "YYYY-MM-DD"
        end:    ISO date string "YYYY-MM-DD"

    Returns:
        DataFrame with columns: period, region, demand_mwh
    """
    url = f"{BASE_URL}/electricity/rto/region-data/data/"
    params = {
        "api_key": EIA_KEY,
        "frequency": "hourly",
        "data[0]": "value",
        "facets[respondent][]": region,
        "facets[type][]": "D",          # D = demand
        "start": start,
        "end": end,
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "length": 5000,
    }

    rows = []
    offset = 0

    while True:
        params["offset"] = offset
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()["response"]

        batch = data.get("data", [])
        if not batch:
            break

        rows.extend(batch)
        total = int(data.get("total", 0))
        offset += len(batch)

        print(f"  [{region}] demand: fetched {offset}/{total} rows")
        if offset >= total:
            break

        time.sleep(0.3)  # be polite to the API

    if not rows:
        print(f"  [{region}] No demand data found for {start}–{end}")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.rename(columns={"period": "datetime", "value": "demand_mwh", "respondent": "region"})
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["demand_mwh"] = pd.to_numeric(df["demand_mwh"], errors="coerce")
    df["region"] = region
    return df[["datetime", "region", "demand_mwh"]].dropna()


def fetch_generation_by_fuel(region: str, start: str, end: str) -> pd.DataFrame:
    """
    Pull hourly generation by fuel type (coal, gas, wind, solar, etc.)

    Returns:
        DataFrame with columns: datetime, region, fuel_type, generation_mwh
    """
    url = f"{BASE_URL}/electricity/rto/fuel-type-data/data/"
    params = {
        "api_key": EIA_KEY,
        "frequency": "hourly",
        "data[0]": "value",
        "facets[respondent][]": region,
        "start": start,
        "end": end,
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "length": 5000,
    }

    rows = []
    offset = 0

    while True:
        params["offset"] = offset
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()["response"]

        batch = data.get("data", [])
        if not batch:
            break

        rows.extend(batch)
        total = int(data.get("total", 0))
        offset += len(batch)
        print(f"  [{region}] generation: fetched {offset}/{total} rows")
        if offset >= total:
            break

        time.sleep(0.3)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = df.rename(columns={
        "period": "datetime",
        "value": "generation_mwh",
        "respondent": "region",
        "fueltype": "fuel_type",
    })
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["generation_mwh"] = pd.to_numeric(df["generation_mwh"], errors="coerce")
    df["region"] = region
    return df[["datetime", "region", "fuel_type", "generation_mwh"]].dropna()


def save_to_db(df: pd.DataFrame, table: str, db_path: str = "energy_grid.db"):
    """Append a DataFrame to a SQLite table (creates if not exists)."""
    if df.empty:
        print(f"  Skipping {table} — empty DataFrame")
        return
    conn = sqlite3.connect(db_path)
    df.to_sql(table, conn, if_exists="append", index=False)
    conn.close()
    print(f"  Saved {len(df):,} rows → {table}")


def run(start: str = "2023-01-01", end: str = "2024-12-31"):
    """
    Main entry point. Fetches all regions and saves to SQLite.
    Start with a short date range (e.g. 1 month) to test, then expand.
    """
    if not EIA_KEY:
        raise ValueError("EIA_API_KEY not found. Add it to your .env file.")

    print(f"\n=== EIA Ingestion: {start} → {end} ===\n")

    all_demand = []
    all_gen = []

    for region in EIA_REGIONS:
        print(f"\n→ Region: {region}")
        try:
            demand_df = fetch_hourly_demand(region, start, end)
            all_demand.append(demand_df)

            gen_df = fetch_generation_by_fuel(region, start, end)
            all_gen.append(gen_df)
        except requests.HTTPError as e:
            print(f"  ERROR {region}: {e}")
            continue

    if all_demand:
        demand_combined = pd.concat(all_demand, ignore_index=True)
        save_to_db(demand_combined, "eia_demand")

    if all_gen:
        gen_combined = pd.concat(all_gen, ignore_index=True)
        save_to_db(gen_combined, "eia_generation")

    print("\n✓ EIA ingestion complete.")


if __name__ == "__main__":
    # Quick test: pull just January 2024 for California
    # Once it works, expand the date range and regions
    run(start="2024-01-01", end="2024-01-31")