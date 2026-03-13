"""
data/fetch_noaa.py
==================
Fetches daily weather data (temperature, wind, precipitation) from
the NOAA Climate Data Online (CDO) API v2.

Free token: https://www.ncdc.noaa.gov/cdo-web/token (emailed instantly)
API docs:   https://www.ncdc.noaa.gov/cdo-web/webservices/v2

Note: NOAA CDO is daily-resolution, not hourly. We'll interpolate to hourly
during the ETL pipeline (Week 2). Good enough for demand correlation.
"""

import os
import time
import sqlite3
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
NOAA_TOKEN = os.getenv("NOAA_TOKEN")

BASE_URL = "https://www.ncdc.noaa.gov/cdo-web/api/v2"

# One representative weather station per EIA region
# These are major-city airport stations (most complete data)
# FIPS = station ID format for GHCND dataset
REGION_STATIONS = {
    "CAL":  "GHCND:USW00023174",   # Los Angeles Int'l Airport
    "TEX":  "GHCND:USW00012960",   # Houston Bush Airport
    "NY":   "GHCND:USW00094728",   # NYC Central Park
    "FLA":  "GHCND:USW00012839",   # Miami Int'l Airport
    "MIDA": "GHCND:USW00013743",   # Washington Dulles
    "MIDW": "GHCND:USW00094846",   # Chicago O'Hare
    "NE":   "GHCND:USW00014739",   # Boston Logan
    "NW":   "GHCND:USW00024233",   # Seattle-Tacoma
    "SE":   "GHCND:USW00003813",   # Atlanta Hartsfield
    "SW":   "GHCND:USW00023183",   # Phoenix Sky Harbor
    "CAR":  "GHCND:USW00013881",   # Charlotte Douglas
    "TEN":  "GHCND:USW00013897",   # Nashville Int'l
}

# Data types we want from NOAA
# TMAX/TMIN in tenths of °C, PRCP in tenths of mm, AWND in tenths of m/s
DATA_TYPES = "TMAX,TMIN,PRCP,AWND,SNOW"


def fetch_station_data(station_id: str, region: str, start: str, end: str) -> pd.DataFrame:
    """
    Fetch daily climate data for a single station.
    NOAA CDO limits: 1000 results per request, 5 requests/second.

    Args:
        station_id: GHCND station ID string
        region:     EIA region label (for joining later)
        start/end:  "YYYY-MM-DD"

    Returns:
        DataFrame: date, region, tmax_c, tmin_c, tavg_c, precip_mm, wind_ms
    """
    url = f"{BASE_URL}/data"
    headers = {"token": NOAA_TOKEN}

    all_rows = []
    offset = 1  # NOAA uses 1-based offset

    while True:
        params = {
            "datasetid": "GHCND",
            "stationid": station_id,
            "datatypeid": DATA_TYPES,
            "startdate": start,
            "enddate": end,
            "limit": 1000,
            "offset": offset,
            "units": "metric",
        }

        resp = requests.get(url, headers=headers, params=params, timeout=30)

        if resp.status_code == 204:
            print(f"  [{region}] No data for this station/period")
            break
        resp.raise_for_status()

        body = resp.json()
        results = body.get("results", [])
        if not results:
            break

        all_rows.extend(results)
        metadata = body.get("metadata", {}).get("resultset", {})
        total = metadata.get("count", 0)
        offset += len(results)
        print(f"  [{region}] weather: fetched {min(offset-1, total)}/{total} rows")

        if offset > total:
            break

        time.sleep(0.25)  # stay under 5 req/sec limit

    if not all_rows:
        return pd.DataFrame()

    # Pivot from long (one row per datatype) to wide (one row per date)
    df_long = pd.DataFrame(all_rows)
    df_long["date"] = pd.to_datetime(df_long["date"]).dt.date
    df_wide = df_long.pivot_table(
        index="date", columns="datatype", values="value", aggfunc="mean"
    ).reset_index()

    # Rename and convert units
    rename = {"TMAX": "tmax_c", "TMIN": "tmin_c", "PRCP": "precip_mm", "AWND": "wind_ms", "SNOW": "snow_mm"}
    df_wide = df_wide.rename(columns={k: v for k, v in rename.items() if k in df_wide.columns})

    # Compute average temp if both max and min exist
    if "tmax_c" in df_wide.columns and "tmin_c" in df_wide.columns:
        df_wide["tavg_c"] = (df_wide["tmax_c"] + df_wide["tmin_c"]) / 2

    df_wide["region"] = region
    df_wide["date"] = pd.to_datetime(df_wide["date"])

    # Ensure expected columns even if NOAA didn't return all types
    for col in ["tmax_c", "tmin_c", "tavg_c", "precip_mm", "wind_ms", "snow_mm"]:
        if col not in df_wide.columns:
            df_wide[col] = None

    return df_wide


def save_to_db(df: pd.DataFrame, db_path: str = "energy_grid.db"):
    if df.empty:
        return
    conn = sqlite3.connect(db_path)
    df.to_sql("noaa_weather", conn, if_exists="append", index=False)
    conn.close()
    print(f"  Saved {len(df):,} rows → noaa_weather")


def run(start: str = "2023-01-01", end: str = "2024-12-31"):
    if not NOAA_TOKEN:
        raise ValueError("NOAA_TOKEN not found. Add it to your .env file.")

    print(f"\n=== NOAA Ingestion: {start} → {end} ===\n")
    all_frames = []

    for region, station_id in REGION_STATIONS.items():
        print(f"\n→ Region: {region} | Station: {station_id}")
        try:
            df = fetch_station_data(station_id, region, start, end)
            all_frames.append(df)
        except requests.HTTPError as e:
            print(f"  ERROR {region}: {e}")

    if all_frames:
        combined = pd.concat(all_frames, ignore_index=True)
        save_to_db(combined)

    print("\n✓ NOAA ingestion complete.")


if __name__ == "__main__":
    run(start="2024-01-01", end="2024-01-31")