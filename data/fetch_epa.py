"""
data/fetch_epa.py
=================
Fetches air quality data (PM2.5, SO2, NOx, Ozone) from the EPA AQS API.
These pollutants directly correlate with fossil fuel plant activity.

Free registration: https://aqs.epa.gov/aqsweb/documents/data_api.html#signup
  → Just POST your email/username to get an API key back immediately.
API docs: https://aqs.epa.gov/aqsweb/documents/data_api.html

Strategy: Pull state-level daily summary data (pre-aggregated by EPA).
Much faster than pulling individual monitor readings.
"""

import os
import time
import sqlite3
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
EPA_EMAIL = os.getenv("EPA_EMAIL")
EPA_KEY   = os.getenv("EPA_API_KEY", "test")   # 'test' key works for exploration

BASE_URL = "https://aqs.epa.gov/data/api"

# Parameter codes we care about
# Full list: https://aqs.epa.gov/aqsweb/documents/codetables/parameters.html
PARAMS = {
    "88101": "pm25",        # PM2.5 (24-hr avg) — most important
    "42401": "so2",         # Sulfur Dioxide
    "42602": "no2",         # Nitrogen Dioxide
    "44201": "ozone",       # Ozone
}

# State FIPS codes → EIA region mapping
# EPA API requires 2-digit FIPS state codes
EIA_REGION_STATES = {
    "CAL":  ["06"],         # California
    "TEX":  ["48"],         # Texas
    "NY":   ["36"],         # New York
    "FLA":  ["12"],         # Florida
    "MIDA": ["24", "11", "51", "10", "42"],  # MD, DC, VA, DE, PA
    "MIDW": ["17", "18", "26", "39", "55"],  # IL, IN, MI, OH, WI
    "NE":   ["09", "23", "25", "33", "44", "50"],  # CT ME MA NH RI VT
    "NW":   ["41", "53", "16"],  # OR, WA, ID
    "SE":   ["01", "13", "28"],  # AL, GA, MS
    "SW":   ["04", "32", "35"],  # AZ, NV, NM
    "CAR":  ["37", "45"],   # NC, SC
    "TEN":  ["47"],         # Tennessee
}


def fetch_state_daily(state_fips: str, param_code: str, param_name: str,
                      year: int, region: str) -> pd.DataFrame:
    """
    Pull daily air quality summaries for a state + parameter + year.
    EPA AQS daily summary endpoint returns statistics across all monitors in the state.
    """
    url = f"{BASE_URL}/dailyData/byState"
    params = {
        "email":   EPA_EMAIL,
        "key":     EPA_KEY,
        "param":   param_code,
        "bdate":   f"{year}0101",
        "edate":   f"{year}1231",
        "state":   state_fips,
    }

    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    body = resp.json()

    if body.get("Header", [{}])[0].get("status") == "No data matched your selection":
        return pd.DataFrame()

    data = body.get("Data", [])
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)

    # EPA returns many columns — keep what matters
    keep = ["date_local", "state_code", "arithmetic_mean", "units_of_measure"]
    df = df[[c for c in keep if c in df.columns]].copy()

    df = df.rename(columns={
        "date_local": "date",
        "state_code": "state_fips",
        "arithmetic_mean": param_name,
    })

    # Average across monitors if multiple per day
    df["date"] = pd.to_datetime(df["date"])
    df[param_name] = pd.to_numeric(df[param_name], errors="coerce")
    df = df.groupby("date")[param_name].mean().reset_index()

    df["state_fips"] = state_fips
    df["region"] = region
    return df


def run(years: list = None, db_path: str = "energy_grid.db"):
    """
    Fetch EPA data for specified years. Each param/state/year = one API call.
    With 12 regions × ~2.5 states avg × 4 params × 2 years ≈ ~240 calls.
    Allow ~10 min for a full run (EPA rate-limits to ~10 req/sec).
    """
    if years is None:
        years = [2024]  # start with one year to test

    if not EPA_EMAIL:
        raise ValueError("EPA_EMAIL not found in .env — required for AQS API registration")

    print(f"\n=== EPA AQS Ingestion: years={years} ===\n")
    all_frames = []

    for region, state_list in EIA_REGION_STATES.items():
        for state_fips in state_list:
            for year in years:
                for param_code, param_name in PARAMS.items():
                    print(f"  [{region}] state={state_fips} param={param_name} year={year}")
                    try:
                        df = fetch_state_daily(state_fips, param_code, param_name, year, region)
                        if not df.empty:
                            all_frames.append(df)
                    except requests.HTTPError as e:
                        print(f"    ERROR: {e}")
                    time.sleep(0.15)  # stay under rate limit

    if not all_frames:
        print("No data fetched — check your EPA_EMAIL in .env")
        return

    # Merge all pollutants for same region/date
    combined = all_frames[0]
    for df in all_frames[1:]:
        combined = pd.merge(combined, df, on=["date", "state_fips", "region"], how="outer")

    conn = sqlite3.connect(db_path)
    combined.to_sql("epa_air_quality", conn, if_exists="append", index=False)
    conn.close()
    print(f"\n✓ Saved {len(combined):,} rows → epa_air_quality")


if __name__ == "__main__":
    # Test: just California, 2024
    # Edit EIA_REGION_STATES above to limit scope during testing
    run(years=[2024])