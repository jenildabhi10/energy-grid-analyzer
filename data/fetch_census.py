import os
import sqlite3
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
CENSUS_KEY = os.getenv("CENSUS_API_KEY")

BASE_URL = "https://api.census.gov/data"

# Map Census state FIPS → EIA region labels
# Same mapping used in fetch_epa.py
FIPS_TO_REGION = {
    "06": "CAL", "48": "TEX", "36": "NY",  "12": "FLA",
    "24": "MIDA","11": "MIDA","51": "MIDA","10": "MIDA","42": "MIDA",
    "17": "MIDW","18": "MIDW","26": "MIDW","39": "MIDW","55": "MIDW",
    "09": "NE",  "23": "NE",  "25": "NE",  "33": "NE",  "44": "NE",  "50": "NE",
    "41": "NW",  "53": "NW",  "16": "NW",
    "01": "SE",  "13": "SE",  "28": "SE",
    "04": "SW",  "32": "SW",  "35": "SW",
    "37": "CAR", "45": "CAR",
    "47": "TEN",
}


def fetch_state_population(year: int = 2022) -> pd.DataFrame:
    """
    Pull state population from ACS 5-Year Estimates.
    B01003_001E = total population estimate.

    Args:
        year: ACS data year (2022 is latest stable as of 2024)

    Returns:
        DataFrame: state_fips, state_name, population, region
    """
    url = f"{BASE_URL}/{year}/acs/acs5"
    params = {
        "get": "NAME,B01003_001E",  # state name + total pop
        "for": "state:*",           # all 50 states + DC
        "key": CENSUS_KEY,
    }

    resp = requests.get(url, params=params, timeout=30)

    # Diagnose bad responses before trying to parse JSON
    if resp.status_code != 200:
        raise ValueError(f"Census API returned HTTP {resp.status_code}.\nResponse: {resp.text[:300]}")

    raw = resp.text.strip()
    if not raw.startswith("["):
        # Census returns plain-text errors (e.g. "Invalid Key")
        raise ValueError(f"Census API error (not JSON):\n{raw[:300]}\n\nCheck your CENSUS_API_KEY in .env")

    rows = resp.json()
    headers = rows[0]
    data = rows[1:]

    df = pd.DataFrame(data, columns=headers)
    df = df.rename(columns={
        "NAME": "state_name",
        "B01003_001E": "population",
        "state": "state_fips",
    })

    df["population"] = pd.to_numeric(df["population"], errors="coerce")
    df["data_year"] = year

    # Map FIPS → EIA region (some states won't map — that's ok)
    df["region"] = df["state_fips"].map(FIPS_TO_REGION)

    print(f"  Fetched population for {len(df)} states (year={year})")
    return df[["state_fips", "state_name", "population", "region", "data_year"]]


def fetch_state_gdp() -> pd.DataFrame:
    """
    Pull state GDP from BEA (Bureau of Economic Analysis).
    Census API doesn't include GDP directly — we use a precomputed
    BEA table via their public API.

    Note: BEA API is separate from Census and requires its own free key.
    For simplicity, we include hardcoded 2022 state GDP (billions USD)
    sourced from BEA Table SAGDP2N. Update yearly from:
    https://www.bea.gov/data/gdp/gdp-state

    Returns:
        DataFrame: state_name, gdp_billions_usd
    """
    # BEA 2022 state GDP — billions of current dollars
    # Source: BEA SAGDP2N, All industry total
    gdp_data = {
        "California": 3598.1, "Texas": 2355.9, "New York": 2053.2,
        "Florida": 1389.7, "Illinois": 1033.9, "Pennsylvania": 907.3,
        "Ohio": 779.0, "Georgia": 756.6, "Washington": 762.8,
        "New Jersey": 728.0, "North Carolina": 706.5, "Virginia": 663.0,
        "Massachusetts": 657.8, "Colorado": 501.3, "Tennessee": 479.5,
        "Michigan": 581.9, "Indiana": 428.1, "Minnesota": 433.1,
        "Arizona": 464.7, "Maryland": 461.5, "Wisconsin": 379.7,
        "Missouri": 370.7, "Connecticut": 324.3, "Oregon": 312.5,
        "South Carolina": 289.8, "Alabama": 265.5, "Louisiana": 269.6,
        "Kentucky": 261.1, "Oklahoma": 244.4, "Iowa": 225.6,
        "Nevada": 223.1, "Arkansas": 155.5, "Mississippi": 133.7,
        "Kansas": 189.3, "Utah": 248.5, "Nebraska": 163.3,
        "New Mexico": 115.3, "Idaho": 113.3, "Hawaii": 94.4,
        "West Virginia": 97.5, "New Hampshire": 100.5, "Maine": 88.1,
        "Rhode Island": 72.4, "Montana": 71.4, "Delaware": 89.6,
        "South Dakota": 65.7, "North Dakota": 73.6, "Alaska": 63.4,
        "Vermont": 44.6, "Wyoming": 52.1, "District of Columbia": 155.5,
    }

    df = pd.DataFrame(list(gdp_data.items()), columns=["state_name", "gdp_billions_usd"])
    print(f"  Loaded GDP for {len(df)} states (BEA 2022)")
    return df


def run(db_path: str = "energy_grid.db"):
    if not CENSUS_KEY:
        raise ValueError("CENSUS_API_KEY not found in .env file.")

    print("\n=== Census/BEA Ingestion ===\n")

    pop_df = fetch_state_population(year=2022)
    gdp_df = fetch_state_gdp()

    # Join population + GDP
    combined = pd.merge(pop_df, gdp_df, on="state_name", how="left")

    # Compute GDP per capita
    combined["gdp_per_capita_usd"] = (
        combined["gdp_billions_usd"] * 1e9 / combined["population"]
    )

    conn = sqlite3.connect(db_path)
    combined.to_sql("census_state_profile", conn, if_exists="replace", index=False)
    conn.close()

    print(f"\n✓ Saved {len(combined)} state profiles → census_state_profile")
    print(combined[["state_name", "population", "gdp_billions_usd", "region"]].head(10).to_string(index=False))


if __name__ == "__main__":
    run()