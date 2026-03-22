import sys
import sqlite3
import pandas as pd

sys.path.append(".")

from fetch_eia    import run as run_eia
from fetch_noaa   import run as run_noaa
from fetch_census import run as run_census

DB_PATH = "../energy_grid.db"

START = "2024-01-01"
END   = "2024-12-31"

def verify_db():
    conn = sqlite3.connect(DB_PATH)
    tables = ["eia_demand", "eia_generation", "noaa_weather", "census_state_profile"]
    print("\n=== Database Summary ===")
    for table in tables:
        try:
            count = pd.read_sql(f"SELECT COUNT(*) as n FROM {table}", conn).iloc[0]["n"]
            print(f"  {table:<28} {count:>10,} rows")
        except Exception:
            print(f"  {table:<28}   (not found)")
    conn.close()

if __name__ == "__main__":
    print(f"Starting full ingestion: {START} → {END}\n")
    errors = []

    try:
        run_eia(start=START, end=END)
    except Exception as e:
        print(f"\n⚠ EIA failed: {e}")
        errors.append("EIA")

    try:
        run_noaa(start=START, end=END)
    except Exception as e:
        print(f"\n⚠ NOAA failed: {e}")
        errors.append("NOAA")

    try:
        run_census()
    except Exception as e:
        print(f"\n⚠ Census failed: {e}")
        errors.append("Census")

    verify_db()

    if errors:
        print(f"\n⚠  Errors in: {', '.join(errors)}")
    else:
        print("\n✓ All sources ingested.")