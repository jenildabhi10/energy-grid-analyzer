import sqlite3, pandas as pd

conn = sqlite3.connect('energy_grid.db')
tables = ['eia_demand', 'eia_generation', 'noaa_weather', 'epa_air_quality', 'census_state_profile']

for t in tables:
    try:
        n = pd.read_sql(f'SELECT COUNT(*) as n FROM {t}', conn).iloc[0]['n']
        print(f'{t:<28} {n:>10,} rows')
    except:
        print(f'{t:<28}  NOT FOUND')

conn.close()