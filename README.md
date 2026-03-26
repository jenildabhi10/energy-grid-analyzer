# ⚡ Energy Grid Stress & Renewable Analyzer

> End-to-end ML system: 3 live government APIs → SQLite → XGBoost + LSTM + K-Means + Isolation Forest → Plotly Dash dashboard

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🔍 What This Project Does

The U.S. power grid faces growing stress as renewable energy scales but demand spikes from heatwaves remain hard to predict. This system ingests live government data, engineers domain-specific features, and runs 4 ML models to forecast grid stress, classify regions, and detect anomalies — all visualized in an interactive dashboard.

**Key results:**
- XGBoost forecaster: **R² = 0.91** on held-out data
- LSTM demand forecast: **MAPE = 3.68%** over a 48-hour horizon
- **13,714 real anomaly events** detected in 2024 U.S. grid data
- U.S. regions clustered into 3 energy archetypes: Renewable Leaders, Balanced Grid, Transitioning

---

## 🗺 Project Structure

```
energy-grid-analyzer/
├── data/
│   ├── fetch_eia.py          # EIA electricity API ingestion
│   ├── fetch_noaa.py         # NOAA weather API ingestion
│   ├── fetch_census.py       # Census Bureau population/GDP
│   ├── merge_pipeline.py     # ETL + 15 feature engineering
│   └── run_ingestion.py      # Master ingestion runner
├── models/
│   ├── xgboost_stress.py     # Grid stress forecaster + SHAP
│   ├── lstm_demand.py        # PyTorch LSTM time-series
│   ├── cluster_states.py     # K-Means region archetypes
│   ├── anomaly_detect.py     # Isolation Forest alerts
│   └── saved/                # Trained model files
├── dashboard/
│   ├── app.py                # Dash entry point
│   ├── layouts.py            # Page layout components
│   └── callbacks.py          # Reactive callbacks
├── notebooks/                # EDA + model training
├── .env.example              # Copy to .env, add your keys
├── .gitignore
└── requirements.txt
```

---

## 🚀 Setup

### 1. Install Python 3.11
Download from https://www.python.org/downloads/  
During install, check **"Add Python to PATH"** ✓

### 2. Install Git
Download from https://git-scm.com/download/win  
Use all default options.

### 3. Clone & set up the project

```cmd
git clone https://github.com/jenildabhi10/energy-grid-analyzer.git
cd energy-grid-analyzer

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
```

### 4. Get your API keys (all free)

| API | Register | Time |
|-----|----------|------|
| EIA | https://www.eia.gov/opendata/register.php | Instant |
| NOAA | https://www.ncdc.noaa.gov/cdo-web/token | ~1 min |
| Census | https://api.census.gov/data/key_signup.html | ~1 min |

### 5. Configure .env

```cmd
copy .env.example .env
```

Fill in your keys:
```
EIA_API_KEY=your_key_here
NOAA_TOKEN=your_token_here
CENSUS_API_KEY=your_key_here
```

### 6. Ingest data

```cmd
cd data
python run_ingestion.py
```

This pulls 2.5M+ hourly records into `energy_grid.db`. Browse it with [DB Browser for SQLite](https://sqlitebrowser.org/).

### 7. Run ETL pipeline

```cmd
python merge_pipeline.py
```

Joins all sources and engineers 15 ML-ready features into a `features` table.

### 8. Train models

```cmd
python models/xgboost_stress.py
python models/cluster_states.py
python models/anomaly_detect.py
python models/lstm_demand.py
```

### 9. Launch dashboard

```cmd
python dashboard/app.py
```

Open **http://localhost:8050** in your browser.

---

## 📊 Data Sources

| Source | Data | Records |
|--------|------|---------|
| [EIA Open Data](https://www.eia.gov/opendata/) | Hourly electricity demand + generation by fuel type | 2.5M+ |
| [NOAA CDO](https://www.ncdc.noaa.gov/cdo-web/) | Daily temperature, wind, precipitation | 9,156 |
| [U.S. Census Bureau](https://api.census.gov/) | State population + GDP | 52 states |

---

## 🤖 ML Models

| Model | Task | Result |
|-------|------|--------|
| XGBoost + SHAP | Grid stress forecasting | R² = 0.91 |
| PyTorch LSTM | 48h demand forecasting | MAPE = 3.68% |
| K-Means + PCA | Region archetype clustering | 3 archetypes |
| Isolation Forest | Anomaly detection | 13,714 events flagged |

---

## ⚙️ Engineered Features

15 domain-specific features including:
- `grid_stress_score` — demand / total generation
- `renewable_pct` — (wind + solar + hydro) / total generation
- `fossil_co2_intensity` — weighted CO₂ per MWh (kg/MWh)
- `demand_lag_24h` / `demand_lag_168h` — time-lag features
- `rolling_7d_avg` — smoothed demand baseline
- `per_capita_demand` — normalized by state population
- `hour_sin` / `hour_cos` — cyclical time encodings
- `peak_hour_flag` — morning & evening peak periods

---

## 📈 Dashboard Features

- 🗺️ **US Choropleth Map** — grid stress by region
- 📈 **Demand Forecast Chart** — XGBoost actual vs predicted
- 🍩 **Generation Mix Donut** — fuel breakdown per region
- 🔥 **Grid Stress Heatmap** — hour × day of week
- 🧩 **Cluster Explorer** — K-Means region archetypes
- 🔴 **Anomaly Alert Feed** — top grid events
- 🎛️ **Scenario Simulator** — model impact of +X% solar/wind live

---

## 🔑 Tech Stack

**Data:** Python · Requests · Pandas · SQLite  
**ML:** XGBoost · PyTorch · Scikit-learn · SHAP  
**Dashboard:** Plotly Dash · Dash Bootstrap Components