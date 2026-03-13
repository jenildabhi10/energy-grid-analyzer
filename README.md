# ⚡ Energy Grid Stress & Renewable Analyzer

> End-to-end ML portfolio project: 4 live government APIs → SQLite → XGBoost + LSTM + Clustering + Anomaly Detection → Plotly Dash dashboard

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🗺 Project Structure

```
energy-grid-analyzer/
├── data/
│   ├── fetch_eia.py          # EIA electricity API
│   ├── fetch_noaa.py         # NOAA weather API
│   ├── fetch_epa.py          # EPA air quality API
│   ├── fetch_census.py       # Census population/GDP
│   └── run_ingestion.py      # Master runner
├── models/                   # Week 3–4 (coming soon)
├── dashboard/                # Week 5 (coming soon)
├── notebooks/                # EDA + model training
├── .env.example              # Copy to .env, add your keys
├── .gitignore
└── requirements.txt
```

---

## 🚀 Windows Setup (One-Time)

### 1. Install Python 3.11
Download from https://www.python.org/downloads/  
During install, check **"Add Python to PATH"** ✓

Verify in a new terminal:
```
python --version   # should print Python 3.11.x
```

### 2. Install Git
Download from https://git-scm.com/download/win  
Use all default options during install.

### 3. Clone & set up the project
Open **Command Prompt** or **Windows Terminal** and run:

```cmd
git clone https://github.com/jenildabhi10/energy-grid-analyzer.git
cd energy-grid-analyzer

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
```

Your terminal prompt will show `(venv)` when the virtual environment is active.  
**Always activate it before working on the project.**

### 4. Get your API keys (all free, all instant)

| API | Where to register | Time |
|-----|------------------|------|
| EIA | https://www.eia.gov/opendata/register.php | ~30 sec |
| NOAA | https://www.ncdc.noaa.gov/cdo-web/token | ~1 min (email) |
| EPA | https://aqs.epa.gov/aqsweb/documents/data_api.html#signup | ~1 min (email) |
| Census | https://api.census.gov/data/key_signup.html | ~1 min (email) |

### 5. Add keys to .env

```cmd
copy .env.example .env
```

Open `.env` in VS Code and fill in your keys:
```
EIA_API_KEY=abc123...
NOAA_TOKEN=xyz456...
EPA_EMAIL=you@email.com
CENSUS_API_KEY=def789...
EPA_API_KEY=gdfb113dc1d
```

### 6. Run your first data fetch (test with 1 month)

```cmd
cd data
python fetch_eia.py
```

If you see rows printing → it's working. Then run all sources:

```cmd
python run_ingestion.py
```

Check `energy_grid.db` appeared in your project folder. Open it with:
- **DB Browser for SQLite** (free): https://sqlitebrowser.org/

---

## 📅 6-Week Build Plan

| Week | Goal | Status |
|------|------|--------|
| 1 | Data ingestion — 4 APIs → SQLite | ✅ |
| 2 | ETL pipeline — 15 engineered features | ⏳ |
| 3 | XGBoost + SHAP forecaster | ⏳ |
| 4 | PyTorch LSTM + K-Means + Isolation Forest | ⏳ |
| 5 | Plotly Dash dashboard (8 visualizations) | ⏳ |
| 6 | Deploy to Hugging Face Spaces + README polish | ⏳ |

---

## 🔑 Tech Stack

**Data:** Python · Requests · Pandas · SQLite  
**ML:** XGBoost · PyTorch · Scikit-learn · SHAP  
**Dashboard:** Plotly Dash · Dash Bootstrap Components  
**Deploy:** Hugging Face Spaces · Docker  

---

## 📊 Resume Impact

- Fused **4 live government APIs** (not Kaggle CSVs)
- Engineered **15 domain features** into a production SQLite feature store
- Trained **4 ML paradigms**: regression, deep learning, clustering, anomaly detection
- Deployed **interactive dashboard** with live scenario simulator
- Total cost: **$0**