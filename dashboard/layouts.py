"""
dashboard/layouts.py
====================
All page layout components for the Energy Grid Dashboard.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sqlite3
import pandas as pd
from dash import dcc, html
import dash_bootstrap_components as dbc

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH  = os.path.join(BASE_DIR, "energy_grid.db")

REGIONS = ["CAL","CAR","FLA","MIDA","MIDW","NE","NW","NY","SE","SW","TEN","TEX"]

REGION_OPTIONS = [{"label": r, "value": r} for r in REGIONS]


def load_kpis() -> dict:
    """Load summary KPIs for the top cards."""
    try:
        conn = sqlite3.connect(DB_PATH)
        df = pd.read_sql("SELECT * FROM features", conn)
        conn.close()

        return {
            "avg_renewable": f"{df['renewable_pct'].mean()*100:.1f}%",
            "avg_stress":    f"{df['grid_stress_score'].mean():.3f}",
            "avg_co2":       f"{df['fossil_co2_intensity'].mean():.0f}",
            "total_demand":  f"{df['demand_mwh'].sum()/1e9:.2f}B",
        }
    except Exception:
        return {
            "avg_renewable": "N/A",
            "avg_stress":    "N/A",
            "avg_co2":       "N/A",
            "total_demand":  "N/A",
        }


def kpi_card(title: str, value: str, color: str = "#00e5ff") -> dbc.Card:
    return dbc.Card(
        dbc.CardBody([
            html.P(title, className="text-muted", style={"fontSize": "11px", "marginBottom": "4px", "letterSpacing": "1px", "textTransform": "uppercase"}),
            html.H4(value, style={"color": color, "fontWeight": "700", "marginBottom": "0"}),
        ]),
        style={"background": "#041624", "border": "1px solid #0d3a5c", "borderRadius": "4px"},
    )


def create_layout():
    kpis = load_kpis()

    return dbc.Container([

        # ── Header ──────────────────────────────────────────
        dbc.Row([
            dbc.Col([
                html.H2("⚡ Energy Grid Stress & Renewable Analyzer",
                        style={"color": "#00e5ff", "fontWeight": "700", "marginBottom": "4px"}),
                html.P("EIA · NOAA · Census  |  XGBoost · LSTM · K-Means · Isolation Forest",
                       className="text-muted", style={"fontSize": "13px"}),
            ], width=8),
            dbc.Col([
                html.Div([
                    dcc.Dropdown(
                        id="region-select",
                        options=REGION_OPTIONS,
                        value="CAL",
                        clearable=False,
                        style={"color": "#000", "fontSize": "13px"},
                    ),
                ], style={"paddingTop": "16px"}),
            ], width=4),
        ], className="mb-3 mt-3"),

        # ── KPI Cards ───────────────────────────────────────
        dbc.Row([
            dbc.Col(kpi_card("Avg Renewable %",   kpis["avg_renewable"], "#b8ff57"), width=3),
            dbc.Col(kpi_card("Avg Grid Stress",   kpis["avg_stress"],    "#00e5ff"), width=3),
            dbc.Col(kpi_card("Avg CO₂ Intensity", kpis["avg_co2"] + " kg/MWh", "#ffb800"), width=3),
            dbc.Col(kpi_card("Total Demand",      kpis["total_demand"] + " MWh", "#ff4d4d"), width=3),
        ], className="mb-4"),

        # ── Row 1: Choropleth + Generation Mix ──────────────
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("🗺️ Grid Stress by Region", style={"color": "#00e5ff", "fontSize": "13px"}),
                    dbc.CardBody([dcc.Graph(id="choropleth-map", style={"height": "350px"})]),
                ], style={"background": "#041624", "border": "1px solid #0d3a5c"}),
            ], width=7),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("🍩 Generation Mix", style={"color": "#00e5ff", "fontSize": "13px"}),
                    dbc.CardBody([dcc.Graph(id="gen-mix-donut", style={"height": "350px"})]),
                ], style={"background": "#041624", "border": "1px solid #0d3a5c"}),
            ], width=5),
        ], className="mb-4"),

        # ── Row 2: Demand Forecast ───────────────────────────
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("📈 Demand Forecast — XGBoost vs Actual",
                                   style={"color": "#00e5ff", "fontSize": "13px"}),
                    dbc.CardBody([dcc.Graph(id="demand-forecast", style={"height": "300px"})]),
                ], style={"background": "#041624", "border": "1px solid #0d3a5c"}),
            ], width=12),
        ], className="mb-4"),

        # ── Row 3: Stress Heatmap + Cluster Explorer ─────────
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("🔥 Grid Stress Heatmap (Hour × Day)",
                                   style={"color": "#00e5ff", "fontSize": "13px"}),
                    dbc.CardBody([dcc.Graph(id="stress-heatmap", style={"height": "300px"})]),
                ], style={"background": "#041624", "border": "1px solid #0d3a5c"}),
            ], width=6),
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("🧩 Region Cluster Explorer",
                                   style={"color": "#00e5ff", "fontSize": "13px"}),
                    dbc.CardBody([dcc.Graph(id="cluster-scatter", style={"height": "300px"})]),
                ], style={"background": "#041624", "border": "1px solid #0d3a5c"}),
            ], width=6),
        ], className="mb-4"),

        # ── Row 4: Scenario Simulator ────────────────────────
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("🎛️ Scenario Simulator — What if renewable capacity changes?",
                                   style={"color": "#b8ff57", "fontSize": "13px"}),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.Label("Solar capacity scale (%)", className="text-muted",
                                           style={"fontSize": "12px"}),
                                dcc.Slider(id="solar-slider", min=50, max=300, step=10, value=100,
                                           marks={50: "50%", 100: "Current", 200: "+100%", 300: "+200%"},
                                           tooltip={"placement": "bottom"}),
                            ], width=6),
                            dbc.Col([
                                html.Label("Wind capacity scale (%)", className="text-muted",
                                           style={"fontSize": "12px"}),
                                dcc.Slider(id="wind-slider", min=50, max=300, step=10, value=100,
                                           marks={50: "50%", 100: "Current", 200: "+100%", 300: "+200%"},
                                           tooltip={"placement": "bottom"}),
                            ], width=6),
                        ], className="mb-3"),
                        dcc.Graph(id="scenario-output", style={"height": "280px"}),
                    ]),
                ], style={"background": "#041624", "border": "1px solid #1a5c2a"}),
            ], width=12),
        ], className="mb-4"),

        # ── Row 5: Anomaly Alert Feed ────────────────────────
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("🔴 Anomaly Alert Feed — Top Grid Events",
                                   style={"color": "#ff4d4d", "fontSize": "13px"}),
                    dbc.CardBody([html.Div(id="anomaly-table")]),
                ], style={"background": "#041624", "border": "1px solid #5c0d0d"}),
            ], width=12),
        ], className="mb-4"),

        # ── Footer ──────────────────────────────────────────
        dbc.Row([
            dbc.Col([
                html.Hr(style={"borderColor": "#0d3a5c"}),
                html.P("Data: EIA Open Data · NOAA CDO · U.S. Census Bureau  |  Models: XGBoost · PyTorch LSTM · K-Means · Isolation Forest",
                       className="text-muted text-center", style={"fontSize": "11px"}),
            ])
        ]),

    ], fluid=True, style={"background": "#020b14", "minHeight": "100vh", "padding": "0 24px 40px"})