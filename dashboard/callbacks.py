"""
dashboard/callbacks.py
======================
All reactive callbacks for the Energy Grid Dashboard.
Every chart updates when the region dropdown changes.
"""

import os
import sys
import sqlite3
import pickle
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import xgboost as xgb
from dash import Input, Output, html
import dash_bootstrap_components as dbc

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH    = os.path.join(BASE_DIR, "energy_grid.db")
SAVE_DIR   = os.path.join(BASE_DIR, "models", "saved")

DARK_TEMPLATE = "plotly_dark"
DARK_BG       = "#020b14"
CARD_BG       = "#041624"

FUEL_COLORS = {
    "gen_coal":    "#888780",
    "gen_gas":     "#ffb800",
    "gen_nuclear": "#7F77DD",
    "gen_hydro":   "#378ADD",
    "gen_solar":   "#EF9F27",
    "gen_wind":    "#639922",
    "gen_other":   "#4a7a96",
}

FUEL_LABELS = {
    "gen_coal":    "Coal",
    "gen_gas":     "Natural Gas",
    "gen_nuclear": "Nuclear",
    "gen_hydro":   "Hydro",
    "gen_solar":   "Solar",
    "gen_wind":    "Wind",
    "gen_other":   "Other",
}

# ── Load assets once at startup ─────────────────────────────

def _load_features():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM features", conn)
    conn.close()
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df

def _load_xgb():
    path = os.path.join(SAVE_DIR, "xgboost_stress.json")
    if not os.path.exists(path):
        return None
    m = xgb.XGBRegressor()
    m.load_model(path)
    return m

def _load_clusters():
    path = os.path.join(SAVE_DIR, "cluster_profiles.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)

def _load_anomalies():
    path = os.path.join(SAVE_DIR, "anomaly_alerts.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df

try:
    DF       = _load_features()
    XGB_MDL  = _load_xgb()
    CLUSTERS = _load_clusters()
    ANOMALIES = _load_anomalies()
    print("✓ All assets loaded successfully")
except Exception as e:
    print(f"⚠ Asset load error: {e}")
    DF = pd.DataFrame()
    XGB_MDL = None
    CLUSTERS = None
    ANOMALIES = None


# ── Helper ──────────────────────────────────────────────────

def region_df(region: str) -> pd.DataFrame:
    return DF[DF["region"] == region].sort_values("datetime")

def dark_fig(fig):
    fig.update_layout(
        template=DARK_TEMPLATE,
        paper_bgcolor=CARD_BG,
        plot_bgcolor=CARD_BG,
        margin=dict(l=10, r=10, t=30, b=10),
        font=dict(color="#c8dde8", size=11),
    )
    return fig


# ── Register all callbacks ───────────────────────────────────

def register_callbacks(app):

    # ── 1. Choropleth Map ────────────────────────────────────
    @app.callback(
        Output("choropleth-map", "figure"),
        Input("region-select", "value"),
    )
    def update_choropleth(_):
        if DF.empty:
            return go.Figure()

        region_avg = DF.groupby("region").agg(
            grid_stress=("grid_stress_score", "mean"),
            renewable_pct=("renewable_pct", "mean"),
        ).reset_index()

        # Map EIA region codes to representative US states for choropleth
        region_to_states = {
            "CAL": ["CA"], "TEX": ["TX"], "NY": ["NY"],
            "FLA": ["FL"], "MIDA": ["PA", "MD", "VA", "DE", "DC"],
            "MIDW": ["IL", "IN", "OH", "MI", "WI"],
            "NE": ["MA", "CT", "RI", "VT", "NH", "ME"],
            "NW": ["WA", "OR", "ID"], "SE": ["GA", "AL", "MS"],
            "SW": ["AZ", "NV", "NM"], "CAR": ["NC", "SC"],
            "TEN": ["TN"],
        }

        rows = []
        for _, row in region_avg.iterrows():
            states = region_to_states.get(row["region"], [])
            for s in states:
                rows.append({
                    "state": s,
                    "region": row["region"],
                    "grid_stress": round(row["grid_stress"], 4),
                    "renewable_pct": round(row["renewable_pct"] * 100, 1),
                })

        map_df = pd.DataFrame(rows)

        fig = px.choropleth(
            map_df,
            locations="state",
            locationmode="USA-states",
            color="grid_stress",
            hover_name="region",
            hover_data={"state": False, "renewable_pct": True, "grid_stress": True},
            color_continuous_scale=["#0d3a5c", "#00e5ff", "#ffb800", "#ff4d4d"],
            scope="usa",
            labels={"grid_stress": "Grid Stress", "renewable_pct": "Renewable %"},
        )
        fig.update_layout(
            template=DARK_TEMPLATE,
            paper_bgcolor=CARD_BG,
            geo=dict(bgcolor=CARD_BG, lakecolor=CARD_BG, landcolor="#071e30",
                     subunitcolor="#0d3a5c"),
            margin=dict(l=0, r=0, t=0, b=0),
            coloraxis_colorbar=dict(title="Stress", thickness=12),
        )
        return fig

    # ── 2. Generation Mix Donut ──────────────────────────────
    @app.callback(
        Output("gen-mix-donut", "figure"),
        Input("region-select", "value"),
    )
    def update_donut(region):
        if DF.empty:
            return go.Figure()

        rdf = region_df(region)
        fuel_cols = [c for c in FUEL_LABELS if c in rdf.columns]
        totals = {FUEL_LABELS[c]: rdf[c].mean() for c in fuel_cols if rdf[c].mean() > 0}
        colors = [FUEL_COLORS[c] for c in fuel_cols if rdf[c].mean() > 0]

        fig = go.Figure(go.Pie(
            labels=list(totals.keys()),
            values=list(totals.values()),
            hole=0.55,
            marker=dict(colors=colors, line=dict(color=CARD_BG, width=2)),
            textinfo="label+percent",
            textfont=dict(size=11),
        ))
        fig.update_layout(
            template=DARK_TEMPLATE,
            paper_bgcolor=CARD_BG,
            showlegend=False,
            margin=dict(l=10, r=10, t=10, b=10),
            annotations=[dict(text=region, x=0.5, y=0.5, font_size=18,
                              font_color="#00e5ff", showarrow=False)],
        )
        return fig

    # ── 3. Demand Forecast Chart ─────────────────────────────
    @app.callback(
        Output("demand-forecast", "figure"),
        Input("region-select", "value"),
    )
    def update_forecast(region):
        if DF.empty or XGB_MDL is None:
            return go.Figure()

        rdf = region_df(region).tail(500).copy()

        FEATURE_COLS = [
            "renewable_pct", "fossil_pct", "fossil_co2_intensity",
            "heat_index", "demand_lag_24h", "demand_lag_168h",
            "rolling_7d_avg", "per_capita_demand", "peak_hour_flag",
            "hour_sin", "hour_cos", "month_sin", "month_cos",
        ]
        feat_cols = [c for c in FEATURE_COLS if c in rdf.columns]
        X = rdf[feat_cols].fillna(0)
        rdf["predicted"] = XGB_MDL.predict(X)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=rdf["datetime"], y=rdf["demand_mwh"],
            name="Actual", line=dict(color="#00e5ff", width=1.5),
        ))
        fig.add_trace(go.Scatter(
            x=rdf["datetime"], y=rdf["predicted"],
            name="XGBoost Forecast", line=dict(color="#b8ff57", width=1.5, dash="dash"),
        ))
        fig.update_layout(
            template=DARK_TEMPLATE,
            paper_bgcolor=CARD_BG, plot_bgcolor=CARD_BG,
            margin=dict(l=40, r=10, t=10, b=30),
            legend=dict(orientation="h", y=1.1),
            xaxis_title="", yaxis_title="Demand (MWh)",
            font=dict(color="#c8dde8", size=11),
        )
        return fig

    # ── 4. Grid Stress Heatmap ───────────────────────────────
    @app.callback(
        Output("stress-heatmap", "figure"),
        Input("region-select", "value"),
    )
    def update_heatmap(region):
        if DF.empty:
            return go.Figure()

        rdf = region_df(region).copy()
        rdf["day_name"] = rdf["datetime"].dt.day_name()

        day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        pivot = rdf.pivot_table(
            index="day_of_week", columns="hour",
            values="grid_stress_score", aggfunc="mean"
        )
        pivot.index = [day_order[i] if i < len(day_order) else i for i in pivot.index]

        fig = go.Figure(go.Heatmap(
            z=pivot.values,
            x=[f"{h:02d}:00" for h in pivot.columns],
            y=list(pivot.index),
            colorscale=[[0, "#0d3a5c"], [0.5, "#00e5ff"], [1, "#ff4d4d"]],
            colorbar=dict(title="Stress", thickness=10),
        ))
        dark_fig(fig)
        fig.update_layout(
            xaxis_title="Hour of Day",
            yaxis_title="",
            margin=dict(l=80, r=10, t=10, b=40),
        )
        return fig

    # ── 5. Cluster Scatter ───────────────────────────────────
    @app.callback(
        Output("cluster-scatter", "figure"),
        Input("region-select", "value"),
    )
    def update_clusters(selected_region):
        if CLUSTERS is None:
            return go.Figure()

        color_map = {
            "Renewable Leader": "#b8ff57",
            "Balanced Grid":    "#00e5ff",
            "Transitioning":    "#ffb800",
            "Fossil Dependent": "#ff4d4d",
            "High Demand":      "#7F77DD",
        }

        fig = go.Figure()
        for archetype in CLUSTERS["archetype"].unique():
            sub = CLUSTERS[CLUSTERS["archetype"] == archetype]
            fig.add_trace(go.Scatter(
                x=sub["renewable_pct"],
                y=sub["per_capita_demand"],
                mode="markers+text",
                name=archetype,
                text=sub["region"],
                textposition="top center",
                marker=dict(
                    size=16,
                    color=color_map.get(archetype, "#888"),
                    line=dict(width=2, color="#020b14"),
                    symbol=["star" if r == selected_region else "circle"
                            for r in sub["region"]],
                ),
                textfont=dict(size=10),
            ))

        dark_fig(fig)
        fig.update_layout(
            xaxis_title="Avg Renewable %",
            yaxis_title="Per Capita Demand",
            legend=dict(orientation="h", y=-0.25, font=dict(size=10)),
            margin=dict(l=50, r=10, t=10, b=80),
        )
        return fig

    # ── 6. Scenario Simulator ────────────────────────────────
    @app.callback(
        Output("scenario-output", "figure"),
        Input("region-select", "value"),
        Input("solar-slider",   "value"),
        Input("wind-slider",    "value"),
    )
    def update_scenario(region, solar_scale, wind_scale):
        if DF.empty or XGB_MDL is None:
            return go.Figure()

        FEATURE_COLS = [
            "renewable_pct", "fossil_pct", "fossil_co2_intensity",
            "heat_index", "demand_lag_24h", "demand_lag_168h",
            "rolling_7d_avg", "per_capita_demand", "peak_hour_flag",
            "hour_sin", "hour_cos", "month_sin", "month_cos",
        ]

        rdf = region_df(region).tail(168).copy()
        feat_cols = [c for c in FEATURE_COLS if c in rdf.columns]
        X_base = rdf[feat_cols].fillna(0).copy()

        # Baseline prediction
        baseline = XGB_MDL.predict(X_base)

        # Scenario: scale renewable % based on sliders
        X_scenario = X_base.copy()
        solar_factor = solar_scale / 100
        wind_factor  = wind_scale  / 100
        avg_factor   = (solar_factor + wind_factor) / 2

        if "renewable_pct" in X_scenario.columns:
            X_scenario["renewable_pct"] = (X_scenario["renewable_pct"] * avg_factor).clip(0, 1)
        if "fossil_pct" in X_scenario.columns:
            X_scenario["fossil_pct"] = (1 - X_scenario["renewable_pct"]).clip(0, 1)
        if "fossil_co2_intensity" in X_scenario.columns:
            X_scenario["fossil_co2_intensity"] *= (1 / avg_factor if avg_factor > 0 else 1)

        scenario = XGB_MDL.predict(X_scenario)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=baseline, name="Baseline",
            line=dict(color="#00e5ff", width=2),
        ))
        fig.add_trace(go.Scatter(
            y=scenario,
            name=f"Solar×{solar_scale/100:.1f} Wind×{wind_scale/100:.1f}",
            line=dict(color="#b8ff57", width=2, dash="dash"),
            fill="tonexty", fillcolor="rgba(184,255,87,0.05)",
        ))

        dark_fig(fig)
        fig.update_layout(
            xaxis_title="Hours (last 168h)",
            yaxis_title="Predicted Demand (MWh)",
            legend=dict(orientation="h", y=1.1),
            margin=dict(l=50, r=10, t=10, b=40),
        )
        return fig

    # ── 7. Anomaly Alert Table ───────────────────────────────
    @app.callback(
        Output("anomaly-table", "children"),
        Input("region-select", "value"),
    )
    def update_anomalies(region):
        if ANOMALIES is None or ANOMALIES.empty:
            return html.P("No anomaly data found. Run models/anomaly_detect.py first.",
                          className="text-muted")

        # Show top 10 anomalies for selected region, or all regions
        reg_ano = ANOMALIES[ANOMALIES["region"] == region] if region in ANOMALIES["region"].values else ANOMALIES
        top = reg_ano.head(10)

        if top.empty:
            return html.P(f"No anomalies detected for {region}.", className="text-muted")

        rows = []
        for _, row in top.iterrows():
            atype  = row.get("anomaly_type", "Unknown")
            color  = "#ff4d4d" if "Spike" in str(atype) else "#ffb800"
            rows.append(html.Tr([
                html.Td(str(row["datetime"])[:16],          style={"fontSize": "12px", "padding": "6px 10px"}),
                html.Td(row.get("region", ""),              style={"fontSize": "12px", "padding": "6px 10px"}),
                html.Td(html.Span(atype, style={"color": color, "fontWeight": "600", "fontSize": "12px"}),
                        style={"padding": "6px 10px"}),
                html.Td(f"{row.get('demand_mwh', 0):,.0f} MWh",
                        style={"fontSize": "12px", "padding": "6px 10px"}),
                html.Td(f"{row.get('grid_stress_score', 0):.3f}",
                        style={"fontSize": "12px", "padding": "6px 10px"}),
            ]))

        return dbc.Table(
            [
                html.Thead(html.Tr([
                    html.Th("Time",         style={"fontSize": "11px", "color": "#4a7a96"}),
                    html.Th("Region",       style={"fontSize": "11px", "color": "#4a7a96"}),
                    html.Th("Type",         style={"fontSize": "11px", "color": "#4a7a96"}),
                    html.Th("Demand",       style={"fontSize": "11px", "color": "#4a7a96"}),
                    html.Th("Stress Score", style={"fontSize": "11px", "color": "#4a7a96"}),
                ])),
                html.Tbody(rows),
            ],
            bordered=False, hover=True, responsive=True,
            style={"background": CARD_BG, "color": "#c8dde8"},
        )