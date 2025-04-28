# ─────────────────────────────────────────────────────────────
# streamlit_app.py   —  Alberta Load Forecast Dashboard
# Samira · 2025
# ─────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
ROOT       = Path(__file__).resolve().parents[1]
CLEAN_DIR  = ROOT / "data" / "clean"
LOAD_PQ    = CLEAN_DIR / "load_long.parquet"
PRED_PQ    = CLEAN_DIR / "preds.parquet"

# ------------------------------------------------------------
# Load parquet files (cached)
# ------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_all():
    load_long = pd.read_parquet(LOAD_PQ)
    preds     = pd.read_parquet(PRED_PQ)
    for d in (load_long, preds):
        d["timestamp"] = pd.to_datetime(d["timestamp"])
    return load_long, preds

load_long, preds = load_all()

# ── **keep only regions that have predictions** ─────────────
regions = sorted(preds["region"].unique())

# ------------------------------------------------------------
# Sidebar controls
# ------------------------------------------------------------
st.sidebar.title("⚙️ Controls")

default_idx = regions.index("Calgary") if "Calgary" in regions else 0
region = st.sidebar.selectbox("Region", regions, index=default_idx)

max_hours = int((load_long.timestamp.max() - load_long.timestamp.min())
                .total_seconds() // 3600)
hours_back = st.sidebar.slider("Last N hours", 24, max_hours, 240, step=24)

theme  = st.sidebar.radio("Theme", ["Light", "Dark"], horizontal=True)
px.defaults.template = "plotly_dark" if theme == "Dark" else "plotly_white"

# ------------------------------------------------------------
# Helper: window filter
# ------------------------------------------------------------
def last_n(df, n):
    end = df.timestamp.max()
    start = end - pd.Timedelta(hours=n)
    return df[df.timestamp.between(start, end)]

ld = last_n(load_long[load_long.region == region], hours_back)
pr = last_n(preds[preds.region == region], hours_back)

# ------------------------------------------------------------
# KPI strip
# ------------------------------------------------------------
today_peak = ld[ld.timestamp.dt.date == ld.timestamp.iloc[-1].date()].load_MW.max()
avg_7d     = ld[ld.timestamp >= ld.timestamp.max() - pd.Timedelta(days=7)].load_MW.mean()
anom_cnt   = int(pr.anomaly.sum())

k1, k2, k3, k4 = st.columns(4)
k1.metric("Latest",       f"{ld.load_MW.iloc[-1]:,.0f} MW")
k2.metric("Peak today",   f"{today_peak:,.0f}")
k3.metric("7-day avg",    f"{avg_7d:,.0f}")
k4.metric("Anomalies",    anom_cnt)
st.markdown("---")

# ------------------------------------------------------------
# Tabs
# ------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Trend", "📈 Forecast", "🚨 Anomaly", "🌡 Compare"])

# ---- Tab 1: Trend ------------------------------------------
with tab1:
    st.subheader(f"Load Trend — {region}")
    fig = go.Figure(go.Scatter(x=ld.timestamp, y=ld.load_MW, name="Load"))
    fig.update_layout(height=350, yaxis_title="MW")
    st.plotly_chart(fig, use_container_width=True)

# ---- Tab 2: Forecast ---------------------------------------
with tab2:
    st.subheader("P10 / P50 / P90 Forecast")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pr.timestamp, y=pr.p50, name="P50", line=dict(width=2)))
    fig.add_trace(go.Scatter(x=pr.timestamp, y=pr.p90, name="P90", line=dict(dash="dot")))
    fig.add_trace(go.Scatter(x=pr.timestamp, y=pr.p10, name="P10",
                             fill="tonexty", fillcolor="rgba(30,144,255,0.15)",
                             line=dict(dash="dot")))
    fig.add_trace(go.Scatter(x=ld.timestamp, y=ld.load_MW, name="Actual",
                             line=dict(color="black", width=1)))
    fig.update_layout(height=400, yaxis_title="MW")
    st.plotly_chart(fig, use_container_width=True)

# ---- Tab 3: Anomaly ----------------------------------------
with tab3:
    st.subheader("Anomaly Radar")
    an = pr[pr.anomaly]
    if an.empty:
        st.info("No anomalies detected in this window 🎉")
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=pr.timestamp, y=pr.p50, name="P50", line=dict(width=1)))
        fig.add_trace(go.Scatter(x=an.timestamp, y=an.p50, mode="markers",
                                 marker=dict(size=8, color="red"), name="Anomaly"))
        fig.update_layout(height=350, yaxis_title="MW")
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(an.iloc[::-1][["timestamp", "actual", "p50", "p10", "p90"]]
                     .style.format("{:.1f}"))

# ---- Tab 4: Compare ----------------------------------------
with tab4:
    st.subheader("Hour × Month Heat-map")

    base = (
        load_long
        .assign(month=load_long.timestamp.dt.month,
                hour =load_long.timestamp.dt.hour)
        .groupby(["region", "month", "hour"], as_index=False)
        .agg(mean_MW=("load_MW", "mean"))
    )

    region_df = (base[base.region == region]
                 .pivot(index="month", columns="hour", values="mean_MW")
                 .sort_index())

    fig_cmp = px.imshow(
        region_df,
        aspect="auto",
        labels=dict(x="Hour", y="Month", color="MW"),
        title=f"Average Load — {region} (Month × Hour)",
    )
    st.plotly_chart(fig_cmp, use_container_width=True)

# ------------------------------------------------------------
# Downloads
# ------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.download_button("⬇ Actual window CSV",
                           ld.to_csv(index=False).encode(),
                           file_name=f"{region}_load_window.csv")
st.sidebar.download_button("⬇ Forecast window CSV",
                           pr.to_csv(index=False).encode(),
                           file_name=f"{region}_forecast_window.csv")

st.sidebar.caption("© 2025 Samira · Demo Dashboard")
