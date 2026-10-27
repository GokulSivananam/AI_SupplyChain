"""
app.py
------
AI Supply Chain Disruption Prediction — Forecasting Dashboard
Shows future weekly disruption risk predictions based on historical data.
"""

import os
import sys
import io
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ── Path resolution ──────────────────────────────────────────────────────────
SRC_DIR  = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR)
if not os.path.isdir(os.path.join(BASE_DIR, "data")):
    BASE_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, SRC_DIR)

from data_preprocessing import load_data, add_or_validate_target, TARGET_COLUMN
from forecast import (
    build_weekly_series, forecast_disruption_rate,
    plot_forecast, plot_forecast_table, plot_risk_calendar,
    plot_category_forecast, forecast_summary,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Supply Chain Disruption Forecast",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #1e1e2e;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    .risk-high   { color: #e74c3c; font-weight: bold; font-size: 1.4rem; }
    .risk-medium { color: #f39c12; font-weight: bold; font-size: 1.4rem; }
    .risk-low    { color: #2ecc71; font-weight: bold; font-size: 1.4rem; }
    .section-header { font-size: 1.2rem; font-weight: 600; margin-top: 1rem; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("📈 Forecast Controls")
st.sidebar.markdown("---")

forecast_weeks = st.sidebar.slider(
    "Weeks to forecast ahead", min_value=4, max_value=26, value=12, step=2
)
forecast_method = st.sidebar.selectbox(
    "Forecast model", ["Prophet", "Holt-Winters"],
    help="Prophet captures trend + seasonality. Holt-Winters is a simpler fallback."
)
method_arg = "prophet" if forecast_method == "Prophet" else "holtwinters"

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Risk Thresholds**
- 🟢 Low  : < 35 %
- 🟡 Medium: 35–65 %
- 🔴 High  : > 65 %
""")
st.sidebar.markdown("---")
st.sidebar.caption(
    "Data: Global Supply Chain Risk & Logistics (2024–2026)\n\n"
    "⚠️ Forecasts are estimates. Always verify with domain experts."
)

# ── Load data ─────────────────────────────────────────────────────────────────
CSV_PATH = os.path.join(BASE_DIR, "data", "supply_chain_data.csv")

@st.cache_data(show_spinner=False)
def load_and_prepare(csv_path: str):
    df = load_data(csv_path)
    df = add_or_validate_target(df)
    return df

@st.cache_data(show_spinner=False)
def run_forecast(weeks: int, method: str):
    df      = load_and_prepare(CSV_PATH)
    weekly  = build_weekly_series(df)
    fc, mod = forecast_disruption_rate(weekly, periods=weeks, method=method)
    summ    = forecast_summary(fc)
    return df, weekly, fc, mod, summ

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🚚 AI Supply Chain Disruption Forecast")
st.markdown(
    "Predicting weekly disruption risk for upcoming shipment cycles "
    "using historical supply chain patterns."
)
st.markdown("---")

# ── Run forecast ──────────────────────────────────────────────────────────────
with st.spinner(f"Running {forecast_method} forecast for {forecast_weeks} weeks…"):
    try:
        df, weekly, forecast_df, used_method, summary = run_forecast(
            forecast_weeks, method_arg
        )
    except Exception as e:
        st.error(f"Forecast failed: {e}")
        st.stop()

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — KPI Summary Cards
# ═════════════════════════════════════════════════════════════════════════════
st.subheader(f"📊 Next {forecast_weeks} Weeks at a Glance")

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric(
    label="Avg Disruption Rate",
    value=f"{summary['avg_rate']*100:.1f}%",
    help="Average predicted disruption rate across forecast window"
)
c2.metric(
    label="Peak Risk Week",
    value=f"{summary['peak_rate']*100:.1f}%",
    delta=summary["peak_week"],
    delta_color="inverse",
    help="The week with the highest predicted disruption risk"
)
c3.metric(
    label="🟢 Low-Risk Weeks",
    value=summary["low_weeks"],
)
c4.metric(
    label="🟡 Medium-Risk Weeks",
    value=summary["medium_weeks"],
)
c5.metric(
    label="🔴 High-Risk Weeks",
    value=summary["high_weeks"],
    delta="Action needed" if summary["high_weeks"] > 0 else "None",
    delta_color="inverse" if summary["high_weeks"] > 0 else "off",
)

st.markdown("---")

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Main Forecast Chart
# ═════════════════════════════════════════════════════════════════════════════
st.subheader("📉 Historical Trend + Future Forecast")

fig_main = plot_forecast(weekly, forecast_df, used_method)
st.plotly_chart(fig_main, use_container_width=True)

st.caption(
    f"Model: **{used_method}**  |  "
    f"Trained on **{len(weekly)} weeks** of historical data (Jan 2024 – Oct 2026)  |  "
    "Shaded band = 80% confidence interval"
)

st.markdown("---")

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Risk Calendar
# ═════════════════════════════════════════════════════════════════════════════
st.subheader("📅 Week-by-Week Risk Calendar")

fig_cal = plot_risk_calendar(forecast_df)
st.plotly_chart(fig_cal, use_container_width=True)

st.markdown("---")

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Forecast Table + Download
# ═════════════════════════════════════════════════════════════════════════════
st.subheader("🗓️ Detailed Forecast Table")

table_df = plot_forecast_table(forecast_df)

# Colour-code Risk_Level column
def colour_risk(val):
    if "High"   in str(val): return "color: #e74c3c; font-weight: bold"
    if "Medium" in str(val): return "color: #f39c12; font-weight: bold"
    if "Low"    in str(val): return "color: #2ecc71; font-weight: bold"
    return ""

st.dataframe(
    table_df.style.map(colour_risk, subset=["Risk_Level"]),
    use_container_width=True,
    hide_index=True,
)

csv_buf = io.StringIO()
table_df.to_csv(csv_buf, index=False)
st.download_button(
    label="⬇️ Download Forecast as CSV",
    data=csv_buf.getvalue(),
    file_name="disruption_forecast.csv",
    mime="text/csv",
)

st.markdown("---")

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Category Breakdown
# ═════════════════════════════════════════════════════════════════════════════
st.subheader("🔍 Historical Disruption Rate by Category")

tab1, tab2, tab3 = st.tabs([
    "🚛 By Transport Mode",
    "🏭 By Supplier",
    "📍 By Location",
])

with tab1:
    fig = plot_category_forecast(df, "Transportation modes")
    if fig:
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Each line shows the weekly disruption rate for one transport mode over time.")
    else:
        st.info("Column 'Transportation modes' not found in dataset.")

with tab2:
    fig = plot_category_forecast(df, "Supplier name")
    if fig:
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Each line shows the weekly disruption rate for one supplier over time.")
    else:
        st.info("Column 'Supplier name' not found in dataset.")

with tab3:
    fig = plot_category_forecast(df, "Location")
    if fig:
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Each line shows the weekly disruption rate by origin location over time.")
    else:
        st.info("Column 'Location' not found in dataset.")

st.markdown("---")

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Date Lookup
# ═════════════════════════════════════════════════════════════════════════════
st.subheader("🔎 Look Up a Specific Future Date")

col_date, col_result = st.columns([1, 2])

with col_date:
    lookup_date = st.date_input(
        "Select a date",
        value=pd.Timestamp.today().date() + pd.Timedelta(days=7),
        min_value=pd.Timestamp.today().date(),
        max_value=(pd.Timestamp.today() + pd.Timedelta(weeks=forecast_weeks)).date(),
    )
    lookup_btn = st.button("🔍 Get Forecast", type="primary")

if lookup_btn:
    fore_only = forecast_df[forecast_df["is_forecast"]].copy()
    if fore_only.empty:
        st.warning("No forecast data in range.")
    else:
        fore_only["diff"] = (fore_only["ds"] - pd.Timestamp(lookup_date)).abs()
        nearest   = fore_only.loc[fore_only["diff"].idxmin()]
        prob      = float(nearest["yhat"])
        low_b     = float(nearest["yhat_lower"])
        high_b    = float(nearest["yhat_upper"])
        week_str  = nearest["ds"].strftime("%d %b %Y")

        if prob < 0.35:
            badge = "🟢 LOW RISK"
            colour = "green"
            advice = "Standard monitoring applies. No immediate action required."
        elif prob < 0.65:
            badge = "🟡 MEDIUM RISK"
            colour = "orange"
            advice = "Increase monitoring frequency. Confirm carrier schedules in advance."
        else:
            badge = "🔴 HIGH RISK"
            colour = "red"
            advice = "Activate contingency plans. Consider backup suppliers and expedited routes."

        with col_result:
            st.markdown(f"### Week of **{week_str}**")
            r1, r2, r3 = st.columns(3)
            r1.metric("Predicted Rate",  f"{prob*100:.1f}%")
            r2.metric("Lower Bound (80%)", f"{low_b*100:.1f}%")
            r3.metric("Upper Bound (80%)", f"{high_b*100:.1f}%")
            st.markdown(f"**Risk Level:** :{colour}[{badge}]")
            st.info(f"💡 **Recommended Action:** {advice}")

st.markdown("---")

# ═════════════════════════════════════════════════════════════════════════════
# SECTION 7 — About
# ═════════════════════════════════════════════════════════════════════════════
with st.expander("ℹ️ How this forecast works"):
    st.markdown(f"""
    **Data preparation**
    - The dataset contains 100 shipment records with no date column.
    - Dates are assigned spanning **Jan 2024 → Oct 2026** (one per shipment in order).
    - Each row is labelled `Disruption_Risk = 1` when any of these signals is present:
      - Inspection result = Fail
      - Defect rate > 2.5 %
      - Lead time > 20 days

    **Forecasting model: {used_method}**
    - Shipment records are aggregated into a **weekly disruption rate** time series.
    - {used_method} learns the historical trend and projects it forward.
    - The shaded band represents an **80% confidence interval** — actual values
      should fall within this range 80% of the time.

    **Limitations**
    - The dataset is small (100 rows / ~100 weeks). More data improves accuracy.
    - Dates are assigned synthetically — real timestamps would improve reliability.
    - This is a decision-support tool. Always verify with domain experts.
    """)
