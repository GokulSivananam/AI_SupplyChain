"""
forecast.py
-----------
Time-series forecasting of supply chain disruption risk for upcoming days.

Strategy
--------
The raw dataset has no date column. We assign realistic shipment dates
spanning Jan 2024 → Oct 2026 (matching the dataset's stated coverage),
one shipment per row in order. This gives us ~100 weekly data points when
aggregated, which is sufficient for short-term forecasting.

Two forecasting approaches are used:
  1. Prophet  – Facebook's trend + seasonality model (primary)
  2. Holt-Winters Exponential Smoothing – fallback if Prophet fails

Both predict:
  - Daily / weekly disruption RATE (0–1)
  - Confidence interval (upper / lower band)

Output returned as a DataFrame ready for Plotly charts.
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# ── make sibling imports work ──────────────────────────────
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)
from data_preprocessing import (
    load_data, add_or_validate_target, TARGET_COLUMN,
    DEFECT_THRESHOLD, LEAD_TIME_THRESHOLD,
)


# ─────────────────────────────────────────────────────────────
# 1. Assign shipment dates to the dataset
# ─────────────────────────────────────────────────────────────
def assign_dates(df: pd.DataFrame,
                 start: str = "2024-01-01",
                 end:   str = "2026-10-10") -> pd.DataFrame:
    """
    Assign one shipment date per row, evenly spaced between start and end.
    Adds a 'Shipment_Date' column (datetime) and 'Week' column (week-start Monday).
    """
    df = df.copy()
    dates = pd.date_range(start=start, end=end, periods=len(df))
    df["Shipment_Date"] = dates
    df["Week"] = df["Shipment_Date"].dt.to_period("W").apply(lambda p: p.start_time)
    return df


# ─────────────────────────────────────────────────────────────
# 2. Build weekly time series
# ─────────────────────────────────────────────────────────────
def build_weekly_series(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate the labelled DataFrame into a weekly disruption rate series.

    Returns a DataFrame with columns:
        ds  – week start date (datetime, Prophet convention)
        y   – disruption rate for that week (0.0 – 1.0)
        n   – shipments in that week
    """
    df = assign_dates(df)
    weekly = (df.groupby("Week")[TARGET_COLUMN]
                .agg(disruptions="sum", n="count")
                .reset_index()
                .rename(columns={"Week": "ds"}))
    weekly["y"] = weekly["disruptions"] / weekly["n"]
    weekly = weekly.sort_values("ds").reset_index(drop=True)
    return weekly[["ds", "y", "n"]]


# ─────────────────────────────────────────────────────────────
# 3. Per-category weekly series (by transport mode / supplier)
# ─────────────────────────────────────────────────────────────
def build_category_series(df: pd.DataFrame,
                           col: str = "Transportation modes") -> pd.DataFrame:
    """
    Returns a wide DataFrame: ds | category_A | category_B | …
    Each column is the weekly disruption rate for that category.
    """
    df = assign_dates(df)
    if col not in df.columns:
        return pd.DataFrame()

    records = []
    for cat, grp in df.groupby(col):
        weekly = (grp.groupby("Week")[TARGET_COLUMN]
                     .agg(disruptions="sum", n="count")
                     .reset_index()
                     .rename(columns={"Week": "ds"}))
        weekly["rate"] = weekly["disruptions"] / weekly["n"]
        weekly = weekly.rename(columns={"rate": cat})
        records.append(weekly[["ds", cat]])

    if not records:
        return pd.DataFrame()

    from functools import reduce
    merged = reduce(lambda a, b: pd.merge(a, b, on="ds", how="outer"), records)
    merged = merged.sort_values("ds").fillna(0)
    return merged


# ─────────────────────────────────────────────────────────────
# 4. Prophet forecast
# ─────────────────────────────────────────────────────────────
def prophet_forecast(weekly: pd.DataFrame,
                     periods: int = 30,
                     freq: str = "W") -> pd.DataFrame:
    """
    Fit a Prophet model on the weekly disruption rate and forecast
    `periods` steps ahead.

    Returns a DataFrame with:
        ds, yhat, yhat_lower, yhat_upper, is_forecast (bool)
    """
    from prophet import Prophet

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        interval_width=0.80,
        changepoint_prior_scale=0.3,
    )
    model.fit(weekly[["ds", "y"]])

    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)

    # Clip probabilities to [0, 1]
    for col in ["yhat", "yhat_lower", "yhat_upper"]:
        forecast[col] = forecast[col].clip(0, 1)

    forecast["is_forecast"] = forecast["ds"] > weekly["ds"].max()
    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper", "is_forecast"]]


# ─────────────────────────────────────────────────────────────
# 5. Holt-Winters fallback forecast
# ─────────────────────────────────────────────────────────────
def holtwinters_forecast(weekly: pd.DataFrame,
                         periods: int = 30) -> pd.DataFrame:
    """
    Holt-Winters Exponential Smoothing forecast — fallback when Prophet
    is unavailable or fails.
    """
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    series = weekly["y"].values
    n = len(series)

    # Need at least 2 * seasonal period for additive; fall back to simple if too short
    try:
        model = ExponentialSmoothing(
            series, trend="add", seasonal=None, initialization_method="estimated"
        ).fit(optimized=True)
    except Exception:
        model = ExponentialSmoothing(
            series, trend=None, seasonal=None
        ).fit(optimized=True)

    preds = model.forecast(periods)
    preds = np.clip(preds, 0, 1)

    last_date = weekly["ds"].max()
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(weeks=1),
        periods=periods, freq="W"
    )

    hist_df = weekly[["ds", "y"]].rename(columns={"y": "yhat"})
    hist_df["yhat_lower"] = hist_df["yhat"]
    hist_df["yhat_upper"] = hist_df["yhat"]
    hist_df["is_forecast"] = False

    fore_df = pd.DataFrame({
        "ds"          : future_dates,
        "yhat"        : preds,
        "yhat_lower"  : np.clip(preds - 0.10, 0, 1),
        "yhat_upper"  : np.clip(preds + 0.10, 0, 1),
        "is_forecast" : True,
    })

    return pd.concat([hist_df, fore_df], ignore_index=True)


# ─────────────────────────────────────────────────────────────
# 6. Smart forecast dispatcher
# ─────────────────────────────────────────────────────────────
def forecast_disruption_rate(weekly: pd.DataFrame,
                              periods: int = 30,
                              method: str = "auto") -> tuple[pd.DataFrame, str]:
    """
    Run Prophet or Holt-Winters depending on availability and `method`.

    Parameters
    ----------
    weekly  : output of build_weekly_series()
    periods : number of future weeks to forecast
    method  : 'prophet' | 'holtwinters' | 'auto'

    Returns
    -------
    (forecast_df, method_used_str)
    """
    if method == "holtwinters":
        return holtwinters_forecast(weekly, periods), "Holt-Winters"

    try:
        result = prophet_forecast(weekly, periods)
        return result, "Prophet"
    except Exception as e:
        print(f"[forecast] Prophet failed ({e}), falling back to Holt-Winters")
        return holtwinters_forecast(weekly, periods), "Holt-Winters"


# ─────────────────────────────────────────────────────────────
# 7. Risk label for forecasted weeks
# ─────────────────────────────────────────────────────────────
def label_forecast_risk(forecast_df: pd.DataFrame) -> pd.DataFrame:
    """Add a Risk_Level column to the forecast DataFrame."""
    def _label(p):
        if p < 0.35:   return "🟢 Low"
        elif p < 0.65: return "🟡 Medium"
        else:          return "🔴 High"

    df = forecast_df.copy()
    df["Risk_Level"] = df["yhat"].apply(_label)
    return df


# ─────────────────────────────────────────────────────────────
# 8. Plotly charts
# ─────────────────────────────────────────────────────────────
def plot_forecast(weekly: pd.DataFrame,
                  forecast_df: pd.DataFrame,
                  method: str = "Prophet") -> go.Figure:
    """
    Line chart showing historical disruption rate + forecast with
    confidence band.
    """
    hist = forecast_df[~forecast_df["is_forecast"]]
    fore = forecast_df[ forecast_df["is_forecast"]]

    fig = go.Figure()

    # Confidence band
    fig.add_trace(go.Scatter(
        x=pd.concat([fore["ds"], fore["ds"].iloc[::-1]]),
        y=pd.concat([fore["yhat_upper"], fore["yhat_lower"].iloc[::-1]]),
        fill="toself",
        fillcolor="rgba(255,165,0,0.15)",
        line=dict(color="rgba(255,255,255,0)"),
        name="80% Confidence Band",
        showlegend=True,
    ))

    # Historical actual
    fig.add_trace(go.Scatter(
        x=weekly["ds"], y=weekly["y"],
        mode="lines+markers",
        name="Historical Disruption Rate",
        line=dict(color="#3498db", width=2),
        marker=dict(size=6),
    ))

    # Forecast line
    fig.add_trace(go.Scatter(
        x=fore["ds"], y=fore["yhat"],
        mode="lines+markers",
        name=f"{method} Forecast",
        line=dict(color="#e74c3c", width=2, dash="dash"),
        marker=dict(size=5),
    ))

    # Vertical line at forecast start
    split_date = weekly["ds"].max()
    fig.add_vline(
        x=split_date, line_dash="dot", line_color="gray",
        annotation_text="Forecast Start", annotation_position="top right",
    )

    # Risk threshold lines
    fig.add_hline(y=0.35, line_dash="dot", line_color="#2ecc71",
                  annotation_text="Low/Medium threshold (35%)",
                  annotation_position="bottom right")
    fig.add_hline(y=0.65, line_dash="dot", line_color="#e74c3c",
                  annotation_text="Medium/High threshold (65%)",
                  annotation_position="bottom right")

    fig.update_layout(
        title=f"Supply Chain Disruption Rate Forecast ({method})",
        xaxis_title="Week",
        yaxis_title="Disruption Rate",
        yaxis=dict(tickformat=".0%", range=[0, 1.05]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=500,
    )
    return fig


def plot_forecast_table(forecast_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a clean summary table of upcoming forecast weeks.
    """
    fore = forecast_df[forecast_df["is_forecast"]].copy()
    fore = label_forecast_risk(fore)
    fore["Week"] = fore["ds"].dt.strftime("%Y-%m-%d")
    fore["Disruption Rate"] = (fore["yhat"] * 100).round(1).astype(str) + "%"
    fore["Lower Bound"]     = (fore["yhat_lower"] * 100).round(1).astype(str) + "%"
    fore["Upper Bound"]     = (fore["yhat_upper"] * 100).round(1).astype(str) + "%"
    return fore[["Week", "Disruption Rate", "Lower Bound",
                 "Upper Bound", "Risk_Level"]].reset_index(drop=True)


def plot_category_forecast(df: pd.DataFrame,
                            col: str = "Transportation modes") -> go.Figure:
    """
    Line chart of historical weekly disruption rate split by a category
    (transport mode, supplier, etc.).
    """
    cat_series = build_category_series(df, col)
    if cat_series.empty:
        return None

    cats = [c for c in cat_series.columns if c != "ds"]
    colors = px.colors.qualitative.Set2

    fig = go.Figure()
    for i, cat in enumerate(cats):
        fig.add_trace(go.Scatter(
            x=cat_series["ds"],
            y=cat_series[cat],
            mode="lines+markers",
            name=cat,
            line=dict(color=colors[i % len(colors)], width=2),
        ))

    fig.update_layout(
        title=f"Historical Disruption Rate by {col}",
        xaxis_title="Week",
        yaxis_title="Disruption Rate",
        yaxis=dict(tickformat=".0%", range=[0, 1.05]),
        height=400,
    )
    return fig


def plot_risk_calendar(forecast_df: pd.DataFrame) -> go.Figure:
    """
    Bar chart of upcoming weeks colour-coded by risk level.
    """
    fore = forecast_df[forecast_df["is_forecast"]].copy()
    fore = label_forecast_risk(fore)

    color_map = {"🟢 Low": "#2ecc71", "🟡 Medium": "#f39c12", "🔴 High": "#e74c3c"}

    fig = go.Figure()
    for risk, grp in fore.groupby("Risk_Level"):
        fig.add_trace(go.Bar(
            x=grp["ds"].dt.strftime("%b %d"),
            y=grp["yhat"],
            name=risk,
            marker_color=color_map.get(risk, "#95a5a6"),
            text=(grp["yhat"] * 100).round(1).astype(str) + "%",
            textposition="outside",
        ))

    fig.update_layout(
        title="Upcoming Week-by-Week Disruption Risk",
        xaxis_title="Week Starting",
        yaxis_title="Predicted Disruption Rate",
        yaxis=dict(tickformat=".0%", range=[0, 1.1]),
        barmode="stack",
        height=430,
    )
    return fig


# ─────────────────────────────────────────────────────────────
# 9. Summary statistics for the forecast period
# ─────────────────────────────────────────────────────────────
def forecast_summary(forecast_df: pd.DataFrame) -> dict:
    """Return key stats about the forecast window."""
    fore = forecast_df[forecast_df["is_forecast"]]
    labeled = label_forecast_risk(fore)

    risk_counts = labeled["Risk_Level"].value_counts().to_dict()
    return {
        "forecast_weeks"  : len(fore),
        "avg_rate"        : round(float(fore["yhat"].mean()), 4),
        "peak_rate"       : round(float(fore["yhat"].max()),  4),
        "peak_week"       : fore.loc[fore["yhat"].idxmax(), "ds"].strftime("%Y-%m-%d"),
        "low_weeks"       : risk_counts.get("🟢 Low",    0),
        "medium_weeks"    : risk_counts.get("🟡 Medium", 0),
        "high_weeks"      : risk_counts.get("🔴 High",   0),
    }


# ─────────────────────────────────────────────────────────────
# CLI self-test
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    base = os.path.dirname(SRC_DIR)
    csv  = os.path.join(base, "data", "supply_chain_data.csv")

    df = load_data(csv)
    df = add_or_validate_target(df)

    weekly = build_weekly_series(df)
    print(f"Weekly series: {len(weekly)} weeks\n{weekly.head()}\n")

    forecast_df, method = forecast_disruption_rate(weekly, periods=12)
    print(f"Forecast method: {method}")
    print(forecast_df[forecast_df["is_forecast"]][["ds","yhat","yhat_lower","yhat_upper"]].to_string())

    summary = forecast_summary(forecast_df)
    print(f"\nForecast summary: {summary}")
