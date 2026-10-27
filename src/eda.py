"""
eda.py
------
Exploratory Data Analysis for the Supply Chain dataset.
All charts are generated from ACTUAL dataset columns only.
Returns Matplotlib / Plotly figures so they can be embedded in Streamlit.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for server use
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

TARGET = "Disruption_Risk"


# ─────────────────────────────────────────────────────────
# Helper: safe column access
# ─────────────────────────────────────────────────────────
def _col_present(df: pd.DataFrame, col: str) -> bool:
    return col in df.columns


# ─────────────────────────────────────────────────────────
# 1. Class balance bar chart
# ─────────────────────────────────────────────────────────
def plot_class_balance(df: pd.DataFrame):
    """Bar chart: Disruption vs Non-disruption counts."""
    counts = df[TARGET].value_counts().reset_index()
    counts.columns = [TARGET, "Count"]
    counts["Label"] = counts[TARGET].map({0: "No Disruption", 1: "Disruption"})

    fig = px.bar(
        counts, x="Label", y="Count",
        color="Label",
        color_discrete_map={"No Disruption": "#2ecc71", "Disruption": "#e74c3c"},
        title="Disruption vs Non-Disruption Count",
        text="Count",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Number of Shipments")
    return fig


# ─────────────────────────────────────────────────────────
# 2. Disruptions by Transportation Mode
# ─────────────────────────────────────────────────────────
def plot_disruption_by_transport(df: pd.DataFrame):
    """Grouped bar: disruption rate per transport mode."""
    if not _col_present(df, "Transportation modes"):
        return None

    grp = (df.groupby("Transportation modes")[TARGET]
           .mean()
           .reset_index()
           .rename(columns={TARGET: "Disruption Rate"}))
    grp["Disruption Rate %"] = (grp["Disruption Rate"] * 100).round(1)

    fig = px.bar(
        grp, x="Transportation modes", y="Disruption Rate %",
        color="Disruption Rate %",
        color_continuous_scale="RdYlGn_r",
        title="Disruption Rate by Transportation Mode (%)",
        text="Disruption Rate %",
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(xaxis_title="Transportation Mode",
                      yaxis_title="Disruption Rate (%)",
                      coloraxis_showscale=False)
    return fig


# ─────────────────────────────────────────────────────────
# 3. Disruptions by Supplier
# ─────────────────────────────────────────────────────────
def plot_disruption_by_supplier(df: pd.DataFrame):
    """Bar chart: disruption rate per supplier."""
    if not _col_present(df, "Supplier name"):
        return None

    grp = (df.groupby("Supplier name")[TARGET]
           .agg(["sum", "count"])
           .reset_index()
           .rename(columns={"sum": "Disruptions", "count": "Total"}))
    grp["Rate %"] = (100 * grp["Disruptions"] / grp["Total"]).round(1)

    fig = px.bar(
        grp.sort_values("Rate %", ascending=False),
        x="Supplier name", y="Rate %",
        color="Rate %",
        color_continuous_scale="Reds",
        title="Disruption Rate by Supplier (%)",
        text="Rate %",
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(xaxis_title="Supplier",
                      yaxis_title="Disruption Rate (%)",
                      coloraxis_showscale=False)
    return fig


# ─────────────────────────────────────────────────────────
# 4. Disruptions by Location / Region
# ─────────────────────────────────────────────────────────
def plot_disruption_by_location(df: pd.DataFrame):
    """Bar chart: disruption rate per location."""
    if not _col_present(df, "Location"):
        return None

    grp = (df.groupby("Location")[TARGET]
           .mean()
           .reset_index()
           .rename(columns={TARGET: "Disruption Rate"}))
    grp["Rate %"] = (grp["Disruption Rate"] * 100).round(1)

    fig = px.bar(
        grp.sort_values("Rate %", ascending=False),
        x="Location", y="Rate %",
        color="Rate %",
        color_continuous_scale="OrRd",
        title="Disruption Rate by Location (%)",
        text="Rate %",
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(xaxis_title="Location",
                      yaxis_title="Disruption Rate (%)",
                      coloraxis_showscale=False)
    return fig


# ─────────────────────────────────────────────────────────
# 5. Disruptions by Shipping Carrier
# ─────────────────────────────────────────────────────────
def plot_disruption_by_carrier(df: pd.DataFrame):
    """Bar chart: disruption rate per carrier."""
    if not _col_present(df, "Shipping carriers"):
        return None

    grp = (df.groupby("Shipping carriers")[TARGET]
           .mean()
           .reset_index()
           .rename(columns={TARGET: "Rate"}))
    grp["Rate %"] = (grp["Rate"] * 100).round(1)

    fig = px.bar(
        grp.sort_values("Rate %", ascending=False),
        x="Shipping carriers", y="Rate %",
        color="Rate %",
        color_continuous_scale="Blues",
        title="Disruption Rate by Shipping Carrier (%)",
        text="Rate %",
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(xaxis_title="Carrier",
                      yaxis_title="Disruption Rate (%)",
                      coloraxis_showscale=False)
    return fig


# ─────────────────────────────────────────────────────────
# 6. Lead Time vs Disruption (box plot)
# ─────────────────────────────────────────────────────────
def plot_lead_time_vs_disruption(df: pd.DataFrame):
    """Box plot: lead time distribution for disrupted vs non-disrupted."""
    col = None
    for c in ["Lead time", "Lead times"]:
        if _col_present(df, c):
            col = c
            break
    if col is None:
        return None

    plot_df = df[[col, TARGET]].copy()
    plot_df["Status"] = plot_df[TARGET].map({0: "No Disruption", 1: "Disruption"})

    fig = px.box(
        plot_df, x="Status", y=col,
        color="Status",
        color_discrete_map={"No Disruption": "#3498db", "Disruption": "#e74c3c"},
        title=f"Lead Time Distribution by Disruption Status",
        points="all",
    )
    fig.update_layout(xaxis_title="Disruption Status",
                      yaxis_title="Lead Time (days)",
                      showlegend=False)
    return fig


# ─────────────────────────────────────────────────────────
# 7. Defect Rate distribution
# ─────────────────────────────────────────────────────────
def plot_defect_rate_distribution(df: pd.DataFrame):
    """Histogram: defect rate coloured by disruption status."""
    if not _col_present(df, "Defect rates"):
        return None

    plot_df = df[["Defect rates", TARGET]].copy()
    plot_df["Status"] = plot_df[TARGET].map({0: "No Disruption", 1: "Disruption"})

    fig = px.histogram(
        plot_df, x="Defect rates", color="Status",
        nbins=20,
        color_discrete_map={"No Disruption": "#2ecc71", "Disruption": "#e74c3c"},
        barmode="overlay",
        opacity=0.75,
        title="Defect Rate Distribution by Disruption Status",
    )
    fig.update_layout(xaxis_title="Defect Rate (%)",
                      yaxis_title="Count")
    return fig


# ─────────────────────────────────────────────────────────
# 8. Shipping cost distribution
# ─────────────────────────────────────────────────────────
def plot_shipping_cost_distribution(df: pd.DataFrame):
    """Histogram: shipping cost coloured by disruption status."""
    if not _col_present(df, "Shipping costs"):
        return None

    plot_df = df[["Shipping costs", TARGET]].copy()
    plot_df["Status"] = plot_df[TARGET].map({0: "No Disruption", 1: "Disruption"})

    fig = px.histogram(
        plot_df, x="Shipping costs", color="Status",
        nbins=20,
        color_discrete_map={"No Disruption": "#3498db", "Disruption": "#e74c3c"},
        barmode="overlay",
        opacity=0.75,
        title="Shipping Cost Distribution by Disruption Status",
    )
    fig.update_layout(xaxis_title="Shipping Cost",
                      yaxis_title="Count")
    return fig


# ─────────────────────────────────────────────────────────
# 9. Correlation heatmap
# ─────────────────────────────────────────────────────────
def plot_correlation_heatmap(df: pd.DataFrame):
    """Plotly heatmap of numerical feature correlations."""
    num_df = df.select_dtypes(include=["int64", "float64"])
    if num_df.shape[1] < 2:
        return None

    corr = num_df.corr().round(2)
    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title="Correlation Heatmap – Numerical Features",
        aspect="auto",
    )
    fig.update_layout(width=700, height=600)
    return fig


# ─────────────────────────────────────────────────────────
# 10. Shipping time vs disruption
# ─────────────────────────────────────────────────────────
def plot_shipping_time_vs_disruption(df: pd.DataFrame):
    """Violin plot: shipping time by disruption status."""
    if not _col_present(df, "Shipping times"):
        return None

    plot_df = df[["Shipping times", TARGET]].copy()
    plot_df["Status"] = plot_df[TARGET].map({0: "No Disruption", 1: "Disruption"})

    fig = px.violin(
        plot_df, x="Status", y="Shipping times",
        color="Status",
        color_discrete_map={"No Disruption": "#9b59b6", "Disruption": "#e74c3c"},
        box=True, points="all",
        title="Shipping Time Distribution by Disruption Status",
    )
    fig.update_layout(xaxis_title="Disruption Status",
                      yaxis_title="Shipping Time (days)",
                      showlegend=False)
    return fig


# ─────────────────────────────────────────────────────────
# Save all EDA charts to outputs/
# ─────────────────────────────────────────────────────────
def save_all_charts(df: pd.DataFrame, output_dir: str = "outputs"):
    """Generate and save every available chart as PNG."""
    import os, plotly.io as pio
    os.makedirs(output_dir, exist_ok=True)

    charts = {
        "class_balance"         : plot_class_balance(df),
        "disruption_by_transport": plot_disruption_by_transport(df),
        "disruption_by_supplier" : plot_disruption_by_supplier(df),
        "disruption_by_location" : plot_disruption_by_location(df),
        "disruption_by_carrier"  : plot_disruption_by_carrier(df),
        "lead_time_vs_disruption": plot_lead_time_vs_disruption(df),
        "defect_rate_dist"       : plot_defect_rate_distribution(df),
        "shipping_cost_dist"     : plot_shipping_cost_distribution(df),
        "correlation_heatmap"    : plot_correlation_heatmap(df),
        "shipping_time_dist"     : plot_shipping_time_vs_disruption(df),
    }

    for name, fig in charts.items():
        if fig is not None:
            out_path = os.path.join(output_dir, f"{name}.png")
            try:
                pio.write_image(fig, out_path, scale=2)
                print(f"[EDA] Saved {out_path}")
            except Exception as e:
                print(f"[EDA] Could not save {name}: {e}")


# ─────────────────────────────────────────────────────────
# Quick self-test
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from data_preprocessing import preprocess

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base, "data", "supply_chain_data.csv")

    *_, df = preprocess(csv_path)
    save_all_charts(df, output_dir=os.path.join(base, "outputs"))
    print("EDA charts saved to outputs/")
