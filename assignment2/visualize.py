"""
Visualize a recorded trajectory CSV.
Usage:
    python visualize.py                        # loads trajectory.csv
    python visualize.py --file figure8.csv
"""

import argparse
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

parser = argparse.ArgumentParser()
parser.add_argument("--file", default="trajectory.csv")
args = parser.parse_args()

df = pd.read_csv(args.file)
df["t"] = df["timestamp"] - df["timestamp"].iloc[0]  # relative time from 0

# ── 3D trajectory ────────────────────────────────────────────────────────────
fig3d = go.Figure()
fig3d.add_trace(go.Scatter3d(
    x=df["x"], y=df["y"], z=df["z"],
    mode="lines+markers",
    marker=dict(size=2, color=df["t"], colorscale="Viridis", showscale=True,
                colorbar=dict(title="time (s)")),
    line=dict(color="lightgray", width=1),
    name="trajectory",
))
fig3d.update_layout(
    title="3D Trajectory",
    scene=dict(xaxis_title="x (m)", yaxis_title="y (m)", zaxis_title="z (m)"),
)
fig3d.show()

# ── x/y/z vs time ────────────────────────────────────────────────────────────
fig_ts = make_subplots(rows=3, cols=1, shared_xaxes=True,
                       subplot_titles=("x", "y", "z"))
for i, axis in enumerate(["x", "y", "z"], start=1):
    fig_ts.add_trace(go.Scatter(x=df["t"], y=df[axis], mode="lines", name=axis), row=i, col=1)
    fig_ts.update_yaxes(title_text=f"{axis} (m)", row=i, col=1)
fig_ts.update_xaxes(title_text="time (s)", row=3, col=1)
fig_ts.update_layout(title="Position vs Time", showlegend=False)
fig_ts.show()

# ── frame interval (dropped frame detector) ──────────────────────────────────
dt = df["timestamp"].diff().dropna()
fig_dt = go.Figure()
fig_dt.add_trace(go.Scatter(x=df["t"].iloc[1:], y=dt * 1000, mode="lines", name="dt"))
fig_dt.add_hline(y=dt.median() * 1000 * 2, line_dash="dash", line_color="red",
                 annotation_text="2× median (dropped frame threshold)")
fig_dt.update_layout(title="Frame Interval (dropped frame detector)",
                     xaxis_title="time (s)", yaxis_title="dt (ms)")
fig_dt.show()
