### Summer Internship - Earth Embeddings
### Showcase - Risk & Clustering Interactive App
### By Edgar Daniel

"""
Low-data interactive showcase: a CDMX street map colored by regression
risk tier (left), next to a PCA feature-space scatter colored by k=4
spherical k-means cluster (right). Selecting a street in either view
highlights the same street in the other.

Reads only the small precomputed sample from ``data/sample`` (see
``src/showcase/prepare_sample.py``) — no model inference happens here.

Run:  streamlit run src/showcase/app.py
"""

### -------------------------------------------------------------------------------
### Libraries and parameters-------------------------------------------------------

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import geopandas as gpd
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.unsupervised.clustering_kmeans import PaletteConfig, categorical_palette
from src.utils.style import DEFAULT as PALETTE
from src.visuals.geo_applications import REGRESSION_RISK_LABELS, RISK_COLORS

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_PATH = REPO_ROOT / "data" / "sample" / "showcase_streets_2025.parquet"

K = 4
CLUSTER_COLORS = categorical_palette(PaletteConfig(), K)
MARKER_SIZE = 1
HIGHLIGHT_SIZE = 20
LINE_WIDTH = 1.5
HIGHLIGHT_LINE_WIDTH = 5
HIGHLIGHT_HALO_WIDTH = 16
HIGHLIGHT_HALO_OPACITY = 0.30

### -------------------------------------------------------------------------------
### Data -----------------------------------------------------------------------


@st.cache_data(show_spinner="Loading sample data...")
def load_sample() -> pd.DataFrame:
    """Load the cached, precomputed sample (real street LineString
    geometry + name, predictions, risk tier, cluster and PCA)."""
    return gpd.read_parquet(SAMPLE_PATH)


### -------------------------------------------------------------------------------
### Figures ----------------------------------------------------------------------


def _street_line_trace(
    sub: pd.DataFrame, color: str, name: str,
    width: float = LINE_WIDTH, showlegend: bool = True, opacity: float = 1.0,
) -> go.Scattermapbox:
    """One Scattermapbox line trace covering every street in ``sub``.

    Each street's own coordinates are followed by a ``None`` break, so
    they render as separate segments rather than one connected path;
    every vertex carries that street's CVEGEO as ``customdata`` so a
    click anywhere along a line resolves to the street it belongs to.
    """
    lons, lats, customdata, hovertext = [], [], [], []
    for _, row in sub.iterrows():
        label = f"{row['street_name']} ({row['CVEGEO']})"
        for lon, lat in row.geometry.coords:
            lons.append(lon)
            lats.append(lat)
            customdata.append(row["CVEGEO"])
            hovertext.append(label)
        lons.append(None)
        lats.append(None)
        customdata.append(None)
        hovertext.append(None)

    return go.Scattermapbox(
        lat=lats, lon=lons, mode="lines",
        line=dict(width=width, color=color),
        name=name, customdata=customdata,
        hovertext=hovertext, hoverinfo="text",
        showlegend=showlegend, opacity=opacity,
    )


def build_map_figure(df: pd.DataFrame, selected_cvegeo: str | None) -> go.Figure:
    """CDMX street map (every street's real LineString geometry), one
    trace per regression risk tier, colored with the house risk
    palette, on a Carto Positron basemap.

    The selected street is redrawn twice on top: a wide, translucent
    "halo" trace followed by a thin solid one, so it reads as a
    highlighted glow against the light basemap rather than just a
    thicker line.
    """
    fig = go.Figure()
    for tier, color in RISK_COLORS.items():
        sub = df[df["risk_reg"] == tier]
        fig.add_trace(_street_line_trace(sub, color, REGRESSION_RISK_LABELS[tier]))

    if selected_cvegeo is not None:
        row = df[df["CVEGEO"] == selected_cvegeo]
        fig.add_trace(_street_line_trace(
            row, PALETTE.bad, "Selected (halo)",
            width=HIGHLIGHT_HALO_WIDTH, showlegend=False,
            opacity=HIGHLIGHT_HALO_OPACITY,
        ))
        fig.add_trace(_street_line_trace(
            row, PALETTE.bad, "Selected",
            width=HIGHLIGHT_LINE_WIDTH, showlegend=False,
        ))

    fig.update_layout(
        mapbox=dict(
            style="carto-positron", zoom=9.5,
            center=dict(lat=df["lat"].mean(), lon=df["lon"].mean()),
        ),
        margin=dict(l=0, r=0, t=0, b=0), height=560,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
        clickmode="event+select",
    )
    return fig


def build_feature_figure(df: pd.DataFrame, selected_cvegeo: str | None) -> go.Figure:
    """PCA feature-space scatter, one trace per k-means cluster, colored
    with the house categorical palette."""
    fig = go.Figure()
    for c, color in zip(sorted(df["cluster"].unique()), CLUSTER_COLORS):
        sub = df[df["cluster"] == c]
        fig.add_trace(go.Scatter(
            x=sub["PC1"], y=sub["PC2"], mode="markers",
            marker=dict(size=MARKER_SIZE, color=color),
            name=f"cluster {c}",
            customdata=sub["CVEGEO"], hovertext=sub["CVEGEO"],
            hoverinfo="text",
        ))

    if selected_cvegeo is not None:
        row = df[df["CVEGEO"] == selected_cvegeo]
        fig.add_trace(go.Scatter(
            x=row["PC1"], y=row["PC2"], mode="markers",
            marker=dict(
                size=HIGHLIGHT_SIZE, color=PALETTE.bad,
                line=dict(width=2, color="white"),
            ),
            name="Selected", customdata=row["CVEGEO"], showlegend=False,
        ))

    fig.update_layout(
        xaxis_title="PC1", yaxis_title="PC2",
        margin=dict(l=10, r=10, t=0, b=10), height=560,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
        plot_bgcolor=PALETTE.background, paper_bgcolor=PALETTE.background,
        clickmode="event+select",
    )
    return fig


def _selected_cvegeo_from_event(event) -> str | None:
    """Pull the clicked point's CVEGEO out of a plotly_chart selection event."""
    if not event:
        return None
    points = event.get("selection", {}).get("points", [])
    if not points:
        return None
    return points[0].get("customdata")


### -------------------------------------------------------------------------------
### App ---------------------------------------------------------------------------


st.set_page_config(page_title="CDMX Risk & Clustering Showcase", layout="wide")

st.title("CDMX Street Risk & Embedding Clusters — 2025")
st.caption(
    "Regression-risk map (left) and k=4 spherical k-means clusters "
    "projected on PCA (right), for every street in CDMX. "
    "Click a street in either panel to highlight it in the other."
)

df = load_sample()
st.session_state.setdefault("selected_cvegeo", None)

col_map, col_feat = st.columns(2)

with col_map:
    st.subheader("Predicted Regression Risk")
    map_event = st.plotly_chart(
        build_map_figure(df, st.session_state["selected_cvegeo"]),
        use_container_width=True, key="map_chart",
        on_select="rerun", selection_mode=["points"],
    )

with col_feat:
    st.subheader("Embedding Feature Space (PCA + K-Means)")
    feat_event = st.plotly_chart(
        build_feature_figure(df, st.session_state["selected_cvegeo"]),
        use_container_width=True, key="scatter_chart",
        on_select="rerun", selection_mode=["points"],
    )

new_selection = (
    _selected_cvegeo_from_event(map_event)
    or _selected_cvegeo_from_event(feat_event)
)
if new_selection and new_selection != st.session_state["selected_cvegeo"]:
    st.session_state["selected_cvegeo"] = new_selection
    st.rerun()

st.divider()

if st.session_state["selected_cvegeo"]:
    row = df.loc[df["CVEGEO"] == st.session_state["selected_cvegeo"]].iloc[0]
    st.subheader(f"Selected street — {row['street_name']} ({row['CVEGEO']})")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predicted incidents", f"{row['y_cnt_pred']:.3f}")
    c2.metric("Incident probability", f"{row['y_ind_proba']:.1%}")
    c3.metric("Risk tier", row["risk_label"])
    c4.metric("Cluster", int(row["cluster"]))
else:
    st.caption("No street selected yet — click a point on either chart above.")
