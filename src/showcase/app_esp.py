### Summer Internship - Earth Embeddings
### Showcase - App interactiva de riesgo y clustering (versión en español)
### By Edgar Daniel

"""
Muestra interactiva de bajo consumo de datos: un mapa de calles de la CDMX
coloreado por nivel de riesgo de regresión (izquierda), junto a una
dispersión del espacio de features en PCA coloreada por clúster de k-means
esférico con k=4 (derecha). Al seleccionar una calle en cualquiera de las
dos vistas se resalta la misma calle en la otra.

Copia en español de ``src.showcase.app``: la lógica es idéntica, únicamente
cambian los textos de la interfaz y de las figuras. Las etiquetas de riesgo
se derivan de ``risk_reg`` para no depender del texto en inglés guardado en
el parquet.

Lee únicamente la muestra precalculada de ``data/sample`` (ver
``src/showcase/prepare_sample.py``) — aquí no se ejecuta ninguna inferencia.

Ejecutar:  streamlit run src/showcase/app_esp.py
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
from src.visuals.geo_applications_esp import REGRESSION_RISK_LABELS, RISK_COLORS

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

# Etiqueta corta del nivel de riesgo, derivada de ``risk_reg``: el parquet
# guarda ``risk_label`` en inglés, así que aquí no se usa.
RISK_LABELS_ESP = {1: "Bajo", 2: "Medio", 3: "Alto"}

### -------------------------------------------------------------------------------
### Data -----------------------------------------------------------------------


@st.cache_data(show_spinner="Cargando los datos de muestra...")
def load_sample() -> pd.DataFrame:
    """Carga la muestra precalculada en caché (geometría LineString real de la
    calle + nombre, predicciones, nivel de riesgo, clúster y PCA)."""
    return gpd.read_parquet(SAMPLE_PATH)


### -------------------------------------------------------------------------------
### Figures ----------------------------------------------------------------------


def _street_line_trace(
    sub: pd.DataFrame, color: str, name: str,
    width: float = LINE_WIDTH, showlegend: bool = True, opacity: float = 1.0,
) -> go.Scattermapbox:
    """Una traza de líneas Scattermapbox que cubre todas las calles de ``sub``.

    Las coordenadas de cada calle van seguidas de un corte ``None``, de modo
    que se dibujan como segmentos separados y no como una sola ruta conectada;
    cada vértice lleva el CVEGEO de su calle como ``customdata``, así que un
    clic en cualquier punto de la línea se resuelve a la calle correspondiente.
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
    """Mapa de calles de la CDMX (geometría LineString real de cada calle), una
    traza por nivel de riesgo de regresión, coloreada con la paleta de riesgo
    de la casa, sobre un mapa base Carto Positron.

    La calle seleccionada se redibuja dos veces encima: una traza ancha y
    translúcida a modo de "halo" seguida de una delgada y sólida, para que se
    lea como un resplandor resaltado sobre el mapa base claro y no solo como
    una línea más gruesa.
    """
    fig = go.Figure()
    for tier, color in RISK_COLORS.items():
        sub = df[df["risk_reg"] == tier]
        fig.add_trace(_street_line_trace(sub, color, REGRESSION_RISK_LABELS[tier]))

    if selected_cvegeo is not None:
        row = df[df["CVEGEO"] == selected_cvegeo]
        fig.add_trace(_street_line_trace(
            row, PALETTE.bad, "Seleccionada (halo)",
            width=HIGHLIGHT_HALO_WIDTH, showlegend=False,
            opacity=HIGHLIGHT_HALO_OPACITY,
        ))
        fig.add_trace(_street_line_trace(
            row, PALETTE.bad, "Seleccionada",
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
    """Dispersión del espacio de features en PCA, una traza por clúster de
    k-means, coloreada con la paleta categórica de la casa."""
    fig = go.Figure()
    for c, color in zip(sorted(df["cluster"].unique()), CLUSTER_COLORS):
        sub = df[df["cluster"] == c]
        fig.add_trace(go.Scatter(
            x=sub["PC1"], y=sub["PC2"], mode="markers",
            marker=dict(size=MARKER_SIZE, color=color),
            name=f"clúster {c}",
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
            name="Seleccionada", customdata=row["CVEGEO"], showlegend=False,
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
    """Extrae el CVEGEO del punto seleccionado en un evento de plotly_chart."""
    if not event:
        return None
    points = event.get("selection", {}).get("points", [])
    if not points:
        return None
    return points[0].get("customdata")


### -------------------------------------------------------------------------------
### App ---------------------------------------------------------------------------


st.set_page_config(page_title="Riesgo y Clustering CDMX", layout="wide")

st.title("Riesgo Vial y Clústeres de Embeddings en la CDMX — 2025")
st.caption(
    "Mapa de riesgo por regresión (izquierda) y clústeres de k-means esférico "
    "con k=4 proyectados en PCA (derecha), para cada calle de la CDMX. "
    "Haz clic en una calle en cualquiera de los dos paneles para resaltarla "
    "en el otro."
)

df = load_sample()
st.session_state.setdefault("selected_cvegeo", None)

col_map, col_feat = st.columns(2)

with col_map:
    st.subheader("Riesgo predicho por regresión")
    map_event = st.plotly_chart(
        build_map_figure(df, st.session_state["selected_cvegeo"]),
        use_container_width=True, key="map_chart",
        on_select="rerun", selection_mode=["points"],
    )

with col_feat:
    st.subheader("Espacio de features del embedding (PCA + K-Means)")
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
    st.subheader(f"Calle seleccionada — {row['street_name']} ({row['CVEGEO']})")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Incidentes predichos", f"{row['y_cnt_pred']:.3f}")
    c2.metric("Probabilidad de incidente", f"{row['y_ind_proba']:.1%}")
    c3.metric("Nivel de riesgo",
              RISK_LABELS_ESP.get(int(row["risk_reg"]), str(row["risk_reg"])))
    c4.metric("Clúster", int(row["cluster"]))
else:
    st.caption("Aún no hay ninguna calle seleccionada — haz clic en un punto de "
               "cualquiera de las gráficas de arriba.")
