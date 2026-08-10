### Summer Internship - Earth Embeddings
### EDA - Constructores de gráficas (versión en español)
### By Edgar Daniel


"""

Lógica central del EDA para los datos de delitos de la CDMX: paleta,
preparación de datos y constructores de gráficas, con todos los textos
visibles (títulos, ejes, leyendas) en español.

Copia en español de ``src.eda.core``: la lógica es idéntica, únicamente
cambian las cadenas que se renderizan en las figuras.

Cada constructor público regresa un artefacto en memoria (``Figure`` de
matplotlib, ``bytes`` de GIF o un ``folium.Map``); nada se escribe a disco.

La paleta vive en ``src.utils.style``; los alias de color a nivel de módulo
se conservan por compatibilidad con las importaciones que dependen de ellos.

"""

### -------------------------------------------------------------------------------
### Libraries and parameters-------------------------------------------------------

from __future__ import annotations

import io
import logging
import sys
from pathlib import Path
from typing import Dict, Iterable

# Allow both `python -m src.eda.core_esp` and direct `python src/eda/core_esp.py`.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import folium
import geopandas as gpd
import h3
import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from folium.plugins import HeatMap
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image as PILImage

from src.utils.geom import split_streets_at_intersections
from src.utils.style import (
    DEFAULT as PALETTE,
    THOUSANDS as _THOUSANDS,
    Palette,
    apply_theme as _apply_theme,
)

logger = logging.getLogger("crime_eda_esp")

### -------------------------------------------------------------------------------
### Paths -------------------------------------------------------------------------

# Default parameters for working functions

REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_FILE = REPO_ROOT / "data" / "raw" / "carpetasFGJ.csv"
PD_SHP_PATH = REPO_ROOT / "data" / "clean" / "pd_shapes" / "09m.shp"
STREETS_SHP_PATH = REPO_ROOT / "data" / "clean" / "pd_shapes" / "09e.shp"

SELECTED_COLUMNS_DICT = {
    "delito": "crime",
    "categoria_delito": "crime_cat",
    "municipio_hecho": "district",
    "mes_hecho": "month",
    "anio_hecho": "year",
    "hora_hecho": "hour",
    "fecha_hecho": "date",
    "latitud": "lat",
    "longitud": "lon",
}

START_YEAR = 2016
GEO_INDEX_COL = "CVEGEO"
BBOX = {"lat_lo": 19.04, "lat_hi": 19.75, "lon_lo": -99.40, "lon_hi": -98.95}

# DBSCAN parameters
EPSG_PROJ = "EPSG:6372"
EPSILON_M = 500
MIN_SAMPLES = 35
ALPHA_HULL = 0.001

### -------------------------------------------------------------------------------
### Palette aliases (single source: src.utils.style) ------------------------------

BG = PALETTE.background
PANEL = PALETTE.panel
INK = PALETTE.ink
MUTED = PALETTE.muted
GRID = PALETTE.grid

NAVY = PALETTE.navy
BLUE = PALETTE.blue
SKY = PALETTE.sky
GRAY = PALETTE.gray

GOOD = PALETTE.good   # escala buena/mala de valores faltantes
BAD = PALETTE.bad     # reservado para resaltados adversos

QUAL = list(PALETTE.qual)
SEQ = PALETTE.seq_cmap()
BAD_SEQ = PALETTE.bad_cmap()   # rampa roja reservada para y_cnt / conteos adversos

DOW = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]
MES_ES = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio", "Julio",
          "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"]


def apply_theme(palette: Palette = PALETTE) -> None:
    """Aplica el tema claro corporativo compartido (ver ``src.utils.style``)."""
    _apply_theme(palette)


### -------------------------------------------------------------------------------
### Functions ---------------------------------------------------------------------


# Data loading and preparation

def format_columns(
    df: pd.DataFrame,
    selected_columns: Dict[str, str] = SELECTED_COLUMNS_DICT,
) -> pd.DataFrame:
    """Renombra las columnas crudas a sus nombres canónicos (idempotente)."""
    return df.rename(columns=selected_columns)


def load_raw(
    path: str | Path = INPUT_FILE,
    selected_columns: Dict[str, str] = SELECTED_COLUMNS_DICT,
) -> pd.DataFrame:
    """Lee únicamente las columnas necesarias para el EDA."""
    logger.info("Leyendo %s", path)
    wanted = set(selected_columns.keys())
    return pd.read_csv(path, usecols=lambda c: c in wanted)


def prepare(
    df: pd.DataFrame,
    selected_columns: Dict[str, str] = SELECTED_COLUMNS_DICT,
) -> pd.DataFrame:
    """Normaliza los tipos de dato y deriva las columnas temporales auxiliares."""
    df = format_columns(df, selected_columns)

    df = df.filter(list(selected_columns.values())).dropna().drop_duplicates().copy()

    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["date"] = pd.to_datetime(df["date"], errors="coerce", format="%Y-%m-%d")
    df["hour"] = pd.to_datetime(
        df["hour"], format="%H:%M:%S", errors="coerce").dt.hour
    df["dow"] = df["date"].dt.dayofweek
    df["month_num"] = df["date"].dt.month

    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    return df


def filter_data(
    df: pd.DataFrame,
    year_range: tuple[int, int],
    categories: Iterable[str] | None = None,
    crimes: Iterable[str] | None = None,
    districts: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Aplica los filtros del tablero; ``None``/vacío significa sin restricción."""
    lo, hi = year_range
    out = df[df["year"].between(lo, hi)]
    if categories:
        out = out[out["crime_cat"].isin(list(categories))]
    if crimes:
        out = out[out["crime"].isin(list(crimes))]
    if districts:
        out = out[out["district"].isin(list(districts))]
    return out.copy()


def geo_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separa los registros dentro y fuera del área delimitadora."""
    geo = df.dropna(subset=["lat", "lon"])
    mask = (geo["lat"].between(BBOX["lat_lo"], BBOX["lat_hi"])
            & geo["lon"].between(BBOX["lon_lo"], BBOX["lon_hi"]))
    return geo[mask].copy(), geo[~mask].copy()


# Static plots — volumes

def top_crime_types(df: pd.DataFrame, n: int = 15) -> Figure:
    top = df["crime"].value_counts().head(n).sort_values()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top.index, top.values, color=BLUE, height=0.6)
    ax.set_xlabel("Incidentes")
    ax.set_title(f"Top {n} tipos de delito")
    ax.xaxis.set_major_formatter(_THOUSANDS)
    ax.grid(axis="x")
    ax.spines["bottom"].set_visible(False)
    for i, v in enumerate(top.values):
        ax.text(v * 1.005, i, f"{v:,}", va="center", fontsize=8, color=GRAY)
    return fig


def incidents_by_category(df: pd.DataFrame) -> Figure:
    cat = df["crime_cat"].value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(cat.index, cat.values, color=BLUE, height=0.6)
    ax.set_xlabel("Incidentes")
    ax.set_title("Incidentes por categoría de delito")
    ax.xaxis.set_major_formatter(_THOUSANDS)
    ax.grid(axis="x")
    ax.spines["bottom"].set_visible(False)
    for i, v in enumerate(cat.values):
        ax.text(v * 1.005, i, f"{v:,}", va="center", fontsize=8, color=GRAY)
    return fig


def annual_volume(df: pd.DataFrame) -> Figure:
    """Volumen anual con etiquetas interanuales: rojo al subir, tenue al bajar."""
    annual = df.groupby("year").size()
    yoy = annual.pct_change() * 100
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(annual.index, annual.values, color=NAVY, width=0.6, zorder=2)
    ax.set_ylabel("Incidentes")
    ax.set_title("Volumen anual de incidentes")
    ax.yaxis.set_major_formatter(_THOUSANDS)
    ax.grid(axis="y")
    for yr, val, pct in zip(annual.index[1:], annual.values[1:], yoy.values[1:]):
        color = BAD if pct >= 0 else MUTED
        ax.text(yr, val + annual.max() * 0.015, f"{pct:+.1f}%",
                ha="center", fontsize=9, color=color, fontweight="bold")
    return fig


# Static plots — temporal

def monthly_time_series(df: pd.DataFrame) -> Figure:
    ts = df.set_index("date").resample("ME").size().rename("n")
    ma3 = ts.rolling(3, center=True).mean()
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(ts.index, ts.values, color=GRID, lw=1, zorder=1)
    ax.plot(ts.index, ma3.values, color=BLUE, lw=2.5, zorder=2,
            label="Media móvil de 3 meses")
    ax.fill_between(ts.index, ma3.values, alpha=0.1, color=BLUE, zorder=1)
    ax.set_ylabel("Incidentes")
    ax.set_title("Incidentes mensuales")
    ax.yaxis.set_major_formatter(_THOUSANDS)
    ax.legend()
    ax.grid(axis="y")
    return fig


def time_series_by_category(df: pd.DataFrame) -> Figure:
    ts_cat = (df.groupby(["date", "crime_cat"]).size()
              .unstack(fill_value=0).resample("ME").sum())
    cat_order = ts_cat.sum().sort_values(ascending=False).index
    colors = PALETTE.categorical(len(cat_order))
    fig, ax = plt.subplots(figsize=(13, 5))
    for col, color in zip(cat_order, colors):
        ax.plot(ts_cat.index, ts_cat[col], color=color, lw=2, label=col)
    ax.set_ylabel("Incidentes")
    ax.set_title("Incidentes mensuales por categoría de delito")
    ax.yaxis.set_major_formatter(_THOUSANDS)
    ax.legend(ncol=2)
    ax.grid(axis="y")
    return fig


def seasonality_month_dow(df: pd.DataFrame) -> Figure:
    pivot = (df.dropna(subset=["dow", "month_num"])
             .groupby(["month_num", "dow"]).size()
             .unstack(fill_value=0).reindex(range(1, 13)))
    pivot.index = MES_ES
    pivot.columns = [DOW[int(i)] for i in pivot.columns]
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot, ax=ax, cmap=SEQ, linewidths=0.4, linecolor=BG,
                cbar_kws={"label": "Incidentes", "shrink": 0.6})
    ax.set_title("Estacionalidad — Mes x Día de la semana")
    ax.set_xlabel("")
    ax.set_ylabel("")
    return fig


def hourly_rhythm(df: pd.DataFrame) -> Figure:
    pivot = (df.dropna(subset=["dow", "hour"])
             .groupby(["dow", "hour"]).size()
             .unstack(fill_value=0))
    pivot.index = [DOW[int(i)] for i in pivot.index]
    fig, ax = plt.subplots(figsize=(13, 4))
    sns.heatmap(pivot, ax=ax, cmap=SEQ, linewidths=0,
                cbar_kws={"label": "Incidentes", "shrink": 0.6})
    ax.set_title("Ritmo horario — Hora x Día de la semana")
    ax.set_xlabel("Hora (0-23)")
    ax.set_ylabel("")
    return fig


def incidents_by_hour(df: pd.DataFrame) -> Figure:
    hc = df["hour"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(hc.index, hc.values, color=BLUE, width=0.7)
    ax.set_xlabel("Hora del día (0-23)")
    ax.set_ylabel("Incidentes")
    ax.set_title("Incidentes por hora del día")
    ax.yaxis.set_major_formatter(_THOUSANDS)
    ax.grid(axis="y")
    return fig


def intraday_by_category(df: pd.DataFrame) -> Figure:
    ph = df.groupby(["crime_cat", "hour"]).size().unstack(fill_value=0)
    ph_norm = ph.div(ph.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(13, 5))
    sns.heatmap(ph_norm, ax=ax, cmap=SEQ, linewidths=0.4, linecolor=BG,
                cbar_kws={"label": "Fracción por fila", "shrink": 0.6})
    ax.set_title("Ritmo intradiario por categoría de delito (normalizado por fila)")
    ax.set_xlabel("Hora (0-23)")
    ax.set_ylabel("")
    return fig

# Static plots — spatial

def spatial_outliers(geo: pd.DataFrame, geo_out: pd.DataFrame) -> Figure:
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(geo["lon"], geo["lat"], s=0.3, alpha=0.15,
               color=BLUE, rasterized=True, label="Dentro del área")
    if len(geo_out):
        ax.scatter(geo_out["lon"], geo_out["lat"], s=12, alpha=0.8,
                   color=BAD, zorder=5, label=f"Atípicos (n={len(geo_out):,})")
    ax.set_title("Distribución espacial — Filtro de área delimitadora")
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")
    ax.set_aspect("equal")
    ax.legend(markerscale=5)
    ax.grid(True, lw=0.3)
    return fig


def _add_colorbar(fig: Figure, ax, mappable, label: str):
    """Añade una leyenda de escala de color para que las magnitudes sombreadas
    sean legibles.

    Quien genere animaciones debe pasar un mappable con norma *fija*:
    ``_fig_to_pil`` guarda con ``bbox_inches="tight"``, así que etiquetas de
    ancho variable entre cuadros producirían GIFs con cuadros de distinto
    tamaño.
    """
    # El cax ligado al divisor sigue la caja de los ejes, así la barra conserva
    # la altura del gráfico con ``set_aspect("equal")`` en vez de excederla.
    cax = make_axes_locatable(ax).append_axes("right", size="4%", pad=0.15)
    cbar = fig.colorbar(mappable, cax=cax)
    cbar.set_label(label, color=MUTED, fontsize=9)
    cbar.ax.tick_params(colors=MUTED, labelsize=8)
    cbar.outline.set_edgecolor(GRID)
    return cbar


def kde_overall(geo: pd.DataFrame) -> Figure:
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.kdeplot(data=geo, x="lon", y="lat",
                cmap=SEQ, fill=True, thresh=0.03, ax=ax)
    if ax.collections:                      # vacío si no se puede ajustar el KDE
        _add_colorbar(fig, ax, ax.collections[0], "Densidad estimada")
    ax.set_title("Densidad delictiva — KDE general")
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")
    ax.set_aspect("equal")
    return fig


def kde_top3_categories(geo: pd.DataFrame) -> Figure:
    top3 = geo["crime_cat"].value_counts().head(3).index.tolist()
    fig, ax = plt.subplots(figsize=(8, 8))
    for cat, color in zip(top3, PALETTE.categorical(len(top3))):
        sub = geo[geo["crime_cat"] == cat]
        sns.kdeplot(data=sub, x="lon", y="lat", color=color, fill=False,
                    thresh=0.1, levels=5, ax=ax, label=cat, linewidths=1.8)
    ax.set_title("Densidad delictiva — Las 3 categorías principales")
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")
    ax.set_aspect("equal")
    ax.legend()
    return fig


def h3_heatmap(geo: pd.DataFrame, resolution: int = 8) -> folium.Map:
    if geo.empty:
        raise ValueError("h3_heatmap necesita al menos un registro geolocalizado.")
    cells = geo.apply(
        lambda r: h3.latlng_to_cell(r["lat"], r["lon"], resolution), axis=1)
    h3c = cells.value_counts().reset_index()
    h3c.columns = ["h3_cell", "count"]
    h3c["lat"] = h3c["h3_cell"].apply(lambda x: h3.cell_to_latlng(x)[0])
    h3c["lng"] = h3c["h3_cell"].apply(lambda x: h3.cell_to_latlng(x)[1])
    m = folium.Map(location=[geo["lat"].mean(), geo["lon"].mean()],
                   zoom_start=11, tiles="CartoDB positron")
    HeatMap(h3c[["lat", "lng", "count"]].values.tolist(),
            radius=10, blur=15, max_zoom=12).add_to(m)
    return m


# Animations (yearly GIFs) — return raw bytes

def _frames_to_gif(frames: list[PILImage.Image], duration: int = 1000) -> bytes:
    if not frames:
        raise ValueError("No hay cuadros que animar (¿selección vacía?).")
    buf = io.BytesIO()
    frames[0].save(buf, format="GIF", save_all=True, append_images=frames[1:],
                   duration=duration, loop=0, optimize=True)
    return buf.getvalue()


def _fig_to_pil(fig: Figure, dpi: int = 130) -> PILImage.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor=BG, bbox_inches="tight")
    buf.seek(0)
    img = PILImage.open(buf).copy()
    plt.close(fig)
    return img


def gif_scatter_by_year(geo: pd.DataFrame) -> bytes:
    years = sorted(geo["year"].dropna().unique())
    x0, x1 = geo["lon"].min(), geo["lon"].max()
    y0, y1 = geo["lat"].min(), geo["lat"].max()
    frames = []
    for year in years:
        sub = geo[geo["year"] == year]
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(sub["lon"], sub["lat"], s=0.3, alpha=0.3, color=BLUE)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect("equal")
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")
        ax.set_title(f"Ubicación de los delitos — {year}")
        frames.append(_fig_to_pil(fig, dpi=120))
    return _frames_to_gif(frames)


def gif_kde_by_year(geo: pd.DataFrame) -> bytes:
    years = sorted(geo["year"].dropna().unique())
    x0, x1 = geo["lon"].min(), geo["lon"].max()
    y0, y1 = geo["lat"].min(), geo["lat"].max()
    frames = []
    for year in years:
        sub = geo[geo["year"] == year]
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(geo["lon"], geo["lat"], c=GRAY, s=0.5, alpha=0.1)
        sns.kdeplot(data=sub, x="lon", y="lat", cmap=SEQ,
                    fill=True, thresh=0.45, ax=ax, alpha=0.7)
        ax.scatter(sub["lon"], sub["lat"], s=0.3, alpha=0.15,
                   color=GRAY, rasterized=True)
        # Seaborn renormaliza el KDE en cada cuadro, así que la leyenda usa una
        # escala relativa fija 0-1: etiquetas idénticas mantienen el tamaño.
        _add_colorbar(fig, ax,
                      plt.cm.ScalarMappable(cmap=SEQ, norm=plt.Normalize(0, 1)),
                      "Densidad relativa")
        ax.set_title(f"Densidad delictiva — {year}")
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")
        ax.set_aspect("equal")
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        frames.append(_fig_to_pil(fig))
    return _frames_to_gif(frames)


def _concave_hull(pts_xy: np.ndarray, alpha: float = ALPHA_HULL):
    import shapely
    from shapely.geometry import MultiPoint
    mp = MultiPoint(pts_xy)
    if len(pts_xy) < 4:
        return mp.convex_hull
    try:
        hull = shapely.concave_hull(mp, ratio=alpha)
        return hull if not hull.is_empty else mp.convex_hull
    except AttributeError:
        return mp.convex_hull


def gif_dbscan_by_year(geo: pd.DataFrame) -> bytes:
    from pyproj import Transformer
    from sklearn.cluster import DBSCAN

    to_proj = Transformer.from_crs(4326, EPSG_PROJ, always_xy=True)
    x0, x1 = geo["lon"].min(), geo["lon"].max()
    y0, y1 = geo["lat"].min(), geo["lat"].max()
    years = sorted(geo["year"].dropna().unique())
    frames = []
    for year in years:
        sub = geo.loc[geo["year"] == year, ["lon", "lat"]]
        if sub.empty:
            continue
        mx, my = to_proj.transform(sub["lon"].values, sub["lat"].values)
        coords_m = np.column_stack([mx, my])
        labels = DBSCAN(eps=EPSILON_M, min_samples=MIN_SAMPLES,
                        n_jobs=-1).fit_predict(coords_m)
        n_clusters = len([lab for lab in np.unique(labels) if lab != -1])
        noise_pct = round((labels == -1).sum() / len(labels) * 100, 1)

        fig, ax = plt.subplots(figsize=(11, 5), facecolor=BG)
        fig.suptitle(f"DBSCAN — {year}   e={EPSILON_M}m   mín_pts={MIN_SAMPLES}   "
                     f"clústeres: {n_clusters}   ruido: {noise_pct}%",
                     fontsize=11, fontweight="bold", color=INK, y=1.01)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect("equal")
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")
        ax.set_title("Pertenencia a clúster")
        noise = labels == -1
        ax.scatter(sub["lon"][noise], sub["lat"][noise],
                   s=0.3, alpha=0.2, color=GRAY)
        if n_clusters > 0:
            cmap = matplotlib.colormaps["Blues"].resampled(n_clusters)
            ax.scatter(sub["lon"][~noise], sub["lat"][~noise],
                       c=labels[~noise], cmap=cmap,
                       s=0.5, alpha=0.5, vmin=0, vmax=n_clusters)
        handles = [mpatches.Patch(color=GRAY, alpha=0.5,
                                  label=f"ruido ({noise_pct}%)"),
                   mpatches.Patch(color=BLUE, alpha=0.6,
                                  label=f"{n_clusters} clústeres")]
        ax.legend(handles=handles, loc="lower right")
        fig.tight_layout()
        frames.append(_fig_to_pil(fig, dpi=130))
    return _frames_to_gif(frames)


# CVEGEO (district-level) aggregation, animation and distribution

def load_districts(path: str | Path = PD_SHP_PATH) -> gpd.GeoDataFrame:
    return gpd.read_file(path)


def cvegeo_aggregate(df: pd.DataFrame, pd_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Agrega el conteo de incidentes por alcaldía (CVEGEO) y año."""

    # Build point GeoDataFrame
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["lon"], df["lat"]),
        crs="EPSG:4326",
    ).to_crs(pd_gdf.crs)

    # Ensure merge key is treated as a string (preserves leading zeros)
    polygons = pd_gdf[[GEO_INDEX_COL, "geometry"]].copy()
    polygons[GEO_INDEX_COL] = polygons[GEO_INDEX_COL].astype(str)

    # Spatial join and aggregation
    agg = (
        gdf.sjoin(
            polygons,
            how="inner",
            predicate="intersects",
        )
        .drop(columns="geometry")
        .groupby([GEO_INDEX_COL, "year"], as_index=False)
        .agg(cnt=("date", "count"))
    )

    agg[GEO_INDEX_COL] = agg[GEO_INDEX_COL].astype(str)

    # One polygon per (GEO_INDEX, year)
    return gpd.GeoDataFrame(
        agg.merge(polygons, on=GEO_INDEX_COL, how="left"),
        geometry="geometry",
        crs=pd_gdf.crs,
    )


def gif_cvegeo_by_year(crime_gdf: gpd.GeoDataFrame) -> bytes:
    gdf = crime_gdf.copy().to_crs(4326)
    years = sorted(gdf["year"].dropna().unique())
    xmin, ymin, xmax, ymax = gdf.total_bounds
    frames = []
    for year in years:
        fig, ax = plt.subplots(figsize=(6, 6))
        sub = gdf.loc[gdf["year"] == year]
        if not sub.empty:
            sub.plot(ax=ax, column="cnt", alpha=0.7, linewidth=0, aspect=None,
                     cmap=SEQ
                     )
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")
        ax.set_title(f"Polígonos delictivos — {year}")
        frames.append(_fig_to_pil(fig, dpi=120))
    return _frames_to_gif(frames)


def _count_distribution(gdf: gpd.GeoDataFrame, level: str) -> Figure:
    """Histograma / ECDF / diagrama de caja de los conteos anuales, para
    cualquier nivel de agregación."""
    years = sorted(gdf["year"].dropna().unique())
    palette = sns.light_palette(BLUE, n_colors=len(years) + 1)[1:]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), dpi=100)
    for year, color in zip(years, palette):
        sub = gdf.loc[gdf["year"] == year, "cnt"].dropna()
        axes[0].hist(sub, bins=35, density=True, alpha=0.45, color=color,
                     label=int(year))
    axes[0].set_xlabel("cnt")
    axes[0].set_ylabel("Densidad")
    axes[0].set_title(f"Distribución de conteos por año — {level}")
    axes[0].legend(fontsize=8)
    for year, color in zip(years, palette):
        sub = np.sort(gdf.loc[gdf["year"] == year, "cnt"].dropna().values)
        if len(sub) == 0:
            continue
        ecdf = np.arange(1, len(sub) + 1) / len(sub)
        axes[1].plot(sub, ecdf, color=color, linewidth=2, label=int(year))
    axes[1].set_xlabel("cnt")
    axes[1].set_ylabel(r"$F_n(x)$")
    axes[1].set_title(f"CDF empírica por año — {level}")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3, linestyle="--")
    box_style = {"boxes": MUTED, "whiskers": MUTED, "medians": SKY, "caps": MUTED}
    gdf.dropna(subset=["cnt"]).boxplot(column="cnt", by="year", ax=axes[2],
                                       patch_artist=True, color=box_style)
    for patch, color in zip(axes[2].patches, palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    axes[2].set_title(f"Diagrama de caja por año — {level}")
    axes[2].set_xlabel("Año")
    axes[2].set_ylabel("cnt")
    fig.suptitle("")
    fig.tight_layout()
    return fig


def cvegeo_distribution(crime_gdf: gpd.GeoDataFrame) -> Figure:
    return _count_distribution(crime_gdf, "Alcaldía (CVEGEO)")


# Street-segment level: split at intersections, aggregate, animate, distribute

def load_streets(path: str | Path = STREETS_SHP_PATH) -> gpd.GeoDataFrame:
    """Lee la red vial y la divide en las intersecciones (lento, conviene cachearlo)."""
    streets_gdf = (
        gpd.read_file(path)
        .assign(STREET_ID=lambda x: x[GEO_INDEX_COL] + x["CVE_ENT"] + x["CVE_MUN"]
                + x["CVE_LOC"] + x["CVEVIAL"] + x["CVESEG"])
        .drop(columns=[GEO_INDEX_COL, "CVE_ENT", "CVE_MUN", "CVE_LOC",
                       "CVEVIAL", "CVESEG"])
    )
    return (split_streets_at_intersections(streets_gdf, id_col="STREET_ID")
            .rename(columns={"STREET_ID": GEO_INDEX_COL}))


def street_aggregate(
    df: pd.DataFrame, streets_split: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """Asigna cada incidente al segmento de calle más cercano y cuenta por
    segmento y año."""
    gdf = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df["lon"], df["lat"]),
        crs="EPSG:4326")
    agg = (gdf.to_crs(streets_split.crs)
           .sjoin_nearest(streets_split.filter([GEO_INDEX_COL, "geometry"]),
                          how="left")
           .drop(columns="geometry")
           .groupby([GEO_INDEX_COL, "year"])
           .agg(cnt=("date", "count"))
           .reset_index())
    return gpd.GeoDataFrame(
        streets_split[[GEO_INDEX_COL, "geometry"]].merge(agg, on=GEO_INDEX_COL,
                                                         how="left"),
        crs=streets_split.crs, geometry="geometry")


def gif_street_by_year(street_gdf: gpd.GeoDataFrame) -> bytes:
    """Animación anual de los segmentos de calle sombreados por número de incidentes."""
    gdf = street_gdf.to_crs(4326)
    years = sorted(gdf["year"].dropna().unique())
    xmin, ymin, xmax, ymax = gdf.total_bounds
    vmax = gdf["cnt"].quantile(0.99)
    vmax = float(vmax) if pd.notna(vmax) and vmax > 0 else 1.0
    frames = []
    for year in years:
        fig, ax = plt.subplots(figsize=(6, 6))
        gdf.plot(ax=ax, color=GRID, linewidth=0.2, aspect=None)
        active = gdf[(gdf["year"] == year) & (gdf["cnt"] > 0)]
        if not active.empty:
            active.plot(ax=ax, column="cnt", cmap=SEQ, linewidth=0.9,
                        vmin=0, vmax=vmax, aspect=None)
        # Se añade siempre (vmax es global): una leyenda que solo apareciera en
        # los años con datos cambiaría el tamaño de esos cuadros.
        _add_colorbar(fig, ax,
                      plt.cm.ScalarMappable(cmap=SEQ,
                                            norm=plt.Normalize(0, vmax)),
                      "Incidentes por segmento")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")
        ax.set_title(f"Segmentos de calle por número de delitos — {year}")
        frames.append(_fig_to_pil(fig, dpi=120))
    return _frames_to_gif(frames)


def street_distribution(street_gdf: gpd.GeoDataFrame) -> Figure:
    return _count_distribution(street_gdf, "Segmento de calle")


# Export helpers

def fig_to_png_bytes(fig: Figure, dpi: int = 150) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor=BG)
    return buf.getvalue()


def export_all(artifacts: dict[str, bytes | str], out_dir: str | Path) -> list[Path]:
    """Escribe cada artefacto en caché (PNG/GIF/HTML) en ``out_dir``."""
    out = Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for name, data in artifacts.items():
        path = out / name
        mode, payload = ("w", data) if isinstance(data, str) else ("wb", data)
        with open(path, mode) as fh:
            fh.write(payload)
        written.append(path)
        logger.info("Exportado %s", path)
    return written


### -------------------------------------------------------------------------------
### Main (cleans up the raw data source) ------------------------------------------

if __name__ == "__main__":

    from src.utils.prod import init_logger

    logger = init_logger()

    INPUT_PATH = "./data/raw/carpetasFGJ.csv"
    OUTPUT_PATH = "./data/clean/carpetasFGJ.csv"

    logger.info(f"Limpiando el archivo: {INPUT_PATH}")
    try:
        # ``prepare`` renombra internamente, así que load -> prepare es toda la cadena
        prepare(
            load_raw(INPUT_PATH, SELECTED_COLUMNS_DICT),
            SELECTED_COLUMNS_DICT,
        ).to_csv(OUTPUT_PATH, index=False)

        logger.info(f"Archivo {INPUT_PATH} limpiado y guardado en {OUTPUT_PATH}.")
    except Exception as e:
        logger.error(f"Error al limpiar el archivo {INPUT_PATH}. \n {e}")
