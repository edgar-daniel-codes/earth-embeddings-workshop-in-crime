### Summer Internship - Earth Embeddings
### EDA - Core plot builders
### By Edgar Daniel


"""

Core EDA logic for CDMX crime data: palette, data prep and plot builders.

Every public builder returns an in-memory artefact (matplotlib ``Figure``,
GIF ``bytes`` or a ``folium.Map``); nothing is written to disk.

The palette lives in ``src.utils.style``; the module-level color aliases
below are kept for backwards compatibility with downstream imports.

"""

### -------------------------------------------------------------------------------
### Libraries and parameters-------------------------------------------------------

from __future__ import annotations

import io
import logging
import sys
from pathlib import Path
from typing import Dict, Iterable

# Allow both `python -m src.eda.core` and direct `python src/eda/core.py`.
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

logger = logging.getLogger("crime_eda")

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

GOOD = PALETTE.good   # good/bad missingness scale
BAD = PALETTE.bad     # reserved for adverse/bad highlights

QUAL = list(PALETTE.qual)
SEQ = PALETTE.seq_cmap()
BAD_SEQ = PALETTE.bad_cmap()   # red ramp reserved for y_cnt / adverse counts

DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MES_ES = ["Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio", "Julio",
          "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"]


def apply_theme(palette: Palette = PALETTE) -> None:
    """Apply the shared corporate light theme (see ``src.utils.style``)."""
    _apply_theme(palette)


### -------------------------------------------------------------------------------
### Functions ---------------------------------------------------------------------


# Data loading and preparation

def format_columns(
    df: pd.DataFrame,
    selected_columns: Dict[str, str] = SELECTED_COLUMNS_DICT,
) -> pd.DataFrame:
    """Rename raw source columns to their canonical names (idempotent)."""
    return df.rename(columns=selected_columns)


def load_raw(
    path: str | Path = INPUT_FILE,
    selected_columns: Dict[str, str] = SELECTED_COLUMNS_DICT,
) -> pd.DataFrame:
    """Read only the columns needed for the EDA."""
    logger.info("Reading %s", path)
    wanted = set(selected_columns.keys())
    return pd.read_csv(path, usecols=lambda c: c in wanted)


def prepare(
    df: pd.DataFrame,
    selected_columns: Dict[str, str] = SELECTED_COLUMNS_DICT,
) -> pd.DataFrame:
    """Normalise dtypes and derive the temporal helper columns."""
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
    """Apply the dashboard filters; ``None``/empty means no restriction."""
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
    """Split rows into bounding-box inliers and outliers."""
    geo = df.dropna(subset=["lat", "lon"])
    mask = (geo["lat"].between(BBOX["lat_lo"], BBOX["lat_hi"])
            & geo["lon"].between(BBOX["lon_lo"], BBOX["lon_hi"]))
    return geo[mask].copy(), geo[~mask].copy()


# Static plots — volumes

def top_crime_types(df: pd.DataFrame, n: int = 15) -> Figure:
    top = df["crime"].value_counts().head(n).sort_values()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top.index, top.values, color=BLUE, height=0.6)
    ax.set_xlabel("Incidents")
    ax.set_title(f"Top {n} Crime Types")
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
    ax.set_xlabel("Incidents")
    ax.set_title("Incidents by Crime Category")
    ax.xaxis.set_major_formatter(_THOUSANDS)
    ax.grid(axis="x")
    ax.spines["bottom"].set_visible(False)
    for i, v in enumerate(cat.values):
        ax.text(v * 1.005, i, f"{v:,}", va="center", fontsize=8, color=GRAY)
    return fig


def annual_volume(df: pd.DataFrame) -> Figure:
    """Annual volume with YoY labels: red when rising (bad), muted when falling."""
    annual = df.groupby("year").size()
    yoy = annual.pct_change() * 100
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(annual.index, annual.values, color=NAVY, width=0.6, zorder=2)
    ax.set_ylabel("Incidents")
    ax.set_title("Annual Incident Volume")
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
            label="3-month moving avg")
    ax.fill_between(ts.index, ma3.values, alpha=0.1, color=BLUE, zorder=1)
    ax.set_ylabel("Incidents")
    ax.set_title("Monthly Incidents")
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
    ax.set_ylabel("Incidents")
    ax.set_title("Monthly Incidents by Crime Category")
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
                cbar_kws={"label": "Incidents", "shrink": 0.6})
    ax.set_title("Seasonality — Month x Day of Week")
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
                cbar_kws={"label": "Incidents", "shrink": 0.6})
    ax.set_title("Hourly Rhythm — Hour x Day of Week")
    ax.set_xlabel("Hour (0-23)")
    ax.set_ylabel("")
    return fig


def incidents_by_hour(df: pd.DataFrame) -> Figure:
    hc = df["hour"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.bar(hc.index, hc.values, color=BLUE, width=0.7)
    ax.set_xlabel("Hour of Day (0-23)")
    ax.set_ylabel("Incidents")
    ax.set_title("Incidents by Hour of Day")
    ax.yaxis.set_major_formatter(_THOUSANDS)
    ax.grid(axis="y")
    return fig


def intraday_by_category(df: pd.DataFrame) -> Figure:
    ph = df.groupby(["crime_cat", "hour"]).size().unstack(fill_value=0)
    ph_norm = ph.div(ph.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(13, 5))
    sns.heatmap(ph_norm, ax=ax, cmap=SEQ, linewidths=0.4, linecolor=BG,
                cbar_kws={"label": "Row fraction", "shrink": 0.6})
    ax.set_title("Intra-day Rhythm by Crime Category (row-normalised)")
    ax.set_xlabel("Hour (0-23)")
    ax.set_ylabel("")
    return fig

# Static plots — spatial

def spatial_outliers(geo: pd.DataFrame, geo_out: pd.DataFrame) -> Figure:
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(geo["lon"], geo["lat"], s=0.3, alpha=0.15,
               color=BLUE, rasterized=True, label="Inliers")
    if len(geo_out):
        ax.scatter(geo_out["lon"], geo_out["lat"], s=12, alpha=0.8,
                   color=BAD, zorder=5, label=f"Outliers (n={len(geo_out):,})")
    ax.set_title("Spatial Distribution — Bounding Box Filter")
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")
    ax.set_aspect("equal")
    ax.legend(markerscale=5)
    ax.grid(True, lw=0.3)
    return fig


def _add_colorbar(fig: Figure, ax, mappable, label: str):
    """Attach a themed colour-scale legend so shaded magnitudes are readable.

    Animated callers must pass a mappable with a *fixed* norm: ``_fig_to_pil``
    saves with ``bbox_inches="tight"``, so tick labels that change width
    between frames would yield GIF frames of differing size.
    """
    # A divider-backed cax tracks the axes box, so the bar keeps the plot's
    # height under ``set_aspect("equal")`` instead of overshooting it.
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
    if ax.collections:                      # empty when the KDE cannot be fit
        _add_colorbar(fig, ax, ax.collections[0], "Estimated density")
    ax.set_title("Crime Density — Overall KDE")
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
    ax.set_title("Crime Density — Top 3 Categories")
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")
    ax.set_aspect("equal")
    ax.legend()
    return fig


def h3_heatmap(geo: pd.DataFrame, resolution: int = 8) -> folium.Map:
    if geo.empty:
        raise ValueError("h3_heatmap needs at least one geolocated row.")
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
        raise ValueError("No frames to animate (empty selection?).")
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
        ax.set_title(f"Crime Location Scatter — {year}")
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
        # Seaborn renormalizes the KDE per frame, so the legend is a fixed
        # 0-1 relative scale — identical labels keep every frame the same size.
        _add_colorbar(fig, ax,
                      plt.cm.ScalarMappable(cmap=SEQ, norm=plt.Normalize(0, 1)),
                      "Relative density")
        ax.set_title(f"Crime Density — {year}")
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
        fig.suptitle(f"DBSCAN — {year}   e={EPSILON_M}m   min_pts={MIN_SAMPLES}   "
                     f"clusters: {n_clusters}   noise: {noise_pct}%",
                     fontsize=11, fontweight="bold", color=INK, y=1.01)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_aspect("equal")
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")
        ax.set_title("Cluster membership")
        noise = labels == -1
        ax.scatter(sub["lon"][noise], sub["lat"][noise],
                   s=0.3, alpha=0.2, color=GRAY)
        if n_clusters > 0:
            cmap = matplotlib.colormaps["Blues"].resampled(n_clusters)
            ax.scatter(sub["lon"][~noise], sub["lat"][~noise],
                       c=labels[~noise], cmap=cmap,
                       s=0.5, alpha=0.5, vmin=0, vmax=n_clusters)
        handles = [mpatches.Patch(color=GRAY, alpha=0.5,
                                  label=f"noise ({noise_pct}%)"),
                   mpatches.Patch(color=BLUE, alpha=0.6,
                                  label=f"{n_clusters} clusters")]
        ax.legend(handles=handles, loc="lower right")
        fig.tight_layout()
        frames.append(_fig_to_pil(fig, dpi=130))
    return _frames_to_gif(frames)


# CVEGEO (district-level) aggregation, animation and distribution

def load_districts(path: str | Path = PD_SHP_PATH) -> gpd.GeoDataFrame:
    return gpd.read_file(path)


def cvegeo_aggregate(df: pd.DataFrame, pd_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Aggregate incident counts per district (CVEGEO) and year."""

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
        ax.set_title(f"Crime Polygons — {year}")
        frames.append(_fig_to_pil(fig, dpi=120))
    return _frames_to_gif(frames)


def _count_distribution(gdf: gpd.GeoDataFrame, level: str) -> Figure:
    """Shared hist / ECDF / boxplot of yearly counts for any aggregation level."""
    years = sorted(gdf["year"].dropna().unique())
    palette = sns.light_palette(BLUE, n_colors=len(years) + 1)[1:]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), dpi=100)
    for year, color in zip(years, palette):
        sub = gdf.loc[gdf["year"] == year, "cnt"].dropna()
        axes[0].hist(sub, bins=35, density=True, alpha=0.45, color=color,
                     label=int(year))
    axes[0].set_xlabel("cnt")
    axes[0].set_ylabel("Density")
    axes[0].set_title(f"{level} Count Distribution by Year")
    axes[0].legend(fontsize=8)
    for year, color in zip(years, palette):
        sub = np.sort(gdf.loc[gdf["year"] == year, "cnt"].dropna().values)
        if len(sub) == 0:
            continue
        ecdf = np.arange(1, len(sub) + 1) / len(sub)
        axes[1].plot(sub, ecdf, color=color, linewidth=2, label=int(year))
    axes[1].set_xlabel("cnt")
    axes[1].set_ylabel(r"$F_n(x)$")
    axes[1].set_title(f"{level} Empirical CDF by Year")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3, linestyle="--")
    box_style = {"boxes": MUTED, "whiskers": MUTED, "medians": SKY, "caps": MUTED}
    gdf.dropna(subset=["cnt"]).boxplot(column="cnt", by="year", ax=axes[2],
                                       patch_artist=True, color=box_style)
    for patch, color in zip(axes[2].patches, palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    axes[2].set_title(f"{level} Boxplot by Year")
    axes[2].set_xlabel("Year")
    axes[2].set_ylabel("cnt")
    fig.suptitle("")
    fig.tight_layout()
    return fig


def cvegeo_distribution(crime_gdf: gpd.GeoDataFrame) -> Figure:
    return _count_distribution(crime_gdf, "District (CVEGEO)")


# Street-segment level: split at intersections, aggregate, animate, distribute

def load_streets(path: str | Path = STREETS_SHP_PATH) -> gpd.GeoDataFrame:
    """Read the street network and split it at intersections (slow, cache it)."""
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
    """Snap incidents to the nearest street segment and count per segment-year."""
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
    """Yearly animation of street segments shaded by incident count."""
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
        # Added unconditionally (vmax is global): a legend that appeared only
        # on non-empty years would resize those frames.
        _add_colorbar(fig, ax,
                      plt.cm.ScalarMappable(cmap=SEQ,
                                            norm=plt.Normalize(0, vmax)),
                      "Incidents per segment")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")
        ax.set_title(f"Street Segments by Crime Count — {year}")
        frames.append(_fig_to_pil(fig, dpi=120))
    return _frames_to_gif(frames)


def street_distribution(street_gdf: gpd.GeoDataFrame) -> Figure:
    return _count_distribution(street_gdf, "Street Segment")


# Export helpers

def fig_to_png_bytes(fig: Figure, dpi: int = 150) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor=BG)
    return buf.getvalue()


def export_all(artifacts: dict[str, bytes | str], out_dir: str | Path) -> list[Path]:
    """Write every cached artefact (PNG/GIF/HTML bytes or str) to ``out_dir``."""
    out = Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for name, data in artifacts.items():
        path = out / name
        mode, payload = ("w", data) if isinstance(data, str) else ("wb", data)
        with open(path, mode) as fh:
            fh.write(payload)
        written.append(path)
        logger.info("Exported %s", path)
    return written


### -------------------------------------------------------------------------------
### Main (cleans up the raw data source) ------------------------------------------

if __name__ == "__main__":

    from src.utils.prod import init_logger

    logger = init_logger()

    INPUT_PATH = "./data/raw/carpetasFGJ.csv"
    OUTPUT_PATH = "./data/clean/carpetasFGJ.csv"

    logger.info(f"Working on cleaning the file: {INPUT_PATH}")
    try:
        # ``prepare`` renames internally, so load -> prepare is the whole chain
        prepare(
            load_raw(INPUT_PATH, SELECTED_COLUMNS_DICT),
            SELECTED_COLUMNS_DICT,
        ).to_csv(OUTPUT_PATH, index=False)

        logger.info(f"File {INPUT_PATH} cleaned and saved in {OUTPUT_PATH}.")
    except Exception as e:
        logger.error(f"Error cleaning the file {INPUT_PATH}. \n {e}")
