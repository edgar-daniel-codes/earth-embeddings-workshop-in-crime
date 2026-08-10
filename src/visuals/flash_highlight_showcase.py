### Summer Internship - Earth Embeddings
### Visuals - Flash Highlight Showcase
### By Edgar Daniel

"""
Animated GIF showcase pairing physical space and embedding feature space.

Each frame highlights exactly one of the top-incidence CDMX streets: a
red point at its location on a CartoDB Positron basemap with the CDMX
boundary and street network overlaid (left), and a red point at its
position in each of the three pairwise embedding scatter plots on the
right. Every background — the basemap, boundary, street network and
embedding scatter — is rendered once and reused across frames; only the
highlight point changes per frame.

Two variants are built: PCA (PC1-PC2, PC1-PC3, PC2-PC3) and SVD
(SV1-SV2, SV1-SV3, SV2-SV3), sharing the same orchestration logic.

Run:  python -m src.visuals.flash_highlight_showcase
"""

### -------------------------------------------------------------------------------
### Libraries and parameters-------------------------------------------------------

from __future__ import annotations

import io
import sys
import warnings
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import contextily as cx
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image as PILImage

from src.eda.core import _frames_to_gif
from src.unsupervised.dim_reduction import compute_pca, compute_svd
from src.utils.dim_reduction import add_colorbar, compute_colors, global_color_norm
from src.utils.style import DEFAULT as PALETTE
from src.utils.style import apply_theme

REPO_ROOT = Path(__file__).resolve().parents[2]
PARQUET_PATH = (
    REPO_ROOT / "data" / "proc" / "training_sets" / "cdmx_asaltos.parquet"
)
BOUNDARY_SHP_PATH = REPO_ROOT / "data" / "spatial" / "pd" / "09ent.shp"
OUT_PATH = (
    REPO_ROOT / "docs" / "resources" / "unsupervised" / "flash_highlight_2023.gif"
)
OUT_PATH_SVD = (
    REPO_ROOT / "docs" / "resources" / "unsupervised" / "flash_highlight_svd_2023.gif"
)

YEAR = 2023
TOP_N = 20
FEATURE_COLUMNS = [f"A{i:02d}" for i in range(64)]
PC_PAIRS = [("PC1", "PC2"), ("PC1", "PC3"), ("PC2", "PC3")]
SVD_PAIRS = [("SV1", "SV2"), ("SV1", "SV3"), ("SV2", "SV3")]

FRAME_DURATION_MS = 600
FRAME_DPI = 120
FIGSIZE = (11, 8)
MAP_WIDTH_RATIO = 2.4

MAP_CRS = 3857   # Web Mercator, required by the contextily basemap tiles
MAP_BASEMAP_SOURCE = cx.providers.CartoDB.Positron
MAP_BOUNDARY_PAD_FRAC = 0.03

NETWORK_COLOR = "#4B5563"
NETWORK_LINEWIDTH = 0.25
BOUNDARY_COLOR = PALETTE.navy
BOUNDARY_LINEWIDTH = 1.6

BACKGROUND_MARKER_SIZE = 10
HIGHLIGHT_MARKER_SIZE = 90
HIGHLIGHT_COLOR = PALETTE.bad

### -------------------------------------------------------------------------------
### Data preparation ----------------------------------------------------------------


def load_year_data(path: str | Path = PARQUET_PATH, year: int = YEAR) -> gpd.GeoDataFrame:
    """Load the training set and filter to a single year.

    The result is reindexed to a contiguous range so map rows and PCA
    rows stay aligned by position.
    """
    gdf = gpd.read_parquet(path)
    gdf = gdf[gdf["year"] == year].reset_index(drop=True)
    if gdf.empty:
        raise ValueError(f"No rows for year={year} in {path}.")
    return gdf


def top_incidence_index(gdf: gpd.GeoDataFrame, top_n: int = TOP_N,
                        target_col: str = "y_cnt") -> pd.Index:
    """Return the positional index of the ``top_n`` rows by ``target_col``,
    ordered from the highest value down."""
    return gdf[target_col].nlargest(top_n).index


def load_boundary(path: str | Path = BOUNDARY_SHP_PATH) -> gpd.GeoDataFrame:
    """Load the CDMX entidad boundary, reprojected to Web Mercator."""
    return gpd.read_file(path).to_crs(MAP_CRS)


### -------------------------------------------------------------------------------
### Rendering -------------------------------------------------------------------


def _strip_axis(ax: plt.Axes) -> None:
    """Remove ticks, lock the aspect ratio, and hide spines on a map axis."""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_visible(False)


def _render_frame(fig: plt.Figure, dpi: int = FRAME_DPI) -> PILImage.Image:
    """Render the current state of a persistent figure to a PIL image.

    Unlike ``eda.core``'s per-frame GIF builders, this figure is not
    closed or rebuilt: the same figure and axes are reused across every
    frame of this animation, so only the changed artists are redrawn.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor=PALETTE.background)
    buf.seek(0)
    return PILImage.open(buf).copy()


def _build_base_figure(
    gdf_map: gpd.GeoDataFrame, boundary: gpd.GeoDataFrame,
    embedding: pd.DataFrame, colors: np.ndarray, cmap, norm, year: int,
    pairs: list[tuple[str, str]] = PC_PAIRS,
):
    """Build the persistent figure: a large map axis with the CartoDB
    Positron basemap, the CDMX boundary and the street network drawn
    once, and three embedding scatter axes (one per entry in ``pairs``)
    whose background is also drawn once — every background is left
    unchanged across frames.

    ``gdf_map`` and ``boundary`` must already be in ``MAP_CRS`` (Web
    Mercator), which the basemap tiles require.

    Returns
    -------
    tuple
        ``(fig, ax_map, embed_axes)``.
    """
    fig = plt.figure(figsize=FIGSIZE)
    gs = fig.add_gridspec(len(pairs), 2, width_ratios=[MAP_WIDTH_RATIO, 1],
                          hspace=0.4, wspace=0.3)
    ax_map = fig.add_subplot(gs[:, 0])
    embed_axes = [fig.add_subplot(gs[i, 1]) for i in range(len(pairs))]

    gdf_map.plot(ax=ax_map, color=NETWORK_COLOR, linewidth=NETWORK_LINEWIDTH)
    boundary.boundary.plot(ax=ax_map, color=BOUNDARY_COLOR,
                           linewidth=BOUNDARY_LINEWIDTH)

    # Fix the view to the boundary's extent (with padding) *after* plotting,
    # since .plot() autoscales the axes to whatever it just drew; contextily
    # then fetches tiles matching this final, explicit extent.
    minx, miny, maxx, maxy = boundary.total_bounds
    pad_x = (maxx - minx) * MAP_BOUNDARY_PAD_FRAC
    pad_y = (maxy - miny) * MAP_BOUNDARY_PAD_FRAC
    ax_map.set_xlim(minx - pad_x, maxx + pad_x)
    ax_map.set_ylim(miny - pad_y, maxy + pad_y)

    cx.add_basemap(ax_map, source=MAP_BASEMAP_SOURCE, crs=gdf_map.crs)

    ax_map.set_title("CDMX Street Network", fontsize=12, pad=10)
    _strip_axis(ax_map)

    for ax, (px, py) in zip(embed_axes, pairs):
        ax.scatter(
            embedding[px], embedding[py], c=colors,
            s=BACKGROUND_MARKER_SIZE, linewidths=0, rasterized=True,
        )
        ax.set_title(f"{px} vs {py}", fontsize=10, pad=4)
        ax.set_xlabel(px, fontsize=8)
        ax.set_ylabel(py, fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        f"Physical space vs. embedding space — incident hotspots, {year}",
        fontsize=13,
    )
    fig.subplots_adjust(left=0.05, right=0.90, bottom=0.06, top=0.90)
    add_colorbar(fig, norm, cmap, "Incidents (y_cnt)")

    return fig, ax_map, embed_axes


def _street_centroid(gdf: gpd.GeoDataFrame, street_idx: int) -> tuple[float, float]:
    """Return the (x, y) centroid of one street, for marker placement only.

    The geographic-CRS approximation warning geopandas raises for this
    is expected here and suppressed.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        centroid = gdf.loc[street_idx, "geometry"].centroid
    return centroid.x, centroid.y


def _update_map_highlight(
    ax_map: plt.Axes, gdf: gpd.GeoDataFrame, street_idx: int,
    rank: int, top_n: int, prev_artist,
):
    """Update the map's highlight point in place on the cached street
    network, without redrawing the background. Returns the new artist.
    """
    if prev_artist is not None:
        prev_artist.remove()

    cx, cy = _street_centroid(gdf, street_idx)
    artist = ax_map.scatter(
        [cx], [cy], s=HIGHLIGHT_MARKER_SIZE, color=HIGHLIGHT_COLOR,
        edgecolors="white", linewidths=0.8, zorder=5,
    )

    cnt = int(gdf.loc[street_idx, "y_cnt"])
    cvegeo = gdf.loc[street_idx, "CVEGEO"]
    ax_map.set_title(
        f"Rank {rank}/{top_n} — {cnt} incidents\nCVEGEO {cvegeo}",
        fontsize=11, pad=10,
    )
    return artist


### -------------------------------------------------------------------------------
### Orchestration -------------------------------------------------------------------


def build_showcase_gif(
    parquet_path: str | Path = PARQUET_PATH,
    year: int = YEAR,
    top_n: int = TOP_N,
    duration: int = FRAME_DURATION_MS,
    out_path: str | Path = OUT_PATH,
    compute_fn=compute_pca,
    pairs: list[tuple[str, str]] = PC_PAIRS,
) -> Path:
    """Build and save the flash-highlight showcase GIF.

    One frame per top-incidence street: a close-up of that street on the
    map, and a red point marking its position in each cached embedding
    scatter panel.

    Parameters
    ----------
    parquet_path : str | Path
        Training-set parquet with embedding columns, ``y_cnt`` and geometry.
    year : int
        Year to filter to (the parquet spans multiple years).
    top_n : int
        Number of highest-``y_cnt`` streets to cycle through, one per frame.
    duration : int
        Milliseconds per frame.
    out_path : str | Path
        Destination GIF path.
    compute_fn : callable
        Embedding function with the ``compute_pca``/``compute_svd`` contract:
        ``(X, n_components) -> (embedding_df, explained_variance_ratio)``.
    pairs : list[tuple[str, str]]
        Column pairs to plot, matching ``compute_fn``'s output columns.

    Returns
    -------
    Path
        ``out_path``, after the GIF has been written.
    """
    apply_theme()

    gdf = load_year_data(parquet_path, year)
    gdf_map = gdf.to_crs(MAP_CRS)
    boundary = load_boundary()
    top_idx = top_incidence_index(gdf, top_n)

    embedding, _ = compute_fn(gdf[FEATURE_COLUMNS], n_components=3)
    norm = global_color_norm(gdf, "y_cnt")
    cmap = PALETTE.bad_cmap()
    colors = compute_colors(gdf["y_cnt"], norm, cmap)

    fig, ax_map, embed_axes = _build_base_figure(
        gdf_map, boundary, embedding, colors, cmap, norm, year, pairs
    )
    highlight_artists: list = [None] * len(embed_axes)
    map_highlight_artist = None

    frames = []
    for rank, street_idx in enumerate(top_idx, start=1):
        map_highlight_artist = _update_map_highlight(
            ax_map, gdf_map, street_idx, rank, top_n, map_highlight_artist
        )

        for j, (ax, (px, py)) in enumerate(zip(embed_axes, pairs)):
            if highlight_artists[j] is not None:
                highlight_artists[j].remove()
            highlight_artists[j] = ax.scatter(
                [embedding.loc[street_idx, px]], [embedding.loc[street_idx, py]],
                s=HIGHLIGHT_MARKER_SIZE, color=HIGHLIGHT_COLOR,
                edgecolors="white", linewidths=0.8, zorder=5,
            )

        frames.append(_render_frame(fig))

    plt.close(fig)
    gif_bytes = _frames_to_gif(frames, duration=duration)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(gif_bytes)
    return out_path


### -------------------------------------------------------------------------------
### Main --------------------------------------------------------------------------

if __name__ == "__main__":
    from src.utils.prod import init_logger

    logger = init_logger()

    logger.info(f"Building PCA flash-highlight showcase GIF for year={YEAR}...")
    try:
        path = build_showcase_gif(compute_fn=compute_pca, pairs=PC_PAIRS, out_path=OUT_PATH)
        logger.info(f"Saved PCA showcase GIF to {path}")
    except Exception as e:
        logger.error(f"Error building the PCA flash-highlight showcase GIF. {e}")

    logger.info(f"Building SVD flash-highlight showcase GIF for year={YEAR}...")
    try:
        path = build_showcase_gif(compute_fn=compute_svd, pairs=SVD_PAIRS, out_path=OUT_PATH_SVD)
        logger.info(f"Saved SVD showcase GIF to {path}")
    except Exception as e:
        logger.error(f"Error building the SVD flash-highlight showcase GIF. {e}")
