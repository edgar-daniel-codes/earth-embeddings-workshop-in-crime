### Summer Internship - Earth Embeddings
### Visuals - Muestra animada de resaltado (versión en español)
### By Edgar Daniel

"""
GIF animado que empareja el espacio físico con el espacio de features del
embedding, con todos los textos en español.

Cada cuadro resalta exactamente una de las calles de la CDMX con mayor
incidencia: un punto rojo en su ubicación sobre un mapa base CartoDB
Positron con el límite de la CDMX y la red vial superpuestos (izquierda), y
un punto rojo en su posición en cada una de las tres dispersiones por pares
del embedding a la derecha. Todo el fondo — mapa base, límite, red vial y
dispersiones — se dibuja una sola vez y se reutiliza en todos los cuadros;
solo cambia el punto resaltado por cuadro.

Se construyen dos variantes: PCA (PC1-PC2, PC1-PC3, PC2-PC3) y SVD
(SV1-SV2, SV1-SV3, SV2-SV3), compartiendo la misma orquestación.

Copia en español de ``src.visuals.flash_highlight_showcase``: la lógica es
idéntica; los GIF se guardan con el sufijo ``_esp`` para no sobrescribir las
versiones en inglés.

Ejecutar:  python -m src.visuals.flash_highlight_showcase_esp
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

from src.eda.core_esp import _frames_to_gif
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
    REPO_ROOT / "docs" / "resources" / "unsupervised" / "flash_highlight_2023_esp.gif"
)
OUT_PATH_SVD = (
    REPO_ROOT / "docs" / "resources" / "unsupervised" / "flash_highlight_svd_2023_esp.gif"
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

MAP_CRS = 3857   # Web Mercator, requerido por los mosaicos del mapa base
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
    """Carga el conjunto de entrenamiento y lo filtra a un solo año.

    El resultado se reindexa a un rango contiguo para que las filas del mapa y
    las del PCA permanezcan alineadas por posición.
    """
    gdf = gpd.read_parquet(path)
    gdf = gdf[gdf["year"] == year].reset_index(drop=True)
    if gdf.empty:
        raise ValueError(f"No hay registros para year={year} en {path}.")
    return gdf


def top_incidence_index(gdf: gpd.GeoDataFrame, top_n: int = TOP_N,
                        target_col: str = "y_cnt") -> pd.Index:
    """Regresa el índice posicional de los ``top_n`` registros con mayor
    ``target_col``, ordenados de mayor a menor."""
    return gdf[target_col].nlargest(top_n).index


def load_boundary(path: str | Path = BOUNDARY_SHP_PATH) -> gpd.GeoDataFrame:
    """Carga el límite de la entidad CDMX, reproyectado a Web Mercator."""
    return gpd.read_file(path).to_crs(MAP_CRS)


### -------------------------------------------------------------------------------
### Rendering -------------------------------------------------------------------


def _strip_axis(ax: plt.Axes) -> None:
    """Quita las marcas, fija la relación de aspecto y oculta los bordes."""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_visible(False)


def _render_frame(fig: plt.Figure, dpi: int = FRAME_DPI) -> PILImage.Image:
    """Renderiza el estado actual de una figura persistente a una imagen PIL.

    A diferencia de los constructores de GIF de ``eda.core_esp``, esta figura no
    se cierra ni se reconstruye: la misma figura y los mismos ejes se reutilizan
    en cada cuadro de la animación, de modo que solo se redibujan los artistas
    que cambiaron.
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
    """Construye la figura persistente: un eje grande de mapa con el mapa base
    CartoDB Positron, el límite de la CDMX y la red vial dibujados una sola vez,
    y tres ejes de dispersión del embedding (uno por entrada de ``pairs``) cuyo
    fondo también se dibuja una sola vez — todo el fondo queda intacto entre
    cuadros.

    ``gdf_map`` y ``boundary`` ya deben estar en ``MAP_CRS`` (Web Mercator), que
    es lo que requieren los mosaicos del mapa base.

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

    # Fija la vista a la extensión del límite (con margen) *después* de graficar,
    # ya que .plot() reescala los ejes a lo que acaba de dibujar; contextily
    # descarga entonces los mosaicos que corresponden a esta extensión final.
    minx, miny, maxx, maxy = boundary.total_bounds
    pad_x = (maxx - minx) * MAP_BOUNDARY_PAD_FRAC
    pad_y = (maxy - miny) * MAP_BOUNDARY_PAD_FRAC
    ax_map.set_xlim(minx - pad_x, maxx + pad_x)
    ax_map.set_ylim(miny - pad_y, maxy + pad_y)

    cx.add_basemap(ax_map, source=MAP_BASEMAP_SOURCE, crs=gdf_map.crs)

    ax_map.set_title("Red vial de la CDMX", fontsize=12, pad=10)
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
        f"Espacio físico vs. espacio del embedding — focos de incidencia, {year}",
        fontsize=13,
    )
    fig.subplots_adjust(left=0.05, right=0.90, bottom=0.06, top=0.90)
    add_colorbar(fig, norm, cmap, "Incidentes (y_cnt)")

    return fig, ax_map, embed_axes


def _street_centroid(gdf: gpd.GeoDataFrame, street_idx: int) -> tuple[float, float]:
    """Regresa el centroide (x, y) de una calle, solo para colocar el marcador.

    La advertencia de aproximación en CRS geográfico que lanza geopandas es
    esperada aquí y se suprime.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        centroid = gdf.loc[street_idx, "geometry"].centroid
    return centroid.x, centroid.y


def _update_map_highlight(
    ax_map: plt.Axes, gdf: gpd.GeoDataFrame, street_idx: int,
    rank: int, top_n: int, prev_artist,
):
    """Actualiza en el lugar el punto resaltado del mapa sobre la red vial
    cacheada, sin redibujar el fondo. Regresa el nuevo artista.
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
        f"Posición {rank}/{top_n} — {cnt} incidentes\nCVEGEO {cvegeo}",
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
    """Construye y guarda el GIF de resaltado animado.

    Un cuadro por calle de mayor incidencia: un acercamiento a esa calle en el
    mapa y un punto rojo marcando su posición en cada panel de dispersión del
    embedding en caché.

    Parameters
    ----------
    parquet_path : str | Path
        Parquet del conjunto de entrenamiento con columnas de embedding,
        ``y_cnt`` y geometría.
    year : int
        Año al cual filtrar (el parquet abarca varios años).
    top_n : int
        Número de calles con mayor ``y_cnt`` a recorrer, una por cuadro.
    duration : int
        Milisegundos por cuadro.
    out_path : str | Path
        Ruta destino del GIF.
    compute_fn : callable
        Función de embedding con el contrato de ``compute_pca``/``compute_svd``:
        ``(X, n_components) -> (embedding_df, explained_variance_ratio)``.
    pairs : list[tuple[str, str]]
        Pares de columnas a graficar, acordes a la salida de ``compute_fn``.

    Returns
    -------
    Path
        ``out_path``, una vez escrito el GIF.
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

    logger.info(f"Construyendo el GIF de resaltado con PCA para year={YEAR}...")
    try:
        path = build_showcase_gif(compute_fn=compute_pca, pairs=PC_PAIRS, out_path=OUT_PATH)
        logger.info(f"GIF con PCA guardado en {path}")
    except Exception as e:
        logger.error(f"Error al construir el GIF de resaltado con PCA. {e}")

    logger.info(f"Construyendo el GIF de resaltado con SVD para year={YEAR}...")
    try:
        path = build_showcase_gif(compute_fn=compute_svd, pairs=SVD_PAIRS, out_path=OUT_PATH_SVD)
        logger.info(f"GIF con SVD guardado en {path}")
    except Exception as e:
        logger.error(f"Error al construir el GIF de resaltado con SVD. {e}")
