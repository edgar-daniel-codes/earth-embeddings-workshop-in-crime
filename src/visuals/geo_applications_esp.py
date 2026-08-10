### Summer Internship - Earth Embeddings
### Visuals - Aplicaciones geográficas / mapas de riesgo (versión en español)
### By Edgar Daniel

"""
Mapas de riesgo a nivel segmento de calle para tres aplicaciones, con
todos los textos visibles (títulos y leyendas) en español:

* Clustering: asignaciones de clúster de k-means (2022-2024) mapeadas a un
  nivel de riesgo mediante un mapeo manual clúster -> nivel por año.
* Regresión: el modelo campeón de regresión persistido, aplicado a datos
  nuevos y clasificado por umbrales fijos sobre el conteo esperado
  (``y_cnt_pred``).
* Clasificación: el modelo campeón de clasificación persistido, aplicado a
  los mismos datos nuevos y clasificado por umbrales fijos sobre la
  probabilidad de incidente (``y_ind_proba``).

Copia en español de ``src.visuals.geo_applications``: la lógica es idéntica;
las figuras se guardan con el sufijo ``_esp`` para no sobrescribir las
versiones en inglés.

Ejecutar:  python -m src.visuals.geo_applications_esp
"""

### -------------------------------------------------------------------------------
### Libraries and parameters-------------------------------------------------------

from __future__ import annotations

import logging
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from src.supervised.predict import load_champion, predict

logger = logging.getLogger("geo_applications_esp")

### -------------------------------------------------------------------------------
### Paths -------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = REPO_ROOT / "docs" / "resources" / "unsupervised"

CLUSTER_ASSIGNMENTS_PATH = (
    REPO_ROOT / "data" / "infer" / "kmeans_cluster_assignments.parquet"
)
STREETS_SHP_PATH = REPO_ROOT / "data" / "proc" / "pd" / "09e_corner_split.shp"

NEW_DATA_YEAR = 2025
NEW_DATA_PATH = (
    REPO_ROOT / "data" / "proc" / "training_sets" / "cdmx_asaltos_2025.parquet"
)
SCORES_OUTPUT_PATH = (
    REPO_ROOT / "data" / "infer" / f"predict_risk_{NEW_DATA_YEAR}.parquet"
)

### -------------------------------------------------------------------------------
### Cluster-plot parameters ---------------------------------------------------------

SELECTED_K = 4
YEARS = [2022, 2023, 2024]
SAMPLE_MUNS = ["003", "014", "016", "010"]

MAPPINGS_BY_YEAR = [
    {0: 1, 1: 2, 2: 2, 3: 3},
    {0: 1, 1: 3, 2: 2, 3: 2},
    {0: 1, 1: 2, 2: 2, 3: 3},
]

RISK_COLORS = {
    1: "#A6B1BB",
    2: "#4C6B87",
    3: "#C0392B",
}
RISK_LABELS = {1: "Bajo", 2: "Medio", 3: "Alto"}

### -------------------------------------------------------------------------------
### Fixed risk thresholds -----------------------------------------------------------

# Cortes sobre y_ind_proba, en la escala natural de probabilidad [0, 1].
CLASSIFICATION_THRESHOLDS = (0.2, 0.50)
CLASSIFICATION_RISK_LABELS = {
    1: "Bajo (<0.20)",
    2: "Medio (0.20-0.5)",
    3: "Alto (>=0.5)",
}

# Cortes sobre y_cnt_pred, escalados x1/100 desde cortes de conteo de 1 y 5
# para coincidir con el rango de salida del modelo campeón (máximo observado
# ~0.98 en datos reales).
REGRESSION_THRESHOLDS = (0.01, 0.05)
REGRESSION_RISK_LABELS = {
    1: "Bajo (<0.01)",
    2: "Medio (0.01-0.05)",
    3: "Alto (>=0.05)",
}

# Títulos de tarea usados en los encabezados de las figuras.
REGRESSION_TASK_TITLE = "Regresión"
CLASSIFICATION_TASK_TITLE = "Clasificación"

### -------------------------------------------------------------------------------
### Shared risk-palette helpers -----------------------------------------------------


def _risk_cmap_norm(risk_colors: dict = RISK_COLORS):
    """Construye un colormap categórico y su norma de frontera a partir de un
    mapeo nivel -> color.

    Parameters
    ----------
    risk_colors : dict
        Mapeo de nivel de riesgo (entero) a color hexadecimal.

    Returns
    -------
    tuple
        ``(cmap, norm)`` listos para pasarse a ``GeoDataFrame.plot``.
    """
    tiers = sorted(risk_colors)
    cmap = mcolors.ListedColormap([risk_colors[t] for t in tiers])
    boundaries = np.array([tiers[0] - 0.5, *[t + 0.5 for t in tiers]])
    norm = mcolors.BoundaryNorm(boundaries, cmap.N)
    return cmap, norm


def _risk_legend_handles(risk_colors: dict = RISK_COLORS,
                         risk_labels: dict = RISK_LABELS) -> list[Patch]:
    """Construye los parches de leyenda para los niveles de riesgo."""
    return [Patch(facecolor=risk_colors[t], edgecolor="none", label=risk_labels[t])
            for t in sorted(risk_colors)]


def _strip_axis(ax: plt.Axes) -> None:
    """Quita las marcas, fija la relación de aspecto y oculta los bordes."""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_visible(False)


def _save(fig: plt.Figure, out_path: str | Path, dpi: int) -> None:
    """Guarda una figura en ``out_path``, creando los directorios necesarios."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


### -------------------------------------------------------------------------------
### Cluster-derived risk ------------------------------------------------------------


def load_cluster_assignments(path: str | Path = CLUSTER_ASSIGNMENTS_PATH) -> pd.DataFrame:
    """Carga las asignaciones de k-means (columnas: year, CVEGEO, k, cluster)."""
    return pd.read_parquet(path)


def load_street_geometry(path: str | Path = STREETS_SHP_PATH) -> gpd.GeoDataFrame:
    """Carga la geometría de los segmentos de calle con su clave y nombre de alcaldía."""
    return gpd.read_file(path)[["CVEGEO", "MUN", "NOM_MUN", "geometry"]]


def build_mun_name_lookup(gdf_geom: gpd.GeoDataFrame) -> dict[str, str]:
    """Construye una tabla de búsqueda clave de alcaldía -> nombre de alcaldía."""
    return (gdf_geom[["MUN", "NOM_MUN"]]
            .drop_duplicates()
            .set_index("MUN")["NOM_MUN"]
            .to_dict())


def plot_cluster_risk_sample_grid(
    gdf_clusters: pd.DataFrame,
    gdf_geom: gpd.GeoDataFrame,
    years: list[int],
    mappings: list[dict],
    muns: list[str],
    k: int,
    cmap, norm,
    out_path: str | Path,
) -> plt.Figure:
    """Grafica el riesgo derivado de clústeres para alcaldías de muestra a lo
    largo de los años.

    Una fila por año, una columna por alcaldía en ``muns``. ``mappings`` da el
    mapeo id de clúster -> nivel de riesgo de cada entrada de ``years``, en el
    mismo orden.

    Raises
    ------
    ValueError
        Si ``years`` y ``mappings`` tienen longitudes distintas.
    """
    if len(years) != len(mappings):
        raise ValueError(
            f"years ({len(years)}) y mappings ({len(mappings)}) deben tener "
            f"la misma longitud.")

    n_rows, n_cols = len(years), len(muns)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 3.2 * n_rows))
    axes = np.atleast_2d(axes).reshape(n_rows, n_cols)

    for i, (year, mapping) in enumerate(zip(years, mappings)):
        for j, mun in enumerate(muns):
            ax = axes[i, j]
            gdf_plot = gpd.GeoDataFrame(
                gdf_clusters
                .query("k == @k and year == @year")
                .drop(columns=["k"])
                .merge(gdf_geom, on="CVEGEO", how="left")
                .query("MUN == @mun")
                .assign(cluster=lambda x: x["cluster"].map(mapping)),
                geometry="geometry", crs=gdf_geom.crs,
            )
            gdf_plot.plot(
                column="cluster", cmap=cmap, norm=norm, ax=ax, linewidth=1.4
            )

            if i == 0:
                label = gdf_plot["NOM_MUN"].iloc[0] if len(gdf_plot) else mun
                ax.set_title(label, fontsize=11, pad=8)
            if j == 0:
                ax.set_ylabel(
                    str(year), fontsize=11, rotation=0,
                    ha="right", va="center", labelpad=24,
                )
            _strip_axis(ax)

    fig.legend(
        handles=_risk_legend_handles(), loc="lower center", ncol=3,
        frameon=False, fontsize=11, bbox_to_anchor=(0.5, -0.02),
        handlelength=1.2, handleheight=1.2, columnspacing=1.5,
        title="Nivel de riesgo",
    )
    fig.tight_layout(rect=[0, 0.03, 1, 1])

    _save(fig, out_path, dpi=360)
    return fig


def plot_cluster_risk_full_cdmx(
    gdf_clusters: pd.DataFrame,
    gdf_geom: gpd.GeoDataFrame,
    years: list[int],
    mappings: list[dict],
    k: int,
    cmap, norm,
    out_path: str | Path,
) -> plt.Figure:
    """Grafica el riesgo derivado de clústeres para toda la CDMX, un panel por año.

    Raises
    ------
    ValueError
        Si ``years`` y ``mappings`` tienen longitudes distintas.
    """
    if len(years) != len(mappings):
        raise ValueError(
            f"years ({len(years)}) y mappings ({len(mappings)}) deben tener "
            f"la misma longitud.")

    fig, axes = plt.subplots(1, len(years), figsize=(5.5 * len(years), 6))
    axes = np.atleast_1d(axes)

    for ax, year, mapping in zip(axes, years, mappings):
        gdf_plot = gpd.GeoDataFrame(
            gdf_clusters
            .query("k == @k and year == @year")
            .drop(columns=["k"])
            .merge(gdf_geom, on="CVEGEO", how="left")
            .assign(cluster=lambda x: x["cluster"].map(mapping)),
            geometry="geometry", crs=gdf_geom.crs,
        )
        gdf_plot.plot(
            column="cluster", cmap=cmap, norm=norm, ax=ax, linewidth=0.5
        )
        ax.set_title(str(year), fontsize=12, pad=8)
        _strip_axis(ax)

    fig.legend(
        handles=_risk_legend_handles(), loc="lower center", ncol=3,
        frameon=False, fontsize=11, bbox_to_anchor=(0.5, -0.04),
        handlelength=1.2, handleheight=1.2, columnspacing=1.5,
        title="Nivel de riesgo",
    )
    fig.suptitle("Riesgo derivado de clústeres — CDMX completa", fontsize=13)
    fig.tight_layout(rect=[0, 0.06, 1, 0.94])

    _save(fig, out_path, dpi=300)
    return fig


### -------------------------------------------------------------------------------
### Regression / classification champion risk -------------------------------------


def bucket_fixed(values: pd.Series, thresholds: tuple[float, float]) -> pd.Series:
    """Agrupa valores continuos en 3 niveles de riesgo con umbrales fijos.

    Parameters
    ----------
    values : pd.Series
        Puntaje continuo a agrupar.
    thresholds : tuple[float, float]
        Cortes ``(low_hi, med_hi)``: los valores por debajo de ``low_hi`` van al
        nivel 1, ``[low_hi, med_hi)`` al nivel 2 y ``>= med_hi`` al nivel 3.

    Returns
    -------
    pd.Series
        Niveles enteros en ``{1, 2, 3}``, indexados como ``values``.
    """
    lo, hi = thresholds
    return pd.cut(values, bins=[-np.inf, lo, hi, np.inf],
                  labels=[1, 2, 3]).astype(int)


def load_champion_labels() -> tuple[str, str]:
    """Regresa las etiquetas de los modelos campeones ``(clasificación, regresión)``."""
    _, _, clf_info = load_champion("classification")
    _, _, reg_info = load_champion("regression")
    return clf_info["label"], reg_info["label"]


def score_new_data(gdf_new: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Puntúa datos nuevos con los modelos campeones y añade los niveles de riesgo.

    Parameters
    ----------
    gdf_new : gpd.GeoDataFrame
        Datos nuevos con las columnas de features que esperan los modelos campeones.

    Returns
    -------
    gpd.GeoDataFrame
        CVEGEO, CVE_MUN, geometry, cada columna de salida de ``predict()``,
        ``risk_reg`` (desde ``y_cnt_pred``) y ``risk_clf`` (desde ``y_ind_proba``).
    """
    preds = predict(gdf_new)
    preds["risk_reg"] = bucket_fixed(preds["y_cnt_pred"], REGRESSION_THRESHOLDS)
    preds["risk_clf"] = bucket_fixed(preds["y_ind_proba"], CLASSIFICATION_THRESHOLDS)
    out = gdf_new[["CVEGEO", "CVE_MUN", "geometry"]].join(preds)
    return gpd.GeoDataFrame(out, geometry="geometry", crs=gdf_new.crs)


def plot_scored_risk_sample_grid(
    preds_gdf: gpd.GeoDataFrame,
    mun_names: dict[str, str],
    muns: list[str],
    risk_col: str,
    year_label: int,
    model_label: str,
    task_title: str,
    risk_labels: dict,
    cmap, norm,
    out_path: str | Path,
) -> plt.Figure:
    """Grafica el riesgo del modelo campeón para alcaldías de muestra, en una fila.

    Parameters
    ----------
    preds_gdf : gpd.GeoDataFrame
        Salida de :func:`score_new_data`.
    risk_col : str
        Columna a graficar, ``"risk_reg"`` o ``"risk_clf"``.
    model_label : str
        Etiqueta del modelo campeón, mostrada como título de la leyenda.
    task_title : str
        Nombre de la tarea, mostrado en el título de la figura (p. ej. "Regresión").
    risk_labels : dict
        Mapeo nivel -> etiqueta para la leyenda.
    """
    fig, axes = plt.subplots(1, len(muns), figsize=(3.2 * len(muns), 3.6))
    axes = np.atleast_1d(axes)

    for ax, mun in zip(axes, muns):
        sub = preds_gdf.query("CVE_MUN == @mun")
        sub.plot(column=risk_col, cmap=cmap, norm=norm, ax=ax, linewidth=1.4)
        label = mun_names.get(mun, mun) if len(sub) else mun
        ax.set_title(label, fontsize=11, pad=8)
        _strip_axis(ax)

    axes[0].set_ylabel(
        str(year_label), fontsize=11, rotation=0,
        ha="right", va="center", labelpad=24,
    )

    fig.legend(
        handles=_risk_legend_handles(risk_labels=risk_labels),
        title=model_label, loc="lower center", ncol=3, frameon=False,
        fontsize=11, bbox_to_anchor=(0.5, -0.06), handlelength=1.2,
        handleheight=1.2, columnspacing=1.5,
    )
    fig.suptitle(f"Riesgo por {task_title} — {year_label}", fontsize=13)
    fig.tight_layout(rect=[0, 0.09, 1, 0.90])

    _save(fig, out_path, dpi=360)
    return fig


def plot_scored_risk_full_cdmx(
    preds_gdf: gpd.GeoDataFrame,
    risk_col: str,
    year_label: int,
    model_label: str,
    task_title: str,
    risk_labels: dict,
    cmap, norm,
    out_path: str | Path,
) -> plt.Figure:
    """Grafica el riesgo del modelo campeón para toda la CDMX, en un solo panel.

    Parameters
    ----------
    preds_gdf : gpd.GeoDataFrame
        Salida de :func:`score_new_data`.
    risk_col : str
        Columna a graficar, ``"risk_reg"`` o ``"risk_clf"``.
    model_label : str
        Etiqueta del modelo campeón, mostrada como título de la leyenda.
    task_title : str
        Nombre de la tarea, mostrado en el título de la figura (p. ej. "Regresión").
    risk_labels : dict
        Mapeo nivel -> etiqueta para la leyenda.
    """
    fig, ax = plt.subplots(figsize=(7, 7))
    preds_gdf.plot(column=risk_col, cmap=cmap, norm=norm, ax=ax, linewidth=0.3)
    ax.set_title(f"Riesgo por {task_title} — CDMX completa, {year_label}",
                 fontsize=12, pad=10)
    _strip_axis(ax)

    fig.legend(
        handles=_risk_legend_handles(risk_labels=risk_labels),
        title=model_label, loc="lower center", ncol=3, frameon=False,
        fontsize=11, bbox_to_anchor=(0.5, -0.04), handlelength=1.2,
        handleheight=1.2, columnspacing=1.5,
    )
    fig.tight_layout(rect=[0, 0.07, 1, 1])

    _save(fig, out_path, dpi=300)
    return fig


def load_new_data(path: str | Path = NEW_DATA_PATH) -> gpd.GeoDataFrame:
    """Carga los datos nuevos a puntuar con los modelos campeones persistidos."""
    return gpd.read_parquet(path)


### -------------------------------------------------------------------------------
### Main --------------------------------------------------------------------------


def main() -> None:
    """Ejecuta las tres aplicaciones de mapas de riesgo y persiste sus salidas."""
    logger.info("Cargando asignaciones de clúster y geometría de calles...")
    gdf_clusters = load_cluster_assignments()
    gdf_geom = load_street_geometry()
    cmap, norm = _risk_cmap_norm()
    mun_names = build_mun_name_lookup(gdf_geom)

    try:
        plot_cluster_risk_sample_grid(
            gdf_clusters, gdf_geom, YEARS, MAPPINGS_BY_YEAR, SAMPLE_MUNS,
            SELECTED_K, cmap, norm,
            FIG_DIR / "kmeans_clustering_cluster_sample_01_esp.png")
        logger.info("Guardada la retícula de riesgo por alcaldías de muestra.")
    except Exception as e:
        logger.error(f"Error al construir la retícula de riesgo por alcaldías. {e}")

    try:
        plot_cluster_risk_full_cdmx(
            gdf_clusters, gdf_geom, YEARS, MAPPINGS_BY_YEAR, SELECTED_K,
            cmap, norm, FIG_DIR / "kmeans_clustering_full_cdmx_esp.png")
        logger.info("Guardado el mapa de riesgo de clústeres para toda la CDMX.")
    except Exception as e:
        logger.error(f"Error al construir el mapa de clústeres de toda la CDMX. {e}")

    logger.info(f"Puntuando {NEW_DATA_PATH} con los modelos campeones persistidos...")
    try:
        gdf_new = load_new_data()
        preds_gdf = score_new_data(gdf_new)
        clf_label, reg_label = load_champion_labels()

        SCORES_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        preds_gdf.to_parquet(SCORES_OUTPUT_PATH)
        logger.info(
            f"Persistidos {len(preds_gdf):,} registros puntuados en {SCORES_OUTPUT_PATH}"
        )
    except Exception as e:
        logger.error(f"Error al puntuar los datos nuevos con los modelos campeones. {e}")
        logger.info("geo_applications_esp finalizado.")
        return

    try:
        plot_scored_risk_sample_grid(
            preds_gdf, mun_names, SAMPLE_MUNS, risk_col="risk_reg",
            year_label=NEW_DATA_YEAR, model_label=reg_label,
            task_title=REGRESSION_TASK_TITLE,
            risk_labels=REGRESSION_RISK_LABELS,
            cmap=cmap, norm=norm,
            out_path=FIG_DIR / f"regression_risk_{NEW_DATA_YEAR}_sample_muns_esp.png")
        plot_scored_risk_full_cdmx(
            preds_gdf, risk_col="risk_reg", year_label=NEW_DATA_YEAR,
            model_label=reg_label, task_title=REGRESSION_TASK_TITLE,
            risk_labels=REGRESSION_RISK_LABELS, cmap=cmap, norm=norm,
            out_path=FIG_DIR / f"regression_risk_{NEW_DATA_YEAR}_full_cdmx_esp.png")
        logger.info(f"Guardados los mapas de riesgo de regresión ({reg_label}).")
    except Exception as e:
        logger.error(f"Error al construir los mapas de riesgo de regresión. {e}")

    try:
        plot_scored_risk_sample_grid(
            preds_gdf, mun_names, SAMPLE_MUNS, risk_col="risk_clf",
            year_label=NEW_DATA_YEAR, model_label=clf_label,
            task_title=CLASSIFICATION_TASK_TITLE,
            risk_labels=CLASSIFICATION_RISK_LABELS,
            cmap=cmap, norm=norm,
            out_path=FIG_DIR / f"classification_risk_{NEW_DATA_YEAR}_sample_muns_esp.png")
        plot_scored_risk_full_cdmx(
            preds_gdf, risk_col="risk_clf", year_label=NEW_DATA_YEAR,
            model_label=clf_label, task_title=CLASSIFICATION_TASK_TITLE,
            risk_labels=CLASSIFICATION_RISK_LABELS, cmap=cmap, norm=norm,
            out_path=FIG_DIR / f"classification_risk_{NEW_DATA_YEAR}_full_cdmx_esp.png")
        logger.info(f"Guardados los mapas de riesgo de clasificación ({clf_label}).")
    except Exception as e:
        logger.error(f"Error al construir los mapas de riesgo de clasificación. {e}")

    logger.info("geo_applications_esp finalizado.")


if __name__ == "__main__":
    from src.utils.prod import init_logger

    logger = init_logger()
    main()
