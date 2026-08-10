### Summer Internship - Earth Embeddings
### Visuals - Geo Applications (Risk Maps)
### By Edgar Daniel

"""
Street-segment risk maps for three applications:

* Clustering: k-means cluster assignments (2022-2024) mapped to a risk
  tier through a manual, year-specific cluster-to-tier mapping.
* Regression: the persisted regression champion model scored on new
  data, tiered by fixed thresholds on the expected count (``y_cnt_pred``).
* Classification: the persisted classification champion model scored on
  the same new data, tiered by fixed thresholds on the incident
  probability (``y_ind_proba``).

Run:  python -m src.visuals.geo_applications
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

logger = logging.getLogger("geo_applications")

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
RISK_LABELS = {1: "Low", 2: "Medium", 3: "High"}

### -------------------------------------------------------------------------------
### Fixed risk thresholds -----------------------------------------------------------

# y_ind_proba cutpoints, on the natural [0, 1] probability scale.
CLASSIFICATION_THRESHOLDS = (0.2, 0.50)
CLASSIFICATION_RISK_LABELS = {
    1: "Low (<0.20)",
    2: "Medium (0.20-0.5)",
    3: "High (>=0.5)",
}

# y_cnt_pred cutpoints, scaled x1/100 from count cuts of 1 and 5 to match
# the champion model's output range (observed max ~0.98 on real data).
REGRESSION_THRESHOLDS = (0.01, 0.05)
REGRESSION_RISK_LABELS = {
    1: "Low (<0.01)",
    2: "Medium (0.01-0.05)",
    3: "High (>=0.05)",
}

### -------------------------------------------------------------------------------
### Shared risk-palette helpers -----------------------------------------------------


def _risk_cmap_norm(risk_colors: dict = RISK_COLORS):
    """Build a categorical colormap and boundary norm from a tier -> color
    mapping.

    Parameters
    ----------
    risk_colors : dict
        Mapping of integer risk tier to hex color.

    Returns
    -------
    tuple
        ``(cmap, norm)`` ready to pass to ``GeoDataFrame.plot``.
    """
    tiers = sorted(risk_colors)
    cmap = mcolors.ListedColormap([risk_colors[t] for t in tiers])
    boundaries = np.array([tiers[0] - 0.5, *[t + 0.5 for t in tiers]])
    norm = mcolors.BoundaryNorm(boundaries, cmap.N)
    return cmap, norm


def _risk_legend_handles(risk_colors: dict = RISK_COLORS,
                         risk_labels: dict = RISK_LABELS) -> list[Patch]:
    """Build legend patch handles for the risk tiers."""
    return [Patch(facecolor=risk_colors[t], edgecolor="none", label=risk_labels[t])
            for t in sorted(risk_colors)]


def _strip_axis(ax: plt.Axes) -> None:
    """Remove ticks, lock the aspect ratio, and hide spines on a map axis."""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_visible(False)


def _save(fig: plt.Figure, out_path: str | Path, dpi: int) -> None:
    """Save a figure to ``out_path``, creating parent directories as needed."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


### -------------------------------------------------------------------------------
### Cluster-derived risk ------------------------------------------------------------


def load_cluster_assignments(path: str | Path = CLUSTER_ASSIGNMENTS_PATH) -> pd.DataFrame:
    """Load k-means cluster assignments (columns: year, CVEGEO, k, cluster)."""
    return pd.read_parquet(path)


def load_street_geometry(path: str | Path = STREETS_SHP_PATH) -> gpd.GeoDataFrame:
    """Load street-segment geometry with its municipio code and name."""
    return gpd.read_file(path)[["CVEGEO", "MUN", "NOM_MUN", "geometry"]]


def build_mun_name_lookup(gdf_geom: gpd.GeoDataFrame) -> dict[str, str]:
    """Build a municipio-code to municipio-name lookup table."""
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
    """Plot cluster-derived risk for sample municipios across years.

    One row per year, one column per municipio in ``muns``. ``mappings``
    gives the cluster-id -> risk-tier mapping for each entry in ``years``,
    in the same order.

    Raises
    ------
    ValueError
        If ``years`` and ``mappings`` differ in length.
    """
    if len(years) != len(mappings):
        raise ValueError(
            f"years ({len(years)}) and mappings ({len(mappings)}) must be "
            f"the same length.")

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
    """Plot cluster-derived risk for the full CDMX extent, one panel per year.

    Raises
    ------
    ValueError
        If ``years`` and ``mappings`` differ in length.
    """
    if len(years) != len(mappings):
        raise ValueError(
            f"years ({len(years)}) and mappings ({len(mappings)}) must be "
            f"the same length.")

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
    )
    fig.suptitle("Cluster-Derived Risk — Full CDMX", fontsize=13)
    fig.tight_layout(rect=[0, 0.06, 1, 0.94])

    _save(fig, out_path, dpi=300)
    return fig


### -------------------------------------------------------------------------------
### Regression / classification champion risk -------------------------------------


def bucket_fixed(values: pd.Series, thresholds: tuple[float, float]) -> pd.Series:
    """Bucket continuous values into 3 risk tiers using fixed thresholds.

    Parameters
    ----------
    values : pd.Series
        Continuous score to bucket.
    thresholds : tuple[float, float]
        ``(low_hi, med_hi)`` cutpoints: values below ``low_hi`` map to
        tier 1, ``[low_hi, med_hi)`` to tier 2, ``>= med_hi`` to tier 3.

    Returns
    -------
    pd.Series
        Integer tiers in ``{1, 2, 3}``, indexed like ``values``.
    """
    lo, hi = thresholds
    return pd.cut(values, bins=[-np.inf, lo, hi, np.inf],
                  labels=[1, 2, 3]).astype(int)


def load_champion_labels() -> tuple[str, str]:
    """Return the ``(classification, regression)`` champion model labels."""
    _, _, clf_info = load_champion("classification")
    _, _, reg_info = load_champion("regression")
    return clf_info["label"], reg_info["label"]


def score_new_data(gdf_new: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Score new rows with the persisted champion models and attach risk tiers.

    Parameters
    ----------
    gdf_new : gpd.GeoDataFrame
        New data with the feature columns the champion models expect.

    Returns
    -------
    gpd.GeoDataFrame
        CVEGEO, CVE_MUN, geometry, every ``predict()`` output column,
        ``risk_reg`` (from ``y_cnt_pred``) and ``risk_clf`` (from
        ``y_ind_proba``).
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
    """Plot champion-model risk for sample municipios, one row.

    Parameters
    ----------
    preds_gdf : gpd.GeoDataFrame
        Output of :func:`score_new_data`.
    risk_col : str
        Column to plot, either ``"risk_reg"`` or ``"risk_clf"``.
    model_label : str
        Champion model label, shown in the legend title.
    task_title : str
        Task name, shown in the figure title (e.g. "Regression").
    risk_labels : dict
        Tier -> label mapping for the legend.
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
    fig.suptitle(f"{task_title} Risk — {year_label}", fontsize=13)
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
    """Plot champion-model risk for the full CDMX extent, single panel.

    Parameters
    ----------
    preds_gdf : gpd.GeoDataFrame
        Output of :func:`score_new_data`.
    risk_col : str
        Column to plot, either ``"risk_reg"`` or ``"risk_clf"``.
    model_label : str
        Champion model label, shown in the legend title.
    task_title : str
        Task name, shown in the figure title (e.g. "Regression").
    risk_labels : dict
        Tier -> label mapping for the legend.
    """
    fig, ax = plt.subplots(figsize=(7, 7))
    preds_gdf.plot(column=risk_col, cmap=cmap, norm=norm, ax=ax, linewidth=0.3)
    ax.set_title(f"{task_title} Risk — Full CDMX, {year_label}", fontsize=12, pad=10)
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
    """Load the new data to score with the persisted champion models."""
    return gpd.read_parquet(path)


### -------------------------------------------------------------------------------
### Main --------------------------------------------------------------------------


def main() -> None:
    """Run all three risk-map applications and persist their outputs."""
    logger.info("Loading cluster assignments and street geometry...")
    gdf_clusters = load_cluster_assignments()
    gdf_geom = load_street_geometry()
    cmap, norm = _risk_cmap_norm()
    mun_names = build_mun_name_lookup(gdf_geom)

    try:
        plot_cluster_risk_sample_grid(
            gdf_clusters, gdf_geom, YEARS, MAPPINGS_BY_YEAR, SAMPLE_MUNS,
            SELECTED_K, cmap, norm,
            FIG_DIR / "kmeans_clustering_cluster_sample_01.png")
        logger.info("Saved cluster sample-municipio risk grid.")
    except Exception as e:
        logger.error(f"Error building the cluster sample-municipio risk grid. {e}")

    try:
        plot_cluster_risk_full_cdmx(
            gdf_clusters, gdf_geom, YEARS, MAPPINGS_BY_YEAR, SELECTED_K,
            cmap, norm, FIG_DIR / "kmeans_clustering_full_cdmx.png")
        logger.info("Saved cluster full-CDMX risk grid.")
    except Exception as e:
        logger.error(f"Error building the cluster full-CDMX risk grid. {e}")

    logger.info(f"Scoring {NEW_DATA_PATH} with the persisted champion models...")
    try:
        gdf_new = load_new_data()
        preds_gdf = score_new_data(gdf_new)
        clf_label, reg_label = load_champion_labels()

        SCORES_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        preds_gdf.to_parquet(SCORES_OUTPUT_PATH)
        logger.info(
            f"Persisted {len(preds_gdf):,} scored rows to {SCORES_OUTPUT_PATH}"
        )
    except Exception as e:
        logger.error(f"Error scoring new data with the champion models. {e}")
        logger.info("geo_applications finished.")
        return

    try:
        plot_scored_risk_sample_grid(
            preds_gdf, mun_names, SAMPLE_MUNS, risk_col="risk_reg",
            year_label=NEW_DATA_YEAR, model_label=reg_label,
            task_title="Regression", risk_labels=REGRESSION_RISK_LABELS,
            cmap=cmap, norm=norm,
            out_path=FIG_DIR / f"regression_risk_{NEW_DATA_YEAR}_sample_muns.png")
        plot_scored_risk_full_cdmx(
            preds_gdf, risk_col="risk_reg", year_label=NEW_DATA_YEAR,
            model_label=reg_label, task_title="Regression",
            risk_labels=REGRESSION_RISK_LABELS, cmap=cmap, norm=norm,
            out_path=FIG_DIR / f"regression_risk_{NEW_DATA_YEAR}_full_cdmx.png")
        logger.info(f"Saved regression champion risk maps ({reg_label}).")
    except Exception as e:
        logger.error(f"Error building the regression champion risk maps. {e}")

    try:
        plot_scored_risk_sample_grid(
            preds_gdf, mun_names, SAMPLE_MUNS, risk_col="risk_clf",
            year_label=NEW_DATA_YEAR, model_label=clf_label,
            task_title="Classification", risk_labels=CLASSIFICATION_RISK_LABELS,
            cmap=cmap, norm=norm,
            out_path=FIG_DIR / f"classification_risk_{NEW_DATA_YEAR}_sample_muns.png")
        plot_scored_risk_full_cdmx(
            preds_gdf, risk_col="risk_clf", year_label=NEW_DATA_YEAR,
            model_label=clf_label, task_title="Classification",
            risk_labels=CLASSIFICATION_RISK_LABELS, cmap=cmap, norm=norm,
            out_path=FIG_DIR / f"classification_risk_{NEW_DATA_YEAR}_full_cdmx.png")
        logger.info(f"Saved classification champion risk maps ({clf_label}).")
    except Exception as e:
        logger.error(f"Error building the classification champion risk maps. {e}")

    logger.info("geo_applications finished.")


if __name__ == "__main__":
    from src.utils.prod import init_logger

    logger = init_logger()
    main()
