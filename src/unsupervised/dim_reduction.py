### Summer Internship - Earth Embeddings
### Unsupervised - Dimensionality reduction
### By Edgar Daniel


"""

Small module to calculate and save some dimensionality reduction
techniques for the given training set.

``y_cnt`` (adverse event counts) is rendered on the reserved red ramp
from the shared house style by default.

"""

### -------------------------------------------------------------------------------
### Libraries and parameters-------------------------------------------------------

from __future__ import annotations

import sys
from pathlib import Path

# Allow both `python -m src.unsupervised.dim_reduction` and direct execution.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD

from src.utils.dim_reduction import (
    add_colorbar,
    compute_colors,
    global_color_norm,
    load_years_data,
    standardize,
    year_slice,
)
from src.utils.style import apply_theme


### -------------------------------------------------------------------------------
### Functions and classes ---------------------------------------------------------


# Embeddings


def compute_pca(X, n_components: int = 3):
    """PCA on standardized features -> (embedding df, explained variance)."""
    X_scaled = standardize(X)

    pca = PCA(n_components=n_components, svd_solver="randomized")
    embedding = pca.fit_transform(X_scaled)
    columns = [f"PC{i + 1}" for i in range(n_components)]

    return (
        pd.DataFrame(embedding, columns=columns),
        pca.explained_variance_ratio_,
    )


def compute_svd(X, n_components: int = 3):
    """Truncated SVD on scaled, uncentered features -> (embedding df, explained variance).

    Unlike ``compute_pca``, features are only scaled to unit variance, not
    mean-centered, so the singular vectors reflect true SVD directions from
    the origin rather than PCA's centered variance directions.
    """
    X_scaled = standardize(X, center=False)

    svd = TruncatedSVD(n_components=n_components)
    embedding = svd.fit_transform(X_scaled)
    columns = [f"SV{i + 1}" for i in range(n_components)]

    return (
        pd.DataFrame(embedding, columns=columns),
        svd.explained_variance_ratio_,
    )


def compute_mahalanobis(X, reg: float = 1e-6) -> np.ndarray:
    """Mahalanobis distance of every row to its own sample mean.

    Uses a ridge-regularized covariance (cov + reg * I) so the Cholesky
    factorization stays stable even if some AlphaEarth embedding
    dimensions end up near-collinear for a given year. Distances are
    computed via a whitening transform (solve, not matrix inverse), so
    the cost is one O(n * p^2) matmul - trivial at p=64 even for
    10M+ rows.
    """
    X_scaled = standardize(X)

    mean = X_scaled.mean(axis=0)
    cov = np.cov(X_scaled, rowvar=False)
    cov += reg * np.eye(cov.shape[0], dtype=cov.dtype)

    L = np.linalg.cholesky(cov)
    centered = X_scaled - mean
    whitened = np.linalg.solve(L, centered.T).T

    return np.sqrt(np.sum(whitened ** 2, axis=1))


def bucket_y_cnt(
    y,
    bins=(-np.inf, 5, np.inf),
    labels=("0-5", ">5"),
):
    """Bucket the target into count ranges for grouped comparisons."""
    return pd.cut(y, bins=bins, labels=labels)


def describe_mahalanobis_by_bucket(
    distances,
    y,
    bins=(-np.inf, 5, np.inf),
    labels=("0-5", ">5"),
):
    """Summary statistics of the distances within each target bucket."""
    bucket = bucket_y_cnt(y, bins, labels)

    return (
        pd.Series(distances, index=y.index, name="mahalanobis")
        .groupby(bucket, observed=True)
        .describe()
    )


# Plotting


# PCA Figure

def plot_pca_by_year(
    parquet_path,
    feature_columns,
    years,
    pairs,
    figsize=(18, 15),
    target_col: str = "y_cnt",
    cmap=None,
):
    """Grid of PCA scatter panels: one row per year, one column per PC pair.

    ``cmap=None`` -> house red ramp (``y_cnt`` is an adverse count).
    """
    apply_theme()

    df = load_years_data(parquet_path, years, feature_columns, target_col)
    norm = global_color_norm(df, target_col)

    fig, axes = plt.subplots(
        len(years),
        len(pairs),
        figsize=figsize,
        sharex="col",
        sharey="col",
    )

    axes = np.atleast_2d(axes)

    for i, year in enumerate(years):

        X, y = year_slice(df, year, feature_columns, target_col)
        embedding, evr = compute_pca(X)

        colors = compute_colors(y, norm, cmap)

        for j, (pcx, pcy) in enumerate(pairs):

            ax = axes[i, j]
            ax.scatter(
                embedding[pcx],
                embedding[pcy],
                c=colors,
                s=20,
                linewidths=0,
                rasterized=True,
            )

            if i == 0:
                var_x = evr[int(pcx[2:]) - 1] * 100
                var_y = evr[int(pcy[2:]) - 1] * 100
                ax.set_title(f"{pcx} ({var_x:.1f}%) vs {pcy} ({var_y:.1f}%)")

            if j == 0:
                ax.set_ylabel(f"{year}\n{pcy}")
            if i == len(years) - 1:
                ax.set_xlabel(pcx)

            ax.grid(alpha=0.2)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    fig.subplots_adjust(
        left=0.08, right=0.91, bottom=0.07, top=0.94,
        wspace=0.18, hspace=0.18,
    )

    add_colorbar(fig, norm, cmap, "Crime count")

    return fig


# SVD Figure

def plot_svd_by_year(
    parquet_path,
    feature_columns,
    years,
    pairs,
    figsize=(18, 15),
    target_col: str = "y_cnt",
    cmap=None,
):
    """Grid of SVD scatter panels: one row per year, one column per SV pair.

    Same flow as ``plot_pca_by_year`` but backed by ``compute_svd``, so
    projections are anchored to the origin rather than centered. Points
    stay colored/alpha-weighted by ``y_cnt`` (house red ramp by default)
    to keep the count-anchored view comparable across methods.
    """
    apply_theme()

    df = load_years_data(parquet_path, years, feature_columns, target_col)
    norm = global_color_norm(df, target_col)

    fig, axes = plt.subplots(
        len(years),
        len(pairs),
        figsize=figsize,
        sharex="col",
        sharey="col",
    )

    axes = np.atleast_2d(axes)

    for i, year in enumerate(years):

        X, y = year_slice(df, year, feature_columns, target_col)
        embedding, evr = compute_svd(X)

        colors = compute_colors(y, norm, cmap)

        for j, (svx, svy) in enumerate(pairs):

            ax = axes[i, j]
            ax.scatter(
                embedding[svx],
                embedding[svy],
                c=colors,
                s=20,
                linewidths=0,
                rasterized=True,
            )

            if i == 0:
                var_x = evr[int(svx[2:]) - 1] * 100
                var_y = evr[int(svy[2:]) - 1] * 100
                ax.set_title(f"{svx} ({var_x:.1f}%) vs {svy} ({var_y:.1f}%)")

            if j == 0:
                ax.set_ylabel(f"{year}\n{svy}")
            if i == len(years) - 1:
                ax.set_xlabel(svx)

            ax.grid(alpha=0.2)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    fig.subplots_adjust(
        left=0.08, right=0.91, bottom=0.07, top=0.94,
        wspace=0.18, hspace=0.18,
    )

    add_colorbar(fig, norm, cmap, "Crime count")

    return fig


# Mahalanobis Figure

def plot_mahalanobis_by_year(
    parquet_path,
    feature_columns,
    years,
    figsize=(4, 4),
    target_col: str = "y_cnt",
    bins=(-np.inf, 5, np.inf),
    labels=("0-5", ">5"),
    reg: float = 1e-6,
):
    """Per-year boxplots of Mahalanobis distance by target bucket."""
    apply_theme()

    df = load_years_data(parquet_path, years, feature_columns, target_col)

    fig, axes = plt.subplots(
        1,
        len(years),
        figsize=(figsize[0] * len(years), figsize[1]),
        sharey=True,
    )
    axes = np.atleast_1d(axes)  

    summaries = []

    for ax, year in zip(axes, years):

        X, y = year_slice(df, year, feature_columns, target_col)

        distances = compute_mahalanobis(X, reg=reg)
        bucket = bucket_y_cnt(y, bins, labels)

        summary = describe_mahalanobis_by_bucket(distances, y, bins, labels)
        summary.insert(0, "year", year)
        summaries.append(summary)

        groups = [distances[(bucket == label).to_numpy()] for label in labels]

        ax.boxplot(groups, tick_labels=labels, showfliers=False)

        ax.set_title(str(year))
        ax.set_xlabel("y_cnt bucket")
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("Mahalanobis distance")

    fig.tight_layout()

    return fig, pd.concat(summaries)


### -------------------------------------------------------------------------------
### Main --------------------------------------------------------------------------

if __name__ == "__main__":

    from src.utils.prod import init_logger

    logger = init_logger()

    PARQUET = "./data/proc/training_sets/cdmx_asaltos.parquet"
    FIG_PATH = "./docs/resources/unsupervised/"

    feature_columns = [f"A{i:02d}" for i in range(64)]

    years = [2022, 2023, 2024]

    pca_pairs = [
        ("PC1", "PC2"),
        ("PC1", "PC3"),
        ("PC2", "PC3"),
    ]

    svd_pairs = [
        ("SV1", "SV2"),
        ("SV1", "SV3"),
        ("SV2", "SV3"),
    ]

    logger.info("Start process for dim reduction.")

    logger.info("Working with PCA. ")
    try:
        fig = plot_pca_by_year(
            parquet_path=PARQUET,
            feature_columns=feature_columns,
            years=years,
            pairs=pca_pairs,
        )

        fig.suptitle(
            "PCA projection of Alpha Earth embeddings by year",
            fontsize=18,
        )

        fig.savefig(FIG_PATH + "pca_yearly_comp.png", dpi=300)
        logger.info("PCA process ended. ")
    except Exception as e:
        logger.error(f"Error working on PCA. {e}")

    logger.info("Working with SVD. ")
    try:
        fig = plot_svd_by_year(
            parquet_path=PARQUET,
            feature_columns=feature_columns,
            years=years,
            pairs=svd_pairs,
        )

        fig.suptitle(
            "SVD projection of Alpha Earth embeddings by year",
            fontsize=18,
        )

        fig.savefig(FIG_PATH + "svd_yearly_comp.png", dpi=300)
        logger.info("SVD process ended. ")
    except Exception as e:
        logger.error(f"Error working on SVD. {e}")

    logger.info("Working with Mahalanobis distance. ")
    try:
        fig, summary = plot_mahalanobis_by_year(
            parquet_path=PARQUET,
            feature_columns=feature_columns,
            years=years,
        )

        fig.suptitle(
            "Mahalanobis distance by y_cnt bucket, by year",
            fontsize=14,
        )

        fig.savefig(FIG_PATH + "mahalanobis_by_year.png", dpi=300)
        summary.to_csv(FIG_PATH + "mahalanobis_by_year_summary.csv")
        logger.info("Mahalanobis distance process ended. ")
    except Exception as e:
        logger.error(f"Error working on Mahalanobis distance. {e}")

    logger.info("Dimensionality reduction methods ended.")
