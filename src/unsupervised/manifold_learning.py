### Summer Internship - Earth Embeddings
### Unsupervised - Manifold learning
### By Edgar Daniel


"""

Small module to calculate and save some manifold learning techniques for
the given training set (t-SNE, UMAP, LLE, MDS), each bounded to a sample
or fit-on-sample/transform-in-batches so cost stays tractable at 10M+ rows.

``y_cnt`` (adverse event counts) is rendered on the reserved red ramp
from the shared house style by default.

"""

### -------------------------------------------------------------------------------
### Libraries and parameters-------------------------------------------------------

from __future__ import annotations

import sys
from pathlib import Path

# Allow both `python -m src.unsupervised.manifold_learning` and direct execution.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import MDS, TSNE, LocallyLinearEmbedding
from umap import UMAP

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


def _sample_indices(n: int, sample_size: int, random_state: int) -> np.ndarray:
    """Random row subset (shared sampling pattern for all methods)."""
    if n <= sample_size:
        return np.arange(n)
    return (
        pd.Series(range(n))
        .sample(sample_size, random_state=random_state)
        .to_numpy()
    )


# Embeddings


def compute_tsne(
    X,
    perplexity: int = 50,
    learning_rate="auto",
    init: str = "pca",
    max_iter: int = 500,
    random_state: int = 42,
    sample_size: int = 50_000,
):
    """t-SNE bounded to a subsample - same approach as ``compute_umap``, but
    t-SNE can only ever show the sample, not the full dataset.
    """
    X_scaled = standardize(X)
    sample_idx = _sample_indices(len(X_scaled), sample_size, random_state)
    X_scaled = X_scaled[sample_idx]

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=learning_rate,
        init=init,
        max_iter=max_iter,
        random_state=random_state,
    )
    return tsne.fit_transform(X_scaled), sample_idx


def compute_umap(
    X,
    n_components: int = 2,
    sample_size: int = 100_000,
    batch_size: int = 100_000,
    random_state: int = 42,
    low_memory: bool = True,
    **umap_kwargs,
):
    """Fit UMAP on a representative sample, then transform the full dataset
    in batches: fit cost stays bounded by ``sample_size`` and transform
    cost is linear in row count.
    """
    X_scaled = standardize(X)

    if len(X_scaled) <= sample_size:
        umap = UMAP(
            n_components=n_components,
            random_state=random_state,
            low_memory=low_memory,
            **umap_kwargs,
        )
        return umap.fit_transform(X_scaled)

    sample_idx = _sample_indices(len(X_scaled), sample_size, random_state)
    X_sample = X_scaled[sample_idx]

    umap = UMAP(
        n_components=n_components,
        random_state=random_state,
        transform_seed=random_state,
        low_memory=low_memory,
        **umap_kwargs,
    )
    umap.fit(X_sample)

    embedding = np.empty((len(X_scaled), n_components), dtype=np.float32)
    for i in range(0, len(X_scaled), batch_size):
        j = min(i + batch_size, len(X_scaled))
        embedding[i:j] = umap.transform(X_scaled[i:j])

    return embedding


def compute_lle(
    X,
    n_components: int = 2,
    n_neighbors: int = 10,
    sample_size: int = 5_000,
    batch_size: int = 100_000,
    random_state: int = 42,
    eigen_solver: str = "arpack",
    n_jobs: int = -1,
    **lle_kwargs,
):
    """Fit LLE on a representative sample, then transform the full dataset
    in batches - same fit-on-sample pattern as ``compute_umap``, since
    sklearn's LocallyLinearEmbedding supports transforming new points via
    their reconstruction weights against the fitted neighbors.
    """
    X_scaled = standardize(X)

    if len(X_scaled) <= sample_size:
        lle = LocallyLinearEmbedding(
            n_components=n_components,
            n_neighbors=n_neighbors,
            random_state=random_state,
            eigen_solver=eigen_solver,
            n_jobs=n_jobs,
            **lle_kwargs,
        )
        return lle.fit_transform(X_scaled)

    sample_idx = _sample_indices(len(X_scaled), sample_size, random_state)
    X_sample = X_scaled[sample_idx]

    lle = LocallyLinearEmbedding(
        n_components=n_components,
        n_neighbors=n_neighbors,
        random_state=random_state,
        eigen_solver=eigen_solver,
        n_jobs=n_jobs,
        **lle_kwargs,
    )
    lle.fit(X_sample)

    embedding = np.empty((len(X_scaled), n_components), dtype=np.float32)
    for i in range(0, len(X_scaled), batch_size):
        j = min(i + batch_size, len(X_scaled))
        embedding[i:j] = lle.transform(X_scaled[i:j])

    return embedding


def compute_mds(
    X,
    n_components: int = 2,
    sample_size: int = 15_000,
    random_state: int = 69,
    n_init: int = 1,
    init: str = "random",
    normalized_stress: bool = False,
    **mds_kwargs,
):
    """Classical MDS needs the full pairwise dissimilarity matrix and an
    iterative SMACOF majorization over it - O(n^2) time and memory.
    Bounded to a subsample exactly like ``compute_tsne``.
    """
    X_scaled = standardize(X)
    sample_idx = _sample_indices(len(X_scaled), sample_size, random_state)
    X_scaled = X_scaled[sample_idx]

    mds = MDS(
        n_components=n_components,
        random_state=random_state,
        n_init=n_init,
        init=init,
        normalized_stress=normalized_stress,
        **mds_kwargs,
    )
    return mds.fit_transform(X_scaled), sample_idx


# Plotting


def _plot_embedding_by_year(
    parquet_path,
    feature_columns,
    years,
    embed_fn,
    axis_label: str,
    figsize=(10, 18),
    target_col: str = "y_cnt",
    alpha: float = 1.0,
    cmap=None,
):
    """Shared one-column-per-year scatter layout for every manifold method.

    ``embed_fn(X, y) -> (embedding ndarray, y_plot)`` computes the 2-D
    embedding and the target slice aligned with it (methods that subsample
    return ``y.iloc[sample_idx]``). ``cmap=None`` -> house red ramp.
    """
    apply_theme()

    df = load_years_data(parquet_path, years, feature_columns, target_col)
    norm = global_color_norm(df, target_col)

    fig, axes = plt.subplots(len(years), 1, figsize=figsize)
    axes = np.atleast_1d(axes)  

    for ax, year in zip(axes, years):

        X, y = year_slice(df, year, feature_columns, target_col)
        embedding, y_plot = embed_fn(X, y)

        colors = compute_colors(y_plot, norm, cmap)

        ax.scatter(
            embedding[:, 0],
            embedding[:, 1],
            c=colors,
            s=18,
            edgecolors="none",
            alpha = alpha,
        )

        ax.set_title(str(year))
        ax.set_xlabel(f"{axis_label} 1")
        ax.set_ylabel(f"{axis_label} 2")
        ax.grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0, 0.92, 0.97])

    add_colorbar(fig, norm, cmap, "Crime count",
                 rect=(0.94, 0.15, 0.015, 0.70))

    return fig


def plot_tsne_by_year(
    parquet_path,
    feature_columns,
    years,
    figsize=(10, 18),
    target_col: str = "y_cnt",
    cmap=None,
    **tsne_kwargs,
):
    def embed(X, y):
        embedding, sample_idx = compute_tsne(X, **tsne_kwargs)
        return embedding, y.iloc[sample_idx]

    return _plot_embedding_by_year(
        parquet_path, feature_columns, years, embed, "Component",
        figsize=figsize, target_col=target_col, cmap=cmap)


def plot_umap_by_year(
    parquet_path,
    feature_columns,
    years,
    figsize=(10, 18),
    target_col: str = "y_cnt",
    cmap=None,
    sample_size: int = 100_000,
    **umap_kwargs,
):
    def embed(X, y):
        return compute_umap(X, sample_size=sample_size, **umap_kwargs), y

    return _plot_embedding_by_year(
        parquet_path, feature_columns, years, embed, "UMAP",
        figsize=figsize, target_col=target_col, cmap=cmap)


def plot_lle_by_year(
    parquet_path,
    feature_columns,
    years,
    figsize=(10, 18),
    target_col: str = "y_cnt",
    cmap=None,
    sample_size: int = 5_000,
    **lle_kwargs,
):
    def embed(X, y):
        return compute_lle(X, sample_size=sample_size, **lle_kwargs), y

    return _plot_embedding_by_year(
        parquet_path, feature_columns, years, embed, "LLE",
        figsize=figsize, target_col=target_col, cmap=cmap)


def plot_mds_by_year(
    parquet_path,
    feature_columns,
    years,
    figsize=(10, 18),
    target_col: str = "y_cnt",
    cmap=None,
    **mds_kwargs,
):
    def embed(X, y):
        embedding, sample_idx = compute_mds(X, **mds_kwargs)
        return embedding, y.iloc[sample_idx]

    return _plot_embedding_by_year(
        parquet_path, feature_columns, years, embed, "MDS",
        figsize=figsize, target_col=target_col, cmap=cmap, 
        alpha=2.0)


### -------------------------------------------------------------------------------
### Main --------------------------------------------------------------------------

if __name__ == "__main__":

    from src.utils.prod import init_logger

    logger = init_logger()

    PARQUET = "./data/proc/training_sets/cdmx_asaltos.parquet"
    FIG_PATH = "./docs/resources/unsupervised/"

    feature_columns = [f"A{i:02d}" for i in range(64)]

    years = [2022, 2023, 2024]

    logger.info("Start process for Manifold Learning.")

    logger.info("Working with t-SNE. ")
    try:
        fig = plot_tsne_by_year(
            parquet_path=PARQUET,
            feature_columns=feature_columns,
            years=years,
            perplexity=50,
            max_iter=500,
        )

        fig.suptitle(
            "t-SNE projection of feature space by year",
            fontsize=18,
        )

        fig.savefig(FIG_PATH + "tsne_by_year_and_incidence.png", dpi=300)
        logger.info("t-SNE process ended. ")
    except Exception as e:
        logger.error(f"Error working on t-SNE. {e}")

    logger.info("Working with UMAP. ")
    try:
        fig = plot_umap_by_year(
            parquet_path=PARQUET,
            feature_columns=feature_columns,
            years=years,
            sample_size=100_000,
            n_neighbors=30,
            min_dist=0.1,
            metric="euclidean",
        )

        fig.suptitle(
            "UMAP projection of AlphaEarth embeddings by year",
            fontsize=18,
        )

        fig.savefig(FIG_PATH + "umap_by_year.png", dpi=300)
        logger.info("UMAP process ended. ")
    except Exception as e:
        logger.error(f"Error working on UMAP. {e}")

    logger.info("Working with LLE. ")
    try:
        fig = plot_lle_by_year(
            parquet_path=PARQUET,
            feature_columns=feature_columns,
            years=years,
            sample_size=5_000,
            n_neighbors=10,
        )

        fig.suptitle(
            "LLE projection of AlphaEarth embeddings by year",
            fontsize=18,
        )

        fig.savefig(FIG_PATH + "lle_by_year.png", dpi=300)
        logger.info("LLE process ended. ")
    except Exception as e:
        logger.error(f"Error working on LLE. {e}")

    logger.info("Working with MDS. ")
    try:
        fig = plot_mds_by_year(
            parquet_path=PARQUET,
            feature_columns=feature_columns,
            years=years,
            sample_size=10_000,
        )

        fig.suptitle(
            "MDS projection of AlphaEarth embeddings by year",
            fontsize=18,
        )

        fig.savefig(FIG_PATH + "mds_by_year.png", dpi=300)
        logger.info("MDS process ended. ")
    except Exception as e:
        logger.error(f"Error working on MDS. {e}")

    logger.info("Manifold learning methods ended.")
