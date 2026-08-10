### Summer Internship - Earth Embeddings
### Visuals - Embedding Structure Showcase
### By Edgar Daniel

"""
1x3 comparison of the AlphaEarth embedding structure at city level:
PCA and k-means clustering fit on a large representative sample of the
full per-pixel dataset (~15.9M rows for CDMX in a given year), and a
t-SNE projection from a much smaller, fast representative sample. Every
computation reuses existing functions from ``unsupervised.dim_reduction``,
``unsupervised.clustering_kmeans`` and ``unsupervised.manifold_learning``.

Run:  python -m src.visuals.embedding_structure_showcase
"""

### -------------------------------------------------------------------------------
### Libraries and parameters-------------------------------------------------------

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.dataset as pds

from src.unsupervised.clustering_kmeans import (
    ModelConfig,
    PaletteConfig,
    categorical_palette,
    load_normalized_features,
    spherical_kmeans,
)
from src.unsupervised.dim_reduction import compute_pca
from src.unsupervised.manifold_learning import compute_tsne
from src.utils.style import DEFAULT as PALETTE
from src.utils.style import apply_theme

REPO_ROOT = Path(__file__).resolve().parents[2]
YEAR = 2023
EMBEDDINGS_PATH = (
    REPO_ROOT / "data" / "proc" / "embeddings" / "alpha_earth" / "cdmx" / f"year={YEAR}"
)
OUT_DIR = REPO_ROOT / "docs" / "resources" / "unsupervised"
OUT_PATH = OUT_DIR / f"embedding_structure_{YEAR}.png"

FEATURE_COLUMNS = [f"A{i:02d}" for i in range(64)]

# The full per-pixel dataset is ~15.9M rows/year; PCA and k-means both fit
# a random sample this size without materializing the full population in
# memory (see load_representative_sample) - statistically equivalent to
# the true population for variance structure and cluster centers, at a
# fraction of the memory cost.
FULL_SAMPLE_SIZE = 2_000_000
PLOT_SAMPLE_SIZE = 50_000    # further subsample, for scatter legibility only
TSNE_SAMPLE_SIZE = 20_000    # small, fast representative sample for t-SNE

K = 4
RNG_SEED = 42

### -------------------------------------------------------------------------------
### Data preparation ----------------------------------------------------------------


def load_representative_sample(
    path: str | Path = EMBEDDINGS_PATH,
    feature_columns: list[str] = FEATURE_COLUMNS,
    max_rows: int = FULL_SAMPLE_SIZE,
    seed: int = RNG_SEED,
) -> tuple[pd.DataFrame, int]:
    """Draw a random, city-wide representative sample of the embeddings.

    Streams the Hive-partitioned dataset one fragment (municipio) at a
    time, downsampling each fragment immediately, so the full dataset is
    never held in memory at once.

    Returns
    -------
    tuple
        ``(sample_df, total_rows)`` — the sample (float32) and the true
        row count of the full dataset.
    """
    dataset = pds.dataset(path, format="parquet", partitioning="hive")
    total_rows = dataset.count_rows()
    frac = min(1.0, max_rows / total_rows)
    rng = np.random.default_rng(seed)

    parts = []
    for fragment in dataset.get_fragments():
        table = fragment.to_table(columns=feature_columns)
        n = table.num_rows
        if n == 0:
            continue
        take_n = min(max(1, round(n * frac)), n)
        idx = rng.choice(n, size=take_n, replace=False)
        parts.append(table.take(idx).to_pandas())

    sample = pd.concat(parts, ignore_index=True).astype(np.float32)
    return sample, total_rows


def fit_kmeans(
    X_norm: np.ndarray, k: int = K, seed: int = RNG_SEED,
):
    """Fit spherical k-means on already L2-normalized features.

    Returns
    -------
    tuple
        ``(labels, model)``.
    """
    cfg = ModelConfig(k_range=[k], k_list=[k], rng_seed=seed)
    model = spherical_kmeans(X_norm, k, cfg)
    labels = model.predict(X_norm)
    return labels, model


### -------------------------------------------------------------------------------
### Rendering -------------------------------------------------------------------


def _style_scatter_axis(ax: plt.Axes) -> None:
    """Light grid, no top/right spines — shared scatter-panel styling."""
    ax.grid(alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def build_embedding_structure_figure(
    pca_embedding: pd.DataFrame,
    pca_evr: np.ndarray,
    cluster_labels: np.ndarray,
    tsne_embedding: np.ndarray,
    tsne_labels: np.ndarray,
    cluster_colors: list[str],
    total_rows: int,
    year: int,
    k: int,
) -> plt.Figure:
    """Build the 1x3 comparison figure: PCA, PCA colored by k-means
    cluster, and t-SNE colored by the same clusters.
    """
    apply_theme()
    fig, (ax_pca, ax_kmeans, ax_tsne) = plt.subplots(1, 3, figsize=(16, 5.2))

    # Panel 1: plain PCA, city-wide sample.
    ax_pca.scatter(
        pca_embedding["PC1"], pca_embedding["PC2"], s=4, alpha=0.3,
        color=PALETTE.blue, linewidths=0, rasterized=True,
    )
    ax_pca.set_title(f"PCA — City-wide Sample (n={total_rows:,})", fontsize=12)
    ax_pca.set_xlabel(f"PC1 ({pca_evr[0] * 100:.1f}%)")
    ax_pca.set_ylabel(f"PC2 ({pca_evr[1] * 100:.1f}%)")
    _style_scatter_axis(ax_pca)

    # Panel 2: same PCA coordinates, colored by k-means cluster.
    for c, color in zip(sorted(np.unique(cluster_labels)), cluster_colors):
        mask = cluster_labels == c
        ax_kmeans.scatter(
            pca_embedding.loc[mask, "PC1"], pca_embedding.loc[mask, "PC2"],
            s=4, alpha=0.4, color=color, linewidths=0, rasterized=True,
            label=f"cluster {c}",
        )
    ax_kmeans.set_title(f"K-Means Clusters (k={k}) — PCA View", fontsize=12)
    ax_kmeans.set_xlabel(f"PC1 ({pca_evr[0] * 100:.1f}%)")
    ax_kmeans.set_ylabel(f"PC2 ({pca_evr[1] * 100:.1f}%)")
    _style_scatter_axis(ax_kmeans)

    # Panel 3: t-SNE on a small, fast representative sample, same clusters.
    for c, color in zip(sorted(np.unique(cluster_labels)), cluster_colors):
        mask = tsne_labels == c
        ax_tsne.scatter(
            tsne_embedding[mask, 0], tsne_embedding[mask, 1],
            s=6, alpha=0.5, color=color, linewidths=0, rasterized=True,
        )
    ax_tsne.set_title(
        f"t-SNE — Fast Representative Sample (n={len(tsne_labels):,})",
        fontsize=12,
    )
    ax_tsne.set_xlabel("t-SNE 1")
    ax_tsne.set_ylabel("t-SNE 2")
    _style_scatter_axis(ax_tsne)

    handles, labels = ax_kmeans.get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=k, frameon=False,
        fontsize=9, bbox_to_anchor=(0.5, -0.04),
    )

    fig.suptitle(
        f"AlphaEarth Embedding Structure — CDMX, {year}", fontsize=14,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    return fig


### -------------------------------------------------------------------------------
### Orchestration -------------------------------------------------------------------


def build_embedding_structure_showcase(
    embeddings_path: str | Path = EMBEDDINGS_PATH,
    year: int = YEAR,
    k: int = K,
    full_sample_size: int = FULL_SAMPLE_SIZE,
    plot_sample_size: int = PLOT_SAMPLE_SIZE,
    tsne_sample_size: int = TSNE_SAMPLE_SIZE,
    out_path: str | Path = OUT_PATH,
) -> Path:
    """Build and save the embedding structure showcase figure.

    Parameters
    ----------
    embeddings_path : str | Path
        Hive-partitioned per-pixel AlphaEarth embeddings for one year.
    year : int
        Year label, used in titles only.
    k : int
        Number of k-means clusters.
    full_sample_size : int
        Rows sampled city-wide for the PCA fit and the k-means fit.
    plot_sample_size : int
        Further subsample of the above, used only to keep the PCA/k-means
        scatter panels legible and fast to render.
    tsne_sample_size : int
        Rows sampled (from the full sample) for the fast t-SNE panel.
    out_path : str | Path
        Destination PNG path.

    Returns
    -------
    Path
        ``out_path``, after the figure has been written.
    """
    rng = np.random.default_rng(RNG_SEED)

    sample, total_rows = load_representative_sample(
        embeddings_path, FEATURE_COLUMNS, full_sample_size, RNG_SEED,
    )

    pca_embedding, pca_evr = compute_pca(sample[FEATURE_COLUMNS], n_components=2)

    X_norm = load_normalized_features(sample, FEATURE_COLUMNS)
    cluster_labels, _ = fit_kmeans(X_norm, k, RNG_SEED)

    tsne_idx = rng.choice(len(sample), size=min(tsne_sample_size, len(sample)),
                          replace=False)
    tsne_embedding, tsne_sample_idx = compute_tsne(
        sample.iloc[tsne_idx][FEATURE_COLUMNS], sample_size=tsne_sample_size,
    )
    tsne_labels = cluster_labels[tsne_idx][tsne_sample_idx]

    plot_idx = rng.choice(len(sample), size=min(plot_sample_size, len(sample)),
                          replace=False)
    palette_cfg = PaletteConfig()
    cluster_colors = categorical_palette(palette_cfg, k)

    fig = build_embedding_structure_figure(
        pca_embedding.iloc[plot_idx].reset_index(drop=True),
        pca_evr,
        cluster_labels[plot_idx],
        tsne_embedding,
        tsne_labels,
        cluster_colors,
        total_rows,
        year,
        k,
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


### -------------------------------------------------------------------------------
### Main --------------------------------------------------------------------------

if __name__ == "__main__":
    from src.utils.prod import init_logger

    logger = init_logger()
    logger.info(f"Building embedding structure showcase for year={YEAR}...")
    try:
        path = build_embedding_structure_showcase()
        logger.info(f"Saved embedding structure showcase to {path}")
    except Exception as e:
        logger.error(f"Error building the embedding structure showcase. {e}")
