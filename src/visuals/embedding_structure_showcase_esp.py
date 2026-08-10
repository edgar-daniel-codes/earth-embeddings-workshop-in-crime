### Summer Internship - Earth Embeddings
### Visuals - Estructura de los embeddings (versión en español)
### By Edgar Daniel

"""
Comparación 1x3 de la estructura de los embeddings de AlphaEarth a nivel
ciudad: PCA y k-means ajustados sobre una muestra representativa grande del
conjunto completo por píxel (~15.9M registros para la CDMX en un año dado),
y una proyección t-SNE a partir de una muestra representativa mucho más
pequeña y rápida. Cada cálculo reutiliza las funciones existentes de
``unsupervised.dim_reduction``, ``unsupervised.clustering_kmeans`` y
``unsupervised.manifold_learning``.

Copia en español de ``src.visuals.embedding_structure_showcase``: la lógica
es idéntica; la figura se guarda con el sufijo ``_esp`` para no sobrescribir
la versión en inglés.

Ejecutar:  python -m src.visuals.embedding_structure_showcase_esp
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
OUT_PATH = OUT_DIR / f"embedding_structure_{YEAR}_esp.png"

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
    """Toma una muestra aleatoria representativa de toda la ciudad.

    Recorre el dataset particionado tipo Hive un fragmento (alcaldía) a la
    vez, submuestreando cada fragmento de inmediato, de modo que el conjunto
    completo nunca se mantiene en memoria.

    Returns
    -------
    tuple
        ``(sample_df, total_rows)`` — la muestra (float32) y el número real de
        registros del conjunto completo.
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
    """Ajusta k-means esférico sobre features ya normalizadas en L2.

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
    """Rejilla ligera, sin bordes superior/derecho — estilo compartido."""
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
    """Construye la figura comparativa 1x3: PCA, PCA coloreado por clúster de
    k-means y t-SNE coloreado con los mismos clústeres.
    """
    apply_theme()
    fig, (ax_pca, ax_kmeans, ax_tsne) = plt.subplots(1, 3, figsize=(16, 5.2))

    # Panel 1: plain PCA, city-wide sample.
    ax_pca.scatter(
        pca_embedding["PC1"], pca_embedding["PC2"], s=4, alpha=0.3,
        color=PALETTE.blue, linewidths=0, rasterized=True,
    )
    ax_pca.set_title(f"PCA — Muestra de toda la ciudad (n={total_rows:,})",
                     fontsize=12)
    ax_pca.set_xlabel(f"PC1 ({pca_evr[0] * 100:.1f}%)")
    ax_pca.set_ylabel(f"PC2 ({pca_evr[1] * 100:.1f}%)")
    _style_scatter_axis(ax_pca)

    # Panel 2: same PCA coordinates, colored by k-means cluster.
    for c, color in zip(sorted(np.unique(cluster_labels)), cluster_colors):
        mask = cluster_labels == c
        ax_kmeans.scatter(
            pca_embedding.loc[mask, "PC1"], pca_embedding.loc[mask, "PC2"],
            s=4, alpha=0.4, color=color, linewidths=0, rasterized=True,
            label=f"clúster {c}",
        )
    ax_kmeans.set_title(f"Clústeres K-Means (k={k}) — Vista PCA", fontsize=12)
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
        f"t-SNE — Muestra representativa rápida (n={len(tsne_labels):,})",
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
        f"Estructura de los embeddings AlphaEarth — CDMX, {year}", fontsize=14,
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
    """Construye y guarda la figura de estructura de los embeddings.

    Parameters
    ----------
    embeddings_path : str | Path
        Embeddings AlphaEarth por píxel, particionados tipo Hive, de un año.
    year : int
        Año, usado únicamente en los títulos.
    k : int
        Número de clústeres de k-means.
    full_sample_size : int
        Registros muestreados de toda la ciudad para el ajuste de PCA y k-means.
    plot_sample_size : int
        Submuestra adicional de la anterior, usada solo para mantener legibles y
        rápidos los paneles de dispersión de PCA/k-means.
    tsne_sample_size : int
        Registros muestreados (de la muestra completa) para el panel rápido de t-SNE.
    out_path : str | Path
        Ruta destino del PNG.

    Returns
    -------
    Path
        ``out_path``, una vez escrita la figura.
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
    logger.info(f"Construyendo la figura de estructura de embeddings para year={YEAR}...")
    try:
        path = build_embedding_structure_showcase()
        logger.info(f"Figura de estructura de embeddings guardada en {path}")
    except Exception as e:
        logger.error(f"Error al construir la figura de estructura de embeddings. {e}")
