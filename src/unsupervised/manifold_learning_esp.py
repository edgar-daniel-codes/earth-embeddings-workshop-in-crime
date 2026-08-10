### Summer Internship - Earth Embeddings
### Unsupervised - Aprendizaje de variedades (versión en español)
### By Edgar Daniel


"""

Módulo para calcular y guardar algunas técnicas de aprendizaje de variedades
para el conjunto de entrenamiento dado (t-SNE, UMAP, LLE, MDS), cada una
acotada a una muestra o ajustada sobre muestra y transformada por lotes para
que el costo se mantenga tratable con más de 10M de registros. Todos los
textos de las figuras están en español.

``y_cnt`` (conteos de eventos adversos) se dibuja por defecto sobre la rampa
roja reservada del estilo compartido de la casa.

Copia en español de ``src.unsupervised.manifold_learning``: la lógica es
idéntica; las figuras se guardan con el sufijo ``_esp`` para no sobrescribir
las versiones en inglés.

"""

### -------------------------------------------------------------------------------
### Libraries and parameters-------------------------------------------------------

from __future__ import annotations

import sys
from pathlib import Path

# Allow both `python -m src.unsupervised.manifold_learning_esp` and direct execution.
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
    """Subconjunto aleatorio de filas (patrón de muestreo común a todos los métodos)."""
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
    """t-SNE acotado a una submuestra - mismo enfoque que ``compute_umap``, pero
    t-SNE solo puede mostrar la muestra, nunca el conjunto completo.
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
    """Ajusta UMAP sobre una muestra representativa y luego transforma el
    conjunto completo por lotes: el costo del ajuste queda acotado por
    ``sample_size`` y el de la transformación es lineal en el número de filas.
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
    """Ajusta LLE sobre una muestra representativa y luego transforma el
    conjunto completo por lotes - el mismo patrón de ajuste sobre muestra que
    ``compute_umap``, ya que LocallyLinearEmbedding de sklearn permite
    transformar puntos nuevos con sus pesos de reconstrucción frente a los
    vecinos ajustados.
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
    """El MDS clásico necesita la matriz completa de disimilitudes por pares y
    una mayorización iterativa SMACOF sobre ella - O(n^2) en tiempo y memoria.
    Se acota a una submuestra exactamente igual que ``compute_tsne``.
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
    """Diseño compartido de una columna por año para cada método de variedades.

    ``embed_fn(X, y) -> (embedding ndarray, y_plot)`` calcula el embedding en
    2-D y la porción del objetivo alineada con él (los métodos que submuestrean
    regresan ``y.iloc[sample_idx]``). ``cmap=None`` -> rampa roja de la casa.
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

    add_colorbar(fig, norm, cmap, "Número de delitos",
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
        parquet_path, feature_columns, years, embed, "Componente",
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

    logger.info("Inicio del proceso de aprendizaje de variedades.")

    logger.info("Trabajando con t-SNE. ")
    try:
        fig = plot_tsne_by_year(
            parquet_path=PARQUET,
            feature_columns=feature_columns,
            years=years,
            perplexity=50,
            max_iter=500,
        )

        fig.suptitle(
            "Proyección t-SNE del espacio de features por año",
            fontsize=18,
        )

        fig.savefig(FIG_PATH + "tsne_by_year_and_incidence_esp.png", dpi=300)
        logger.info("Proceso de t-SNE finalizado. ")
    except Exception as e:
        logger.error(f"Error al trabajar con t-SNE. {e}")

    logger.info("Trabajando con UMAP. ")
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
            "Proyección UMAP de los embeddings de AlphaEarth por año",
            fontsize=18,
        )

        fig.savefig(FIG_PATH + "umap_by_year_esp.png", dpi=300)
        logger.info("Proceso de UMAP finalizado. ")
    except Exception as e:
        logger.error(f"Error al trabajar con UMAP. {e}")

    logger.info("Trabajando con LLE. ")
    try:
        fig = plot_lle_by_year(
            parquet_path=PARQUET,
            feature_columns=feature_columns,
            years=years,
            sample_size=5_000,
            n_neighbors=10,
        )

        fig.suptitle(
            "Proyección LLE de los embeddings de AlphaEarth por año",
            fontsize=18,
        )

        fig.savefig(FIG_PATH + "lle_by_year_esp.png", dpi=300)
        logger.info("Proceso de LLE finalizado. ")
    except Exception as e:
        logger.error(f"Error al trabajar con LLE. {e}")

    logger.info("Trabajando con MDS. ")
    try:
        fig = plot_mds_by_year(
            parquet_path=PARQUET,
            feature_columns=feature_columns,
            years=years,
            sample_size=10_000,
        )

        fig.suptitle(
            "Proyección MDS de los embeddings de AlphaEarth por año",
            fontsize=18,
        )

        fig.savefig(FIG_PATH + "mds_by_year_esp.png", dpi=300)
        logger.info("Proceso de MDS finalizado. ")
    except Exception as e:
        logger.error(f"Error al trabajar con MDS. {e}")

    logger.info("Métodos de aprendizaje de variedades finalizados.")
