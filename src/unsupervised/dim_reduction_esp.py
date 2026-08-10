### Summer Internship - Earth Embeddings
### Unsupervised - Reducción de dimensionalidad (versión en español)
### By Edgar Daniel


"""

Módulo para calcular y guardar algunas técnicas de reducción de
dimensionalidad para el conjunto de entrenamiento dado, con todos los
textos de las figuras en español.

``y_cnt`` (conteos de eventos adversos) se dibuja por defecto sobre la
rampa roja reservada del estilo compartido de la casa.

Copia en español de ``src.unsupervised.dim_reduction``: la lógica es
idéntica; las figuras se guardan con el sufijo ``_esp`` para no
sobrescribir las versiones en inglés.

"""

### -------------------------------------------------------------------------------
### Libraries and parameters-------------------------------------------------------

from __future__ import annotations

import sys
from pathlib import Path

# Allow both `python -m src.unsupervised.dim_reduction_esp` and direct execution.
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
    """PCA sobre features estandarizadas -> (df del embedding, varianza explicada)."""
    X_scaled = standardize(X)

    pca = PCA(n_components=n_components, svd_solver="randomized")
    embedding = pca.fit_transform(X_scaled)
    columns = [f"PC{i + 1}" for i in range(n_components)]

    return (
        pd.DataFrame(embedding, columns=columns),
        pca.explained_variance_ratio_,
    )


def compute_svd(X, n_components: int = 3):
    """SVD truncada sobre features escaladas y sin centrar -> (df del embedding,
    varianza explicada).

    A diferencia de ``compute_pca``, las features solo se escalan a varianza
    unitaria, no se centran en la media, de modo que los vectores singulares
    reflejan direcciones SVD verdaderas desde el origen en vez de las
    direcciones de varianza centradas del PCA.
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
    """Distancia de Mahalanobis de cada registro a su propia media muestral.

    Usa una covarianza regularizada tipo ridge (cov + reg * I) para que la
    factorización de Cholesky se mantenga estable aunque algunas dimensiones
    del embedding de AlphaEarth resulten casi colineales en un año dado. Las
    distancias se calculan mediante una transformación de blanqueo (solve, no
    inversa de matriz), así que el costo es una sola multiplicación de matrices
    O(n * p^2) - trivial con p=64 incluso para más de 10M de registros.
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
    """Agrupa la variable objetivo en rangos de conteo para comparaciones."""
    return pd.cut(y, bins=bins, labels=labels)


def describe_mahalanobis_by_bucket(
    distances,
    y,
    bins=(-np.inf, 5, np.inf),
    labels=("0-5", ">5"),
):
    """Estadísticos de resumen de las distancias dentro de cada grupo objetivo."""
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
    """Retícula de dispersiones PCA: una fila por año, una columna por par de PC.

    ``cmap=None`` -> rampa roja de la casa (``y_cnt`` es un conteo adverso).
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

    add_colorbar(fig, norm, cmap, "Número de delitos")

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
    """Retícula de dispersiones SVD: una fila por año, una columna por par de SV.

    Mismo flujo que ``plot_pca_by_year`` pero respaldado por ``compute_svd``, de
    modo que las proyecciones están ancladas al origen en vez de centradas. Los
    puntos siguen coloreados y ponderados por transparencia según ``y_cnt``
    (rampa roja de la casa por defecto) para mantener comparables las vistas.
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

    add_colorbar(fig, norm, cmap, "Número de delitos")

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
    """Diagramas de caja por año de la distancia de Mahalanobis por grupo objetivo."""
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
        ax.set_xlabel("Grupo de y_cnt")
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("Distancia de Mahalanobis")

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

    logger.info("Inicio del proceso de reducción de dimensionalidad.")

    logger.info("Trabajando con PCA. ")
    try:
        fig = plot_pca_by_year(
            parquet_path=PARQUET,
            feature_columns=feature_columns,
            years=years,
            pairs=pca_pairs,
        )

        fig.suptitle(
            "Proyección PCA de los embeddings de Alpha Earth por año",
            fontsize=18,
        )

        fig.savefig(FIG_PATH + "pca_yearly_comp_esp.png", dpi=300)
        logger.info("Proceso de PCA finalizado. ")
    except Exception as e:
        logger.error(f"Error al trabajar con PCA. {e}")

    logger.info("Trabajando con SVD. ")
    try:
        fig = plot_svd_by_year(
            parquet_path=PARQUET,
            feature_columns=feature_columns,
            years=years,
            pairs=svd_pairs,
        )

        fig.suptitle(
            "Proyección SVD de los embeddings de Alpha Earth por año",
            fontsize=18,
        )

        fig.savefig(FIG_PATH + "svd_yearly_comp_esp.png", dpi=300)
        logger.info("Proceso de SVD finalizado. ")
    except Exception as e:
        logger.error(f"Error al trabajar con SVD. {e}")

    logger.info("Trabajando con la distancia de Mahalanobis. ")
    try:
        fig, summary = plot_mahalanobis_by_year(
            parquet_path=PARQUET,
            feature_columns=feature_columns,
            years=years,
        )

        fig.suptitle(
            "Distancia de Mahalanobis por grupo de y_cnt, por año",
            fontsize=14,
        )

        fig.savefig(FIG_PATH + "mahalanobis_by_year_esp.png", dpi=300)
        summary.to_csv(FIG_PATH + "mahalanobis_by_year_summary_esp.csv")
        logger.info("Proceso de distancia de Mahalanobis finalizado. ")
    except Exception as e:
        logger.error(f"Error al trabajar con la distancia de Mahalanobis. {e}")

    logger.info("Métodos de reducción de dimensionalidad finalizados.")
