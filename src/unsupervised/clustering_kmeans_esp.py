### Summer Internship - Earth Embeddings
### Unsupervised - Clustering (k-means esférico, versión en español)
### By Edgar Daniel


"""
clustering_kmeans_esp.py
========================

Pipeline de clustering con k-means esférico para la comparación entre años de
una variable de conteo (``y_cnt``) condicionada a la pertenencia al clúster.
Todos los textos visibles de las figuras están en español.

Copia en español de ``src.unsupervised.clustering_kmeans``: la lógica es
idéntica; cada figura se guarda con el sufijo ``_esp`` para no sobrescribir
las versiones en inglés.

Notas de diseño
---------------
* Geometría: se asume que las features viven en [-1, 1]^d sin garantía de
  norma unitaria. Normalizamos las filas en L2 para que la distancia
  euclidiana sobre la esfera sea una función monótona de la similitud coseno
  (||u - v||^2 = 2 - 2cos(u,v)), lo que permite ejecutar un k-means esférico
  *verdadero* (renormalizando los centroides después de cada ajuste parcial)
  usando ``MiniBatchKMeans`` como solucionador interno.

* Variable de salida: ``y_cnt`` es un conteo entero no negativo (eventos por
  unidad espacial). Deliberadamente NO usamos KDE ni violines para ella: un
  kernel gaussiano coloca masa de densidad continua en valores no enteros y
  negativos. En su lugar reportamos la PMF empírica y la ECDF, que sí admiten
  una comparación puntual válida entre clústeres (por ejemplo, mediante
  diagnósticos visuales tipo Kolmogorov-Smirnov sobre el panel de la ECDF).

* Tamaño de efecto por encima de la significancia: con n ~ 10^6, los valores p
  de Kruskal-Wallis colapsan a cero numérico ante prácticamente cualquier
  efecto no nulo. Por eso tratamos epsilon^2 = (H - k + 1) / (n - k) como la
  cantidad sustantiva y a p como una formalidad.

Uso
---
    python -m src.unsupervised.clustering_kmeans_esp --config config/kmeans_conf.yml

o programáticamente:

    from src.unsupervised.clustering_kmeans_esp import PipelineConfig, run_pipeline
    cfg = PipelineConfig.from_yaml("config/kmeans_conf.yml")
    result = run_pipeline(cfg)
"""

### -------------------------------------------------------------------------------
### Libraries and parameters-------------------------------------------------------

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

# Allow both `python -m src.unsupervised.clustering_kmeans_esp` and direct execution.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from scipy import stats
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (calinski_harabasz_score, davies_bouldin_score,
                             silhouette_score)
from sklearn.preprocessing import normalize

from src.utils.style import DEFAULT as HOUSE


logger = logging.getLogger("cluster_pipeline_esp")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S"))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)

# Sufijo añadido al nombre de cada figura para no pisar las versiones en inglés.
FIG_SUFFIX = "_esp"

### -------------------------------------------------------------------------------
### Style -------------------------------------------------------------------------

@dataclass(frozen=True)
class DataConfig:
    parquet_path: str
    feature_columns: Sequence[str]
    key_column: str                        # identificador por registro, preservado entre años
    year_column: str = "year"
    sort_column: str = "cnt"
    ind_column: str = "y_ind"
    cnt_column: str = "y_cnt"


@dataclass(frozen=True)
class ModelConfig:
    k_range: Sequence[int]                 # barrido de exploración, p. ej. list(range(2, 15))
    k_list: Sequence[int]                  # k finales a ajustar y comparar, p. ej. [4, 7]
    rng_seed: int = 69
    batch_size: int = 10_000
    n_init_sweep: int = 5
    max_iter_sweep: int = 200
    reassignment_ratio: float = 0.01
    spherical_n_iter: int = 50
    spherical_tol: float = 1e-4
    sil_sample_size: int = 20_000


@dataclass(frozen=True)
class PaletteConfig:
    """Paleta parametrizable de tinta mínima.

    Los valores por defecto vienen del estilo compartido de la casa
    (``src.utils.style``); cada campo se puede sobrescribir desde el bloque
    ``viz.palette`` del YAML.
    """
    base_color: str = HOUSE.navy            # acento azul marino (fuente de la rampa)
    highlight_color: str = HOUSE.bad        # rojo, solo líneas de referencia/anotaciones
    ink_color: str = HOUSE.ink              # texto / etiquetas de ejes
    axis_color: str = HOUSE.muted           # bordes / etiquetas de marcas
    grid_color: str = HOUSE.grid            # líneas de rejilla
    background_color: str = HOUSE.background
    font_family: str = HOUSE.font_family
    n_shades: int = 9                       # máximo de niveles categóricos a pregenerar
    dpi: int = 220
    fig_format: str = "png"
    categorical_hex: Optional[Sequence[str]] = None  # anulación explícita, tiene prioridad


@dataclass(frozen=True)
class VizConfig:
    fig_path: str
    palette: PaletteConfig = field(default_factory=PaletteConfig)
    max_display_count: Optional[int] = None
    pca_sample_size: int = 20_000


@dataclass(frozen=True)
class OutputConfig:
    labeled_parquet_path: Optional[str] = None
    sweep_table_path: Optional[str] = None
    results_long_path: Optional[str] = None
    effect_table_path: Optional[str] = None
    cluster_assignments_path: Optional[str] = None


@dataclass(frozen=True)
class PipelineConfig:
    data: DataConfig
    model: ModelConfig
    viz: VizConfig
    output: OutputConfig = field(default_factory=OutputConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
        return cls(
            data=DataConfig(**raw["data"]),
            model=ModelConfig(**raw["model"]),
            viz=VizConfig(
                fig_path=raw["viz"]["fig_path"],
                palette=PaletteConfig(**raw["viz"].get("palette", {})),
                max_display_count=raw["viz"].get("max_display_count"),
                pca_sample_size=raw["viz"].get("pca_sample_size", 20_000),
            ),
            output=OutputConfig(**raw.get("output", {})),
        )

    def validate(self) -> None:
        if not set(self.model.k_list).issubset(set(self.model.k_range)):
            logger.warning("k_list %s no está contenido por completo en k_range %s — "
                           "el barrido no cubrirá todos los k finales.",
                           self.model.k_list, self.model.k_range)
        if len(self.data.feature_columns) == 0:
            raise ValueError("data.feature_columns está vacío.")
        Path(self.viz.fig_path).mkdir(parents=True, exist_ok=True)



def apply_house_style(palette: PaletteConfig) -> None:
    """Fija un tema de matplotlib/seaborn con alta razón de tinta-datos: sin
    adornos, rejilla solo horizontal, tinta desaturada y bordes superior y
    derecho eliminados. Idempotente — puede llamarse varias veces."""
    sns.set_theme(style="white", context="notebook")
    plt.rcParams.update({
        "figure.facecolor": palette.background_color,
        "axes.facecolor": palette.background_color,
        "savefig.facecolor": palette.background_color,
        "axes.edgecolor": palette.axis_color,
        "axes.labelcolor": palette.ink_color,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "grid.color": palette.grid_color,
        "grid.linewidth": 0.8,
        "xtick.color": palette.axis_color,
        "ytick.color": palette.axis_color,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "text.color": palette.ink_color,
        "font.family": palette.font_family,
        "font.size": 10,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "figure.titlesize": 12,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlecolor": palette.ink_color,
        "savefig.dpi": palette.dpi,
    })


# Conjunto categórico de orden fijo para la identidad del clúster (dispersión
# PCA, paneles PMF/ECDF, barras apiladas de proporciones)
_CLUSTER_HEX: tuple[str, ...] = HOUSE.qual


def categorical_palette(palette: PaletteConfig, n: int) -> list[str]:
    """``n`` colores para distinguir clústeres: una anulación explícita vía
    ``categorical_hex`` tiene prioridad; de lo contrario toma en orden fijo del
    conjunto categórico de la casa. Más allá de eso (raro), recurre a sombrear
    ``palette.base_color`` de oscuro a claro."""
    if palette.categorical_hex is not None:
        if len(palette.categorical_hex) < n:
            raise ValueError(
                f"categorical_hex tiene {len(palette.categorical_hex)} colores, "
                f"se necesitan {n}.")
        return list(palette.categorical_hex[:n])
    if n <= len(_CLUSTER_HEX):
        return list(_CLUSTER_HEX[:n])
    ramp = sns.dark_palette(palette.base_color, n_colors=max(n, 2) + 1,
                            reverse=False)
    return [mpl_to_hex(c) for c in ramp[1: n + 1]]


def mpl_to_hex(rgb) -> str:
    import matplotlib.colors as mcolors
    return mcolors.to_hex(rgb)



def strip_axis(ax: plt.Axes) -> None:
    """Quita las marcas (conservando las etiquetas) — toque final de tinta mínima."""
    ax.tick_params(length=0)
    ax.spines["bottom"].set_visible(True)


### -------------------------------------------------------------------------------
### Functions and classes ---------------------------------------------------------


def load_normalized_features(gdf: pd.DataFrame, feature_columns: Sequence[str]) -> np.ndarray:
    X = gdf[list(feature_columns)].to_numpy(dtype=np.float32)
    return normalize(X, norm="l2", axis=1)


def spherical_kmeans(
    X: np.ndarray,
    k: int,
    cfg: ModelConfig,
) -> MiniBatchKMeans:
    """K-means esférico verdadero: MiniBatchKMeans como motor del paso de Lloyd,
    con los centroides reproyectados a la esfera unitaria después de cada ajuste
    parcial para que las actualizaciones euclidianas al cuadrado sigan siendo
    consistentes con el coseno."""
    batch_size = min(cfg.batch_size, X.shape[0])
    n = X.shape[0]

    mbk = MiniBatchKMeans(
        n_clusters=k, batch_size=batch_size, n_init=1,
        random_state=cfg.rng_seed, max_iter=1,
    )
    mbk.partial_fit(X[:batch_size])
    mbk.cluster_centers_ = normalize(mbk.cluster_centers_, norm="l2")
    prev = mbk.cluster_centers_.copy()

    for it in range(cfg.spherical_n_iter):
        rng = np.random.default_rng(cfg.rng_seed + it + 1)
        idx = rng.choice(n, size=batch_size, replace=False)
        mbk.partial_fit(X[idx])
        mbk.cluster_centers_ = normalize(mbk.cluster_centers_, norm="l2")
        shift = np.linalg.norm(mbk.cluster_centers_ - prev)
        if shift < cfg.spherical_tol:
            logger.debug("spherical_kmeans(k=%d) convergió en la iteración %d (shift=%.2e)",
                         k, it, shift)
            break
        prev = mbk.cluster_centers_.copy()
    return mbk


def run_k_sweep(X_norm: np.ndarray, year, cfg: ModelConfig) -> pd.DataFrame:
    """Barrido de selección de modelo sobre cfg.k_range: inercia,
    Calinski-Harabasz, Davies-Bouldin (datos completos) y silueta (submuestreada
    por tratabilidad)."""
    n = X_norm.shape[0]
    rng = np.random.default_rng(cfg.rng_seed)
    sil_idx = rng.choice(n, size=min(cfg.sil_sample_size, n), replace=False)

    records = []
    for k in cfg.k_range:
        mbk = MiniBatchKMeans(
            n_clusters=k, batch_size=min(cfg.batch_size, n),
            n_init=cfg.n_init_sweep, max_iter=cfg.max_iter_sweep,
            random_state=cfg.rng_seed, reassignment_ratio=cfg.reassignment_ratio,
        )
        labels = mbk.fit_predict(X_norm)
        try:
            sil = silhouette_score(X_norm[sil_idx], labels[sil_idx],
                                   metric="euclidean")
        except ValueError:
            sil = float("nan")
        rec = {
            "year": year,
            "k": k,
            "inertia": mbk.inertia_,
            "calinski_harabasz": calinski_harabasz_score(X_norm, labels),
            "davies_bouldin": davies_bouldin_score(X_norm, labels),
            "silhouette": sil,
        }
        records.append(rec)
        logger.info("[año=%s] k=%2d | inercia=%.1f | CH=%.1f | DB=%.3f | silueta=%.3f",
                    year, k, rec["inertia"], rec["calinski_harabasz"], rec["davies_bouldin"], rec["silhouette"])
    return pd.DataFrame.from_records(records)


def empirical_pmf(y: np.ndarray, support: np.ndarray) -> np.ndarray:
    """P(Y = v) para v en `support`, con los valores más allá de support.max()
    agrupados en el último cubo (desbordamiento)."""
    y_capped = np.minimum(y, support.max())
    counts = np.bincount(y_capped.astype(np.int64), minlength=support.max() + 1)[: support.max() + 1]
    return counts[support] / counts.sum()




def empirical_ecdf(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ECDF escalonada continua por la derecha: regresa (valores únicos ordenados, F(v))."""
    vals, counts = np.unique(y, return_counts=True)
    cdf = np.cumsum(counts) / counts.sum()
    return vals, cdf


def resolve_display_support(y_all: np.ndarray, max_display_count: Optional[int]) -> np.ndarray:
    cap = max_display_count if max_display_count is not None else int(np.percentile(y_all, 99))
    cap = max(cap, int(y_all.min()) + 1)
    return np.arange(int(y_all.min()), cap + 1)


def plot_discrete_distributions(
    df_year: pd.DataFrame,
    cluster_col: str,
    value_col: str,
    year,
    k_star: int,
    viz: VizConfig,
) -> plt.Figure:
    """Diagnóstico pareado para una variable discreta: PMF empírica (izquierda,
    barras agrupadas, soporte truncado compartido con un cubo de desbordamiento)
    y ECDF (derecha, líneas escalonadas) — ambos descriptores válidos de una
    distribución de conteos, a diferencia de un KDE o violín superpuesto."""
    apply_house_style(viz.palette)
    clusters = sorted(df_year[cluster_col].unique())
    colors = categorical_palette(viz.palette, len(clusters))
    support = resolve_display_support(df_year[value_col].to_numpy(), viz.max_display_count)
    cap = support.max()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Panel PMF: barras agrupadas entre clústeres, desplazadas en x
    width = 0.8 / len(clusters)
    for i, (c, color) in enumerate(zip(clusters, colors)):
        y_c = df_year.loc[df_year[cluster_col] == c, value_col].to_numpy()
        pmf = empirical_pmf(y_c, support)
        offset = (i - (len(clusters) - 1) / 2) * width
        axes[0].bar(support + offset, pmf, width=width, color=color, label=f"clúster {c}")
    xticks = list(support)
    xticklabels = [str(v) for v in support[:-1]] + [f"≥{cap}"]
    axes[0].set_xticks(xticks)
    axes[0].set_xticklabels(xticklabels)
    axes[0].set_title(f"PMF — año {year}, k={k_star}")
    axes[0].set_xlabel(value_col)
    axes[0].set_ylabel("P(Y = y)")
    axes[0].legend(loc="upper right")

    # Panel ECDF: una línea escalonada por clúster
    for c, color in zip(clusters, colors):
        y_c = df_year.loc[df_year[cluster_col] == c, value_col].to_numpy()
        vals, cdf = empirical_ecdf(y_c)
        vals_capped = np.minimum(vals, cap)
        axes[1].step(vals_capped, cdf, where="post", color=color, label=f"clúster {c}", linewidth=1.6)
    axes[1].set_xlim(support.min(), cap)
    axes[1].set_ylim(0, 1.02)
    axes[1].set_title(f"ECDF — año {year}, k={k_star}")
    axes[1].set_xlabel(f"{value_col} (truncado en {cap})")
    axes[1].set_ylabel("F(y)")
    axes[1].legend(loc="lower right")

    for ax in axes:
        strip_axis(ax)
    fig.tight_layout()
    return fig


def plot_cluster_size_share(df_k: pd.DataFrame, k_star: int, viz: VizConfig) -> plt.Figure:
    """Barras apiladas de la proporción del tamaño de cada clúster por año — un
    uso legítimo de barras apiladas, ya que las proporciones de una partición
    discreta (la pertenencia al clúster) son exactamente lo que el apilamiento
    representa, a diferencia de superponer densidades continuas."""
    apply_house_style(viz.palette)
    size_table = df_k.groupby(["year", "cluster"]).size().unstack(fill_value=0)
    size_frac = size_table.div(size_table.sum(axis=1), axis=0)
    colors = categorical_palette(viz.palette, size_frac.shape[1])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    size_frac.plot(kind="bar", stacked=True, ax=ax, color=colors, width=0.7, edgecolor="none")
    ax.set_title(f"Proporción del tamaño de los clústeres por año — k={k_star}")
    ax.set_ylabel("proporción de observaciones")
    ax.set_xlabel("")
    ax.legend(title="clúster", bbox_to_anchor=(1.02, 1), loc="upper left")
    strip_axis(ax)
    fig.tight_layout()
    return fig



def plot_sweep_diagnostics(sweep_df: pd.DataFrame, year, k_list: Sequence[int], viz: VizConfig) -> plt.Figure:
    apply_house_style(viz.palette)
    accent = viz.palette.base_color
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    panels = [
        ("inertia", "Inercia (WCSS) vs k", axes[0, 0]),
        ("silhouette", "Silueta (submuestra) vs k", axes[0, 1]),
        ("calinski_harabasz", "Calinski-Harabasz vs k", axes[1, 0]),
        ("davies_bouldin", "Davies-Bouldin vs k", axes[1, 1]),
    ]
    for col, title, ax in panels:
        ax.plot(sweep_df["k"], sweep_df[col], marker="o", markersize=4,
                color=accent, linewidth=1.4)
        ax.set_title(f"{title} — año {year}")
        ax.set_xlabel("k")
        for k_star in k_list:
            ax.axvline(k_star, color=viz.palette.highlight_color, linestyle="--",
                       alpha=0.5, linewidth=1)
        strip_axis(ax)
    fig.tight_layout()
    return fig


def plot_pca_projection(
    X_norm: np.ndarray,
    labels: np.ndarray,
    year,
    k_star: int,
    viz: VizConfig,
    rng_seed: int,
) -> plt.Figure:
    apply_house_style(viz.palette)
    n = X_norm.shape[0]
    rng = np.random.default_rng(rng_seed)
    idx = rng.choice(n, size=min(viz.pca_sample_size, n), replace=False)

    pca = PCA(n_components=2, random_state=rng_seed)
    X_2d = pca.fit_transform(X_norm[idx])
    labels_2d = labels[idx]
    clusters = sorted(np.unique(labels_2d))
    colors = categorical_palette(viz.palette, len(clusters))

    fig, ax = plt.subplots(figsize=(7, 6))
    for c, color in zip(clusters, colors):
        m = labels_2d == c
        ax.scatter(X_2d[m, 0], X_2d[m, 1], s=5, alpha=0.6, color=color, label=f"clúster {c}", linewidths=0)
    ax.legend(title="clúster", loc="best")
    ax.set_title(f"PCA — año {year}, k={k_star}")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    strip_axis(ax)
    fig.tight_layout()
    return fig


def plot_effect_size_trend(eff_table: pd.DataFrame, k_star: int, viz: VizConfig) -> plt.Figure:
    apply_house_style(viz.palette)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(eff_table["year"], eff_table["epsilon_sq"], marker="o",
            color=viz.palette.base_color, linewidth=1.6)
    ax.set_title(f"Fuerza de separación entre clústeres (ε²) por año — k={k_star}")
    ax.set_xlabel("año")
    ax.set_ylabel("ε²")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    strip_axis(ax)
    fig.tight_layout()
    return fig


def kruskal_effect_size(df_k: pd.DataFrame, k_star: int, value_col: str) -> pd.DataFrame:
    """H de Kruskal-Wallis y epsilon^2 = (H - k + 1) / (n - k), por año. Se
    reporta en lugar de depender de p (que se satura cerca de cero con n grande
    sin importar la magnitud del efecto)."""
    rows = []
    for year, g in df_k.groupby("year"):
        groups = [gg[value_col].to_numpy() for _, gg in g.groupby("cluster")]
        if len(groups) < 2:
            logger.warning("el año=%s tiene %d clúster(es); se omite Kruskal-Wallis.",
                           year, len(groups))
            continue
        try:
            H, p = stats.kruskal(*groups)
        except ValueError:
            # Todos los valores idénticos entre grupos: no hay separación que probar.
            logger.warning("año=%s: valores de %s idénticos; se omite la prueba.",
                           year, value_col)
            continue
        eps_sq = (H - k_star + 1) / (len(g) - k_star)
        rows.append({"year": year, "H": H, "p": p, "epsilon_sq": eps_sq})
    return pd.DataFrame(rows, columns=["year", "H", "p", "epsilon_sq"])


### -------------------------------------------------------------------------------
### Main --------------------------------------------------------------------------


@dataclass
class PipelineResult:
    gdf_labeled: pd.DataFrame
    sweep_table: pd.DataFrame
    results_long: pd.DataFrame
    effect_tables: dict[int, pd.DataFrame]
    models: dict[tuple, MiniBatchKMeans]
    cluster_assignments: pd.DataFrame


def _savefig(fig: plt.Figure, viz: VizConfig, name: str) -> None:
    path = Path(viz.fig_path) / f"{name}{FIG_SUFFIX}.{viz.palette.fig_format}"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("figura guardada -> %s", path)


def run_pipeline(cfg: PipelineConfig, gdf_all: Optional[pd.DataFrame] = None) -> PipelineResult:
    """De principio a fin: barrido -> ajuste -> diagnósticos de la distribución
    discreta -> comparación del tamaño de efecto entre años. `gdf_all` puede
    inyectarse directamente (por ejemplo, ya cargado con geopandas) para
    mantener este módulo agnóstico de la capa de E/S geográfica; en otro caso se
    lee de cfg.data.parquet_path con pandas (parquet no requiere un lector geo a
    menos que la geometría se necesite más adelante)."""
    cfg.validate()
    apply_house_style(cfg.viz.palette)

    if gdf_all is None:
        gdf_all = pd.read_parquet(cfg.data.parquet_path)

    years = sorted(gdf_all[cfg.data.year_column].unique())
    sweep_frames, results_long, effect_tables, models = [], [], {}, {}

    # Etapa 1: barrido, por año
    for year in years:
        gdf_y = gdf_all.query(f"{cfg.data.year_column} == @year").sort_values(cfg.data.sort_column)
        X_norm = load_normalized_features(gdf_y, cfg.data.feature_columns)
        sweep_df = run_k_sweep(X_norm, year, cfg.model)
        sweep_df["year"] = year
        sweep_frames.append(sweep_df)
        fig = plot_sweep_diagnostics(sweep_df, year, cfg.model.k_list, cfg.viz)
        _savefig(fig, cfg.viz, f"kmeans_sweep_diagnostics_{year}")

    sweep_table = pd.concat(sweep_frames, ignore_index=True)

    # Etapa 2: ajustes finales por año x k
    for year in years:
        gdf_y = gdf_all.query(f"{cfg.data.year_column} == @year").sort_values(cfg.data.sort_column)
        X_norm = load_normalized_features(gdf_y, cfg.data.feature_columns)
        key_vals = gdf_y[cfg.data.key_column].to_numpy()
        y_ind = gdf_y[cfg.data.ind_column].to_numpy()
        y_cnt = gdf_y[cfg.data.cnt_column].to_numpy()

        for k_star in cfg.model.k_list:
            model = spherical_kmeans(X_norm, k_star, cfg.model)
            labels = model.predict(X_norm)
            models[(year, k_star)] = model

            col_name = f"y_k{k_star}"
            gdf_all.loc[gdf_y.index, col_name] = labels

            fig_pca = plot_pca_projection(X_norm, labels, year, k_star, cfg.viz, cfg.model.rng_seed)
            _savefig(fig_pca, cfg.viz, f"kmeans_pca_{year}_k{k_star}")

            results_long.append(pd.DataFrame({
                "year": year, "k": k_star, "cluster": labels,
                cfg.data.key_column: key_vals,
                cfg.data.ind_column: y_ind, cfg.data.cnt_column: y_cnt,
            }))

    results_long_df = pd.concat(results_long, ignore_index=True)

    # Etapa 3: comparación entre años por k
    for k_star in cfg.model.k_list:
        df_k = results_long_df.query("k == @k_star")

        table_k = (df_k.groupby(["year", "cluster"])[cfg.data.cnt_column]
                        .describe(percentiles=[.25, .5, .75]).round(3))
        logger.info("\n=== k=%d — %s por año x clúster ===\n%s", k_star, cfg.data.cnt_column, table_k)

        eff_table = kruskal_effect_size(df_k, k_star, cfg.data.cnt_column)
        effect_tables[k_star] = eff_table
        logger.info("\n=== k=%d — tamaño de efecto de Kruskal-Wallis ===\n%s", k_star, eff_table)

        for year, g in df_k.groupby("year"):
            fig_dist = plot_discrete_distributions(g, "cluster", cfg.data.cnt_column, year, k_star, cfg.viz)
            _savefig(fig_dist, cfg.viz, f"kmeans_dist_{year}_k{k_star}")

        fig_eff = plot_effect_size_trend(eff_table, k_star, cfg.viz)
        _savefig(fig_eff, cfg.viz, f"kmeans_epsilon_trend_k{k_star}")

        fig_size = plot_cluster_size_share(df_k, k_star, cfg.viz)
        _savefig(fig_size, cfg.viz, f"kmeans_yearly_sizeshare_k{k_star}")

    # mapa de asignaciones entre años, con las claves necesarias para unir después
    # contra el conjunto de entrenamiento original (year, key_column, k, cluster)
    cluster_assignments_df = results_long_df[
        ["year", cfg.data.key_column, "k", "cluster"]
    ].reset_index(drop=True)

    # persistencia opcional
    if cfg.output.labeled_parquet_path:
        gdf_all.to_parquet(cfg.output.labeled_parquet_path)
    if cfg.output.sweep_table_path:
        sweep_table.to_parquet(cfg.output.sweep_table_path)
    if cfg.output.results_long_path:
        results_long_df.to_parquet(cfg.output.results_long_path)
    if cfg.output.effect_table_path:
        pd.concat(
            [t.assign(k=k) for k, t in effect_tables.items()], ignore_index=True
        ).to_parquet(cfg.output.effect_table_path)
    if cfg.output.cluster_assignments_path:
        cluster_assignments_df.to_parquet(cfg.output.cluster_assignments_path)
        logger.info("asignaciones de clúster guardadas -> %s", cfg.output.cluster_assignments_path)

    return PipelineResult(
        gdf_labeled=gdf_all,
        sweep_table=sweep_table,
        results_long=results_long_df,
        effect_tables=effect_tables,
        models=models,
        cluster_assignments=cluster_assignments_df,
    )

### -------------------------------------------------------------------------------
### Main --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pipeline de clustering con k-means esférico.")
    parser.add_argument("--config", required=True, type=str,
                        help="Ruta al archivo YAML de configuración.")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    cfg = PipelineConfig.from_yaml(args.config)
    result = run_pipeline(cfg)
    logger.info("Pipeline completo. Combinaciones año x k ajustadas: %d", len(result.models))


if __name__ == "__main__":
    main()
